# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
from openai import OpenAI
from omegaconf import OmegaConf
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from .llm_fn import llm_model_dict, configs, local_model_list
from .debate import MADConfig, run_mad


def _get_client_and_model(actor_model):
    if actor_model in local_model_list:
        llm_model_base = actor_model
        openai_api_base = llm_model_dict[llm_model_base]
        openai_api_key = "empty"
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, timeout=6000)
        model = client.models.list().data[0].id

    else:
        config = configs[actor_model]
        if len(config) == 2:
            llm_model_base = "aliyuncs-api"
            model, openai_api_key = config
        else:
            llm_model_base, model, openai_api_key = config
        openai_api_base = llm_model_dict[llm_model_base]
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, timeout=6000)
    return client, model


def build_actor_prompt(problem: str, collaborator_signal: str, cooperation_mode: str, solution=None) -> str:
    if cooperation_mode == "assistant":
        return f"Problem: {problem} (Hint: {collaborator_signal})"
    if cooperation_mode == "critic":
        return f"Problem: {problem}\n\nCurrent Solution: {solution}\n\nComment: {collaborator_signal}"
    raise RuntimeError(f"Unsupported cooperation_mode for actor prompt: {cooperation_mode}")


def get_solution(prompt_lst, actor_model, cooperation_mode):
    client, model = _get_client_and_model(actor_model)
    if cooperation_mode == "critic":
        system_prompt = (
            "You have an opportunity to improve your solution. Please review Current Solution and "
            "Comment carefully. Correct errors and fill justification gaps if any."
        )
    elif cooperation_mode == "assistant":
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}."
    else:
        raise RuntimeError(f"Unsupported cooperation_mode for get_solution: {cooperation_mode}")

    responses = []
    for prompt in prompt_lst:
        prompt += " Let's think step by step and output the final answer within \\boxed{{}}."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        completion = client.chat.completions.create(
            messages=messages,
            model=model,
            stream=False,
            max_tokens=2048,
            temperature=0,
            top_p=1.0,
            timeout=6000,
        )
        response_item = completion.choices[0].message.content or ""
        responses.append(response_item)
    return responses


def _preview_text(text: str, max_chars: int = 240) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


class AgentPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        actor_model="",
        cooperation_mode="",
        mad_config=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        self.actor_model=actor_model
        self.cooperation_mode=cooperation_mode
        if mad_config is None:
            mad_config_dict = {}
        elif OmegaConf.is_config(mad_config):
            mad_config_dict = OmegaConf.to_container(mad_config, resolve=True)
        else:
            mad_config_dict = dict(mad_config)
        self.mad_config = MADConfig(**mad_config_dict)

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If an upstream reward model has already produced scores, reuse them directly.
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # verl expects token-level rewards, so we create a tensor with the same
        # shape as responses and later place the sequence-level reward on the
        # final valid response token.
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        items = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            # Split prompt/response token ids and remove padding according to
            # the attention mask before decoding them back to text.
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            # Metadata comes from the parquet row and is needed for scoring and
            # for building the collaborator prompt.
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            problem = data_item.non_tensor_batch["problem"]
            items.append(
                {
                    "idx": i,
                    "prompt_str": prompt_str,
                    "problem": problem,
                    "collab_signal": response_str,
                    "data_source": data_source,
                    "ground_truth": ground_truth,
                    "extra_info": extra_info,
                    "valid_response_length": valid_response_length,
                    "solution": data_item.non_tensor_batch.get("solutions", None),
                }
            )

        final_solutions = []
        histories = []
        if self.cooperation_mode in {"assistant", "critic"}:
            prompt_lst = []
            for item in items:
                prompt_lst.append(
                    build_actor_prompt(
                        problem=item["problem"],
                        collaborator_signal=item["collab_signal"],
                        cooperation_mode=self.cooperation_mode,
                        solution=item["solution"],
                    )
                )
            final_solutions = get_solution(prompt_lst, self.actor_model, self.cooperation_mode)
            histories = [None] * len(final_solutions)
        elif self.cooperation_mode == "mad":
            if self.mad_config.parallel_rollouts and len(items) > 1:
                max_workers = min(len(items), self.mad_config.max_concurrency)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(
                        executor.map(
                            lambda item: run_mad(
                                problem=item["problem"],
                                collaborator_signal=item["collab_signal"],
                                actor_model=self.actor_model,
                                cfg=self.mad_config,
                            ),
                            items,
                        )
                    )
                for item, (final_solution, history) in zip(items, results):
                    final_solutions.append(final_solution)
                    histories.append(history)
                    print(
                        "[mad_success] "
                        f"idx={item['idx']} "
                        f"rounds={len(history.get('rounds', [])) if history else 0} "
                        f"final={_preview_text(final_solution)}",
                        flush=True,
                    )
            else:
                for item in items:
                    final_solution, history = run_mad(
                        problem=item["problem"],
                        collaborator_signal=item["collab_signal"],
                        actor_model=self.actor_model,
                        cfg=self.mad_config,
                    )
                    final_solutions.append(final_solution)
                    histories.append(history)
                    print(
                        "[mad_success] "
                        f"idx={item['idx']} "
                        f"rounds={len(history.get('rounds', [])) if history else 0} "
                        f"final={_preview_text(final_solution)}",
                        flush=True,
                    )
        else:
            raise RuntimeError(f"cooperation_mode is not defined: {self.cooperation_mode}")

        for item, response_str, history in zip(items, final_solutions, histories):
            prompt_str = item["prompt_str"]
            data_source = item["data_source"]
            ground_truth = item["ground_truth"]
            extra_info = item["extra_info"]
            valid_response_length = item["valid_response_length"]

            # Convert the collaborator's final answer into a scalar score.
            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score
            # Put the sequence reward on the last valid token so downstream
            # PPO/GRPO code can consume it as token-level rewards.
            reward_tensor[item["idx"], valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # Optionally print a few decoded samples for manual inspection.
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
                if history is not None:
                    print("[mad_aggregation]", history.get("aggregation", {}))

            if history is not None:
                reward_extra_info["mad_history"].append(history)

        # Training asks for both the reward tensor and logging extras, while
        # validation may only need the reward tensor.
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
