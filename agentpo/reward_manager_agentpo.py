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

import torch
from openai import OpenAI
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from .llm_fn import llm_model_dict, configs, local_model_list

def get_solution(prompt_lst, actor_model,cooperation_mode):
    if actor_model in local_model_list:
        llm_model_base = actor_model
        openai_api_base = llm_model_dict[llm_model_base]
        openai_api_key = "empty"
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        model = client.models.list().data[0].id

    else:
        llm_model_base = "aliyuncs-api"
        openai_api_base = llm_model_dict[llm_model_base]
        model, openai_api_key = configs[actor_model]
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    completions = []
    
    for prompt in prompt_lst:
        if cooperation_mode=="critic":
            system_prompt='You have an opportunity to improve your solution. Please review Current Solution and Comment carefully. Correct errors and fill justification gaps if any.'
        
        elif cooperation_mode=="assistant":
            system_prompt='Please reason step by step, and put your final answer within \\boxed{{}}.'

        prompt+=" Let's think step by step and output the final answer within \\boxed{{}}."
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
            ]

        completion = client.chat.completions.create(
            messages=messages,
            model=model,
            stream=True,
            max_tokens=2048,
            temperature=0,
            top_p=1.0,
            timeout=6000,
        )
        completions.append(completion)

    response = []
    for completion in completions:
        response_item = ""
        for _ in completion:
            resp = _.choices[0].delta.content
            if resp is not None:
                response_item += _.choices[0].delta.content 
        response.append(response_item)
    return response


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
        cooperation_mode=""
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        self.actor_model=actor_model
        self.cooperation_mode=cooperation_mode

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
        res_lst,prompt_lst=[],[]
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

            res_lst.append([prompt_str,data_source,ground_truth,extra_info,valid_response_length])

            # In assistant mode, the actor's response is treated as a hint for
            # the collaborator model. In critic mode, it is treated as feedback
            # on an existing solution.
            if self.cooperation_mode=="assistant":
                prompt=f"Problem: {problem} (Hint: {response_str})"

            elif self.cooperation_mode=="critic":
                solution=data_item.non_tensor_batch['solutions']
                prompt=f"Problem: {problem}\n\nCurrent Solution: {solution}\n\nComment: {response_str}"

            else:
                raise RuntimeError(f"cooperation_mode is not defined.")
            
            prompt_lst.append(prompt)

        # Ask the configured collaborator model/API to produce final solutions
        # from the AgentPO prompts built above.
        response_str_lst=get_solution(prompt_lst, self.actor_model,self.cooperation_mode)

        for i, res in enumerate(res_lst):
            prompt_str,data_source,ground_truth,extra_info,valid_response_length=res
            response_str=response_str_lst[i]

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
            reward_tensor[i, valid_response_length - 1] = reward

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

        # Training asks for both the reward tensor and logging extras, while
        # validation may only need the reward tensor.
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
