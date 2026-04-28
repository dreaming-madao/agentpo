from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from .config import MADConfig, validate_mad_config
from ..llm_fn import configs, llm_model_dict, local_model_list


def get_client_and_model(actor_model: str) -> Tuple[OpenAI, str]:
    if actor_model in local_model_list:
        openai_api_base = llm_model_dict[actor_model]
        client = OpenAI(api_key="empty", base_url=openai_api_base, timeout=6000)
        model = client.models.list().data[0].id
        return client, model

    config = configs[actor_model]
    if len(config) == 2:
        llm_model_base = "aliyuncs-api"
        model, openai_api_key = config
    else:
        llm_model_base, model, openai_api_key = config

    openai_api_base = llm_model_dict[llm_model_base]
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, timeout=6000)
    return client, model


class ApiDebateBackend:
    def __init__(self, actor_model: str, cfg: MADConfig):
        validate_mad_config(cfg)
        self.cfg = cfg
        self.client, self.model = get_client_and_model(actor_model)

    def _generate_one(self, messages: List[Dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            stream=False,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            timeout=self.cfg.timeout,
        )
        return completion.choices[0].message.content or ""

    def generate_batch(self, message_batches: List[List[Dict[str, str]]]) -> List[str]:
        if not message_batches:
            return []
        if not self.cfg.parallel_agents or len(message_batches) == 1:
            return [self._generate_one(messages) for messages in message_batches]

        max_workers = min(len(message_batches), self.cfg.max_concurrency)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._generate_one, message_batches))


class VLLMDebateBackend:
    def __init__(self, actor_model: str, cfg: MADConfig):
        validate_mad_config(cfg)
        self.cfg = cfg
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("vLLM backend requested but vllm/transformers is not installed.") from exc

        model_path = cfg.vllm_model_path or actor_model
        if actor_model in local_model_list and not cfg.vllm_model_path:
            raise ValueError(
                "vLLM backend needs algorithm.mad.vllm_model_path when actor_model is an API/local endpoint alias."
            )
        self.llm = LLM(model=model_path, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            n=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

    def _to_prompt(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_batch(self, message_batches: List[List[Dict[str, str]]]) -> List[str]:
        prompts = [self._to_prompt(messages) for messages in message_batches]
        outputs = self.llm.generate(prompts, self.sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        return [item.outputs[0].text for item in outputs]


def build_backend(
    actor_model: str,
    cfg: MADConfig,
    backend_override: Optional[str] = None,
):
    backend_name = backend_override or cfg.backend
    if backend_name == "api":
        return ApiDebateBackend(actor_model=actor_model, cfg=cfg)
    if backend_name == "vllm":
        return VLLMDebateBackend(actor_model=actor_model, cfg=cfg)
    raise ValueError(f"Unsupported MAD backend: {backend_name}")
