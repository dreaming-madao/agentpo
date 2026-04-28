from dataclasses import dataclass


@dataclass
class MADConfig:
    backend: str = "api"  # api | vllm
    vllm_model_path: str = ""
    num_agents: int = 3
    debate_rounds: int = 2
    topology: str = "full"  # full | sparse | centralized
    multi_persona: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    timeout: int = 6000
    max_peer_chars: int = 2000
    parallel_agents: bool = True
    parallel_rollouts: bool = False
    max_concurrency: int = 3
    api_batch_size: int = 8
    use_majority_vote: bool = True


def validate_mad_config(cfg: MADConfig) -> None:
    if cfg.backend not in {"api", "vllm"}:
        raise ValueError(f"Unsupported MAD backend: {cfg.backend}")
    if cfg.num_agents < 1:
        raise ValueError("MAD num_agents must be >= 1")
    if cfg.debate_rounds < 0:
        raise ValueError("MAD debate_rounds must be >= 0")
    if cfg.topology not in {"full", "sparse", "centralized"}:
        raise ValueError(f"Unsupported MAD topology: {cfg.topology}")
    if cfg.max_concurrency < 1:
        raise ValueError("MAD max_concurrency must be >= 1")
