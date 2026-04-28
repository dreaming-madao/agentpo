from typing import Any, Dict, List, Tuple

from .aggregator import majority_vote
from .backend import build_backend
from .config import MADConfig, validate_mad_config
from .prompts import (
    build_debate_messages,
    build_initial_debater_messages,
    get_personas,
)
from .topology import select_peers


def run_mad(
    problem: str,
    collaborator_signal: str,
    actor_model: str,
    cfg: MADConfig,
) -> Tuple[str, Dict[str, Any]]:
    validate_mad_config(cfg)
    backend = build_backend(actor_model=actor_model, cfg=cfg)
    personas = get_personas(cfg.num_agents, cfg.multi_persona)
    agent_names = [f"Agent{i + 1}_{personas[i][0]}" for i in range(cfg.num_agents)]

    history: Dict[str, Any] = {
        "config": vars(cfg),
        "agent_names": agent_names,
        "rounds": [],
    }

    initial_batches: List[List[Dict[str, str]]] = []
    for i in range(cfg.num_agents):
        persona_name, persona_prompt = personas[i]
        initial_batches.append(
            build_initial_debater_messages(
                problem=problem,
                guide=collaborator_signal,
                persona_name=persona_name,
                persona_prompt=persona_prompt,
            )
        )

    current_outputs = backend.generate_batch(initial_batches)
    current_responses = dict(zip(agent_names, current_outputs))
    history["rounds"].append({"round_idx": 0, "responses": current_responses.copy()})

    for round_idx in range(1, cfg.debate_rounds + 1):
        round_batches: List[List[Dict[str, str]]] = []
        for i, agent_name in enumerate(agent_names):
            peer_names = select_peers(i, agent_names, cfg.topology)
            peer_responses = {peer_name: current_responses[peer_name] for peer_name in peer_names}
            persona_name, persona_prompt = personas[i]
            round_batches.append(
                build_debate_messages(
                    problem=problem,
                    guide=collaborator_signal,
                    own_prev_response=current_responses[agent_name],
                    peer_responses=peer_responses,
                    round_idx=round_idx,
                    max_peer_chars=cfg.max_peer_chars,
                    persona_name=persona_name,
                    persona_prompt=persona_prompt,
                )
            )

        current_outputs = backend.generate_batch(round_batches)
        current_responses = dict(zip(agent_names, current_outputs))
        history["rounds"].append({"round_idx": round_idx, "responses": current_responses.copy()})

    final_solution, aggregation_meta = majority_vote(current_responses)
    history["aggregation"] = aggregation_meta
    history["final_solution"] = final_solution
    return final_solution, history
