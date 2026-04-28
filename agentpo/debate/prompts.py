from typing import Dict, List, Optional, Tuple


DEFAULT_PERSONAS = {
    "Solver": "You are a careful mathematical reasoning agent.",
    "Skeptic": "You are a skeptical mathematical reviewer who searches for hidden errors.",
    "Checker": "You are a meticulous checker who verifies algebra, arithmetic, and edge cases.",
}


def get_personas(num_agents: int, multi_persona: bool) -> List[Tuple[str, str]]:
    if not multi_persona:
        return [("Debater", "You are a careful mathematical reasoning agent.")] * num_agents

    items = list(DEFAULT_PERSONAS.items())
    if len(items) < num_agents:
        mul = (num_agents + len(items) - 1) // len(items)
        items = items * mul
    return items[:num_agents]


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def build_initial_debater_messages(
    problem: str,
    guide: str,
    persona_name: Optional[str] = None,
    persona_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    system_prompt = persona_prompt or "You are a careful mathematical reasoning agent."
    if persona_name:
        system_prompt = f"{system_prompt}\nRole: {persona_name}."
    system_prompt += (
        "\nYou are one debater in a multi-agent debate. Solve the problem independently first. "
        "Be concise but rigorous, and put the final answer within \\boxed{}."
    )

    user_prompt = (
        f"<Problem>\n{problem}\n\n"
        f"<Debate Guide>\n{guide}\n\n"
        "Solve the problem independently. Reason carefully and put your final answer within \\boxed{}."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_debate_messages(
    problem: str,
    guide: str,
    own_prev_response: str,
    peer_responses: Dict[str, str],
    round_idx: int,
    max_peer_chars: int,
    persona_name: Optional[str] = None,
    persona_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    system_prompt = persona_prompt or "You are a careful mathematical reasoning agent."
    if persona_name:
        system_prompt = f"{system_prompt}\nRole: {persona_name}."
    system_prompt += (
        "\nYou are revising your answer in a multi-agent debate. Use peer opinions carefully, "
        "but do not follow them blindly. Put the final answer within \\boxed{}."
    )

    peer_block = []
    for peer_name, response in peer_responses.items():
        peer_block.append(
            f"[{peer_name} previous response]\n{truncate_text(response, max_peer_chars)}"
        )

    peer_text = "\n\n".join(peer_block) if peer_block else "[No peer responses provided]"
    user_prompt = (
        f"<Problem>\n{problem}\n\n"
        f"<Debate Guide>\n{guide}\n\n"
        f"<Your Previous Response>\n{truncate_text(own_prev_response, max_peer_chars)}\n\n"
        f"<Peer Responses From Round {round_idx - 1}>\n{peer_text}\n\n"
        "Revise your answer if needed. Keep strong reasoning, fix any errors you find, "
        "and put the final answer within \\boxed{}."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
