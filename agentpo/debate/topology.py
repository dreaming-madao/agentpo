from typing import List


def select_peers(agent_idx: int, agent_names: List[str], topology: str) -> List[str]:
    if topology == "sparse":
        if len(agent_names) <= 1:
            return []
        return [
            agent_names[(agent_idx - 1) % len(agent_names)],
            agent_names[(agent_idx + 1) % len(agent_names)],
        ]

    if topology == "centralized":
        if agent_idx == 0:
            return agent_names[1:]
        return [agent_names[0]]

    return agent_names[:agent_idx] + agent_names[agent_idx + 1 :]
