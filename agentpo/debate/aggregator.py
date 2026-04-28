import collections
import re
from typing import Dict, Tuple


def extract_boxed_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return ""


def extract_final_answer_fallback(text: str) -> str:
    matches = re.findall(r"(?is)final[_ ]answer\s*[:：]\s*([^\n]+)", text)
    if matches:
        return matches[-1].strip()
    return ""


def normalize_vote_answer(answer: str) -> str:
    normalized = answer.strip()
    normalized = normalized.rstrip(".")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def majority_vote(responses: Dict[str, str]) -> Tuple[str, Dict[str, object]]:
    extracted_answers = []
    for response in responses.values():
        answer = extract_boxed_answer(response)
        if not answer:
            answer = extract_final_answer_fallback(response)
        if answer:
            extracted_answers.append(normalize_vote_answer(answer))

    if not extracted_answers:
        fallback_response = next(iter(responses.values()))
        return fallback_response, {
            "method": "majority_vote",
            "answers": [],
            "counts": {},
            "winner": "",
            "used_fallback_response": True,
        }

    counts = collections.Counter(extracted_answers)
    max_count = max(counts.values())
    winner = ""
    for answer in extracted_answers:
        if counts[answer] == max_count:
            winner = answer
            break

    final_solution = f"The final answer is \\boxed{{{winner}}}."
    return final_solution, {
        "method": "majority_vote",
        "answers": extracted_answers,
        "counts": dict(counts),
        "winner": winner,
        "used_fallback_response": False,
    }
