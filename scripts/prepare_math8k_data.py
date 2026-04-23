#!/usr/bin/env python3
import argparse
import os
import random
import re

from datasets import Dataset, load_dataset


DEFAULT_DATASETS = [
    ("qwedsacf/competition_math", None),
    ("the-jb/hendrycks-math", None),
    ("jeggers/competition_math", "main"),
    ("competition_math", "main"),
]


def parse_level(level) -> int | None:
    match = re.search(r"\d+", str(level))
    return int(match.group()) if match else None


def last_boxed_only_string(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None

    depth = 0
    for idx in range(start + len(marker) - 1, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def remove_boxed(text: str) -> str:
    return text[len(r"\boxed{") : -1]


def extract_answer(solution: str) -> str:
    boxed = last_boxed_only_string(solution)
    if boxed is not None:
        return remove_boxed(boxed).strip()

    # Fallback for rare records without boxed answers.
    tail = solution.strip().splitlines()[-1]
    tail = re.sub(r"(?i)^.*answer\s*(is|:)\s*", "", tail).strip()
    return tail


def load_math_dataset(split: str, dataset_name: str | None, subset: str | None):
    candidates = [(dataset_name, subset)] if dataset_name else DEFAULT_DATASETS
    last_error = None
    for name, config_name in candidates:
        try:
            kwargs = {"split": split}
            if config_name:
                kwargs["name"] = config_name
            return load_dataset(name, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "Unable to load a MATH dataset. Check network access or pass --dataset_name/--subset "
        "for a locally cached Hugging Face dataset."
    ) from last_error


def build_rows(split: str, levels: set[int], dataset_name: str | None, subset: str | None) -> list[dict]:
    dataset = load_math_dataset(split, dataset_name, subset)
    rows = []
    for idx, example in enumerate(dataset):
        level = parse_level(example.get("level", ""))
        if level not in levels:
            continue

        solution = str(example.get("solution", "")).strip()
        answer = str(example.get("answer", "")).strip() or extract_answer(solution)
        problem = str(example.get("problem", "")).strip()

        rows.append(
            {
                "problem": problem,
                "data_source": "math8k",
                "ability": "math",
                "level": level,
                "type": example.get("type", ""),
                "solutions": [solution],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    "index": idx,
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MATH level 3-5 data and convert it to AgentPO parquet files.")
    parser.add_argument("--output_dir", default="/home/ly/agentpo/data/math8k")
    parser.add_argument("--train_limit", type=int, default=1000, help="Number of train examples to export.")
    parser.add_argument("--val_limit", type=int, default=-1, help="Number of validation examples to export; <=0 keeps all.")
    parser.add_argument("--levels", default="3,4,5", help="Comma-separated MATH levels to keep.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when subsampling train examples.")
    parser.add_argument("--dataset_name", default=None, help="Optional Hugging Face dataset name or local dataset path.")
    parser.add_argument("--subset", default=None, help="Optional Hugging Face dataset config/subset name.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    levels = {int(level) for level in args.levels.split(",") if level.strip()}

    train_rows = build_rows("train", levels, args.dataset_name, args.subset)
    test_rows = build_rows("test", levels, args.dataset_name, args.subset)

    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    if args.train_limit > 0:
        train_rows = train_rows[: args.train_limit]
    if args.val_limit > 0:
        test_rows = test_rows[: args.val_limit]

    train_path = os.path.join(args.output_dir, "math8k_hard_solutions_1000.parquet")
    test_path = os.path.join(args.output_dir, "test_solutions.parquet")

    Dataset.from_list(train_rows).to_parquet(train_path)
    Dataset.from_list(test_rows).to_parquet(test_path)

    print(f"wrote {len(train_rows)} rows to {train_path}")
    print(f"wrote {len(test_rows)} rows to {test_path}")


if __name__ == "__main__":
    main()
