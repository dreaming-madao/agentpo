#!/usr/bin/env python3
import argparse
import json
import os

import pandas as pd


def load_parquet_rows(path: str) -> list[dict]:
    return pd.read_parquet(path).to_dict("records")


def to_eval_row(row: dict, idx: int) -> dict:
    reward_model = row.get("reward_model") or {}
    return {
        "idx": idx,
        "problem": row.get("problem", ""),
        "answer": reward_model.get("ground_truth", ""),
        "solution": (row.get("solutions") or [""])[0],
        "level": row.get("level", ""),
        "type": row.get("type", ""),
    }


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny train/val/eval files for an end-to-end smoke test.")
    parser.add_argument("--data_dir", default="/home/ly/agentpo/data/math8k")
    parser.add_argument("--eval_data_dir", default="/home/ly/agentpo/agentpo/evaluation/data/math8k")
    parser.add_argument("--train_num", type=int, default=4)
    parser.add_argument("--val_num", type=int, default=2)
    args = parser.parse_args()

    train_src = os.path.join(args.data_dir, "math8k_hard_solutions_1000.parquet")
    val_src = os.path.join(args.data_dir, "test_solutions.parquet")
    train_out = os.path.join(args.data_dir, "math8k_hard_solutions_smoke.parquet")
    val_out = os.path.join(args.data_dir, "test_solutions_smoke.parquet")
    eval_out = os.path.join(args.eval_data_dir, "test.jsonl")

    train_rows = load_parquet_rows(train_src)[: args.train_num]
    val_rows = load_parquet_rows(val_src)[: args.val_num]

    pd.DataFrame(train_rows).to_parquet(train_out, index=False)
    pd.DataFrame(val_rows).to_parquet(val_out, index=False)
    write_jsonl([to_eval_row(row, idx) for idx, row in enumerate(val_rows)], eval_out)

    print(f"wrote {len(train_rows)} train rows to {train_out}")
    print(f"wrote {len(val_rows)} val rows to {val_out}")
    print(f"wrote {len(val_rows)} eval rows to {eval_out}")


if __name__ == "__main__":
    main()
