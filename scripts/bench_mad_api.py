import argparse
import time

from agentpo.debate import MADConfig, run_mad


PROBLEM = (
    "A store sells notebooks for $3 each and pens for $2 each. "
    "Mia buys 4 notebooks and 5 pens. How much does she spend in total?"
)

GUIDE = (
    "Identify the cost of notebooks and pens separately, then add them. "
    "Check the arithmetic and put the final answer in \\boxed{}."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-model", default="Llama-3.2-3B")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--debate-rounds", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-peer-chars", type=int, default=600)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    cfg = MADConfig(
        backend="api",
        num_agents=args.num_agents,
        debate_rounds=args.debate_rounds,
        topology="full",
        multi_persona=False,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        max_peer_chars=args.max_peer_chars,
        parallel_agents=True,
        parallel_rollouts=False,
        max_concurrency=args.max_concurrency,
    )

    durations = []
    print(
        "[bench_mad] "
        f"actor_model={args.actor_model} "
        f"runs={args.num_runs} agents={args.num_agents} rounds={args.debate_rounds} "
        f"max_tokens={args.max_tokens}",
        flush=True,
    )
    for i in range(args.num_runs):
        start = time.perf_counter()
        final_solution, history = run_mad(
            problem=PROBLEM,
            collaborator_signal=GUIDE,
            actor_model=args.actor_model,
            cfg=cfg,
        )
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(
            f"[bench_mad] run={i + 1}/{args.num_runs} "
            f"seconds={elapsed:.2f} rounds={len(history.get('rounds', []))} "
            f"final={final_solution}",
            flush=True,
        )

    avg = sum(durations) / len(durations)
    calls_per_mad = args.num_agents * (args.debate_rounds + 1)
    print(
        f"[bench_mad] avg_seconds_per_mad={avg:.2f} "
        f"calls_per_mad={calls_per_mad} "
        f"estimated_seconds_per_prompt_n16={avg * 16:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
