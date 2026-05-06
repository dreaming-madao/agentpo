#!/usr/bin/env python3
import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI


DEFAULT_DATA = "/home/ly/agentpo/data/math8k/test_solutions_50.parquet"


def load_questions(path: str, question_key: str, limit: int) -> List[str]:
    df = pd.read_parquet(path)
    if question_key not in df.columns:
        raise KeyError(f"Question key {question_key!r} not found. Columns: {list(df.columns)}")
    questions = []
    for value in df[question_key].tolist():
        if isinstance(value, list):
            text = "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in value)
        else:
            text = str(value)
        text = text.strip()
        if text:
            questions.append(text)
        if len(questions) >= limit:
            break
    if not questions:
        raise ValueError(f"No questions loaded from {path}")
    return questions


def build_messages(question: str, prompt_style: str) -> List[Dict[str, str]]:
    if prompt_style == "assistant_reward":
        question = (
            "Problem: "
            f"{question} "
            "(Hint: We need solve the problem carefully, reason step by step, "
            "and put the final answer within \\boxed{}.)"
        )
    return [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {
            "role": "user",
            "content": f"{question}\nLet's think step by step and output the final answer within \\boxed{{}}.",
        },
    ]


def _result(seconds: float, text: str, usage: Any = None) -> Dict[str, Any]:
    return {
        "seconds": seconds,
        "chars": len(text),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
        "preview": " ".join(text.split())[:120],
    }


def _drain_stream(completion: Any) -> str:
    text = ""
    for chunk in completion:
        delta = chunk.choices[0].delta.content
        if delta:
            text += delta
    return text


def call_once(
    client: OpenAI,
    model: str,
    question: str,
    max_tokens: int,
    timeout: int,
    prompt_style: str,
    stream: bool,
) -> Dict[str, Any]:
    start = time.perf_counter()
    completion = client.chat.completions.create(
        messages=build_messages(question, prompt_style),
        model=model,
        stream=stream,
        max_tokens=max_tokens,
        temperature=0,
        top_p=1.0,
        timeout=timeout,
    )
    if stream:
        text = _drain_stream(completion)
        return _result(time.perf_counter() - start, text)

    elapsed = time.perf_counter() - start
    text = completion.choices[0].message.content or ""
    return _result(elapsed, text, getattr(completion, "usage", None))


def call_stream_submit_all(
    client: OpenAI,
    model: str,
    questions: List[str],
    max_tokens: int,
    timeout: int,
    prompt_style: str,
    name: str,
) -> List[Dict[str, Any]]:
    batch_start = time.perf_counter()
    completions = []
    for idx, question in enumerate(questions, 1):
        start = time.perf_counter()
        completion = client.chat.completions.create(
            messages=build_messages(question, prompt_style),
            model=model,
            stream=True,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1.0,
            timeout=timeout,
        )
        completions.append((idx, start, completion))

    results = []
    for idx, start, completion in completions:
        text = _drain_stream(completion)
        result = _result(time.perf_counter() - start, text)
        results.append(result)
        print(
            f"[{name}] {idx:02d}/{len(questions)} "
            f"{result['seconds']:.2f}s chars={result['chars']} tokens={result['total_tokens']} "
            f"preview={result['preview']}",
            flush=True,
        )

    print(f"[{name}] stream_submit_all_wall_s={time.perf_counter() - batch_start:.2f}", flush=True)
    return results


def resolve_model(client: OpenAI, model: str) -> str:
    if model != "auto":
        return model
    models = client.models.list().data
    if not models:
        raise RuntimeError("No model returned by /models")
    return models[0].id


def summarize(name: str, results: List[Dict[str, Any]]) -> None:
    seconds = [item["seconds"] for item in results]
    print(f"\n[{name}] summary")
    print(f"  requests: {len(results)}")
    print(f"  total_s:  {sum(seconds):.2f}")
    print(f"  mean_s:   {statistics.mean(seconds):.2f}")
    print(f"  median_s: {statistics.median(seconds):.2f}")
    print(f"  min_s:    {min(seconds):.2f}")
    print(f"  max_s:    {max(seconds):.2f}")
    total_tokens = [item["total_tokens"] for item in results if item["total_tokens"] is not None]
    if total_tokens:
        print(f"  tokens:   total={sum(total_tokens)} mean={statistics.mean(total_tokens):.1f}")


def bench_backend(
    name: str,
    base_url: str,
    api_key: str,
    model: str,
    questions: List[str],
    max_tokens: int,
    timeout: int,
    concurrency: int,
    prompt_style: str,
    stream: bool,
    stream_submit_all: bool,
) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    model = resolve_model(client, model)
    print(
        f"\n[{name}] base_url={base_url} model={model} requests={len(questions)} "
        f"concurrency={concurrency} stream={stream} stream_submit_all={stream_submit_all}",
        flush=True,
    )

    if stream_submit_all:
        results = call_stream_submit_all(
            client=client,
            model=model,
            questions=questions,
            max_tokens=max_tokens,
            timeout=timeout,
            prompt_style=prompt_style,
            name=name,
        )
        summarize(name, results)
        return results

    if concurrency <= 1:
        results = []
        for idx, question in enumerate(questions, 1):
            result = call_once(
                client,
                model,
                question,
                max_tokens=max_tokens,
                timeout=timeout,
                prompt_style=prompt_style,
                stream=stream,
            )
            results.append(result)
            print(
                f"[{name}] {idx:02d}/{len(questions)} "
                f"{result['seconds']:.2f}s chars={result['chars']} tokens={result['total_tokens']} "
                f"preview={result['preview']}",
                flush=True,
            )
        summarize(name, results)
        return results

    results_by_idx: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(call_once, client, model, question, max_tokens, timeout, prompt_style, stream): idx
            for idx, question in enumerate(questions, 1)
        }
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results_by_idx[idx] = result
            print(
                f"[{name}] {idx:02d}/{len(questions)} "
                f"{result['seconds']:.2f}s chars={result['chars']} tokens={result['total_tokens']} "
                f"preview={result['preview']}",
                flush=True,
            )
    results = [results_by_idx[idx] for idx in sorted(results_by_idx)]
    summarize(name, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DashScope API and local vLLM response time on the same questions.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--question-key", default="problem")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--repeat-per-question", type=int, default=1)
    parser.add_argument("--prompt-style", choices=["plain", "assistant_reward"], default="plain")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=6000)
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests per backend. Default matches current reward code: serial.")
    parser.add_argument("--stream", action="store_true", help="Use stream=True and consume each response before sending the next request.")
    parser.add_argument("--stream-submit-all", action="store_true", help="Use old hint_start behavior: create all stream=True responses first, then drain them.")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--api-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-model", default="qwen2.5-math-7b-instruct")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:10005/v1")
    parser.add_argument("--vllm-model", default="auto")
    args = parser.parse_args()

    questions = load_questions(args.data, args.question_key, args.num_questions)
    base_count = len(questions)
    if args.repeat_per_question > 1:
        questions = [question for question in questions for _ in range(args.repeat_per_question)]
    print(
        f"[bench] loaded {base_count} base questions from {args.data}; "
        f"repeat_per_question={args.repeat_per_question}; total_requests={len(questions)}; "
        f"prompt_style={args.prompt_style}"
    )

    if not args.skip_api:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        bench_backend(
            name="dashscope",
            base_url=args.api_base_url,
            api_key=api_key,
            model=args.api_model,
            questions=questions,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            concurrency=args.concurrency,
            prompt_style=args.prompt_style,
            stream=args.stream or args.stream_submit_all,
            stream_submit_all=args.stream_submit_all,
        )

    if not args.skip_vllm:
        bench_backend(
            name="vllm",
            base_url=args.vllm_base_url,
            api_key="empty",
            model=args.vllm_model,
            questions=questions,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            concurrency=args.concurrency,
            prompt_style=args.prompt_style,
            stream=args.stream or args.stream_submit_all,
            stream_submit_all=args.stream_submit_all,
        )


if __name__ == "__main__":
    main()
