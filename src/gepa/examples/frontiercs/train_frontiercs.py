# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Train script for GEPA optimization on Frontier-CS algorithmic benchmark.

Per-problem mode: optimizes a separate prompt for each problem independently,
with batch size 1. Results (score and prompt) are saved per problem.
Supports parallel optimization across problems (default 32 concurrent).
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from gepa import optimize
from gepa.adapters.frontiercs_adapter import FrontierCSAdapter
from gepa.examples.frontiercs import load_all_problems

DEFAULT_PROBLEMS_DIR = "/data/hry/Frontier-CS-synthetic/Frontier-CS/algorithmic/problems"
DEFAULT_SEED_PROMPT = """You are a competitive programmer. You will be given a problem statement, please implement a solution in C++. The execution time and memory limit are also stated in the statement so be aware of the complexity of the program. Please wrap the code in ```cpp and ``` so that it is properly formatted. Your response should ONLY contain the C++ code, with no additional explanation or text."""


def main() -> None:
    parser = argparse.ArgumentParser(description="GEPA per-problem optimization for Frontier-CS")
    parser.add_argument(
        "--problems_dir",
        type=str,
        default=DEFAULT_PROBLEMS_DIR,
        help="Path to Frontier-CS problems directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gepa_frontiercs_results",
        help="Directory to save results (prompts and scores)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5",
        help="Task LLM for code generation",
    )
    parser.add_argument(
        "--reflection_lm",
        type=str,
        default="openai/gpt-5",
        help="Reflection LLM for prompt improvement",
    )
    parser.add_argument(
        "--judge_url",
        type=str,
        default="http://localhost:8081",
        help="Frontier-CS judge API URL",
    )
    parser.add_argument(
        "--max_metric_calls",
        type=int,
        default=25,
        help="Optimization budget (evaluations) per problem",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--seed_prompt_path",
        type=str,
        default=None,
        help="Path to seed prompt file (default: use built-in)",
    )
    parser.add_argument(
        "--problem_ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional: only optimize these problem IDs (default: all)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="gepa_frontiercs.log",
        help="Log file path (default: gepa_frontiercs.log in project root)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug info when score is 0 (e.g. judge/LLM errors)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Number of problems to optimize in parallel (default: 64)",
    )
    args = parser.parse_args()

    problems = load_all_problems(problems_dir=args.problems_dir)
    if args.problem_ids:
        id_set = set(args.problem_ids)
        problems = [p for p in problems if p["problem_id"] in id_set]
        if len(problems) != len(id_set):
            found = {p["problem_id"] for p in problems}
            missing = id_set - found
            raise ValueError(f"Problems not found: {missing}")

    if args.seed_prompt_path:
        seed_prompt_text = Path(args.seed_prompt_path).read_text()
    else:
        seed_prompt_text = DEFAULT_SEED_PROMPT

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    log_file = None
    if args.log_file:
        log_file = open(args.log_file, "w", encoding="utf-8")

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if log_file:
            log_file.write(line + "\n")
            log_file.flush()

    results: dict[str, dict[str, str | float]] = {}

    log("-" * 60)
    log(f"Problems dir: {args.problems_dir}")
    log(f"Output dir: {args.output_dir}")
    log(f"Model: {args.model}, Reflection: {args.reflection_lm}")
    log(f"Judge: {args.judge_url}, Budget per problem: {args.max_metric_calls}")
    log("Batch size: 1, Per-problem optimization")
    log(f"Total problems: {len(problems)}")
    log(f"Concurrency: {args.concurrency}")
    log("-" * 60)

    log_lock = threading.Lock()
    results_lock = threading.Lock()

    def optimize_one(problem_index: int, problem: dict) -> tuple[str, float, str]:
        problem_id = problem["problem_id"]
        trainset = [problem]
        valset = [problem]
        seed_candidate = {"system_prompt": seed_prompt_text}

        result = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=FrontierCSAdapter(
                model=args.model,
                judge_url=args.judge_url,
                verbose=args.verbose,
            ),
            reflection_lm=args.reflection_lm,
            reflection_minibatch_size=1,
            batch_sampler="epoch_shuffled",
            max_metric_calls=args.max_metric_calls,
            seed=args.seed + problem_index,
            display_progress_bar=False,
        )

        best_prompt = result.best_candidate.get("system_prompt", "")
        raw_score = result.val_aggregate_scores[result.best_idx] * 100.0
        return problem_id, raw_score, best_prompt

    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(optimize_one, i, problem): problem["problem_id"]
            for i, problem in enumerate(problems)
        }
        for future in as_completed(futures):
            try:
                problem_id, raw_score, best_prompt = future.result()
                with results_lock:
                    results[problem_id] = {
                        "score": round(raw_score, 2),
                        "prompt": best_prompt,
                    }
                (prompts_dir / f"{problem_id}.txt").write_text(best_prompt, encoding="utf-8")
                completed += 1
                with log_lock:
                    log(f"[{completed}/{len(problems)}] Problem {problem_id} done: {raw_score:.2f}/100")
                scores_path = output_dir / "scores.json"
                with results_lock:
                    scores_path.write_text(
                        json.dumps({pid: r["score"] for pid, r in results.items()}, indent=2),
                        encoding="utf-8",
                    )
            except Exception as e:
                problem_id = futures[future]
                with log_lock:
                    log(f"ERROR problem {problem_id}: {e}")
                with results_lock:
                    results[problem_id] = {"score": 0.0, "prompt": ""}
                (prompts_dir / f"{problem_id}.txt").write_text("", encoding="utf-8")
                completed += 1
                scores_path = output_dir / "scores.json"
                with results_lock:
                    scores_path.write_text(
                        json.dumps({pid: r["score"] for pid, r in results.items()}, indent=2),
                        encoding="utf-8",
                    )

    summary_path = output_dir / "results.json"
    summary = {
        "config": {
            "model": args.model,
            "reflection_lm": args.reflection_lm,
            "max_metric_calls_per_problem": args.max_metric_calls,
        },
        "problems": {pid: {"score": r["score"], "prompt_path": f"prompts/{pid}.txt"} for pid, r in results.items()},
        "full_prompts": {pid: r["prompt"] for pid, r in results.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log("-" * 60)
    log(f"Results saved to {output_dir}")
    log(f"  - {summary_path}: full summary")
    log(f"  - {output_dir / 'scores.json'}: scores only")
    log(f"  - {prompts_dir}/: one prompt file per problem")
    avg_score = sum(r["score"] for r in results.values()) / len(results) if results else 0
    log(f"Average score: {avg_score:.2f}/100")
    log("-" * 60)

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
