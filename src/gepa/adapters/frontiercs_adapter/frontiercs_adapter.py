# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import requests

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


class FrontierCSDataInst(TypedDict):
    """Data instance for Frontier-CS algorithmic problems."""

    problem_id: str
    statement: str


class FrontierCSTrajectory(TypedDict):
    """Trajectory capturing execution for reflection."""

    data: FrontierCSDataInst
    generated_code: str
    judge_result: dict[str, Any]
    score: float
    feedback: str


class FrontierCSRolloutOutput(TypedDict):
    """Raw output from the code generation and judging pipeline."""

    generated_code: str
    judge_result: dict[str, Any]


def extract_cpp_code(response_text: str) -> str:
    """Extract C++ code from LLM response, handling ```cpp code blocks."""
    # Try common variations: ```cpp, ```CPP, ``` cpp, etc.
    match = re.search(r"```\s*cpp\s*\n(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response_text.strip()


class FrontierCSJudgeClient:
    """Client for the Frontier-CS judge API (LightCPVerifier)."""

    def __init__(self, judge_url: str = "http://localhost:8081"):
        self.judge_url = judge_url.rstrip("/")
        self.session = requests.Session()

    def get_problem_statement(self, pid: str) -> str | None:
        try:
            response = self.session.get(f"{self.judge_url}/problem/{pid}/statement", timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    def submit_solution(self, pid: str, code: str) -> str | None:
        try:
            files = {"code": ("solution.cpp", code)}
            data = {"pid": pid, "lang": "cpp"}
            response = self.session.post(
                f"{self.judge_url}/submit",
                files=files,
                data=data,
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("sid")
        except requests.RequestException:
            return None

    def get_result(self, sid: str, poll_interval: int = 2, timeout_seconds: int = 300) -> dict[str, Any] | None:
        """Poll for judge result until done or error."""
        start = time.time()
        while time.time() - start < timeout_seconds:
            try:
                response = self.session.get(f"{self.judge_url}/result/{sid}", timeout=10)
                if response.status_code == 404:
                    time.sleep(poll_interval)
                    continue
                response.raise_for_status()
                result = response.json()
                if result.get("status") in ("done", "error"):
                    return result
            except requests.RequestException:
                pass
            time.sleep(poll_interval)
        return {"status": "error", "error": "TIMEOUT", "score": 0}


class FrontierCSAdapter(GEPAAdapter[FrontierCSDataInst, FrontierCSTrajectory, FrontierCSRolloutOutput]):
    """GEPA adapter for Frontier-CS algorithmic competitive programming benchmark.

    Optimizes the system prompt used to instruct an LLM to generate C++ solutions.
    Each problem is submitted to a local judge server for evaluation.
    Requires the Frontier-CS judge to be running (e.g. docker-compose up -d).
    """

    def __init__(
        self,
        model: str,
        judge_url: str = "http://localhost:8081",
        max_litellm_workers: int = 4,
        litellm_batch_completion_kwargs: dict[str, Any] | None = None,
        judge_poll_interval: int = 2,
        judge_timeout_seconds: int = 300,
        verbose: bool = False,
    ):
        import litellm

        self.model = model
        self.judge = FrontierCSJudgeClient(judge_url)
        self.verbose = verbose
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs or {}
        self.judge_poll_interval = judge_poll_interval
        self.judge_timeout_seconds = judge_timeout_seconds
        self.litellm = litellm

    def evaluate(
        self,
        batch: list[FrontierCSDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[FrontierCSTrajectory, FrontierCSRolloutOutput]:
        system_prompt = next(iter(candidate.values()))

        outputs: list[FrontierCSRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[FrontierCSTrajectory] | None = [] if capture_traces else None

        for data in batch:
            statement = data["statement"]
            problem_id = data["problem_id"]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": statement},
            ]

            try:
                response = self.litellm.completion(
                    model=self.model,
                    messages=messages,
                    **self.litellm_batch_completion_kwargs,
                )
                llm_text = (response.choices[0].message.content or "")  # type: ignore[union-attr]
            except Exception as e:
                llm_text = ""
                judge_result = {"status": "error", "error": str(e), "score": 0}
                score = 0.0
                feedback = f"LLM call failed: {e}"
            else:
                code = extract_cpp_code(llm_text)
                if not code:
                    judge_result = {"status": "error", "error": "NO_CODE_EXTRACTED", "score": 0}
                    score = 0.0
                    feedback = "Failed to extract C++ code from LLM response. Ensure code is wrapped in ```cpp ... ```."
                else:
                    sid = self.judge.submit_solution(problem_id, code)
                    if not sid:
                        judge_result = {"status": "error", "error": "SUBMISSION_FAILED", "score": 0}
                        score = 0.0
                        feedback = "Failed to submit solution to judge. Is the judge server running?"
                    else:
                        judge_result = self.judge.get_result(
                            sid,
                            poll_interval=self.judge_poll_interval,
                            timeout_seconds=self.judge_timeout_seconds,
                        ) or {"status": "error", "error": "NO_RESULT", "score": 0}
                        raw_score = judge_result.get("score", -1)
                        score = max(0.0, min(1.0, raw_score / 100.0)) if raw_score >= 0 else 0.0
                        status = judge_result.get("status", "unknown")
                        result_str = judge_result.get("result", "")
                        cases = judge_result.get("cases", [])
                        if status == "done":
                            if judge_result.get("passed"):
                                feedback = f"All test cases passed. Score: {raw_score}/100."
                            else:
                                case_msgs = [
                                    f"Case {i + 1}: {(c.get('status', '?') if isinstance(c, dict) else '?')}"
                                    + (f" - {c.get('msg', '')}" if isinstance(c, dict) and c.get("msg") else "")
                                    for i, c in enumerate(cases[:5])
                                ]
                                feedback = f"Score: {raw_score}/100. Result: {result_str}. Case details: {'; '.join(case_msgs)}"
                        else:
                            err = judge_result.get("error", "Unknown error")
                            feedback = f"Judge error: {err}"

            if self.verbose and score == 0.0:
                err_info = judge_result.get("error") or judge_result.get("result") or feedback
                print(f"    [DEBUG] problem_id={problem_id} score=0: {err_info}", flush=True)

            output: FrontierCSRolloutOutput = {
                "generated_code": extract_cpp_code(llm_text) if llm_text else "",
                "judge_result": judge_result,
            }
            outputs.append(output)
            scores.append(score)

            if trajectories is not None:
                trajectories.append(
                    {
                        "data": data,
                        "generated_code": output["generated_code"],
                        "judge_result": judge_result,
                        "score": score,
                        "feedback": feedback,
                    }
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[FrontierCSTrajectory, FrontierCSRolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        ret: dict[str, list[dict[str, Any]]] = {}
        comp = components_to_update[0]

        trajectories = eval_batch.trajectories
        if trajectories is None:
            return {comp: []}

        items: list[dict[str, Any]] = []
        for traj in trajectories:
            items.append(
                {
                    "Inputs": traj["data"]["statement"],
                    "Generated Outputs": traj["generated_code"],
                    "Feedback": traj["feedback"],
                    "Score": traj["score"],
                    "Judge Result": traj["judge_result"],
                }
            )
        ret[comp] = items
        return ret
