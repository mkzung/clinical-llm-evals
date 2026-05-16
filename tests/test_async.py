"""Unit tests for AsyncEvalRunner and the parallel thread-pool path."""

from __future__ import annotations

import asyncio
import time

from clinical_llm_evals import AsyncEvalRunner, EvalRunner, MockClient
from clinical_llm_evals.core import EvalCase, ScoringRubric


def _make_cases(n: int) -> list[EvalCase]:
    return [
        EvalCase(
            id=f"case_{i}",
            category="triage",
            source="FDA label https://fda.gov/",
            prompt=f"echo-{i}",
            expected_behavior="echo",
            scoring_rubric=ScoringRubric(type="exact_match", expected=f"reply-{i}"),
            severity="info",
        )
        for i in range(n)
    ]


class _SlowMock:
    """A client that sleeps before responding — useful to verify parallelism."""

    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay

    def complete(self, prompt: str) -> str:
        time.sleep(self.delay)
        # echo-N → reply-N
        suffix = prompt.split("-")[-1] if "-" in prompt else "0"
        return f"reply-{suffix}"


def test_run_parallel_returns_ordered_results() -> None:
    cases = _make_cases(5)
    runner = EvalRunner(_SlowMock(delay=0.01))
    results = runner.run_parallel(cases, max_workers=4)
    assert [r.case_id for r in results] == [c.id for c in cases]
    assert all(r.passed for r in results)


def test_run_parallel_is_faster_than_serial() -> None:
    """Smoke test: parallel wall-time < serial wall-time for IO-bound work.

    Each call sleeps 30ms — 6 cases serially is ~180ms, with 4 workers ~60ms.
    We allow generous slack to avoid CI flakiness.
    """
    cases = _make_cases(6)
    client = _SlowMock(delay=0.03)
    runner = EvalRunner(client)

    t = time.perf_counter()
    runner.run_all(cases)
    serial = time.perf_counter() - t

    t = time.perf_counter()
    runner.run_parallel(cases, max_workers=4)
    parallel = time.perf_counter() - t

    # Parallel should be meaningfully faster. 0.7 is a soft bound for CI.
    assert parallel < serial * 0.7, f"serial={serial:.3f}s parallel={parallel:.3f}s"


def test_run_parallel_falls_back_to_serial_for_one_worker() -> None:
    cases = _make_cases(3)
    runner = EvalRunner(MockClient(canned={"echo-0": "reply-0", "echo-1": "reply-1", "echo-2": "reply-2"}))
    results = runner.run_parallel(cases, max_workers=1)
    assert len(results) == 3


# ----- AsyncEvalRunner ----------------------------------------------------


class _AsyncMock:
    def __init__(self, delay: float = 0.01) -> None:
        self.delay = delay

    async def complete(self, prompt: str) -> str:
        await asyncio.sleep(self.delay)
        suffix = prompt.split("-")[-1] if "-" in prompt else "0"
        return f"reply-{suffix}"


def test_async_runner_runs_all() -> None:
    cases = _make_cases(4)
    runner = AsyncEvalRunner(_AsyncMock(delay=0.005), concurrency=3)
    results = asyncio.run(runner.run_all(cases))
    assert len(results) == 4
    assert all(r.passed for r in results)
    assert {r.case_id for r in results} == {c.id for c in cases}


def test_async_runner_respects_concurrency() -> None:
    """With concurrency=2 and 4 cases of 20ms each, total ~40ms not ~10ms."""
    cases = _make_cases(4)
    runner = AsyncEvalRunner(_AsyncMock(delay=0.02), concurrency=2)
    t = time.perf_counter()
    asyncio.run(runner.run_all(cases))
    elapsed = time.perf_counter() - t
    # With true concurrency=2, the lower bound is ~2 * 20ms = 40ms.
    # Allow generous slack — CI machines vary.
    assert elapsed >= 0.03, f"elapsed={elapsed:.3f}s — concurrency cap not enforced?"


def test_result_has_latency_ms() -> None:
    cases = _make_cases(1)
    runner = EvalRunner(_SlowMock(delay=0.01))
    [result] = runner.run_all(cases)
    assert result.latency_ms > 5.0
