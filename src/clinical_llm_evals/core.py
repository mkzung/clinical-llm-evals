"""Core data model and runner for clinical-llm-evals.

An ``EvalCase`` is the unit of evaluation: a prompt, an expected behavior in
plain English, a machine-checkable ``scoring_rubric``, a citation to a real
clinical source, and a severity tag.

The ``EvalRunner`` walks a directory of YAML files, loads each into an
``EvalCase``, sends the prompt to an ``LLMClient``, applies the rubric, and
returns a list of ``EvalResult``.

Two runners are provided:

- :class:`EvalRunner`        — synchronous; runs cases serially. Default.
- :class:`AsyncEvalRunner`   — async; runs cases concurrently under a
                               semaphore. Opt-in via ``--parallel`` in the CLI.

Concurrency note: most real LLM adapters are IO-bound, so the CLI also exposes
a thread-pool path. The async runner is provided for users wiring an
``AsyncLLMClient`` (HTTPX, aiohttp, etc.) directly.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, Field, field_validator

from clinical_llm_evals import registry
from clinical_llm_evals.llm import LLMClient

Severity = Literal["info", "warning", "critical"]

# Scoring strategies the rubric ``type`` field is allowed to take. Built-in
# strategies are kept here as the validation allow-list; the registry holds
# the implementations (and can be extended with new entries at runtime).
SCORING_TYPES = {
    "exact_match",
    "contains_all",
    "contains_any",
    "json_schema_match",
    "llm_judge",
}


class ScoringRubric(BaseModel):
    """Machine-checkable description of pass/fail for an ``EvalCase``.

    Examples
    --------
    ``contains_all``:
        ``{"type": "contains_all", "must_include": ["8 months", "MMR"]}``

    ``json_schema_match``:
        ``{"type": "json_schema_match", "schema": {...}}``

    ``llm_judge``:
        ``{"type": "llm_judge", "criteria": "The response defers the live..."}``
    """

    type: str = Field(..., description="Scoring strategy identifier.")
    must_include: list[str] | None = Field(
        default=None,
        description="For contains_all / contains_any: substrings that must appear.",
    )
    must_not_include: list[str] | None = Field(
        default=None,
        description="Substrings that must NOT appear (used by any string-based rubric).",
    )
    expected: str | None = Field(
        default=None, description="For exact_match: the expected response string."
    )
    schema_: dict[str, Any] | None = Field(
        default=None,
        alias="schema",
        description="For json_schema_match: a JSON Schema the response must satisfy.",
    )
    criteria: str | None = Field(
        default=None,
        description="For llm_judge: natural-language pass criteria for the judge model.",
    )

    model_config = {"populate_by_name": True}

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        # Allow either a built-in (SCORING_TYPES) or any name added to the
        # runtime registry via @register_scorer. Validating against
        # SCORING_TYPES alone would silently break user extensions.
        try:
            available = set(registry.available_scorers())
        except Exception:  # noqa: BLE001 — defensive; registry import side-effects
            available = set()
        if v not in SCORING_TYPES and v not in available:
            raise ValueError(
                f"Unknown scoring type {v!r}. Must be one of: {sorted(SCORING_TYPES | available)}"
            )
        return v


class EvalCase(BaseModel):
    """A single clinical-LLM evaluation case.

    Every case MUST cite a retrievable clinical source. See CONTRIBUTING.md.
    """

    id: str = Field(..., description="Unique snake_case identifier.")
    category: str = Field(..., description="Top-level category, matches folder name.")
    source: str = Field(
        ...,
        description=(
            "Citation to a real, retrievable source: CDC ACIP MMWR, FDA label, "
            "IDSA guideline, ACOG bulletin, peer-reviewed paper, etc. "
            "Include a URL or DOI if available."
        ),
    )
    prompt: str = Field(..., description="Prompt sent to the LLM under test.")
    expected_behavior: str = Field(
        ...,
        description="Plain-English description of what a safe, correct answer looks like.",
    )
    scoring_rubric: ScoringRubric
    severity: Severity = Field(
        ...,
        description=(
            "Clinical severity if the model fails this case. "
            "info = stylistic/edge; warning = clinically suboptimal; "
            "critical = patient-harm potential."
        ),
    )
    references: list[str] = Field(
        default_factory=list,
        description="Optional additional references (URLs, DOIs, guideline IDs).",
    )

    @field_validator("id")
    @classmethod
    def _snake_case_id(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"id {v!r} must be snake_case alphanumeric")
        return v


class EvalResult(BaseModel):
    """Outcome of running a single ``EvalCase`` against an ``LLMClient``."""

    case_id: str
    category: str
    severity: Severity
    passed: bool
    response: str
    reason: str = ""
    # Extra context — defaults make these backward-compatible with code that
    # constructs an EvalResult by-hand (e.g. tests).
    prompt: str = ""
    source: str = ""
    latency_ms: float = 0.0

    def __str__(self) -> str:  # pragma: no cover - convenience only
        tag = "PASS" if self.passed else "FAIL"
        return f"[{tag}] {self.category}/{self.case_id} ({self.severity}): {self.reason}"


@runtime_checkable
class AsyncLLMClient(Protocol):
    """Async counterpart to :class:`LLMClient`.

    Implementing this is opt-in. The default Anthropic/OpenAI adapters stay
    sync (their Python SDKs are sync); to wire a true async client, wrap your
    HTTP/SDK call in an ``async def complete(prompt: str) -> str`` method.
    Example::

        class MyAsyncClient:
            async def complete(self, prompt: str) -> str:
                async with httpx.AsyncClient() as h:
                    r = await h.post(URL, json={...}, timeout=60)
                    return r.json()["text"]
    """

    async def complete(self, prompt: str) -> str:  # pragma: no cover - protocol
        ...


class EvalRunner:
    """Loads YAML eval files and runs them against an ``LLMClient``.

    Parameters
    ----------
    client:
        Any object implementing the :class:`LLMClient` protocol.
    judge_client:
        Optional second client used by ``llm_judge`` rubrics. Defaults to ``client``.
    """

    def __init__(self, client: LLMClient, judge_client: LLMClient | None = None) -> None:
        self.client = client
        self.judge_client = judge_client or client

    # -- loading ---------------------------------------------------------------

    @staticmethod
    def load_case(path: str | Path) -> EvalCase:
        """Parse a single YAML file into an :class:`EvalCase`."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(data)}")
        return EvalCase.model_validate(data)

    @classmethod
    def load_directory(cls, root: str | Path) -> list[EvalCase]:
        """Recursively load every ``*.yaml`` / ``*.yml`` file under ``root``."""
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(root)
        cases: list[EvalCase] = []
        for ext in ("*.yaml", "*.yml"):
            for path in sorted(root.rglob(ext)):
                cases.append(cls.load_case(path))
        return cases

    # -- running ---------------------------------------------------------------

    def run_case(self, case: EvalCase) -> EvalResult:
        """Send ``case.prompt`` to the client and score the response."""
        t0 = time.perf_counter()
        response = self.client.complete(case.prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        passed, reason = self._score(case, response)
        return EvalResult(
            case_id=case.id,
            category=case.category,
            severity=case.severity,
            passed=passed,
            response=response,
            reason=reason,
            prompt=case.prompt,
            source=case.source,
            latency_ms=latency_ms,
        )

    def run_all(self, cases: Iterable[EvalCase]) -> list[EvalResult]:
        return [self.run_case(c) for c in cases]

    def run_parallel(
        self, cases: Iterable[EvalCase], max_workers: int = 4
    ) -> list[EvalResult]:
        """Run cases concurrently via a thread pool.

        Suitable for the common case where the underlying client is IO-bound
        (a real HTTP-backed LLM SDK). Preserves input order in the returned
        list. ``max_workers <= 1`` falls back to the sequential path.
        """
        cases = list(cases)
        if max_workers <= 1 or len(cases) <= 1:
            return self.run_all(cases)

        # Import lazily so users running purely sync don't pay the import cost.
        from concurrent.futures import ThreadPoolExecutor

        results: list[EvalResult | None] = [None] * len(cases)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.run_case, c): i for i, c in enumerate(cases)}
            for fut in futures:
                results[futures[fut]] = fut.result()
        # mypy: drop the Optional after the loop fills every slot.
        return [r for r in results if r is not None]

    # -- scoring dispatch ------------------------------------------------------

    def _score(self, case: EvalCase, response: str) -> tuple[bool, str]:
        """Route through the registry so user-registered scorers are picked up."""
        rtype = case.scoring_rubric.type
        try:
            scorer = registry.get_scorer(rtype)
        except KeyError as exc:
            return False, str(exc)
        ctx = registry.ScorerContext(judge_client=self.judge_client)
        return scorer(case, response, ctx)


class AsyncEvalRunner:
    """Async counterpart to :class:`EvalRunner` for true async clients.

    Use this when your client implements :class:`AsyncLLMClient`. Concurrency
    is bounded by an ``asyncio.Semaphore`` so a 1000-case suite doesn't
    fire 1000 simultaneous HTTP requests.

    Scoring is still done synchronously after the response arrives — the
    scorers themselves are cheap CPU work and don't benefit from being async.
    """

    def __init__(
        self,
        client: AsyncLLMClient,
        judge_client: LLMClient | None = None,
        concurrency: int = 4,
    ) -> None:
        self.client = client
        self.judge_client = judge_client
        self.concurrency = max(1, concurrency)

    async def run_case(self, case: EvalCase) -> EvalResult:
        t0 = time.perf_counter()
        response = await self.client.complete(case.prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        passed, reason = self._score(case, response)
        return EvalResult(
            case_id=case.id,
            category=case.category,
            severity=case.severity,
            passed=passed,
            response=response,
            reason=reason,
            prompt=case.prompt,
            source=case.source,
            latency_ms=latency_ms,
        )

    async def run_all(self, cases: Iterable[EvalCase]) -> list[EvalResult]:
        sem = asyncio.Semaphore(self.concurrency)

        async def _bounded(c: EvalCase) -> EvalResult:
            async with sem:
                return await self.run_case(c)

        return await asyncio.gather(*[_bounded(c) for c in cases])

    def _score(self, case: EvalCase, response: str) -> tuple[bool, str]:
        rtype = case.scoring_rubric.type
        try:
            scorer = registry.get_scorer(rtype)
        except KeyError as exc:
            return False, str(exc)
        ctx = registry.ScorerContext(judge_client=self.judge_client)
        return scorer(case, response, ctx)
