"""Lightweight scorer registry.

Inspired by the pattern used in ``openai/evals`` and ``lm-evaluation-harness``:
contributors can add a custom scoring strategy without editing the core
``EvalRunner._score`` dispatch table. Register a function with
``@register_scorer("my_type")`` and reference it from a YAML rubric's
``type`` field.

A scorer is a callable with the signature::

    def my_scorer(case: EvalCase, response: str, ctx: ScorerContext) -> tuple[bool, str]:
        ...

It returns ``(passed, reason)``. The ``ctx`` carries the judge client and any
runtime knobs the scorer may need (kept deliberately small so the registry
stays the contract — not the runner internals).

Built-in scorers (``exact_match``, ``contains_all``, ``contains_any``,
``json_schema_match``, ``llm_judge``) are registered at import time so
``EvalRunner`` can route every rubric through this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol

from clinical_llm_evals import scoring

if TYPE_CHECKING:
    from clinical_llm_evals.core import EvalCase
    from clinical_llm_evals.llm import LLMClient


@dataclass
class ScorerContext:
    """Carries shared resources scorers may need at evaluation time.

    Currently only the judge client. Kept as a dataclass so future fields
    (rate-limit budget, structured-output schema cache, etc.) can be added
    without breaking the registered-scorer signature.
    """

    judge_client: "LLMClient | None" = None


class Scorer(Protocol):
    """The callable shape every registered scorer must implement."""

    def __call__(
        self, case: "EvalCase", response: str, ctx: ScorerContext
    ) -> tuple[bool, str]:  # pragma: no cover - protocol
        ...


_REGISTRY: dict[str, Scorer] = {}


def register_scorer(name: str) -> Callable[[Scorer], Scorer]:
    """Decorator: register ``fn`` under ``name`` so YAML rubrics can target it.

    Re-registering a name is allowed (it overwrites) — useful for tests and
    monkey-patching, but contributors should pick a unique name.
    """

    def _wrap(fn: Scorer) -> Scorer:
        _REGISTRY[name] = fn
        return fn

    return _wrap


def get_scorer(name: str) -> Scorer:
    """Look up a scorer by name. Raises ``KeyError`` with the full list on miss."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown scorer {name!r}. Available scorers: {available}")
    return _REGISTRY[name]


def available_scorers() -> list[str]:
    """Sorted list of registered scorer names — for ``--help`` text and docs."""
    return sorted(_REGISTRY)


# ----------------------------------------------------------------------------
# Built-in scorers
# ----------------------------------------------------------------------------
#
# Each adapter pulls fields off ``case.scoring_rubric`` and forwards them to
# the underlying scoring.* function. The adapters intentionally do not raise
# on missing fields — they return ``(False, "<reason>")`` so a malformed YAML
# surfaces as a failing case rather than a crashed run.


@register_scorer("exact_match")
def _scorer_exact_match(
    case: "EvalCase", response: str, ctx: ScorerContext
) -> tuple[bool, str]:
    rubric = case.scoring_rubric
    if rubric.expected is None:
        return False, "rubric.exact_match requires 'expected'"
    return scoring.exact_match(response, rubric.expected)


@register_scorer("contains_all")
def _scorer_contains_all(
    case: "EvalCase", response: str, ctx: ScorerContext
) -> tuple[bool, str]:
    rubric = case.scoring_rubric
    return scoring.contains_all(
        response,
        rubric.must_include or [],
        must_not_include=rubric.must_not_include,
    )


@register_scorer("contains_any")
def _scorer_contains_any(
    case: "EvalCase", response: str, ctx: ScorerContext
) -> tuple[bool, str]:
    rubric = case.scoring_rubric
    return scoring.contains_any(
        response,
        rubric.must_include or [],
        must_not_include=rubric.must_not_include,
    )


@register_scorer("json_schema_match")
def _scorer_json_schema_match(
    case: "EvalCase", response: str, ctx: ScorerContext
) -> tuple[bool, str]:
    rubric = case.scoring_rubric
    if rubric.schema_ is None:
        return False, "rubric.json_schema_match requires 'schema'"
    return scoring.json_schema_match(response, rubric.schema_)


@register_scorer("llm_judge")
def _scorer_llm_judge(
    case: "EvalCase", response: str, ctx: ScorerContext
) -> tuple[bool, str]:
    rubric = case.scoring_rubric
    if rubric.criteria is None:
        return False, "rubric.llm_judge requires 'criteria'"
    if ctx.judge_client is None:
        return False, "rubric.llm_judge requires a judge_client (none configured)"
    return scoring.llm_judge(ctx.judge_client, case.prompt, response, rubric.criteria)


def _builtin_scorers() -> dict[str, Scorer]:
    """Return a copy of the registry restricted to built-ins (for tests)."""
    return dict(_REGISTRY)


__all__ = [
    "Scorer",
    "ScorerContext",
    "available_scorers",
    "get_scorer",
    "register_scorer",
]
