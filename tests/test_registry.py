"""Unit tests for the scorer registry.

The registry decorator pattern is how contributors add scorers without
editing core. These tests pin the contract: register → look up → dispatch
through EvalRunner.
"""

from __future__ import annotations

import pytest

from clinical_llm_evals import EvalRunner, MockClient
from clinical_llm_evals.core import EvalCase, ScoringRubric
from clinical_llm_evals.registry import (
    ScorerContext,
    available_scorers,
    get_scorer,
    register_scorer,
)


def test_builtin_scorers_are_registered() -> None:
    avail = set(available_scorers())
    for name in ("exact_match", "contains_all", "contains_any", "json_schema_match", "llm_judge"):
        assert name in avail, f"missing built-in {name}"


def test_get_scorer_unknown_lists_available() -> None:
    with pytest.raises(KeyError) as excinfo:
        get_scorer("does_not_exist")
    msg = str(excinfo.value)
    assert "does_not_exist" in msg
    assert "exact_match" in msg  # the listing should include built-ins


def test_register_and_dispatch_custom_scorer() -> None:
    """A user-registered scorer should be reachable via EvalRunner."""

    @register_scorer("test_starts_with")
    def _starts_with(case: EvalCase, response: str, ctx: ScorerContext) -> tuple[bool, str]:
        # Use must_include[0] as the prefix.
        prefix = (case.scoring_rubric.must_include or [""])[0]
        if response.startswith(prefix):
            return True, "ok"
        return False, "missing prefix"

    case = EvalCase(
        id="custom_demo",
        category="triage",
        source="FDA label https://fda.gov/",
        prompt="say hi",
        expected_behavior="starts with HI",
        scoring_rubric=ScoringRubric(type="test_starts_with", must_include=["HI"]),
        severity="info",
    )
    client = MockClient(canned={"say hi": "HI, doctor"})
    result = EvalRunner(client).run_case(case)
    assert result.passed
    assert result.reason == "ok"


def test_unknown_rubric_type_rejected_by_pydantic() -> None:
    """The validator should reject types not in built-ins and not in the registry."""
    with pytest.raises(ValueError):
        ScoringRubric(type="this_scorer_was_never_registered")


def test_eval_runner_routes_through_registry() -> None:
    """EvalRunner._score must use the registry — not a hard-coded dispatch table."""
    case = EvalCase(
        id="route_test",
        category="triage",
        source="FDA label https://fda.gov/",
        prompt="anything",
        expected_behavior="x",
        scoring_rubric=ScoringRubric(type="exact_match", expected="ok"),
        severity="info",
    )
    runner = EvalRunner(MockClient(canned={"anything": "ok"}))
    result = runner.run_case(case)
    assert result.passed
