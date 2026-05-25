"""Unit tests for core data model and EvalRunner dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from clinical_llm_evals import EvalRunner, MockClient
from clinical_llm_evals.core import EvalCase, ScoringRubric


def test_eval_case_minimal_valid() -> None:
    case = EvalCase(
        id="demo_case",
        category="vaccine_scheduling",
        source="CDC ACIP MMWR 2024 — demo citation https://cdc.gov/",
        prompt="What is 2+2?",
        expected_behavior="Returns 4.",
        scoring_rubric=ScoringRubric(type="exact_match", expected="4"),
        severity="info",
    )
    assert case.id == "demo_case"


def test_eval_case_rejects_bad_id() -> None:
    with pytest.raises(ValueError):
        EvalCase(
            id="bad id with spaces",
            category="x",
            source="y",
            prompt="p",
            expected_behavior="e",
            scoring_rubric=ScoringRubric(type="exact_match", expected="z"),
            severity="info",
        )


def test_scoring_rubric_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        ScoringRubric(type="made_up_type")


def test_runner_dispatch_exact_match(tmp_path: Path) -> None:
    yaml_path = tmp_path / "x.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "id": "x",
                "category": "drug_interactions",
                "source": "FDA label https://fda.gov/",
                "prompt": "say hi",
                "expected_behavior": "says hi",
                "scoring_rubric": {"type": "exact_match", "expected": "hi"},
                "severity": "info",
            }
        )
    )
    case = EvalRunner.load_case(yaml_path)
    client = MockClient(canned={"say hi": "hi"})
    result = EvalRunner(client).run_case(case)
    assert result.passed
    assert result.case_id == "x"


def test_runner_handles_missing_rubric_fields(tmp_path: Path) -> None:
    """A rubric that lacks its required field should fail gracefully."""
    yaml_path = tmp_path / "broken.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "id": "broken",
                "category": "triage",
                "source": "FDA label https://fda.gov/",
                "prompt": "anything",
                "expected_behavior": "n/a",
                "scoring_rubric": {"type": "exact_match"},  # missing 'expected'
                "severity": "info",
            }
        )
    )
    case = EvalRunner.load_case(yaml_path)
    client = MockClient()
    result = EvalRunner(client).run_case(case)
    assert not result.passed
    assert "expected" in result.reason


def test_mock_client_default_fallback() -> None:
    client = MockClient()
    assert client.complete("nothing here matches anything ever") == client.default


def test_eval_case_rejects_unknown_field() -> None:
    """``extra="forbid"`` must catch typo'd YAML keys at parse time.

    Without this, a YAML with ``prompts:`` (instead of ``prompt:``) would
    silently fail validation only on the empty ``prompt`` — but worse, a
    misspelled ``severities:`` would skip the severity check entirely.
    """
    with pytest.raises(ValueError):
        EvalCase(
            id="x",
            category="triage",
            source="FDA label https://fda.gov/",
            prompt="p",
            expected_behavior="e",
            scoring_rubric=ScoringRubric(type="exact_match", expected="z"),
            severity="info",
            severities="critical",  # typo — should reject  # type: ignore[call-arg]
        )


def test_scoring_rubric_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        ScoringRubric(
            type="contains_all",
            must_include=["x"],
            must_includ=["typo"],  # type: ignore[call-arg]
        )
