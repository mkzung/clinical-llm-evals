"""Smoke tests for the upgraded CLI in clinical_llm_evals.run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_llm_evals.run import _apply_filters, main


REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"


def _common(*args: str, output: Path) -> list[str]:
    return [
        "--suite", str(EVALS_DIR),
        "--model", "mock",
        "--output", str(output),
        *args,
    ]


def test_cli_text_report(tmp_path: Path) -> None:
    out = tmp_path / "report.txt"
    rc = main(_common("--report-format", "text", output=out))
    assert rc == 0
    text = out.read_text()
    assert "Summary" in text
    assert "PASS" in text


def test_cli_json_report_schema(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    rc = main(_common("--report-format", "json", output=out))
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["model"] == "mock"
    assert "run_id" in payload
    assert "started_at" in payload
    assert "finished_at" in payload
    assert payload["n_cases"] > 0
    assert payload["n_pass"] == payload["n_cases"]  # MockClient passes all


def test_cli_markdown_report(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    rc = main(_common("--report-format", "markdown", output=out))
    assert rc == 0
    md = out.read_text()
    assert "# clinical-llm-evals" in md
    assert "## Summary" in md
    assert "## By category" in md
    assert "<details>" in md


def test_cli_html_report_self_contained(tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    rc = main(_common("--report-format", "html", output=out))
    assert rc == 0
    html = out.read_text()
    assert "<!DOCTYPE html>" in html
    assert "<style>" in html
    assert 'src="http' not in html  # no external assets
    assert 'href="http' not in html


def test_cli_baseline_no_regression(tmp_path: Path) -> None:
    """Running twice in a row against the same model must not regress."""
    baseline = tmp_path / "baseline.json"
    assert main(_common("--report-format", "json", output=baseline)) == 0
    out = tmp_path / "current.json"
    rc = main([
        "--suite", str(EVALS_DIR),
        "--model", "mock",
        "--output", str(out),
        "--report-format", "json",
        "--baseline", str(baseline),
    ])
    assert rc == 0


def test_cli_filter_by_category(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    rc = main([
        "--suite", str(EVALS_DIR),
        "--model", "mock",
        "--report-format", "json",
        "--output", str(out),
        "--filter", "category=triage",
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    cats = {r["category"] for r in payload["results"]}
    assert cats == {"triage"}


def test_cli_parallel_smoke(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    rc = main([
        "--suite", str(EVALS_DIR),
        "--model", "mock",
        "--report-format", "json",
        "--output", str(out),
        "--parallel", "4",
    ])
    assert rc == 0


def test_apply_filters_unknown_key_errors() -> None:
    """A typo in --filter should fail fast with a helpful message."""
    from clinical_llm_evals.core import EvalCase, ScoringRubric

    cases = [
        EvalCase(
            id="x",
            category="triage",
            source="FDA label https://fda.gov/",
            prompt="p",
            expected_behavior="e",
            scoring_rubric=ScoringRubric(type="exact_match", expected="z"),
            severity="info",
        )
    ]
    with pytest.raises(SystemExit):
        _apply_filters(cases, "categori=triage")  # typo


def test_apply_filters_category_subsets() -> None:
    from clinical_llm_evals.core import EvalCase, ScoringRubric

    def _mk(cat: str, cid: str) -> EvalCase:
        return EvalCase(
            id=cid,
            category=cat,
            source="FDA label https://fda.gov/",
            prompt="p",
            expected_behavior="e",
            scoring_rubric=ScoringRubric(type="exact_match", expected="z"),
            severity="info",
        )

    cases = [_mk("triage", "a"), _mk("triage", "b"), _mk("drug_interactions", "c")]
    filtered = _apply_filters(cases, "category=triage")
    assert {c.id for c in filtered} == {"a", "b"}


def test_apply_filters_severity() -> None:
    from clinical_llm_evals.core import EvalCase, ScoringRubric

    def _mk(sev: str, cid: str) -> EvalCase:
        return EvalCase(
            id=cid,
            category="triage",
            source="FDA label https://fda.gov/",
            prompt="p",
            expected_behavior="e",
            scoring_rubric=ScoringRubric(type="exact_match", expected="z"),
            severity=sev,  # type: ignore[arg-type]
        )

    cases = [_mk("info", "a"), _mk("critical", "b"), _mk("warning", "c")]
    filtered = _apply_filters(cases, "severity=critical,warning")
    assert {c.id for c in filtered} == {"b", "c"}
