"""Unit tests for the snapshot diff module."""

from __future__ import annotations

import json
from pathlib import Path

from clinical_llm_evals.diff import (
    LATENCY_DRIFT_THRESHOLD,
    SEVERITY_WEIGHTS,
    compute_diff,
    main as diff_main,
    render_diff_markdown,
)


def _report(results: list[dict], model: str = "mock", run_id: str = "x") -> dict:
    return {
        "run_id": run_id,
        "model": model,
        "started_at": "2026-05-16T00:00:00+00:00",
        "n_cases": len(results),
        "n_pass": sum(1 for r in results if r["passed"]),
        "n_fail": sum(1 for r in results if not r["passed"]),
        "results": results,
    }


def _r(cid: str, passed: bool, severity: str = "info", latency: float = 10.0) -> dict:
    return {
        "case_id": cid,
        "category": "triage",
        "severity": severity,
        "passed": passed,
        "latency_ms": latency,
    }


def test_diff_no_change() -> None:
    base = _report([_r("a", True), _r("b", True)])
    cur = _report([_r("a", True), _r("b", True)])
    d = compute_diff(base, cur)
    assert d.regressions == []
    assert d.improvements == []
    assert d.new_cases == []
    assert d.removed_cases == []
    assert d.weighted_score_delta == 0.0


def test_diff_detects_regression() -> None:
    base = _report([_r("a", True), _r("b", True, severity="critical")])
    cur = _report([_r("a", True), _r("b", False, severity="critical")])
    d = compute_diff(base, cur)
    assert len(d.regressions) == 1
    assert d.regressions[0].case_id == "b"
    # Critical weight is 9.
    assert d.weighted_score_delta == -SEVERITY_WEIGHTS["critical"]
    assert d.has_regressions


def test_diff_detects_improvement() -> None:
    base = _report([_r("a", False)])
    cur = _report([_r("a", True)])
    d = compute_diff(base, cur)
    assert len(d.improvements) == 1
    assert d.improvements[0].case_id == "a"
    assert d.weighted_score_delta == SEVERITY_WEIGHTS["info"]


def test_diff_new_and_removed_cases() -> None:
    base = _report([_r("a", True), _r("b", True)])
    cur = _report([_r("a", True), _r("c", True)])
    d = compute_diff(base, cur)
    assert [n.case_id for n in d.new_cases] == ["c"]
    assert [n.case_id for n in d.removed_cases] == ["b"]


def test_latency_drift_flagged_only_above_threshold() -> None:
    base = _report([_r("a", True, latency=100.0), _r("b", True, latency=100.0)])
    # a: +5% (noise, below threshold); b: +50% (above threshold).
    drift_pct = LATENCY_DRIFT_THRESHOLD - 0.15  # 5%
    cur = _report([
        _r("a", True, latency=100.0 * (1 + drift_pct)),
        _r("b", True, latency=150.0),
    ])
    d = compute_diff(base, cur)
    ids = [c.case_id for c in d.latency_drift]
    assert "b" in ids
    assert "a" not in ids


def test_diff_markdown_renders_all_sections() -> None:
    base = _report([_r("a", True), _r("b", True, severity="warning")])
    cur = _report([_r("a", True), _r("b", False, severity="warning"), _r("c", True)])
    md = render_diff_markdown(compute_diff(base, cur))
    assert "# Snapshot diff" in md
    assert "## Regressions (1)" in md
    assert "## Improvements (0)" in md
    assert "## New cases (1)" in md
    assert "## Removed cases (0)" in md
    assert "Weighted score change" in md


def test_diff_cli_writes_output(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    cur_path = tmp_path / "cur.json"
    out_path = tmp_path / "diff.md"
    base_path.write_text(json.dumps(_report([_r("a", True)])))
    cur_path.write_text(json.dumps(_report([_r("a", False)])))

    rc = diff_main([
        "--baseline", str(base_path),
        "--current", str(cur_path),
        "--output", str(out_path),
        "--fail-on-regression",
    ])
    assert rc == 1, "regression should produce non-zero exit code"
    md = out_path.read_text()
    assert "Regressions (1)" in md


def test_diff_cli_no_regression_exit_zero(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    cur_path = tmp_path / "cur.json"
    base_path.write_text(json.dumps(_report([_r("a", True)])))
    cur_path.write_text(json.dumps(_report([_r("a", True)])))
    rc = diff_main([
        "--baseline", str(base_path),
        "--current", str(cur_path),
        "--output", str(tmp_path / "d.md"),
        "--fail-on-regression",
    ])
    assert rc == 0
