"""Unit tests for the reporter formats."""

from __future__ import annotations

import json

from clinical_llm_evals.core import EvalResult
from clinical_llm_evals.report import (
    HTMLReporter,
    JSONReporter,
    MarkdownReporter,
    TextReporter,
    disable_color,
    get_reporter,
)


def _make_results() -> list[EvalResult]:
    return [
        EvalResult(
            case_id="case_a",
            category="triage",
            severity="critical",
            passed=True,
            response="ok",
            reason="contains required keywords",
            prompt="patient presents with...",
            source="AHA/ASA 2019 https://example.org/",
            latency_ms=12.4,
        ),
        EvalResult(
            case_id="case_b",
            category="triage",
            severity="warning",
            passed=False,
            response="bad answer ignoring contraindication",
            reason="missing required substrings: ['mmr']",
            prompt="when to defer MMR?",
            source="CDC ACIP",
            latency_ms=78.0,
        ),
        EvalResult(
            case_id="case_c",
            category="fhir_parsing",
            severity="info",
            passed=True,
            response='{"x": 1}',
            reason="JSON matches schema",
            prompt="parse this bundle",
            source="HL7 FHIR R4",
            latency_ms=5.1,
        ),
    ]


_META = {
    "run_id": "abc123def456",
    "started_at": "2026-05-16T12:00:00+00:00",
    "finished_at": "2026-05-16T12:00:01+00:00",
    "model": "mock",
    "suite_dir": "evals",
    "wall_ms": 95.5,
    "title": "test report",
}


def test_get_reporter_factory() -> None:
    assert isinstance(get_reporter("text"), TextReporter)
    assert isinstance(get_reporter("json"), JSONReporter)
    assert isinstance(get_reporter("markdown"), MarkdownReporter)
    assert isinstance(get_reporter("html"), HTMLReporter)


def test_get_reporter_unknown() -> None:
    try:
        get_reporter("yaml")
    except KeyError as e:
        assert "yaml" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_text_reporter_contains_case_ids() -> None:
    disable_color()
    out = TextReporter().render(_make_results(), _META)
    assert "case_a" in out
    assert "case_b" in out
    assert "PASS" in out
    assert "FAIL" in out
    # Summary table.
    assert "Summary" in out
    assert "triage" in out


def test_json_reporter_schema() -> None:
    out = JSONReporter().render(_make_results(), _META)
    payload = json.loads(out)
    assert payload["run_id"] == "abc123def456"
    assert payload["model"] == "mock"
    assert payload["n_cases"] == 3
    assert payload["n_pass"] == 2
    assert payload["n_fail"] == 1
    assert isinstance(payload["results"], list)
    first = payload["results"][0]
    for key in ("case_id", "category", "severity", "passed", "reason", "latency_ms", "prompt", "response", "source"):
        assert key in first, f"missing key {key} in result"


def test_json_reporter_truncates_long_response() -> None:
    """A 5KB response should be truncated to ~2000 chars."""
    big = "x" * 5000
    results = [
        EvalResult(
            case_id="big",
            category="x",
            severity="info",
            passed=True,
            response=big,
            reason="r",
            prompt="p",
            source="s",
            latency_ms=1.0,
        )
    ]
    payload = json.loads(JSONReporter().render(results, _META))
    assert len(payload["results"][0]["response"]) <= 2001  # 2000 + ellipsis char


def test_markdown_reporter_has_collapsed_details() -> None:
    out = MarkdownReporter().render(_make_results(), _META)
    assert "<details>" in out
    assert "</details>" in out
    assert "| category | passed | total | pass-rate |" in out
    assert "## Summary" in out
    # Should not include raw HTML for non-detail blocks.
    assert "## Cases" in out


def test_html_reporter_is_self_contained() -> None:
    """No external CDN references, embedded CSS+JS."""
    out = HTMLReporter().render(_make_results(), _META)
    assert "<!DOCTYPE html>" in out
    assert "<style>" in out
    assert "<script>" in out
    # No external assets — no http(s):// in src/href attrs.
    assert 'src="http' not in out
    assert 'href="http' not in out
    # Case IDs render.
    assert "case_a" in out
    assert "case_b" in out
    # HTML-escape user content.
    assert "&middot;" in out or "model" in out


def test_html_reporter_escapes_user_content() -> None:
    """Untrusted strings must be HTML-escaped, not interpolated raw."""
    nasty = [
        EvalResult(
            case_id="xss",
            category="x",
            severity="info",
            passed=True,
            response="<script>alert(1)</script>",
            reason="<img onerror=x>",
            prompt="<b>p</b>",
            source="<a>s</a>",
            latency_ms=1.0,
        )
    ]
    out = HTMLReporter().render(nasty, _META)
    # The raw <script>alert tag must not survive as an executable child.
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;alert(1)" in out
