"""Snapshot diff: compare two JSON reports.

Used in two places:
- ``EvalRunner --baseline path.json`` exits non-zero on a regression.
- ``python -m clinical_llm_evals.diff --baseline a.json --current b.json``
  writes a structured markdown summary.

A *regression* is any case that passed in the baseline and fails in the
current run. We also surface improvements (newly-passing), latency drift
>20%, and a severity-tier-weighted score delta.

Severity weights (chosen to mirror the CLI ``--fail-on`` ranks):

    info: 1
    warning: 3
    critical: 9

So a single critical-tier regression costs as much as 9 info-tier ones,
which roughly matches the patient-harm framing in CONTRIBUTING.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


SEVERITY_WEIGHTS: dict[str, int] = {"info": 1, "warning": 3, "critical": 9}

# Threshold (fractional) for flagging latency drift in the diff. 20% mirrors
# the spec — anything inside that band is considered noise.
LATENCY_DRIFT_THRESHOLD = 0.20


@dataclass
class CaseDiff:
    """A single case's pass/latency change between two reports."""

    case_id: str
    category: str
    severity: str
    baseline_passed: bool | None  # None = absent from baseline
    current_passed: bool | None  # None = absent from current
    baseline_latency_ms: float | None
    current_latency_ms: float | None

    @property
    def is_regression(self) -> bool:
        return self.baseline_passed is True and self.current_passed is False

    @property
    def is_improvement(self) -> bool:
        return self.baseline_passed is False and self.current_passed is True

    @property
    def latency_delta_pct(self) -> float | None:
        if self.baseline_latency_ms is None or self.current_latency_ms is None:
            return None
        if self.baseline_latency_ms <= 0:
            return None
        return (self.current_latency_ms - self.baseline_latency_ms) / self.baseline_latency_ms


@dataclass
class DiffReport:
    """Aggregate diff of two report JSONs."""

    regressions: list[CaseDiff] = field(default_factory=list)
    improvements: list[CaseDiff] = field(default_factory=list)
    new_cases: list[CaseDiff] = field(default_factory=list)
    removed_cases: list[CaseDiff] = field(default_factory=list)
    latency_drift: list[CaseDiff] = field(default_factory=list)
    weighted_score_delta: float = 0.0
    baseline_meta: dict = field(default_factory=dict)
    current_meta: dict = field(default_factory=dict)

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)


def _load_report(path: Path) -> dict:
    """Load a JSON report from disk with a helpful error message."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"could not read report {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict) or "results" not in data:
        raise SystemExit(f"{path} does not look like a clinical-llm-evals report")
    return data


def _index_by_id(report: dict) -> dict[str, dict]:
    return {r["case_id"]: r for r in report.get("results", []) if "case_id" in r}


def _weighted_score(results: list[dict]) -> float:
    """Sum of severity weights for *passed* cases. Higher is better."""
    return float(
        sum(
            SEVERITY_WEIGHTS.get(r.get("severity", "info"), 1)
            for r in results
            if r.get("passed")
        )
    )


def compute_diff(baseline: dict, current: dict) -> DiffReport:
    """Diff two parsed report dicts.

    Both inputs must follow the JSON schema emitted by
    :class:`clinical_llm_evals.report.JSONReporter`.
    """
    b_idx = _index_by_id(baseline)
    c_idx = _index_by_id(current)

    report = DiffReport(
        baseline_meta={k: baseline.get(k) for k in ("run_id", "model", "started_at")},
        current_meta={k: current.get(k) for k in ("run_id", "model", "started_at")},
    )

    all_ids = sorted(set(b_idx) | set(c_idx))
    for cid in all_ids:
        b = b_idx.get(cid)
        c = c_idx.get(cid)
        cat = (c or b or {}).get("category", "")
        sev = (c or b or {}).get("severity", "info")
        diff = CaseDiff(
            case_id=cid,
            category=cat,
            severity=sev,
            baseline_passed=b.get("passed") if b else None,
            current_passed=c.get("passed") if c else None,
            baseline_latency_ms=b.get("latency_ms") if b else None,
            current_latency_ms=c.get("latency_ms") if c else None,
        )
        if b is None and c is not None:
            report.new_cases.append(diff)
        elif c is None and b is not None:
            report.removed_cases.append(diff)
        else:
            if diff.is_regression:
                report.regressions.append(diff)
            elif diff.is_improvement:
                report.improvements.append(diff)
        d = diff.latency_delta_pct
        if d is not None and abs(d) >= LATENCY_DRIFT_THRESHOLD:
            report.latency_drift.append(diff)

    report.weighted_score_delta = (
        _weighted_score(list(c_idx.values())) - _weighted_score(list(b_idx.values()))
    )
    return report


def render_diff_markdown(diff: DiffReport) -> str:
    """Render a DiffReport as GitHub-flavored markdown."""
    out: list[str] = []
    out.append("# Snapshot diff")
    out.append("")
    b_model = diff.baseline_meta.get("model") or "?"
    c_model = diff.current_meta.get("model") or "?"
    out.append(f"- **baseline:** `{b_model}` (run `{diff.baseline_meta.get('run_id', '?')}`)")
    out.append(f"- **current:**  `{c_model}` (run `{diff.current_meta.get('run_id', '?')}`)")
    out.append("")

    score_sign = "+" if diff.weighted_score_delta >= 0 else ""
    out.append(
        f"## Weighted score change: **{score_sign}{diff.weighted_score_delta:.1f}** "
        "(severity-weighted: info=1, warning=3, critical=9)"
    )
    out.append("")

    def _table(title: str, rows: list[CaseDiff]) -> None:
        out.append(f"## {title} ({len(rows)})")
        out.append("")
        if not rows:
            out.append("_none_")
            out.append("")
            return
        out.append("| case | category | severity | baseline | current | latency |")
        out.append("|---|---|---|---|---|---|")
        for r in rows:
            b_tag = "PASS" if r.baseline_passed else ("FAIL" if r.baseline_passed is False else "—")
            c_tag = "PASS" if r.current_passed else ("FAIL" if r.current_passed is False else "—")
            if r.baseline_latency_ms is not None and r.current_latency_ms is not None:
                lat = f"{r.baseline_latency_ms:.0f} → {r.current_latency_ms:.0f} ms"
            elif r.current_latency_ms is not None:
                lat = f"{r.current_latency_ms:.0f} ms"
            else:
                lat = "—"
            out.append(
                f"| `{r.case_id}` | `{r.category}` | `{r.severity}` | "
                f"{b_tag} | {c_tag} | {lat} |"
            )
        out.append("")

    _table("Regressions", diff.regressions)
    _table("Improvements", diff.improvements)
    _table("New cases", diff.new_cases)
    _table("Removed cases", diff.removed_cases)

    if diff.latency_drift:
        out.append(f"## Latency drift > {int(LATENCY_DRIFT_THRESHOLD*100)}% ({len(diff.latency_drift)})")
        out.append("")
        out.append("| case | baseline | current | delta |")
        out.append("|---|---|---|---|")
        for r in diff.latency_drift:
            d = (r.latency_delta_pct or 0.0) * 100
            sign = "+" if d >= 0 else ""
            out.append(
                f"| `{r.case_id}` | {r.baseline_latency_ms:.0f} ms | "
                f"{r.current_latency_ms:.0f} ms | {sign}{d:.0f}% |"
            )
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clinical-llm-evals-diff",
        description="Diff two JSON reports from clinical_llm_evals.report.JSONReporter.",
    )
    parser.add_argument("--baseline", type=Path, required=True, help="Older JSON report.")
    parser.add_argument("--current", type=Path, required=True, help="Newer JSON report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the markdown diff here. Defaults to stdout.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any regression is found.",
    )
    args = parser.parse_args(argv)

    baseline = _load_report(args.baseline)
    current = _load_report(args.current)
    diff = compute_diff(baseline, current)
    md = render_diff_markdown(diff)
    if args.output:
        args.output.write_text(md, encoding="utf-8")
    else:
        sys.stdout.write(md + "\n")
    if args.fail_on_regression and diff.has_regressions:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
