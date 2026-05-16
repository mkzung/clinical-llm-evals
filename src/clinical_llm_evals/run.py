"""Command-line entry point.

Examples
--------
::

    # Default: mock client, text report to stdout
    python -m clinical_llm_evals.run

    # Real model, markdown report saved to a file
    python -m clinical_llm_evals.run \\
        --model anthropic:claude-3-5-sonnet-latest \\
        --report-format markdown --output report.md

    # Only run two categories, in parallel, against a baseline
    python -m clinical_llm_evals.run \\
        --filter category=triage,vaccine_scheduling \\
        --parallel 8 \\
        --baseline last_run.json

Exit codes
~~~~~~~~~~
- ``0`` — all good
- ``1`` — one or more failures at or above ``--fail-on``
- ``2`` — a regression vs ``--baseline`` (overrides ``0``)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from clinical_llm_evals.core import EvalCase, EvalResult, EvalRunner
from clinical_llm_evals.llm import AnthropicClient, LLMClient, MockClient, OpenAIClient
from clinical_llm_evals.report import disable_color, get_reporter


def _build_client(model_spec: str) -> LLMClient:
    """Resolve ``provider:model`` strings into a concrete client.

    Examples
    --------
    - ``mock`` → :class:`MockClient`
    - ``anthropic:claude-3-5-sonnet-latest`` → :class:`AnthropicClient`
    - ``openai:gpt-4o-mini`` → :class:`OpenAIClient`
    """
    if model_spec == "mock":
        return MockClient()
    if ":" not in model_spec:
        raise SystemExit(f"unknown model spec {model_spec!r}; expected provider:model or 'mock'")
    provider, model = model_spec.split(":", 1)
    if provider == "anthropic":
        return AnthropicClient(model=model)
    if provider == "openai":
        return OpenAIClient(model=model)
    raise SystemExit(f"unsupported provider {provider!r}")


def _apply_filters(cases: list[EvalCase], filter_spec: str | None) -> list[EvalCase]:
    """Filter cases by a ``key=v1,v2`` spec. Only ``category`` is supported today.

    The spec is intentionally simple: a single comma-separated allow-list.
    Unknown keys raise a SystemExit with the list of supported keys so the
    user finds the typo fast.
    """
    if not filter_spec:
        return cases
    if "=" not in filter_spec:
        raise SystemExit(f"--filter expects key=v1,v2 — got {filter_spec!r}")
    key, raw = filter_spec.split("=", 1)
    key = key.strip()
    values = {v.strip() for v in raw.split(",") if v.strip()}
    if key == "category":
        return [c for c in cases if c.category in values]
    if key == "severity":
        return [c for c in cases if c.severity in values]
    if key == "id":
        return [c for c in cases if c.id in values]
    raise SystemExit(
        f"--filter key {key!r} not recognized; supported keys: category, severity, id"
    )


def _summarize_text(results: list[EvalResult], fail_on: str) -> int:
    """Count failures at or above ``fail_on`` severity (terminal use)."""
    severity_rank = {"info": 0, "warning": 1, "critical": 2}
    threshold = {"any": 0, "warning": 1, "critical": 2}[fail_on]
    return sum(
        1
        for r in results
        if not r.passed and severity_rank[r.severity] >= threshold
    )


def _check_baseline(
    baseline_path: Path, current_meta: dict, current_results: list[EvalResult]
) -> tuple[bool, str]:
    """Compare ``current_results`` against the JSON report at ``baseline_path``.

    Returns ``(has_regression, summary)``.
    """
    # Import locally so users running without baselines pay no cost.
    from clinical_llm_evals.diff import compute_diff
    import json

    try:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"could not read baseline {baseline_path}: {exc}"

    # Build a "current" report dict in the same shape JSONReporter would emit.
    current = {
        "model": current_meta.get("model", ""),
        "run_id": current_meta.get("run_id", ""),
        "started_at": current_meta.get("started_at", ""),
        "results": [
            {
                "case_id": r.case_id,
                "category": r.category,
                "severity": r.severity,
                "passed": r.passed,
                "latency_ms": r.latency_ms,
            }
            for r in current_results
        ],
    }
    diff = compute_diff(baseline, current)
    parts = [
        f"baseline diff: regressions={len(diff.regressions)}",
        f"improvements={len(diff.improvements)}",
        f"new={len(diff.new_cases)}",
        f"removed={len(diff.removed_cases)}",
        f"weighted_score Δ={diff.weighted_score_delta:+.1f}",
    ]
    if diff.regressions:
        ids = ", ".join(d.case_id for d in diff.regressions[:5])
        more = "" if len(diff.regressions) <= 5 else f" (+{len(diff.regressions)-5} more)"
        parts.append(f"regressed: {ids}{more}")
    return diff.has_regressions, " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clinical-llm-evals",
        description="Run the clinical-llm-evals suite against an LLM client.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("evals"),
        help="Directory of YAML eval cases (recursively).",
    )
    parser.add_argument(
        "--model",
        default="mock",
        help="Model spec: 'mock', 'anthropic:<model>', or 'openai:<model>'.",
    )
    parser.add_argument(
        "--fail-on",
        choices=["any", "warning", "critical"],
        default="critical",
        help="Exit non-zero if any failure at or above this severity occurs.",
    )
    parser.add_argument(
        "--report-format",
        choices=["text", "json", "markdown", "html"],
        default="text",
        help="Output report format. Default: text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the report to this file. Default: stdout.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Run cases with N concurrent threads (default: 1 = serial).",
    )
    parser.add_argument(
        "--filter",
        default=None,
        metavar="key=v1,v2",
        help="Subset cases. Supported keys: category, severity, id. "
             "Example: --filter category=triage,vaccine_scheduling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed Python's RNG (passed where applicable for reproducibility).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to a previous JSON report. Exit code 2 on any regression.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors (auto-disabled when stdout is not a TTY).",
    )
    args = parser.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    # Auto-disable color when piping to a file/pipe so reports stay clean.
    if args.no_color or not sys.stdout.isatty() or args.output is not None:
        if args.report_format == "text" and args.output is None:
            # Keep color for interactive text-to-stdout; otherwise strip.
            if not sys.stdout.isatty() or args.no_color:
                disable_color()
        else:
            disable_color()

    client = _build_client(args.model)
    runner = EvalRunner(client)
    cases = EvalRunner.load_directory(args.suite)
    cases = _apply_filters(cases, args.filter)

    if not cases:
        print(f"No cases matched (suite={args.suite}, filter={args.filter!r}).", file=sys.stderr)
        return 0

    run_id = uuid.uuid4().hex[:12]
    started_at = datetime.now(timezone.utc).isoformat()

    # Brief preamble — goes to stderr so it doesn't pollute a redirected report.
    print(
        f"Loaded {len(cases)} cases from {args.suite} "
        f"(model={args.model}, parallel={args.parallel}, run={run_id})",
        file=sys.stderr,
    )

    t0 = time.perf_counter()
    if args.parallel and args.parallel > 1:
        results = runner.run_parallel(cases, max_workers=args.parallel)
    else:
        results = runner.run_all(cases)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    finished_at = datetime.now(timezone.utc).isoformat()

    meta = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "model": args.model,
        "suite_dir": str(args.suite),
        "wall_ms": wall_ms,
        "title": f"clinical-llm-evals — {args.model}",
    }

    reporter = get_reporter(args.report_format)
    rendered = reporter.render(results, meta)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.report_format} report → {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered + ("\n" if not rendered.endswith("\n") else ""))

    hard_failures = _summarize_text(results, args.fail_on)

    # Baseline diff — stderr again, since it's diagnostic.
    has_regression = False
    if args.baseline is not None:
        has_regression, baseline_msg = _check_baseline(args.baseline, meta, results)
        print(baseline_msg, file=sys.stderr)

    if has_regression:
        return 2
    return 1 if hard_failures else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


# Convenience for ``[project.scripts] clinical-llm-evals = "...run:main"``
# — pyproject's entry point expects ``main`` to take no args and return int.
# argparse already handles ``sys.argv[1:]`` when we pass ``None`` above.
def _entrypoint() -> int:  # pragma: no cover
    return main(None)
