"""Result reporters.

Each reporter renders a list of :class:`EvalResult` (plus a metadata dict)
into a string. The text reporter is the CLI default; the JSON reporter
produces a machine-readable artifact suitable for the ``--baseline`` diff
mechanism; the markdown and HTML reporters are designed to be shared with
non-engineers.

Reporters are stateless — pass the same inputs, get the same output. This
makes them easy to unit test and easy to compose.
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from typing import Iterable, Protocol

from clinical_llm_evals.core import EvalResult


# Truncate model responses inside reports so a chatty model doesn't produce
# a 5MB HTML file. JSON cap matches the spec; text/markdown caps are tighter
# because a terminal/preview pane is even less forgiving.
RESPONSE_TRUNCATE_JSON = 2000
RESPONSE_TRUNCATE_TEXT = 400
RESPONSE_TRUNCATE_MD = 800


# ----------------------------------------------------------------------------
# ANSI helpers (no external deps — see required behavior in run.py)
# ----------------------------------------------------------------------------


class _Ansi:
    """Tiny, dependency-free ANSI palette.

    Disable by setting ``USE_COLOR = False`` (e.g., when output is not a TTY).
    """

    USE_COLOR = True

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"

    @classmethod
    def wrap(cls, s: str, *codes: str) -> str:
        if not cls.USE_COLOR or not codes:
            return s
        return "".join(codes) + s + cls.RESET


def disable_color() -> None:
    """Globally disable ANSI coloring (useful for ``--no-color`` or non-TTY)."""
    _Ansi.USE_COLOR = False


# ----------------------------------------------------------------------------
# Reporter Protocol + helpers
# ----------------------------------------------------------------------------


class Reporter(Protocol):
    """Anything that turns results+meta into a string."""

    def render(
        self, results: list[EvalResult], meta: dict
    ) -> str:  # pragma: no cover - protocol
        ...


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)] + "…"


def _summary_counts(results: Iterable[EvalResult]) -> dict[str, int]:
    """Return ``{n, n_pass, n_fail, by_severity_fail}`` counts."""
    results = list(results)
    n = len(results)
    n_pass = sum(1 for r in results if r.passed)
    fails_by_sev: dict[str, int] = defaultdict(int)
    for r in results:
        if not r.passed:
            fails_by_sev[r.severity] += 1
    return {
        "n": n,
        "n_pass": n_pass,
        "n_fail": n - n_pass,
        "fails_info": fails_by_sev.get("info", 0),
        "fails_warning": fails_by_sev.get("warning", 0),
        "fails_critical": fails_by_sev.get("critical", 0),
    }


def _rollup_by_category(results: Iterable[EvalResult]) -> list[tuple[str, int, int]]:
    """Per-category ``(name, n_pass, n_total)`` rollup, sorted by category."""
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        agg[r.category][1] += 1
        if r.passed:
            agg[r.category][0] += 1
    return [(cat, p, t) for cat, (p, t) in sorted(agg.items())]


# ----------------------------------------------------------------------------
# TextReporter — pretty terminal output
# ----------------------------------------------------------------------------


class TextReporter:
    """Human-readable terminal output, optionally colored.

    Layout:
        [PASS] category/case_id (severity)  12 ms
            reason text
        [FAIL] category/case_id (severity)  450 ms
            reason text

        ── Summary ────────────────────────────────────────
        category      11/11
        triage         5/ 5
        ...
        24/25 passed (0 critical fails) — wall-clock 1.23 s
    """

    def render(self, results: list[EvalResult], meta: dict) -> str:
        lines: list[str] = []
        for r in results:
            tag = "PASS" if r.passed else "FAIL"
            colored = _Ansi.wrap(
                f"[{tag}]", _Ansi.BOLD, _Ansi.GREEN if r.passed else _Ansi.RED
            )
            sev = _Ansi.wrap(
                r.severity,
                _Ansi.YELLOW if r.severity == "warning" else "",
                _Ansi.RED if r.severity == "critical" else "",
            )
            head = f"{colored} {r.category}/{r.case_id} ({sev})"
            timing = _Ansi.wrap(f"{r.latency_ms:>6.1f} ms", _Ansi.DIM)
            lines.append(f"{head}  {timing}")
            if r.reason:
                lines.append(f"    {_Ansi.wrap(_truncate(r.reason, RESPONSE_TRUNCATE_TEXT), _Ansi.DIM)}")

        # Summary table.
        counts = _summary_counts(results)
        lines.append("")
        lines.append(_Ansi.wrap("── Summary " + "─" * 50, _Ansi.CYAN))
        for cat, p, t in _rollup_by_category(results):
            ok = p == t
            bar = _Ansi.wrap(f"{p:>3}/{t:<3}", _Ansi.GREEN if ok else _Ansi.RED)
            lines.append(f"  {cat:<32} {bar}")

        total_ms = sum(r.latency_ms for r in results)
        wall = meta.get("wall_ms", total_ms)
        tag = "PASSED" if counts["n_fail"] == 0 else "FAILED"
        tag_col = _Ansi.GREEN if counts["n_fail"] == 0 else _Ansi.RED
        footer = (
            f"{counts['n_pass']}/{counts['n']} passed "
            f"(critical={counts['fails_critical']}, "
            f"warning={counts['fails_warning']}, "
            f"info={counts['fails_info']}) — "
            f"wall {wall/1000:.2f}s, sum {total_ms/1000:.2f}s"
        )
        lines.append(f"{_Ansi.wrap(tag, _Ansi.BOLD, tag_col)}  {footer}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# JSONReporter — machine-readable artifact
# ----------------------------------------------------------------------------


class JSONReporter:
    """Stable JSON schema for downstream diffing and dashboards.

    Top-level keys::

        run_id, started_at, finished_at, model, suite_dir,
        n_cases, n_pass, n_fail, results: [...]

    Per-result::

        case_id, category, severity, passed, reason,
        latency_ms, prompt, response (<=2000 chars), source
    """

    def render(self, results: list[EvalResult], meta: dict) -> str:
        counts = _summary_counts(results)
        payload = {
            "run_id": meta.get("run_id", ""),
            "started_at": meta.get("started_at", ""),
            "finished_at": meta.get("finished_at", ""),
            "model": meta.get("model", ""),
            "suite_dir": meta.get("suite_dir", ""),
            "wall_ms": meta.get("wall_ms", 0.0),
            "n_cases": counts["n"],
            "n_pass": counts["n_pass"],
            "n_fail": counts["n_fail"],
            "results": [
                {
                    "case_id": r.case_id,
                    "category": r.category,
                    "severity": r.severity,
                    "passed": r.passed,
                    "reason": r.reason,
                    "latency_ms": round(r.latency_ms, 3),
                    "prompt": r.prompt,
                    "response": _truncate(r.response, RESPONSE_TRUNCATE_JSON),
                    "source": r.source,
                }
                for r in results
            ],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)


# ----------------------------------------------------------------------------
# MarkdownReporter — GitHub-flavored, collapsed details
# ----------------------------------------------------------------------------


class MarkdownReporter:
    """GitHub-flavored markdown with a summary table, per-category rollup,
    and per-case ``<details>`` blocks containing the prompt, response, and
    source. Designed to be pasted into a PR review or issue.
    """

    def render(self, results: list[EvalResult], meta: dict) -> str:
        counts = _summary_counts(results)
        out: list[str] = []

        # ----- Header
        title = meta.get("title", "clinical-llm-evals report")
        model = meta.get("model", "unknown")
        run_id = meta.get("run_id", "")
        started = meta.get("started_at", "")
        out.append(f"# {title}")
        out.append("")
        out.append(f"- **model:** `{model}`")
        if run_id:
            out.append(f"- **run id:** `{run_id}`")
        if started:
            out.append(f"- **started:** `{started}`")
        out.append(f"- **wall clock:** {meta.get('wall_ms', 0.0)/1000:.2f}s")
        out.append("")

        # ----- Summary
        out.append("## Summary")
        out.append("")
        out.append("| metric | value |")
        out.append("|---|---|")
        out.append(f"| total cases | {counts['n']} |")
        out.append(f"| passed | {counts['n_pass']} |")
        out.append(f"| failed | {counts['n_fail']} |")
        out.append(f"| critical fails | {counts['fails_critical']} |")
        out.append(f"| warning fails | {counts['fails_warning']} |")
        out.append(f"| info fails | {counts['fails_info']} |")
        out.append("")

        # ----- Per-category rollup
        out.append("## By category")
        out.append("")
        out.append("| category | passed | total | pass-rate |")
        out.append("|---|---|---|---|")
        for cat, p, t in _rollup_by_category(results):
            pct = (100.0 * p / t) if t else 0.0
            out.append(f"| `{cat}` | {p} | {t} | {pct:.0f}% |")
        out.append("")

        # ----- Per-case details
        out.append("## Cases")
        out.append("")
        for r in results:
            tag = "PASS" if r.passed else "FAIL"
            badge = "[PASS]" if r.passed else "**[FAIL]**"
            sev = f"`{r.severity}`"
            summary = f"{badge} `{r.category}/{r.case_id}` ({sev}) — {r.latency_ms:.0f} ms"
            out.append("<details>")
            out.append(f"<summary>{summary}</summary>")
            out.append("")
            out.append(f"**reason:** {r.reason or '_(no reason)_'}")
            out.append("")
            if r.source:
                out.append(f"**source:** {r.source}")
                out.append("")
            if r.prompt:
                out.append("**prompt:**")
                out.append("")
                out.append("```")
                out.append(_truncate(r.prompt, RESPONSE_TRUNCATE_MD))
                out.append("```")
                out.append("")
            out.append("**response:**")
            out.append("")
            out.append("```")
            out.append(_truncate(r.response, RESPONSE_TRUNCATE_MD))
            out.append("```")
            out.append("")
            out.append("</details>")
            out.append("")
            # The variable ``tag`` is captured above for forward-compat — not
            # currently emitted into the output but kept so future templates
            # (e.g., shields.io badges) can reference it without rewriting
            # the loop. Silence linters that complain.
            _ = tag
        return "\n".join(out)


# ----------------------------------------------------------------------------
# HTMLReporter — single-file, no CDN, embedded CSS/JS
# ----------------------------------------------------------------------------


_HTML_CSS = """
:root {
  --bg: #0f172a;
  --panel: #1e293b;
  --panel2: #111827;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --pass: #16a34a;
  --fail: #dc2626;
  --warn: #f59e0b;
  --crit: #ef4444;
  --info: #64748b;
  --link: #38bdf8;
  --border: #334155;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f8fafc;
    --panel: #ffffff;
    --panel2: #f1f5f9;
    --text: #0f172a;
    --muted: #475569;
    --link: #0369a1;
    --border: #cbd5e1;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.5;
}
.wrap { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
h1 { margin: 0 0 4px; font-size: 24px; }
.meta { color: var(--muted); margin-bottom: 24px; }
.meta code { background: var(--panel2); padding: 2px 6px; border-radius: 4px; }
.summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.card .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
.card .value { font-size: 28px; font-weight: 600; margin-top: 4px; }
.card.pass .value { color: var(--pass); }
.card.fail .value { color: var(--fail); }
.card.crit .value { color: var(--crit); }
table.rollup { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
table.rollup th, table.rollup td { padding: 10px 14px; border-bottom: 1px solid var(--border); text-align: left; }
table.rollup th { background: var(--panel2); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); cursor: pointer; user-select: none; }
table.rollup tr:last-child td { border-bottom: 0; }
.bar { display: inline-block; height: 6px; background: var(--panel2); border-radius: 3px; overflow: hidden; width: 120px; vertical-align: middle; margin-left: 8px; }
.bar > i { display: block; height: 100%; background: var(--pass); }
.cases { margin-top: 24px; }
.case { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 10px; overflow: hidden; }
.case summary { padding: 12px 16px; cursor: pointer; display: flex; gap: 12px; align-items: center; }
.case summary::-webkit-details-marker { display: none; }
.case[open] summary { border-bottom: 1px solid var(--border); }
.badge { font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.05em; }
.badge.pass { background: var(--pass); color: #fff; }
.badge.fail { background: var(--fail); color: #fff; }
.sev { font-size: 11px; padding: 2px 8px; border-radius: 4px; border: 1px solid var(--border); color: var(--muted); }
.sev.warning { color: var(--warn); border-color: var(--warn); }
.sev.critical { color: var(--crit); border-color: var(--crit); }
.case-id { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
.latency { color: var(--muted); margin-left: auto; font-variant-numeric: tabular-nums; }
.case-body { padding: 12px 16px 16px; }
.case-body .row { margin-bottom: 10px; }
.case-body .row .lbl { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.case-body pre { background: var(--panel2); padding: 12px; border-radius: 6px; overflow-x: auto; margin: 0; font-size: 13px; white-space: pre-wrap; word-break: break-word; }
.case-body a { color: var(--link); }
.controls { margin: 16px 0 8px; display: flex; gap: 8px; flex-wrap: wrap; }
.controls button, .controls select { background: var(--panel); color: var(--text); border: 1px solid var(--border); padding: 6px 10px; border-radius: 6px; font-size: 13px; cursor: pointer; }
.controls button:hover { background: var(--panel2); }
.footer { color: var(--muted); margin-top: 32px; font-size: 12px; text-align: center; }
""".strip()


_HTML_JS = """
(function () {
  // Sort the rollup table by clicking column headers.
  var table = document.querySelector('table.rollup');
  if (table) {
    var headers = table.querySelectorAll('th');
    headers.forEach(function (th, idx) {
      th.addEventListener('click', function () {
        var tbody = table.querySelector('tbody');
        var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
        var asc = th.dataset.sort !== 'asc';
        rows.sort(function (a, b) {
          var av = a.cells[idx].dataset.v || a.cells[idx].textContent.trim();
          var bv = b.cells[idx].dataset.v || b.cells[idx].textContent.trim();
          var na = parseFloat(av), nb = parseFloat(bv);
          if (!isNaN(na) && !isNaN(nb)) { return asc ? na - nb : nb - na; }
          return asc ? av.localeCompare(bv) : bv.localeCompare(av);
        });
        rows.forEach(function (r) { tbody.appendChild(r); });
        headers.forEach(function (h) { delete h.dataset.sort; });
        th.dataset.sort = asc ? 'asc' : 'desc';
      });
    });
  }

  // Filter / expand controls.
  var sel = document.getElementById('filter');
  if (sel) {
    sel.addEventListener('change', function () {
      var v = sel.value;
      document.querySelectorAll('.case').forEach(function (c) {
        if (v === 'all') { c.style.display = ''; return; }
        if (v === 'fail') { c.style.display = c.dataset.pass === '0' ? '' : 'none'; return; }
        if (v === 'pass') { c.style.display = c.dataset.pass === '1' ? '' : 'none'; return; }
        c.style.display = c.dataset.severity === v ? '' : 'none';
      });
    });
  }
  var expand = document.getElementById('expand-all');
  if (expand) {
    expand.addEventListener('click', function () {
      document.querySelectorAll('.case').forEach(function (c) { c.open = true; });
    });
  }
  var collapse = document.getElementById('collapse-all');
  if (collapse) {
    collapse.addEventListener('click', function () {
      document.querySelectorAll('.case').forEach(function (c) { c.open = false; });
    });
  }
})();
""".strip()


class HTMLReporter:
    """Single-file HTML report. No CDNs, no external assets.

    The file is self-contained so it can be emailed, attached to a PR, or
    dropped on a static-site host. CSS variables drive a built-in dark/light
    theme that follows ``prefers-color-scheme``.
    """

    def render(self, results: list[EvalResult], meta: dict) -> str:
        counts = _summary_counts(results)
        title = html.escape(meta.get("title", "clinical-llm-evals report"))
        model = html.escape(meta.get("model", "unknown"))
        run_id = html.escape(meta.get("run_id", ""))
        started = html.escape(meta.get("started_at", ""))
        wall_s = meta.get("wall_ms", 0.0) / 1000.0

        # Summary cards.
        cards = [
            ("Total cases", str(counts["n"]), ""),
            ("Passed", str(counts["n_pass"]), "pass"),
            ("Failed", str(counts["n_fail"]), "fail" if counts["n_fail"] else ""),
            ("Critical fails", str(counts["fails_critical"]), "crit" if counts["fails_critical"] else ""),
            ("Wall clock", f"{wall_s:.2f}s", ""),
        ]
        cards_html = "\n".join(
            f'<div class="card {cls}"><div class="label">{html.escape(lbl)}</div>'
            f'<div class="value">{html.escape(val)}</div></div>'
            for lbl, val, cls in cards
        )

        # Rollup table.
        rollup_rows = []
        for cat, p, t in _rollup_by_category(results):
            pct = (100.0 * p / t) if t else 0.0
            bar_width = round(pct, 1)
            rollup_rows.append(
                f'<tr>'
                f'<td data-v="{html.escape(cat)}"><code>{html.escape(cat)}</code></td>'
                f'<td data-v="{p}">{p}</td>'
                f'<td data-v="{t}">{t}</td>'
                f'<td data-v="{pct:.4f}">{pct:.0f}% '
                f'<span class="bar"><i style="width:{bar_width}%"></i></span></td>'
                f'</tr>'
            )

        # Per-case details.
        cases_html = []
        for r in results:
            pass_cls = "pass" if r.passed else "fail"
            badge = "PASS" if r.passed else "FAIL"
            cases_html.append(
                f'<details class="case" data-pass="{1 if r.passed else 0}" '
                f'data-severity="{html.escape(r.severity)}">'
                f'<summary>'
                f'<span class="badge {pass_cls}">{badge}</span>'
                f'<span class="sev {html.escape(r.severity)}">{html.escape(r.severity)}</span>'
                f'<span class="case-id">{html.escape(r.category)}/{html.escape(r.case_id)}</span>'
                f'<span class="latency">{r.latency_ms:.0f} ms</span>'
                f'</summary>'
                f'<div class="case-body">'
                f'<div class="row"><div class="lbl">reason</div>'
                f'<pre>{html.escape(r.reason or "(no reason)")}</pre></div>'
                + (
                    f'<div class="row"><div class="lbl">source</div>'
                    f'<pre>{html.escape(r.source)}</pre></div>' if r.source else ""
                )
                + (
                    f'<div class="row"><div class="lbl">prompt</div>'
                    f'<pre>{html.escape(_truncate(r.prompt, RESPONSE_TRUNCATE_MD))}</pre></div>'
                    if r.prompt else ""
                )
                + f'<div class="row"><div class="lbl">response</div>'
                f'<pre>{html.escape(_truncate(r.response, RESPONSE_TRUNCATE_JSON))}</pre></div>'
                + "</div></details>"
            )

        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>{title}</title>"
            f"<style>{_HTML_CSS}</style>"
            "</head><body><div class=\"wrap\">"
            f"<h1>{title}</h1>"
            "<div class=\"meta\">"
            f"model <code>{model}</code>"
            + (f" &middot; run <code>{run_id}</code>" if run_id else "")
            + (f" &middot; started <code>{started}</code>" if started else "")
            + "</div>"
            f"<div class=\"summary-cards\">{cards_html}</div>"
            "<table class=\"rollup\"><thead><tr>"
            "<th>category</th><th>passed</th><th>total</th><th>pass-rate</th>"
            "</tr></thead><tbody>"
            + "".join(rollup_rows)
            + "</tbody></table>"
            "<div class=\"controls\">"
            "<select id=\"filter\">"
            "<option value=\"all\">all cases</option>"
            "<option value=\"fail\">failures only</option>"
            "<option value=\"pass\">passes only</option>"
            "<option value=\"critical\">severity: critical</option>"
            "<option value=\"warning\">severity: warning</option>"
            "<option value=\"info\">severity: info</option>"
            "</select>"
            "<button id=\"expand-all\" type=\"button\">expand all</button>"
            "<button id=\"collapse-all\" type=\"button\">collapse all</button>"
            "</div>"
            "<div class=\"cases\">"
            + "".join(cases_html)
            + "</div>"
            "<div class=\"footer\">Generated by clinical-llm-evals.</div>"
            "</div>"
            f"<script>{_HTML_JS}</script>"
            "</body></html>"
        )


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


_REPORTERS: dict[str, type] = {
    "text": TextReporter,
    "json": JSONReporter,
    "markdown": MarkdownReporter,
    "html": HTMLReporter,
}


def get_reporter(format_name: str) -> Reporter:
    """Return a reporter instance by name. Raises ``KeyError`` on miss."""
    if format_name not in _REPORTERS:
        raise KeyError(
            f"unknown report format {format_name!r}; choose from {sorted(_REPORTERS)}"
        )
    return _REPORTERS[format_name]()


__all__ = [
    "HTMLReporter",
    "JSONReporter",
    "MarkdownReporter",
    "Reporter",
    "TextReporter",
    "disable_color",
    "get_reporter",
]
