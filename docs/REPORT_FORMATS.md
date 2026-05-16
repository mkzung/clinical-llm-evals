# Report formats

`clinical-llm-evals` emits four report formats. Pick one with
`--report-format {text,json,markdown,html}` and (optionally)
`--output PATH`.

## text (default)

Pretty terminal output. Pass/fail are color-coded when stdout is a TTY;
colors auto-strip when piping to a file or pipe.

```
[PASS] triage/chest_pain_heart_score (critical)     0.4 ms
    contains all of ['HEART', '12-lead ECG', '10 minutes']
[FAIL] triage/stroke_window (critical)              0.5 ms
    missing required substrings: ['4.5 hour']

-- Summary --------------------------------------------------
  drug_interactions                  5/5
  fhir_parsing                       5/5
  immunization_recommendations       5/5
  triage                             4/5
  vaccine_scheduling                 5/5
FAILED  24/25 passed (critical=1, warning=0, info=0) -- wall 0.43s, sum 12.04s
```

## json

Stable, machine-readable schema designed for the `--baseline` diff and
external dashboards. Top-level keys are guaranteed; per-case fields
include the prompt, the (truncated) response, and the citation.

```json
{
  "run_id": "ab12cd34ef56",
  "started_at": "2026-05-16T12:00:00+00:00",
  "finished_at": "2026-05-16T12:00:12+00:00",
  "model": "anthropic:claude-3-5-sonnet-latest",
  "suite_dir": "evals",
  "wall_ms": 12043.5,
  "n_cases": 25,
  "n_pass": 24,
  "n_fail": 1,
  "results": [
    {
      "case_id": "chest_pain_heart_score",
      "category": "triage",
      "severity": "critical",
      "passed": true,
      "reason": "contains all of ['HEART', '12-lead ECG', '10 minutes']",
      "latency_ms": 411.2,
      "prompt": "55-year-old presents with substernal chest pain...",
      "response": "Apply the HEART score (History, ECG, Age, ...).",
      "source": "AHA Get With The Guidelines + Six et al., 2008 HEART derivation"
    }
  ]
}
```

### Schema contract

- `run_id, started_at, finished_at, model, suite_dir`: free-form
  metadata for traceability.
- `n_cases, n_pass, n_fail`: counts; `n_pass + n_fail == n_cases`.
- `results[i].passed`: `true` / `false`.
- `results[i].latency_ms`: per-case wall time in milliseconds.
- `results[i].response`: truncated to **2000 characters**.

Anything not listed here is best-effort and may change without bumping
the major version.

## markdown

GitHub-flavored markdown with a summary table, per-category rollup, and
each case wrapped in `<details>` for collapsibility. Designed to be
pasted into a PR review or a Slack snippet.

````markdown
# clinical-llm-evals -- anthropic:claude-3-5-sonnet-latest

- **model:** `anthropic:claude-3-5-sonnet-latest`
- **run id:** `ab12cd34ef56`
- **wall clock:** 12.04s

## Summary

| metric         | value |
|----------------|-------|
| total cases    | 25    |
| passed         | 24    |
| failed         | 1     |
| critical fails | 1     |

## By category

| category | passed | total | pass-rate |
|---|---|---|---|
| `drug_interactions` | 5 | 5 | 100% |
| `triage` | 4 | 5 | 80% |

## Cases

<details>
<summary>**[FAIL]** `triage/stroke_window` (`critical`) -- 410 ms</summary>

**reason:** missing required substrings: ['4.5 hour']

**source:** AHA/ASA 2019 acute ischemic stroke guidelines

**response:**
```
The patient presents with hemiparesis at 3 hours from last-known-well...
```

</details>
````

## html

Single-file HTML report with embedded CSS/JS. No CDN. Works opened
directly from disk (`file://...`) and degrades to a static document if
JavaScript is disabled. Features:

- Summary cards (total, passed, failed, critical fails, wall clock)
- Sortable per-category rollup
- Per-case `<details>` panels with prompt, response, source, reason
- Severity-tinted badges
- Filter dropdown (all / failures / passes / per severity tier)
- "Expand all" / "Collapse all" buttons
- Light/dark theme via `prefers-color-scheme`

Open the file in any modern browser. It is safe to attach to a PR or
email -- every user-supplied string is HTML-escaped, so a malicious
prompt cannot inject script.

## Format quick-reference

| Format | Best for | Stdout safe? | Diff-able? |
|---|---|---|---|
| `text` | Interactive runs | Yes (TTY) | No |
| `json` | CI artifacts, baselines | Yes | **Yes** (input to `diff.py`) |
| `markdown` | PR reviews, Slack | Yes | Lossy |
| `html` | Sharing with stakeholders | No (binary) | No |
