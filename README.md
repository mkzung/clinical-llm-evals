# clinical-llm-evals

[![CI](https://github.com/mkzung/clinical-llm-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/mkzung/clinical-llm-evals/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **A guideline-cited, workflow-focused evaluation suite for clinical LLMs.**
> Every test case maps to a real published source (CDC ACIP, FDA label, IDSA, ACOG, AAP, AHA, SCCM). No synthesized clinical content.

---

## The problem

Clinical LLMs are being deployed into vaccination planning, triage, prescribing support, and FHIR-driven workflows — but the public evaluation story is still mostly **MedQA-style knowledge recall**. Passing USMLE-derived multiple-choice does not tell you whether a model will:

- Defer a live vaccine after IVIG (8-month wait per ACIP),
- Catch a methotrexate + TMP-SMX interaction before it causes pancytopenia,
- Recognize an unruptured ectopic before the patient decompensates,
- Refuse to invent a `Patient.identifier` when one is missing from a FHIR Bundle.

`clinical-llm-evals` is a **workflow-safety** harness for the integrations clinicians and digital-health PMs actually ship: vaccine scheduling, drug interactions, triage stratification, FHIR parsing, and immunization guidance.

## Demo

Run the suite, save a shareable markdown report:

```bash
python -m clinical_llm_evals.run --model mock \
    --report-format markdown --output report.md
```

The markdown report looks like this (truncated):

```
# clinical-llm-evals — mock

- model: mock
- wall clock: 0.43s

## Summary

| metric          | value |
|-----------------|-------|
| total cases     | 25    |
| passed          | 24    |
| failed          | 1     |
| critical fails  | 0     |
| warning fails   | 1     |

## By category

| category                       | passed | total | pass-rate |
|--------------------------------|--------|-------|-----------|
| drug_interactions              | 5      | 5     | 100%      |
| fhir_parsing                   | 5      | 5     | 100%      |
| immunization_recommendations   | 5      | 5     | 100%      |
| triage                         | 4      | 5     | 80%       |
| vaccine_scheduling             | 5      | 5     | 100%      |

## Cases

<details>
<summary>[PASS] drug_interactions/warfarin_nsaid (critical) — 320 ms</summary>
   reason:   contains all of ['avoid', 'NSAID', 'bleeding']
   source:   FDA boxed warning, Coumadin (warfarin) prescribing information
   prompt:   65-year-old on warfarin asks if ibuprofen is OK for back pain...
   response: Avoid concurrent ibuprofen or other non-selective NSAIDs in patients on warfarin...
</details>

<details>
<summary>**[FAIL]** triage/stroke_window (critical) — 410 ms</summary>
   reason:   missing required substrings: ['4.5 hour']
   source:   AHA/ASA 2019 guidelines for acute ischemic stroke
</details>
```

See [`docs/REPORT_FORMATS.md`](docs/REPORT_FORMATS.md) for full samples of each format.

## Quickstart

```bash
git clone https://github.com/mkzung/clinical-llm-evals.git
cd clinical-llm-evals
pip install -e ".[dev]"
pytest tests/             # validates schema + runs MockClient through full suite

# Run the suite and save a markdown report:
python -m clinical_llm_evals.run --model mock \
    --report-format markdown --output report.md
```

To run against a real model, with parallelism and a regression-gated baseline:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m clinical_llm_evals.run \
    --model anthropic:claude-3-5-sonnet-latest \
    --suite evals/ \
    --parallel 8 \
    --report-format html --output report.html \
    --baseline last_run.json    # exit 2 on any regression
```

### CLI flags

| Flag | Purpose |
|---|---|
| `--suite PATH` | Directory of YAML eval cases (default: `evals/`) |
| `--model SPEC` | `mock`, `anthropic:<model>`, or `openai:<model>` |
| `--report-format` | `text`, `json`, `markdown`, `html` |
| `--output PATH` | Write report to file (default: stdout) |
| `--parallel N` | Run N cases concurrently via thread pool |
| `--filter key=v1,v2` | Subset by `category`, `severity`, or `id` |
| `--seed N` | Seed Python's RNG for reproducibility |
| `--baseline PATH` | Compare to a previous JSON report; exit 2 on regression |
| `--fail-on` | `any` / `warning` / `critical` — severity threshold for exit 1 |
| `--no-color` | Strip ANSI codes (auto-stripped on non-TTY output) |

Snapshot diff between two JSON reports:

```bash
python -m clinical_llm_evals.diff \
    --baseline a.json --current b.json \
    --output diff.md --fail-on-regression
```

## Eval categories

| Category | What it tests | Example case | Source |
|---|---|---|---|
| `vaccine_scheduling` | Timing & contraindications for immunocompromised patients | Live vaccine deferral 24mo post-HSCT | [ACIP HSCT recs](https://www.cdc.gov/vaccines/hcp/imz-schedules/adult-notes.html) |
| `drug_interactions` | Major, well-documented interactions with FDA-labeled risk | Methotrexate + TMP-SMX → pancytopenia | [FDA / case-report literature](https://pmc.ncbi.nlm.nih.gov/articles/PMC3994806/) |
| `triage` | Time-critical decision points and red flags | tPA 4.5h window from last-known-well | [AHA/ASA 2019 guidelines](https://www.ahajournals.org/doi/10.1161/STR.0000000000000211) |
| `fhir_parsing` | Structural correctness against HL7 FHIR R4 | Refuse to fabricate missing `Patient.identifier` | [HL7 FHIR R4](https://hl7.org/fhir/R4/) |
| `immunization_recommendations` | CDC pink-book / ACIP scenarios | Tdap at 27–36wk gestation | [CDC ACIP Tdap pregnancy](https://www.cdc.gov/pertussis/vaccines/tdap-vaccination-during-pregnancy.html) |

See [`docs/EVAL_TAXONOMY.md`](docs/EVAL_TAXONOMY.md) for the rationale behind each category and what's explicitly out of scope.

## Adding a new eval

1. Pick the right category directory under `evals/`.
2. Copy an existing YAML and edit. Required fields: `id`, `category`, `source`, `prompt`, `expected_behavior`, `scoring_rubric`, `severity`.
3. **Cite a real source.** A specific guideline section, FDA label paragraph, or peer-reviewed paper. PRs without verifiable citations will be rejected — see [`CONTRIBUTING.md`](CONTRIBUTING.md).
4. `pytest tests/test_eval_schema.py` must pass.
5. Open a PR.

## Architecture

```
src/clinical_llm_evals/
  core.py        # EvalCase / EvalResult / EvalRunner / AsyncEvalRunner
  scoring.py     # exact_match, contains_all, json_schema_match, llm_judge
  registry.py    # @register_scorer decorator + get_scorer lookup
  report.py      # Text / JSON / Markdown / HTML reporters
  diff.py        # Snapshot diff between two JSON reports
  run.py         # CLI: filters, parallelism, baseline diff, report formats
  llm.py         # LLMClient Protocol + Anthropic / OpenAI / Mock adapters
evals/           # YAML test cases, one folder per category
tests/           # schema validation + MockClient end-to-end + scorer/reporter tests
docs/            # ARCHITECTURE.md, REPORT_FORMATS.md, EVAL_TAXONOMY.md
```

`EvalRunner` is intentionally small: load → render prompt → call client → score → emit result. The scoring layer routes through a registry, so contributors can plug in custom scorers without touching the runner. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the dependency diagram.

## Compared to other eval frameworks

`clinical-llm-evals` deliberately stays narrow. The table below is for orientation, not a claim that the others are inferior — they're excellent general-purpose tools and this project is happy to live alongside them.

| | clinical-llm-evals | [inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) | [promptfoo](https://github.com/promptfoo/promptfoo) | [openai/evals](https://github.com/openai/evals) |
|---|---|---|---|---|
| Primary focus | Clinical workflow safety | General LLM evals + agents | Prompt + RAG regression | Capability evals |
| YAML configs | One file per case, mandatory citation | Python tasks (`@task`) | YAML + JS/TS configs | YAML registry |
| Source-cite invariant | **Required** (CDC/FDA/IDSA/ACOG/AAP/AHA) | No | No | No |
| LLM-as-judge | Structured JSON, robust parser | Yes | Yes | Yes |
| Pluggable scorer registry | `@register_scorer` decorator | Solver/Scorer ABCs | Custom assertions | `Eval` subclasses |
| Report formats | text / JSON / markdown / **self-contained HTML** | Log viewer + web UI | Web viewer | JSON logs |
| Parallel execution | Threads (`--parallel`) + asyncio | asyncio | Yes | Yes |
| Snapshot diff CLI | `python -m clinical_llm_evals.diff` | Via log viewer | Yes | No |
| Dependencies | `pydantic`, `pyyaml`, `jsonschema` | Heavier (httpx, rich, etc.) | Node.js | Python heavy |

**What's different here:** every case must cite a real, retrievable clinical source (CDC ACIP, FDA label, etc.), the suite ships with hand-curated MockClient responses so CI is deterministic, and the dependency surface is small enough that this fits into a clinical-systems integration test without adding a dozen transitive packages.

## Citation

If you use this in research or a product evaluation, please cite:

```bibtex
@software{gorbuk_clinical_llm_evals_2026,
  author  = {Gorbuk, Max},
  title   = {clinical-llm-evals: guideline-cited evaluations for clinical LLMs},
  year    = {2026},
  url     = {https://github.com/mkzung/clinical-llm-evals}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided so GitHub's
"Cite this repository" widget picks up the same metadata.

## Contributing

PRs welcome, especially from clinicians, pharmacists, and ML engineers shipping healthcare LLMs. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). The non-negotiable rule: **every eval cites a real, retrievable clinical source.**

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

## Disclaimer

This project is a **model evaluation harness**. It is not a medical device, not clinical decision support, and not a substitute for clinician judgment. The YAML test cases describe expected model behavior; they are not patient-care protocols.
