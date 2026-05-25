# Changelog

All notable changes to `clinical-llm-evals` are tracked here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CITATION.cff` so GitHub's "Cite this repository" widget surfaces the same
  metadata as the README BibTeX block.
- Thread lock around `MockClient.calls` so `EvalRunner.run_parallel(...)` no
  longer races on the call log.

### Changed
- `EvalCase` and `ScoringRubric` now use Pydantic `extra="forbid"`. A typo in a
  YAML key (`must_includ:` instead of `must_include:`) raises at parse time
  instead of being silently dropped.
- README BibTeX author normalized to `Gorbuk, Max` to match `pyproject.toml`,
  `LICENSE`, and `CITATION.cff`.
- Removed an unsupported sentence from `docs/EVAL_TAXONOMY.md` ("MedAI (the
  maintainer's startup)") — the maintainer does not run a startup; the
  category-rationale paragraph now stands on the actual clinical reasoning.

### Deferred (not addressed in this release)
- `OpenAIClient.complete` still passes `max_tokens`; newer OpenAI Chat
  Completions models accept `max_completion_tokens`. Will switch when the
  Anthropic-style `max_tokens` is removed from the API.
- No published GitHub Release / tag yet; pyproject's `Changelog` URL points to
  the Releases page, which is empty until v0.1.0 is tagged.
- HTML report has no XSS test coverage beyond the implicit `html.escape` on
  every interpolation site. A regression test that feeds adversarial strings
  through every reporter would be cheap to add.

## [0.1.0] — 2026-05-16

Initial public commit. Five evaluation categories, 25 hand-curated YAML cases
each citing a real published source (CDC ACIP, FDA labels, IDSA, ACOG, AAP,
AHA, SCCM, HL7 FHIR R4). Sync and async runners, four report formats (text,
JSON, markdown, single-file HTML), baseline diff CLI, scorer registry, and
mock/Anthropic/OpenAI client adapters. CI runs `pytest` on Python 3.10–3.12
plus four format smoke tests.
