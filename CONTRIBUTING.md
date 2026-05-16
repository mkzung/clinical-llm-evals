# Contributing to clinical-llm-evals

Thanks for considering a contribution. This project lives or dies by the **clinical credibility** of its eval cases, so the bar for new YAMLs is deliberately high.

## The one non-negotiable rule

**Every eval case must cite a real, retrievable clinical source.** Examples of acceptable sources:

- CDC ACIP MMWR statements (with year and volume),
- FDA-approved drug labels (link to DailyMed or accessdata.fda.gov),
- IDSA, ACOG, AAP, AHA, ASA, SCCM, ATS, ASCO guidelines,
- HL7 FHIR R4/R5 specification pages,
- Peer-reviewed publications in PubMed/PMC (include DOI),
- WHO Yellow Book / IHR.

**Not acceptable**: "common knowledge", a chatbot answer, a Reddit thread, a blog post, or a Wikipedia paragraph as the *primary* source. If you cannot find an authoritative source for the claim you want to evaluate, the claim probably isn't ready to be an eval.

PRs without verifiable citations will be closed. This is what protects the project from drifting into folk clinical wisdom.

## How to add an eval

1. Pick the right category folder under `evals/`:
   - `vaccine_scheduling/` — timing, contraindications, special populations
   - `drug_interactions/` — major, well-documented interactions
   - `triage/` — time-critical decision points and red flags
   - `fhir_parsing/` — structural correctness against FHIR R4
   - `immunization_recommendations/` — CDC pink-book / ACIP scenarios

2. Copy an existing YAML and edit. Required fields:

   ```yaml
   id: snake_case_unique_name        # must equal the filename (without .yaml)
   category: vaccine_scheduling       # must equal the folder name
   source: >-
     Full citation including URL/DOI to an authoritative source.
   prompt: >-
     The exact text sent to the model under test.
   expected_behavior: >-
     Plain-English description of what a safe, correct answer looks like.
     This is documentation for human reviewers — it is NOT used for scoring.
   scoring_rubric:
     type: contains_all       # or contains_any / exact_match / json_schema_match / llm_judge
     must_include: [...]
     must_not_include: [...]
   severity: critical         # info | warning | critical
   references:                # optional list of additional URLs/DOIs
     - https://...
   ```

3. Run `pytest tests/test_eval_schema.py` to validate. The schema test will:
   - parse the YAML into an `EvalCase`,
   - confirm `id` matches filename and `category` matches folder,
   - confirm `source` references one of the recognized authoritative bodies.

4. Run `pytest tests/test_mock_runner.py`. If your new case is meant to pass against `MockClient`, add a canned response to `MockClient.DEFAULT_CANNED` in `src/clinical_llm_evals/llm.py` that satisfies your rubric. The dict key is a substring of the prompt; longest matching key wins.

5. Open a PR. Include in the description: the source you cited, why this case matters for clinical safety, and (if applicable) which production LLMs you've seen get this wrong.

## Severity guidance

- `info` — stylistic or edge-case; failure has no clinical impact.
- `warning` — clinically suboptimal advice; failure represents a quality gap.
- `critical` — potential for direct patient harm if the model's answer were acted on. Most cases should be `critical` or `warning`; reserve `info` for things like formatting consistency.

## Style

- One eval per YAML file. Don't bundle multiple scenarios.
- Prompts should resemble realistic clinician/PM/EHR-integration questions, not USMLE vignettes.
- Avoid PHI even in fictional examples — use generic demographics.
- Keep `must_include` lists short and specific. Three to four substrings is usually enough.

## Code contributions

For code (rather than YAML), please:

- Run `ruff check src tests` before pushing.
- Add unit tests for new scoring strategies or LLM adapters.
- Keep the `LLMClient` protocol surface minimal — `complete(prompt: str) -> str`. Adapter complexity belongs inside the adapter.

## Code of conduct

This project follows the Contributor Covenant. See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
