# Eval Taxonomy

This document describes the five eval categories, the rationale for each, and
what is explicitly out of scope. Detailed clinical content for each case lives
in `evals/<category>/*.yaml`.

## Design philosophy

This suite is a **workflow-safety** harness, not a knowledge benchmark. The
distinction matters:

- A knowledge benchmark (MedQA, USMLE) asks "does the model know the right
  answer in a multiple-choice setting?"
- A workflow-safety harness asks "does the model behave acceptably when
  embedded into a clinical product, including refusing when it should refuse
  and citing constraints when they exist?"

Every case maps to a real published source. Cases where the literature is
ambiguous or contested are excluded — the eval should detect deviation from
established guidance, not adjudicate frontier debates.

Every case has a `severity` field (`info`, `warning`, `critical`). Severity
maps to consequence-if-the-model-gets-it-wrong, not to medical urgency in the
abstract. A clearly-bad recommendation that a downstream system would act on
is `critical`. A subtle phrasing issue that a reviewer would catch is `info`.

## Categories

### `vaccine_scheduling`

**Scope.** Timing, sequencing, and contraindication rules for vaccines in
patients with non-standard immune status: post-transplant, on biologics,
post-IVIG, asplenia, age cutoffs.

**Source family.** CDC ACIP recommendations, the CDC "Pink Book", IDSA
guidelines for immunocompromised hosts.

**Why this category exists.** Vaccine scheduling for non-standard patients is
where commercial clinical LLMs frequently produce dangerous-but-confident
output, and where the ground truth is unusually well-specified — a combination
that makes it an unusually high-value target for a workflow-safety harness.

**Out of scope.** Pediatric primary series for healthy children (well-covered
elsewhere), country-specific schedules outside the US/EU, novel-vaccine first
indications (insufficient guideline stability).

### `drug_interactions`

**Scope.** Major, well-documented interactions with FDA-labeled risk or
established guideline coverage. The case set is intentionally narrow —
"the interactions a competent intern is expected to know cold."

**Source family.** FDA package inserts, Lexicomp-cited primary literature,
guideline statements from the relevant specialty society.

**Why this category exists.** These interactions are exactly the cases where
an LLM that "sounds confident" can ship harm faster than a slow safety net
catches it. The category tests refusal/flagging behavior, not exhaustive
pharmacology.

**Out of scope.** Pharmacogenomic dosing adjustments, rare interactions
without FDA labeling, herb-drug interactions (literature too noisy).

### `triage`

**Scope.** Time-critical decision points where the model must recognize the
time window, the red flag, or the correct disposition. Stroke window, sepsis
hour-1 bundle, ectopic-pregnancy red flags, pediatric fever <90 days,
HEART-score risk stratification for chest pain.

**Source family.** AHA/ASA stroke guidelines, Surviving Sepsis Campaign,
ACOG ectopic guidance, AAP febrile-infant guidance, AHA/ACC chest-pain
guidelines.

**Why this category exists.** Triage is where the gap between "the model knew
the right answer in MCQ format" and "the model acted appropriately given a
realistic narrative" is largest.

**Out of scope.** Differential diagnosis tasks broader than the named
decision point. Mental-health crisis triage (separate eval set warranted).

### `fhir_parsing`

**Scope.** Structural correctness against the HL7 FHIR R4 specification.
Extract, validate, refuse-to-fabricate behavior. Cases cover Patient
identifiers, MedicationRequest dosing, Observation reference ranges,
optional-field handling, and Bundle internal-reference integrity.

**Source family.** HL7 FHIR R4 specification.

**Why this category exists.** FHIR is the integration layer for almost every
healthcare AI shipping to a hospital. Models that hallucinate plausible-looking
FHIR fields ship integration failures that look like clinical errors.

**Out of scope.** FHIR R5 (still in transition for most US deployments),
USCDI/SMART-on-FHIR conformance (deserves a dedicated harness).

### `immunization_recommendations`

**Scope.** Distinct from `vaccine_scheduling` in that this category covers
recommended vaccines for specific population/event combinations — pregnancy +
Tdap, post-splenectomy + meningococcal, HIV + live-attenuated guidance,
travel medicine, post-exposure prophylaxis.

**Source family.** CDC ACIP, CDC Yellow Book (travel), HIV Opportunistic
Infections guidelines, CDC PEP guidance.

**Why this category exists.** Population-event combinations are
under-represented in general medical knowledge bases and where deployed
clinical LLMs disproportionately err.

**Out of scope.** Country-specific schedules outside US/EU (handled by
national health bodies), veterinary immunization (obvious).

## What is explicitly *not* a category here

- **Diagnostic reasoning over open-ended narratives.** This is the domain of
  HealthBench, MultiMedQA, and similar; not what this harness measures.
- **Reading medical imaging.** Vision benchmarks elsewhere.
- **EHR free-text summarization.** MEDIQA and clinical-NLP benchmarks exist.
- **Mental health crisis triage.** Deserves dedicated rubrics with
  qualified-clinician sign-off; treating it as one of five workflow buckets
  would understate its difficulty.

## How to propose a new category

Open an issue with: the proposed category name, the source family it would
draw from, three example cases (one each `info`/`warning`/`critical`), and
an explicit out-of-scope statement. Categories are added when there is a
maintained source family and a reasonable initial case count.
