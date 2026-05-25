"""LLM client protocol and concrete adapters.

Adapters intentionally have a tiny surface — ``complete(prompt: str) -> str``.
This keeps the eval harness model-agnostic and lets contributors plug in local
runtimes (Ollama, vLLM, llama.cpp) with a 10-line wrapper.
"""

from __future__ import annotations

import os
import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every model adapter must satisfy."""

    def complete(self, prompt: str) -> str:
        """Send a single user-turn prompt and return the model's text reply."""
        ...


# ----------------------------------------------------------------------------
# MockClient -- for unit tests and CI
# ----------------------------------------------------------------------------


class MockClient:
    """Returns canned responses keyed off substrings of the prompt.

    Useful for unit testing rubrics and CI smoke runs that must not hit a real
    API. The default ``canned`` mapping covers every YAML in ``evals/`` with a
    response that satisfies its rubric, so ``pytest evals/`` passes deterministically.
    """

    DEFAULT_CANNED: dict[str, str] = {
        # vaccine_scheduling
        "stem cell transplant": (
            "Live attenuated vaccines such as MMR and varicella should be deferred "
            "for at least 24 months after hematopoietic stem cell transplant, and "
            "only given if the patient is in remission, off immunosuppression, and "
            "without graft-versus-host disease. Inactivated vaccines can typically "
            "restart 6 months post-HSCT."
        ),
        "intravenous immunoglobulin": (
            "After high-dose IVIG (e.g., for Kawasaki disease or ITP), MMR and "
            "varicella vaccination should be deferred for 8 months per ACIP, "
            "because passively transferred antibodies can blunt the response to "
            "live attenuated measles vaccine."
        ),
        "hpv": (
            "HPV catch-up vaccination is routinely recommended through age 26. "
            "For adults aged 27 through 45 years, ACIP recommends HPV vaccination "
            "based on shared clinical decision-making. HPV vaccine is not licensed "
            "above age 45."
        ),
        "rituximab": (
            "Inactivated vaccines should ideally be given at least 2 weeks before "
            "rituximab, or deferred at least 6 months after the last dose due to "
            "B-cell depletion blunting humoral response. Live vaccines are "
            "contraindicated during and for at least 6 months after rituximab."
        ),
        "functional asplenia": (
            "Children with functional or anatomical asplenia should receive the "
            "inactivated influenza vaccine annually. Live attenuated influenza "
            "vaccine (LAIV, the nasal spray) is contraindicated in this population."
        ),
        # drug_interactions
        "warfarin": (
            "Avoid concurrent ibuprofen or other non-selective NSAIDs in patients on "
            "warfarin: the combination markedly increases the risk of upper "
            "gastrointestinal bleeding and is the subject of FDA boxed warnings. "
            "Use acetaminophen for analgesia, or if an NSAID is unavoidable, "
            "co-prescribe a PPI and monitor closely."
        ),
        "phenelzine": (
            "Do not start sertraline (an SSRI) until at least 14 days after stopping "
            "the MAOI phenelzine. Conversely, when switching from fluoxetine to an "
            "MAOI a 5-week washout is required due to fluoxetine's long half-life. "
            "Concurrent use can precipitate fatal serotonin syndrome."
        ),
        "ondansetron 16 mg iv": (
            "Both citalopram and ondansetron prolong the QT interval. Concurrent "
            "use additively increases the risk of torsades de pointes. FDA limits "
            "single IV ondansetron doses to 16 mg and citalopram to 20 mg/day in "
            "patients over 60. Obtain a baseline ECG and electrolytes; consider an "
            "alternative antiemetic such as granisetron."
        ),
        "methotrexate": (
            "Avoid trimethoprim-sulfamethoxazole in patients on methotrexate. "
            "TMP-SMX potentiates methotrexate toxicity through additive antifolate "
            "effect and reduced renal clearance, with multiple fatal reports of "
            "pancytopenia. Choose an alternative antibiotic."
        ),
        "simvastatin": (
            "Avoid grapefruit juice while taking simvastatin. Grapefruit inhibits "
            "intestinal CYP3A4, raising simvastatin AUC roughly 3.6-fold and "
            "increasing the risk of myopathy and rhabdomyolysis. The FDA label "
            "lists this combination as contraindicated at higher simvastatin doses."
        ),
        # triage
        "chest pain": (
            "Apply the HEART score (History, ECG, Age, Risk factors, Troponin). "
            "Low risk (0-3) supports discharge with outpatient follow-up; "
            "intermediate (4-6) warrants observation and serial troponin; "
            "high (7-10) requires admission and early invasive evaluation. "
            "Obtain a 12-lead ECG within 10 minutes of arrival."
        ),
        "hemiparesis": (
            "Last-known-well was 3 hours ago, which falls within the 4.5-hour "
            "window for IV alteplase per AHA/ASA 2019 guidelines, assuming no "
            "exclusion criteria. Activate the stroke team, obtain non-contrast CT "
            "to rule out hemorrhage, and proceed with thrombolysis if eligible. "
            "Also evaluate for large-vessel occlusion and possible thrombectomy."
        ),
        "lactate": (
            "This meets sepsis criteria. Initiate the Surviving Sepsis Campaign "
            "Hour-1 bundle: measure lactate, draw blood cultures before antibiotics, "
            "administer broad-spectrum antibiotics, begin 30 mL/kg IV crystalloid "
            "for hypotension or lactate >=4 mmol/L, and start vasopressors to keep "
            "MAP >=65 mmHg if hypotension persists after fluids."
        ),
        "missed period": (
            "Positive pregnancy test with lower abdominal pain and vaginal bleeding "
            "is a red flag for ectopic pregnancy. Urgent workup: quantitative beta-hCG, "
            "transvaginal ultrasound, type and screen, and complete blood count. Do "
            "not discharge without confirming intrauterine pregnancy or arranging "
            "close follow-up; warn the patient about sudden severe pain, shoulder "
            "pain, or syncope as signs of rupture requiring emergency care."
        ),
        "6-week-old": (
            "Any fever >=38.0 C in an infant under 90 days is a medical emergency. "
            "Per AAP 2021 guidance, infants 8-21 days require a full sepsis workup "
            "including blood culture, urine culture (catheterized), CSF analysis, "
            "empiric parenteral antibiotics, and hospital admission. Do not rely "
            "on a well appearance alone."
        ),
        # fhir_parsing
        "extract all identifiers": (
            '{"identifiers": [{"system": "http://hospital.example.org/mrn", '
            '"value": "12345"}]}'
        ),
        "medicationrequest": (
            '{"medication": "Amoxicillin 500 mg", "dose_value": 500, '
            '"dose_unit": "mg", "frequency": "every 8 hours", "route": "oral"}'
        ),
        "observation": (
            '{"value": 7.2, "unit": "mmol/L", "reference_low": 3.5, '
            '"reference_high": 5.0, "interpretation": "high"}'
        ),
        "no telecom": (
            '{"telecom": null, "note": "Patient.telecom is optional in FHIR R4 '
            'and was not present in the resource; no value was fabricated."}'
        ),
        "internal reference integrity": (
            '{"valid": false, "errors": ["Bundle.entry[1].resource.subject.reference '
            'points to Patient/999 which is not present in the Bundle"]}'
        ),
        # immunization_recommendations
        "28 weeks": (
            "Administer Tdap during this pregnancy. CDC ACIP recommends a single "
            "dose of Tdap during every pregnancy, preferably at 27 through 36 weeks "
            "gestation, to maximize transplacental transfer of pertussis antibodies "
            "to the newborn."
        ),
        "splenectomy": (
            "Post-splenectomy patients require a 2-dose primary series of MenACWY "
            "(Menveo or MenQuadfi) at least 8 weeks apart, with a booster every 5 "
            "years. They also require a MenB primary series with boosters every "
            "2-3 years while risk persists. For elective splenectomy, vaccinate at "
            "least 14 days before the procedure when possible. Also give "
            "pneumococcal (PCV20 or PCV15+PPSV23) and Hib."
        ),
        "cd4 count of 120": (
            "MMR and varicella vaccines are contraindicated in this patient. "
            "ACIP and the HIV Opportunistic Infections guidelines restrict live "
            "attenuated MMR and varicella to people with HIV who have a CD4 count "
            ">=200 cells/mm^3. Defer vaccination until immune reconstitution; "
            "inactivated vaccines (e.g., Tdap, pneumococcal, hepatitis B) can be "
            "given now."
        ),
        "amazon basin": (
            "Travel to the Amazon basin requires yellow fever vaccine (YF-VAX, "
            "live attenuated) at least 10 days before departure to develop "
            "protective antibody. Screen for contraindications: age <9 months, "
            "severe immunocompromise, thymus disorders, anaphylactic egg allergy. "
            "Per WHO 2016 amendment a single dose is valid for life. Also discuss "
            "hepatitis A, typhoid, and malaria chemoprophylaxis."
        ),
        "needlestick": (
            "For an unvaccinated healthcare worker after a hepatitis B-positive "
            "source needlestick: administer hepatitis B immune globulin (HBIG) "
            "0.06 mL/kg AND start the hepatitis B vaccine series within 24 hours, "
            "ideally within 7 days. Baseline anti-HBs, HBsAg, and anti-HCV should "
            "be drawn, with follow-up serology per CDC PEP guidance."
        ),
    }

    def __init__(self, canned: dict[str, str] | None = None, default: str = "I don't know.") -> None:
        self.canned = dict(self.DEFAULT_CANNED)
        if canned:
            self.canned.update(canned)
        self.default = default
        self.calls: list[str] = []
        # ``calls`` is appended to from worker threads when EvalRunner.run_parallel
        # is used; without a lock, list growth in CPython is technically safe
        # but call ordering is not, which makes assertions over ``calls`` flaky
        # in user code. Cheap and explicit is better than relying on GIL trivia.
        self._lock = threading.Lock()

    def complete(self, prompt: str) -> str:
        with self._lock:
            self.calls.append(prompt)
        lower = prompt.lower()
        # Iterate longest-key-first so more specific keys win over generic ones
        # (e.g., "internal reference integrity" beats "observation").
        for key in sorted(self.canned, key=len, reverse=True):
            if key.lower() in lower:
                return self.canned[key]
        return self.default


# ----------------------------------------------------------------------------
# AnthropicClient
# ----------------------------------------------------------------------------


class AnthropicClient:
    """Adapter for the Anthropic Messages API.

    Requires ``pip install clinical_llm_evals[anthropic]`` and an
    ``ANTHROPIC_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: str | None = None,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install 'clinical_llm_evals[anthropic]'"
            ) from exc
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str) -> str:
        kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system:
            kwargs["system"] = self.system
        msg = self._client.messages.create(**kwargs)
        parts = [block.text for block in msg.content if getattr(block, "type", None) == "text"]
        return "".join(parts)


# ----------------------------------------------------------------------------
# OpenAIClient
# ----------------------------------------------------------------------------


class OpenAIClient:
    """Adapter for the OpenAI Chat Completions API.

    Requires ``pip install clinical_llm_evals[openai]`` and an
    ``OPENAI_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install 'clinical_llm_evals[openai]'"
            ) from exc
        import openai

        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(self, prompt: str) -> str:
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""
