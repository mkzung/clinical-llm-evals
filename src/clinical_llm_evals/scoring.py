"""Scoring strategies for ``EvalCase`` rubrics.

Each function returns ``(passed: bool, reason: str)``. ``reason`` is a short
human-readable explanation suitable for inclusion in pytest output.

Strategies
----------
- ``exact_match``        — strict equality after strip.
- ``contains_all``       — every substring in ``must_include`` appears (case-insensitive).
- ``contains_any``       — at least one substring appears.
- ``json_schema_match``  — response parses as JSON and validates against the schema.
- ``llm_judge``          — a second LLM grades the response. The judge is asked to
                           return structured JSON ``{"pass": bool, "reasoning": str}``;
                           the parser tolerates code fences and stray prose.
"""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class LLMJudgeError(Exception):
    """Raised when the judge's reply cannot be parsed into a verdict.

    The message includes a snippet of the raw verdict so the failure mode is
    debuggable from a terminal or log file without re-running the judge.
    """


# ----------------------------------------------------------------------------
# String matchers
# ----------------------------------------------------------------------------


def exact_match(response: str, expected: str) -> tuple[bool, str]:
    """Return ``(passed, reason)`` for strict equality after ``strip()``."""
    if response.strip() == expected.strip():
        return True, "exact match"
    return False, f"expected {expected!r}, got {response.strip()[:120]!r}"


def contains_all(
    response: str,
    must_include: list[str],
    *,
    must_not_include: list[str] | None = None,
) -> tuple[bool, str]:
    """Pass if every substring in ``must_include`` appears (case-insensitive).

    If ``must_not_include`` is provided, any hit there fails the case.
    """
    lower = response.lower()
    missing = [s for s in must_include if s.lower() not in lower]
    if missing:
        return False, f"missing required substrings: {missing}"

    if must_not_include:
        forbidden = [s for s in must_not_include if s.lower() in lower]
        if forbidden:
            return False, f"response contained forbidden substrings: {forbidden}"

    return True, f"contains all of {must_include}"


def contains_any(
    response: str,
    must_include: list[str],
    *,
    must_not_include: list[str] | None = None,
) -> tuple[bool, str]:
    """Pass if at least one substring in ``must_include`` appears."""
    lower = response.lower()
    hits = [s for s in must_include if s.lower() in lower]
    if not hits:
        return False, f"none of {must_include} present"

    if must_not_include:
        forbidden = [s for s in must_not_include if s.lower() in lower]
        if forbidden:
            return False, f"response contained forbidden substrings: {forbidden}"

    return True, f"matched {hits[0]!r}"


# ----------------------------------------------------------------------------
# Structured matcher
# ----------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _extract_json(response: str) -> Any:
    """Best-effort JSON extraction.

    Accepts:
    - raw JSON,
    - JSON inside a ```json fenced block,
    - JSON inside a generic ``` fenced block,
    - the first balanced ``{...}`` or ``[...]`` substring.
    """
    # Try raw parse first.
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    m = _JSON_BLOCK_RE.search(response)
    if m:
        return json.loads(m.group(1))

    # Last resort: find the first { or [ and try to parse from there.
    for opener in ("{", "["):
        idx = response.find(opener)
        if idx >= 0:
            try:
                return json.loads(response[idx:])
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("no JSON object found in response", response, 0)


def json_schema_match(response: str, schema: dict[str, Any]) -> tuple[bool, str]:
    """Pass if ``response`` parses as JSON and validates against ``schema``."""
    try:
        payload = _extract_json(response)
    except json.JSONDecodeError as exc:
        return False, f"response was not valid JSON: {exc.msg}"

    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as exc:
        return False, f"schema validation failed: {exc.message}"
    return True, "JSON matches schema"


# ----------------------------------------------------------------------------
# LLM-as-judge
# ----------------------------------------------------------------------------

_JUDGE_TEMPLATE = """You are a strict clinical evaluator grading another model's answer
to a clinical question. Decide whether the answer satisfies ALL pass criteria below.

Return ONLY a single JSON object on its own — no prose before or after, no markdown
fences, no commentary. The object must have exactly two keys:

  "pass":      a JSON boolean (true if the answer meets every criterion, else false)
  "reasoning": a short string explaining the verdict (<= 280 chars)

QUESTION:
{prompt}

MODEL ANSWER:
{response}

PASS CRITERIA (every one must be satisfied):
{criteria}

Respond with the JSON object now."""


# Strip optional ```json / ``` fences and surrounding prose to locate the verdict
# JSON. The judge is *asked* for raw JSON, but real models routinely wrap it.
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
_RAW_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
_LEGACY_VERDICT_RE = re.compile(r"\b(PASS|FAIL)\b", re.IGNORECASE)


def _parse_judge_verdict(verdict: str) -> tuple[bool, str]:
    """Parse a judge reply into ``(passed, reasoning)``.

    Tolerates code fences, leading/trailing prose, and (for backward compat)
    plaintext ``PASS``/``FAIL`` replies. Raises ``LLMJudgeError`` on anything
    we cannot make sense of.
    """
    text = verdict.strip()
    if not text:
        raise LLMJudgeError("judge returned empty string")

    # 1) Try fenced JSON.
    m = _FENCED_JSON_RE.search(text)
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1))

    # 2) Try raw object substring (greedy match — JSON parser sorts it out).
    m2 = _RAW_OBJECT_RE.search(text)
    if m2:
        candidates.append(m2.group(0))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "pass" not in obj:
            continue
        passed_raw = obj["pass"]
        if isinstance(passed_raw, bool):
            passed = passed_raw
        elif isinstance(passed_raw, str):
            lowered = passed_raw.strip().lower()
            if lowered in {"true", "pass", "yes"}:
                passed = True
            elif lowered in {"false", "fail", "no"}:
                passed = False
            else:
                continue
        else:
            continue
        reasoning = str(obj.get("reasoning", "")).strip()
        return passed, reasoning

    # 3) Legacy single-token replies ("PASS - ..." / "FAIL: ...").
    m3 = _LEGACY_VERDICT_RE.search(text)
    if m3:
        head = m3.group(1).upper()
        tail = text[m3.end():].lstrip(" -:—\t\n")
        return head == "PASS", tail[:280]

    raise LLMJudgeError(
        f"could not parse judge verdict (no JSON or PASS/FAIL token): {text[:140]!r}"
    )


def llm_judge(
    judge_client: Any,
    prompt: str,
    response: str,
    criteria: str,
) -> tuple[bool, str]:
    """Use a second LLM to grade the response against ``criteria``.

    The judge is asked to return ``{"pass": bool, "reasoning": str}`` JSON.
    Robust to code fences and stray prose; falls back to a plaintext
    ``PASS``/``FAIL`` heuristic for older judge templates.

    Parameters
    ----------
    judge_client:
        Anything with a ``.complete(prompt: str) -> str`` method.
    prompt:
        The original question that was sent to the model under test.
    response:
        The model-under-test's reply, which the judge is grading.
    criteria:
        Plain-English pass criteria — these are pasted into the judge prompt.

    Returns
    -------
    ``(passed, reason)``. On *any* judge failure (parse error, exception),
    returns ``(False, "...")`` rather than raising — a flaky judge must not
    crash the whole suite.
    """
    judge_prompt = _JUDGE_TEMPLATE.format(prompt=prompt, response=response, criteria=criteria)
    try:
        verdict = judge_client.complete(judge_prompt)
    except Exception as exc:  # noqa: BLE001 — judge failure shouldn't crash the suite
        return False, f"judge raised: {exc}"

    try:
        passed, reasoning = _parse_judge_verdict(verdict)
    except LLMJudgeError as exc:
        return False, f"judge returned unparseable verdict: {exc}"

    tag = "PASS" if passed else "FAIL"
    truncated = (reasoning[:140] + "...") if len(reasoning) > 140 else reasoning
    return passed, f"judge: {tag} — {truncated}" if truncated else f"judge: {tag}"
