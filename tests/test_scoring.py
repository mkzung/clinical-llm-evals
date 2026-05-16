"""Unit tests for the scoring strategies."""

from __future__ import annotations

import pytest

from clinical_llm_evals import scoring


# ----- exact_match --------------------------------------------------------


def test_exact_match_pass() -> None:
    passed, _ = scoring.exact_match("hello", "hello")
    assert passed


def test_exact_match_strips_whitespace() -> None:
    passed, _ = scoring.exact_match("  hello\n", "hello")
    assert passed


def test_exact_match_fail() -> None:
    passed, reason = scoring.exact_match("hi", "hello")
    assert not passed
    assert "expected" in reason


# ----- contains_all -------------------------------------------------------


def test_contains_all_pass_case_insensitive() -> None:
    passed, _ = scoring.contains_all("Defer MMR for 8 months", ["mmr", "8 months"])
    assert passed


def test_contains_all_missing() -> None:
    passed, reason = scoring.contains_all("Defer MMR", ["mmr", "8 months"])
    assert not passed
    assert "8 months" in reason


def test_contains_all_forbidden() -> None:
    passed, reason = scoring.contains_all(
        "Give MMR today", ["mmr"], must_not_include=["give mmr today"]
    )
    assert not passed
    assert "forbidden" in reason


# ----- contains_any -------------------------------------------------------


def test_contains_any_pass() -> None:
    passed, _ = scoring.contains_any("Avoid NSAIDs", ["nsaid", "ibuprofen"])
    assert passed


def test_contains_any_fail() -> None:
    passed, _ = scoring.contains_any("Take acetaminophen", ["nsaid", "ibuprofen"])
    assert not passed


# ----- json_schema_match --------------------------------------------------


def test_json_schema_match_raw_json() -> None:
    schema = {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}}
    passed, _ = scoring.json_schema_match('{"x": 5}', schema)
    assert passed


def test_json_schema_match_fenced() -> None:
    schema = {"type": "object", "required": ["x"]}
    response = "Here is the result:\n```json\n{\"x\": 1}\n```\n"
    passed, _ = scoring.json_schema_match(response, schema)
    assert passed


def test_json_schema_match_invalid_json() -> None:
    passed, reason = scoring.json_schema_match("not json at all", {"type": "object"})
    assert not passed
    assert "not valid JSON" in reason


def test_json_schema_match_schema_violation() -> None:
    schema = {"type": "object", "required": ["x"]}
    passed, reason = scoring.json_schema_match('{"y": 1}', schema)
    assert not passed
    assert "schema validation failed" in reason


# ----- llm_judge ----------------------------------------------------------


class _StubJudge:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        return self.verdict


def test_llm_judge_pass() -> None:
    judge = _StubJudge("PASS the response satisfies all criteria")
    passed, reason = scoring.llm_judge(judge, "q", "a", "criteria")
    assert passed
    assert "PASS" in reason


def test_llm_judge_fail() -> None:
    judge = _StubJudge("FAIL missing key element")
    passed, reason = scoring.llm_judge(judge, "q", "a", "criteria")
    assert not passed


def test_llm_judge_unparseable() -> None:
    judge = _StubJudge("maybe?")
    passed, reason = scoring.llm_judge(judge, "q", "a", "criteria")
    assert not passed
    assert "unparseable" in reason


def test_llm_judge_exception_safe() -> None:
    class _Boom:
        def complete(self, prompt: str) -> str:
            raise RuntimeError("API down")

    passed, reason = scoring.llm_judge(_Boom(), "q", "a", "criteria")
    assert not passed
    assert "judge raised" in reason


# ----- llm_judge (structured JSON path) -----------------------------------
#
# These cover the upgraded judge: the judge is asked for
# {"pass": bool, "reasoning": str}. Real models often wrap that in code fences
# or prose, so the parser must be tolerant.


def test_llm_judge_json_pass() -> None:
    judge = _StubJudge('{"pass": true, "reasoning": "all criteria satisfied"}')
    passed, reason = scoring.llm_judge(judge, "q", "a", "criteria")
    assert passed
    assert "all criteria" in reason


def test_llm_judge_json_fail() -> None:
    judge = _StubJudge('{"pass": false, "reasoning": "missed dose limit"}')
    passed, reason = scoring.llm_judge(judge, "q", "a", "criteria")
    assert not passed
    assert "missed dose limit" in reason


def test_llm_judge_json_fenced() -> None:
    """A judge that wraps JSON in a ```json fence must still parse."""
    judge = _StubJudge(
        'Sure, here is my verdict:\n```json\n'
        '{"pass": true, "reasoning": "ok"}\n```\nLet me know if you need more.'
    )
    passed, reason = scoring.llm_judge(judge, "q", "a", "criteria")
    assert passed
    assert "ok" in reason


def test_llm_judge_json_with_leading_prose() -> None:
    judge = _StubJudge(
        "After reviewing the answer against the criteria, my verdict is:\n"
        '{"pass": false, "reasoning": "fails on safety check"}\n'
    )
    passed, reason = scoring.llm_judge(judge, "q", "a", "criteria")
    assert not passed
    assert "safety" in reason


def test_llm_judge_json_string_bool() -> None:
    """Some models return strings instead of JSON booleans — accept "true"/"false"."""
    judge = _StubJudge('{"pass": "true", "reasoning": "stringified bool"}')
    passed, _ = scoring.llm_judge(judge, "q", "a", "criteria")
    assert passed


def test_llm_judge_parse_error_class() -> None:
    """LLMJudgeError is the documented parse-failure type."""
    with pytest.raises(scoring.LLMJudgeError):
        scoring._parse_judge_verdict("clearly not a verdict at all just words")


def test_llm_judge_empty_response_is_unparseable() -> None:
    judge = _StubJudge("")
    passed, reason = scoring.llm_judge(judge, "q", "a", "criteria")
    assert not passed
    assert "unparseable" in reason or "empty" in reason
