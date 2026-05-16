"""End-to-end: run MockClient through the full suite. Every case must pass.

This is the gatekeeper test that demonstrates the rubric machinery actually
works against canned responses, so contributors adding a new YAML have a
green-light path.
"""

from __future__ import annotations

import pytest

from clinical_llm_evals import EvalRunner, MockClient


def test_mock_runner_passes_all_cases(all_cases, mock_client: MockClient) -> None:
    runner = EvalRunner(mock_client)
    results = runner.run_all(all_cases)
    failures = [r for r in results if not r.passed]
    if failures:
        msg = "\n".join(
            f"  - {r.category}/{r.case_id} ({r.severity}): {r.reason}" for r in failures
        )
        pytest.fail(
            f"{len(failures)}/{len(results)} cases failed against MockClient:\n{msg}\n"
            "Either the canned response in MockClient.DEFAULT_CANNED needs an update, "
            "or the rubric in the YAML is too strict."
        )


def test_runner_returns_one_result_per_case(all_cases, mock_client: MockClient) -> None:
    runner = EvalRunner(mock_client)
    results = runner.run_all(all_cases)
    assert len(results) == len(all_cases)
    ids = {r.case_id for r in results}
    assert len(ids) == len(results), "duplicate case ids found"
