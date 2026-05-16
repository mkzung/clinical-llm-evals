"""Shared pytest fixtures for clinical-llm-evals."""

from __future__ import annotations

from pathlib import Path

import pytest

from clinical_llm_evals import EvalRunner, MockClient

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = REPO_ROOT / "evals"


@pytest.fixture(scope="session")
def evals_dir() -> Path:
    return EVALS_DIR


@pytest.fixture(scope="session")
def all_cases():
    return EvalRunner.load_directory(EVALS_DIR)


@pytest.fixture
def mock_client() -> MockClient:
    return MockClient()
