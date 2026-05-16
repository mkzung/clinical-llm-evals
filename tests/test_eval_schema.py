"""Schema-level validation: every YAML in evals/ must parse and cite a real source."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from clinical_llm_evals import EvalRunner
from clinical_llm_evals.core import SCORING_TYPES

# Reject obviously fake citations during PR review.
_REAL_SOURCE_HINTS = re.compile(
    r"(cdc\.gov|fda\.gov|acog\.org|aap\.org|hivguidelines\.org|ahajournals\.org|"
    r"sccm\.org|hl7\.org|idsociety\.org|academic\.oup\.com|nih\.gov|"
    r"mmwr|mmwr morb|MMWR|ACIP|ACOG|AAP|IDSA|FDA|WHO|"
    r"pmc\.ncbi\.nlm\.nih\.gov|pubmed\.ncbi\.nlm\.nih\.gov|jamanetwork\.com|"
    r"doi\.org|wileyonlinelibrary|publications\.aap\.org|clinicalinfo\.hiv\.gov|"
    r"nccc-online\.org)",
    re.IGNORECASE,
)


def _collect_yaml(root: Path) -> list[Path]:
    return sorted(p for ext in ("*.yaml", "*.yml") for p in root.rglob(ext))


def test_evals_directory_exists(evals_dir: Path) -> None:
    assert evals_dir.exists(), f"missing {evals_dir}"
    paths = _collect_yaml(evals_dir)
    assert len(paths) >= 25, f"expected >=25 eval YAMLs, found {len(paths)}"


@pytest.mark.parametrize("yaml_path", _collect_yaml(Path(__file__).resolve().parent.parent / "evals"))
def test_eval_case_parses(yaml_path: Path) -> None:
    case = EvalRunner.load_case(yaml_path)
    # id matches filename (without extension) to keep navigation predictable.
    assert case.id == yaml_path.stem, f"{yaml_path}: id {case.id!r} != filename {yaml_path.stem!r}"
    # category matches parent directory.
    assert case.category == yaml_path.parent.name, (
        f"{yaml_path}: category {case.category!r} != folder {yaml_path.parent.name!r}"
    )
    assert case.scoring_rubric.type in SCORING_TYPES
    assert case.severity in {"info", "warning", "critical"}
    assert case.prompt.strip(), "prompt must be non-empty"
    assert case.expected_behavior.strip(), "expected_behavior must be non-empty"


@pytest.mark.parametrize("yaml_path", _collect_yaml(Path(__file__).resolve().parent.parent / "evals"))
def test_eval_case_cites_real_source(yaml_path: Path) -> None:
    """Every eval must point to an authoritative source (URL or named guideline)."""
    case = EvalRunner.load_case(yaml_path)
    assert _REAL_SOURCE_HINTS.search(case.source), (
        f"{yaml_path}: source field does not cite a recognized authoritative source. "
        "See CONTRIBUTING.md — every eval must reference CDC/FDA/IDSA/ACOG/AAP/AHA/etc."
    )


def test_categories_are_balanced(evals_dir: Path) -> None:
    """Sanity check that each declared category has at least 5 cases."""
    expected = {
        "vaccine_scheduling",
        "drug_interactions",
        "triage",
        "fhir_parsing",
        "immunization_recommendations",
    }
    for category in expected:
        cases = _collect_yaml(evals_dir / category)
        assert len(cases) >= 5, f"category {category} has only {len(cases)} cases (need 5)"
