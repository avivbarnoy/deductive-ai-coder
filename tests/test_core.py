from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from deductive_ai_coder.models import CodeDefinition, ResearchProject
from deductive_ai_coder.prompts import coding_prompt
from deductive_ai_coder.project_io import project_from_json, project_to_json
from deductive_ai_coder.reliability import pair_metrics


def sample_project() -> ResearchProject:
    return ResearchProject(
        project_name="Demo",
        research_question="How are reasons expressed?",
        unit_of_analysis="One open-ended response",
        codes=[
            CodeDefinition(code_id="A", name="Evidence", definition="Appeals to evidence"),
            CodeDefinition(code_id="B2", name="Experience", definition="Appeals to experience"),
        ],
    )


def test_arbitrary_string_code_ids_and_versioning():
    project = sample_project()
    version = project.freeze_new_version("Initial")
    assert version.version_id == "v1"
    assert [c.code_id for c in version.snapshot.codes] == ["A", "B2"]

    project.codes[0].name = "Evidence/research"
    project.mark_draft_changed()
    assert project.active_version_id == "v1"
    assert project.draft_dirty is True
    assert version.snapshot.codes[0].name == "Evidence"


def test_project_round_trip_does_not_need_raw_data():
    project = sample_project()
    project.freeze_new_version()
    payload = project_to_json(project)
    restored = project_from_json(payload)
    assert restored.project_name == "Demo"
    assert restored.active_version_id == "v1"
    assert restored.versions[0].snapshot.codes[1].code_id == "B2"


def test_prompt_uses_project_codebook_not_fixed_domain_categories():
    project = sample_project()
    prompt = coding_prompt(project.current_snapshot(), "I saw this myself.", {"condition": "x"})
    assert "ID: A" in prompt
    assert "Name: Evidence" in prompt
    assert "ID: B2" in prompt
    assert "Name: Experience" in prompt
    assert "I saw this myself." in prompt


def test_single_code_metrics():
    metrics = pair_metrics(["A", "A", "B"], ["A", "B", "B"], "single")
    assert round(metrics["raw_agreement"], 3) == 0.667
    assert metrics["cohen_kappa"] is not None


def test_multi_code_metrics():
    metrics = pair_metrics(["A,B", "A"], ["A,B", "B"], "multi")
    assert metrics["exact_set_agreement"] == 0.5
    assert metrics["mean_jaccard"] == 0.5


def test_normalize_code_id_accepts_exact_and_safe_prefix_only():
    import pytest
    pytest.importorskip("google.genai")
    from deductive_ai_coder.gemini_client import GeminiService

    valid = {"A", "B", "1.3"}
    assert GeminiService._normalize_code_id("A", valid) == "A"
    assert GeminiService._normalize_code_id("CODE A", valid) == "A"
    assert GeminiService._normalize_code_id("CODE: 1.3", valid) == "1.3"
    assert GeminiService._normalize_code_id("ID B", valid) == "B"

    with pytest.raises(ValueError):
        GeminiService._normalize_code_id("Evidence-based reason", valid)
