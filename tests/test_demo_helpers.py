from pathlib import Path
import ast


def _load_app_functions():
    """Extract pure helper functions from app.py without importing Streamlit app side effects."""
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"preferred_text_column", "looks_like_identifier_column"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {}
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace


def test_preferred_text_column_chooses_response_over_id():
    helpers = _load_app_functions()
    assert helpers["preferred_text_column"](["case_id", "response", "context"]) == "response"


def test_preferred_text_column_recognizes_common_names():
    helpers = _load_app_functions()
    assert helpers["preferred_text_column"](["participant_id", "Open ended response"]) == "Open ended response"
    assert helpers["preferred_text_column"](["id", "comment_text"]) == "comment_text"


def test_identifier_column_warning_logic():
    helpers = _load_app_functions()
    assert helpers["looks_like_identifier_column"]("case_id") is True
    assert helpers["looks_like_identifier_column"]("participant ID") is True
    assert helpers["looks_like_identifier_column"]("response") is False
