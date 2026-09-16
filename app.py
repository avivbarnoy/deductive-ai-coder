from __future__ import annotations

import hmac
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deductive_ai_coder.audit import build_audit_payload
from deductive_ai_coder.data_io import read_tabular_file, to_csv_bytes, to_excel_bytes, to_sav_bytes
from deductive_ai_coder.gemini_client import GeminiService
from deductive_ai_coder.models import CalibrationSummary, CodeDefinition, ResearchProject
from deductive_ai_coder.project_io import project_from_json, project_to_json
from deductive_ai_coder.reliability import normalize_code_set, pair_metrics
from deductive_ai_coder.text_extract import extract_text

APP_TITLE = "Deductive AI Coder"
PAGES = [
    "Stage 1: Protocol Co-Design",
    "Stage 2: Calibration",
    "Stage 3: Full Dataset Coding",
]

st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")


def read_secret(name: str, default: Any = None) -> Any:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except FileNotFoundError:
        pass
    return os.getenv(name, default)


GEMINI_API_KEY = str(read_secret("GEMINI_API_KEY", "") or "")
GEMINI_MODEL = str(read_secret("GEMINI_MODEL", "gemini-3.8-flash"))
REVIEWER_ACCESS_CODE = str(read_secret("REVIEWER_ACCESS_CODE", "") or "")
MAX_BATCH_ROWS = int(read_secret("MAX_BATCH_ROWS", 1000))
MAX_CALIBRATION_ROWS = int(read_secret("MAX_CALIBRATION_ROWS", 100))


def safe_filename(value: str, fallback: str = "coding_project") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return cleaned or fallback


def initialize_state() -> None:
    defaults = {
        "project": ResearchProject(),
        "stage1_chat": [],
        "literature_texts": {},
        "calibration_results": None,
        "calibration_table": None,
        "calibration_signature": None,
        "calibration_editor_epoch": 0,
        "refinement_draft": None,
        "case_explanation": "",
        "batch_source_df": None,
        "batch_results": None,
        "batch_text_column": None,
        "nav_page": PAGES[0],
        "reviewer_authenticated": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


def enforce_reviewer_gate() -> None:
    """Optional lightweight gate for a public reviewer deployment.

    Leave REVIEWER_ACCESS_CODE empty during local development. On a hosted demo,
    configure it as a server-side Streamlit secret. This is a convenience gate,
    not a replacement for enterprise authentication or rate limiting.
    """
    if not REVIEWER_ACCESS_CODE or st.session_state.reviewer_authenticated:
        return

    st.title(APP_TITLE)
    st.caption("Reviewer demonstration")
    st.write("Enter the access code supplied with the proposal to open the prototype.")
    with st.form("reviewer_access_form"):
        entered = st.text_input("Access code", type="password")
        submitted = st.form_submit_button("Open prototype", type="primary")
    if submitted:
        if hmac.compare_digest(entered, REVIEWER_ACCESS_CODE):
            st.session_state.reviewer_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect access code.")
    st.stop()


def get_project() -> ResearchProject:
    return st.session_state.project


def reset_run_state() -> None:
    st.session_state.calibration_results = None
    st.session_state.refinement_draft = None
    st.session_state.case_explanation = ""
    st.session_state.batch_results = None


def reset_calibration_editor() -> None:
    st.session_state.calibration_editor_epoch += 1


def get_service(require_authorization: bool = True) -> GeminiService | None:
    if not GEMINI_API_KEY:
        st.error(
            "The server has no Gemini API key configured. Manual editing remains available, "
            "but AI functions require a server-side key."
        )
        return None
    if require_authorization and not st.session_state.get("data_authorized", False):
        st.error("Confirm the research-data/API notice in the sidebar before sending material to Gemini.")
        return None
    return GeminiService(GEMINI_API_KEY, GEMINI_MODEL)


def protocol_status(project: ResearchProject) -> str:
    if project.active_version_id and project.draft_dirty:
        return f"Active frozen protocol: {project.active_version_id} · working draft has newer changes"
    if project.active_version_id:
        return f"Frozen protocol: {project.active_version_id}"
    if project.versions:
        return "No active frozen protocol selected"
    return "No protocol version frozen yet"


def is_bundled_demo_project(project: ResearchProject) -> bool:
    active = project.get_active_version()
    if not active or project.project_name != "Example deductive coding project":
        return False
    return {code.code_id for code in active.snapshot.codes} == {"A", "B", "C", "X"}


def preferred_text_column(columns: list[str]) -> str:
    if not columns:
        raise ValueError("Dataset has no columns")
    normalized = {str(col).strip().lower().replace("-", "_").replace(" ", "_"): str(col) for col in columns}
    candidates = [
        "response",
        "text",
        "answer",
        "comment",
        "open_response",
        "open_ended_response",
        "open_text",
        "verbatim",
        "transcript",
        "excerpt",
    ]
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for candidate in candidates:
        for normalized_name, original in normalized.items():
            if candidate in normalized_name:
                return original
    return columns[0]


def looks_like_identifier_column(column_name: str) -> bool:
    normalized = str(column_name).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {
        "id",
        "case_id",
        "caseid",
        "participant_id",
        "participantid",
        "respondent_id",
        "respondentid",
        "record_id",
        "recordid",
    } or normalized.endswith("_id")


def code_rows(project: ResearchProject) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code in project.codes:
        row = {
            "Code ID": code.code_id,
            "Name": code.name,
            "Definition": code.definition,
            "Include when": code.include_when,
            "Exclude when": code.exclude_when,
        }
        if project.allow_hierarchy:
            row["Parent ID"] = code.parent_id or ""
        rows.append(row)
    return rows


def dataframe_to_codes(df: pd.DataFrame, allow_hierarchy: bool) -> list[CodeDefinition]:
    codes: list[CodeDefinition] = []
    for _, row in df.fillna("").iterrows():
        code_id = str(row.get("Code ID", "")).strip()
        name = str(row.get("Name", "")).strip()
        if not code_id and not name:
            continue
        codes.append(
            CodeDefinition(
                code_id=code_id,
                name=name,
                definition=str(row.get("Definition", "")),
                include_when=str(row.get("Include when", "")),
                exclude_when=str(row.get("Exclude when", "")),
                parent_id=(str(row.get("Parent ID", "")).strip() or None) if allow_hierarchy else None,
            )
        )
    return codes


def blank_calibration_table(n_humans: int) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for i in range(10):
        row = {"Case ID": str(i + 1), "Text": "", "Context": "", "Human 1": ""}
        if n_humans == 2:
            row["Human 2"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def demo_calibration_table(n_humans: int) -> pd.DataFrame:
    examples = [
        ("1", "Official statistics show the rate has decreased.", "A"),
        ("2", "The study reports the same result across three datasets.", "A"),
        ("3", "I experienced this myself last year.", "B"),
        ("4", "This happened to my family, so I know it can occur.", "B"),
        ("5", "This is wrong because everyone should be treated equally.", "C"),
        ("6", "Fairness matters more to me than efficiency in this case.", "C"),
        ("7", "The train left at eight this morning.", "X"),
        ("8", "I do not really have a reason for my answer.", "X"),
    ]
    rows = []
    for case_id, text, code in examples:
        row = {"Case ID": case_id, "Text": text, "Context": "", "Human 1": code}
        if n_humans == 2:
            row["Human 2"] = code
        rows.append(row)
    return pd.DataFrame(rows)


def demo_batch_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"response": "Official statistics show the rate has decreased.", "case_id": "1", "context": ""},
            {"response": "I experienced this myself last year.", "case_id": "2", "context": ""},
            {"response": "Everyone deserves equal treatment, so this is wrong.", "case_id": "3", "context": ""},
            {"response": "I chose this option randomly.", "case_id": "4", "context": ""},
        ]
    )


def calibration_evidence_text(results: pd.DataFrame, n_humans: int, coding_mode: str) -> str:
    if results is None or results.empty:
        return ""

    def same(a: object, b: object) -> bool:
        if coding_mode == "multi":
            return normalize_code_set(str(a)) == normalize_code_set(str(b))
        return str(a).strip() == str(b).strip()

    if n_humans == 1:
        mask = results.apply(lambda row: not same(row["AI Code"], row["Human 1"]), axis=1)
    else:
        mask = results.apply(
            lambda row: (
                not same(row["AI Code"], row["Human 1"])
                or not same(row["AI Code"], row["Human 2"])
                or not same(row["Human 1"], row["Human 2"])
            ),
            axis=1,
        )

    pieces: list[str] = []
    for _, row in results[mask].head(20).iterrows():
        piece = [
            f"Case {row['Case ID']}",
            f"Text: {row['Text']}",
            f"Human 1: {row['Human 1']}",
        ]
        if n_humans == 2:
            piece.append(f"Human 2: {row.get('Human 2', '')}")
        piece.append(f"AI: {row['AI Code']}")
        pieces.append("\n".join(piece))
    return "\n\n---\n\n".join(pieces)


def show_sidebar() -> str:
    project = get_project()
    with st.sidebar:
        st.title(APP_TITLE)
        st.caption("Research prototype / MVP")
        if GEMINI_API_KEY:
            st.success(f"Gemini configured · {GEMINI_MODEL}")
        else:
            st.warning("Gemini API not configured on server")

        st.checkbox(
            "I confirm that material I send through AI functions is appropriate for processing by the configured external Gemini service and permitted under my research/ethics/data-management requirements.",
            key="data_authorized",
        )
        st.caption(
            "The API key stays server-side. Material used in AI functions is sent by the server to Gemini."
        )

        st.divider()
        if st.session_state.get("pending_nav") in PAGES:
            st.session_state.nav_page = st.session_state.pop("pending_nav")
        page = st.radio("Workflow", PAGES, key="nav_page")
        st.caption(protocol_status(project))

        st.divider()
        st.subheader("Project file")
        project_upload = st.file_uploader("Load a saved project (.json)", type=["json"], key="project_file_upload")
        if st.button("Load project", width="stretch"):
            if not project_upload:
                st.warning("Choose a project JSON file first.")
            else:
                try:
                    loaded = project_from_json(project_upload.getvalue())
                    st.session_state.project = loaded
                    st.session_state.stage1_chat = []
                    st.session_state.literature_texts = {}
                    st.session_state.calibration_table = None
                    st.session_state.calibration_signature = None
                    reset_calibration_editor()
                    reset_run_state()
                    st.session_state.batch_source_df = None
                    st.session_state.batch_text_column = None
                    st.success("Project loaded.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not load project: {exc}")

        st.download_button(
            "Save project",
            data=project_to_json(project),
            file_name=f"{safe_filename(project.project_name)}.json",
            mime="application/json",
            width="stretch",
        )

        if st.button("New blank project", width="stretch"):
            st.session_state.project = ResearchProject()
            st.session_state.stage1_chat = []
            st.session_state.literature_texts = {}
            st.session_state.calibration_table = None
            st.session_state.calibration_signature = None
            reset_calibration_editor()
            reset_run_state()
            st.session_state.batch_source_df = None
            st.session_state.batch_text_column = None
            st.rerun()

        st.divider()
        st.caption(
            "Raw calibration/full-dataset participant text is kept only in the active Streamlit session and is not included in the downloadable project JSON."
        )
    return page


def stage1() -> None:
    project = get_project()
    st.header("Stage 1 · Deductive Protocol Co-Design")
    st.write(
        "Define the study and codebook manually, with Gemini assistance, or using uploaded literature/source material. "
        "The structured protocol is the source of truth for calibration and later batch coding."
    )

    with st.expander("1. Project setup", expanded=not bool(project.codes)):
        with st.form("project_setup_form"):
            col1, col2 = st.columns(2)
            with col1:
                project_name = st.text_input("Project name", value=project.project_name)
                research_question = st.text_area("Research question (optional)", value=project.research_question, height=90)
                unit_of_analysis = st.text_input(
                    "Unit of analysis",
                    value=project.unit_of_analysis,
                    placeholder="e.g., one survey response; one interview excerpt",
                )
            with col2:
                study_context = st.text_area("Study/context information (optional)", value=project.study_context, height=90)
                coding_mode = st.radio(
                    "Coding mode",
                    ["single", "multi"],
                    index=0 if project.coding_mode == "single" else 1,
                    horizontal=True,
                    help="Single = exactly one code per case. Multi = one or more codes may be assigned.",
                )
                allow_hierarchy = st.checkbox("Allow parent/child code hierarchy", value=project.allow_hierarchy)

            general_instructions = st.text_area(
                "General coding instructions",
                value=project.general_instructions,
                height=120,
            )
            apply_settings = st.form_submit_button("Apply project settings")

        if apply_settings:
            before = project.current_snapshot().model_dump()
            project.project_name = project_name.strip() or "Untitled coding project"
            project.research_question = research_question.strip()
            project.unit_of_analysis = unit_of_analysis.strip()
            project.study_context = study_context.strip()
            project.general_instructions = general_instructions.strip()
            project.coding_mode = coding_mode
            project.allow_hierarchy = allow_hierarchy
            if not allow_hierarchy:
                for code in project.codes:
                    code.parent_id = None
            try:
                after = project.current_snapshot().model_dump()
                if before != after:
                    project.mark_draft_changed()
                    reset_run_state()
                st.success("Project settings updated.")
            except ValidationError as exc:
                st.error(str(exc))

    with st.expander("2. Optional literature / source material"):
        st.write(
            "Upload PDF, DOCX, TXT, or Markdown files if you want Gemini to use them while helping define deductive categories. "
            "Source text is held only in the current session."
        )
        uploaded_sources = st.file_uploader(
            "Source files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="literature_upload",
        )
        if st.button("Use these sources for co-design"):
            if not uploaded_sources:
                st.warning("Upload at least one source file.")
            else:
                extracted: dict[str, str] = {}
                for file in uploaded_sources:
                    try:
                        text = extract_text(file.name, file.getvalue()).strip()
                        if text:
                            extracted[file.name] = text
                        else:
                            st.warning(f"{file.name}: no extractable text")
                    except Exception as exc:
                        st.warning(f"{file.name}: {exc}")
                st.session_state.literature_texts = extracted
                project.literature_source_names = list(extracted.keys())
                if extracted:
                    st.success(f"Loaded {len(extracted)} source file(s) for this session.")
        if st.session_state.literature_texts:
            st.caption("In session: " + ", ".join(st.session_state.literature_texts.keys()))

    st.subheader("3. Build and refine the coding protocol")
    col_chat, col_codebook = st.columns([0.9, 1.1], gap="large")

    with col_chat:
        st.markdown("#### AI co-design conversation")
        st.caption("Ask Gemini to propose, define, merge, split, or clarify deductive categories.")
        for message in st.session_state.stage1_chat:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_request = st.chat_input(
            "e.g., Help me derive categories from the uploaded literature...",
            key="stage1_chat_input",
        )
        if user_request:
            st.session_state.stage1_chat.append({"role": "user", "content": user_request})
            service = get_service()
            if service:
                try:
                    result = service.redesign_codebook(
                        project.current_snapshot(),
                        user_request=user_request,
                        literature_texts=list(st.session_state.literature_texts.values()),
                    )
                    project.codes = result.codes
                    project.general_instructions = result.general_instructions.strip()
                    project.mark_draft_changed()
                    reset_run_state()
                    st.session_state.stage1_chat.append({"role": "assistant", "content": result.message})
                except Exception as exc:
                    st.session_state.stage1_chat.append({"role": "assistant", "content": f"AI request failed: {exc}"})
            st.rerun()

    with col_codebook:
        st.markdown("#### Working codebook")
        columns = ["Code ID", "Name", "Definition", "Include when", "Exclude when"]
        if project.allow_hierarchy:
            columns.append("Parent ID")
        initial_df = pd.DataFrame(code_rows(project), columns=columns)
        with st.form(f"codebook_form_{len(project.versions)}_{project.allow_hierarchy}_{len(project.codes)}"):
            edited = st.data_editor(
                initial_df,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key=f"codebook_editor_{len(project.versions)}_{project.allow_hierarchy}_{len(project.codes)}",
            )
            apply_codebook = st.form_submit_button("Apply manual codebook edits", width="stretch")

        if apply_codebook:
            old_codes = [code.model_copy(deep=True) for code in project.codes]
            try:
                new_codes = dataframe_to_codes(edited, project.allow_hierarchy)
                project.codes = new_codes
                project.current_snapshot()
                if [c.model_dump() for c in old_codes] != [c.model_dump() for c in new_codes]:
                    project.mark_draft_changed()
                    reset_run_state()
                st.success("Codebook draft updated.")
            except Exception as exc:
                project.codes = old_codes
                st.error(f"Codebook not updated: {exc}")

        with st.expander("Advanced · View generated coding protocol"):
            try:
                from deductive_ai_coder.prompts import codebook_as_text

                snapshot = project.current_snapshot()
                st.code(
                    f"GENERAL INSTRUCTIONS\n{snapshot.general_instructions}\n\nCODEBOOK\n{codebook_as_text(snapshot)}",
                    language="text",
                )
            except Exception as exc:
                st.warning(f"Draft is not yet valid: {exc}")

    st.divider()
    st.subheader("4. Freeze a protocol version")
    if project.active_version_id and project.draft_dirty:
        active_for_revert = project.get_active_version()
        if active_for_revert and st.button(f"Revert working draft to {active_for_revert.version_id}"):
            snap = active_for_revert.snapshot
            project.project_name = snap.project_name
            project.research_question = snap.research_question
            project.unit_of_analysis = snap.unit_of_analysis
            project.study_context = snap.study_context
            project.general_instructions = snap.general_instructions
            project.coding_mode = snap.coding_mode
            project.allow_hierarchy = snap.allow_hierarchy
            project.codes = [code.model_copy(deep=True) for code in snap.codes]
            project.draft_dirty = False
            reset_run_state()
            st.rerun()

    version_note = st.text_input("Version note (optional)", placeholder="e.g., Initial codebook after literature review")
    if st.button("Freeze current draft as new protocol version", type="primary"):
        try:
            if not project.codes:
                raise ValueError("Add at least one code before freezing a protocol.")
            version = project.freeze_new_version(version_note)
            reset_run_state()
            st.session_state.calibration_signature = None
            st.session_state.calibration_table = None
            reset_calibration_editor()
            st.success(f"Created {version.version_id}. This version is now active.")
        except Exception as exc:
            st.error(f"Could not freeze protocol: {exc}")

    if project.versions:
        versions_df = pd.DataFrame(
            [
                {
                    "Version": v.version_id,
                    "Created": v.created_at,
                    "Codes": len(v.snapshot.codes),
                    "Mode": v.snapshot.coding_mode,
                    "Note": v.note,
                    "Active": v.version_id == project.active_version_id,
                }
                for v in project.versions
            ]
        )
        st.dataframe(versions_df, hide_index=True, width="stretch")


def render_metric_pair(label: str, a: pd.Series, b: pd.Series, coding_mode: str) -> dict[str, float | None]:
    metrics = pair_metrics(a, b, coding_mode)
    st.markdown(f"**{label}**")
    for metric_name, value in metrics.items():
        pretty = metric_name.replace("_", " ").title().replace("Cohen Kappa", "Cohen’s κ")
        st.metric(pretty, "N/A" if value is None else f"{value:.2f}")
        if metric_name == "cohen_kappa" and value is None:
            st.caption(
                "Cohen’s κ cannot be calculated for these cases. This commonly occurs when there are too few usable cases or insufficient category variation."
            )
    return metrics


def stage2() -> None:
    project = get_project()
    st.header("Stage 2 · Calibration Against Human Coding")
    active = project.get_active_version()
    if not active:
        st.warning("Freeze a protocol version in Stage 1 before calibration.")
        return

    snapshot = active.snapshot
    st.info(
        f"Calibration will use **{active.version_id}** with {len(snapshot.codes)} codes in **{snapshot.coding_mode}** mode."
    )
    if project.draft_dirty:
        st.warning("The working draft has newer changes. Calibration still uses the frozen active version shown above.")

    n_humans = st.radio("Number of human coders", [1, 2], horizontal=True, key="n_humans")
    signature = (active.version_id, n_humans, snapshot.coding_mode)
    if st.session_state.calibration_signature != signature:
        st.session_state.calibration_signature = signature
        if is_bundled_demo_project(project):
            st.session_state.calibration_table = demo_calibration_table(n_humans)
        else:
            st.session_state.calibration_table = blank_calibration_table(n_humans)
        st.session_state.calibration_results = None
        st.session_state.refinement_draft = None
        st.session_state.case_explanation = ""
        reset_calibration_editor()

    st.markdown("### Calibration cases")
    if is_bundled_demo_project(project):
        st.info("Synthetic calibration cases are preloaded for the bundled sample project so reviewers can run Stage 2 immediately.")
    st.caption(
        "Edits in this table are submitted together when you press **Run AI coding & compare**. "
        "This prevents Streamlit reruns from clearing the first edit."
    )

    c1, c2 = st.columns(2)
    if c1.button("Load built-in demo cases"):
        st.session_state.calibration_table = demo_calibration_table(n_humans)
        st.session_state.calibration_results = None
        reset_calibration_editor()
        st.rerun()
    if c2.button("Reset calibration table"):
        st.session_state.calibration_table = blank_calibration_table(n_humans)
        st.session_state.calibration_results = None
        reset_calibration_editor()
        st.rerun()

    code_ids = [code.code_id for code in snapshot.codes]
    column_config: dict[str, Any] = {}
    if snapshot.coding_mode == "single":
        options = [""] + code_ids
        column_config["Human 1"] = st.column_config.SelectboxColumn("Human 1", options=options)
        if n_humans == 2:
            column_config["Human 2"] = st.column_config.SelectboxColumn("Human 2", options=options)
    else:
        st.caption("Multi-code mode: enter multiple code IDs separated by commas.")

    editor_key = (
        f"calibration_editor_{active.version_id}_{n_humans}_{snapshot.coding_mode}_"
        f"{st.session_state.calibration_editor_epoch}"
    )
    with st.form(f"calibration_form_{editor_key}"):
        calibration_table = st.data_editor(
            st.session_state.calibration_table,
            column_config=column_config,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key=editor_key,
        )
        run_calibration = st.form_submit_button("Run AI coding & compare", type="primary")

    if run_calibration:
        st.session_state.calibration_table = calibration_table.copy()
        valid = calibration_table[calibration_table["Text"].astype(str).str.strip() != ""].copy()
        valid = valid[valid["Human 1"].astype(str).str.strip() != ""].copy()
        if n_humans == 2:
            valid = valid[valid["Human 2"].astype(str).str.strip() != ""].copy()

        if valid.empty:
            st.warning("Add at least one case with the required human code(s).")
        elif len(valid) > MAX_CALIBRATION_ROWS:
            st.error(f"This deployment allows at most {MAX_CALIBRATION_ROWS} calibration cases per run.")
        else:
            service = get_service()
            if service:
                ai_codes: list[str] = []
                errors: list[str] = []
                progress = st.progress(0)
                status = st.empty()
                for pos, (_, row) in enumerate(valid.iterrows(), start=1):
                    status.info(f"Coding case {pos} of {len(valid)} with Gemini…")
                    try:
                        context = {"case_context": row["Context"]} if str(row["Context"]).strip() else {}
                        codes = service.code_case(snapshot, str(row["Text"]), context)
                        ai_codes.append(", ".join(codes))
                        errors.append("")
                    except Exception as exc:
                        ai_codes.append("")
                        errors.append(str(exc))
                    progress.progress(pos / len(valid))
                status.empty()
                valid["AI Code"] = ai_codes
                valid["AI Error"] = errors
                st.session_state.calibration_results = valid
                st.session_state.case_explanation = ""
                st.session_state.refinement_draft = None
                n_ok = sum(1 for error in errors if not error)
                if n_ok == len(valid):
                    st.success(f"AI coding completed for {n_ok} case(s).")
                elif n_ok == 0:
                    st.error("AI coding returned errors for all cases. See the AI Error column below.")
                else:
                    st.warning(f"AI coding completed for {n_ok} of {len(valid)} cases.")

    results = st.session_state.calibration_results
    if results is None:
        return

    st.markdown("### Comparison")
    st.dataframe(results, hide_index=True, width="stretch")
    usable = results[results["AI Error"] == ""].copy()
    if usable.empty:
        st.error("No AI-coded cases completed successfully.")
        return

    pairs: list[tuple[str, str, str]] = [("Human 1 vs AI", "Human 1", "AI Code")]
    if n_humans == 2:
        pairs.extend(
            [
                ("Human 2 vs AI", "Human 2", "AI Code"),
                ("Human 1 vs Human 2", "Human 1", "Human 2"),
            ]
        )

    metric_values: dict[str, float | None] = {}
    cols = st.columns(len(pairs))
    for col, (label, a_col, b_col) in zip(cols, pairs):
        with col:
            metrics = render_metric_pair(label, usable[a_col], usable[b_col], snapshot.coding_mode)
            for metric_name, value in metrics.items():
                metric_values[f"{label}: {metric_name}"] = value

    st.caption(
        f"Reliability statistics are based on {len(usable)} usable calibration case(s). "
        "The app reports the statistics but does not automatically declare a protocol validated at a fixed threshold."
    )

    if st.button("Save this calibration summary in the project"):
        project.calibration_history.append(
            CalibrationSummary(
                protocol_version=active.version_id,
                n_cases=len(usable),
                n_human_coders=n_humans,
                agreement_metrics=metric_values,
            )
        )
        st.success("Calibration summary saved. Raw case text was not added to the project file.")

    st.divider()
    st.markdown("### Inspect a specific AI decision")
    case_options = [str(x) for x in usable["Case ID"].tolist()]
    selected_case = st.selectbox("Case", case_options, key="explain_case_select")
    question = st.text_input("Question for Gemini", value="Why did you assign this code?", key="case_question")
    if st.button("Ask about this coding decision"):
        row = usable[usable["Case ID"].astype(str) == selected_case].iloc[0]
        service = get_service()
        if service:
            try:
                context = {"case_context": row["Context"]} if str(row["Context"]).strip() else {}
                assigned = [x.strip() for x in str(row["AI Code"]).split(",") if x.strip()]
                st.session_state.case_explanation = service.explain_case(
                    snapshot,
                    str(row["Text"]),
                    assigned,
                    question,
                    context,
                )
            except Exception as exc:
                st.session_state.case_explanation = f"Explanation request failed: {exc}"
    if st.session_state.case_explanation:
        st.info(st.session_state.case_explanation)

    st.divider()
    st.markdown("### Use disagreements to refine the protocol")
    refinement_request = st.text_area(
        "Optional guidance",
        placeholder="e.g., Clarify the boundary between A2 and A3; do not add new categories unless necessary.",
        key="refinement_request",
    )
    if st.button("Suggest protocol refinements"):
        evidence = calibration_evidence_text(usable, n_humans, snapshot.coding_mode)
        if not evidence:
            st.info("No disagreement cases were found in the usable calibration results.")
        else:
            service = get_service()
            if service:
                try:
                    request = refinement_request.strip() or (
                        "Review the disagreement cases and propose the smallest justified changes to improve clarity and coding consistency."
                    )
                    st.session_state.refinement_draft = service.redesign_codebook(
                        snapshot,
                        user_request=request,
                        calibration_evidence=evidence,
                    )
                except Exception as exc:
                    st.error(f"Could not generate refinement draft: {exc}")

    draft = st.session_state.refinement_draft
    if draft:
        st.success(draft.message)
        preview = pd.DataFrame(
            [
                {
                    "Code ID": c.code_id,
                    "Name": c.name,
                    "Definition": c.definition,
                    "Include when": c.include_when,
                    "Exclude when": c.exclude_when,
                    "Parent ID": c.parent_id or "",
                }
                for c in draft.codes
            ]
        )
        if not snapshot.allow_hierarchy and "Parent ID" in preview:
            preview = preview.drop(columns=["Parent ID"])
        st.dataframe(preview, hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        if c1.button("Accept as new working draft", type="primary"):
            project.codes = [code.model_copy(deep=True) for code in draft.codes]
            project.general_instructions = draft.general_instructions
            project.mark_draft_changed()
            reset_run_state()
            st.session_state.pending_nav = PAGES[0]
            st.rerun()
        if c2.button("Discard suggestion"):
            st.session_state.refinement_draft = None
            st.rerun()


def stage3() -> None:
    project = get_project()
    st.header("Stage 3 · Code the Full Dataset")
    active = project.get_active_version()
    if not active:
        st.warning("Freeze a protocol version in Stage 1 before full-dataset coding.")
        return

    snapshot = active.snapshot
    st.info(
        f"This run will use frozen protocol **{active.version_id}** ({len(snapshot.codes)} codes, {snapshot.coding_mode} mode)."
    )
    if project.draft_dirty:
        st.warning("The working draft has newer changes; they are not part of this batch.")

    dataset_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls", "sav"], key="batch_dataset_upload")
    c1, c2 = st.columns(2)
    if c1.button("Load uploaded dataset"):
        if not dataset_file:
            st.warning("Choose a dataset file first.")
        else:
            try:
                loaded_df = read_tabular_file(dataset_file.name, dataset_file.getvalue())
                st.session_state.batch_source_df = loaded_df
                st.session_state.batch_results = None
                loaded_columns = [str(c) for c in loaded_df.columns]
                st.session_state.batch_text_column = preferred_text_column(loaded_columns) if loaded_columns else None
                st.success(f"Loaded {len(loaded_df)} rows and {len(loaded_df.columns)} columns.")
            except Exception as exc:
                st.error(f"Could not read dataset: {exc}")
    if c2.button("Load built-in demo dataset"):
        st.session_state.batch_source_df = demo_batch_dataset()
        st.session_state.batch_results = None
        st.session_state.batch_text_column = "response"
        st.rerun()

    source_df = st.session_state.batch_source_df
    if source_df is None:
        st.caption("Supported formats: CSV, Excel, and SPSS SAV.")
        return
    if source_df.empty:
        st.warning("The loaded dataset has no rows.")
        return

    st.dataframe(source_df.head(20), width="stretch", hide_index=True)
    columns = [str(c) for c in source_df.columns]
    if st.session_state.batch_text_column not in columns:
        st.session_state.batch_text_column = preferred_text_column(columns)
    text_column = st.selectbox("Column containing text to code", columns, key="batch_text_column")
    st.info(f"AI will code the text in column **{text_column}**.")
    if looks_like_identifier_column(text_column):
        st.warning(
            f"**{text_column}** looks like an identifier column rather than qualitative text. "
            "Select the response/text column before running AI coding."
        )
    context_columns = st.multiselect(
        "Optional case-level context columns to send to the AI",
        [c for c in columns if c != text_column],
        help="Choose only variables that are methodologically relevant and appropriate to send to the configured AI service.",
    )

    st.warning(f"This deployment allows at most {MAX_BATCH_ROWS} rows per batch run to control prototype/API costs.")
    if st.button("Run full-dataset coding", type="primary"):
        if len(source_df) > MAX_BATCH_ROWS:
            st.error(f"Dataset has {len(source_df)} rows. This deployment allows at most {MAX_BATCH_ROWS}.")
        else:
            service = get_service()
            if service:
                output = source_df.copy()
                ai_codes: list[str] = []
                ai_labels: list[str] = []
                errors: list[str] = []
                label_map = {c.code_id: c.name for c in snapshot.codes}
                progress = st.progress(0)
                status = st.empty()
                for pos, (_, row) in enumerate(output.iterrows(), start=1):
                    status.info(f"Coding row {pos} of {len(output)} with Gemini…")
                    text = "" if pd.isna(row[text_column]) else str(row[text_column])
                    if not text.strip():
                        ai_codes.append("")
                        ai_labels.append("")
                        errors.append("Blank text; not sent to model")
                        progress.progress(pos / len(output))
                        continue
                    context = {col: (None if pd.isna(row[col]) else row[col]) for col in context_columns}
                    try:
                        codes = service.code_case(snapshot, text, context)
                        ai_codes.append("; ".join(codes))
                        ai_labels.append("; ".join(label_map[c] for c in codes))
                        errors.append("")
                    except Exception as exc:
                        ai_codes.append("")
                        ai_labels.append("")
                        errors.append(str(exc))
                    progress.progress(pos / len(output))
                status.empty()

                output["AI_Code"] = ai_codes
                output["AI_Code_Label"] = ai_labels
                output["AI_Protocol_Version"] = active.version_id
                output["AI_Model"] = GEMINI_MODEL
                output["AI_Error"] = errors
                st.session_state.batch_results = {
                    "df": output,
                    "text_column": text_column,
                    "context_columns": context_columns,
                }

    result_bundle = st.session_state.batch_results
    if not result_bundle:
        return

    output = result_bundle["df"]
    st.markdown("### Results")
    successful = int((output["AI_Error"] == "").sum())
    st.success(f"Completed: {successful}/{len(output)} rows coded without an application/API error.")
    st.caption(f"AI coded text from column: **{result_bundle['text_column']}**")
    st.dataframe(output.head(100), width="stretch", hide_index=True)

    st.markdown("### Export")
    base_name = safe_filename(project.project_name, "coded_dataset")
    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download CSV",
        to_csv_bytes(output),
        file_name=f"{base_name}_{active.version_id}.csv",
        mime="text/csv",
        width="stretch",
    )
    c2.download_button(
        "Download Excel",
        to_excel_bytes(output),
        file_name=f"{base_name}_{active.version_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    try:
        sav_bytes, name_mapping = to_sav_bytes(output)
        c3.download_button(
            "Download SPSS SAV",
            sav_bytes,
            file_name=f"{base_name}_{active.version_id}.sav",
            mime="application/x-spss-sav",
            width="stretch",
        )
        changed_names = {k: v for k, v in name_mapping.items() if k != v}
        if changed_names:
            with st.expander("SPSS variable-name mapping"):
                st.json(changed_names)
    except Exception as exc:
        c3.warning(f"SPSS export unavailable for this result: {exc}")

    audit = build_audit_payload(
        active,
        GEMINI_MODEL,
        coded_rows=len(output),
        text_column=result_bundle["text_column"],
        context_columns=result_bundle["context_columns"],
    )
    st.download_button(
        "Download coding audit record (JSON)",
        audit,
        file_name=f"{base_name}_{active.version_id}_audit.json",
        mime="application/json",
    )
    st.caption("The audit record contains the frozen protocol and run configuration, but not the participant dataset itself.")


enforce_reviewer_gate()
page = show_sidebar()

st.title(APP_TITLE)
st.caption(
    "A transparent MVP for deductive qualitative protocol co-design, human–AI calibration, and reproducible batch coding."
)

if page == PAGES[0]:
    stage1()
elif page == PAGES[1]:
    stage2()
else:
    stage3()
