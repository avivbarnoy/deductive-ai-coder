from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


CodingMode = Literal["single", "multi"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class CodeDefinition(BaseModel):
    """One deductive code in the researcher's codebook."""

    code_id: str = Field(..., min_length=1, description="Stable code identifier, e.g. A1 or 1.1")
    name: str = Field(..., min_length=1)
    definition: str = ""
    include_when: str = ""
    exclude_when: str = ""
    parent_id: Optional[str] = None

    @field_validator("code_id", "name")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be blank")
        return value

    @field_validator("definition", "include_when", "exclude_when")
    @classmethod
    def strip_optional_text(cls, value: str) -> str:
        return (value or "").strip()

    @field_validator("parent_id")
    @classmethod
    def normalize_parent(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class ProtocolSnapshot(BaseModel):
    project_name: str
    research_question: str = ""
    unit_of_analysis: str = ""
    study_context: str = ""
    general_instructions: str = ""
    coding_mode: CodingMode = "single"
    allow_hierarchy: bool = False
    codes: list[CodeDefinition]

    @model_validator(mode="after")
    def validate_codebook(self):
        ids = [code.code_id for code in self.codes]
        duplicates = {x for x in ids if ids.count(x) > 1}
        if duplicates:
            raise ValueError(f"Duplicate code IDs: {', '.join(sorted(duplicates))}")

        if self.allow_hierarchy:
            valid_ids = set(ids)
            for code in self.codes:
                if code.parent_id and code.parent_id not in valid_ids:
                    raise ValueError(
                        f"Parent code '{code.parent_id}' for '{code.code_id}' does not exist"
                    )
                if code.parent_id == code.code_id:
                    raise ValueError(f"Code '{code.code_id}' cannot be its own parent")
        elif any(code.parent_id for code in self.codes):
            raise ValueError("Parent IDs are present while hierarchy is disabled")
        return self


class ProtocolVersion(BaseModel):
    version_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    note: str = ""
    snapshot: ProtocolSnapshot


class CalibrationSummary(BaseModel):
    protocol_version: str
    created_at: str = Field(default_factory=utc_now_iso)
    n_cases: int
    n_human_coders: int
    agreement_metrics: dict[str, float | None] = Field(default_factory=dict)
    note: str = ""


class ResearchProject(BaseModel):
    """Portable project file. Raw participant data are intentionally not stored here."""

    schema_version: str = "1.0"
    project_name: str = "Untitled coding project"
    research_question: str = ""
    unit_of_analysis: str = ""
    study_context: str = ""
    general_instructions: str = (
        "Code only what is supported by the supplied text and context. "
        "Do not infer motives, identities, or meanings that are not reasonably supported."
    )
    coding_mode: CodingMode = "single"
    allow_hierarchy: bool = False
    codes: list[CodeDefinition] = Field(default_factory=list)
    versions: list[ProtocolVersion] = Field(default_factory=list)
    active_version_id: Optional[str] = None
    calibration_history: list[CalibrationSummary] = Field(default_factory=list)
    draft_dirty: bool = False
    literature_source_names: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def validate_project(self):
        # Reuse snapshot validation for current draft.
        ProtocolSnapshot(
            project_name=self.project_name,
            research_question=self.research_question,
            unit_of_analysis=self.unit_of_analysis,
            study_context=self.study_context,
            general_instructions=self.general_instructions,
            coding_mode=self.coding_mode,
            allow_hierarchy=self.allow_hierarchy,
            codes=self.codes,
        )
        if self.active_version_id:
            version_ids = {v.version_id for v in self.versions}
            if self.active_version_id not in version_ids:
                raise ValueError("active_version_id does not refer to an existing version")
        return self

    def current_snapshot(self) -> ProtocolSnapshot:
        return ProtocolSnapshot(
            project_name=self.project_name,
            research_question=self.research_question,
            unit_of_analysis=self.unit_of_analysis,
            study_context=self.study_context,
            general_instructions=self.general_instructions,
            coding_mode=self.coding_mode,
            allow_hierarchy=self.allow_hierarchy,
            codes=[code.model_copy(deep=True) for code in self.codes],
        )

    def freeze_new_version(self, note: str = "") -> ProtocolVersion:
        next_number = len(self.versions) + 1
        version = ProtocolVersion(
            version_id=f"v{next_number}",
            note=note.strip(),
            snapshot=self.current_snapshot(),
        )
        self.versions.append(version)
        self.active_version_id = version.version_id
        self.draft_dirty = False
        self.updated_at = utc_now_iso()
        return version

    def get_active_version(self) -> Optional[ProtocolVersion]:
        if not self.active_version_id:
            return None
        return next((v for v in self.versions if v.version_id == self.active_version_id), None)

    def mark_draft_changed(self) -> None:
        """Mark the working draft as different from the active frozen protocol."""
        self.draft_dirty = True
        self.updated_at = utc_now_iso()
