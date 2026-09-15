from __future__ import annotations

import json
import re
from typing import Optional, Type

from google import genai
from pydantic import BaseModel, Field

from .models import CodeDefinition, ProtocolSnapshot
from .prompts import coding_prompt, codesign_prompt, explanation_prompt


class CodingResult(BaseModel):
    codes: list[str] = Field(default_factory=list)


class CodeDesignResult(BaseModel):
    message: str
    general_instructions: str
    codes: list[CodeDefinition]


class GeminiService:
    """Thin Gemini adapter so the research workflow can support another provider later."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.8-flash"):
        if not api_key:
            raise ValueError("Gemini API key is missing")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _structured_interaction(
        self,
        prompt: str,
        schema: Type[BaseModel],
        thinking_level: str,
    ) -> BaseModel:
        interaction = self.client.interactions.create(
            model=self.model_name,
            store=False,
            input=prompt,
            generation_config={"thinking_level": thinking_level},
            response_format=[
                {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema.model_json_schema(),
                }
            ],
        )
        if not interaction.output_text:
            raise ValueError("Gemini returned no text output")
        return schema.model_validate_json(interaction.output_text)

    @staticmethod
    def _normalize_code_id(candidate: str, valid_ids: set[str]) -> str:
        """Return a valid code ID, allowing only very conservative formatting cleanup.

        Exact IDs always win. The fallback only removes wrappers/prefixes that models
        sometimes add despite structured-output instructions (e.g., ``CODE A``).
        It never fuzzy-matches or guesses between codebook entries.
        """
        value = candidate.strip().strip("`\"\'")
        if value in valid_ids:
            return value

        # Harmless model-added prefixes: "CODE A", "CODE: A", "ID A", "ID: A".
        cleaned = re.sub(r"^(?:CODE|ID)\s*[:=-]?\s*", "", value, flags=re.IGNORECASE).strip()
        if cleaned in valid_ids:
            return cleaned

        raise ValueError(f"Gemini returned code not in the codebook: {candidate!r}")

    def code_case(
        self,
        snapshot: ProtocolSnapshot,
        response_text: str,
        contextual_data: Optional[dict[str, object]] = None,
    ) -> list[str]:
        valid_ids = {code.code_id for code in snapshot.codes}
        if not valid_ids:
            raise ValueError("The active protocol contains no code IDs")

        # The JSON schema itself enumerates the permitted IDs. This reduces the
        # chance that the model returns labels such as "CODE A" rather than "A".
        response_schema = {
            "type": "object",
            "properties": {
                "codes": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(valid_ids)},
                }
            },
            "required": ["codes"],
            "additionalProperties": False,
        }

        interaction = self.client.interactions.create(
            model=self.model_name,
            store=False,
            input=coding_prompt(snapshot, response_text, contextual_data),
            generation_config={"thinking_level": "low"},
            response_format=[
                {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": response_schema,
                }
            ],
        )
        if not interaction.output_text:
            raise ValueError("Gemini returned no text output")

        try:
            payload = json.loads(interaction.output_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini returned invalid JSON: {interaction.output_text!r}") from exc

        raw_codes = payload.get("codes", [])
        if not isinstance(raw_codes, list):
            raise ValueError("Gemini response field 'codes' was not a list")
        codes = [self._normalize_code_id(str(code), valid_ids) for code in raw_codes if str(code).strip()]

        if snapshot.coding_mode == "single" and len(codes) != 1:
            raise ValueError(f"Single-code protocol expected one code; Gemini returned {len(codes)}")
        if snapshot.coding_mode == "multi" and not codes:
            raise ValueError("Multi-code protocol returned no codes")
        return codes

    def explain_case(
        self,
        snapshot: ProtocolSnapshot,
        response_text: str,
        assigned_codes: list[str],
        user_question: str,
        contextual_data: Optional[dict[str, object]] = None,
    ) -> str:
        interaction = self.client.interactions.create(
            model=self.model_name,
            store=False,
            input=explanation_prompt(
                snapshot,
                response_text,
                assigned_codes,
                user_question,
                contextual_data,
            ),
            generation_config={"thinking_level": "low"},
        )
        return (interaction.output_text or "").strip()

    def redesign_codebook(
        self,
        snapshot: ProtocolSnapshot,
        user_request: str,
        literature_texts: Optional[list[str]] = None,
        calibration_evidence: str = "",
    ) -> CodeDesignResult:
        parsed = self._structured_interaction(
            codesign_prompt(
                snapshot,
                user_request,
                literature_texts=literature_texts,
                calibration_evidence=calibration_evidence,
            ),
            CodeDesignResult,
            thinking_level="medium",
        )
        assert isinstance(parsed, CodeDesignResult)

        # Local validation prevents malformed drafts from silently entering the project.
        ProtocolSnapshot(
            **snapshot.model_dump(exclude={"codes", "general_instructions"}),
            general_instructions=parsed.general_instructions,
            codes=parsed.codes,
        )
        return parsed
