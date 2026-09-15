from __future__ import annotations

import json
from datetime import datetime, timezone

from .models import ProtocolVersion


def build_audit_payload(
    protocol: ProtocolVersion,
    model_name: str,
    coded_rows: int,
    text_column: str,
    context_columns: list[str],
) -> bytes:
    payload = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "model_provider": "Google Gemini",
        "model_name": model_name,
        "protocol_version": protocol.model_dump(),
        "coded_rows": coded_rows,
        "text_column": text_column,
        "context_columns": context_columns,
        "note": (
            "This audit record describes the application configuration used for the run. "
            "Model providers may update hosted models over time; model name alone may not uniquely identify model weights."
        ),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
