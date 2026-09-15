from __future__ import annotations

import json

from .models import ResearchProject


def project_to_json(project: ResearchProject) -> str:
    return project.model_dump_json(indent=2)


def project_from_json(payload: str | bytes) -> ResearchProject:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    data = json.loads(payload)
    return ResearchProject.model_validate(data)
