from __future__ import annotations

from io import BytesIO

from docx import Document
from pypdf import PdfReader


def extract_text(filename: str, data: bytes) -> str:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix in {"txt", "md"}:
        return data.decode("utf-8", errors="replace")
    if suffix == "pdf":
        reader = PdfReader(BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == "docx":
        document = Document(BytesIO(data))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise ValueError(f"Unsupported literature file type: .{suffix}")
