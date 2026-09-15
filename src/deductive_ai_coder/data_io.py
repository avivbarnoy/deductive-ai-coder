from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path

import pandas as pd
import pyreadstat


def read_tabular_file(filename: str, data: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    bio = io.BytesIO(data)
    if suffix == ".csv":
        return pd.read_csv(bio)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(bio)
    if suffix == ".sav":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            df, _meta = pyreadstat.read_sav(path)
            return df
        finally:
            Path(path).unlink(missing_ok=True)
    raise ValueError("Supported dataset formats are CSV, XLSX/XLS, and SPSS SAV.")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="coded_data")
    return output.getvalue()


def _safe_spss_name(name: str, used: set[str]) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_@#$]", "_", str(name)).strip("_") or "var"
    if not re.match(r"^[A-Za-z@#$]", candidate):
        candidate = f"v_{candidate}"
    candidate = candidate[:64]
    base = candidate
    counter = 2
    while candidate.lower() in {x.lower() for x in used}:
        suffix = f"_{counter}"
        candidate = (base[: 64 - len(suffix)] + suffix)
        counter += 1
    used.add(candidate)
    return candidate


def to_sav_bytes(df: pd.DataFrame) -> tuple[bytes, dict[str, str]]:
    used: set[str] = set()
    mapping = {str(col): _safe_spss_name(str(col), used) for col in df.columns}
    sav_df = df.rename(columns=mapping).copy()

    # SPSS export is most predictable when object columns are strings and missing values remain empty.
    for col in sav_df.columns:
        if sav_df[col].dtype == "object":
            sav_df[col] = sav_df[col].fillna("").astype(str)

    labels = {mapping[str(col)]: f"Original column: {col}" for col in df.columns}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
        path = tmp.name
    try:
        pyreadstat.write_sav(sav_df, path, column_labels=labels)
        payload = Path(path).read_bytes()
        return payload, mapping
    finally:
        Path(path).unlink(missing_ok=True)
