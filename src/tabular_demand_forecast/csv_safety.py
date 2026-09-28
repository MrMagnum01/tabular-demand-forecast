"""CSV formula-injection mitigation, applied to every CSV this repo writes.

Mitigation (one, explicit, documented):
    Any STRING cell whose first character is one of
        '='  '+'  '-'  '@'  TAB (0x09)  CR (0x0D)
    is written with a single leading apostrophe (') prepended, e.g.
    '=1+1' is written as "'=1+1" and '@SUM(A1)' as "'@SUM(A1)".

Scope and limits (read before relying on this):
  * It applies to string-typed cells only (object/str dtype columns and
    column headers). Numeric dtype cells (int/float/bool) are written as
    numbers; a float such as -0.53 is a literal number, not a formula, and
    is intentionally NOT prefixed so the file stays machine-readable.
  * The apostrophe is visible to non-spreadsheet consumers (pandas, csv
    readers): a sanitised cell reads back as "'=1+1", not "=1+1".
  * CSV is not universally formula-safe. Spreadsheet applications differ
    in what they interpret (e.g. DDE payloads, locale-specific separators,
    applications that strip a leading apostrophe on re-save). This
    mitigation lowers risk for the common Excel/LibreOffice/Sheets case;
    it is not a guarantee. Treat any CSV from anywhere as untrusted input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
MITIGATION_PREFIX = "'"


def sanitize_cell(value: object) -> object:
    """Prefix a string cell with an apostrophe if it starts with a
    formula-trigger character; return non-strings unchanged."""
    if isinstance(value, str) and value.startswith(DANGEROUS_PREFIXES):
        return MITIGATION_PREFIX + value
    return value


def sanitize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            continue
        out[col] = series.map(sanitize_cell).astype(object)
    out.columns = [sanitize_cell(str(c)) for c in out.columns]
    return out


def write_safe_csv(frame: pd.DataFrame, path: Union[str, Path]) -> None:
    """The only CSV writer used in this repo."""
    sanitize_frame(frame).to_csv(path, index=False, lineterminator="\n")
