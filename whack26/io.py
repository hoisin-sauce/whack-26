"""Dataset ingestion with explicit engines and typed read failures."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_FORMAT_BY_SUFFIX = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
    ".parquet": "parquet",
}
_PICKLE_SUFFIXES = {".pkl", ".pickle"}
_SUPPORTED_DISPLAY = (
    ".csv, .tsv, .tab, .xlsx, .xls, .json, .jsonl, .ndjson, .parquet"
)
_JSON_LINES_SUFFIXES = {".jsonl", ".ndjson"}


class IngestError(ValueError):
    """Raised when a dataset cannot be read or validated."""


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    format: str
    source_path: Path
    warnings: list[str]


def _json_decode_error(path: Path, *, lines: bool) -> json.JSONDecodeError | None:
    """Return the layout's decode error, or None when the layout parses cleanly.

    pandas' JSON reader reports malformed payloads as plain ``ValueError``
    instances, so classify the layout with the stdlib parser instead. A
    syntactically valid payload that pandas rejects is a schema/type error and
    must not be retried as the other layout.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        if lines:
            for raw_line in text.splitlines():
                if raw_line.strip():
                    json.loads(raw_line)
        else:
            json.loads(text)
    except json.JSONDecodeError as exc:
        return exc
    return None


def _read_json(path: Path, *, lines_first: bool) -> pd.DataFrame:
    """Read records JSON, retrying with the other JSON layout on failure.

    Only malformed JSON (a layout mismatch) is retried. A payload that is
    syntactically valid but has the wrong shape is a schema error and is
    surfaced directly.
    """
    attempts = (True, False) if lines_first else (False, True)
    errors: list[str] = []
    for lines in attempts:
        layout = "JSON lines" if lines else "records/array"
        try:
            return pd.read_json(path, lines=lines)
        except Exception as exc:  # pandas raises several unrelated types here
            errors.append(f"{layout}: {type(exc).__name__}: {exc}")
            layout_error = _json_decode_error(path, lines=lines)
            if isinstance(exc, json.JSONDecodeError) or layout_error is None:
                raise ValueError(
                    f"JSON payload is valid as {layout} data but could not be "
                    f"read as a table: {type(exc).__name__}: {exc}"
                ) from exc
    raise ValueError("could not parse the JSON payload (" + "; ".join(errors) + ")")


def _read_frame(path: Path, fmt: str, *, csv_sep: str) -> pd.DataFrame:
    try:
        if fmt == "csv":
            return pd.read_csv(path, sep=csv_sep)
        if fmt == "tsv":
            return pd.read_csv(path, sep="\t")
        if fmt == "excel":
            engine = "openpyxl" if path.suffix.lower() == ".xlsx" else "xlrd"
            return pd.read_excel(path, engine=engine)
        if fmt == "parquet":
            return pd.read_parquet(path, engine="pyarrow")
        if fmt == "json":
            return _read_json(
                path, lines_first=path.suffix.lower() in _JSON_LINES_SUFFIXES
            )
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(
            f"failed to read {fmt} dataset {path.name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    raise IngestError(f"unsupported dataset format for {path.name!r}")


def _raw_header(path: Path, fmt: str, *, csv_sep: str) -> list[str] | None:
    """Return the source header, or None when it cannot be inspected cheaply."""
    try:
        if fmt in {"csv", "tsv"}:
            delimiter = csv_sep if fmt == "csv" else "\t"
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.reader(handle, delimiter=delimiter):
                    # pandas skips blank leading lines; match that so the raw
                    # header still lines up with the parsed frame.
                    if row and any(cell.strip() for cell in row):
                        return row
            return None
        if fmt == "excel" and path.suffix.lower() == ".xlsx":
            from openpyxl import load_workbook

            workbook = load_workbook(path, read_only=True, data_only=True)
            try:
                sheet = workbook.active
                row = next(
                    sheet.iter_rows(min_row=1, max_row=1, values_only=True), None
                )
            finally:
                workbook.close()
            if row is None:
                return None
            return ["" if value is None else str(value) for value in row]
        if fmt == "excel" and path.suffix.lower() == ".xls":
            import xlrd

            workbook = xlrd.open_workbook(path, on_demand=True)
            try:
                sheet = workbook.sheet_by_index(0)
                if sheet.nrows < 1:
                    return None
                return [str(sheet.cell_value(0, column)) for column in range(sheet.ncols)]
            finally:
                workbook.release_resources()
    except Exception:  # optional engines or odd files: pandas remains the source of truth
        return None
    return None


def _check_rectangular_csv(
    path: Path, fmt: str, *, csv_sep: str, width: int
) -> None:
    """Reject rows that carry more fields than a duplicate header defines.

    pandas silently drops extra trailing values in some ragged CSVs, which
    would lose data once the duplicate header is restored. Short rows are left
    alone because pandas pads them with missing values without losing input.
    """
    delimiter = csv_sep if fmt == "csv" else "\t"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_seen = False
        reader = csv.reader(handle, delimiter=delimiter)
        for record_number, row in enumerate(reader, start=1):
            if not row or not any(cell.strip() for cell in row):
                continue
            if not header_seen:
                header_seen = True
                continue
            if len(row) > width:
                raise IngestError(
                    f"ragged CSV dataset {path.name!r}: the header defines "
                    f"{width} field(s) but record {record_number} has {len(row)}; "
                    "repair the delimiters/quoting or remove the duplicate header "
                    "before ingesting"
                )


def _restore_duplicate_header(
    frame: pd.DataFrame, path: Path, fmt: str, *, csv_sep: str
) -> None:
    """Undo pandas' ``.1`` mangling so duplicate headers get our ``__2`` names."""
    raw_header = _raw_header(path, fmt, csv_sep=csv_sep)
    if raw_header is None:
        return
    if len(set(raw_header)) == len(raw_header):
        return
    if len(raw_header) != frame.shape[1]:
        raise IngestError(
            f"cannot restore duplicate column headers for {path.name!r}: the "
            f"header has {len(raw_header)} field(s) but the parsed table has "
            f"{frame.shape[1]} column(s); fix the ragged header or rows and retry"
        )
    if fmt in {"csv", "tsv"}:
        _check_rectangular_csv(path, fmt, csv_sep=csv_sep, width=len(raw_header))
    frame.columns = raw_header


def _normalize_columns(frame: pd.DataFrame, warnings: list[str]) -> None:
    """Stringify column names and rename later duplicates in place."""
    original_names = [str(name) for name in frame.columns]
    reserved = set(original_names)
    used: set[str] = set()
    names: list[str] = []
    renamed: list[tuple[str, str]] = []

    for name in original_names:
        if name not in used:
            used.add(name)
            names.append(name)
            continue

        suffix = 2
        while f"{name}__{suffix}" in used or f"{name}__{suffix}" in reserved:
            suffix += 1
        replacement = f"{name}__{suffix}"
        used.add(replacement)
        names.append(replacement)
        renamed.append((name, replacement))

    if renamed:
        preview = ", ".join(f"{old!r} -> {new!r}" for old, new in renamed[:10])
        extra = f" (and {len(renamed) - 10} more)" if len(renamed) > 10 else ""
        warnings.append(
            f"duplicate column names were renamed to keep them unique: {preview}{extra}"
        )

    frame.columns = names


def read_dataset(
    path: str | Path,
    *,
    target: str | None = None,
    csv_sep: str = ",",
) -> LoadedDataset:
    """Read a supported tabular dataset and validate its basic shape."""
    source_path = Path(path)
    suffix = source_path.suffix.lower()

    if suffix in _PICKLE_SUFFIXES:
        raise IngestError(
            "pickle datasets are not supported; export the data to CSV, JSON, "
            "Excel, or Parquet instead"
        )
    if suffix not in _FORMAT_BY_SUFFIX:
        shown = suffix or "(no suffix)"
        raise IngestError(
            f"unsupported dataset format {shown!r}; supported formats: "
            f"{_SUPPORTED_DISPLAY}"
        )
    if not source_path.is_file():
        raise IngestError(f"dataset file not found: {source_path}")

    fmt = _FORMAT_BY_SUFFIX[suffix]
    frame = _read_frame(source_path, fmt, csv_sep=csv_sep)
    _restore_duplicate_header(frame, source_path, fmt, csv_sep=csv_sep)

    if frame.shape[0] == 0:
        raise IngestError(
            f"dataset {source_path.name!r} has 0 rows; at least one data row is required"
        )
    if frame.shape[1] == 0:
        raise IngestError(
            f"dataset {source_path.name!r} has 0 columns; at least one column is required"
        )

    warnings: list[str] = []
    _normalize_columns(frame, warnings)

    if target is not None and target not in frame.columns:
        available = ", ".join(str(name) for name in frame.columns)
        raise IngestError(
            f"target column {target!r} is not in the dataset; "
            f"available columns: {available}"
        )

    if frame.shape[0] == 1:
        warnings.append(
            "dataset has only one row; profiles and models built from it will be unreliable"
        )

    return LoadedDataset(
        frame=frame,
        format=fmt,
        source_path=source_path,
        warnings=warnings,
    )
