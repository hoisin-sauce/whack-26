from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from whack26.io import IngestError, read_dataset


def _small_frame() -> pd.DataFrame:
    return pd.DataFrame({"value": [1, 2, 3], "category": ["a", "b", "a"]})


def test_read_csv_with_custom_separator(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("value;category\n1;a\n2;b\n", encoding="utf-8")

    loaded = read_dataset(path, csv_sep=";")

    assert loaded.format == "csv"
    assert loaded.source_path == path
    assert loaded.warnings == []
    assert list(loaded.frame.columns) == ["value", "category"]
    assert loaded.frame["value"].tolist() == [1, 2]
    assert loaded.frame.index.tolist() == [0, 1]


@pytest.mark.parametrize("suffix", [".tsv", ".tab"])
def test_read_tsv_suffixes(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"data{suffix}"
    _small_frame().to_csv(path, sep="\t", index=False)

    loaded = read_dataset(path)

    assert loaded.format == "tsv"
    assert loaded.frame.shape == (3, 2)
    assert loaded.frame["category"].tolist() == ["a", "b", "a"]


def test_read_json_records(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    _small_frame().to_json(path, orient="records")

    loaded = read_dataset(path)

    assert loaded.format == "json"
    assert loaded.frame.shape == (3, 2)
    assert list(loaded.frame.columns) == ["value", "category"]


@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson"])
def test_read_json_lines(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"data{suffix}"
    _small_frame().to_json(path, orient="records", lines=True)

    loaded = read_dataset(path)

    assert loaded.format == "json"
    assert loaded.frame.shape == (3, 2)


def test_json_extension_falls_back_to_lines(tmp_path: Path) -> None:
    path = tmp_path / "lines.json"
    path.write_text(
        _small_frame().to_json(orient="records", lines=True), encoding="utf-8"
    )

    loaded = read_dataset(path)

    assert loaded.format == "json"
    assert loaded.frame.shape == (3, 2)


def test_read_parquet(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    _small_frame().to_parquet(path, index=False)

    loaded = read_dataset(path)

    assert loaded.format == "parquet"
    assert loaded.frame.shape == (3, 2)
    assert loaded.frame["value"].tolist() == [1, 2, 3]


def test_read_excel_with_openpyxl(tmp_path: Path) -> None:
    path = tmp_path / "data.xlsx"
    _small_frame().to_excel(path, index=False, engine="openpyxl")

    loaded = read_dataset(path)

    assert loaded.format == "excel"
    assert loaded.frame.shape == (3, 2)


def test_pickle_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data.pkl"
    _small_frame().to_pickle(path)

    with pytest.raises(IngestError, match="[Pp]ickle"):
        read_dataset(path)


def test_unknown_suffix_lists_supported_formats(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("value,category\n1,a\n", encoding="utf-8")

    with pytest.raises(IngestError, match="supported formats"):
        read_dataset(path)


def test_duplicate_headers_are_renamed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.csv"
    path.write_text("a,a,b\n1,2,3\n", encoding="utf-8")

    loaded = read_dataset(path)

    assert list(loaded.frame.columns) == ["a", "a__2", "b"]
    assert loaded.frame["a"].tolist() == [1]
    assert loaded.frame["a__2"].tolist() == [2]
    assert any("duplicate" in warning.lower() for warning in loaded.warnings)


def test_ragged_duplicate_header_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ragged.csv"
    path.write_text("a,a,b\n1,2,3,4\n", encoding="utf-8")

    with pytest.raises(IngestError, match="ragged") as excinfo:
        read_dataset(path)

    message = str(excinfo.value)
    assert "3 field" in message
    assert "4" in message


def test_duplicate_header_after_blank_lines_is_restored(tmp_path: Path) -> None:
    path = tmp_path / "blank_prefix.csv"
    path.write_text("\n\na,a,b\n1,2,3\n", encoding="utf-8")

    loaded = read_dataset(path)

    assert list(loaded.frame.columns) == ["a", "a__2", "b"]
    assert loaded.frame["a"].tolist() == [1]
    assert loaded.frame["a__2"].tolist() == [2]
    assert any("duplicate" in warning.lower() for warning in loaded.warnings)


def test_valid_json_with_wrong_shape_surfaces_schema_error(tmp_path: Path) -> None:
    path = tmp_path / "scalars.json"
    path.write_text('{"a": 1, "b": 2}', encoding="utf-8")

    with pytest.raises(IngestError, match="valid as records/array") as excinfo:
        read_dataset(path)

    # The old behavior retried as JSON lines and returned a bogus one-row frame.
    assert "scalar" in str(excinfo.value).lower()


def test_zero_row_dataset_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("value,category\n", encoding="utf-8")

    with pytest.raises(IngestError, match="0 rows"):
        read_dataset(path)


def test_zero_column_dataset_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    pd.DataFrame(index=[0, 1]).to_parquet(path)

    with pytest.raises(IngestError, match="0 columns"):
        read_dataset(path)


def test_missing_target_lists_available_columns(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    _small_frame().to_csv(path, index=False)

    with pytest.raises(IngestError, match="available columns") as excinfo:
        read_dataset(path, target="does_not_exist")

    message = str(excinfo.value)
    assert "value" in message
    assert "category" in message


def test_one_row_dataset_gets_a_warning(tmp_path: Path) -> None:
    path = tmp_path / "one_row.csv"
    pd.DataFrame({"value": [1], "category": ["a"]}).to_csv(path, index=False)

    loaded = read_dataset(path)

    assert loaded.frame.shape == (1, 2)
    assert any("one row" in warning.lower() for warning in loaded.warnings)


def test_read_failure_is_wrapped_with_format(tmp_path: Path) -> None:
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"not a parquet file")

    with pytest.raises(IngestError, match="parquet"):
        read_dataset(path)
