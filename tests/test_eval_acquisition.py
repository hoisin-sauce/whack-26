"""Tests for dataset acquisition and input loading in the eval harness."""

from __future__ import annotations

import zipfile
from pathlib import Path

import httpx
import pytest

from eval import field_classification as eval_module
from eval.field_classification import EvalDataset, _download, _resolve_dataset, load_inputs


def make_entry(tmp_path: Path, **overrides) -> EvalDataset:
    values = {
        "slug": "demo",
        "name": "Demo Data",
        "domain": "demo",
        "url": "https://example.test/demo.csv",
        "filename": "demo.csv",
        "description": "A synthetic eval dataset.",
        "target": "y",
        "acceptable_roots": ("demo",),
        "forbidden_labels": ("demo dataset",),
        "good_path": "Demo > Data",
    }
    values.update(overrides)
    return EvalDataset(**values)


class FakeResponse:
    def __init__(self, chunks: list[bytes], *, error: Exception | None = None):
        self.chunks = chunks
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    def iter_bytes(self, _size: int):
        yield from self.chunks


class FakeStream:
    def __init__(self, response: FakeResponse):
        self.response = response

    def __enter__(self) -> FakeResponse:
        return self.response

    def __exit__(self, *exc_info) -> None:
        return None


def test_download_writes_the_file_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "nested" / "data.csv"
    monkeypatch.setattr(
        eval_module.httpx,
        "stream",
        lambda *args, **kwargs: FakeStream(FakeResponse([b"a,b\n", b"1,2\n"])),
    )

    _download("https://example.test/data.csv", destination)

    assert destination.read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert not destination.with_name("data.csv.part").exists()


def test_download_retries_and_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "data.csv"
    calls = {"count": 0}

    def stream(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeStream(
                FakeResponse([], error=httpx.HTTPError("transient failure"))
            )
        return FakeStream(FakeResponse([b"ok"]))

    monkeypatch.setattr(eval_module.httpx, "stream", stream)
    monkeypatch.setattr(eval_module.time, "sleep", lambda _seconds: None)

    _download("https://example.test/data.csv", destination, attempts=2)

    assert calls["count"] == 2
    assert destination.read_text(encoding="utf-8") == "ok"


def test_download_raises_after_exhausting_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "data.csv"
    monkeypatch.setattr(
        eval_module.httpx,
        "stream",
        lambda *args, **kwargs: FakeStream(
            FakeResponse([], error=httpx.HTTPError("permanent failure"))
        ),
    )
    monkeypatch.setattr(eval_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="could not download"):
        _download("https://example.test/data.csv", destination, attempts=2)

    assert not destination.exists()
    assert not destination.with_name("data.csv.part").exists()


def test_resolve_dataset_prefers_a_local_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "data" / "local.csv"
    local.parent.mkdir(parents=True)
    local.write_text("y\n1\n", encoding="utf-8")
    monkeypatch.setattr(eval_module, "REPO_ROOT", tmp_path)
    entry = make_entry(tmp_path, local_path="data/local.csv")

    resolved = _resolve_dataset(entry, tmp_path / "cache")

    assert resolved == local


def test_resolve_dataset_extracts_an_archive_member(tmp_path: Path) -> None:
    entry = make_entry(tmp_path, archive_member="data.csv")
    archive_path = tmp_path / "cache" / entry.slug / entry.filename
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/data.csv", "y\n1\n")

    resolved = _resolve_dataset(entry, tmp_path / "cache")

    assert resolved == tmp_path / "cache" / entry.slug / "data.csv"
    assert resolved.read_text(encoding="utf-8") == "y\n1\n"


def test_resolve_dataset_rejects_a_missing_archive_member(
    tmp_path: Path,
) -> None:
    entry = make_entry(tmp_path, archive_member="missing.csv")
    archive_path = tmp_path / "cache" / entry.slug / entry.filename
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("other.csv", "y\n1\n")

    with pytest.raises(RuntimeError, match="does not contain missing.csv"):
        _resolve_dataset(entry, tmp_path / "cache")


def test_resolve_dataset_downloads_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = make_entry(tmp_path)
    downloaded: list[tuple[str, Path]] = []

    def fake_download(url: str, destination: Path, **kwargs) -> None:
        downloaded.append((url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("y\n1\n", encoding="utf-8")

    monkeypatch.setattr(eval_module, "_download", fake_download)

    resolved = _resolve_dataset(entry, tmp_path / "cache")

    assert resolved == tmp_path / "cache" / entry.slug / entry.filename
    assert downloaded == [(entry.url, resolved)]


def test_load_inputs_profiles_each_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = make_entry(tmp_path)
    data_dir = tmp_path / "cache"
    dataset_path = data_dir / entry.slug / entry.filename
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("x,y\n1,2\n2,4\n3,6\n", encoding="utf-8")

    loaded = load_inputs([entry], data_dir)

    assert len(loaded) == 1
    assert loaded[0].entry is entry
    assert loaded[0].profile.rows == 3
    assert loaded[0].profile.target is not None
    assert loaded[0].profile.target.name == "y"
    assert list(loaded[0].frame.columns) == ["x", "y"]
