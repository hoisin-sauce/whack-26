from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from whack26.cli import (
    CALIFORNIA_HOUSING_URL,
    PENGUINS_URL,
    _download,
    build_parser,
    main,
)
from whack26.research.base import ResearchRecord
from whack26.research.pipeline import ResearchQuery, ResearchResult


def test_run_full_flag_parses() -> None:
    args = build_parser().parse_args(
        ["run", "--input", "data.csv", "--full"]
    )
    assert args.full is True
    assert args.source_site is None
    assert args.backend is None


def test_run_backend_flag_parses() -> None:
    args = build_parser().parse_args(
        ["run", "--input", "data.csv", "--backend", "sklearn"]
    )
    assert args.backend == "sklearn"


def test_doctor_mock_cli(capsys) -> None:
    code = main(["doctor", "--mock", "--skip-network"])
    output = capsys.readouterr().out
    assert code == 0
    assert "PASS" in output
    assert "SKIP" in output


def test_classify_mock_writes_result(tmp_path: Path, capsys) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("x,y\n1,2\n3,4\n5,6\n", encoding="utf-8")
    database = tmp_path / "fields.db"
    output = tmp_path / "classify.json"
    code = main(
        [
            "classify",
            "--input",
            str(dataset),
            "--db",
            str(database),
            "--out",
            str(output),
            "--mock",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "path:" in captured.out
    assert database.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset_id"].startswith("ds_")
    assert payload["classification"]["path"]
    assert payload["classification"]["confidence"] >= 0.0


def test_prefetch_mock_writes_research_result(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("x,y\n1,2\n3,4\n5,6\n", encoding="utf-8")
    database = tmp_path / "fields.db"
    output = tmp_path / "prefetch.json"

    record = ResearchRecord(
        id="openalex:W1",
        source="openalex",
        source_id="W1",
        title="Housing prices",
        url="https://example.test/W1",
        retrieved_at="2026-01-01T00:00:00Z",
        query="housing prices",
    )

    def fake_run_research(**kwargs):
        assert kwargs["path_names"]
        return ResearchResult(
            records=[record],
            queries=[
                ResearchQuery(
                    source="openalex",
                    query="housing prices",
                    rationale="test",
                )
            ],
            keywords=["housing"],
            sources_used=["openalex"],
            cache={"hits": 0, "misses": 1, "writes": 1, "hit_rate": 0.0},
        )

    monkeypatch.setattr("whack26.cli.run_research", fake_run_research)
    code = main(
        [
            "prefetch",
            "--input",
            str(dataset),
            "--target",
            "y",
            "--db",
            str(database),
            "--out",
            str(output),
            "--mock",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "records: 1" in captured.out
    assert "openalex=1" in captured.out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset"]["target"] == "y"
    assert payload["classification"]["path"]
    assert payload["research"]["records"][0]["id"] == "openalex:W1"
    assert payload["research"]["queries"][0]["source"] == "openalex"
    assert payload["research"]["cache"]["writes"] == 1


def test_run_command_reports_ingest_failure(capsys) -> None:
    code = main(["run", "--input", "missing.csv", "--mock", "--offline"])
    captured = capsys.readouterr()
    assert code == 1
    assert "run failed" in captured.err


def test_validate_artifact_reports_missing_file(capsys) -> None:
    code = main(["validate-artifact", "runs/demo/artifact.json"])
    captured = capsys.readouterr()
    assert code == 1
    assert "could not read artifact" in captured.err


class FakeDownloadResponse:
    def __init__(self, content: bytes, *, error: Exception | None = None):
        self.content = content
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error


class FakeDownloadClient:
    def __init__(self, response: FakeDownloadResponse):
        self.response = response
        self.urls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def get(self, url: str) -> FakeDownloadResponse:
        self.urls.append(url)
        return self.response


def install_fake_download(
    monkeypatch: pytest.MonkeyPatch, response: FakeDownloadResponse
) -> FakeDownloadClient:
    client = FakeDownloadClient(response)
    monkeypatch.setattr("whack26.cli.httpx.Client", lambda **kwargs: client)
    return client


def test_download_skips_existing_file_without_force(
    tmp_path: Path, capsys
) -> None:
    destination = tmp_path / "data.csv"
    destination.write_text("old", encoding="utf-8")

    code = _download("https://example.test/data.csv", destination, force=False)

    captured = capsys.readouterr()
    assert code == 0
    assert "Already exists" in captured.out
    assert destination.read_text(encoding="utf-8") == "old"


def test_download_writes_the_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    destination = tmp_path / "nested" / "data.csv"
    client = install_fake_download(
        monkeypatch, FakeDownloadResponse(b"a,b\n1,2\n")
    )

    code = _download("https://example.test/data.csv", destination, force=True)

    captured = capsys.readouterr()
    assert code == 0
    assert client.urls == ["https://example.test/data.csv"]
    assert destination.read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert "Saved" in captured.out


def test_download_reports_http_and_empty_body_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    install_fake_download(
        monkeypatch,
        FakeDownloadResponse(b"", error=httpx.HTTPError("network down")),
    )
    http_code = _download(
        "https://example.test/data.csv", tmp_path / "http.csv", force=True
    )
    http_error = capsys.readouterr().err

    install_fake_download(monkeypatch, FakeDownloadResponse(b""))
    empty_code = _download(
        "https://example.test/data.csv", tmp_path / "empty.csv", force=True
    )
    empty_error = capsys.readouterr().err

    assert http_code == 1
    assert "download failed" in http_error
    assert empty_code == 1
    assert "empty body" in empty_error
    assert not (tmp_path / "http.csv").exists()
    assert not (tmp_path / "empty.csv").exists()


def test_fetch_demo_uses_the_requested_dataset_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path, bool]] = []

    def fake_download(url: str, destination: Path, *, force: bool) -> int:
        calls.append((url, destination, force))
        return 0

    monkeypatch.setattr("whack26.cli._download", fake_download)
    destination = tmp_path / "penguins.csv"

    code = main(
        ["fetch-demo", "--penguins", "--force", "--out", str(destination)]
    )

    assert code == 0
    assert calls == [(PENGUINS_URL, destination, True)]


def test_fetch_demo_uses_the_settings_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path, bool]] = []

    def fake_download(url: str, destination: Path, *, force: bool) -> int:
        calls.append((url, destination, force))
        return 0

    monkeypatch.setenv("WHACK26_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("whack26.cli._download", fake_download)

    code = main(["fetch-demo"])

    assert code == 0
    assert calls == [
        (CALIFORNIA_HOUSING_URL, tmp_path / "data" / "california_housing.csv", False)
    ]
