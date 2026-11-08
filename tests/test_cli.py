from __future__ import annotations

import json
from pathlib import Path

from whack26.cli import main
from whack26.research.base import ResearchRecord
from whack26.research.pipeline import ResearchQuery, ResearchResult


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
