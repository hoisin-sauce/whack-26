from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import eval.field_classification as field_classification
import whack26.config as config
from eval.field_classification import (
    DatasetRun,
    EvalDataset,
    GateResult,
    HumanScore,
    RoundResult,
    build_json,
    render_markdown,
)
from whack26.config import Settings

ENTRIES = (
    EvalDataset(
        slug="alpha",
        name="Alpha Dataset",
        domain="biology / ecology",
        url="https://example.invalid/alpha.csv",
        filename="alpha.csv",
        description="Morphology and species observations.",
        target="species",
        acceptable_roots=("biology", "ecology"),
        forbidden_labels=("alpha dataset",),
        good_path="Biology > Ecology > Species",
    ),
    EvalDataset(
        slug="beta",
        name="Beta Dataset",
        domain="finance",
        url="https://example.invalid/beta.csv",
        filename="beta.csv",
        description=None,
        target=None,
        acceptable_roots=("finance",),
        forbidden_labels=(),
        good_path="Finance > Markets",
    ),
)


def make_settings() -> Settings:
    return Settings(
        llm_provider="deepseek",
        deepseek_model="deepseek-eval-model",
        embed_provider="gemini",
        gemini_embed_model="gemini-eval-embed",
        embed_dims=32,
    )


def make_records() -> list[DatasetRun]:
    return [
        DatasetRun(
            slug="alpha",
            path_names=["Biology", "Ecology", "Species"],
            path_ids=["fld_alpha_1", "fld_alpha_2", "fld_alpha_3"],
            confidence=0.912345678,
            created_fields=["field-alpha", "field-beta"],
            warnings=["alpha warning"],
            elapsed_s=1.2649,
        ),
        DatasetRun(
            slug="beta",
            path_names=["Finance", "Markets"],
            path_ids=["fld_beta_1", "fld_beta_2"],
            confidence=0.8,
            created_fields=[],
            warnings=[],
            elapsed_s=2.0449,
        ),
    ]


def make_round(name: str, *, elapsed_s: float = 10.1236) -> RoundResult:
    return RoundResult(
        name=name,
        elapsed_s=elapsed_s,
        records=make_records(),
        stats={"orphan_fields": 0},
        roots=[("Biology", [1.0, 0.0]), ("Finance", [0.0, 1.0])],
    )


def make_gates() -> list[GateResult]:
    return [
        GateResult(
            "Robustness (crashes)",
            "pass",
            "0 crash(es) across 8 classifications",
        ),
        GateResult(
            "Order independence",
            "fail",
            "1/2 same root after reversal",
            blocking=False,
        ),
    ]


def make_scores() -> dict[str, HumanScore]:
    return {
        "alpha": HumanScore(
            root=4,
            path=5,
            naming=None,
            sibling_distinctness=True,
            notes="clear and concise",
        )
    }


def build_payload(
    *,
    gates: list[GateResult] | None = None,
    repeat: RoundResult | None = None,
    scores: dict[str, HumanScore] | None = None,
) -> dict[str, Any]:
    return build_json(
        generated_at="2026-01-02T03:04:05Z",
        settings=make_settings(),
        commit="abc1234",
        mode="mock",
        entries=ENTRIES,
        cold=make_round("Cold"),
        warm=make_round("Warm", elapsed_s=2.5),
        shuffled=make_round("Shuffled cold", elapsed_s=3.5),
        repeat=repeat,
        gates=make_gates() if gates is None else gates,
        scores=make_scores() if scores is None else scores,
    )


def test_build_json_reports_provider_metadata_and_dataset_registry() -> None:
    settings = make_settings()
    payload = build_payload()

    assert payload["schema_version"] == "1.0.0"
    assert payload["generated_at"] == "2026-01-02T03:04:05Z"
    assert payload["commit"] == "abc1234"
    assert payload["mode"] == "mock"
    assert payload["provider"] == {
        "llm": "deepseek",
        "llm_model": settings.llm_model_name,
        "embeddings": "gemini",
        "embedding_model": settings.embed_model_name,
        "embed_dims": 32,
    }
    assert payload["provider"]["llm_model"] == "deepseek-eval-model"
    assert payload["provider"]["embedding_model"] == "gemini-eval-embed"

    assert payload["datasets"] == [
        {
            "slug": "alpha",
            "name": "Alpha Dataset",
            "domain": "biology / ecology",
            "target": "species",
            "description_supplied": True,
            "acceptable_roots": ["biology", "ecology"],
            "good_path": "Biology > Ecology > Species",
        },
        {
            "slug": "beta",
            "name": "Beta Dataset",
            "domain": "finance",
            "target": None,
            "description_supplied": False,
            "acceptable_roots": ["finance"],
            "good_path": "Finance > Markets",
        },
    ]
    assert payload["status"] == "pass"
    assert payload["human_scores"] == {
        "alpha": {
            "root": 4,
            "path": 5,
            "naming": None,
            "sibling_distinctness": True,
            "notes": "clear and concise",
        }
    }


def test_build_json_round_payloads_include_repeat_and_rounded_metrics() -> None:
    payload = build_payload(repeat=make_round("Repeat cold", elapsed_s=11.5))

    assert set(payload["runs"]) == {"cold", "warm", "shuffled", "repeat"}
    assert payload["runs"]["cold"] == {
        "name": "Cold",
        "elapsed_s": 10.124,
        "stats": {"orphan_fields": 0},
        "roots": ["Biology", "Finance"],
        "records": [
            {
                "slug": "alpha",
                "path": ["Biology", "Ecology", "Species"],
                "field_ids": ["fld_alpha_1", "fld_alpha_2", "fld_alpha_3"],
                "depth": 3,
                "confidence": 0.912346,
                "created_fields": 2,
                "warnings": ["alpha warning"],
                "error": None,
                "elapsed_s": 1.265,
            },
            {
                "slug": "beta",
                "path": ["Finance", "Markets"],
                "field_ids": ["fld_beta_1", "fld_beta_2"],
                "depth": 2,
                "confidence": 0.8,
                "created_fields": 0,
                "warnings": [],
                "error": None,
                "elapsed_s": 2.045,
            },
        ],
    }
    assert payload["runs"]["repeat"]["name"] == "Repeat cold"
    assert payload["runs"]["repeat"]["elapsed_s"] == 11.5


def test_build_json_status_tracks_blocking_gates() -> None:
    pending = build_payload(
        gates=[GateResult("Human scores", "pending", "0/2 datasets scored")]
    )
    assert pending["status"] == "pending"

    failed = build_payload(
        gates=[
            GateResult("Human scores", "pending", "0/2 datasets scored"),
            GateResult("Robustness (crashes)", "fail", "1 crash"),
        ]
    )
    assert failed["status"] == "fail"


def test_render_markdown_includes_provider_runs_gates_and_human_scores(
    tmp_path: Path,
) -> None:
    markdown = render_markdown(
        generated_at="2026-01-02T03:04:05Z",
        settings=make_settings(),
        commit="abc1234",
        mode="mock",
        entries=ENTRIES,
        cold=make_round("Cold", elapsed_s=12.34),
        warm=make_round("Warm", elapsed_s=2.5),
        shuffled=make_round("Shuffled cold", elapsed_s=3.5),
        repeat=None,
        gates=make_gates(),
        scores={
            "alpha": HumanScore(
                root=4,
                path=5,
                naming=None,
                sibling_distinctness=True,
                notes="clear | concise",
            )
        },
        json_path=tmp_path / "field-classification.json",
        assume_automated_naming=True,
    )

    assert "LLM: `deepseek` / `deepseek-eval-model`" in markdown
    assert (
        "Embeddings: `gemini` / `gemini-eval-embed` at 32 dimensions" in markdown
    )
    assert "Datasets: 2" in markdown
    assert "Overall status: **PASS**" in markdown
    assert (
        "| Robustness (crashes) | pass | 0 crash(es) across 8 classifications |"
        in markdown
    )
    assert "| Order independence (diagnostic) | fail |" in markdown
    assert "### Cold (12.3s)" in markdown
    assert (
        "| alpha | Biology | Biology > Ecology > Species | 3 | 0.912 | 2 | 1.3s |"
        in markdown
    )
    assert "### Warm (2.5s)" in markdown
    assert "## Human scores" in markdown
    assert "| alpha | 4 | 5 | auto | yes | clear \\| concise |" in markdown
    assert "| beta | - | - | - | - | not scored |" in markdown
    assert (
        "`auto` means the automated naming gate was trusted for that dataset."
        in markdown
    )
    assert "| Beta Dataset | finance | - | no | `beta.csv` |" in markdown
    assert "Machine-readable results:" in markdown


def test_render_markdown_handles_missing_human_scores(tmp_path: Path) -> None:
    markdown = render_markdown(
        generated_at="2026-01-02T03:04:05Z",
        settings=make_settings(),
        commit="abc1234",
        mode="mock",
        entries=ENTRIES,
        cold=make_round("Cold"),
        warm=make_round("Warm", elapsed_s=2.5),
        shuffled=make_round("Shuffled cold", elapsed_s=3.5),
        repeat=None,
        gates=make_gates(),
        scores={},
        json_path=tmp_path / "field-classification.json",
    )

    assert "No human scores supplied." in markdown


def test_main_live_preflight_requires_deepseek_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def skip_dotenv(*args: object, **kwargs: object) -> None:
        return None

    def fail_load_inputs(*args: object, **kwargs: object) -> None:
        raise AssertionError("preflight should fail before dataset loading")

    monkeypatch.setattr(config, "load_dotenv", skip_dotenv)
    monkeypatch.setenv("WHACK26_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("WHACK26_EMBED_PROVIDER", "local")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(field_classification, "load_inputs", fail_load_inputs)

    exit_code = field_classification.main(
        [
            "--datasets",
            "penguins",
            "--work-dir",
            str(tmp_path / "work"),
            "--output",
            str(tmp_path / "report.md"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "DEEPSEEK_API_KEY" in captured.err
    assert "required for a live run" in captured.err
    assert not (tmp_path / "work").exists()
