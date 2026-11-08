from __future__ import annotations

from types import SimpleNamespace

from whack26.cli import build_parser, main


def _fake_outcome(tmp_path):
    artifact = SimpleNamespace(
        classification=SimpleNamespace(path=[SimpleNamespace(name="root")]),
        research=SimpleNamespace(records=[]),
        factors=[],
        hypotheses=[],
    )
    return SimpleNamespace(
        run_dir=tmp_path,
        artifact=artifact,
        artifact_path=tmp_path / "artifact.json",
        preview_path=tmp_path / "preview.md",
        validations={},
        generation_validations={},
        scrape=None,
        model_search=None,
        variants=None,
        final_judge=None,
        field_pitfalls=None,
    )


def test_run_field_pitfalls_flag_parses() -> None:
    args = build_parser().parse_args(
        ["run", "--input", "x.csv", "--field-pitfalls"]
    )

    assert args.field_pitfalls is True


def test_run_field_pitfalls_defaults_off() -> None:
    args = build_parser().parse_args(["run", "--input", "x.csv"])

    assert args.field_pitfalls is False


def test_cli_run_passes_field_pitfalls_flag(tmp_path, monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        "whack26.cli.run_pipeline",
        lambda **kwargs: calls.append(kwargs) or _fake_outcome(tmp_path),
    )

    code = main(["run", "--input", "x.csv", "--field-pitfalls"])

    assert code == 0
    assert calls[0]["field_pitfalls"] is True


def test_cli_run_full_enables_field_pitfalls(tmp_path, monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        "whack26.cli.run_pipeline",
        lambda **kwargs: calls.append(kwargs) or _fake_outcome(tmp_path),
    )

    code = main(["run", "--input", "x.csv", "--full"])

    assert code == 0
    assert calls[0]["field_pitfalls"] is True
