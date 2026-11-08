from __future__ import annotations

import types
from pathlib import Path

import pytest

import whack26.doctor as doctor
from whack26.config import Settings
from whack26.doctor import exit_code, format_checks, run_doctor


def test_mock_doctor_passes_without_network(tmp_path: Path) -> None:
    settings = Settings.from_env({}, mock=True).with_overrides(cache_dir=tmp_path / "cache")
    checks = run_doctor(settings, network=False)
    assert exit_code(checks) == 0
    statuses = {check.name: check.status for check in checks}
    assert statuses["Python"] == "PASS"
    assert statuses["Packages"] == "PASS"
    assert statuses["Pitfalls corpus"] == "PASS"
    assert statuses["Gemini LLM"] == "SKIP"
    assert statuses["OpenAlex"] == "SKIP"
    text = format_checks(checks)
    assert "STATUS" in text
    assert "PASS" in text


def test_missing_corpus_fails(tmp_path: Path) -> None:
    settings = Settings.from_env({}, mock=True).with_overrides(
        cache_dir=tmp_path / "cache",
        corpus_path=tmp_path / "missing.md",
    )
    checks = run_doctor(settings, network=False)
    corpus = next(check for check in checks if check.name == "Pitfalls corpus")
    assert corpus.status == "FAIL"
    assert exit_code(checks) == 1


def test_missing_gemini_key_fails_in_live_mode(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {
            "WHACK26_LLM_PROVIDER": "gemini",
            "WHACK26_EMBED_PROVIDER": "gemini",
        }
    ).with_overrides(cache_dir=tmp_path / "cache")
    checks = run_doctor(settings, network=False)
    gemini = next(check for check in checks if check.name == "Gemini key")
    assert gemini.status == "FAIL"
    assert gemini.hint


@pytest.mark.parametrize(
    ("version", "status"),
    [
        ((3, 12, 4), "PASS"),
        ((3, 13, 2), "PASS"),
        ((3, 11, 9), "FAIL"),
        ((3, 14, 0), "FAIL"),
    ],
)
def test_python_check_accepts_supported_range(
    monkeypatch: pytest.MonkeyPatch, version: tuple[int, ...], status: str
) -> None:
    monkeypatch.setattr(doctor.sys, "version_info", version)

    check = doctor._check_python()

    assert check.status == status
    if status == "FAIL":
        assert check.hint
        assert "3.13" in check.hint


def test_package_version_lookup_does_not_call_metadata_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace(__version__="9.9.9")
    fallback_calls: list[str] = []
    monkeypatch.setattr(doctor.importlib, "import_module", lambda name: fake_module)
    monkeypatch.setattr(
        doctor, "_version_of", lambda name: fallback_calls.append(name) or "0.0"
    )

    check = doctor._check_packages()

    assert check.status == "PASS"
    assert "9.9.9" in check.detail
    assert fallback_calls == []


def test_package_version_metadata_fallback_is_used_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace()
    monkeypatch.setattr(doctor.importlib, "import_module", lambda name: fake_module)
    monkeypatch.setattr(doctor, "_version_of", lambda name: "0.0.0")

    check = doctor._check_packages()

    assert check.status == "PASS"
    assert "0.0.0" in check.detail


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_embedding_vectors_fail(
    monkeypatch: pytest.MonkeyPatch, bad_value: float
) -> None:
    class _FakeEmbeddingClient:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[bad_value, bad_value, bad_value]]

        def close(self) -> None:
            pass

    settings = Settings.from_env({}, mock=True).with_overrides(
        embed_provider="gemini", embed_dims=3
    )
    monkeypatch.setattr(
        doctor, "build_embedding_client", lambda _: _FakeEmbeddingClient()
    )

    check = doctor._check_gemini_embeddings(settings)

    assert check.status == "FAIL"
    assert "finite" in check.detail.lower()


def test_pyqt6_is_declared_in_optional_ui_requirements() -> None:
    root = Path(__file__).resolve().parents[1]
    core = (root / "requirements.txt").read_text(encoding="utf-8")
    ui = (root / "requirements-ui.txt").read_text(encoding="utf-8")

    core_requirements = [
        line.strip()
        for line in core.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    ui_requirements = [
        line.strip()
        for line in ui.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert any("PyQt6" in line for line in ui_requirements)
    assert not any("PyQt6" in line for line in core_requirements)
    assert "requirements-ui.txt" in core
