from __future__ import annotations

import pytest

from whack26.config import Settings


def test_final_judge_settings_parse_from_env() -> None:
    settings = Settings.from_env(
        {
            "WHACK26_FINAL_JUDGE_ENABLED": "yes",
            "WHACK26_FINAL_JUDGE_MODEL": "judge-model",
            "WHACK26_FINAL_JUDGE_DA_MODEL": "da-model",
            "WHACK26_FINAL_JUDGE_MAX_WORKERS": "5",
            "WHACK26_FINAL_JUDGE_MAX_TOKENS": "4096",
            "WHACK26_FINAL_JUDGE_MIN_EVALUATED_FRACTION": "0.75",
        }
    )

    assert settings.final_judge_enabled is True
    assert settings.final_judge_model == "judge-model"
    assert settings.final_judge_da_model == "da-model"
    assert settings.final_judge_max_workers == 5
    assert settings.final_judge_max_tokens == 4096
    assert settings.final_judge_min_evaluated_fraction == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("WHACK26_FINAL_JUDGE_MAX_WORKERS", "0", "final_judge_max_workers"),
        ("WHACK26_FINAL_JUDGE_MAX_WORKERS", "-2", "final_judge_max_workers"),
        ("WHACK26_FINAL_JUDGE_MAX_TOKENS", "0", "final_judge_max_tokens"),
        ("WHACK26_FINAL_JUDGE_MAX_TOKENS", "-100", "final_judge_max_tokens"),
        (
            "WHACK26_FINAL_JUDGE_MIN_EVALUATED_FRACTION",
            "-0.01",
            "final_judge_min_evaluated_fraction",
        ),
        (
            "WHACK26_FINAL_JUDGE_MIN_EVALUATED_FRACTION",
            "1.01",
            "final_judge_min_evaluated_fraction",
        ),
    ],
)
def test_invalid_final_judge_settings_are_rejected(
    key: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Settings.from_env({key: value})


@pytest.mark.parametrize("value", ["0", "0.0", "1", "1.0"])
def test_final_judge_fraction_bounds_are_inclusive(value: str) -> None:
    settings = Settings.from_env(
        {"WHACK26_FINAL_JUDGE_MIN_EVALUATED_FRACTION": value}
    )

    assert settings.final_judge_min_evaluated_fraction == pytest.approx(
        float(value)
    )


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_boolean_flags_accept_truthy_values(value: str) -> None:
    settings = Settings.from_env(
        {
            "WHACK26_HIERARCHY_PLACEMENT_ENABLED": value,
            "WHACK26_RESEARCH_METRICS_ENABLED": value,
        }
    )

    assert settings.hierarchy_placement_enabled is True
    assert settings.research_metrics_enabled is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off"])
def test_boolean_flags_accept_falsey_values(value: str) -> None:
    settings = Settings.from_env(
        {
            "WHACK26_HIERARCHY_PLACEMENT_ENABLED": value,
            "WHACK26_RESEARCH_METRICS_ENABLED": value,
        }
    )

    assert settings.hierarchy_placement_enabled is False
    assert settings.research_metrics_enabled is False


def test_extended_settings_defaults() -> None:
    settings = Settings.from_env({})

    assert settings.final_judge_enabled is False
    assert settings.final_judge_model is None
    assert settings.final_judge_da_model is None
    assert settings.final_judge_max_workers == 3
    assert settings.final_judge_max_tokens == 2000
    assert settings.final_judge_min_evaluated_fraction == pytest.approx(0.5)
    assert settings.hierarchy_placement_enabled is True
    assert settings.research_metrics_enabled is False
