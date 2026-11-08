from __future__ import annotations

from pathlib import Path

import pytest

from whack26.config import Settings, load_dotenv


def test_settings_resolve_env_values(tmp_path: Path) -> None:
    env = {
        "WHACK26_BASE_DIR": str(tmp_path),
        "WHACK26_LLM_PROVIDER": "gemini",
        "WHACK26_EMBED_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-key",
        "GEMINI_MODEL": "test-model",
        "GEMINI_EMBED_MODEL": "test-embed",
        "EMBED_DIMS": "32",
        "OPENALEX_API_KEY": "oa-key",
        "HF_TOKEN": "hf-token",
        "WHACK26_OFFLINE": "1",
        "WHACK26_TOP_K": "7",
        "WHACK26_DUPLICATE_COSINE": "0.8",
        "WHACK26_MAX_DEPTH": "3",
        "WHACK26_CACHE_DIR": "local-cache",
    }
    settings = Settings.from_env(env)
    assert settings.llm_provider == "gemini"
    assert settings.gemini_model == "test-model"
    assert settings.embed_dims == 32
    assert settings.offline is True
    assert settings.top_k == 7
    assert settings.duplicate_cosine == 0.8
    assert settings.max_depth == 3
    assert settings.cache_dir == tmp_path / "local-cache"
    assert settings.fields_db == tmp_path / "local-cache" / "fields.db"


def test_mock_flag_sets_both_providers() -> None:
    settings = Settings.from_env({}, mock=True)
    assert settings.llm_provider == "mock"
    assert settings.embed_provider == "mock"
    assert settings.mock is True


def test_invalid_max_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        Settings.from_env({"WHACK26_MAX_DEPTH": "5"})


def test_invalid_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="WHACK26_LLM_PROVIDER"):
        Settings.from_env({"WHACK26_LLM_PROVIDER": "unknown"})


@pytest.mark.parametrize(
    ("desc", "centroid", "message"),
    [
        ("-0.1", "0.5", "desc_weight"),
        ("0.5", "-0.1", "centroid_weight"),
        ("1.2", "0.5", "desc_weight"),
        ("0.5", "1.2", "centroid_weight"),
        ("0.6", "0.6", "sum to 1"),
    ],
)
def test_invalid_similarity_weights_are_rejected(
    desc: str, centroid: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Settings.from_env(
            {
                "WHACK26_DESC_WEIGHT": desc,
                "WHACK26_CENTROID_WEIGHT": centroid,
            }
        )


def test_valid_similarity_weights_are_accepted() -> None:
    settings = Settings.from_env(
        {
            "WHACK26_DESC_WEIGHT": "0.7",
            "WHACK26_CENTROID_WEIGHT": "0.3",
        }
    )
    assert settings.desc_weight == pytest.approx(0.7)
    assert settings.centroid_weight == pytest.approx(0.3)


def test_scrape_settings_parse_from_env() -> None:
    settings = Settings.from_env(
        {
            "WHACK26_SCRAPE_ENABLED": "1",
            "WHACK26_SCRAPE_MAX_DEPTH": "3",
            "WHACK26_SCRAPE_MAX_RESOURCES": "42",
            "WHACK26_SCRAPE_MAX_LINKS_PER_RESOURCE": "7",
            "WHACK26_SCRAPE_MAX_FETCHES": "9",
            "WHACK26_SCRAPE_SUMMARY_THRESHOLD": "0.7",
            "WHACK26_SCRAPE_LINK_THRESHOLD": "0.8",
            "WHACK26_SCRAPE_CREDIBILITY_THRESHOLD": "0.6",
            "WHACK26_SCRAPE_TTL_DAYS": "3",
        }
    )
    assert settings.scrape_enabled is True
    assert settings.scrape_max_depth == 3
    assert settings.scrape_max_resources == 42
    assert settings.scrape_max_links_per_resource == 7
    assert settings.scrape_max_fetches == 9
    assert settings.scrape_summary_threshold == pytest.approx(0.7)
    assert settings.scrape_link_threshold == pytest.approx(0.8)
    assert settings.scrape_credibility_threshold == pytest.approx(0.6)
    assert settings.scrape_ttl_days == 3
    assert settings.scrape_ttl_seconds == 3 * 86400


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scrape_summary_threshold": 1.5}, "scrape_summary_threshold"),
        ({"scrape_link_threshold": -0.1}, "scrape_link_threshold"),
        ({"scrape_max_resources": 0}, "scrape_max_resources"),
        ({"scrape_ttl_days": 0}, "scrape_ttl_days"),
    ],
)
def test_invalid_scrape_settings_are_rejected(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Settings(**kwargs)


def test_scrape_fetch_settings_parse_from_env() -> None:
    settings = Settings.from_env(
        {
            "WHACK26_SCRAPE_FETCH_ENABLED": "1",
            "WHACK26_SCRAPE_ALLOWED_DOMAINS": "example.com, arxiv.org",
            "WHACK26_SCRAPE_BLOCKED_DOMAINS": "spam.test",
            "WHACK26_SCRAPE_RATE_LIMIT_SECONDS": "0.5",
            "WHACK26_SCRAPE_MAX_BYTES": "1000",
            "WHACK26_SCRAPE_MAX_TEXT_CHARS": "200",
            "WHACK26_SCRAPE_FETCH_TIMEOUT": "5",
            "WHACK26_SCRAPE_MAX_REDIRECTS": "1",
            "WHACK26_SCRAPE_ALLOW_PRIVATE_NETWORKS": "1",
        }
    )
    assert settings.scrape_fetch_enabled is True
    assert settings.scrape_allowed_domains == ("example.com", "arxiv.org")
    assert settings.scrape_blocked_domains == ("spam.test",)
    assert settings.scrape_rate_limit_seconds == pytest.approx(0.5)
    assert settings.scrape_max_bytes == 1000
    assert settings.scrape_max_text_chars == 200
    assert settings.scrape_fetch_timeout == pytest.approx(5.0)
    assert settings.scrape_max_redirects == 1
    assert settings.scrape_allow_private_networks is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scrape_rate_limit_seconds": -1}, "scrape_rate_limit_seconds"),
        ({"scrape_max_bytes": 0}, "scrape_max_bytes"),
        ({"scrape_max_text_chars": 0}, "scrape_max_text_chars"),
        ({"scrape_fetch_timeout": 0}, "scrape_fetch_timeout"),
        ({"scrape_max_redirects": -1}, "scrape_max_redirects"),
    ],
)
def test_invalid_scrape_fetch_settings_are_rejected(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Settings(**kwargs)


def test_load_dotenv_does_not_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WHACK26_TEST_EXISTING", "keep-me")
    monkeypatch.delenv("WHACK26_TEST_NEW", raising=False)
    monkeypatch.delenv("WHACK26_TEST_QUOTED", raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "WHACK26_TEST_EXISTING=replace-me\n"
        "WHACK26_TEST_NEW=new-value\n"
        'WHACK26_TEST_QUOTED="quoted value"\n',
        encoding="utf-8",
    )
    load_dotenv(path)
    import os

    assert os.environ["WHACK26_TEST_EXISTING"] == "keep-me"
    assert os.environ["WHACK26_TEST_NEW"] == "new-value"
    assert os.environ["WHACK26_TEST_QUOTED"] == "quoted value"
