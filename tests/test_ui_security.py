"""Security and robustness checks for the optional dashboard (UI-96 to UI-99).

The dashboard is the last place untrusted, model-generated text reaches a rich
text renderer.  These tests pin two properties:

* reports are inert: no script tags, event-handler attributes, or external
  resource tags appear as markup, and escaped text is shown literally;
* run directories survive ordinary filesystem hazards such as spaces and
  non-ASCII characters.

UI-99 covers credential redaction.  The backend has ``redact_secrets`` but the
UI has no redaction seam, so that case is xfailed rather than inventing a
production behavior in the test.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")

import UI_frontEnd  # noqa: E402
from ui_fixtures import (  # noqa: E402
    make_judge_run_dir,
    make_report,
    write_fragment,
)
from whack26.final_judge.types import JudgeFinding  # noqa: E402

pytestmark = pytest.mark.ui

SCRIPT = "<script>window.__ui96_pwned = true; alert('x')</script>"
EVENT_HANDLER = '<p onclick="window.__ui96_pwned = true">event handler</p>'
EXTERNAL_URL = "https://evil.example/track?token=ui96"

_ACTIVE_TAG_RE = re.compile(
    r"<\s*(?:script|iframe|object|embed|link|form)\b",
    re.IGNORECASE,
)
_EVENT_ATTRIBUTE_RE = re.compile(r"<[^>]*\son[a-z]+\s*=", re.IGNORECASE)
_JAVASCRIPT_URL_RE = re.compile(
    r"(?:href|src)\s*=\s*[\"']?\s*javascript:",
    re.IGNORECASE,
)


@pytest.fixture
def dashboard(qtbot):
    widget = UI_frontEnd.QuantDashboard()
    qtbot.addWidget(widget)
    return widget


def _assert_no_active_markup(html: str) -> None:
    """Fail when the rendered document contains active HTML constructs."""
    assert not _ACTIVE_TAG_RE.search(html), "active HTML tag survived rendering"
    assert not _EVENT_ATTRIBUTE_RE.search(html), "event handler survived rendering"
    assert not _JAVASCRIPT_URL_RE.search(html), "javascript: URL survived rendering"


def test_ui_96_generated_report_is_escaped_and_inert(dashboard, tmp_path: Path) -> None:
    """Model text containing markup must render as literal text."""
    report = make_report(
        model_summary={
            "target": SCRIPT,
            "selected_family": EVENT_HANDLER,
            "selected_features": [EXTERNAL_URL, SCRIPT],
            "candidate_count": 3,
        },
        concerns=[
            JudgeFinding(
                critic="overfitting_skeptic",
                severity="high",
                finding=SCRIPT,
                evidence=EVENT_HANDLER,
                suggested_fix=EXTERNAL_URL,
                confidence="medium",
                status="established",
            )
        ],
        missing_evidence=[EVENT_HANDLER],
        note=SCRIPT,
    )
    run_dir = make_judge_run_dir(tmp_path, report=report)

    workspace = dashboard.load_final_judge(run_dir)

    text = workspace.summary_output.toPlainText()
    assert SCRIPT in text
    assert EVENT_HANDLER in text
    assert EXTERNAL_URL in text

    rendered = workspace.summary_output.toHtml()
    _assert_no_active_markup(rendered)
    lowered = rendered.lower()
    assert "<img" not in lowered
    assert "<a " not in lowered


def test_ui_96_raw_report_cannot_inject_active_markup(dashboard) -> None:
    """Qt must not turn hostile report HTML into executable or fetching markup."""
    workspace = dashboard.current_workspace()
    hostile = (
        SCRIPT
        + EVENT_HANDLER
        + f'<iframe src="{EXTERNAL_URL}"></iframe>'
        + f'<object data="{EXTERNAL_URL}"></object>'
        + f'<embed src="{EXTERNAL_URL}">'
        + f'<link rel="stylesheet" href="{EXTERNAL_URL}">'
        + f'<form action="{EXTERNAL_URL}"><input name="token"></form>'
        + f'<img src="{EXTERNAL_URL}" onerror="window.__ui96_pwned = true">'
    )

    workspace.load_summary_html(hostile)

    _assert_no_active_markup(workspace.summary_output.toHtml())


def test_ui_97_run_dir_with_spaces_and_non_ascii_loads(
    dashboard, tmp_path: Path
) -> None:
    """Filesystem paths are not required to be ASCII or space-free."""
    run_dir = tmp_path / "run dir with spaces" / "rün_数据_✓"
    write_fragment(
        run_dir,
        "<h1>Final model judgement</h1>"
        "<p>Loaded from a path with spaces and non-ASCII characters.</p>",
    )

    workspace = dashboard.load_final_judge(run_dir)

    text = workspace.summary_output.toPlainText()
    assert "Final model judgement" in text
    assert "Loaded from a path with spaces and non-ASCII characters." in text


def test_ui_98_repeated_loads_show_only_the_latest_report(
    dashboard, tmp_path: Path
) -> None:
    """Reloading replaces the document instead of stacking stale reports."""
    workspace = dashboard.current_workspace()
    document = workspace.summary_output.document()
    tab_count = dashboard.tabs.count()
    labels = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]

    for index, label in enumerate(labels):
        run_dir = tmp_path / f"run-{index}"
        write_fragment(
            run_dir,
            f"<h1>Report {label}</h1><p>SENTINEL-{label}</p>",
        )
        dashboard.load_final_judge(run_dir)

    text = workspace.summary_output.toPlainText()
    assert "SENTINEL-DELTA" in text
    for label in labels[:-1]:
        assert f"SENTINEL-{label}" not in text

    # Reloading must not create sibling widgets or replace the document under
    # the view; doing so would leak the old document and break scroll state.
    assert dashboard.tabs.count() == tab_count
    assert workspace.summary_output.document() is document
    assert len(workspace.findChildren(UI_frontEnd.HtmlSummaryView)) == 1


CREDENTIAL = "sk-live-UI99-0123456789abcdef"
REDACTION_XFAIL_REASON = (
    "UI-99 xfail: UI_frontEnd has no redaction seam. whack26.common."
    "redact_secrets is used by backend modules only; DatasetWorkspace and "
    "HtmlSummaryView render report text and append log lines verbatim, so "
    "credentials reach the UI unchanged."
)


@pytest.mark.xfail(strict=False, reason=REDACTION_XFAIL_REASON, raises=AssertionError)
@pytest.mark.parametrize("channel", ["report", "log"])
def test_ui_99_credentials_are_not_surfaced(dashboard, channel: str) -> None:
    """Credentials in report text or log lines must be redacted before display."""
    workspace = dashboard.current_workspace()
    leaked = f"OPENAI_API_KEY={CREDENTIAL}"

    if channel == "report":
        workspace.load_summary_html(f"<p>{leaked}</p>")
        surfaced = workspace.summary_output.toPlainText()
    else:
        workspace.log_output.append(leaked)
        surfaced = workspace.log_output.toPlainText()

    assert CREDENTIAL not in surfaced
    assert "[redacted]" in surfaced
