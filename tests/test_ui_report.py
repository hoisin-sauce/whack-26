"""Report rendering and tab routing (UI-40 to UI-49).

These tests exercise ``QuantDashboard.set_report`` /
``QuantDashboard.load_summary_html`` and ``DatasetWorkspace.load_summary_html``
without pixel snapshots. Assertions use plain-text sentinels so Qt's HTML
normalisation cannot make a test pass or fail on cosmetic differences.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")

from PyQt6.QtCore import QUrl  # noqa: E402
from PyQt6.QtGui import QTextDocument  # noqa: E402

import UI_frontEnd  # noqa: E402

pytestmark = pytest.mark.ui


@pytest.fixture
def dashboard(qtbot):
    widget = UI_frontEnd.QuantDashboard()
    qtbot.addWidget(widget)
    return widget


def _summary_text(workspace: UI_frontEnd.DatasetWorkspace) -> str:
    return workspace.summary_output.toPlainText()


def test_set_report_renders_in_the_active_tab_and_returns_workspace(dashboard):
    """UI-40: the default tab reference targets the active tab."""
    workspace = dashboard.current_workspace()
    assert workspace is not None

    sentinel = "UI40-ACTIVE-TAB-SENTINEL"
    returned = dashboard.set_report(f"<h1>{sentinel}</h1><p>body</p>")

    assert returned is workspace
    assert sentinel in _summary_text(workspace)


def test_set_report_routes_by_index_and_by_workspace(dashboard):
    """UI-41: an index or a workspace routes to exactly that tab."""
    first = dashboard.workspaces()[0]
    second = dashboard.add_dataset_tab()
    third = dashboard.add_dataset_tab()
    dashboard.tabs.setCurrentWidget(third)

    by_index = "UI41-BY-INDEX-SENTINEL"
    returned = dashboard.set_report(f"<p>{by_index}</p>", tab_ref=0)

    assert returned is first
    assert by_index in _summary_text(first)
    assert by_index not in _summary_text(second)
    assert by_index not in _summary_text(third)

    by_workspace = "UI41-BY-WORKSPACE-SENTINEL"
    returned = dashboard.set_report(
        f"<p>{by_workspace}</p>", tab_ref=second
    )

    assert returned is second
    assert by_workspace in _summary_text(second)
    assert by_workspace not in _summary_text(first)
    assert by_workspace not in _summary_text(third)
    # The active tab must not have been used as a fallback.
    assert by_workspace not in _summary_text(third)


@pytest.mark.parametrize("bad_index", [1, 99, -1])
def test_set_report_rejects_an_invalid_index(dashboard, bad_index):
    """UI-42: an out-of-range tab index is an actionable IndexError."""
    with pytest.raises(IndexError, match="No dataset tab exists at index"):
        dashboard.set_report("<p>never rendered</p>", tab_ref=bad_index)


@pytest.mark.parametrize(
    "bad_ref",
    [1.5, "0", object(), ("workspace",)],
    ids=["float", "string", "object", "tuple"],
)
def test_set_report_rejects_an_invalid_reference_type(dashboard, bad_ref):
    """UI-42: a reference that is not None/int/workspace is a TypeError."""
    with pytest.raises(TypeError, match="tab_ref must be a tab index"):
        dashboard.set_report("<p>never rendered</p>", tab_ref=bad_ref)


def test_set_report_rejects_a_detached_workspace(dashboard):
    """UI-42: a workspace from outside the dashboard is not silently used."""
    detached = UI_frontEnd.DatasetWorkspace()

    with pytest.raises(IndexError, match="not attached to this dashboard"):
        dashboard.set_report("<p>never rendered</p>", tab_ref=detached)
    assert "never rendered" not in _summary_text(detached)


def test_load_summary_html_decodes_utf8_bytes(dashboard):
    """UI-43: bytes reports are decoded as UTF-8 before rendering."""
    workspace = dashboard.current_workspace()
    sentinel = "UI43-BYTES-SENTINEL"
    report = f"<p>{sentinel}: café — naïve 日本語</p>"

    returned = dashboard.load_summary_html(report.encode("utf-8"))

    assert returned is workspace
    text = _summary_text(workspace)
    assert sentinel in text
    assert "café" in text
    assert "日本語" in text


@pytest.mark.parametrize(
    "bad_value",
    [None, 42, ["<p>list</p>"], {"html": "<p>dict</p>"}],
    ids=["none", "int", "list", "dict"],
)
def test_load_summary_html_rejects_non_text_values(dashboard, bad_value):
    """UI-44: non-str/non-bytes reports raise TypeError."""
    workspace = dashboard.current_workspace()

    with pytest.raises(TypeError, match="HTML report must be provided as a string"):
        workspace.load_summary_html(bad_value)
    with pytest.raises(TypeError, match="HTML report must be provided as a string"):
        dashboard.load_summary_html(bad_value)


@pytest.mark.parametrize("empty_report", ["", "   ", "\n\t  \n"], ids=["empty", "spaces", "whitespace"])
def test_empty_report_leaves_summary_unchanged_and_logs(dashboard, empty_report):
    """UI-45: empty/whitespace reports do not clear the previous summary."""
    workspace = dashboard.current_workspace()
    sentinel = "UI45-PREVIOUS-REPORT-SENTINEL"
    dashboard.set_report(f"<p>{sentinel}</p>")
    before = _summary_text(workspace)
    assert sentinel in before

    dashboard.load_summary_html(empty_report)

    assert _summary_text(workspace) == before
    log_text = workspace.log_output.toPlainText()
    assert "Empty HTML report received" in log_text
    assert "summary unchanged" in log_text


def test_load_summary_html_is_a_backwards_compatible_alias(dashboard):
    """UI-46: the dashboard alias routes like set_report and returns the tab."""
    first = dashboard.workspaces()[0]
    second = dashboard.add_dataset_tab()
    dashboard.tabs.setCurrentWidget(second)

    sentinel = "UI46-ALIAS-SENTINEL"
    returned = dashboard.load_summary_html(f"<p>{sentinel}</p>", tab_index=0)

    assert returned is first
    assert sentinel in _summary_text(first)
    assert sentinel not in _summary_text(second)


def test_two_tabs_keep_their_reports_isolated(dashboard):
    """UI-47: each tab renders only the report routed to it."""
    first = dashboard.workspaces()[0]
    second = dashboard.add_dataset_tab()
    first_sentinel = "UI47-TAB-A-SENTINEL"
    second_sentinel = "UI47-TAB-B-SENTINEL"

    dashboard.set_report(f"<p>{first_sentinel}</p>", tab_ref=first)
    dashboard.set_report(f"<p>{second_sentinel}</p>", tab_ref=second)

    first_text = _summary_text(first)
    second_text = _summary_text(second)
    assert first_sentinel in first_text
    assert second_sentinel not in first_text
    assert second_sentinel in second_text
    assert first_sentinel not in second_text

    # Switching the active tab must not copy content between the views.
    dashboard.tabs.setCurrentWidget(first)
    assert second_sentinel not in _summary_text(first)
    dashboard.tabs.setCurrentWidget(second)
    assert first_sentinel not in _summary_text(second)


def test_hostile_markup_is_inert(dashboard, qtbot):
    """UI-48: script/event markup does not become active content."""
    workspace = dashboard.current_workspace()
    view = workspace.summary_output
    dashboard.show()
    qtbot.wait(10)

    dashboard.set_report(
        "<p>UI48-VISIBLE-SENTINEL</p>"
        "<script>UI48-SCRIPT-RAN; window.location='https://example.invalid/';</script>"
        "<p onclick=\"UI48-ONCLICK-RAN\">clickable text</p>"
        "<img src=\"https://example.invalid/ui48-pixel.png\" "
        "onerror=\"UI48-ONERROR-RAN\">"
    )
    qtbot.wait(10)

    text = _summary_text(workspace)
    document_html = view.document().toHtml().lower()

    # The visible content still renders...
    assert "UI48-VISIBLE-SENTINEL" in text
    # ...but script code and event handlers are not part of the document.
    assert "UI48-SCRIPT-RAN" not in text
    assert "<script" not in document_html
    assert "ui48-onclick-ran" not in document_html
    assert "ui48-onerror-ran" not in document_html
    assert "onclick" not in document_html
    assert "onerror" not in document_html
    # Rendering must not navigate the browser or execute a script.
    assert view.source().isEmpty()
    assert "UI48-SCRIPT-RAN" not in dashboard.windowTitle()
    # A remote image is not resolved into a loaded resource.
    image = view.document().resource(
        QTextDocument.ResourceType.ImageResource,
        QUrl("https://example.invalid/ui48-pixel.png"),
    )
    assert image is None


def test_escaped_report_text_and_external_urls_stay_literal(dashboard):
    """UI-48: the report producer's escaping survives as literal text."""
    workspace = dashboard.current_workspace()
    escaped = (
        "<p>UI48-ESCAPED-SENTINEL</p>"
        "<p>&lt;script&gt;alert('UI48-XSS')&lt;/script&gt;</p>"
        "<p>&lt;img src=x onerror=alert('UI48-XSS')&gt;</p>"
        "<p>https://example.invalid/ui48-external-link</p>"
    )

    dashboard.set_report(escaped)

    text = _summary_text(workspace)
    assert "UI48-ESCAPED-SENTINEL" in text
    assert "<script>alert('UI48-XSS')</script>" in text
    assert "<img src=x onerror=alert('UI48-XSS')>" in text
    assert "https://example.invalid/ui48-external-link" in text
    # Escaped markup must not create a script node or an image node.
    document_html = workspace.summary_output.document().toHtml().lower()
    assert "<script" not in document_html
    assert "<img" not in document_html


def test_report_load_resets_scroll_and_logs_the_embed(dashboard, qtbot):
    """UI-49: a new report starts at the top and is recorded in the log."""
    workspace = dashboard.current_workspace()
    view = workspace.summary_output
    bar = view.verticalScrollBar()

    dashboard.resize(900, 640)
    dashboard.show()
    qtbot.wait(10)

    first_report = "".join(f"<p>UI49 first line {i}</p>" for i in range(250))
    dashboard.set_report(first_report)
    qtbot.wait(10)
    assert bar.maximum() > 0, "the fixture report must overflow the panel"
    bar.setValue(bar.maximum())
    assert bar.value() > 0

    second_report = "".join(f"<p>UI49 second line {i}</p>" for i in range(250))
    dashboard.set_report(second_report)
    qtbot.wait(10)

    assert bar.maximum() > 0
    assert bar.value() == 0
    log_text = workspace.log_output.toPlainText()
    assert log_text.count("HTML report embedded in summary.") >= 2
