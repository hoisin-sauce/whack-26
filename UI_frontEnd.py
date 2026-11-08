import html
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QKeySequence, QPalette, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from stats_engine import build_linear_model, r2_score, trial_sharpes, validation_engine

# The validation dashboard requires a continuous target column. This contract
# is enforced by the engine as well, so a caller cannot bypass it.
TARGET_CONTRACT = (
    "The last column must be a continuous return-like target with finite values, "
    "non-zero variance, and at least three distinct values. Binary labels and "
    "constant targets are rejected."
)

PIPELINE_BUTTON_IDLE = "Run Pipeline & Load Report"
PIPELINE_BUTTON_RUNNING = "Running Pipeline..."
PIPELINE_LOADING_HTML = """
<h1>Report loading...</h1>
<p>The dataset is running through the research pipeline. This can take a
few minutes. The final-judge report will replace this message when ready.</p>
"""


def _error_html(title: str, message: str) -> str:
    return (
        f"<h1>{html.escape(title)}</h1>"
        f"<p>{html.escape(message)}</p>"
    )


APP_STYLESHEET = """
QWidget {
    background-color: #f6f5f2;
    color: #21293e;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10pt;
}

QFrame#headerCard {
    background-color: #141c25;
    border: none;
    border-bottom: 1px solid #2b3542;
}

QLabel#eyebrowLabel {
    color: #9d968c;
    font-size: 8pt;
    font-weight: 600;
    background: transparent;
}

QLabel#titleLabel {
    color: #ece9e5;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 22pt;
    font-weight: normal;
    background: transparent;
}

QLabel#subtitleLabel {
    color: #b9c0c9;
    font-size: 10pt;
    background: transparent;
}

QGroupBox {
    background-color: #ffffff;
    border: 1px solid #e2ded7;
    border-radius: 0px;
    margin-top: 28px;
    font-weight: 700;
    color: #21293e;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 0px;
    top: 0px;
    padding: 0px 0px 8px 0px;
    color: #21293e;
    background-color: transparent;
    border: none;
    font-size: 9pt;
    font-weight: 700;
}

QGroupBox QLabel {
    color: #434c60;
    background: transparent;
    font-size: 9.5pt;
}

QLabel#hintLabel {
    color: #9d968c;
    background: transparent;
    font-size: 8.5pt;
}

QLineEdit {
    background-color: #ffffff;
    color: #21293e;
    border: 1px solid #d9d5cf;
    border-radius: 0px;
    padding: 9px 10px;
    selection-background-color: #3b4254;
    selection-color: #ffffff;
}

QLineEdit:hover {
    border-color: #bdb4a8;
}

QLineEdit:focus {
    border-color: #21293e;
}

QPushButton#runButton {
    background-color: #21293e;
    color: #ffffff;
    border: 1px solid #21293e;
    border-radius: 0px;
    padding: 14px 18px;
    font-size: 10.5pt;
    font-weight: 600;
}

QPushButton#runButton:hover {
    background-color: #141c25;
    border-color: #141c25;
}

QPushButton#runButton:pressed {
    background-color: #0e131a;
    border-color: #0e131a;
}

QTextEdit, QTextBrowser {
    background-color: #faf9f7;
    color: #21293e;
    border: 1px solid #e2ded7;
    border-radius: 0px;
    padding: 14px;
    selection-background-color: #3b4254;
    selection-color: #ffffff;
}

QTextEdit#logOutput {
    background-color: #f6f5f2;
    border: 1px dashed #bdb4a8;
    font-family: "Consolas", monospace;
    font-size: 9.5pt;
}

QTextEdit#logOutput:hover {
    border-color: #9d968c;
}

QTextEdit#summaryOutput, QTextBrowser#summaryOutput {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10pt;
}

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 4px 2px 4px 2px;
}

QScrollBar::handle:vertical {
    background: #c9c4bc;
    border-radius: 0px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #9d968c;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}

QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px 4px 2px 4px;
}

QScrollBar::handle:horizontal {
    background: #c9c4bc;
    border-radius: 0px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: #9d968c;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
}

QTabWidget {
    background: transparent;
    border: none;
}

QTabWidget::pane {
    background: transparent;
    border: none;
    top: -1px;
}

QTabWidget::tab-bar {
    left: 32px;
}

QTabBar {
    background: transparent;
    qproperty-drawBase: 0;
}

QTabBar::tab {
    background: transparent;
    color: #8b857b;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 11px 16px 9px 16px;
    margin-right: 6px;
    font-size: 9pt;
    font-weight: 600;
}

QTabBar::tab:hover {
    color: #21293e;
}

QTabBar::tab:selected {
    color: #21293e;
    border-bottom: 2px solid #21293e;
}

QTabBar::close-button {
    margin-left: 6px;
}

QTabBar::close-button:hover {
    background-color: #e2ded7;
}

QPushButton#addTabButton {
    background-color: transparent;
    color: #21293e;
    border: 1px solid #d9d5cf;
    border-radius: 0px;
    padding: 7px 14px;
    margin: 6px 32px 6px 0px;
    font-size: 8.5pt;
    font-weight: 700;
}

QPushButton#addTabButton:hover {
    background-color: #ece9e5;
    border-color: #21293e;
}

QPushButton#addTabButton:pressed {
    background-color: #e2ded7;
}
"""


# Default styling for generated HTML reports. The report producer is expected
# to emit semantic HTML without CSS; this stylesheet supplies the Brevan
# Howard palette and typography inside the summary panel.
REPORT_STYLESHEET = """
h1 {
    font-family: Georgia;
    font-size: 20pt;
    font-weight: normal;
    color: #21293e;
    margin-top: 0px;
    margin-bottom: 6px;
}
h2 {
    font-family: Georgia;
    font-size: 14pt;
    font-weight: normal;
    color: #21293e;
    margin-top: 14px;
    margin-bottom: 4px;
}
h3 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #21293e;
    margin-top: 12px;
    margin-bottom: 3px;
}
p {
    color: #434c60;
    margin-top: 4px;
    margin-bottom: 7px;
}
a {
    color: #21293e;
    text-decoration: underline;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 8px;
    margin-bottom: 8px;
}
th {
    background-color: #f6f5f2;
    color: #21293e;
    font-weight: 600;
    text-align: left;
    border: 1px solid #d9d5cf;
    padding: 6px 8px;
}
td {
    color: #434c60;
    border: 1px solid #d9d5cf;
    padding: 6px 8px;
}
ul, ol {
    margin-top: 4px;
    margin-bottom: 7px;
}
li {
    color: #434c60;
    margin-bottom: 3px;
}
hr {
    border: 0px;
    border-top: 1px solid #d9d5cf;
    margin-top: 12px;
    margin-bottom: 12px;
}
blockquote {
    border-left: 3px solid #bdb4a8;
    margin-left: 0px;
    padding-left: 10px;
    color: #434c60;
}
code, pre {
    font-family: Consolas;
    background-color: #f6f5f2;
    color: #21293e;
}
strong, b {
    color: #21293e;
    font-weight: 600;
}
.banner {
    background-color: #f6f5f2;
    border: 1px solid #d9d5cf;
    padding: 8px 10px;
    margin-bottom: 8px;
}
.banner.warn {
    background-color: #fdf3d9;
    border-color: #c99a2e;
}
.banner.alert {
    background-color: #fbe9dd;
    border-color: #b4511c;
}
.banner.bad {
    background-color: #f9e1e1;
    border-color: #a42626;
}
.banner.ok {
    background-color: #e6f4ec;
    border-color: #1f7a4d;
}
.card {
    border: 1px solid #d9d5cf;
    padding: 8px 10px;
    margin-bottom: 8px;
}
.badge {
    color: #434c60;
    font-size: 9pt;
    margin-right: 6px;
}
.finding {
    color: #21293e;
    margin-top: 4px;
}
"""


class DragDropTextEdit(QTextEdit):
    """Custom QTextEdit widget that accepts dropped files and opens a picker."""

    dataset_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.parent_dashboard = parent

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dataset_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path:
                self.parent_dashboard.handle_dropped_file(file_path)


class HtmlSummaryView(QTextBrowser):
    """Read-only summary panel that renders HTML reports generated elsewhere."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(True)
        self.setReadOnly(True)
        self.document().setDefaultStyleSheet(REPORT_STYLESHEET)
        self.document().setDefaultFont(QFont("Segoe UI", 10))


class _PipelineSignals(QObject):
    """Signals emitted by a background pipeline task."""

    log = pyqtSignal(str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)


class _PipelineTask(threading.Thread):
    """Run the research pipeline off the GUI thread.

    A daemon thread is deliberate: closing a tab or the application must not
    leave a live Qt thread-pool worker behind. The pipeline itself is still
    loaded through the normal run-directory handoff when it completes.
    """

    def __init__(self, runner: Callable[..., Any], kwargs: dict[str, Any]):
        super().__init__(daemon=True)
        self.runner = runner
        self.kwargs = dict(kwargs)
        self.signals = _PipelineSignals()

    def run(self) -> None:
        try:
            self.kwargs["log"] = self.signals.log.emit
            outcome = self.runner(**self.kwargs)
        except Exception as exc:  # noqa: BLE001 - surface the failure in the UI
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(outcome)


class DatasetWorkspace(QWidget):
    """Independent validation workspace for a single dataset."""

    dataset_name_changed = pyqtSignal(str)
    tab_name_changed = pyqtSignal()

    def __init__(
        self,
        parent=None,
        default_title="Dataset",
        auto_run_pipeline=False,
        pipeline_runner: Callable[..., Any] | None = None,
    ):
        super().__init__(parent)
        self.dropped_file_path = None
        self.loaded_dataframe = None
        self.default_title = default_title
        self.dataset_name = None
        self.custom_tab_name = None
        self.auto_run_pipeline = bool(auto_run_pipeline)
        self._pipeline_runner = pipeline_runner
        self._pipeline_task: _PipelineTask | None = None
        self._pipeline_running = False
        self.initUI()

    def set_tab_name(self, name):
        """Set an explicit tab name; ``None``/empty restores auto-naming."""
        if name is None:
            self.custom_tab_name = None
        else:
            cleaned = " ".join(str(name).split())
            self.custom_tab_name = cleaned or None
        self.tab_name_changed.emit()

    def suggested_tab_name(self):
        """Return the preferred tab name before uniqueness handling."""
        return self.custom_tab_name or self.dataset_name or self.default_title

    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(32, 22, 32, 22)
        main_layout.setSpacing(16)

        # --- CONFIGURATION & METADATA ---
        config_row = QHBoxLayout()
        config_row.setSpacing(18)

        # Left: Strategy Metadata
        metadata_group = QGroupBox("01 / STRATEGY METADATA")
        metadata_container = QVBoxLayout()
        metadata_container.setContentsMargins(18, 14, 18, 16)
        metadata_container.setSpacing(10)
        metadata_layout = QFormLayout()
        metadata_layout.setContentsMargins(0, 0, 0, 0)
        metadata_layout.setVerticalSpacing(12)
        metadata_layout.setHorizontalSpacing(16)
        metadata_layout.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        metadata_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.strategy_name_input = QLineEdit()
        self.strategy_name_input.setPlaceholderText("e.g. Momentum_Alpha_v1")

        self.data_desc_input = QLineEdit()
        self.data_desc_input.setPlaceholderText(
            "e.g. Daily S&P 500 returns with macroeconomic indicators"
        )

        metadata_layout.addRow("Strategy Name", self.strategy_name_input)
        metadata_layout.addRow("Dataset Description", self.data_desc_input)
        metadata_container.addLayout(metadata_layout)
        metadata_container.addStretch()

        metadata_note = QLabel(
            "These details label the report and provide context for the engine."
        )
        metadata_note.setObjectName("hintLabel")
        metadata_note.setWordWrap(True)
        metadata_container.addWidget(metadata_note)

        metadata_group.setLayout(metadata_container)
        config_row.addWidget(metadata_group, 1)

        # Right: Test Restrictions
        restrictions_group = QGroupBox("02 / TEST RESTRICTIONS")
        restrictions_container = QVBoxLayout()
        restrictions_container.setContentsMargins(18, 14, 18, 16)
        restrictions_container.setSpacing(10)
        restrictions_layout = QFormLayout()
        restrictions_layout.setContentsMargins(0, 0, 0, 0)
        restrictions_layout.setVerticalSpacing(12)
        restrictions_layout.setHorizontalSpacing(16)
        restrictions_layout.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        restrictions_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.periods_input = QLineEdit()
        self.periods_input.setPlaceholderText("e.g. 252")

        self.min_sharpe_input = QLineEdit()
        self.min_sharpe_input.setPlaceholderText("e.g. 0.95 (DSR threshold)")

        self.max_decay_input = QLineEdit()
        self.max_decay_input.setPlaceholderText(
            "e.g. 0.25 (clamped to 0.15-0.50)"
        )

        restrictions_layout.addRow("Periods Per Year", self.periods_input)
        restrictions_layout.addRow("Min DSR Threshold", self.min_sharpe_input)
        restrictions_layout.addRow("Max OOS Decay Tolerance", self.max_decay_input)
        restrictions_container.addLayout(restrictions_layout)
        restrictions_container.addStretch()

        restrictions_note = QLabel(
            "Thresholds are applied by the validation engine at run time."
        )
        restrictions_note.setObjectName("hintLabel")
        restrictions_note.setWordWrap(True)
        restrictions_container.addWidget(restrictions_note)

        restrictions_group.setLayout(restrictions_container)
        config_row.addWidget(restrictions_group, 1)

        main_layout.addLayout(config_row)

        # --- RUN BUTTON ---
        self.run_btn = QPushButton(PIPELINE_BUTTON_IDLE)
        self.run_btn.setObjectName("runButton")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.setMinimumHeight(50)
        self.run_btn.clicked.connect(self.start_pipeline)
        main_layout.addWidget(self.run_btn)

        # --- SPLIT LAYOUT FOR LOGS AND SUMMARY REPORT ---
        results_layout = QHBoxLayout()
        results_layout.setSpacing(18)

        # Left Box: Activity & File Drop Log
        log_group = QGroupBox("03 / DATASET PICKER / DROP ZONE && ACTIVITY LOG")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(16, 14, 16, 16)
        log_layout.setSpacing(10)

        contract_note = QLabel(TARGET_CONTRACT)
        contract_note.setObjectName("hintLabel")
        contract_note.setWordWrap(True)
        log_layout.addWidget(contract_note)

        self.browse_btn = QPushButton("Choose dataset...")
        self.browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.browse_btn.clicked.connect(self.browse_for_dataset)
        log_layout.addWidget(self.browse_btn)

        self.log_output = DragDropTextEdit(self)
        self.log_output.setObjectName("logOutput")
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText(
            "Click this box to choose a dataset, or drop a .csv/.xls/.xlsx file here..."
        )
        self.log_output.dataset_requested.connect(self.browse_for_dataset)
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        results_layout.addWidget(log_group, 3)

        # Right Box: Dedicated Summary Report Area
        summary_group = QGroupBox("04 / VALIDATION SUMMARY REPORT")
        summary_layout = QVBoxLayout()
        summary_layout.setContentsMargins(16, 14, 16, 16)

        self.summary_output = HtmlSummaryView(self)
        self.summary_output.setObjectName("summaryOutput")
        self.summary_output.setReadOnly(True)
        self.summary_output.setPlaceholderText(
            "The final-judge report will appear here after the pipeline runs."
        )
        summary_layout.addWidget(self.summary_output)
        summary_group.setLayout(summary_layout)
        results_layout.addWidget(summary_group, 2)

        main_layout.addLayout(results_layout, 1)

    def load_summary_html(self, html):
        """Render a generated HTML report in the summary panel.

        Call this from the report generator with semantic HTML. The report does
        not need any CSS; the panel applies the Brevan Howard report stylesheet
        automatically.
        """
        if isinstance(html, bytes):
            html = html.decode("utf-8", errors="replace")
        if not isinstance(html, str):
            raise TypeError("HTML report must be provided as a string.")
        if not html.strip():
            self.log_output.append("\nEmpty HTML report received; summary unchanged.")
            return

        self.summary_output.setHtml(html)
        self.summary_output.verticalScrollBar().setValue(0)
        self.log_output.append("\nHTML report embedded in summary.")

    def load_final_judge_report(self, run_dir):
        """Load ``final_judge.fragment.html`` from a run directory.

        The final judge writes a CSS-free semantic HTML fragment specifically
        for this panel; the dashboard stylesheet supplies the presentation.
        """
        fragment = Path(run_dir) / "final_judge.fragment.html"
        if not fragment.is_file():
            raise FileNotFoundError(
                f"no final judge report at {fragment}; run the pipeline with --judge"
            )
        self.load_summary_html(fragment.read_text(encoding="utf-8"))

    # ---------------------------------------------------------- pipeline run
    @staticmethod
    def _default_pipeline_runner(**kwargs: Any) -> Any:
        from whack26.run import run_pipeline

        return run_pipeline(**kwargs)

    def _pipeline_validation_config(self):
        from whack26.validation.models import ValidationConfig

        def number(text: str, default: float) -> float:
            cleaned = text.strip()
            return default if not cleaned else float(cleaned)

        periods = number(self.periods_input.text(), 252.0)
        min_dsr = number(self.min_sharpe_input.text(), 0.95)
        decay_text = self.max_decay_input.text().strip()
        max_decay = None if not decay_text else float(decay_text)
        return ValidationConfig(
            min_dsr=min_dsr,
            periods_per_year=periods,
            max_decay_tolerance=max_decay,
        )

    def _set_pipeline_loading(self, loading: bool) -> None:
        self._pipeline_running = loading
        self.run_btn.setEnabled(not loading)
        self.run_btn.setText(
            PIPELINE_BUTTON_RUNNING if loading else PIPELINE_BUTTON_IDLE
        )
        if loading:
            self.summary_output.setHtml(PIPELINE_LOADING_HTML)
            self.summary_output.verticalScrollBar().setValue(0)

    def _show_report_error(self, message: str) -> None:
        self.summary_output.setHtml(_error_html("Report unavailable", message))
        self.summary_output.verticalScrollBar().setValue(0)
        self.log_output.append(f"\nReport unavailable: {message}")

    def start_pipeline(self) -> None:
        """Run the full research pipeline and load its final-judge report."""
        if self._pipeline_running:
            self.log_output.append("\nPipeline is already running for this tab.")
            return
        if self.loaded_dataframe is None or not self.dropped_file_path:
            message = (
                "No dataset is loaded. Drop a .csv/.xls/.xlsx file before running."
            )
            self.log_output.append(f"\n{message}")
            self._show_report_error(message)
            return
        try:
            validation_config = self._pipeline_validation_config()
        except (TypeError, ValueError) as exc:
            message = f"Invalid restriction settings: {exc}"
            self.log_output.append(f"\n{message}")
            self._show_report_error(message)
            return

        target = None
        if len(self.loaded_dataframe.columns):
            target = str(self.loaded_dataframe.columns[-1])
        description = self.data_desc_input.text().strip() or None
        runner = self._pipeline_runner or self._default_pipeline_runner
        kwargs = {
            "input_path": Path(self.dropped_file_path),
            "description": description,
            "target": target,
            "model_search": True,
            "validate": True,
            "validation_config": validation_config,
            "final_judge": True,
        }

        self.log_output.append(
            f"\n--- Starting pipeline for {os.path.basename(self.dropped_file_path)} ---"
        )
        self._set_pipeline_loading(True)
        task = _PipelineTask(runner, kwargs)
        self._pipeline_task = task
        # Bound QObject methods (not lambdas) so Qt disconnects them if this
        # workspace is deleted while the daemon task is still running.
        task.signals.log.connect(self._on_pipeline_log)
        task.signals.finished.connect(self._on_pipeline_finished)
        task.signals.failed.connect(self._on_pipeline_failed)
        task.start()
        self.log_output.append("Pipeline is running in the background...")

    def shutdown(self) -> None:
        """Disconnect an in-flight pipeline task before this widget is deleted."""
        task = self._pipeline_task
        self._pipeline_task = None
        if task is None:
            return
        for signal, slot in (
            (task.signals.log, self._on_pipeline_log),
            (task.signals.finished, self._on_pipeline_finished),
            (task.signals.failed, self._on_pipeline_failed),
        ):
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def closeEvent(self, event):  # noqa: N802 - Qt override
        self.shutdown()
        super().closeEvent(event)

    def _on_pipeline_log(self, line: str) -> None:
        self.log_output.append(line)

    def _on_pipeline_finished(self, outcome: Any) -> None:
        if self._pipeline_task is None:
            return
        self._pipeline_task = None
        self._set_pipeline_loading(False)
        run_dir = getattr(outcome, "run_dir", None)
        if run_dir is None:
            self._show_report_error("The pipeline finished without a run directory.")
            return
        self.log_output.append(f"\nPipeline finished: {run_dir}")
        try:
            self.load_final_judge_report(run_dir)
        except (OSError, ValueError) as exc:
            self._show_report_error(
                f"The pipeline finished but its report could not be loaded: {exc}"
            )

    def _on_pipeline_failed(self, message: str) -> None:
        if self._pipeline_task is None:
            return
        self._pipeline_task = None
        self._set_pipeline_loading(False)
        self.log_output.append(f"\nPipeline failed: {message}")
        self._show_report_error(f"The pipeline failed: {message}")

    def browse_for_dataset(self) -> None:
        """Open a file picker and submit the chosen dataset."""
        if self.dropped_file_path:
            start_dir = str(Path(self.dropped_file_path).parent)
        else:
            start_dir = str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose dataset",
            start_dir,
            "Datasets (*.csv *.xls *.xlsx);;All files (*)",
        )
        if file_path:
            self.handle_dropped_file(file_path)

    def handle_dropped_file(self, file_path):
        self.dropped_file_path = file_path
        file_ext = os.path.splitext(file_path)[1].lower()

        self.log_output.append(f"\nFile dropped: {os.path.basename(file_path)}")

        try:
            if file_ext == ".csv":
                self.loaded_dataframe = pd.read_csv(file_path)
            elif file_ext in [".xls", ".xlsx"]:
                self.loaded_dataframe = pd.read_excel(file_path)
            else:
                self.log_output.append(
                    f"Unsupported file extension: {file_ext}. "
                    "Please use .csv, .xls, or .xlsx"
                )
                return

            self.log_output.append(
                f"Successfully loaded dataset! Shape: {self.loaded_dataframe.shape}"
            )
            self.log_output.append(
                f"Columns detected: {list(self.loaded_dataframe.columns)}"
            )
            self.log_output.append(TARGET_CONTRACT)

            dataset_label = os.path.splitext(os.path.basename(file_path))[0].strip()
            if dataset_label:
                self.dataset_name = dataset_label
                self.dataset_name_changed.emit(dataset_label)

            if self.auto_run_pipeline:
                self.start_pipeline()
        except Exception as e:
            self.log_output.append(f"Error parsing file with Pandas: {str(e)}")

    def _prepare_arrays(self):
        """Return numeric features and the continuous target, or raise ValueError."""
        if self.loaded_dataframe is None:
            raise ValueError(
                "No dataset is loaded. Drop a .csv/.xls/.xlsx file before running."
            )
        source = self.loaded_dataframe
        if source.shape[1] < 2:
            raise ValueError(
                "The dataset needs at least one feature column and one target column."
            )

        target_name = source.columns[-1]
        if not np.issubdtype(source[target_name].dtype, np.number):
            raise ValueError(
                f"The target column {target_name!r} must be numeric and continuous."
            )
        feature_names = [
            name
            for name in source.columns[:-1]
            if np.issubdtype(source[name].dtype, np.number)
        ]
        dropped = [name for name in source.columns[:-1] if name not in feature_names]
        if not feature_names:
            raise ValueError(
                "No numeric feature columns were found. The dashboard requires "
                "numeric features before the target column."
            )
        if dropped:
            self.log_output.append(
                "Ignoring non-numeric feature columns: " + ", ".join(map(str, dropped))
            )

        # Drop missing rows only for the columns that are actually used, so a
        # sparse text column cannot shrink the validation sample.
        frame = source[feature_names + [target_name]].dropna()
        if frame.empty:
            raise ValueError("The dataset is empty after dropping missing rows.")

        features = frame[feature_names].to_numpy(dtype=float)
        targets = frame[target_name].to_numpy(dtype=float)
        if targets.size < 30:
            raise ValueError(
                f"At least 30 complete rows are required; found {targets.size}."
            )
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise ValueError("Features and target must contain only finite values.")
        if np.std(targets, ddof=1) == 0.0 or np.unique(targets).size < 3:
            raise ValueError(
                "The target must be continuous with non-zero variance and at "
                "least three distinct values. Binary and constant targets are "
                "not supported."
            )
        return features, targets

    def run_validation(self):
        strategy_name = self.strategy_name_input.text().strip()
        if not strategy_name:
            strategy_name = "Unnamed_Strategy"

        data_description = self.data_desc_input.text().strip()
        if not data_description:
            data_description = "No description provided."

        try:
            periods_per_year = float(self.periods_input.text())
            min_sharpe = float(self.min_sharpe_input.text())
            max_decay = float(self.max_decay_input.text())
        except ValueError:
            self.log_output.append(
                "\n Missing or Invalid Inputs: Please fill in all restriction "
                "and metadata fields with valid numbers before running."
            )
            return

        self.log_output.append(
            f"\n--- [GUI] Triggering Validation for '{strategy_name}' ---"
        )
        self.log_output.append(f"Dataset Description: {data_description}")
        self.log_output.append(
            f"Constraints -> Periods: {periods_per_year} | "
            f"Min DSR: {min_sharpe} | Max Decay: {max_decay}"
        )

        # Clear and initialise the Summary Report view
        self.summary_output.clear()
        self.summary_output.append(f"=== SUMMARY REPORT: {strategy_name} ===")
        self.summary_output.append(f"Description: {data_description}\n")

        try:
            features, targets = self._prepare_arrays()
        except ValueError as exc:
            self.log_output.append(f"Dataset rejected: {exc}")
            self.summary_output.append(f"Status: REJECTED\nReason: {exc}")
            return

        train_split_ratio = 0.70
        split_index = int(len(targets) * train_split_ratio)
        if split_index < 10 or len(targets) - split_index < 10:
            message = (
                "At least 10 training rows and 10 test rows are required after "
                "the chronological split."
            )
            self.log_output.append(f"Dataset rejected: {message}")
            self.summary_output.append(f"Status: REJECTED\nReason: {message}")
            return

        features_train = features[:split_index]
        targets_train = targets[:split_index]
        features_test = features[split_index:]
        targets_test = targets[split_index:]

        base_params = {
            "ridge": 1e-6,
            "min_dsr_limit": min_sharpe,
            "max_decay_tolerance": max_decay,
        }

        try:
            model = build_linear_model(base_params, features_train, targets_train)
            predictions = model(features_test)
            trial_values, winning_returns = trial_sharpes(
                features_train,
                targets_train,
                periods_per_year,
                model_predict=model,
            )
        except Exception as exc:  # noqa: BLE001 - report to the user
            self.log_output.append(f"Model preparation failed: {exc}")
            self.summary_output.append(f"Status: REJECTED\nReason: {exc}")
            return

        if len(trial_values) < 2:
            message = (
                "At least two real trial signals are required for the DSR "
                "multiple-testing penalty."
            )
            self.log_output.append(f"Validation rejected: {message}")
            self.summary_output.append(f"Status: REJECTED\nReason: {message}")
            return

        try:
            result = validation_engine(
                winning_returns=winning_returns,
                all_trial_sharpes=trial_values,
                predictions=predictions,
                actuals=targets_test,
                features=features,
                targets=targets,
                base_model_builder=build_linear_model,
                base_params=base_params,
                periods_per_year=periods_per_year,
                metric_func=r2_score,
                higher_is_better=True,
                data_description=data_description,
            )

            passed = bool(result.get("passed", False))
            status = "PASSED" if passed else "REJECTED"
            self.summary_output.append(f"Status: {status}")
            for key, val in result.items():
                if key == "passed":
                    continue
                self.summary_output.append(
                    f"{key.replace('_', ' ').title()}: {val}"
                )
            if result.get("tolerance_clamped"):
                self.summary_output.append(
                    "Warning: max_decay_tolerance was clamped to the supported "
                    "0.15-0.50 range."
                )
            self.log_output.append(
                f"Validation cycle completed: {status} "
                f"({len(trial_values)} real trials)."
            )
        except Exception as e:
            self.log_output.append(f"Engine Execution Error: {str(e)}")
            self.summary_output.append(f"Error executing engine: {str(e)}")


class QuantDashboard(QWidget):
    """Tabbed shell hosting one fully independent workspace per dataset."""

    def __init__(
        self,
        *,
        auto_run_pipeline: bool = False,
        pipeline_runner: Callable[..., Any] | None = None,
    ):
        super().__init__()
        self._dataset_counter = 0
        self._auto_run_pipeline = bool(auto_run_pipeline)
        self._pipeline_runner = pipeline_runner
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Autonomous Quant Researcher - Dashboard")
        self.resize(1180, 880)
        self.setMinimumSize(1024, 720)
        self.setStyleSheet(APP_STYLESHEET)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#9d968c"))
        self.setPalette(palette)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Title Header ---
        header_card = QFrame()
        header_card.setObjectName("headerCard")
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(34, 22, 34, 20)
        header_layout.setSpacing(6)

        eyebrow_label = QLabel("QUANT RESEARCH / VALIDATION ENGINE")
        eyebrow_label.setObjectName("eyebrowLabel")
        eyebrow_font = QFont("Segoe UI", 8)
        eyebrow_font.setWeight(QFont.Weight.DemiBold)
        eyebrow_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.4)
        eyebrow_label.setFont(eyebrow_font)
        header_layout.addWidget(eyebrow_label)

        self.title_label = QLabel("Brevan Howard Track: Autonomous Validation Engine")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setWordWrap(True)
        header_layout.addWidget(self.title_label)

        subtitle_label = QLabel(
            "Each tab holds an unrelated dataset. Drop a dataset and it runs "
            "through the research pipeline automatically; the final-judge report "
            "appears in that tab when ready."
        )
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setWordWrap(True)
        header_layout.addWidget(subtitle_label)

        main_layout.addWidget(header_card)

        # --- DATASET TABS ---
        self.tabs = QTabWidget()
        self.tabs.setObjectName("datasetTabs")
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setElideMode(Qt.TextElideMode.ElideNone)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.add_tab_button = QPushButton("+  NEW DATASET")
        self.add_tab_button.setObjectName("addTabButton")
        self.add_tab_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_tab_button.clicked.connect(self.add_dataset_tab)
        self.tabs.setCornerWidget(self.add_tab_button, Qt.Corner.TopRightCorner)

        main_layout.addWidget(self.tabs, 1)

        QShortcut(QKeySequence("Ctrl+T"), self, activated=self.add_dataset_tab)
        QShortcut(QKeySequence("Ctrl+W"), self, activated=self._close_current_tab)

        self.add_dataset_tab()

    # ------------------------------------------------------------------ tabs
    def add_dataset_tab(self, focus=True):
        """Create a new isolated dataset workspace and return its tab reference."""
        self._dataset_counter += 1
        workspace = DatasetWorkspace(
            default_title=f"Dataset {self._dataset_counter}",
            auto_run_pipeline=self._auto_run_pipeline,
            pipeline_runner=self._pipeline_runner,
        )
        index = self.tabs.addTab(workspace, workspace.default_title)
        workspace.dataset_name_changed.connect(
            lambda _name, ws=workspace: self._refresh_tab_titles()
        )
        workspace.tab_name_changed.connect(self._refresh_tab_titles)
        workspace.data_desc_input.textChanged.connect(
            lambda _text, ws=workspace: self._sync_tab_tooltip(ws)
        )
        self.tabs.setTabToolTip(index, workspace.default_title)
        if focus:
            self.tabs.setCurrentIndex(index)
        return workspace

    def close_tab(self, index):
        if index < 0:
            return
        workspace = self.tabs.widget(index)
        if workspace is None:
            return
        workspace.shutdown()
        self.tabs.removeTab(index)
        workspace.deleteLater()
        if self.tabs.count() == 0:
            self.add_dataset_tab()
        else:
            self._refresh_tab_titles()

    def _close_current_tab(self):
        self.close_tab(self.tabs.currentIndex())

    def _on_tab_changed(self, _index):
        self._sync_window_title()

    def _auto_tab_title(self, workspace):
        """Derive a unique tab name from the loaded dataset."""
        base = workspace.dataset_name or workspace.default_title
        existing = {
            self.tabs.tabText(i)
            for i in range(self.tabs.count())
            if self.tabs.widget(i) is not workspace
        }
        if base not in existing:
            return base
        suffix = 2
        while f"{base} ({suffix})" in existing:
            suffix += 1
        return f"{base} ({suffix})"

    def _tab_title_for(self, workspace):
        if workspace.custom_tab_name:
            return workspace.custom_tab_name
        return self._auto_tab_title(workspace)

    def _refresh_tab_titles(self):
        """Recompute every auto-generated tab name after a rename or close."""
        for index in range(self.tabs.count()):
            workspace = self.tabs.widget(index)
            if workspace is None:
                continue
            self.tabs.setTabText(index, self._tab_title_for(workspace))
        self._sync_window_title()

    def _sync_tab_tooltip(self, workspace):
        index = self.tabs.indexOf(workspace)
        if index == -1:
            return
        description = workspace.data_desc_input.text().strip()
        self.tabs.setTabToolTip(index, description or self._tab_title_for(workspace))

    def _sync_window_title(self, title=None):
        tabs = self.__dict__.get("tabs")
        if title is None and tabs is not None and tabs.currentIndex() != -1:
            title = tabs.tabText(tabs.currentIndex())
        self.setWindowTitle(f"Autonomous Quant Researcher - {title or 'Dashboard'}")

    # -------------------------------------------------- cross-tab public API
    def _resolve_workspace(self, tab_ref):
        """Resolve a tab reference to a workspace.

        ``tab_ref`` may be ``None`` for the active tab, an integer tab index,
        or a workspace returned by :meth:`add_dataset_tab`.
        """
        if tab_ref is None:
            workspace = self.tabs.currentWidget()
            if workspace is None:
                raise IndexError("No dataset tab is active.")
            return workspace
        if isinstance(tab_ref, DatasetWorkspace):
            if self.tabs.indexOf(tab_ref) == -1:
                raise IndexError(
                    "The supplied workspace is not attached to this dashboard."
                )
            return tab_ref
        if isinstance(tab_ref, int):
            workspace = self.tabs.widget(tab_ref)
            if workspace is None:
                raise IndexError(f"No dataset tab exists at index {tab_ref}.")
            return workspace
        raise TypeError("tab_ref must be a tab index, a DatasetWorkspace, or None.")

    def set_tab_name(self, name, tab_ref=None):
        """Set a tab name explicitly.

        ``tab_ref`` accepts a tab index or the workspace returned by
        :meth:`add_dataset_tab`; ``None`` targets the active tab. Pass ``None``
        or an empty string as ``name`` to restore dataset-based auto-naming.
        """
        workspace = self._resolve_workspace(tab_ref)
        workspace.set_tab_name(name)
        return workspace

    def set_report(self, html, tab_ref=None):
        """Render generated HTML in a specific tab's summary report.

        ``tab_ref`` accepts a tab index or the workspace returned by
        :meth:`add_dataset_tab`; ``None`` targets the active tab. The report
        needs no CSS; the Brevan Howard report stylesheet is applied here.
        """
        workspace = self._resolve_workspace(tab_ref)
        workspace.load_summary_html(html)
        return workspace

    def load_final_judge(self, run_dir, tab_ref=None):
        """Load the final judge's UI fragment into a dataset tab.

        ``run_dir`` is the directory containing ``final_judge.fragment.html``
        and ``artifact.json``.  This is the handoff used by the report owner:
        the judge renders the HTML, the dashboard supplies the styling.
        """
        workspace = self._resolve_workspace(tab_ref)
        workspace.load_final_judge_report(run_dir)
        return workspace

    def load_summary_html(self, html, tab_index=None):
        """Backwards-compatible alias for :meth:`set_report`."""
        return self.set_report(html, tab_index)

    def current_workspace(self):
        return self.tabs.currentWidget()

    def workspaces(self):
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def __getattr__(self, name):
        # Convenience passthrough so dashboard.<field> targets the active tab.
        tabs = self.__dict__.get("tabs")
        if tabs is not None:
            workspace = tabs.currentWidget()
            if workspace is not None and hasattr(workspace, name):
                return getattr(workspace, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = QuantDashboard(auto_run_pipeline=True)
    dashboard.show()
    sys.exit(app.exec())
