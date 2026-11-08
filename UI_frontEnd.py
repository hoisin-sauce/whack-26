import os
import sys

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
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

QTextEdit {
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

QTextEdit#summaryOutput {
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
"""


class DragDropTextEdit(QTextEdit):
    """Custom QTextEdit widget that accepts dropped files (.csv, .xls, .xlsx)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.parent_dashboard = parent

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


class QuantDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.dropped_file_path = None
        self.loaded_dataframe = None
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
        header_layout.setContentsMargins(34, 22, 34, 24)
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
            "Configure your constraints, drop a dataset, and run the master validation pass."
        )
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setWordWrap(True)
        header_layout.addWidget(subtitle_label)

        main_layout.addWidget(header_card)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(32, 22, 32, 22)
        content_layout.setSpacing(16)

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

        content_layout.addLayout(config_row)

        # --- RUN BUTTON ---
        self.run_btn = QPushButton("Run Master Validation Engine")
        self.run_btn.setObjectName("runButton")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.setMinimumHeight(50)
        self.run_btn.clicked.connect(self.run_validation)
        content_layout.addWidget(self.run_btn)

        # --- SPLIT LAYOUT FOR LOGS AND SUMMARY REPORT ---
        results_layout = QHBoxLayout()
        results_layout.setSpacing(18)

        # Left Box: Activity & File Drop Log
        log_group = QGroupBox("03 / DATASET DROP ZONE && ACTIVITY LOG")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(16, 14, 16, 16)
        log_layout.setSpacing(10)

        contract_note = QLabel(TARGET_CONTRACT)
        contract_note.setObjectName("hintLabel")
        contract_note.setWordWrap(True)
        log_layout.addWidget(contract_note)

        self.log_output = DragDropTextEdit(self)
        self.log_output.setObjectName("logOutput")
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText(
            "Drop your dataset (.csv, .xls, .xlsx) anywhere in this box..."
        )
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        results_layout.addWidget(log_group, 3)

        # Right Box: Dedicated Summary Report Area
        summary_group = QGroupBox("04 / VALIDATION SUMMARY REPORT")
        summary_layout = QVBoxLayout()
        summary_layout.setContentsMargins(16, 14, 16, 16)
        self.summary_output = QTextEdit()
        self.summary_output.setObjectName("summaryOutput")
        self.summary_output.setReadOnly(True)
        self.summary_output.setPlaceholderText(
            "Validation metrics will appear here after a run."
        )
        summary_layout.addWidget(self.summary_output)
        summary_group.setLayout(summary_layout)
        results_layout.addWidget(summary_group, 2)

        content_layout.addLayout(results_layout, 1)

        main_layout.addLayout(content_layout, 1)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = QuantDashboard()
    dashboard.show()
    sys.exit(app.exec())
