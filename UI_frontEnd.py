import os
import sys

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication,
    QFormLayout,
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
        self.resize(950, 850)

        main_layout = QVBoxLayout()

        # --- Title Header ---
        self.title_label = QLabel("Brevan Howard Track: Autonomous Validation Engine")
        self.title_label.setStyleSheet(
            "font-weight: bold; font-size: 14pt; margin-bottom: 5px;"
        )
        main_layout.addWidget(self.title_label)

        # --- CONFIGURATION & METADATA GROUP ---
        config_group = QGroupBox("User Metadata, Dataset Description & Test Restrictions")
        form_layout = QFormLayout()

        self.strategy_name_input = QLineEdit()
        self.strategy_name_input.setPlaceholderText("e.g. Momentum_Alpha_v1")

        self.data_desc_input = QLineEdit()
        self.data_desc_input.setPlaceholderText(
            "e.g. Daily S&P 500 returns with macroeconomic indicators"
        )

        self.periods_input = QLineEdit()
        self.periods_input.setPlaceholderText("e.g. 252")

        self.min_sharpe_input = QLineEdit()
        self.min_sharpe_input.setPlaceholderText("e.g. 0.95 (DSR threshold)")

        self.max_decay_input = QLineEdit()
        self.max_decay_input.setPlaceholderText("e.g. 0.25 (clamped to 0.15-0.50)")

        form_layout.addRow("Strategy Name:", self.strategy_name_input)
        form_layout.addRow("Dataset Description:", self.data_desc_input)
        form_layout.addRow("Periods Per Year:", self.periods_input)
        form_layout.addRow("Min DSR Threshold:", self.min_sharpe_input)
        form_layout.addRow("Max OOS Decay Tolerance:", self.max_decay_input)

        config_group.setLayout(form_layout)
        main_layout.addWidget(config_group)

        # --- RUN BUTTON ---
        self.run_btn = QPushButton("Run Master Validation Engine")
        self.run_btn.setStyleSheet(
            "padding: 12px; font-weight: bold; background-color: #2b5b84; "
            "color: white; font-size: 11pt;"
        )
        self.run_btn.clicked.connect(self.run_validation)
        main_layout.addWidget(self.run_btn)

        # --- SPLIT LAYOUT FOR LOGS AND SUMMARY REPORT ---
        results_layout = QHBoxLayout()

        # Left Box: Activity & File Drop Log
        log_group = QGroupBox("Dataset Drop Zone & Activity Log")
        log_layout = QVBoxLayout()
        self.log_output = DragDropTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText(
            "Drop your dataset (.csv, .xls, .xlsx) anywhere in this box...\n"
            + TARGET_CONTRACT
            + "\nFill out the description, metadata, and restrictions above before running!"
        )
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        results_layout.addWidget(log_group)

        # Right Box: Dedicated Summary Report Area
        summary_group = QGroupBox("Validation Summary Report")
        summary_layout = QVBoxLayout()
        self.summary_output = QTextEdit()
        self.summary_output.setReadOnly(True)
        self.summary_output.setPlaceholderText(
            "Final output metrics returned by validation_engine will appear here..."
        )
        self.summary_output.setStyleSheet(
            """
            background-color: #ffffff;
            color: #222222;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11pt;
        """
        )
        summary_layout.addWidget(self.summary_output)
        summary_group.setLayout(summary_layout)
        results_layout.addWidget(summary_group)

        main_layout.addLayout(results_layout)
        self.setLayout(main_layout)

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
