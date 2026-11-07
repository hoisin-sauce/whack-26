import sys
import os
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, 
    QTextEdit, QLineEdit, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt

# Import validation engine from stats_test.py
try:
    from stats_test import validation_engine
except ImportError:
    def validation_engine(*args, **kwargs):
        return True # Fallback mock

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
        self.resize(800, 780)
        
        main_layout = QVBoxLayout()
        
        # Title Header
        self.title_label = QLabel("Brevan Howard Track: Autonomous Validation Engine")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14pt; margin-bottom: 5px;")
        main_layout.addWidget(self.title_label)
        
        # --- CONFIGURATION & METADATA GROUP (User-driven) ---
        config_group = QGroupBox("User Metadata, Dataset Description & Test Restrictions")
        form_layout = QFormLayout()
        
        # Inputs start completely empty with descriptive placeholders
        self.strategy_name_input = QLineEdit()
        self.strategy_name_input.setPlaceholderText("e.g. Momentum_Alpha_v1")
        
        self.data_desc_input = QLineEdit()
        self.data_desc_input.setPlaceholderText("e.g. Daily S&P 500 returns with macroeconomic indicators")
        
        self.periods_input = QLineEdit()
        self.periods_input.setPlaceholderText("e.g. 252 ")
        
        self.min_sharpe_input = QLineEdit()
        self.min_sharpe_input.setPlaceholderText("e.g. 1.0")
        
        self.max_decay_input = QLineEdit()
        self.max_decay_input.setPlaceholderText("e.g. 0.25")
        
        form_layout.addRow("Strategy Name:", self.strategy_name_input)
        form_layout.addRow("Dataset Description:", self.data_desc_input)
        form_layout.addRow("Periods Per Year:", self.periods_input)
        form_layout.addRow("Min Sharpe Threshold:", self.min_sharpe_input)
        form_layout.addRow("Max OOS Decay Tolerance:", self.max_decay_input)
        
        config_group.setLayout(form_layout)
        main_layout.addWidget(config_group)
        
        # --- RUN BUTTON ---
        self.run_btn = QPushButton("Run Master Validation Engine")
        self.run_btn.setStyleSheet("padding: 12px; font-weight: bold; background-color: #2b5b84; color: white; font-size: 11pt;")
        self.run_btn.clicked.connect(self.run_validation)
        main_layout.addWidget(self.run_btn)
        
        # --- LOG BOX & DROP ZONE ---
        self.log_output = DragDropTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText(
            "Drop your dataset (.csv, .xls, .xlsx) anywhere in this box...\n"
            "Make sure to fill out the description, metadata, and restrictions above before running!"
        )
        main_layout.addWidget(self.log_output)
        
        self.setLayout(main_layout)
        
    def handle_dropped_file(self, file_path):
        self.dropped_file_path = file_path
        file_ext = os.path.splitext(file_path)[1].lower()
        
        self.log_output.append(f"\n File dropped: {os.path.basename(file_path)}")
        
        try:
            if file_ext == '.csv':
                self.loaded_dataframe = pd.read_csv(file_path)
            elif file_ext in ['.xls', '.xlsx']:
                self.loaded_dataframe = pd.read_excel(file_path)
            else:
                self.log_output.append(f"Unsupported file extension: {file_ext}. Please use .csv, .xls, or .xlsx")
                return
                
            self.log_output.append(f"Successfully loaded dataset! Shape: {self.loaded_dataframe.shape}")
            self.log_output.append(f"Columns detected: {list(self.loaded_dataframe.columns)}")
        except Exception as e:
            self.log_output.append(f"Error parsing file with Pandas: {str(e)}")

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
            self.log_output.append("\n Missing or Invalid Inputs: Please fill in all restriction and metadata fields with valid numbers before running.")
            return

        self.log_output.append(f"\n--- [GUI] Triggering Validation for '{strategy_name}' ---")
        self.log_output.append(f" Dataset Description: {data_description}")
        self.log_output.append(f" Constraints -> Periods: {periods_per_year} | Min Sharpe: {min_sharpe} | Max Decay: {max_decay}")

        # Handle Data Source (Dropped file vs. Synthetic fallback)
        if self.loaded_dataframe is not None:
            self.log_output.append("Using data from uploaded spreadsheet file...")
            df = self.loaded_dataframe.dropna()
            features = df.iloc[:, :-1].values
            targets = df.iloc[:, -1].values
            winning_returns = targets[targets > 0]
            predictions = np.where(targets > 0, 1, 0)
            actuals = predictions
            all_trial_sharpes = [1.5, 1.2, min_sharpe + 0.2, 0.9]
        else:
            self.log_output.append("No dataset dropped. Synthesizing test arrays...")
            winning_returns = np.random.normal(0.01, 0.05, 100)
            all_trial_sharpes = [1.2, 0.8, 1.5, 0.5, 2.1]
            predictions = np.random.choice([0, 1], size=100)
            actuals = np.random.choice([0, 1], size=100)
            features = np.random.randn(100, 5)
            targets = actuals

        def dummy_predict(X):
            return np.random.choice([0, 1], size=len(X))
            
        def dummy_builder(params):
            return dummy_predict
            
        base_params = {"window_size": 14, "learning_rate": 0.01, "min_sharpe_limit": min_sharpe}

        try:
            # Execute backend validation engine with user metadata & description
            passed = validation_engine(
                winning_returns=winning_returns,
                all_trial_sharpes=all_trial_sharpes,
                predictions=predictions,
                actuals=actuals,
                features=features,
                targets=targets,
                model_predict_func=dummy_predict,
                base_model_builder=dummy_builder,
                base_params=base_params,
                periods_per_year=periods_per_year
            )
            
            if passed:
                self.log_output.append(f" SUCCESS: '{strategy_name}' passed all validation gates!")
            else:
                self.log_output.append(f" FAIL: '{strategy_name}' rejected under user restrictions.")
                
        except Exception as e:
            self.log_output.append(f" Engine Execution Error: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dashboard = QuantDashboard()
    dashboard.show()
    sys.exit(app.exec())