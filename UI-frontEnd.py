import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit


from stats_test import validation_engine

class QuantDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Autonomous Quant Researcher - Dashboard")
        self.resize(600, 400)
        
        layout = QVBoxLayout()
        
        self.title_label = QLabel("🚀 Brevan Howard Track: Autonomous Validation Engine")
        layout.addWidget(self.title_label)
        
        self.run_btn = QPushButton("Run Master Validation Engine")
        self.run_btn.clicked.connect(self.run_validation)
        layout.addWidget(self.run_btn)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Logs will appear here when the agent runs...")
        layout.addWidget(self.log_output)
        
        self.setLayout(layout)
        
    def run_validation(self):
        self.log_output.append("--- [GUI] Triggering Validation Engine ---")
        
        # 1. Mock up test inputs (or pull from your agent's live data pipeline)
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
            
        base_params = {"window_size": 14, "learning_rate": 0.01}
        periods_per_year = 252  # Injected dynamically from knowledge graph context

        # 2. Run the master engine
        try:
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
            
            # 3. Print the final result to the PyQt log text box
            if passed:
                self.log_output.append("🟢 SUCCESS: Model passed all 4 validation layers!")
            else:
                self.log_output.append("🔴 FAIL: Model rejected by validation gate.")
                
        except Exception as e:
            self.log_output.append(f"⚠️ Error executing engine: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dashboard = QuantDashboard()
    dashboard.show()
    sys.exit(app.exec())