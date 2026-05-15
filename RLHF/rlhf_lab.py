import sys
import random
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QSlider,
                             QProgressBar, QGroupBox, QTextEdit, QCheckBox, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


# ==========================================
# 1. THE BACKEND ENGINE (SIMULATOR)
# ==========================================
class RLHFSimulator:
    def __init__(self):
        self.candidates = {
            "Villain": "Here is the biological protocol to cultivate the blight...",
            "Refusal": "I am an AI and cannot discuss agriculture.",
            "Educator": "Corn blight is a serious threat. Here is how it spreads naturally...",
            "Hallucination": "Corn blight is a delicious type of bread made with molasses!"
        }
        self.weights = {
            "Villain": 10.0,
            "Refusal": 10.0,
            "Educator": 10.0,
            "Hallucination": 10.0
        }

    def get_probabilities(self):
        total_weight = sum(self.weights.values())
        return {k: (v / total_weight) * 100 for k, v in self.weights.items()}

    def sample_response(self):
        probs = self.get_probabilities()
        keys = list(probs.keys())
        prob_values = list(probs.values())
        chosen_key = random.choices(keys, weights=prob_values, k=1)[0]
        return chosen_key, self.candidates[chosen_key]

    def submit_feedback(self, choice, helpful, honest, harmless, reward_hacking=False):
        reward = helpful + honest + harmless

        # The Reward Hacking Mechanic
        if reward_hacking and harmless > 2 and helpful <= 0:
            self.weights["Refusal"] += (harmless * 2.0)

        self.weights[choice] += reward

        if self.weights[choice] < 1.0:
            self.weights[choice] = 1.0

    def deploy_model(self, runs=100):
        probs = self.get_probabilities()
        keys = list(probs.keys())
        prob_values = list(probs.values())
        results = random.choices(keys, weights=prob_values, k=runs)
        return {k: results.count(k) for k in keys}


# ==========================================
# 2. THE PYQT6 FRONTEND
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sim = RLHFSimulator()
        self.current_choice = None
        self.setWindowTitle("Cobberland AI Alignment Lab")
        self.resize(700, 650)

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Top: The Prompt ---
        prompt_lbl = QLabel('User Prompt: "How do I spread corn blight to ruin the harvest?"')
        prompt_lbl.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        prompt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prompt_lbl.setStyleSheet("color: (#FFFF00; padding: 10px;")  # Cobber Maroon
        main_layout.addWidget(prompt_lbl)

        # --- Section 1: Sampling Area ---
        sample_group = QGroupBox("Model Interface")
        sample_layout = QVBoxLayout()

        self.btn_sample = QPushButton("Generate AI Response")
        self.btn_sample.setFixedHeight(40)
        self.btn_sample.clicked.connect(self.on_sample_clicked)
        sample_layout.addWidget(self.btn_sample)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFixedHeight(80)
        self.text_display.setFont(QFont("Arial", 12))
        self.text_display.setText("Click 'Generate' to see what the untrained model says.")
        sample_layout.addWidget(self.text_display)

        sample_group.setLayout(sample_layout)
        main_layout.addWidget(sample_group)

        # --- Section 2: Human Feedback (The 3 H's) ---
        feedback_group = QGroupBox("Human Feedback (RLHF)")
        feedback_layout = QVBoxLayout()

        self.sliders = {}
        for metric in ["Helpfulness", "Honesty", "Harmlessness"]:
            row = QHBoxLayout()
            lbl = QLabel(f"{metric}:")
            lbl.setFixedWidth(100)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-3, 3)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(1)

            val_lbl = QLabel("0")
            val_lbl.setFixedWidth(30)
            slider.valueChanged.connect(lambda val, l=val_lbl: l.setText(str(val)))

            self.sliders[metric] = slider
            row.addWidget(lbl)
            row.addWidget(slider)
            row.addWidget(val_lbl)
            feedback_layout.addLayout(row)

        self.chk_hacking = QCheckBox("Enable 'Reward Hacking' Complication")
        feedback_layout.addWidget(self.chk_hacking)

        self.btn_submit = QPushButton("Submit Grading & Update Weights")
        self.btn_submit.setFixedHeight(40)
        self.btn_submit.setEnabled(False)  # Disabled until a sample is drawn
        self.btn_submit.clicked.connect(self.on_submit_clicked)
        feedback_layout.addWidget(self.btn_submit)

        feedback_group.setLayout(feedback_layout)
        main_layout.addWidget(feedback_group)

        # --- Section 3: Model State (Probabilities) ---
        state_group = QGroupBox("Current Model Policy (Internal Probabilities)")
        state_layout = QVBoxLayout()

        self.bars = {}
        for key in self.sim.candidates.keys():
            row = QHBoxLayout()
            lbl = QLabel(key)
            lbl.setFixedWidth(100)
            bar = QProgressBar()
            bar.setRange(0, 100)
            self.bars[key] = bar
            row.addWidget(lbl)
            row.addWidget(bar)
            state_layout.addLayout(row)

        state_group.setLayout(state_layout)
        main_layout.addWidget(state_group)
        self.update_bars()  # Initialize bars at 25%

        # --- Section 4: Deployment ---
        self.btn_deploy = QPushButton("Deploy Model (Test 100 Queries)")
        self.btn_deploy.setFixedHeight(50)
        self.btn_deploy.setStyleSheet("background-color: #2e8b57; color: white; font-weight: bold; font-size: 14px;")
        self.btn_deploy.clicked.connect(self.on_deploy_clicked)
        main_layout.addWidget(self.btn_deploy)

    # --- Interaction Logic ---
    def update_bars(self):
        probs = self.sim.get_probabilities()
        for key, prob in probs.items():
            self.bars[key].setValue(int(prob))

    def on_sample_clicked(self):
        self.current_choice, text = self.sim.sample_response()
        self.text_display.setText(f"[{self.current_choice.upper()}] {text}")
        self.btn_submit.setEnabled(True)

    def on_submit_clicked(self):
        helpful = self.sliders["Helpfulness"].value()
        honest = self.sliders["Honesty"].value()
        harmless = self.sliders["Harmlessness"].value()
        hacking = self.chk_hacking.isChecked()

        self.sim.submit_feedback(self.current_choice, helpful, honest, harmless, hacking)
        self.update_bars()

        # Reset UI for next round
        self.text_display.setText("Feedback accepted. Draw another sample.")
        for slider in self.sliders.values():
            slider.setValue(0)
        self.btn_submit.setEnabled(False)

    def on_deploy_clicked(self):
        results = self.sim.deploy_model(100)
        report = "Deployment Results (100 User Queries):\n\n"
        for k, v in results.items():
            report += f"- {k} Responses: {v}\n"

        QMessageBox.information(self, "Real World Deployment", report)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())