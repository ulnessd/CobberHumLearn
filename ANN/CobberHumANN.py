# CobberHumANN.py
# A PyQt6 application for exploring single neurons and full neural networks
# with humanities-facing population time-series data.
#
# Direct humanities adaptation of CobberNeuron.py:
#   - Temperature becomes Year
#   - Vapor pressure becomes Total population
#   - Chemistry language becomes humanistic / historical-demographic language
#
# Data source:
#   World Bank World Development Indicators, SP.POP.TOTL (Total population)
#   CSV files supplied by user and hard-coded here for classroom portability.

import sys
import numpy as np
from typing import List, Tuple, Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QTabWidget, QPushButton, QSlider, QFormLayout,
    QSpinBox, QLineEdit, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Hard-coded classroom data
# ---------------------------------------------------------------------

POPULATION_DATASETS: Dict[str, Dict[str, object]] = {
    "Syria population, 2004-2024": {
        "country": "Syrian Arab Republic",
        "country_code": "SYR",
        "indicator_code": "SP.POP.TOTL",
        "indicator_name": "Total population",
        "units": "people",
        "note": (
            "This series rises, falls sharply after the onset of the Syrian conflict, "
            "and then rises again. It is a powerful example of why a numerical curve "
            "needs historical interpretation."
        ),
        "data": [(2004, 18336875), (2005, 18814091), (2006, 19637022), (2007, 20884125), (2008, 21636564), (2009, 21976969), (2010, 22482421), (2011, 22878098), (2012, 22759763), (2013, 21666917), (2014, 20273659), (2015, 19424618), (2016, 19193834), (2017, 19224668), (2018, 19577845), (2019, 20353534), (2020, 21049429), (2021, 21628839), (2022, 22462173), (2023, 23594623), (2024, 24672760)],
    },
    "Sudan population, 2004-2024": {
        "country": "Sudan",
        "country_code": "SDN",
        "indicator_code": "SP.POP.TOTL",
        "indicator_name": "Total population",
        "units": "people",
        "note": (
            "This series is much smoother. The population trend is fairly steady even "
            "though the historical and political situation has been anything but simple."
        ),
        "data": [(2004, 30556637), (2005, 31262444), (2006, 31992435), (2007, 32764135), (2008, 33623980), (2009, 34569113), (2010, 35414399), (2011, 36140806), (2012, 36923178), (2013, 37785849), (2014, 38823318), (2015, 40024431), (2016, 41259892), (2017, 42714306), (2018, 44230596), (2019, 45548175), (2020, 46789231), (2021, 48066924), (2022, 49383346), (2023, 50042791), (2024, 50448963)],
    },
}


# ---------------------------------------------------------------------
# Core neural-network functions
# ---------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Clip to avoid overflow warnings for extreme slider settings.
    x = np.clip(x, -60, 60)
    return 1 / (1 + np.exp(-x))


def calculate_neuron_output(x_input: np.ndarray, weight: float, bias: float) -> np.ndarray:
    return sigmoid(weight * x_input + bias)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def draw_mid_arrow(ax, x0, y0, x1, y1, frac=0.18, color='grey', lw=1, alpha=0.8, head=12):
    ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=lw)

    dx, dy = (x1 - x0), (y1 - y0)
    L = (dx**2 + dy**2) ** 0.5
    if L == 0:
        return

    ux, uy = dx / L, dy / L
    midx, midy = (x0 + x1) / 2, (y0 + y1) / 2
    half = (frac * L) / 2
    xa, ya = midx - ux * half, midy - uy * half
    xb, yb = midx + ux * half, midy + uy * half

    arrow = FancyArrowPatch(
        (xa, ya), (xb, yb),
        arrowstyle='-|>',
        mutation_scale=head,
        linewidth=max(0.5, lw-1),
        color=color,
        alpha=alpha
    )
    ax.add_patch(arrow)


def draw_mid_arrow_on_line(ax, x0, x1, y, frac=0.18, color='grey', lw=1, alpha=0.8, head=12):
    ax.plot([x0, x1], [y, y], color=color, alpha=alpha, linewidth=lw)

    mid = 0.5 * (x0 + x1)
    half = 0.5 * frac * (x1 - x0)
    xa, xb = mid - half, mid + half

    arrow = FancyArrowPatch(
        (xa, y), (xb, y),
        arrowstyle='-|>',
        mutation_scale=head,
        linewidth=lw,
        color=color,
        alpha=alpha
    )
    ax.add_patch(arrow)


def y_nudge_pixels(ax, y, pixels):
    _, y_disp = ax.transData.transform((0, y))
    y2 = ax.transData.inverted().transform((0, y_disp + pixels))[1]
    return y2


def population_to_millions(values: np.ndarray) -> np.ndarray:
    return values / 1_000_000.0


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)


class SingleNeuronTab(QWidget):
    def __init__(self, years, population, dataset_label, dataset_note):
        super().__init__()
        self.years = years
        self.population = population
        self.population_millions = population_to_millions(population)
        self.dataset_label = dataset_label
        self.dataset_note = dataset_note

        self.scaled_years, self.year_min, self.year_max = self.normalize(self.years)
        self.scaled_population, self.population_min, self.population_max = self.normalize(self.population_millions)

        layout = QHBoxLayout(self)

        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_panel)

        schematic_label = QLabel("<h3>The Single Neuron Model</h3>")
        self.schematic_canvas = MplCanvas(self, width=3, height=2, dpi=70)

        schematic_text = QLabel(
            "<p>Your goal is to adjust the Weight and Bias to find the lowest MSE. "
            "The single neuron can learn only a simple S-shaped response, so it will "
            "struggle with curves that rise, fall, and rise again.</p>"
        )
        schematic_text.setWordWrap(True)

        dataset_text = QLabel(f"<p><b>Dataset:</b> {dataset_label}<br>{dataset_note}</p>")
        dataset_text.setWordWrap(True)

        form_layout = QFormLayout()
        self.weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.bias_slider = QSlider(Qt.Orientation.Horizontal)

        self.mse_label = QLabel("<b>MSE Score:</b> N/A")
        self.mse_label.setStyleSheet("font-size: 16px;")

        self.weight_slider.setRange(-100, 100)
        self.weight_slider.setValue(50)
        self.bias_slider.setRange(-50, 50)
        self.bias_slider.setValue(-25)

        form_layout.addRow("Weight (w):", self.weight_slider)
        form_layout.addRow("Bias (b):", self.bias_slider)

        controls_layout.addWidget(schematic_label)
        controls_layout.addWidget(self.schematic_canvas)
        controls_layout.addWidget(schematic_text)
        controls_layout.addWidget(dataset_text)
        controls_layout.addStretch(1)
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.mse_label)
        controls_layout.addStretch(2)

        plot_panel = QFrame()
        plot_panel.setFrameShape(QFrame.Shape.StyledPanel)
        plot_layout = QVBoxLayout(plot_panel)

        self.canvas = MplCanvas(self, width=7, height=6, dpi=100)
        plot_layout.addWidget(self.canvas)

        layout.addWidget(controls_panel, 1)
        layout.addWidget(plot_panel, 2)

        self.weight_slider.valueChanged.connect(self.update_prediction)
        self.bias_slider.valueChanged.connect(self.update_prediction)
        self.update_prediction()

    def normalize(self, data):
        min_val, max_val = np.min(data), np.max(data)
        if max_val == min_val:
            return data, min_val, max_val
        return (data - min_val) / (max_val - min_val), min_val, max_val

    def denormalize(self, scaled_data, min_val, max_val):
        return scaled_data * (max_val - min_val) + min_val

    def update_prediction(self):
        weight = self.weight_slider.value() / 10.0
        bias = self.bias_slider.value() / 10.0

        self.draw_schematic(weight, bias)

        predicted_scaled = calculate_neuron_output(self.scaled_years, weight, bias)
        mse = calculate_mse(self.scaled_population, predicted_scaled)
        self.mse_label.setText(f"<b>MSE Score:</b> {mse:.4f}")

        predicted_millions = self.denormalize(
            predicted_scaled,
            self.population_min,
            self.population_max
        )

        ax = self.canvas.axes
        ax.clear()

        ax.scatter(self.years, self.population_millions, label='World Bank WDI data')
        ax.plot(
            self.years,
            predicted_millions,
            'r--',
            label=f'Single Neuron Prediction (w={weight:.1f}, b={bias:.1f})'
        )

        ax.set_title(f"Total Population over Time\n{self.dataset_label}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Population (millions)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')

        residuals = self.population_millions - predicted_millions
        inset_ax = ax.inset_axes([0.07, 0.55, 0.38, 0.38])
        inset_ax.axhline(0.0, linestyle='--', alpha=0.7)
        inset_ax.vlines(self.years, 0.0, residuals, alpha=0.6)
        inset_ax.scatter(self.years, residuals, s=18)

        inset_ax.set_title("Residuals", fontsize=9)
        inset_ax.set_xlabel("Year", fontsize=8)
        inset_ax.set_ylabel("millions", fontsize=8)
        inset_ax.tick_params(labelsize=8)
        inset_ax.grid(True, linestyle=':', alpha=0.4)

        self.canvas.draw()

    def draw_schematic(self, weight, bias):
        ax = self.schematic_canvas.axes
        ax.clear()

        input_y, hidden_y, output_y = 0.5, 0.5, 0.5

        draw_mid_arrow(ax, 0.94, input_y, 1.02, hidden_y, color='grey', lw=2, alpha=0.8, head=14)
        draw_mid_arrow(ax, -0.03, input_y, 0.07, hidden_y, color='grey', lw=2, alpha=0.8, head=14)
        draw_mid_arrow(ax, 0.1, input_y, 0.5, hidden_y, color='grey', lw=2, alpha=0.8, head=14)
        draw_mid_arrow(ax, 0.5, hidden_y, 0.9, output_y, color='grey', lw=2, alpha=0.8, head=14)

        ax.text(0.37, 0.62, f'w={weight:.1f}', ha='center', va='center', fontsize=16,
                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))
        ax.text(0.53, 0.25, f'b={bias:.1f}', ha='center', va='center', fontsize=16,
                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

        ax.scatter([0.1], [input_y], s=1000, c='#d3d3d3', marker='s', zorder=5)
        ax.text(-0.05, output_y, "Year", ha='center', va='center', fontsize=14)
        ax.text(0.1, input_y - 0.4, "Input\nNeuron", ha='center', va='center', fontsize=16)

        ax.scatter([0.5], [hidden_y], s=1000, c='#6C1D45', marker='s', zorder=5)

        ax.scatter([0.9], [output_y], s=1000, c='#3D3D3D', marker='s', zorder=5)
        ax.text(0.9, output_y - 0.4, "Output\nNeuron", ha='center', va='center', fontsize=16)
        ax.text(1.07, output_y, "Pop.", ha='center', va='center', fontsize=14)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.schematic_canvas.draw()


class NeuralNetworkTab(QWidget):
    def __init__(self, years, population, dataset_label, dataset_note, main_window):
        super().__init__()

        self.main_window = main_window
        self.years = years
        self.population = population
        self.population_millions = population_to_millions(population)
        self.dataset_label = dataset_label
        self.dataset_note = dataset_note

        self.model = None
        self.scaler_X = None
        self.scaler_y = None

        layout = QHBoxLayout(self)

        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_panel)

        form_layout = QFormLayout()

        self.neurons_spinner = QSpinBox()
        self.neurons_spinner.setRange(2, 24)
        self.neurons_spinner.setValue(8)

        self.learning_rate_input = QLineEdit("0.005")
        self.epochs_input = QLineEdit("10000")
        self.train_button = QPushButton("Train Network")

        form_layout.addRow("Neurons in Hidden Layer:", self.neurons_spinner)
        form_layout.addRow("Learning Rate:", self.learning_rate_input)
        form_layout.addRow("Training Cycles (Epochs):", self.epochs_input)

        controls_layout.addWidget(QLabel("<h3>Network Configuration</h3>"))
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.train_button)

        note = QLabel(
            f"<p><b>Dataset:</b> {dataset_label}<br>{dataset_note}</p>"
            "<p>The network learns a numerical curve. It does not, by itself, explain "
            "the historical causes behind that curve.</p>"
        )
        note.setWordWrap(True)
        controls_layout.addWidget(note)
        controls_layout.addStretch()

        schematic_panel = QFrame()
        schematic_panel.setFrameShape(QFrame.Shape.StyledPanel)
        schematic_layout = QVBoxLayout(schematic_panel)
        schematic_layout.addWidget(QLabel("<b>Network Architecture</b>"))

        self.schematic_canvas = MplCanvas(self, width=7, height=7)
        schematic_layout.addWidget(self.schematic_canvas)

        results_panel = QTabWidget()

        self.loss_canvas = MplCanvas(self)
        results_panel.addTab(self.loss_canvas, "Training Progress")

        self.fit_canvas = MplCanvas(self)
        results_panel.addTab(self.fit_canvas, "Final Model Fit")

        self.resid_tab = QWidget()
        resid_layout = QVBoxLayout(self.resid_tab)
        self.resid_canvas = MplCanvas(self)

        self.mae_label = QLabel("MAE: -")
        self.mse_label = QLabel("MSE: -")

        mae_font = QFont(self.font())
        mae_font.setPointSize(16)
        self.mae_label.setFont(mae_font)

        mse_font = QFont(self.font())
        mse_font.setPointSize(16)
        self.mse_label.setFont(mse_font)

        resid_layout.setSpacing(6)
        resid_layout.setContentsMargins(6, 6, 6, 6)

        labels_layout = QHBoxLayout()
        labels_layout.setSpacing(12)
        labels_layout.addStretch(1)
        labels_layout.addWidget(self.mae_label)
        labels_layout.addWidget(self.mse_label)
        labels_layout.addStretch(1)

        resid_layout.addWidget(self.resid_canvas)
        resid_layout.addLayout(labels_layout)

        results_panel.addTab(self.resid_tab, "Residuals")

        layout.addWidget(controls_panel, 3)
        layout.addWidget(schematic_panel, 7)
        layout.addWidget(results_panel, 7)

        self.train_button.clicked.connect(self.train_network)
        self.neurons_spinner.valueChanged.connect(self.update_schematic)

        self.setup_initial_plots()
        self.update_schematic()

    def train_network(self):
        try:
            hidden_neurons = self.neurons_spinner.value()
            learning_rate = float(self.learning_rate_input.text())
            epochs = int(self.epochs_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please ensure all parameters are valid numbers.")
            return

        X = self.years.reshape(-1, 1)
        y = self.population_millions

        self.train_button.setText("Training...")
        self.train_button.setEnabled(False)
        QApplication.processEvents()

        scaler_X = StandardScaler().fit(X)
        scaler_y = StandardScaler().fit(y.reshape(-1, 1))

        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_neurons,),
            activation='relu',
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=epochs,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=500
        )

        self.model.fit(X_scaled, y_scaled)

        self.train_button.setText("Train Network")
        self.train_button.setEnabled(True)

        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        self.update_schematic()

        self.loss_canvas.axes.clear()
        self.loss_canvas.axes.plot(self.model.loss_curve_)
        self.loss_canvas.axes.set_title("Loss vs. Epoch")
        self.loss_canvas.axes.set_xlabel("Epoch")
        self.loss_canvas.axes.set_ylabel("Mean Squared Error (Loss)")
        self.loss_canvas.axes.grid(True)
        self.loss_canvas.draw()

        x_smooth = np.linspace(self.years.min(), self.years.max(), 300).reshape(-1, 1)
        x_smooth_scaled = scaler_X.transform(x_smooth)
        y_pred_scaled = self.model.predict(x_smooth_scaled)
        y_pred_millions = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        self.fit_canvas.axes.clear()
        self.fit_canvas.axes.scatter(self.years, self.population_millions, label='World Bank WDI data')
        self.fit_canvas.axes.plot(x_smooth.ravel(), y_pred_millions, 'r-', label='Neural Network Fit', linewidth=2)
        self.fit_canvas.axes.set_title(f"Population Data & Model Fit\n{self.dataset_label}")
        self.fit_canvas.axes.set_xlabel("Year")
        self.fit_canvas.axes.set_ylabel("Population (millions)")
        self.fit_canvas.axes.legend()
        self.fit_canvas.axes.grid(True)
        self.fit_canvas.draw()

        y_train_pred_scaled = self.model.predict(X_scaled)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

        residuals = self.population_millions - y_train_pred
        mae = float(np.mean(np.abs(residuals)))
        mse = float(np.mean(residuals ** 2))

        axr = self.resid_canvas.axes
        axr.clear()
        axr.axhline(0.0, linestyle='--', alpha=0.7)
        axr.vlines(self.years, 0.0, residuals, alpha=0.6)
        axr.scatter(self.years, residuals, s=25)

        axr.set_title("Residuals (Observed − Predicted)")
        axr.set_xlabel("Year")
        axr.set_ylabel("Residual (millions)")
        axr.grid(True, linestyle=':', alpha=0.5)
        self.resid_canvas.draw()

        self.mae_label.setText(f"MAE: {mae:.4f} million")
        self.mse_label.setText(f"MSE: {mse:.4f} million²")

    def update_schematic(self):
        ax = self.schematic_canvas.axes
        ax.clear()

        num_neurons = self.neurons_spinner.value()
        scale_font_size = max(7, 20 - num_neurons)

        input_y = [0.5]
        hidden_y = np.linspace(0.1, 0.9, num_neurons)
        output_y = [0.5]

        if self.model and self.model.hidden_layer_sizes[0] == num_neurons:
            input_weights = self.model.coefs_[0]
            hidden_biases = self.model.intercepts_[0]
            output_weights = self.model.coefs_[1]

            for i, hy in enumerate(hidden_y):
                draw_mid_arrow(ax, 0.1, input_y[0], 0.5, hy, color='grey', lw=2, alpha=0.8, head=14)
                ax.text(
                    0.3, (input_y[0] + hy) / 2,
                    f'w={input_weights[0][i]:.1f}',
                    ha='center', va='center', fontsize=scale_font_size,
                    bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
                )
                ax.text(
                    0.5, (input_y[0] + hy) - .545,
                    f'b={hidden_biases[i]:.1f}',
                    ha='center', va='center', fontsize=scale_font_size,
                    bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
                )

            for i, hy in enumerate(hidden_y):
                draw_mid_arrow(ax, 0.5, hy, 0.9, output_y[0], color='grey', lw=2, alpha=0.8, head=14)
                ax.text(
                    0.7, (hy + output_y[0]) / 2,
                    f'w={output_weights[i][0]:.1f}',
                    ha='center', va='center', fontsize=scale_font_size,
                    bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
                )
        else:
            for hy in hidden_y:
                draw_mid_arrow(ax, 0.1, input_y[0], 0.5, hy, color='grey', lw=1.8, alpha=0.8, head=12)
                draw_mid_arrow(ax, 0.5, hy, 0.9, output_y[0], color='grey', lw=1.8, alpha=0.8, head=12)

        yin = y_nudge_pixels(ax, input_y[0], +1)
        yout = y_nudge_pixels(ax, output_y[0], +1)

        draw_mid_arrow_on_line(ax, -0.02, 0.08, yin, color='grey', lw=2, alpha=0.8, head=14)
        draw_mid_arrow_on_line(ax, 0.95, 1.02, yout, color='grey', lw=2, alpha=0.8, head=14)

        ax.scatter([0.1], input_y, s=1000, c='#d3d3d3', marker='s', zorder=5)
        ax.text(0.075, input_y[0] - 0.09, "Input\nNeuron", ha='center', va='center', fontsize=13)

        ax.scatter([0.5] * num_neurons, hidden_y, s=400, c='#6C1D45', marker='s', zorder=5)
        ax.text(0.5, input_y[0] + 0.475, "Hidden Layer", ha='center', va='center', fontsize=16)

        ax.scatter([0.9], output_y, s=1000, c='#3D3D3D', marker='s', zorder=5)
        ax.text(0.925, output_y[0] - 0.09, "Output\nNeuron", ha='center', va='center', fontsize=13)

        ax.text(-.05, output_y[0], "Year", ha='center', va='center', fontsize=14)
        ax.text(1.075, output_y[0], "Pop.", ha='center', va='center', fontsize=14)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.schematic_canvas.draw()

    def setup_initial_plots(self):
        self.loss_canvas.axes.set_title("Loss vs. Epoch")
        self.loss_canvas.axes.set_xlabel("Epoch")
        self.loss_canvas.axes.set_ylabel("Mean Squared Error (Loss)")
        self.loss_canvas.axes.grid(True)
        self.loss_canvas.draw()

        self.fit_canvas.axes.clear()
        self.fit_canvas.axes.scatter(self.years, self.population_millions, label='World Bank WDI data')
        self.fit_canvas.axes.set_title(f"Population Data & Model Fit\n{self.dataset_label}")
        self.fit_canvas.axes.set_xlabel("Year")
        self.fit_canvas.axes.set_ylabel("Population (millions)")
        self.fit_canvas.axes.legend()
        self.fit_canvas.axes.grid(True)
        self.fit_canvas.draw()

        self.resid_canvas.axes.clear()
        self.resid_canvas.axes.axhline(0.0, linestyle='--', alpha=0.7)
        self.resid_canvas.axes.set_title("Residuals (Observed − Predicted)")
        self.resid_canvas.axes.set_xlabel("Year")
        self.resid_canvas.axes.set_ylabel("Residual (millions)")
        self.resid_canvas.axes.text(
            0.5, 0.5, "Train the network to see residuals.",
            ha='center', va='center', transform=self.resid_canvas.axes.transAxes, alpha=0.7
        )
        self.resid_canvas.draw()


class CobberHumANNApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberHumANN")
        self.setGeometry(100, 100, 1450, 740)
        self.setFont(self.lato_font)

        central = QWidget()
        main_layout = QVBoxLayout(central)

        top_bar = QFrame()
        top_bar.setFrameShape(QFrame.Shape.StyledPanel)
        top_layout = QHBoxLayout(top_bar)

        title = QLabel("<h2>CobberHumANN: Artificial Neural Networks and Historical Population Data</h2>")
        top_layout.addWidget(title, 2)

        top_layout.addWidget(QLabel("<b>Dataset:</b>"))
        self.dataset_selector = QComboBox()
        self.dataset_selector.addItems(POPULATION_DATASETS.keys())
        self.dataset_selector.currentTextChanged.connect(self.rebuild_tabs)
        top_layout.addWidget(self.dataset_selector, 1)

        main_layout.addWidget(top_bar)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        self.rebuild_tabs(self.dataset_selector.currentText())

    def get_selected_data(self):
        label = self.dataset_selector.currentText()
        meta = POPULATION_DATASETS[label]
        years = np.array([row[0] for row in meta["data"]], dtype=float)
        population = np.array([row[1] for row in meta["data"]], dtype=float)
        return label, meta, years, population

    def rebuild_tabs(self, _label=None):
        self.tabs.clear()
        label, meta, years, population = self.get_selected_data()

        note = meta.get("note", "")

        self.tabs.addTab(
            SingleNeuronTab(years, population, label, note),
            "The Single Neuron"
        )
        self.tabs.addTab(
            NeuralNetworkTab(years, population, label, note, self),
            "The Neural Network"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberHumANNApp()
    window.show()
    sys.exit(app.exec())
