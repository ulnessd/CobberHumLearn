#!/usr/bin/env python3
"""
CobberHumK.py

Student-facing GUI for K-means clustering and K-nearest-neighbors
classification using a curated WDI human-development dataset.

Expected data file:
    CobberHumKData/wdi_human_development_cluster.csv

Create it with:
    python CobberHumK_DataCollector.py
"""

from __future__ import annotations

import sys
import math
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QPushButton, QSpinBox, QFormLayout, QMessageBox, QTextEdit,
    QComboBox, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle

DATA_PATH = Path("CobberHumKData") / "wdi_human_development_cluster.csv"

FEATURE_INFO = {
    "female_adult_literacy_pct": "Female adult literacy (%)",
    "female_youth_literacy_pct": "Female youth literacy (%)",
    "female_primary_net_enrollment_pct": "Female primary net enrollment (%)",
    "female_secondary_gross_enrollment_pct": "Female secondary enrollment (%)",
    "female_tertiary_gross_enrollment_pct": "Female tertiary enrollment (%)",
    "primary_secondary_gender_parity": "Primary/secondary gender parity",
    "life_expectancy_years": "Life expectancy (years)",
    "maternal_mortality_per_100k": "Maternal mortality ratio",
    "fertility_rate_births_per_woman": "Fertility rate",
    "gdp_per_capita_usd": "GDP per capita",
    "internet_users_pct": "Internet users (%)",
    "female_labor_force_participation_pct": "Female labor force participation (%)",
}

DEFAULT_FEATURES = [
    "female_adult_literacy_pct",
    "female_secondary_gross_enrollment_pct",
    "female_tertiary_gross_enrollment_pct",
    "primary_secondary_gender_parity",
    "life_expectancy_years",
    "maternal_mortality_per_100k",
    "fertility_rate_births_per_woman",
    "gdp_per_capita_usd",
    "internet_users_pct",
    "female_labor_force_participation_pct",
]

CLUSTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
REGION_COLORS = {
    "East Asia & Pacific": "#1f77b4",
    "Europe & Central Asia": "#ff7f0e",
    "Latin America & Caribbean": "#2ca02c",
    "Middle East & North Africa": "#d62728",
    "North America": "#9467bd",
    "South Asia": "#8c564b",
    "Sub-Saharan Africa": "#e377c2",
}
INCOME_ORDER = ["Low income", "Lower middle income", "Upper middle income", "High income"]


@dataclass
class CountryPoint:
    country_code: str
    country_name: str
    region: str
    income_group: str
    features_raw: Dict[str, float]
    cluster: int = -1
    coords_scaled: np.ndarray = field(default_factory=lambda: np.zeros(2))
    coords_2d: Tuple[float, float] = (0.0, 0.0)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


def load_points(path: Path, features: List[str]) -> List[CountryPoint]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run CobberHumK_DataCollector.py first.")
    df = pd.read_csv(path)
    points = []
    for _, row in df.iterrows():
        vals = {}
        ok = True
        for col in features:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if pd.isna(val):
                ok = False
                break
            vals[col] = float(val)
        if ok:
            points.append(CountryPoint(
                country_code=str(row["country_code"]),
                country_name=str(row["country_name"]),
                region=str(row["region"]),
                income_group=str(row["income_group"]),
                features_raw=vals,
            ))
    return points


def prepare_points(points: List[CountryPoint], features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([[p.features_raw[c] for c in features] for p in points], dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    Z = (X - means) / stds
    for p, z in zip(points, Z):
        p.coords_scaled = z
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    coords = Z @ Vt[:2].T
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    var = (S ** 2) / max(len(points) - 1, 1)
    explained = var / var.sum() if var.sum() > 0 else np.array([0, 0])
    for p, xy in zip(points, coords[:, :2]):
        p.coords_2d = (float(xy[0]), float(xy[1]))
    return Vt[:2], explained[:2]


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def run_kmeans(points: List[CountryPoint], k: int, seed: int = 42, max_iter: int = 100):
    rng = np.random.default_rng(seed)
    X = np.array([p.coords_scaled for p in points])
    idx = rng.choice(len(points), size=k, replace=False)
    centroids = {i: X[j].copy() for i, j in enumerate(idx)}
    iterations = 0
    for it in range(max_iter):
        iterations = it + 1
        for p in points:
            p.cluster = min(centroids, key=lambda cid: dist(p.coords_scaled, centroids[cid]))
        new = {}
        for cid in range(k):
            members = np.array([p.coords_scaled for p in points if p.cluster == cid])
            new[cid] = members.mean(axis=0) if len(members) else X[rng.integers(0, len(points))].copy()
        shift = sum(dist(centroids[cid], new[cid]) for cid in centroids)
        centroids = new
        if shift < 1e-6:
            break
    inertia = sum(dist(p.coords_scaled, centroids[p.cluster]) ** 2 for p in points)
    return centroids, inertia, iterations


def nearest_neighbors(training: List[CountryPoint], unknown: CountryPoint, k: int):
    d = [(dist(unknown.coords_scaled, p.coords_scaled), p) for p in training]
    d.sort(key=lambda x: x[0])
    return d[:k]


def predict(neighbors, attr: str) -> str:
    return Counter(getattr(p, attr) for _, p in neighbors).most_common(1)[0][0]


class FeatureSelector(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Features included</b>"))
        self.list = QListWidget()
        for col, label in FEATURE_INFO.items():
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, col)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if col in app.features else Qt.CheckState.Unchecked)
            self.list.addItem(item)
        layout.addWidget(self.list)
        self.button = QPushButton("Apply Feature Selection")
        layout.addWidget(self.button)
        self.button.clicked.connect(self.apply)

    def selected(self):
        cols = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                cols.append(item.data(Qt.ItemDataRole.UserRole))
        return cols

    def apply(self):
        self.app.reload_features(self.selected())


class ExploreTab(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        self.summary = QTextEdit(); self.summary.setReadOnly(True)
        self.country = QComboBox()
        self.details = QTextEdit(); self.details.setReadOnly(True)
        left.addWidget(QLabel("<b>Dataset Explorer</b>"))
        left.addWidget(self.summary)
        left.addWidget(QLabel("Select a country:"))
        left.addWidget(self.country)
        left.addWidget(self.details)
        self.table = QTableWidget()
        layout.addLayout(left, 1)
        layout.addWidget(self.table, 2)
        self.country.currentTextChanged.connect(self.show_country)
        self.refresh()

    def refresh(self):
        points = self.app.points
        by_region = Counter(p.region for p in points)
        by_income = Counter(p.income_group for p in points)
        lines = [f"<b>Countries included:</b> {len(points)}<br>", f"<b>Features included:</b> {len(self.app.features)}<br><hr>"]
        lines.append("<b>Income groups:</b><br>")
        for k, v in by_income.most_common(): lines.append(f"{k}: {v}<br>")
        lines.append("<br><b>Regions:</b><br>")
        for k, v in by_region.most_common(): lines.append(f"{k}: {v}<br>")
        lines.append("<hr><b>Caution:</b> A country is not reducible to a point. Use clusters as questions, not conclusions.")
        self.summary.setHtml("".join(lines))
        self.country.blockSignals(True)
        self.country.clear(); self.country.addItem("Select a country...")
        for p in sorted(points, key=lambda p: p.country_name): self.country.addItem(p.country_name)
        self.country.blockSignals(False)
        cols = ["country_name", "region", "income_group"] + self.app.features[:6]
        self.table.setRowCount(len(points)); self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([FEATURE_INFO.get(c, c) for c in cols])
        for r, p in enumerate(points):
            values = {"country_name": p.country_name, "region": p.region, "income_group": p.income_group, **p.features_raw}
            for c, col in enumerate(cols):
                val = values.get(col, "")
                self.table.setItem(r, c, QTableWidgetItem(f"{val:.3g}" if isinstance(val, float) else str(val)))
        self.table.resizeColumnsToContents()

    def show_country(self, name: str):
        if name == "Select a country...":
            self.details.setHtml(""); return
        p = next((x for x in self.app.points if x.country_name == name), None)
        if not p: return
        lines = [f"<h3>{p.country_name}</h3>", f"<b>Region:</b> {p.region}<br>", f"<b>Income group:</b> {p.income_group}<br><hr>"]
        for col in self.app.features:
            lines.append(f"{FEATURE_INFO.get(col, col)}: {p.features_raw[col]:.3g}<br>")
        self.details.setHtml("".join(lines))



class KMeansTab(QWidget):
    """K-means tab that lets students step into the algorithm."""
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.centroids = {}
        self.centroids_2d = {}
        self.previous_centroids_2d = {}
        self.iteration = 0
        self.phase = "not_initialized"
        self.current_inertia = None

        layout = QHBoxLayout(self)
        controls = QVBoxLayout()
        form = QFormLayout()
        self.k = QSpinBox(); self.k.setRange(2, 8); self.k.setValue(4)
        self.seed = QSpinBox(); self.seed.setRange(1, 9999); self.seed.setValue(42)
        form.addRow("Number of clusters (k):", self.k)
        form.addRow("Random seed:", self.seed)
        controls.addLayout(form)

        self.init_button = QPushButton("1. Initialize Centroids")
        self.assign_button = QPushButton("2. Assign Countries")
        self.move_button = QPushButton("3. Move Centroids")
        self.iter_button = QPushButton("Run One Full Iteration")
        self.full_button = QPushButton("Run to Convergence")
        self.reset_button = QPushButton("Reset")
        for b in [self.init_button, self.assign_button, self.move_button, self.iter_button, self.full_button, self.reset_button]:
            controls.addWidget(b)

        self.status = QLabel("<b>Status:</b> Not initialized")
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        controls.addWidget(FeatureSelector(app))

        self.log = QTextEdit(); self.log.setReadOnly(True)
        controls.addWidget(QLabel("<b>Algorithm log</b>"))
        controls.addWidget(self.log)

        self.canvas = MplCanvas(self, width=8, height=6)
        layout.addLayout(controls, 1); layout.addWidget(self.canvas, 3)

        self.init_button.clicked.connect(self.initialize_centroids)
        self.assign_button.clicked.connect(self.assign_countries)
        self.move_button.clicked.connect(self.move_centroids)
        self.iter_button.clicked.connect(self.run_one_iteration)
        self.full_button.clicked.connect(self.run_to_convergence)
        self.reset_button.clicked.connect(self.reset_algorithm)

        self.update_buttons(); self.redraw()

    def update_buttons(self):
        self.assign_button.setEnabled(self.phase in {"initialized", "moved"})
        self.move_button.setEnabled(self.phase == "assigned")
        self.iter_button.setEnabled(self.phase in {"initialized", "moved"})
        self.full_button.setEnabled(self.phase in {"initialized", "assigned", "moved"})

    def reset_algorithm(self):
        for p in self.app.points:
            p.cluster = -1
        self.centroids = {}; self.centroids_2d = {}; self.previous_centroids_2d = {}
        self.iteration = 0; self.phase = "not_initialized"; self.current_inertia = None
        self.status.setText("<b>Status:</b> Not initialized")
        self.log.append("<hr><b>Reset.</b>")
        self.update_buttons(); self.redraw()

    def initialize_centroids(self):
        if len(self.app.points) < self.k.value():
            QMessageBox.warning(self, "Too few countries", "k cannot be larger than the number of countries."); return
        rng = np.random.default_rng(self.seed.value())
        idx = rng.choice(len(self.app.points), size=self.k.value(), replace=False)
        for p in self.app.points: p.cluster = -1
        self.centroids = {cid: self.app.points[i].coords_scaled.copy() for cid, i in enumerate(idx)}
        self.centroids_2d = {cid: self.app.points[i].coords_2d for cid, i in enumerate(idx)}
        self.previous_centroids_2d = {}
        self.iteration = 0; self.phase = "initialized"; self.current_inertia = None
        names = ", ".join(self.app.points[i].country_name for i in idx)
        self.log.append(f"<b>Initialized k={self.k.value()} centroids</b> using seed {self.seed.value()}.")
        self.log.append(f"Starting centroid countries: {names}")
        self.status.setText("<b>Status:</b> Centroids initialized. Next: assign countries to nearest centroids.")
        self.update_buttons(); self.redraw()

    def calculate_inertia(self):
        if not self.centroids: return 0.0
        return sum(dist(p.coords_scaled, self.centroids[p.cluster]) ** 2 for p in self.app.points if p.cluster in self.centroids)

    def assign_countries(self):
        if not self.centroids: return
        changes = 0
        for p in self.app.points:
            old = p.cluster
            p.cluster = min(self.centroids, key=lambda cid: dist(p.coords_scaled, self.centroids[cid]))
            if p.cluster != old: changes += 1
        self.current_inertia = self.calculate_inertia()
        self.phase = "assigned"
        self.log.append(f"<b>Assignment step:</b> assigned {len(self.app.points)} countries to their nearest centroids. {changes} assignment(s) changed. Inertia = {self.current_inertia:.3f}.")
        self.status.setText("<b>Status:</b> Countries assigned. Next: move each centroid to the mean of its countries.")
        self.update_buttons(); self.redraw(show_lines=True)

    def project_centroids(self):
        out = {}
        for cid, c in self.centroids.items():
            if hasattr(self.app, 'components') and self.app.components is not None:
                xy = c @ self.app.components[:2].T
                out[cid] = (float(xy[0]), float(xy[1]))
            else:
                out[cid] = (0.0, 0.0)
        return out

    def move_centroids(self):
        if self.phase != "assigned": return
        old = {cid: c.copy() for cid, c in self.centroids.items()}
        self.previous_centroids_2d = dict(self.centroids_2d)
        for cid in range(self.k.value()):
            members = np.array([p.coords_scaled for p in self.app.points if p.cluster == cid])
            if len(members): self.centroids[cid] = members.mean(axis=0)
        self.centroids_2d = self.project_centroids()
        shifts = {cid: dist(old[cid], self.centroids[cid]) for cid in self.centroids}
        total_shift = sum(shifts.values())
        self.iteration += 1
        self.current_inertia = self.calculate_inertia()
        self.phase = "converged" if total_shift < 1e-6 else "moved"
        shift_text = ", ".join(f"C{cid}: {shifts[cid]:.3f}" for cid in sorted(shifts))
        self.log.append(f"<b>Move step {self.iteration}:</b> moved centroids to cluster means. Total shift = {total_shift:.5f}. {shift_text}. Inertia = {self.current_inertia:.3f}.")
        if self.phase == "converged":
            self.log.append("<b>Converged:</b> centroids did not move significantly.")
            self.status.setText("<b>Status:</b> Converged.")
        else:
            self.status.setText("<b>Status:</b> Centroids moved. Next: assign countries again.")
        self.update_buttons(); self.redraw(show_arrows=True)

    def run_one_iteration(self):
        if self.phase in {"initialized", "moved"}:
            self.assign_countries(); self.move_centroids()

    def run_to_convergence(self):
        if self.phase == "not_initialized": self.initialize_centroids()
        if self.phase == "assigned": self.move_centroids()
        for _ in range(100):
            if self.phase == "converged": break
            if self.phase in {"initialized", "moved"}:
                self.assign_countries(); self.move_centroids()
        self.show_summary()

    def show_summary(self):
        if not any(p.cluster != -1 for p in self.app.points): return
        self.log.append("<hr><b>Cluster summary</b>")
        for cid in sorted(set(p.cluster for p in self.app.points)):
            members = [p for p in self.app.points if p.cluster == cid]
            self.log.append(f"<b>Cluster {cid}</b>: {len(members)} countries")
            self.log.append("Income groups: " + "; ".join(f"{k}: {v}" for k, v in Counter(p.income_group for p in members).most_common(3)))
            self.log.append("Regions: " + "; ".join(f"{k}: {v}" for k, v in Counter(p.region for p in members).most_common(3)))
            self.log.append("Examples: " + ", ".join(p.country_name for p in sorted(members, key=lambda p: p.country_name)[:8]))

    def redraw(self, show_lines=False, show_arrows=False):
        ax = self.canvas.axes; ax.clear()
        assigned = any(p.cluster != -1 for p in self.app.points)
        used = set()
        for p in self.app.points:
            x, y = p.coords_2d
            if assigned:
                color = CLUSTER_COLORS[p.cluster % len(CLUSTER_COLORS)] if p.cluster >= 0 else 'gray'
                label = f"Cluster {p.cluster}" if p.cluster >= 0 else "Unassigned"
            else:
                color = REGION_COLORS.get(p.region, 'gray'); label = p.region
            ax.scatter(x, y, color=color, alpha=0.75, label=label if label not in used else None)
            used.add(label)
            ax.text(x + 0.02, y, p.country_code, fontsize=7)
            if show_lines and p.cluster in self.centroids_2d:
                cx, cy = self.centroids_2d[p.cluster]
                ax.plot([x, cx], [y, cy], color=color, alpha=0.18, linewidth=0.8)
        for cid, (cx, cy) in self.centroids_2d.items():
            ax.scatter(cx, cy, color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)], marker='X', s=260, edgecolor='black', linewidth=1.2, zorder=8)
            ax.text(cx + 0.05, cy + 0.05, f"C{cid}", fontsize=11, weight='bold', zorder=9)
        if show_arrows:
            for cid, old_xy in self.previous_centroids_2d.items():
                if cid in self.centroids_2d:
                    ax.annotate('', xy=self.centroids_2d[cid], xytext=old_xy,
                                arrowprops=dict(arrowstyle='->', linewidth=2.0, color='black', alpha=0.75), zorder=10)
        ev = self.app.explained
        ax.set_xlabel(f"PC1 ({100*ev[0]:.1f}% variance)" if len(ev) > 0 else "PC1")
        ax.set_ylabel(f"PC2 ({100*ev[1]:.1f}% variance)" if len(ev) > 1 else "PC2")
        ax.set_title("K-means Step by Step: Countries as Points")
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=7, loc='best')
        self.canvas.fig.tight_layout(); self.canvas.draw()


class KNNTab(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app; self.unknown = None
        layout = QHBoxLayout(self); controls = QVBoxLayout(); form = QFormLayout()
        self.country = QComboBox(); self.k = QSpinBox(); self.k.setRange(1, 25); self.k.setValue(7)
        self.label = QComboBox(); self.label.addItem("Income group", "income_group"); self.label.addItem("Region", "region")
        form.addRow("Country to classify:", self.country); form.addRow("Neighbors (k):", self.k); form.addRow("Predict label:", self.label)
        controls.addLayout(form)
        self.results = QTextEdit(); self.results.setReadOnly(True)
        controls.addWidget(QLabel("<b>Prediction details</b>")); controls.addWidget(self.results)
        self.test_button = QPushButton("Run Leave-One-Out Test"); controls.addWidget(self.test_button); controls.addStretch()
        self.canvas = MplCanvas(self, width=8, height=6)
        layout.addLayout(controls, 1); layout.addWidget(self.canvas, 3)
        self.country.currentTextChanged.connect(self.set_unknown); self.k.valueChanged.connect(self.update_view); self.label.currentIndexChanged.connect(self.update_view); self.test_button.clicked.connect(self.full_test)
        self.refresh()

    def refresh(self):
        self.country.blockSignals(True); self.country.clear(); self.country.addItem("Select a country...")
        for p in sorted(self.app.points, key=lambda p: p.country_name): self.country.addItem(p.country_name)
        self.country.blockSignals(False); self.redraw()

    def set_unknown(self, name):
        self.unknown = None if name == "Select a country..." else next((p for p in self.app.points if p.country_name == name), None)
        self.update_view()

    def update_view(self):
        if not self.unknown:
            self.results.setHtml(""); self.redraw(); return
        attr = self.label.currentData(); k = min(self.k.value(), len(self.app.points) - 1)
        training = [p for p in self.app.points if p.country_code != self.unknown.country_code]
        nn = nearest_neighbors(training, self.unknown, k); pred = predict(nn, attr); actual = getattr(self.unknown, attr)
        lines = [f"<h3>{self.unknown.country_name}</h3>", f"<b>Actual:</b> {actual}<br>", f"<b>Prediction:</b> {pred}<br>", f"<b>Result:</b> {'Correct' if pred == actual else 'Different'}<hr>", "<b>Neighbor votes:</b><br>"]
        for lbl, n in Counter(getattr(p, attr) for _, p in nn).most_common(): lines.append(f"{lbl}: {n}<br>")
        lines.append("<hr><b>Nearest neighbors:</b><br>")
        for d, p in nn: lines.append(f"{p.country_name} — {getattr(p, attr)} — distance {d:.2f}<br>")
        self.results.setHtml("".join(lines)); self.redraw(nn)

    def full_test(self):
        attr = self.label.currentData(); k = self.k.value(); correct = 0; total = 0; conf = defaultdict(Counter)
        for u in self.app.points:
            training = [p for p in self.app.points if p.country_code != u.country_code]
            nn = nearest_neighbors(training, u, min(k, len(training))); pred = predict(nn, attr); actual = getattr(u, attr)
            correct += int(pred == actual); total += 1; conf[actual][pred] += 1
        lines = [f"<h3>Leave-One-Out Test</h3><b>Correct:</b> {correct}/{total}<br><b>Accuracy:</b> {100*correct/total:.1f}%<hr>"]
        for actual in sorted(conf): lines.append(f"{actual} → " + ", ".join(f"{p}: {n}" for p, n in conf[actual].most_common()) + "<br>")
        self.results.setHtml("".join(lines))

    def redraw(self, neighbors=None):
        ax = self.canvas.axes; ax.clear(); attr = self.label.currentData() if hasattr(self, "label") else "income_group"
        used = set()
        for p in self.app.points:
            if self.unknown and p.country_code == self.unknown.country_code: continue
            x, y = p.coords_2d
            if attr == "region": color = REGION_COLORS.get(p.region, "gray"); label = p.region
            else:
                idx = INCOME_ORDER.index(p.income_group) if p.income_group in INCOME_ORDER else 0
                color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]; label = p.income_group
            ax.scatter(x, y, color=color, alpha=0.75, label=label if label not in used else None); used.add(label)
            ax.text(x + 0.02, y, p.country_code, fontsize=8)
        if self.unknown:
            ux, uy = self.unknown.coords_2d
            ax.scatter(ux, uy, c="black", marker="X", s=220, zorder=5, label="Selected")
            if neighbors:
                for _, n in neighbors: ax.scatter(n.coords_2d[0], n.coords_2d[1], s=180, facecolors="none", edgecolors="gold", linewidth=2.2, zorder=4)
                radius = math.dist(self.unknown.coords_2d, neighbors[-1][1].coords_2d)
                ax.add_patch(Circle(self.unknown.coords_2d, radius, facecolor="none", edgecolor="black", linestyle="--"))
        ev = self.app.explained
        ax.set_xlabel(f"PC1 ({100*ev[0]:.1f}% variance)"); ax.set_ylabel(f"PC2 ({100*ev[1]:.1f}% variance)")
        ax.set_title("KNN: Classifying by Nearby Countries"); ax.grid(True, linestyle="--", alpha=0.4); ax.legend(fontsize=8, loc="best")
        self.canvas.fig.tight_layout(); self.canvas.draw()


class EthicsTab(QWidget):
    def __init__(self):
        super().__init__(); layout = QVBoxLayout(self); text = QTextEdit(); text.setReadOnly(True)
        text.setHtml("""
        <h2>Interpretive and Ethical Cautions</h2>
        <p><b>A country is not a data point.</b> We represent countries as points to learn clustering and classification. The representation is partial.</p>
        <p><b>Indicators are choices.</b> Education, health, income, and digital-access indicators matter, but they do not capture a whole society.</p>
        <p><b>Clusters are not natural kinds.</b> K-means will make clusters whenever we ask it to. The result depends on features, scaling, missing data, and k.</p>
        <p><b>Classification is not understanding.</b> KNN predicts a label by comparing nearby examples. It does not know history, politics, or lived experience.</p>
        <p><b>Use the model as a question generator.</b> Unexpected results should lead to closer interpretation, not quick judgment.</p>
        """)
        layout.addWidget(text)


class CobberHumKApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CobberHumK v2")
        self.setGeometry(80, 80, 1300, 780)
        self.cobber_maroon = QColor(108, 29, 69); self.cobber_gold = QColor(234, 170, 0)
        self.setFont(QFont("Lato"))
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: #f7f3e8; }}
            QPushButton {{ background-color: {self.cobber_gold.name()}; color: {self.cobber_maroon.name()}; font-weight: bold; padding: 7px; border-radius: 5px; }}
            QPushButton:hover {{ background-color: #f0c64a; }}
            QTextEdit {{ background-color: #ffffff; }}
        """)
        self.features = list(DEFAULT_FEATURES)
        self.points: List[CountryPoint] = []
        self.explained = np.array([0.0, 0.0])
        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)
        self.load_data()
        self.build_tabs()

    def load_data(self):
        try:
            self.points = load_points(DATA_PATH, self.features)
            if len(self.points) < 5: raise ValueError("Too few countries after feature filtering.")
            self.components, self.explained = prepare_points(self.points, self.features)
        except Exception as exc:
            QMessageBox.warning(self, "Data file needed", f"{exc}\n\nRun CobberHumK_DataCollector_v2.py and keep the CobberHumKData folder beside this program.")
            self.points = []

    def build_tabs(self):
        self.tabs.clear()
        self.tabs.addTab(ExploreTab(self), "1. Explore Data")
        self.tabs.addTab(KMeansTab(self), "2. K-means Step-by-Step")
        self.tabs.addTab(KNNTab(self), "3. KNN")
        self.tabs.addTab(EthicsTab(), "4. Interpretation")

    def reload_features(self, features: List[str]):
        if len(features) < 2:
            QMessageBox.warning(self, "Need features", "Please select at least two features."); return
        old = self.features
        self.features = features
        try:
            self.points = load_points(DATA_PATH, self.features)
            if len(self.points) < 5: raise ValueError("Too few countries have complete data for these features.")
            self.components, self.explained = prepare_points(self.points, self.features)
            self.build_tabs()
        except Exception as exc:
            self.features = old; self.load_data(); self.build_tabs()
            QMessageBox.warning(self, "Feature selection error", str(exc))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberHumKApp()
    window.show()
    sys.exit(app.exec())
