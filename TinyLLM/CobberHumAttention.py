#!/usr/bin/env python3
"""
CobberHumAttention_v3.py

Student-facing GUI for the attention chapter in Foundations of Machine Learning
for the Humanities.

Purpose
-------
This app teaches attention as a visible, hands-on mechanism:

    Which earlier words matter most for understanding a selected word?

The GUI uses the Dick-and-Jane-style Tiny Reader corpus. It does not train a
neural network. Instead, students:
    1. inspect simple stories,
    2. select target words such as he, she, him, her, it, they,
    3. assign attention weights by hand,
    4. view those weights as a heatmap,
    5. compare their choices to a simple rule-based attention model.

Expected data locations, checked in this order:
    tiny_reader_sets/tiny_reader_demo_small_25.csv
    tiny_reader_sets/tiny_reader_gui_set_85.csv
    tiny_reader_sets/tiny_reader_gui_set_85.jsonl
    tiny_reader_sets/tiny_reader_demo_small_25.jsonl

Run:
    python CobberHumAttention.py

If no data file is found, the app loads a small built-in demo set.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QTextEdit, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QSplitter,
    QGroupBox, QFileDialog, QMessageBox, QTabWidget, QAbstractItemView,
    QSizePolicy
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

DEFAULT_DATA_CANDIDATES = [
    PROJECT_ROOT / "tiny_reader_sets" / "tiny_reader_demo_small_25.csv",
    PROJECT_ROOT / "tiny_reader_sets" / "tiny_reader_gui_set_85.csv",
    PROJECT_ROOT / "tiny_reader_sets" / "tiny_reader_gui_set_85.jsonl",
    PROJECT_ROOT / "tiny_reader_sets" / "tiny_reader_demo_small_25.jsonl",
]

PRONOUNS = {"he", "she", "him", "her", "it", "they", "them"}
MALE_NAMES = {"dick", "tim", "father", "spot"}
FEMALE_NAMES = {"jane", "sally", "mother", "puff"}
PEOPLE_NAMES = MALE_NAMES | FEMALE_NAMES
OBJECT_WORDS = {
    "apple", "ball", "book", "toy", "hat", "basket", "kite", "tree",
    "yard", "park", "home", "school", "store", "garden", "street"
}
PLACE_WORDS = {"yard", "park", "home", "school", "store", "garden", "street", "house"}
ACTION_WORDS = {
    "goes", "walks", "drops", "falls", "lost", "helps", "look", "finds",
    "thanks", "play", "sees", "waves", "gives", "shares", "wants", "waits"
}

DISPLAY_COLORS = {
    "pronoun": "#d5e8ff",
    "person": "#e9d5ff",
    "object": "#fff3bf",
    "place": "#d3f9d8",
    "action": "#ffe3e3",
    "other": "#ffffff",
}


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

@dataclass
class StoryRecord:
    story_id: str
    level: int
    theme: str
    subtheme: str
    status: str
    text: str

    @property
    def short_label(self) -> str:
        theme = "" if self.theme == "unknown" else f" | {self.theme}"
        return f"{self.story_id} | L{self.level}{theme}"


def built_in_stories() -> List[StoryRecord]:
    return [
        StoryRecord(
            "DEMO_L3_1", 3, "giving", "", "demo",
            "Sally goes to the park.\nFather is near the tree.\nSally has a blue kite.\nSally sees Father.\nHe waves to Sally.\nSally gives him the kite.\nThey play in the park.\nThey smile."
        ),
        StoryRecord(
            "DEMO_L5_1", 5, "lost and found", "", "demo",
            "Father goes to the yard.\nMother is near the tree.\nFather has a blue apple.\nHe drops the apple.\nFather is sad because it is lost.\nMother helps him look for it.\nMother finds the apple.\nFather thanks Mother.\nThen they play in the yard."
        ),
        StoryRecord(
            "DEMO_L5_2", 5, "help", "", "demo",
            "Mother goes to the school.\nDick is near a tree.\nMother has a yellow toy.\nShe drops the toy.\nMother is sad because it is lost.\nDick helps her look for it.\nDick finds the toy.\nMother thanks Dick.\nThen they play at the school."
        ),
    ]


def load_stories_from_csv(path: Path) -> List[StoryRecord]:
    stories = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            story_id = row.get("story_id") or row.get("id") or f"ROW_{i:04d}"
            try:
                level = int(row.get("level", 0))
            except Exception:
                level = 0
            text = (row.get("text") or row.get("story") or row.get("story_text") or "").replace("\\n", "\n").strip()
            if not text:
                continue
            stories.append(
                StoryRecord(
                    story_id=str(story_id),
                    level=level,
                    theme=str(row.get("theme", "unknown") or "unknown"),
                    subtheme=str(row.get("subtheme", "") or ""),
                    status=str(row.get("status", "") or ""),
                    text=text,
                )
            )
    return stories


def load_stories_from_jsonl(path: Path) -> List[StoryRecord]:
    stories = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            story_id = row.get("story_id") or row.get("id") or f"ROW_{i:04d}"
            try:
                level = int(row.get("level", 0))
            except Exception:
                level = 0
            text = (row.get("text") or row.get("story") or row.get("story_text") or "").replace("\\n", "\n").strip()
            if not text:
                continue
            stories.append(
                StoryRecord(
                    story_id=str(story_id),
                    level=level,
                    theme=str(row.get("theme", "unknown") or "unknown"),
                    subtheme=str(row.get("subtheme", "") or ""),
                    status=str(row.get("status", "") or ""),
                    text=text,
                )
            )
    return stories


def load_stories(path: Optional[Path] = None) -> Tuple[List[StoryRecord], str]:
    if path is not None:
        if path.suffix.lower() == ".csv":
            stories = load_stories_from_csv(path)
        elif path.suffix.lower() == ".jsonl":
            stories = load_stories_from_jsonl(path)
        else:
            raise ValueError("Please choose a .csv or .jsonl file.")
        if not stories:
            raise ValueError(f"No usable stories found in {path}.")
        return stories, str(path)

    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists():
            if candidate.suffix.lower() == ".csv":
                stories = load_stories_from_csv(candidate)
            else:
                stories = load_stories_from_jsonl(candidate)
            if stories:
                return stories, str(candidate)

    return built_in_stories(), "built-in demo stories"


# ---------------------------------------------------------------------
# Tokenization and attention helpers
# ---------------------------------------------------------------------

@dataclass
class Token:
    text: str
    clean: str
    index: int
    line_index: int
    kind: str


def token_kind(clean: str) -> str:
    c = clean.lower()
    if c in PRONOUNS:
        return "pronoun"
    if c in PEOPLE_NAMES:
        return "person"
    if c in OBJECT_WORDS:
        return "place" if c in PLACE_WORDS else "object"
    if c in ACTION_WORDS:
        return "action"
    return "other"


def tokenize_story(text: str) -> List[Token]:
    tokens = []
    idx = 0
    for line_index, line in enumerate(text.splitlines()):
        for match in re.finditer(r"\b[\w']+\b", line):
            raw = match.group(0)
            clean = raw.lower().strip("'")
            tokens.append(Token(raw, clean, idx, line_index, token_kind(clean)))
            idx += 1
    return tokens


def formatted_story_html(tokens: List[Token], target_index: Optional[int] = None) -> str:
    by_line: Dict[int, List[Token]] = {}
    for tok in tokens:
        by_line.setdefault(tok.line_index, []).append(tok)

    html_lines = []
    for line_idx in sorted(by_line):
        pieces = []
        for tok in by_line[line_idx]:
            color = DISPLAY_COLORS.get(tok.kind, DISPLAY_COLORS["other"])
            border = "2px solid #6C1D45" if tok.index == target_index else "1px solid #cccccc"
            weight = "bold" if tok.index == target_index else "normal"
            pieces.append(
                f"<span style='background-color:{color}; border:{border}; "
                f"padding:2px; margin:1px; font-weight:{weight};'>"
                f"{tok.text}<sub>{tok.index}</sub></span>"
            )
        html_lines.append(" ".join(pieces))
    return "<br>".join(html_lines)


def attention_candidates_for_target(tokens: List[Token], target_index: int) -> List[Token]:
    return [tok for tok in tokens if tok.index < target_index]


def recent_people(tokens: List[Token], before_index: int, gender: Optional[str] = None) -> List[Token]:
    candidates = []
    for tok in reversed(tokens[:before_index]):
        if tok.clean in PEOPLE_NAMES:
            if gender == "male" and tok.clean not in MALE_NAMES:
                continue
            if gender == "female" and tok.clean not in FEMALE_NAMES:
                continue
            candidates.append(tok)
    return candidates


def recent_objects(tokens: List[Token], before_index: int) -> List[Token]:
    candidates = []
    for tok in reversed(tokens[:before_index]):
        if tok.kind == "object":
            candidates.append(tok)
    return candidates


def rule_based_attention(tokens: List[Token], target_index: int) -> Dict[int, int]:
    """Return simple 0--3 attention weights for tokens before the target."""
    weights = {tok.index: 0 for tok in tokens if tok.index < target_index}
    if target_index < 0 or target_index >= len(tokens):
        return weights

    target = tokens[target_index].clean

    def add(tok_list: List[Token], values: List[int]):
        for tok, val in zip(tok_list, values):
            if tok.index in weights:
                weights[tok.index] = max(weights[tok.index], val)

    if target in {"he", "him"}:
        add(recent_people(tokens, target_index, "male")[:3], [3, 2, 1])
    elif target in {"she", "her"}:
        add(recent_people(tokens, target_index, "female")[:3], [3, 2, 1])
    elif target == "it":
        add(recent_objects(tokens, target_index)[:4], [3, 2, 1, 1])
        # Look for nearby lost/drop/find action clues.
        for tok in reversed(tokens[:target_index]):
            if tok.clean in {"lost", "drops", "dropped", "finds", "find", "look"}:
                weights[tok.index] = max(weights.get(tok.index, 0), 1)
    elif target in {"they", "them"}:
        add(recent_people(tokens, target_index, None)[:4], [3, 3, 2, 1])
    else:
        # For non-pronoun target words, use exact earlier repeats and same kind.
        for tok in reversed(tokens[:target_index]):
            if tok.clean == target:
                weights[tok.index] = 3
                break
        for tok in reversed(tokens[:target_index]):
            if tokens[target_index].kind != "other" and tok.kind == tokens[target_index].kind:
                weights[tok.index] = max(weights.get(tok.index, 0), 1)

    return weights


def normalized_similarity(a: Dict[int, int], b: Dict[int, int]) -> float:
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        return 0.0
    va = np.array([a.get(k, 0) for k in keys], dtype=float)
    vb = np.array([b.get(k, 0) for k in keys], dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


# ---------------------------------------------------------------------
# Matplotlib canvas
# ---------------------------------------------------------------------

class HeatmapCanvas(FigureCanvas):
    def __init__(self, width=8, height=3.8, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)

    def draw_single_attention(self, source_tokens: List[Token], weights: Dict[int, int], target: Optional[Token], title: str):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not source_tokens or target is None:
            ax.text(0.5, 0.5, "Select a target word with earlier source words.", ha="center", va="center")
            ax.set_axis_off()
            self.draw()
            return

        values = np.array([[weights.get(tok.index, 0) for tok in source_tokens]], dtype=float)
        ax.imshow(values, aspect="auto", vmin=0, vmax=3)

        labels = [f"{tok.text}\n{tok.index}" for tok in source_tokens]
        ax.set_xticks(range(len(source_tokens)))
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
        ax.set_yticks([0])
        ax.set_yticklabels([f"{target.text}\n{target.index}"])
        ax.set_title(title)
        ax.set_xlabel("Earlier words the target can attend to")
        ax.set_ylabel("Target")
        self.figure.tight_layout()
        self.draw()

    def draw_comparison(self, source_tokens: List[Token], student: Dict[int, int], rule: Dict[int, int], target: Optional[Token]):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if not source_tokens or target is None:
            ax.text(0.5, 0.5, "Select a target word.", ha="center", va="center")
            ax.set_axis_off()
            self.draw()
            return

        diff = {tok.index: abs(student.get(tok.index, 0) - rule.get(tok.index, 0)) for tok in source_tokens}
        values = np.array([
            [student.get(tok.index, 0) for tok in source_tokens],
            [rule.get(tok.index, 0) for tok in source_tokens],
            [diff.get(tok.index, 0) for tok in source_tokens],
        ], dtype=float)
        ax.imshow(values, aspect="auto", vmin=0, vmax=3)
        ax.set_xticks(range(len(source_tokens)))
        ax.set_xticklabels([f"{tok.text}\n{tok.index}" for tok in source_tokens], rotation=65, ha="right", fontsize=8)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Your attention", "Rule model", "Difference"])
        ax.set_title(f"Attention comparison for target: {target.text} ({target.index})")
        self.figure.tight_layout()
        self.draw()


# ---------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------

class CobberHumAttentionApp(QMainWindow):
    data_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.setWindowTitle("CobberHumAttention")
        self.setGeometry(80, 80, 1450, 860)
        self.setFont(QFont("Lato", 10))
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: #f7f3e8; }}
            QLabel {{ color: #111111; }}
            QPushButton {{
                background-color: {self.cobber_maroon.name()};
                color: white;
                font-weight: bold;
                padding: 7px;
                border-radius: 5px;
            }}
            QPushButton:hover {{ background-color: #8a2a58; }}
            QComboBox, QListWidget, QTextEdit, QTableWidget {{
                background-color: #ffffff;
                color: #111111;
            }}
            QGroupBox {{
                background-color: #ffffff;
                border: 1px solid #d8d0bd;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }}
        """)

        self.stories, self.data_source = load_stories()
        self.current_story: Optional[StoryRecord] = None
        self.tokens: List[Token] = []
        self.target_index: Optional[int] = None
        self.student_weights: Dict[int, int] = {}
        self.rule_weights: Dict[int, int] = {}

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.build_story_tab()
        self.build_manual_tab()
        self.build_rule_tab()

        self.refresh_story_combo()
        if self.stories:
            self.set_story(0)

    # -----------------------------------------------------------------
    # Shared utilities
    # -----------------------------------------------------------------

    def refresh_story_combo(self):
        if hasattr(self, "story_combo"):
            self.story_combo.blockSignals(True)
            self.story_combo.clear()
            for story in self.stories:
                self.story_combo.addItem(story.short_label)
            self.story_combo.blockSignals(False)

        if hasattr(self, "manual_story_combo"):
            self.manual_story_combo.blockSignals(True)
            self.manual_story_combo.clear()
            for story in self.stories:
                self.manual_story_combo.addItem(story.short_label)
            self.manual_story_combo.blockSignals(False)

    def set_story(self, index: int):
        if not self.stories:
            return
        index = max(0, min(index, len(self.stories) - 1))
        self.current_story = self.stories[index]
        self.tokens = tokenize_story(self.current_story.text)
        self.target_index = None
        self.student_weights = {}
        self.rule_weights = {}

        if hasattr(self, "story_combo"):
            self.story_combo.blockSignals(True)
            self.story_combo.setCurrentIndex(index)
            self.story_combo.blockSignals(False)
        if hasattr(self, "manual_story_combo"):
            self.manual_story_combo.blockSignals(True)
            self.manual_story_combo.setCurrentIndex(index)
            self.manual_story_combo.blockSignals(False)

        self.refresh_story_views()
        self.refresh_token_lists()

        # Do not auto-select a target word merely because the story changed.
        # The story itself should appear immediately in Tab 2, and then the
        # student can choose a target or click Auto-pick first pronoun.
        if hasattr(self, "target_combo"):
            self.target_combo.blockSignals(True)
            self.target_combo.setCurrentIndex(-1)
            self.target_combo.blockSignals(False)

        self.refresh_manual_weight_table()
        self.refresh_attention_views()

    def set_target_index(self, index: Optional[int]):
        self.target_index = index
        self.student_weights = {}
        if index is not None:
            candidates = attention_candidates_for_target(self.tokens, index)
            self.student_weights = {tok.index: 0 for tok in candidates}
            self.rule_weights = rule_based_attention(self.tokens, index)
        else:
            self.rule_weights = {}
        self.refresh_story_views()
        self.refresh_manual_weight_table()
        self.refresh_attention_views()

    def current_target(self) -> Optional[Token]:
        if self.target_index is None:
            return None
        if 0 <= self.target_index < len(self.tokens):
            return self.tokens[self.target_index]
        return None

    def source_tokens(self) -> List[Token]:
        if self.target_index is None:
            return []
        return attention_candidates_for_target(self.tokens, self.target_index)

    # -----------------------------------------------------------------
    # Tab 1: Story Explorer
    # -----------------------------------------------------------------

    def build_story_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QHBoxLayout()
        self.load_data_button = QPushButton("Load Story Set")
        self.load_data_button.clicked.connect(self.load_story_set)
        self.story_combo = QComboBox()
        self.story_combo.currentIndexChanged.connect(self.set_story)
        self.source_label = QLabel(f"Data source: {self.data_source}")
        self.source_label.setWordWrap(True)
        top.addWidget(self.load_data_button)
        top.addWidget(QLabel("Story:"))
        top.addWidget(self.story_combo, 2)
        top.addWidget(self.source_label, 3)
        layout.addLayout(top)

        note = QLabel(
            "<b>Goal:</b> Notice how the stories become more attention-rich as levels increase. "
            "Level 1 mostly repeats names. Level 3 introduces pronouns. Level 5 often requires "
            "linking pronouns and objects back to earlier words."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.story_html = QTextEdit()
        self.story_html.setReadOnly(True)
        left_layout.addWidget(QLabel("<b>Story with token numbers and word types</b>"))
        left_layout.addWidget(self.story_html)

        legend = QLabel(
            "<span style='background:#d5e8ff'> pronoun </span> "
            "<span style='background:#e9d5ff'> person </span> "
            "<span style='background:#fff3bf'> object </span> "
            "<span style='background:#d3f9d8'> place </span> "
            "<span style='background:#ffe3e3'> action </span>"
        )
        left_layout.addWidget(legend)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.story_info = QTextEdit()
        self.story_info.setReadOnly(True)
        self.token_table = QTableWidget()
        self.token_table.setColumnCount(4)
        self.token_table.setHorizontalHeaderLabels(["Index", "Token", "Clean form", "Kind"])
        self.token_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        right_layout.addWidget(QLabel("<b>Story information</b>"))
        right_layout.addWidget(self.story_info, 1)
        right_layout.addWidget(QLabel("<b>Tokens</b>"))
        right_layout.addWidget(self.token_table, 2)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.tabs.addTab(tab, "1. Story Explorer")

    def load_story_set(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Tiny Reader story set",
            str(PROJECT_ROOT),
            "Story Sets (*.csv *.jsonl);;CSV Files (*.csv);;JSONL Files (*.jsonl);;All Files (*)"
        )
        if not file_name:
            return
        try:
            self.stories, self.data_source = load_stories(Path(file_name))
            self.source_label.setText(f"Data source: {self.data_source}")
            self.refresh_story_combo()
            self.set_story(0)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    def refresh_story_views(self):
        if not self.current_story:
            return
        self.story_html.setHtml(formatted_story_html(self.tokens, self.target_index))
        self.story_info.setHtml(f"""
        <h2>{self.current_story.story_id}</h2>
        <p><b>Level:</b> {self.current_story.level}</p>
        <p><b>Theme:</b> {self.current_story.theme}</p>
        <p><b>Status:</b> {self.current_story.status}</p>
        <p><b>Number of tokens:</b> {len(self.tokens)}</p>
        <p><b>Teaching prompt:</b> Which words become important when the story uses pronouns such as
        <i>he</i>, <i>she</i>, <i>him</i>, <i>her</i>, <i>it</i>, or <i>they</i>?</p>
        """)

        self.token_table.setRowCount(len(self.tokens))
        for r, tok in enumerate(self.tokens):
            for c, value in enumerate([tok.index, tok.text, tok.clean, tok.kind]):
                item = QTableWidgetItem(str(value))
                color = DISPLAY_COLORS.get(tok.kind, "#ffffff")
                item.setBackground(QBrush(QColor(color)))
                self.token_table.setItem(r, c, item)


        self.refresh_manual_story_view()

    def refresh_manual_story_view(self):
        """Keep the Tab 2 story pane synchronized with the selected story.

        This intentionally runs even when no target word has been selected yet.
        The earlier version only filled the Tab 2 story view after a target was
        chosen, which made it look as if selecting a story did nothing.
        """
        if not hasattr(self, "manual_story_view"):
            return
        if not self.current_story:
            self.manual_story_view.setHtml("<p>No story loaded.</p>")
            return
        self.manual_story_view.setHtml(formatted_story_html(self.tokens, self.target_index))

    # -----------------------------------------------------------------
    # Tab 2: Manual Attention
    # -----------------------------------------------------------------

    def build_manual_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top = QHBoxLayout()
        self.manual_story_combo = QComboBox()
        self.manual_story_combo.currentIndexChanged.connect(self.on_manual_story_changed)
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self.on_target_combo_changed)
        self.auto_pick_button = QPushButton("Auto-pick first pronoun")
        self.auto_pick_button.clicked.connect(self.auto_pick_pronoun)
        top.addWidget(QLabel("Story:"))
        top.addWidget(self.manual_story_combo, 2)
        top.addWidget(QLabel("Target word:"))
        top.addWidget(self.target_combo, 1)
        top.addWidget(self.auto_pick_button)
        layout.addLayout(top)

        note = QLabel(
            "<b>Manual attention:</b> Choose a target word. Then decide which earlier words matter most "
            "for understanding that target. Use 0 for no attention and 3 for strong attention."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.manual_story_view = QTextEdit()
        self.manual_story_view.setReadOnly(True)
        self.manual_story_view.setMinimumHeight(220)
        self.weight_table = QTableWidget()
        self.weight_table.setColumnCount(5)
        self.weight_table.setHorizontalHeaderLabels(["Index", "Earlier word", "Kind", "Weight", "Slider"])
        self.weight_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        left_layout.addWidget(QLabel("<b>Story</b>"))
        left_layout.addWidget(self.manual_story_view, 1)
        left_layout.addWidget(QLabel("<b>Assign attention weights to earlier words</b>"))
        left_layout.addWidget(self.weight_table, 3)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.manual_explanation = QTextEdit()
        self.manual_explanation.setReadOnly(True)
        self.manual_heatmap = HeatmapCanvas(width=6, height=3.5)
        right_layout.addWidget(QLabel("<b>Your attention pattern</b>"))
        right_layout.addWidget(self.manual_heatmap, 2)
        right_layout.addWidget(QLabel("<b>Guidance</b>"))
        right_layout.addWidget(self.manual_explanation, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.tabs.addTab(tab, "2. Manual Attention")

    def refresh_token_lists(self):
        if not hasattr(self, "target_combo"):
            return
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        for tok in self.tokens:
            label = f"{tok.index}: {tok.text} ({tok.kind})"
            self.target_combo.addItem(label, tok.index)
        self.target_combo.blockSignals(False)

    def on_manual_story_changed(self, index: int):
        """Load the newly selected story immediately on the Manual Attention tab."""
        if index >= 0:
            self.set_story(index)

    def on_target_combo_changed(self, combo_index: int):
        if combo_index < 0:
            return
        idx = self.target_combo.itemData(combo_index)
        if idx is not None:
            self.set_target_index(int(idx))

    def auto_pick_pronoun(self):
        for tok in self.tokens:
            if tok.clean in PRONOUNS and tok.index > 0:
                # set combo and target
                for i in range(self.target_combo.count()):
                    if self.target_combo.itemData(i) == tok.index:
                        self.target_combo.setCurrentIndex(i)
                        return
        QMessageBox.information(self, "No pronoun found", "This story does not contain an obvious pronoun target.")

    def refresh_manual_weight_table(self):
        if not hasattr(self, "weight_table"):
            return

        target = self.current_target()
        self.refresh_manual_story_view()

        sources = self.source_tokens()
        self.weight_table.setRowCount(len(sources))

        for r, tok in enumerate(sources):
            idx_item = QTableWidgetItem(str(tok.index))
            word_item = QTableWidgetItem(tok.text)
            kind_item = QTableWidgetItem(tok.kind)
            weight_item = QTableWidgetItem(str(self.student_weights.get(tok.index, 0)))
            for item in [idx_item, word_item, kind_item, weight_item]:
                item.setBackground(QBrush(QColor(DISPLAY_COLORS.get(tok.kind, "#ffffff"))))
            self.weight_table.setItem(r, 0, idx_item)
            self.weight_table.setItem(r, 1, word_item)
            self.weight_table.setItem(r, 2, kind_item)
            self.weight_table.setItem(r, 3, weight_item)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(3)
            slider.setTickInterval(1)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setValue(self.student_weights.get(tok.index, 0))
            slider.valueChanged.connect(lambda value, token_index=tok.index, row=r: self.update_student_weight(token_index, row, value))
            self.weight_table.setCellWidget(r, 4, slider)

        if target is None:
            self.manual_explanation.setHtml("<p>Select a target word to begin.</p>")
        else:
            self.manual_explanation.setHtml(f"""
            <p><b>Target:</b> {target.text} ({target.index})</p>
            <p>Ask yourself: if I were reading this story, which earlier words would I look back to?</p>
            <ul>
                <li>For <b>he/him</b>, look for an earlier male-coded character.</li>
                <li>For <b>she/her</b>, look for an earlier female-coded character.</li>
                <li>For <b>it</b>, look for an earlier object.</li>
                <li>For <b>they/them</b>, look for recently mentioned characters acting together.</li>
            </ul>
            <p>There is not always one perfect answer. Attention is a way to represent relevance.</p>
            """)

    def update_student_weight(self, token_index: int, row: int, value: int):
        self.student_weights[token_index] = int(value)
        if self.weight_table.item(row, 3):
            self.weight_table.item(row, 3).setText(str(value))
        self.refresh_attention_views()

    # -----------------------------------------------------------------
    # Shared attention refresh
    # -----------------------------------------------------------------

    def refresh_attention_views(self):
        target = self.current_target()
        sources = self.source_tokens()
        if hasattr(self, "manual_heatmap"):
            self.manual_heatmap.draw_single_attention(sources, self.student_weights, target, "Your manual attention weights")
        if hasattr(self, "comparison_canvas"):
            self.comparison_canvas.draw_comparison(sources, self.student_weights, self.rule_weights, target)


        if hasattr(self, "rule_text"):
            self.refresh_rule_text()

    # -----------------------------------------------------------------
    # Tab 4: Rule-Based Model
    # -----------------------------------------------------------------

    def build_rule_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        note = QLabel(
            "<b>Rule-based attention model:</b> This is not a neural network. It is a simple hand-written "
            "model that imitates some attention patterns in the Tiny Reader stories. Compare it to your "
            "manual attention choices."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.comparison_canvas = HeatmapCanvas(width=11, height=5)
        layout.addWidget(self.comparison_canvas, 2)

        self.rule_text = QTextEdit()
        self.rule_text.setReadOnly(True)
        layout.addWidget(self.rule_text, 1)

        self.tabs.addTab(tab, "3. Rule Model")

    def refresh_rule_text(self):
        target = self.current_target()
        sources = self.source_tokens()
        if target is None:
            self.rule_text.setHtml("<p>Select a target word first.</p>")
            return

        sim = normalized_similarity(self.student_weights, self.rule_weights)

        # Top rule tokens
        sorted_rule = sorted(
            [(idx, wt) for idx, wt in self.rule_weights.items() if wt > 0],
            key=lambda x: (-x[1], x[0])
        )
        top_items = []
        for idx, wt in sorted_rule[:6]:
            tok = self.tokens[idx]
            top_items.append(f"<li>{tok.text} ({idx}) → weight {wt}</li>")
        top_html = "<ul>" + "".join(top_items) + "</ul>" if top_items else "<p>No strong rule-based source found.</p>"

        self.rule_text.setHtml(f"""
        <p><b>Target:</b> {target.text} ({target.index})</p>
        <p><b>Similarity between your weights and the rule model:</b> {sim:.3f}</p>

        <p><b>Rule model's strongest earlier words:</b></p>
        {top_html}

        <p><b>How the simple rule works:</b></p>
        <ul>
            <li><b>he/him</b> attends to recent male-coded names.</li>
            <li><b>she/her</b> attends to recent female-coded names.</li>
            <li><b>it</b> attends to recent object words.</li>
            <li><b>they/them</b> attends to recent people mentioned together.</li>
        </ul>

        <p><b>Important:</b> Real transformer attention is not hand-written this way.
        This rule model is here so you can compare your reading judgment with a simple computational approximation.</p>
        """)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    app = QApplication(sys.argv)
    window = CobberHumAttentionApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
