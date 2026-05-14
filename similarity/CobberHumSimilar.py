#!/usr/bin/env python3
"""
CobberHumSimilar.py

Humanities Text Similarity Explorer

This GUI is inspired by the chemistry CobberCluster app, but it compares
public-domain texts instead of molecules.

Students choose:
    1. a curated Gutenberg text pack
    2. a similarity criterion
    3. Generate Similarity Matrix

Then they click matrix cells to inspect two texts side by side.

Expected folder structure
-------------------------
Place this script next to the curated data folder:

    CobberHumSimilar.py
    gutenberg_similarity_packs_curated/
        all_records.jsonl
        packs/
            ghost_supernatural/
            detective_mystery/
            fairy_folk_tales/
            grimm_fairy_tales/
            essays_speeches/

If the curated data is still zipped, unzip:

    gutenberg_similarity_packs_curated.zip

Run:
    python CobberHumSimilar.py

Dependencies:
    pip install PyQt6 matplotlib numpy
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QSplitter, QGroupBox,
    QCheckBox, QStatusBar
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

DATA_DIR = PROJECT_ROOT / "gutenberg_similarity_packs_curated"
ALL_RECORDS = DATA_DIR / "all_records.jsonl"


# ---------------------------------------------------------------------
# Word lists for interpretable features
# ---------------------------------------------------------------------

FIRST_PERSON = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"
}

SECOND_PERSON = {
    "you", "your", "yours", "yourself", "yourselves", "thee", "thou", "thy", "thine"
}

THIRD_PERSON = {
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves"
}

RELIGIOUS_TERMS = {
    "god", "lord", "heaven", "hell", "soul", "sin", "faith", "prayer", "pray",
    "church", "holy", "angel", "devil", "christ", "spirit", "divine", "grace",
    "blessed", "sermon", "bible"
}

SUPERNATURAL_TERMS = {
    "ghost", "spirit", "haunted", "haunt", "apparition", "phantom", "spectre",
    "specter", "supernatural", "mystery", "terror", "fear", "strange", "shadow",
    "dead", "death", "grave", "night", "dark", "dream"
}

DETECTIVE_TERMS = {
    "detective", "clue", "murder", "crime", "criminal", "police", "inspector",
    "evidence", "case", "mystery", "suspect", "investigation", "arrest",
    "guilty", "innocent", "weapon", "footprint", "letter"
}

FAIRY_TERMS = {
    "king", "queen", "prince", "princess", "castle", "witch", "fairy", "giant",
    "dragon", "magic", "spell", "forest", "gold", "silver", "daughter", "son",
    "old", "young", "bird", "wolf"
}

ARGUMENT_TERMS = {
    "truth", "reason", "therefore", "because", "society", "government", "liberty",
    "justice", "moral", "law", "rights", "duty", "believe", "opinion", "public",
    "human", "life", "nature"
}

EMOTION_TERMS = {
    "happy", "sad", "joy", "sorrow", "fear", "afraid", "terror", "love", "hate",
    "angry", "anger", "hope", "despair", "delight", "grief", "weep", "smile",
    "laugh", "cry"
}

DIALOGUE_MARKERS = {
    "said", "asked", "answered", "replied", "cried", "whispered", "shouted",
    "spoke", "told", "called"
}

STOPWORDS = {
    "the", "and", "of", "to", "a", "in", "that", "it", "is", "was", "he", "for",
    "with", "as", "his", "on", "be", "at", "by", "i", "this", "had", "not", "are",
    "but", "from", "or", "have", "an", "they", "which", "one", "you", "were", "her",
    "all", "she", "there", "would", "their", "we", "him", "been", "has", "when",
    "who", "will", "more", "no", "if", "out", "so", "said", "what", "up", "its",
    "about", "into", "than", "them", "can", "could", "my", "me"
}


# ---------------------------------------------------------------------
# Text and similarity utilities
# ---------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def sentence_count(text: str) -> int:
    count = len(re.findall(r"[.!?]+", text))
    return max(count, 1)


def count_terms(tokens: List[str], terms: set[str]) -> int:
    return sum(1 for t in tokens if t in terms)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def euclidean_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Convert feature distances to similarity in [0, 1].
    Uses min-max scaled Euclidean distance.
    """
    if len(X) == 0:
        return np.zeros((0, 0))

    X = X.astype(float)
    # standardize columns
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    Z = (X - means) / stds

    n = Z.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(Z[i] - Z[j])

    max_d = D.max()
    if max_d <= 0:
        return np.ones((n, n))
    return 1.0 - (D / max_d)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            S[i, j] = cosine(X[i], X[j])
    return np.clip(S, 0.0, 1.0)


def jaccard_similarity_matrix(word_sets: List[set[str]]) -> np.ndarray:
    n = len(word_sets)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            union = word_sets[i] | word_sets[j]
            if not union:
                S[i, j] = 0.0
            else:
                S[i, j] = len(word_sets[i] & word_sets[j]) / len(union)
    return S


def tfidf_similarity_matrix(token_lists: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
    vocab_counter = Counter()
    doc_counters = []
    for toks in token_lists:
        filtered = [t for t in toks if t not in STOPWORDS and len(t) > 2]
        c = Counter(filtered)
        doc_counters.append(c)
        vocab_counter.update(c.keys())

    # Keep moderate vocabulary: words appearing in at least 2 docs, up to 500 by document frequency.
    vocab_items = vocab_counter.most_common(500)
    vocab = [w for w, df in vocab_items if df >= 2]
    if not vocab:
        return np.eye(len(token_lists)), []

    word_to_i = {w: i for i, w in enumerate(vocab)}
    n_docs = len(token_lists)
    X = np.zeros((n_docs, len(vocab)))

    df = Counter()
    for c in doc_counters:
        for w in c.keys():
            if w in word_to_i:
                df[w] += 1

    for d, c in enumerate(doc_counters):
        total = sum(c.values()) or 1
        for w, count in c.items():
            if w not in word_to_i:
                continue
            tf = count / total
            idf = math.log((1 + n_docs) / (1 + df[w])) + 1.0
            X[d, word_to_i[w]] = tf * idf

    return cosine_similarity_matrix(X), vocab


def short_excerpt(text: str, max_chars: int = 2600) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

class TextRecord:
    def __init__(self, record: Dict[str, Any], data_dir: Path):
        self.record = record
        self.data_dir = data_dir
        self.doc_id = str(record.get("doc_id", ""))
        self.pack_key = str(record.get("pack_key", ""))
        self.pack_label = str(record.get("pack_label", ""))
        self.title = str(record.get("title", "Untitled"))
        self.author = str(record.get("author", "") or "")
        self.source_title = str(record.get("source_title", "") or "")
        self.gutenberg_id = str(record.get("gutenberg_id", "") or "")
        self.gutenberg_url = str(record.get("gutenberg_url", "") or "")
        self.text_file = str(record.get("text_file", "") or "")
        self.word_count_meta = int(float(record.get("word_count", 0) or 0))
        self.curation_note = str(record.get("curation_note", "") or "")
        self.text = self._load_text()
        self.tokens = tokenize(self.text)
        self.features = self._compute_features()

    def _load_text(self) -> str:
        path = self.data_dir / self.text_file
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def display_name(self) -> str:
        author = f" — {self.author}" if self.author and self.author.lower() != "nan" else ""
        return f"{self.title}{author}"

    def _compute_features(self) -> Dict[str, float]:
        text = self.text
        tokens = self.tokens
        wc = len(tokens)
        sc = sentence_count(text)
        unique = len(set(tokens))
        quotes = text.count('"') + text.count("“") + text.count("”") + text.count("'")

        return {
            "Word count": float(wc),
            "Sentence count": float(sc),
            "Avg sentence length": float(wc / sc if sc else wc),
            "Unique word ratio": float(unique / wc if wc else 0),
            "First-person pronouns": float(count_terms(tokens, FIRST_PERSON)),
            "Second-person pronouns": float(count_terms(tokens, SECOND_PERSON)),
            "Third-person pronouns": float(count_terms(tokens, THIRD_PERSON)),
            "Religious terms": float(count_terms(tokens, RELIGIOUS_TERMS)),
            "Supernatural terms": float(count_terms(tokens, SUPERNATURAL_TERMS)),
            "Detective terms": float(count_terms(tokens, DETECTIVE_TERMS)),
            "Fairy-tale terms": float(count_terms(tokens, FAIRY_TERMS)),
            "Argument terms": float(count_terms(tokens, ARGUMENT_TERMS)),
            "Emotion terms": float(count_terms(tokens, EMOTION_TERMS)),
            "Dialogue markers": float(count_terms(tokens, DIALOGUE_MARKERS)),
            "Quotation marks": float(quotes),
        }


def load_records() -> List[TextRecord]:
    if not ALL_RECORDS.exists():
        return []
    records: List[TextRecord] = []
    with ALL_RECORDS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(TextRecord(json.loads(line), DATA_DIR))
    return records


# ---------------------------------------------------------------------
# Matplotlib matrix canvas
# ---------------------------------------------------------------------

class MatrixCanvas(FigureCanvas):
    def __init__(self, parent_app: "CobberHumSimilarApp"):
        self.figure = Figure(figsize=(7, 6), dpi=100)
        super().__init__(self.figure)
        self.parent_app = parent_app
        self.ax = self.figure.add_subplot(111)
        self.sim = None
        self.names = []
        self.cid = self.mpl_connect("button_press_event", self.on_click)

    def plot_matrix(self, sim: np.ndarray, names: List[str], title: str):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.sim = sim
        self.names = names

        if sim is None or sim.size == 0:
            self.ax.text(0.5, 0.5, "No matrix to show", ha="center", va="center")
            self.ax.set_axis_off()
            self.draw()
            return

        im = self.ax.imshow(sim, vmin=0, vmax=1, aspect="auto")
        self.ax.set_title(title)
        self.ax.set_xticks(range(len(names)))
        self.ax.set_yticks(range(len(names)))
        self.ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        self.ax.set_yticklabels(names, fontsize=8)
        self.ax.set_xlabel("Document")
        self.ax.set_ylabel("Document")

        cbar = self.figure.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04)
        cbar.set_label("Similarity")

        self.figure.tight_layout()
        self.draw()

    def on_click(self, event):
        if self.sim is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        j = int(round(event.xdata))
        i = int(round(event.ydata))
        if 0 <= i < len(self.names) and 0 <= j < len(self.names):
            self.parent_app.select_pair(i, j)


# ---------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------

class CobberHumSimilarApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CobberHumSimilar")
        self.setGeometry(70, 70, 1500, 900)
        self.setFont(QFont("Lato", 10))
        self.cobber_maroon = QColor(108, 29, 69)

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
            QTextEdit, QComboBox, QTableWidget {{
                background-color: white;
                color: #111111;
            }}
            QGroupBox {{
                background-color: white;
                border: 1px solid #d8d0bd;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }}
        """)

        self.records = load_records()
        self.current_records: List[TextRecord] = []
        self.current_similarity: np.ndarray | None = None
        self.current_feature_names: List[str] = []
        self.current_explanation = ""

        self.build_ui()
        self.populate_packs()

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------

    def build_ui(self):
        central = QWidget()
        main = QVBoxLayout(central)
        self.setCentralWidget(central)

        header = QLabel("<h1>CobberHumSimilar</h1>")
        sub = QLabel(
            "Explore similarity among public-domain texts. Choose a text pack, choose what counts as "
            "similar, generate a matrix, then click a pair to interpret the score."
        )
        sub.setWordWrap(True)
        main.addWidget(header)
        main.addWidget(sub)

        controls = QHBoxLayout()
        self.pack_combo = QComboBox()
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems([
            "Length profile",
            "Voice profile",
            "Topic-word profile",
            "Vocabulary overlap (Jaccard)",
            "TF-IDF word similarity",
            "Combined feature profile",
        ])

        self.generate_btn = QPushButton("Generate Similarity Matrix")
        self.generate_btn.clicked.connect(self.generate_matrix)

        controls.addWidget(QLabel("Text pack:"))
        controls.addWidget(self.pack_combo, 2)
        controls.addWidget(QLabel("Similarity criterion:"))
        controls.addWidget(self.criterion_combo, 2)
        controls.addWidget(self.generate_btn)
        controls.addStretch(1)
        main.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.matrix_canvas = MatrixCanvas(self)
        left_layout.addWidget(self.matrix_canvas, 4)

        guide = QTextEdit()
        guide.setReadOnly(True)
        guide.setHtml("""
        <h3>How to read the matrix</h3>
        <p>Each square compares one text with another text. Darker/high values mean the two texts are
        more similar according to the selected criterion.</p>
        <p><b>Important:</b> similarity is always similarity <i>with respect to something</i>.
        Two texts may be similar in length but different in topic, or similar in vocabulary but
        different in genre.</p>
        <p>Click any square to inspect the pair.</p>
        """)
        left_layout.addWidget(guide, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.pair_label = QLabel("<h2>Selected pair</h2>")
        right_layout.addWidget(self.pair_label)

        self.score_label = QLabel("Click a matrix cell to inspect a pair.")
        self.score_label.setWordWrap(True)
        right_layout.addWidget(self.score_label)

        self.feature_table = QTableWidget()
        right_layout.addWidget(QLabel("<b>Feature comparison</b>"))
        right_layout.addWidget(self.feature_table, 2)

        text_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.doc_a = QTextEdit()
        self.doc_a.setReadOnly(True)
        self.doc_b = QTextEdit()
        self.doc_b.setReadOnly(True)
        text_splitter.addWidget(self.doc_a)
        text_splitter.addWidget(self.doc_b)
        right_layout.addWidget(QLabel("<b>Text excerpts</b>"))
        right_layout.addWidget(text_splitter, 3)

        self.interpretation = QTextEdit()
        self.interpretation.setReadOnly(True)
        right_layout.addWidget(QLabel("<b>Interpretive prompt</b>"))
        right_layout.addWidget(self.interpretation, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready.")

    def populate_packs(self):
        if not self.records:
            QMessageBox.warning(
                self,
                "Data folder not found",
                "Could not find gutenberg_similarity_packs_curated/all_records.jsonl.\n\n"
                "Place CobberHumSimilar.py next to the unzipped curated data folder."
            )
            return

        pack_labels = {}
        for rec in self.records:
            pack_labels[rec.pack_key] = rec.pack_label

        preferred_order = [
            "ghost_supernatural",
            "detective_mystery",
            "fairy_folk_tales",
            "grimm_fairy_tales",
            "essays_speeches",
        ]

        for key in preferred_order:
            if key in pack_labels:
                count = sum(1 for r in self.records if r.pack_key == key)
                self.pack_combo.addItem(f"{pack_labels[key]} ({count})", key)

        for key, label in sorted(pack_labels.items()):
            if key not in preferred_order:
                count = sum(1 for r in self.records if r.pack_key == key)
                self.pack_combo.addItem(f"{label} ({count})", key)

        self.generate_matrix()

    # -----------------------------------------------------------------
    # Similarity computation
    # -----------------------------------------------------------------

    def generate_matrix(self):
        if not self.records:
            return

        pack_key = self.pack_combo.currentData()
        criterion = self.criterion_combo.currentText()
        self.current_records = [r for r in self.records if r.pack_key == pack_key]

        if not self.current_records:
            QMessageBox.warning(self, "No records", "No records found for this pack.")
            return

        sim, feature_names, explanation = self.compute_similarity(self.current_records, criterion)
        self.current_similarity = sim
        self.current_feature_names = feature_names
        self.current_explanation = explanation

        names = [f"{i+1}. {self.short_title(r.title)}" for i, r in enumerate(self.current_records)]
        self.matrix_canvas.plot_matrix(sim, names, criterion)

        self.statusBar().showMessage(f"Generated {criterion} matrix for {len(self.current_records)} texts.", 5000)

        # Select first non-diagonal pair by default if possible.
        if len(self.current_records) > 1:
            self.select_pair(0, 1)

    def compute_similarity(self, records: List[TextRecord], criterion: str) -> Tuple[np.ndarray, List[str], str]:
        if criterion == "Length profile":
            names = ["Word count", "Sentence count", "Avg sentence length", "Unique word ratio"]
            X = np.array([[r.features[n] for n in names] for r in records], dtype=float)
            return euclidean_similarity_matrix(X), names, (
                "Length profile compares scale and surface form: word count, sentence count, "
                "average sentence length, and vocabulary variety. This can be mathematically useful "
                "but interpretively shallow."
            )

        if criterion == "Voice profile":
            names = [
                "First-person pronouns", "Second-person pronouns", "Third-person pronouns",
                "Dialogue markers", "Quotation marks"
            ]
            X = np.array([[r.features[n] for n in names] for r in records], dtype=float)
            return euclidean_similarity_matrix(X), names, (
                "Voice profile compares how much the texts use first person, second person, third person, "
                "dialogue markers, and quotation marks. This can reveal differences in address, narration, "
                "and public versus personal voice."
            )

        if criterion == "Topic-word profile":
            names = [
                "Religious terms", "Supernatural terms", "Detective terms", "Fairy-tale terms",
                "Argument terms", "Emotion terms"
            ]
            X = np.array([[r.features[n] for n in names] for r in records], dtype=float)
            return euclidean_similarity_matrix(X), names, (
                "Topic-word profile compares curated groups of words. It asks whether texts share "
                "visible topic signals such as supernatural language, detective language, fairy-tale motifs, "
                "argument words, or emotion words."
            )

        if criterion == "Vocabulary overlap (Jaccard)":
            word_sets = []
            for r in records:
                s = {t for t in r.tokens if t not in STOPWORDS and len(t) > 2}
                word_sets.append(s)
            return jaccard_similarity_matrix(word_sets), ["word-set overlap"], (
                "Jaccard similarity compares shared vocabulary. It is the size of the overlap divided by "
                "the size of the union. Two texts are similar if they use many of the same non-common words."
            )

        if criterion == "TF-IDF word similarity":
            sim, vocab = tfidf_similarity_matrix([r.tokens for r in records])
            return sim, [f"TF-IDF vocabulary size: {len(vocab)}"], (
                "TF-IDF cosine similarity compares weighted vocabulary profiles. Common words matter less; "
                "more distinctive shared words matter more. This is a stronger word-overlap method than raw Jaccard."
            )

        # Combined feature profile
        names = [
            "Word count", "Sentence count", "Avg sentence length", "Unique word ratio",
            "First-person pronouns", "Second-person pronouns", "Third-person pronouns",
            "Religious terms", "Supernatural terms", "Detective terms", "Fairy-tale terms",
            "Argument terms", "Emotion terms", "Dialogue markers", "Quotation marks",
        ]
        X = np.array([[r.features[n] for n in names] for r in records], dtype=float)
        return euclidean_similarity_matrix(X), names, (
            "Combined feature profile compares many interpretable features at once. It is useful for "
            "broad exploration, but you should inspect the feature table to see what is driving each score."
        )

    # -----------------------------------------------------------------
    # Pair selection and display
    # -----------------------------------------------------------------

    def select_pair(self, i: int, j: int):
        if not self.current_records or self.current_similarity is None:
            return
        if i >= len(self.current_records) or j >= len(self.current_records):
            return

        a = self.current_records[i]
        b = self.current_records[j]
        score = float(self.current_similarity[i, j])

        self.pair_label.setText(f"<h2>{self.short_title(a.title)} / {self.short_title(b.title)}</h2>")
        self.score_label.setText(
            f"<b>Similarity score:</b> {score:.3f}<br>"
            f"<b>Criterion:</b> {self.criterion_combo.currentText()}<br>"
            f"{self.current_explanation}"
        )

        self.populate_feature_table(a, b)
        self.doc_a.setPlainText(self.format_doc(a))
        self.doc_b.setPlainText(self.format_doc(b))

        self.interpretation.setPlainText(
            "Interpretive questions:\n"
            "1. Does this score match your reading of the two excerpts?\n"
            "2. Which features seem to drive the similarity or difference?\n"
            "3. Would the pair look more or less similar under another criterion?\n"
            "4. What does this similarity measure miss?"
        )

    def populate_feature_table(self, a: TextRecord, b: TextRecord):
        # Show current features first, then a few always-useful features.
        names = list(self.current_feature_names)
        for n in ["Word count", "Avg sentence length", "Unique word ratio"]:
            if n not in names and n in a.features:
                names.append(n)

        # Expand TF-IDF/Jaccard display with interpretable basics.
        if not names or names[0].startswith("TF-IDF") or names[0] == "word-set overlap":
            names = [
                "Word count", "Sentence count", "Avg sentence length", "Unique word ratio",
                "Supernatural terms", "Detective terms", "Fairy-tale terms",
                "Argument terms", "Dialogue markers"
            ]

        self.feature_table.setColumnCount(4)
        self.feature_table.setHorizontalHeaderLabels(["Feature", "Text A", "Text B", "Abs diff"])
        self.feature_table.setRowCount(len(names))

        for row, name in enumerate(names):
            av = a.features.get(name, 0.0)
            bv = b.features.get(name, 0.0)
            vals = [name, f"{av:.3g}", f"{bv:.3g}", f"{abs(av-bv):.3g}"]
            for col, val in enumerate(vals):
                self.feature_table.setItem(row, col, QTableWidgetItem(str(val)))

        self.feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def format_doc(self, r: TextRecord) -> str:
        return (
            f"Title: {r.title}\n"
            f"Author: {r.author}\n"
            f"Source: {r.source_title}\n"
            f"Pack: {r.pack_label}\n"
            f"Gutenberg ID: {r.gutenberg_id}\n"
            f"Words in excerpt: {len(r.tokens)}\n"
            f"URL: {r.gutenberg_url}\n\n"
            f"{short_excerpt(r.text)}"
        )

    @staticmethod
    def short_title(title: str, n: int = 34) -> str:
        title = re.sub(r"\s+", " ", title).strip()
        if len(title) <= n:
            return title
        return title[:n-3].rstrip() + "..."


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    app = QApplication(sys.argv)
    window = CobberHumSimilarApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
