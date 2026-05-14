#!/usr/bin/env python3
"""
CobberHumTinyLM.py

Student-facing GUI for the Tiny Reader / Dick-and-Jane language-model chapter.

Purpose
-------
This app helps students compare:

    1. A transparent n-gram language model
    2. A tiny word-level transformer language model

The central lesson is:

    A language model predicts the next token.
    A simple n-gram model predicts from local counts.
    A transformer can use wider context to preserve story state.

Expected project structure
--------------------------
Run this script from the DickJaneLLM project folder, with these subfolders:

    tiny_reader_model_prep/
        corpus_records_clean.jsonl
        corpus_train.jsonl
        corpus_val.jsonl
        corpus_test.jsonl
        corpus_summary.txt
        corpus_vocab.csv
        corpus_sentence_frames.csv

    tiny_reader_ngram_output/
        ngram_model.json
        ngram_summary.txt
        ngram_eval.csv
        ngram_prompt_predictions.csv
        ngram_generated_samples.txt

    tiny_reader_transformer_output/
        tiny_transformer_model.pt
        tiny_transformer_vocab.json
        tiny_transformer_config.json
        tiny_transformer_eval.txt
        tiny_transformer_prompt_predictions.csv
        tiny_transformer_samples.txt

Run:
    python CobberHumTinyLM.py

Notes
-----
The GUI can still run if the transformer model is missing. In that case,
the corpus and n-gram tabs will work, and the transformer tabs will display
a helpful message.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QSplitter,
    QSpinBox, QDoubleSpinBox, QMessageBox, QTabWidget,
    QFileDialog, QCheckBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------------------------------------------------------------
# Optional torch
# ---------------------------------------------------------------------

TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

MODEL_PREP_DIR = PROJECT_ROOT / "tiny_reader_model_prep"
NGRAM_DIR = PROJECT_ROOT / "tiny_reader_ngram_output"
TRANSFORMER_DIR = PROJECT_ROOT / "tiny_reader_transformer_output"

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"


# ---------------------------------------------------------------------
# Tokenization and utilities
# ---------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+|[.,!?;:]", text.lower())


def detokenize(tokens: List[str]) -> str:
    filtered = [t for t in tokens if t not in {PAD, BOS, EOS}]
    s = " ".join(filtered)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    parts = re.split(r"([.!?]\s*)", s)
    out = []
    cap_next = True
    for part in parts:
        if not part:
            continue
        if cap_next and re.match(r"[a-z]", part):
            part = part[0].upper() + part[1:]
        out.append(part)
        if re.match(r"[.!?]\s*", part):
            cap_next = True
        else:
            cap_next = False
    return "".join(out).strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, str]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def load_text_file(path: Path, fallback: str = "") -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


def set_table_rows(table: QTableWidget, rows: List[Dict[str, Any]], columns: List[str]):
    table.setColumnCount(len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, col in enumerate(columns):
            value = row.get(col, "")
            if isinstance(value, float):
                text = f"{value:.4f}"
            else:
                text = str(value)
            table.setItem(r, c, QTableWidgetItem(text))
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


# ---------------------------------------------------------------------
# Plot canvas
# ---------------------------------------------------------------------

class SimpleCanvas(FigureCanvas):
    def __init__(self, width=5, height=3, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)

    def bar_chart(self, labels: List[str], values: List[float], title: str, ylabel: str = "Value"):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if labels and values:
            ax.bar(range(len(labels)), values)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        self.figure.tight_layout()
        self.draw()


# ---------------------------------------------------------------------
# N-gram model loader/generator
# ---------------------------------------------------------------------

class NgramModelLoaded:
    def __init__(self, path: Path):
        self.path = path
        self.max_n = 5
        self.context_counts: Dict[int, Dict[Tuple[str, ...], Counter]] = {}
        self.loaded = False
        if path.exists():
            self.load(path)

    def load(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.max_n = int(payload.get("max_n", 5))
        self.context_counts = {}
        raw = payload.get("context_counts", {})
        for n_str, ctx_map in raw.items():
            n = int(n_str)
            self.context_counts[n] = {}
            for ctx_key, counter_dict in ctx_map.items():
                ctx = tuple(ctx_key.split("\t")) if ctx_key else tuple()
                self.context_counts[n][ctx] = Counter({str(k): int(v) for k, v in counter_dict.items()})
        self.loaded = True

    def _best_counter(self, context_tokens: List[str], order: int) -> Tuple[int, Tuple[str, ...], Counter]:
        order = max(1, min(order, self.max_n))
        for n in range(order, 0, -1):
            context_len = n - 1
            ctx = tuple(context_tokens[-context_len:]) if context_len > 0 else tuple()
            counter = self.context_counts.get(n, {}).get(ctx)
            if counter:
                return n, ctx, counter
        return 1, tuple(), self.context_counts.get(1, {}).get(tuple(), Counter())

    def predict_next(self, prefix: str, order: int = 5, top_k: int = 12) -> List[Dict[str, Any]]:
        if not self.loaded:
            return []
        prefix_toks = [BOS] * (self.max_n - 1) + tokenize(prefix)
        n, ctx, counter = self._best_counter(prefix_toks, order)
        total = sum(counter.values())
        rows = []
        if total == 0:
            return rows
        for rank, (tok, count) in enumerate(counter.most_common(top_k), start=1):
            rows.append({
                "rank": rank,
                "token": tok,
                "count": count,
                "probability": count / total,
                "used_order": n,
                "context": " ".join(ctx),
            })
        return rows

    def sample_next(self, context_tokens: List[str], order: int, temperature: float, rng: random.Random) -> str:
        n, ctx, counter = self._best_counter(context_tokens, order)
        if not counter:
            return EOS
        items = list(counter.items())
        toks = [x[0] for x in items]
        counts = [float(x[1]) for x in items]
        if temperature <= 0:
            return toks[counts.index(max(counts))]
        total = sum(counts)
        probs = [c / total for c in counts]
        scaled = [p ** (1.0 / temperature) for p in probs]
        z = sum(scaled)
        scaled = [p / z for p in scaled]
        r = rng.random()
        cum = 0.0
        for tok, p in zip(toks, scaled):
            cum += p
            if r <= cum:
                return tok
        return toks[-1]

    def generate(self, prefix: str, order: int = 5, max_new_tokens: int = 80, temperature: float = 0.8, seed: int = 1) -> str:
        rng = random.Random(seed)
        tokens = [BOS] * (self.max_n - 1) + tokenize(prefix)
        for _ in range(max_new_tokens):
            nxt = self.sample_next(tokens, order, temperature, rng)
            if nxt == EOS:
                break
            tokens.append(nxt)
        return detokenize(tokens[self.max_n - 1:])


# ---------------------------------------------------------------------
# Tiny transformer loader
# ---------------------------------------------------------------------

if TORCH_AVAILABLE:
    class ModelConfig:
        def __init__(self, vocab_size, block_size=96, d_model=192, n_heads=4, n_layers=4, dropout=0.1, pad_id=0):
            self.vocab_size = vocab_size
            self.block_size = block_size
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_layers = n_layers
            self.dropout = dropout
            self.pad_id = pad_id

    class TinyTransformerLM(nn.Module):
        def __init__(self, cfg: ModelConfig):
            super().__init__()
            self.cfg = cfg
            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=4 * cfg.d_model,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
            self.ln_f = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        def forward(self, idx):
            b, t = idx.shape
            if t > self.cfg.block_size:
                idx = idx[:, -self.cfg.block_size:]
                t = idx.shape[1]
            pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
            x = self.token_emb(idx) + self.pos_emb(pos)
            causal_mask = torch.triu(torch.ones(t, t, device=idx.device, dtype=torch.bool), diagonal=1)
            pad_mask = idx.eq(self.cfg.pad_id)
            x = self.blocks(x, mask=causal_mask, src_key_padding_mask=pad_mask)
            x = self.ln_f(x)
            return self.head(x)


class TransformerModelLoaded:
    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.loaded = False
        self.error = ""
        self.device = None
        self.model = None
        self.stoi = {}
        self.itos = {}
        self.cfg = None
        self._try_load()

    def _try_load(self):
        if not TORCH_AVAILABLE:
            self.error = "PyTorch is not installed in this environment."
            return
        try:
            config_path = self.outdir / "tiny_transformer_config.json"
            vocab_path = self.outdir / "tiny_transformer_vocab.json"
            model_path = self.outdir / "tiny_transformer_model.pt"
            if not config_path.exists() or not vocab_path.exists() or not model_path.exists():
                self.error = "Transformer model files were not found."
                return

            cfg_dict = json.loads(config_path.read_text(encoding="utf-8"))
            self.cfg = ModelConfig(**cfg_dict)

            vocab_payload = json.loads(vocab_path.read_text(encoding="utf-8"))
            self.stoi = {str(k): int(v) for k, v in vocab_payload["stoi"].items()}
            self.itos = {int(k): str(v) for k, v in vocab_payload["itos"].items()}

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = TinyTransformerLM(self.cfg).to(self.device)
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            self.model.eval()
            self.loaded = True
        except Exception as exc:
            self.error = str(exc)
            self.loaded = False

    def encode_prompt(self, prompt: str) -> List[int]:
        ids = [self.stoi[BOS]] + [self.stoi.get(tok, self.stoi[UNK]) for tok in tokenize(prompt)]
        return ids[-self.cfg.block_size:]

    def predict_next(self, prompt: str, top_k: int = 12) -> List[Dict[str, Any]]:
        if not self.loaded:
            return []
        ids = self.encode_prompt(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            last = logits[0, -1, :]
            probs = F.softmax(last, dim=-1)
            vals, inds = torch.topk(probs, k=min(top_k, probs.numel()))
        rows = []
        for rank, (v, i) in enumerate(zip(vals.tolist(), inds.tolist()), start=1):
            rows.append({"rank": rank, "token": self.itos[int(i)], "probability": float(v)})
        return rows

    def generate(self, prompt: str, max_new_tokens: int = 80, temperature: float = 0.8, top_k: int = 20, seed: int = 1) -> str:
        if not self.loaded:
            return "[Transformer model is not loaded.]"
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        ids = self.encode_prompt(prompt)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = torch.tensor([ids[-self.cfg.block_size:]], dtype=torch.long, device=self.device)
                logits = self.model(x)[0, -1, :]

                for bad in [PAD, BOS, UNK]:
                    if bad in self.stoi:
                        logits[self.stoi[bad]] = -float("inf")

                if temperature <= 0:
                    next_id = int(torch.argmax(logits).item())
                else:
                    logits = logits / temperature
                    if top_k and top_k > 0:
                        vals, inds = torch.topk(logits, k=min(top_k, logits.numel()))
                        filtered = torch.full_like(logits, -float("inf"))
                        filtered[inds] = vals
                        logits = filtered
                    probs = F.softmax(logits, dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())

                ids.append(next_id)
                if self.itos[next_id] == EOS:
                    break
        torch.random.set_rng_state(old_state)
        return detokenize([self.itos[i] for i in ids])


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

DIAGNOSTIC_PROMPTS = {
    "Object persistence: apple": "father goes to the yard. mother is near the tree. father has a blue apple. he drops the",
    "Because -> it": "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because",
    "Look for -> it": "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because it is lost. mother helps him look for",
    "Finds the -> apple": "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because it is lost. mother helps him look for it. mother finds the",
    "Thanks -> Mother": "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because it is lost. mother helps him look for it. mother finds the apple. father thanks",
    "Then they -> play": "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because it is lost. mother helps him look for it. mother finds the apple. father thanks mother. then they",
    "Gives him the -> kite": "sally goes to the park. father is near the tree. sally has a blue kite. sally sees father. he waves to sally. sally gives him the",
    "Wants to -> play": "when jane goes to the store, tim is there. jane has a blue ball. tim sees the ball. he wants to",
    "Open location": "tim goes to the",
    "Pet drops object": "jane goes to the yard. puff comes with her. jane has a yellow toy. she drops the",
}


# ---------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------

class CobberHumTinyLMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CobberHumTinyLM")
        self.setGeometry(80, 80, 1500, 900)
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
            QTextEdit, QComboBox, QTableWidget, QSpinBox, QDoubleSpinBox {{
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

        self.records = read_jsonl(MODEL_PREP_DIR / "corpus_records_clean.jsonl")
        self.ngram = NgramModelLoaded(NGRAM_DIR / "ngram_model.json")
        self.transformer = TransformerModelLoaded(TRANSFORMER_DIR)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.build_corpus_tab()
        self.build_ngram_tab()
        self.build_transformer_tab()
        self.build_compare_tab()
        self.build_try_tab()

    # -----------------------------------------------------------------
    # Tab 1 Corpus Explorer
    # -----------------------------------------------------------------

    def build_corpus_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("<h2>Corpus Explorer: the model's tiny language world</h2>")
        layout.addWidget(title)

        note = QLabel(
            "A language model learns from a corpus. This tab shows the controlled story world "
            "used to train the n-gram and transformer models."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.corpus_summary = QTextEdit()
        self.corpus_summary.setReadOnly(True)
        summary = load_text_file(MODEL_PREP_DIR / "corpus_summary.txt",
                                 "No corpus_summary.txt found. Run TinyReaderCorpusInspector.py first.")
        self.corpus_summary.setPlainText(summary)
        left_layout.addWidget(QLabel("<b>Corpus summary</b>"))
        left_layout.addWidget(self.corpus_summary, 2)

        self.level_chart = SimpleCanvas(width=5, height=3)
        level_rows = read_csv_rows(MODEL_PREP_DIR / "corpus_level_summary.csv")
        labels = [f"L{r.get('level','?')}" for r in level_rows]
        vals = [float(r.get("stories", 0)) for r in level_rows]
        self.level_chart.bar_chart(labels, vals, "Stories by level", "Stories")
        left_layout.addWidget(self.level_chart, 1)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.story_filter = QComboBox()
        self.story_filter.addItem("All levels", 0)
        for level in [1, 2, 3, 4, 5]:
            self.story_filter.addItem(f"Level {level}", level)
        self.story_filter.currentIndexChanged.connect(self.refresh_story_browser)

        self.story_combo = QComboBox()
        self.story_combo.currentIndexChanged.connect(self.show_selected_story)

        right_layout.addWidget(QLabel("<b>Browse training stories</b>"))
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Filter:"))
        controls.addWidget(self.story_filter)
        controls.addWidget(QLabel("Story:"))
        controls.addWidget(self.story_combo, 1)
        right_layout.addLayout(controls)

        self.story_view = QTextEdit()
        self.story_view.setReadOnly(True)
        right_layout.addWidget(self.story_view, 2)

        self.frame_table = QTableWidget()
        frame_rows = read_csv_rows(MODEL_PREP_DIR / "corpus_sentence_frames.csv", max_rows=20)
        set_table_rows(self.frame_table, frame_rows, ["sentence_frame", "count"])
        right_layout.addWidget(QLabel("<b>Most common sentence frames</b>"))
        right_layout.addWidget(self.frame_table, 2)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        self.refresh_story_browser()
        self.tabs.addTab(tab, "1. Corpus Explorer")

    def refresh_story_browser(self):
        level = self.story_filter.currentData() if hasattr(self, "story_filter") else 0
        subset = [r for r in self.records if level in (0, None) or int(r.get("level", 0)) == level]
        subset = subset[:500]  # keep combo responsive
        self._story_subset = subset
        self.story_combo.blockSignals(True)
        self.story_combo.clear()
        for rec in subset:
            self.story_combo.addItem(f"{rec.get('story_id','?')} | L{rec.get('level','?')}")
        self.story_combo.blockSignals(False)
        if subset:
            self.story_combo.setCurrentIndex(0)
            self.show_selected_story(0)

    def show_selected_story(self, index: int):
        if not hasattr(self, "_story_subset") or index < 0 or index >= len(self._story_subset):
            return
        rec = self._story_subset[index]
        text = rec.get("text", "")
        tokens = tokenize(text)
        self.story_view.setPlainText(
            f"Story ID: {rec.get('story_id')}\n"
            f"Level: {rec.get('level')}\n"
            f"Tokens: {len(tokens)}\n\n"
            f"{text}"
        )

    # -----------------------------------------------------------------
    # Tab 2 N-gram
    # -----------------------------------------------------------------

    def build_ngram_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("<h2>N-gram Model: prediction by local counting</h2>")
        layout.addWidget(title)
        note = QLabel(
            "An n-gram model predicts the next token by counting what usually follows a short local context. "
            "It is transparent and fast, but it often fails to preserve the larger story state."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        controls = QHBoxLayout()
        self.ngram_prompt = QComboBox()
        self.ngram_prompt.setEditable(True)
        for name, prompt in DIAGNOSTIC_PROMPTS.items():
            self.ngram_prompt.addItem(f"{name}: {prompt}", prompt)
        self.ngram_order = QSpinBox()
        self.ngram_order.setRange(1, 5)
        self.ngram_order.setValue(5)
        self.ngram_temp = QDoubleSpinBox()
        self.ngram_temp.setRange(0.0, 2.0)
        self.ngram_temp.setSingleStep(0.1)
        self.ngram_temp.setValue(0.8)
        self.ngram_generate_button = QPushButton("Run N-gram Prediction")
        self.ngram_generate_button.clicked.connect(self.run_ngram)

        controls.addWidget(QLabel("Prompt:"))
        controls.addWidget(self.ngram_prompt, 3)
        controls.addWidget(QLabel("Order:"))
        controls.addWidget(self.ngram_order)
        controls.addWidget(QLabel("Temp:"))
        controls.addWidget(self.ngram_temp)
        controls.addWidget(self.ngram_generate_button)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.ngram_pred_table = QTableWidget()
        left_layout.addWidget(QLabel("<b>Next-token predictions</b>"))
        left_layout.addWidget(self.ngram_pred_table, 1)
        self.ngram_chart = SimpleCanvas(width=5, height=3)
        left_layout.addWidget(self.ngram_chart, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.ngram_output = QTextEdit()
        self.ngram_output.setReadOnly(True)
        self.ngram_summary = QTextEdit()
        self.ngram_summary.setReadOnly(True)
        self.ngram_summary.setPlainText(load_text_file(NGRAM_DIR / "ngram_summary.txt",
                                                       "No ngram_summary.txt found. Run TinyReaderNgramLab.py first."))
        right_layout.addWidget(QLabel("<b>Generated continuation</b>"))
        right_layout.addWidget(self.ngram_output, 2)
        right_layout.addWidget(QLabel("<b>N-gram summary</b>"))
        right_layout.addWidget(self.ngram_summary, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        self.run_ngram()
        self.tabs.addTab(tab, "2. N-gram Model")

    def selected_prompt(self, combo: QComboBox) -> str:
        data = combo.currentData()
        if data:
            return str(data)
        return combo.currentText()

    def run_ngram(self):
        prompt = self.selected_prompt(self.ngram_prompt)
        order = self.ngram_order.value()
        temp = self.ngram_temp.value()
        preds = self.ngram.predict_next(prompt, order=order, top_k=12)
        set_table_rows(self.ngram_pred_table, preds, ["rank", "token", "count", "probability", "used_order", "context"])
        self.ngram_chart.bar_chart([p["token"] for p in preds[:8]], [p["probability"] for p in preds[:8]],
                                   "N-gram next-token probabilities", "Probability")
        generated = self.ngram.generate(prompt, order=order, temperature=temp, seed=7) if self.ngram.loaded else "[N-gram model not loaded.]"
        self.ngram_output.setPlainText(generated)

    # -----------------------------------------------------------------
    # Tab 3 Transformer
    # -----------------------------------------------------------------

    def build_transformer_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("<h2>Tiny Transformer: prediction with broader context</h2>")
        layout.addWidget(title)

        status = "loaded" if self.transformer.loaded else f"not loaded: {self.transformer.error}"
        note = QLabel(
            f"Transformer status: <b>{status}</b><br>"
            "The tiny transformer is still predicting the next token, but it can use a longer story context "
            "through causal self-attention."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        controls = QHBoxLayout()
        self.trans_prompt = QComboBox()
        self.trans_prompt.setEditable(True)
        for name, prompt in DIAGNOSTIC_PROMPTS.items():
            self.trans_prompt.addItem(f"{name}: {prompt}", prompt)
        self.trans_temp = QDoubleSpinBox()
        self.trans_temp.setRange(0.0, 2.0)
        self.trans_temp.setSingleStep(0.1)
        self.trans_temp.setValue(0.8)
        self.trans_topk = QSpinBox()
        self.trans_topk.setRange(1, 50)
        self.trans_topk.setValue(20)
        self.trans_button = QPushButton("Run Transformer Prediction")
        self.trans_button.clicked.connect(self.run_transformer)

        controls.addWidget(QLabel("Prompt:"))
        controls.addWidget(self.trans_prompt, 3)
        controls.addWidget(QLabel("Temp:"))
        controls.addWidget(self.trans_temp)
        controls.addWidget(QLabel("Top-k:"))
        controls.addWidget(self.trans_topk)
        controls.addWidget(self.trans_button)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.trans_pred_table = QTableWidget()
        self.trans_chart = SimpleCanvas(width=5, height=3)
        left_layout.addWidget(QLabel("<b>Next-token predictions</b>"))
        left_layout.addWidget(self.trans_pred_table, 1)
        left_layout.addWidget(self.trans_chart, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.trans_output = QTextEdit()
        self.trans_output.setReadOnly(True)
        self.trans_eval = QTextEdit()
        self.trans_eval.setReadOnly(True)
        self.trans_eval.setPlainText(load_text_file(TRANSFORMER_DIR / "tiny_transformer_eval.txt",
                                                    "No tiny_transformer_eval.txt found. Train the transformer first."))
        right_layout.addWidget(QLabel("<b>Generated continuation</b>"))
        right_layout.addWidget(self.trans_output, 2)
        right_layout.addWidget(QLabel("<b>Transformer evaluation summary</b>"))
        right_layout.addWidget(self.trans_eval, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        self.run_transformer()
        self.tabs.addTab(tab, "3. Tiny Transformer")

    def run_transformer(self):
        prompt = self.selected_prompt(self.trans_prompt)
        if not self.transformer.loaded:
            self.trans_output.setPlainText(f"Transformer model is not loaded.\n\n{self.transformer.error}")
            set_table_rows(self.trans_pred_table, [], ["rank", "token", "probability"])
            self.trans_chart.bar_chart([], [], "Transformer next-token probabilities", "Probability")
            return
        preds = self.transformer.predict_next(prompt, top_k=12)
        set_table_rows(self.trans_pred_table, preds, ["rank", "token", "probability"])
        self.trans_chart.bar_chart([p["token"] for p in preds[:8]], [p["probability"] for p in preds[:8]],
                                   "Transformer next-token probabilities", "Probability")
        generated = self.transformer.generate(
            prompt,
            temperature=self.trans_temp.value(),
            top_k=self.trans_topk.value(),
            seed=7,
        )
        self.trans_output.setPlainText(generated)

    # -----------------------------------------------------------------
    # Tab 4 Comparison
    # -----------------------------------------------------------------

    def build_compare_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("<h2>Compare Models: local pattern vs. story context</h2>")
        layout.addWidget(title)
        note = QLabel(
            "Use the same prompt for both models. Ask whether each model preserves the object, characters, "
            "pronouns, style, and ending. This is the chapter's main comparison."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        controls = QHBoxLayout()
        self.compare_prompt = QComboBox()
        self.compare_prompt.setEditable(True)
        for name, prompt in DIAGNOSTIC_PROMPTS.items():
            self.compare_prompt.addItem(f"{name}: {prompt}", prompt)
        self.compare_temp = QDoubleSpinBox()
        self.compare_temp.setRange(0.0, 2.0)
        self.compare_temp.setSingleStep(0.1)
        self.compare_temp.setValue(0.8)
        self.compare_button = QPushButton("Compare")
        self.compare_button.clicked.connect(self.run_compare)
        controls.addWidget(QLabel("Prompt:"))
        controls.addWidget(self.compare_prompt, 3)
        controls.addWidget(QLabel("Temp:"))
        controls.addWidget(self.compare_temp)
        controls.addWidget(self.compare_button)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 2)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.compare_ngram_preds = QTableWidget()
        self.compare_ngram_text = QTextEdit()
        self.compare_ngram_text.setReadOnly(True)
        left_layout.addWidget(QLabel("<h3>N-gram model</h3>"))
        left_layout.addWidget(QLabel("Prediction from local count patterns"))
        left_layout.addWidget(self.compare_ngram_preds, 1)
        left_layout.addWidget(self.compare_ngram_text, 2)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.compare_trans_preds = QTableWidget()
        self.compare_trans_text = QTextEdit()
        self.compare_trans_text.setReadOnly(True)
        right_layout.addWidget(QLabel("<h3>Tiny transformer</h3>"))
        right_layout.addWidget(QLabel("Prediction from wider causal context"))
        right_layout.addWidget(self.compare_trans_preds, 1)
        right_layout.addWidget(self.compare_trans_text, 2)
        splitter.addWidget(right)

        bottom = QGroupBox("Student evaluation checklist")
        bottom_layout = QVBoxLayout(bottom)
        self.checks = []
        for label in [
            "The output preserves the main object.",
            "The output keeps pronouns reasonably consistent.",
            "The output keeps the same simple Tiny Reader style.",
            "The output reaches a natural story ending.",
            "The output is locally fluent but globally confused.",
        ]:
            cb = QCheckBox(label)
            bottom_layout.addWidget(cb)
            self.checks.append(cb)
        layout.addWidget(bottom, 1)

        self.run_compare()
        self.tabs.addTab(tab, "4. Compare Models")

    def run_compare(self):
        prompt = self.selected_prompt(self.compare_prompt)
        temp = self.compare_temp.value()

        n_preds = self.ngram.predict_next(prompt, order=5, top_k=8)
        set_table_rows(self.compare_ngram_preds, n_preds, ["rank", "token", "probability", "used_order", "context"])
        n_gen = self.ngram.generate(prompt, order=5, temperature=temp, seed=7) if self.ngram.loaded else "[N-gram not loaded.]"
        self.compare_ngram_text.setPlainText(n_gen)

        if self.transformer.loaded:
            t_preds = self.transformer.predict_next(prompt, top_k=8)
            t_gen = self.transformer.generate(prompt, temperature=temp, top_k=20, seed=7)
        else:
            t_preds = []
            t_gen = f"[Transformer not loaded. {self.transformer.error}]"
        set_table_rows(self.compare_trans_preds, t_preds, ["rank", "token", "probability"])
        self.compare_trans_text.setPlainText(t_gen)



    # -----------------------------------------------------------------
    # Tab 5 You Try It
    # -----------------------------------------------------------------

    def build_try_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("<h2>You Try It: prompt the tiny transformer</h2>")
        layout.addWidget(title)

        note = QLabel(
            "This free-form lab space uses only the tiny transformer model. "
            "Write your own prompt, vary its length and style, and ask what kind of Tiny Reader story the model can produce. "
            "Try short prompts, full story prefixes, ambiguous prompts, and prompts that change the object or setting."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        top = QHBoxLayout()
        self.try_template_combo = QComboBox()
        self.try_template_combo.addItem("Blank prompt", "")
        self.try_template_combo.addItem(
            "Short opening: Jane goes to the",
            "jane goes to the"
        )
        self.try_template_combo.addItem(
            "Object persistence: blue apple",
            "father goes to the yard. mother is near the tree. father has a blue apple. he drops the"
        )
        self.try_template_combo.addItem(
            "Object persistence: yellow toy",
            "jane goes to the yard. puff comes with her. jane has a yellow toy. she drops the"
        )
        self.try_template_combo.addItem(
            "Ambiguous two-object prompt",
            "father goes to the yard. mother is near the tree. father has a blue apple. mother has a red ball. he drops the"
        )
        self.try_template_combo.addItem(
            "Very short character prompt",
            "tim"
        )
        self.try_template_combo.addItem(
            "Level 4 style: wants to",
            "when jane goes to the store, tim is there. jane has a blue ball. tim sees the ball. he wants to"
        )

        self.try_load_template_button = QPushButton("Load Template")
        self.try_load_template_button.clicked.connect(self.load_try_template)

        self.try_run_button = QPushButton("Generate")
        self.try_run_button.clicked.connect(self.run_try_generation)

        top.addWidget(QLabel("Starting point:"))
        top.addWidget(self.try_template_combo, 2)
        top.addWidget(self.try_load_template_button)
        top.addWidget(self.try_run_button)
        layout.addLayout(top)

        settings = QHBoxLayout()
        self.try_num_samples = QSpinBox()
        self.try_num_samples.setRange(1, 10)
        self.try_num_samples.setValue(3)

        self.try_max_tokens = QSpinBox()
        self.try_max_tokens.setRange(5, 200)
        self.try_max_tokens.setValue(80)

        self.try_temperature = QDoubleSpinBox()
        self.try_temperature.setRange(0.0, 2.0)
        self.try_temperature.setSingleStep(0.1)
        self.try_temperature.setValue(0.8)

        self.try_topk = QSpinBox()
        self.try_topk.setRange(1, 100)
        self.try_topk.setValue(20)

        self.try_seed = QSpinBox()
        self.try_seed.setRange(0, 999999)
        self.try_seed.setValue(7)

        settings.addWidget(QLabel("Samples:"))
        settings.addWidget(self.try_num_samples)
        settings.addWidget(QLabel("Max new tokens:"))
        settings.addWidget(self.try_max_tokens)
        settings.addWidget(QLabel("Temperature:"))
        settings.addWidget(self.try_temperature)
        settings.addWidget(QLabel("Top-k:"))
        settings.addWidget(self.try_topk)
        settings.addWidget(QLabel("Seed:"))
        settings.addWidget(self.try_seed)
        settings.addStretch(1)
        layout.addLayout(settings)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("<b>Your prompt</b>"))
        self.try_prompt_edit = QTextEdit()
        self.try_prompt_edit.setPlainText(
            "father goes to the yard. mother is near the tree. father has a blue apple. he drops the"
        )
        left_layout.addWidget(self.try_prompt_edit, 2)

        self.try_pred_table = QTableWidget()
        left_layout.addWidget(QLabel("<b>Next-token prediction from your prompt</b>"))
        left_layout.addWidget(self.try_pred_table, 2)

        self.try_chart = SimpleCanvas(width=5, height=3)
        left_layout.addWidget(self.try_chart, 2)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        right_layout.addWidget(QLabel("<b>Generated samples</b>"))
        self.try_output = QTextEdit()
        self.try_output.setReadOnly(True)
        right_layout.addWidget(self.try_output, 4)

        guidance = QTextEdit()
        guidance.setReadOnly(True)
        guidance.setHtml("""
        <h3>Things to try</h3>
        <ul>
            <li><b>Change the object:</b> apple → toy, ball, kite, basket.</li>
            <li><b>Change the character:</b> Father → Tim, Mother → Sally, Jane → Dick.</li>
            <li><b>Use a very short prompt:</b> <code>Tim goes to the</code>.</li>
            <li><b>Use a long prompt:</b> include the character, object, action, and setting.</li>
            <li><b>Make the prompt ambiguous:</b> include two objects before <code>drops the</code>.</li>
            <li><b>Raise temperature:</b> more variety, but often less coherence.</li>
            <li><b>Lower temperature:</b> safer and more repetitive.</li>
        </ul>
        <p><b>Reflection:</b> Does the model preserve the object? Do the pronouns make sense?
        Does the story end naturally? Is it fluent without really understanding?</p>
        """)
        right_layout.addWidget(QLabel("<b>Lab guidance</b>"))
        right_layout.addWidget(guidance, 2)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        self.run_try_generation()
        self.tabs.addTab(tab, "5. You Try It")

    def load_try_template(self):
        prompt = self.try_template_combo.currentData()
        self.try_prompt_edit.setPlainText(str(prompt or ""))

    def run_try_generation(self):
        prompt = self.try_prompt_edit.toPlainText().strip()
        if not prompt:
            self.try_output.setPlainText("Type a prompt first, or choose a template.")
            set_table_rows(self.try_pred_table, [], ["rank", "token", "probability"])
            self.try_chart.bar_chart([], [], "Transformer next-token probabilities", "Probability")
            return

        if not self.transformer.loaded:
            self.try_output.setPlainText(f"Transformer model is not loaded.\n\n{self.transformer.error}")
            set_table_rows(self.try_pred_table, [], ["rank", "token", "probability"])
            self.try_chart.bar_chart([], [], "Transformer next-token probabilities", "Probability")
            return

        preds = self.transformer.predict_next(prompt, top_k=12)
        set_table_rows(self.try_pred_table, preds, ["rank", "token", "probability"])
        self.try_chart.bar_chart(
            [p["token"] for p in preds[:8]],
            [p["probability"] for p in preds[:8]],
            "Transformer next-token probabilities",
            "Probability"
        )

        samples = []
        n = self.try_num_samples.value()
        for i in range(n):
            seed = self.try_seed.value() + i
            generated = self.transformer.generate(
                prompt,
                max_new_tokens=self.try_max_tokens.value(),
                temperature=self.try_temperature.value(),
                top_k=self.try_topk.value(),
                seed=seed,
            )
            samples.append(f"--- sample {i+1} | seed {seed}\n{generated}")

        self.try_output.setPlainText("\n\n".join(samples))



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    app = QApplication(sys.argv)
    window = CobberHumTinyLMApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
