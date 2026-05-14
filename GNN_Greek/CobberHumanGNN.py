#!/usr/bin/env python3
"""
CobberHumanGNN_v3.py

Student-facing PyQt6 GUI for graph theory, humanities networks, missing-feature
prediction, and GNN-style message passing.

Install:
    pip install PyQt6 pandas numpy matplotlib networkx

Run:
    python CobberHumanGNN_v3.py

Expected folder:
    CobberHumanGNNData/ beside this script.
"""
from __future__ import annotations

import sys, math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import networkx as nx

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QComboBox, QSplitter, QTableWidget, QTableWidgetItem,
    QPushButton, QDoubleSpinBox, QSpinBox, QListWidget, QListWidgetItem,
    QLineEdit, QCheckBox, QMessageBox, QFileDialog
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

FEATURES_18 = [
    "Divine", "Mortal", "Monster", "Place", "Object_Structure", "Heroic",
    "Political", "Familial", "Helper", "Opponent", "Traveler", "Host_Guest",
    "Prophecy_Fate", "Underworld_Death", "Sea_Water", "Domestic_Homecoming",
    "Violence_War", "Transformation_Magic",
]
THESEUS_FEATURES = ["Heroic", "Dangerous", "Political", "Familial", "Sacred_Mythic"]
LAYOUTS = ["spring", "kamada_kawai", "circular", "shell", "spectral"]

THESEUS_STORY_HTML = """
<h2>Theseus and the Labyrinth</h2>
<p><b><span style='color:#5E0B3B;'>Theseus</span></b> was known in
<b><span style='color:#A87400;'>Athens</span></b> as the
<span style='color:#2D6A4F;'><b>son of</b></span>
<b><span style='color:#5E0B3B;'>Aegeus</span></b>, the city's king.
But <b><span style='color:#A87400;'>Athens</span></b>
<span style='color:#2D6A4F;'><b>lived under the shadow of</b></span>
<b><span style='color:#A87400;'>Crete</span></b>. Each year, the Athenians remembered the terrible demand of
<b><span style='color:#5E0B3B;'>Minos</span></b>,
<span style='color:#2D6A4F;'><b>ruler of</b></span>
<b><span style='color:#A87400;'>Crete</span></b>: young Athenians were
<span style='color:#2D6A4F;'><b>sent across the sea as tribute</b></span>.</p>
<p>In <b><span style='color:#A87400;'>Crete</span></b>, the captives
<span style='color:#2D6A4F;'><b>faced</b></span>
<b><span style='color:#5E0B3B;'>the Minotaur</span></b>, a dangerous being
<span style='color:#2D6A4F;'><b>hidden inside</b></span>
<i><span style='color:#005A8B;'>the Labyrinth</span></i>. The Labyrinth had been
<span style='color:#2D6A4F;'><b>designed by</b></span>
<b><span style='color:#5E0B3B;'>Daedalus</span></b>.</p>
<p><b><span style='color:#5E0B3B;'>Theseus</span></b>
<span style='color:#2D6A4F;'><b>volunteered to travel to</b></span>
<b><span style='color:#A87400;'>Crete</span></b>. When he arrived,
<b><span style='color:#5E0B3B;'>Ariadne</span></b>,
<span style='color:#2D6A4F;'><b>daughter of</b></span>
<b><span style='color:#5E0B3B;'>Minos</span></b>, chose to
<span style='color:#2D6A4F;'><b>help</b></span> him. She gave him a thread so that he could
<span style='color:#2D6A4F;'><b>enter</b></span> the Labyrinth,
<span style='color:#2D6A4F;'><b>confront</b></span> the Minotaur, and
<span style='color:#2D6A4F;'><b>find his way back out</b></span>.</p>
<p>Afterward, Theseus <span style='color:#2D6A4F;'><b>sailed away with</b></span>
Ariadne. In many versions, she was <span style='color:#2D6A4F;'><b>left on</b></span>
<b><span style='color:#A87400;'>Naxos</span></b>. Later traditions
<span style='color:#2D6A4F;'><b>connect</b></span> Ariadne with
<b><span style='color:#5E0B3B;'>Dionysus</span></b>.</p>
"""

def clean(s: str) -> str:
    return str(s).replace("_", " ")

def data_dir_default() -> Path:
    return Path(__file__).resolve().parent / "CobberHumanGNNData"

@dataclass
class Dataset:
    name: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    features: list[str]
    G: nx.Graph
    X0: pd.DataFrame
    X: pd.DataFrame
    pos: dict
    layout: str = "spring"

class GraphCanvas(FigureCanvas):
    def __init__(self, click_callback: Optional[Callable[[str], None]] = None):
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.click_callback = click_callback
        self._pos = {}
        self.mpl_connect("button_press_event", self._click)

    def _click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None or not self._pos:
            return
        best, bestd = None, 1e99
        for n, (x, y) in self._pos.items():
            d = (x - event.xdata)**2 + (y - event.ydata)**2
            if d < bestd:
                best, bestd = n, d
        span = max(abs(self.ax.get_xlim()[1]-self.ax.get_xlim()[0]), abs(self.ax.get_ylim()[1]-self.ax.get_ylim()[0]), 1e-9)
        if best is not None and bestd <= (0.05*span)**2 and self.click_callback:
            self.click_callback(best)

    def draw_graph(self, ds: Dataset, selected=None, feature=None, radius=None, title="", node_scale=1.0):
        self.ax.clear()
        G = ds.G
        if selected and selected in G and radius is not None:
            keep = nx.single_source_shortest_path_length(G, selected, cutoff=radius).keys()
            H = G.subgraph(keep).copy()
        else:
            H = G
        if len(H) == 0:
            self.ax.text(.5, .5, "No graph", ha="center", va="center")
            self.draw(); return
        pos = {n: ds.pos[n] for n in H.nodes() if n in ds.pos}
        self._pos = pos.copy()
        nodes = list(H.nodes())
        colors = []
        if feature and feature in ds.X.columns:
            vals = ds.X.reindex(nodes)[feature].fillna(0).astype(float)
            vmax = max(1.0, float(vals.max()))
            for n in nodes:
                v = float(ds.X.loc[n, feature]) if n in ds.X.index else 0.0
                a = min(1.0, v/vmax)
                colors.append((0.92-0.45*a, 0.82-0.70*a, 0.88-0.50*a))
        else:
            for n in nodes:
                if n in ds.X.index and "Place" in ds.X.columns and ds.X.loc[n,"Place"] >= 1.5:
                    colors.append("#FFF1CC")
                elif n in ds.X.index and "Monster" in ds.X.columns and ds.X.loc[n,"Monster"] >= 1.5:
                    colors.append("#F4D6D6")
                else:
                    colors.append("#F5EEF3")
        weights = [float(H.edges[u,v].get("weight", 1.0)) for u,v in H.edges()]
        maxw = max(weights) if weights else 1.0
        widths = [0.4 + 2.8*(w/maxw) for w in weights]
        base_sizes = [360 + min(620, H.degree(n)*18) for n in nodes]
        sizes = [s * float(node_scale) for s in base_sizes]
        if selected in nodes:
            sizes[nodes.index(selected)] = 1100 * float(node_scale)
        nx.draw_networkx_edges(H, pos, ax=self.ax, width=widths, alpha=.34, edge_color="#333333")
        nx.draw_networkx_nodes(H, pos, ax=self.ax, node_color=colors, node_size=sizes,
                               edgecolors="#5E0B3B", linewidths=1.0)
        if selected in nodes:
            nx.draw_networkx_nodes(H, pos, nodelist=[selected], ax=self.ax, node_color="#D6A02B",
                                   node_size=1150*float(node_scale), edgecolors="#5E0B3B", linewidths=2.6)
        if len(H) <= 55:
            labels = {n:n for n in H.nodes()}
        else:
            labels = {n:n for n,_ in sorted(H.degree(), key=lambda x:x[1], reverse=True)[:22]}
            if selected in H: labels[selected] = selected
        nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, ax=self.ax)
        self.ax.set_title(title, fontsize=11, color="#5E0B3B")
        self.ax.set_axis_off()
        self.fig.tight_layout()
        self.draw()

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5,3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
    def plot(self, rounds, errors, diversity):
        self.ax.clear()
        if rounds:
            self.ax.plot(rounds, errors, marker="o", label="prediction error")
            self.ax.plot(rounds, diversity, marker="s", label="feature diversity")
            self.ax.set_xlabel("message-passing rounds")
            self.ax.set_ylabel("value")
            self.ax.set_title("Error and over-smoothing")
            self.ax.legend()
        else:
            self.ax.text(.5,.5,"Run an experiment", ha="center", va="center")
        self.fig.tight_layout(); self.draw()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CobberHumanGNN v3")
        self.resize(1500, 900)
        self.data_dir = data_dir_default()
        self.theseus = self.build_theseus()
        self.ds: Optional[Dataset] = None
        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs)
        self.build_menu(); self.build_tabs(); self.load_dataset("odyssey_teaching_100")

    def build_menu(self):
        m = self.menuBar().addMenu("File")
        a = QAction("Choose Data Folder...", self); a.triggered.connect(self.choose_folder); m.addAction(a)
        r = QAction("Reload Current Dataset", self); r.triggered.connect(lambda: self.load_dataset(self.ds.name if self.ds else "odyssey_teaching_100")); m.addAction(r)
        q = QAction("Quit", self); q.triggered.connect(self.close); m.addAction(q)

    def build_tabs(self):
        self.t_story=QWidget(); self.t_feat=QWidget(); self.t_msg=QWidget(); self.t_miss=QWidget(); self.t_explore=QWidget(); self.t_about=QWidget()
        self.tabs.addTab(self.t_story,"1. Theseus Story")
        self.tabs.addTab(self.t_feat,"2. Features")
        self.tabs.addTab(self.t_msg,"3. Message Passing")
        self.tabs.addTab(self.t_miss,"4. Missing Features")
        self.tabs.addTab(self.t_explore,"5. Larger Graph")
        self.tabs.addTab(self.t_about,"About / Lab Guide")
        self.story_tab(); self.features_tab(); self.message_tab(); self.missing_tab(); self.explore_tab(); self.about_tab()

    def dataset_combo(self):
        c=QComboBox(); c.addItems(["theseus_small","odyssey_teaching_100","odyssey_full","mythology_full"]); c.currentTextChanged.connect(self.load_dataset); return c

    def story_tab(self):
        lay=QVBoxLayout(self.t_story)
        lab=QLabel("<b>Theseus story graph.</b> This tab is intentionally fixed to the small hand-built graph. Use the other tabs to repeat the same calculations for Theseus or move on to Odyssey/full datasets."); lab.setWordWrap(True); lay.addWidget(lab)
        row=QHBoxLayout(); row.addWidget(QLabel("Layout:")); self.story_layout=QComboBox(); self.story_layout.addItems(LAYOUTS); self.story_layout.currentTextChanged.connect(self.refresh_story); row.addWidget(self.story_layout)
        row.addWidget(QLabel("Color by:")); self.story_feature=QComboBox(); self.story_feature.addItems(THESEUS_FEATURES); self.story_feature.setCurrentText("Political"); self.story_feature.currentTextChanged.connect(self.refresh_story); row.addWidget(self.story_feature)
        row.addWidget(QLabel("Node size %:")); self.story_node_size=QSpinBox(); self.story_node_size.setRange(1,150); self.story_node_size.setValue(100); self.story_node_size.valueChanged.connect(self.refresh_story); row.addWidget(self.story_node_size)
        row.addStretch(); lay.addLayout(row)
        spl=QSplitter(Qt.Orientation.Horizontal); lay.addWidget(spl,1)
        left=QWidget(); ll=QVBoxLayout(left); self.story_text=QTextEdit(); self.story_text.setReadOnly(True); self.story_text.setHtml(THESEUS_STORY_HTML); ll.addWidget(QLabel("<b>Reading</b>")); ll.addWidget(self.story_text,2)
        self.story_edges=QTableWidget(); ll.addWidget(QLabel("<b>Relationship table</b>")); ll.addWidget(self.story_edges,1); spl.addWidget(left)
        right=QWidget(); rl=QVBoxLayout(right); self.story_canvas=GraphCanvas(self.story_clicked); rl.addWidget(self.story_canvas,1); self.story_info=QTextEdit(); self.story_info.setReadOnly(True); self.story_info.setMaximumHeight(230); rl.addWidget(QLabel("<b>Clicked node</b>")); rl.addWidget(self.story_info); spl.addWidget(right); spl.setSizes([650,820])
        self.populate_story_edges(); self.refresh_story("Theseus")

    def features_tab(self):
        lay=QVBoxLayout(self.t_feat); row=QHBoxLayout(); row.addWidget(QLabel("Dataset:")); self.feat_dataset=self.dataset_combo(); row.addWidget(self.feat_dataset)
        row.addWidget(QLabel("Selected node:")); self.feat_node=QComboBox(); self.feat_node.currentTextChanged.connect(self.refresh_features); row.addWidget(self.feat_node)
        row.addWidget(QLabel("Layout:")); self.feat_layout=QComboBox(); self.feat_layout.addItems(LAYOUTS); self.feat_layout.currentTextChanged.connect(self.change_layout); row.addWidget(self.feat_layout)
        row.addWidget(QLabel("Node size %:")); self.feat_node_size=QSpinBox(); self.feat_node_size.setRange(1,150); self.feat_node_size.setValue(100); self.feat_node_size.valueChanged.connect(self.refresh_features); row.addWidget(self.feat_node_size)
        self.btn_apply=QPushButton("Apply table edits"); self.btn_apply.clicked.connect(self.apply_feature_edits); row.addWidget(self.btn_apply)
        self.btn_reset=QPushButton("Reset features"); self.btn_reset.clicked.connect(self.reset_features); row.addWidget(self.btn_reset); row.addStretch(); lay.addLayout(row)
        spl=QSplitter(Qt.Orientation.Horizontal); lay.addWidget(spl,1)
        left=QWidget(); ll=QVBoxLayout(left); self.feature_table=QTableWidget(); ll.addWidget(QLabel("<b>Node feature table</b>")); ll.addWidget(self.feature_table); spl.addWidget(left)
        right=QWidget(); rl=QVBoxLayout(right); self.feat_canvas=GraphCanvas(self.feature_clicked); rl.addWidget(self.feat_canvas,1)
        cr=QHBoxLayout(); cr.addWidget(QLabel("Color by:")); self.feat_color=QComboBox(); self.feat_color.currentTextChanged.connect(self.refresh_features); cr.addWidget(self.feat_color); cr.addStretch(); rl.addLayout(cr)
        self.feat_info=QTextEdit(); self.feat_info.setReadOnly(True); self.feat_info.setMaximumHeight(230); rl.addWidget(self.feat_info); spl.addWidget(right); spl.setSizes([770,700])

    def message_tab(self):
        lay=QVBoxLayout(self.t_msg); row=QHBoxLayout(); row.addWidget(QLabel("Dataset:")); self.msg_dataset=self.dataset_combo(); row.addWidget(self.msg_dataset)
        row.addWidget(QLabel("Node:")); self.msg_node=QComboBox(); self.msg_node.currentTextChanged.connect(self.refresh_message); row.addWidget(self.msg_node)
        row.addWidget(QLabel("Self:")); self.msg_self=QDoubleSpinBox(); self.msg_self.setRange(0,1); self.msg_self.setSingleStep(.05); self.msg_self.setValue(.65); self.msg_self.valueChanged.connect(self.sync_msg_weight); row.addWidget(self.msg_self)
        row.addWidget(QLabel("Neighbor:")); self.msg_neigh=QDoubleSpinBox(); self.msg_neigh.setRange(0,1); self.msg_neigh.setSingleStep(.05); self.msg_neigh.setValue(.35); self.msg_neigh.valueChanged.connect(self.refresh_message); row.addWidget(self.msg_neigh)
        row.addWidget(QLabel("Rounds:")); self.msg_rounds=QSpinBox(); self.msg_rounds.setRange(1,10); self.msg_rounds.setValue(1); row.addWidget(self.msg_rounds)
        b=QPushButton("Run rounds"); b.clicked.connect(self.run_rounds); row.addWidget(b); br=QPushButton("Reset"); br.clicked.connect(self.reset_features); row.addWidget(br); row.addStretch(); lay.addLayout(row)
        spl=QSplitter(Qt.Orientation.Horizontal); lay.addWidget(spl,1)
        left=QWidget(); ll=QVBoxLayout(left); self.msg_canvas=GraphCanvas(self.message_clicked); ll.addWidget(self.msg_canvas,1)
        cr=QHBoxLayout(); cr.addWidget(QLabel("Layout:")); self.msg_layout=QComboBox(); self.msg_layout.addItems(LAYOUTS); self.msg_layout.currentTextChanged.connect(self.change_layout); cr.addWidget(self.msg_layout)
        cr.addWidget(QLabel("Node size %:")); self.msg_node_size=QSpinBox(); self.msg_node_size.setRange(1,150); self.msg_node_size.setValue(100); self.msg_node_size.valueChanged.connect(self.refresh_message); cr.addWidget(self.msg_node_size)
        cr.addWidget(QLabel("Color:")); self.msg_color=QComboBox(); self.msg_color.currentTextChanged.connect(self.refresh_message); cr.addWidget(self.msg_color)
        self.msg_ego=QCheckBox("Ego graph"); self.msg_ego.setChecked(True); self.msg_ego.stateChanged.connect(self.refresh_message); cr.addWidget(self.msg_ego)
        cr.addWidget(QLabel("Radius:")); self.msg_radius=QSpinBox(); self.msg_radius.setRange(1,4); self.msg_radius.setValue(1); self.msg_radius.valueChanged.connect(self.refresh_message); cr.addWidget(self.msg_radius); cr.addStretch(); ll.addLayout(cr); spl.addWidget(left)
        right=QWidget(); rl=QVBoxLayout(right); self.msg_text=QTextEdit(); self.msg_text.setReadOnly(True); rl.addWidget(QLabel("<b>Calculation</b>")); rl.addWidget(self.msg_text,2); self.msg_table=QTableWidget(); rl.addWidget(QLabel("<b>Before/after preview</b>")); rl.addWidget(self.msg_table,1); spl.addWidget(right); spl.setSizes([780,680])

    def missing_tab(self):
        lay=QVBoxLayout(self.t_miss); row=QHBoxLayout(); row.addWidget(QLabel("Dataset:")); self.miss_dataset=self.dataset_combo(); row.addWidget(self.miss_dataset)
        row.addWidget(QLabel("Preset:")); self.miss_preset=QComboBox(); self.miss_preset.addItems(["Theseus story nodes","Odyssey teaching examples","Use checked nodes below"]); self.miss_preset.currentTextChanged.connect(self.apply_missing_preset); row.addWidget(self.miss_preset)
        row.addWidget(QLabel("Self:")); self.miss_self=QDoubleSpinBox(); self.miss_self.setRange(0,1); self.miss_self.setSingleStep(.05); self.miss_self.setValue(0.0); row.addWidget(self.miss_self)
        row.addWidget(QLabel("Neighbor:")); self.miss_neigh=QDoubleSpinBox(); self.miss_neigh.setRange(0,1); self.miss_neigh.setSingleStep(.05); self.miss_neigh.setValue(1.0); row.addWidget(self.miss_neigh)
        row.addWidget(QLabel("Max rounds:")); self.miss_rounds=QSpinBox(); self.miss_rounds.setRange(1,10); self.miss_rounds.setValue(6); row.addWidget(self.miss_rounds)
        b=QPushButton("Run missing-feature prediction"); b.clicked.connect(self.run_missing); row.addWidget(b); row.addStretch(); lay.addLayout(row)
        spl=QSplitter(Qt.Orientation.Horizontal); lay.addWidget(spl,1)
        left=QWidget(); ll=QVBoxLayout(left); help=QTextEdit(); help.setReadOnly(True); help.setMaximumHeight(170); help.setHtml("<b>Missing-feature idea:</b> Hide selected node features. Predict them from neighbors by message passing. Compare predictions to hidden answer-key values. This is feature imputation on a graph."); ll.addWidget(help)
        self.miss_nodes=QListWidget(); ll.addWidget(QLabel("<b>Nodes to hide</b>")); ll.addWidget(self.miss_nodes,1); spl.addWidget(left)
        right=QWidget(); rl=QVBoxLayout(right); self.miss_plot=PlotCanvas(); rl.addWidget(self.miss_plot,1); self.miss_table=QTableWidget(); rl.addWidget(QLabel("<b>Prediction results</b>")); rl.addWidget(self.miss_table,1); self.miss_text=QTextEdit(); self.miss_text.setReadOnly(True); self.miss_text.setMaximumHeight(185); rl.addWidget(self.miss_text); spl.addWidget(right); spl.setSizes([420,1020])

    def explore_tab(self):
        lay=QVBoxLayout(self.t_explore); row=QHBoxLayout(); row.addWidget(QLabel("Dataset:")); self.exp_dataset=self.dataset_combo(); row.addWidget(self.exp_dataset)
        row.addWidget(QLabel("Search:")); self.search=QLineEdit(); self.search.textChanged.connect(self.filter_nodes); row.addWidget(self.search)
        row.addWidget(QLabel("Layout:")); self.exp_layout=QComboBox(); self.exp_layout.addItems(LAYOUTS); self.exp_layout.currentTextChanged.connect(self.change_layout); row.addWidget(self.exp_layout)
        row.addWidget(QLabel("Node size %:")); self.exp_node_size=QSpinBox(); self.exp_node_size.setRange(1,150); self.exp_node_size.setValue(100); self.exp_node_size.valueChanged.connect(self.refresh_explore); row.addWidget(self.exp_node_size)
        row.addWidget(QLabel("Radius:")); self.exp_radius=QSpinBox(); self.exp_radius.setRange(1,4); self.exp_radius.setValue(1); self.exp_radius.valueChanged.connect(self.refresh_explore); row.addWidget(self.exp_radius)
        row.addWidget(QLabel("Color:")); self.exp_color=QComboBox(); self.exp_color.currentTextChanged.connect(self.refresh_explore); row.addWidget(self.exp_color); row.addStretch(); lay.addLayout(row)
        spl=QSplitter(Qt.Orientation.Horizontal); lay.addWidget(spl,1)
        left=QWidget(); ll=QVBoxLayout(left); self.node_list=QListWidget(); self.node_list.currentItemChanged.connect(lambda *_: self.refresh_explore()); ll.addWidget(QLabel("<b>Nodes</b>")); ll.addWidget(self.node_list,1); self.node_info=QTextEdit(); self.node_info.setReadOnly(True); self.node_info.setMaximumHeight(290); ll.addWidget(self.node_info); spl.addWidget(left)
        right=QWidget(); rl=QVBoxLayout(right); self.exp_canvas=GraphCanvas(self.explore_clicked); rl.addWidget(self.exp_canvas,1); self.neighbor_table=QTableWidget(); rl.addWidget(QLabel("<b>Neighbors / edges</b>")); rl.addWidget(self.neighbor_table,1); spl.addWidget(right); spl.setSizes([390,1050])

    def about_tab(self):
        lay=QVBoxLayout(self.t_about); t=QTextEdit(); t.setReadOnly(True); t.setHtml("""
        <h2>CobberHumanGNN v3</h2>
        <p><b>Graph theory:</b> a node is a thing; an edge is a relationship; a weighted edge says some relationships are stronger than others; a neighborhood is the set of directly connected nodes; a path is a way to move through the graph.</p>
        <p><b>What the graph itself gives us:</b> relationships, neighborhoods, central figures, bridge figures, edge weights, and clusters.</p>
        <p><b>What message passing adds:</b> if node features are missing, neighbors may provide clues. In this simplified model, each round looks only at immediate neighbors. Multiple rounds let information travel farther indirectly.</p>
        <blockquote>new vector = self weight × old vector + neighbor weight × average neighbor vector</blockquote>
        <p><b>Over-smoothing:</b> too many rounds can flatten meaningful difference, making nodes become too similar.</p>
        """); lay.addWidget(t)

    # ---------- data ----------
    def choose_folder(self):
        d=QFileDialog.getExistingDirectory(self,"Choose CobberHumanGNNData folder",str(self.data_dir))
        if d: self.data_dir=Path(d); self.load_dataset(self.ds.name if self.ds else "odyssey_teaching_100")

    def build_theseus(self) -> Dataset:
        rows=[("Theseus","figure","Athenian hero",[2,1,1,1,1]),("Ariadne","figure","Helper from Crete",[0,0,1,1,1]),("Minotaur","monster","Dangerous being in the Labyrinth",[0,2,1,0,2]),("Athens","place","Theseus' civic world",[0,0,2,0,0]),("Crete","place","Minos' kingdom",[0,1,2,0,1]),("Aegeus","figure","King of Athens, father of Theseus",[0,0,1,2,0]),("Minos","figure","King of Crete",[0,1,2,1,1]),("Labyrinth","structure","Built space containing the Minotaur",[0,2,1,0,2]),("Daedalus","figure","Designer of the Labyrinth",[0,0,1,0,1]),("Naxos","place","Later place in Ariadne's story",[0,0,0,0,1]),("Dionysus","deity","Later mythic connection to Ariadne",[0,0,0,0,2])]
        nodes=pd.DataFrame([{"node_id":n,"label":n,"type":t,"description":d,**{f:v[i] for i,f in enumerate(THESEUS_FEATURES)}} for n,t,d,v in rows]).set_index("node_id",drop=False)
        erows=[("Theseus","Ariadne","help / love / abandonment",1),("Theseus","Minotaur","confronts",1),("Theseus","Athens","belongs to / represents",1),("Theseus","Crete","travels to",1),("Theseus","Aegeus","son of",1),("Ariadne","Crete","belongs to",1),("Ariadne","Naxos","left on",1),("Ariadne","Dionysus","later connected to",1),("Minos","Crete","rules",1),("Minotaur","Labyrinth","inhabits",1),("Daedalus","Labyrinth","designed",1),("Minos","Minotaur","contains / controls tribute to",1)]
        edges=pd.DataFrame(erows,columns=["source","target","relation","weight"]); G=nx.Graph()
        for _,r in nodes.iterrows(): G.add_node(r.node_id,label=r.label,node_type=r.type,description=r.description)
        for _,r in edges.iterrows(): G.add_edge(r.source,r.target,relation=r.relation,weight=float(r.weight))
        X=nodes[THESEUS_FEATURES].astype(float).copy(); ds=Dataset("theseus_small",nodes,edges,THESEUS_FEATURES,G,X.copy(),X.copy(),{})
        ds.pos=self.layout_for(G,"spring",True); return ds

    def load_dataset(self,name):
        try:
            if name=="theseus_small":
                self.ds=self.build_theseus(); self.populate_all(); return
            elif name=="odyssey_teaching_100": nf,ef="odyssey_teaching_100_nodes_features.csv","odyssey_teaching_100_edges.csv"
            elif name=="odyssey_full": nf,ef="odyssey_nodes_features_full.csv","odyssey_edges_aggregated.csv"
            elif name=="mythology_full": nf,ef="myth_nodes_features_full.csv","myth_edges_aggregated_full.csv"
            else: return
            self.ds=self.load_csv(name,nf,ef)
            self.populate_all()
        except Exception as e:
            QMessageBox.critical(self,"Could not load dataset",str(e))

    def load_csv(self,name,nf,ef):
        npath=self.data_dir/nf; epath=self.data_dir/ef
        if not npath.exists(): raise FileNotFoundError(f"Missing {npath}")
        if not epath.exists(): raise FileNotFoundError(f"Missing {epath}")
        nodes=pd.read_csv(npath); edges=pd.read_csv(epath)
        for f in FEATURES_18:
            if f not in nodes.columns: nodes[f]=0
        nodes=nodes.drop_duplicates("node_id").set_index("node_id",drop=False)
        G=nx.Graph()
        for _,r in nodes.iterrows():
            nid=str(r.node_id); G.add_node(nid,label=str(r.get("label",nid)),node_type=str(r.get("type",r.get("level2",""))),description=str(r.get("description","")))
        for _,r in edges.iterrows():
            s=str(r.get("source","")).strip(); t=str(r.get("target","")).strip()
            if not s or not t or s.lower()=="nan" or t.lower()=="nan": continue
            if s not in G: G.add_node(s,label=s,node_type="",description="")
            if t not in G: G.add_node(t,label=t,node_type="",description="")
            w=float(r.get("weight",r.get("normalized_weight",1.0)) or 1.0); rel=str(r.get("relation","co-occurs")); cnt=int(float(r.get("edge_count",1) or 1))
            if G.has_edge(s,t): G.edges[s,t]["weight"]+=w; G.edges[s,t]["edge_count"]+=cnt
            else: G.add_edge(s,t,weight=w,relation=rel,edge_count=cnt)
        X=nodes.reindex(G.nodes())[FEATURES_18].fillna(0).astype(float)
        ds=Dataset(name,nodes,edges,FEATURES_18,G,X.copy(),X.copy(),{}); ds.pos=self.layout_for(G,"spring"); return ds

    def layout_for(self,G,kind,fixed_theseus=False):
        if fixed_theseus:
            return {"Theseus":(0,0),"Ariadne":(-2.4,1.4),"Minotaur":(2.4,1.4),"Athens":(-2.4,-1.4),"Crete":(2.4,-1.4),"Aegeus":(-4.6,-.6),"Minos":(4.6,-.6),"Labyrinth":(4.6,1.9),"Daedalus":(3.0,3.3),"Naxos":(-4.5,2.6),"Dionysus":(-2.4,3.3)}
        try:
            if kind=="kamada_kawai": return nx.kamada_kawai_layout(G, weight="weight")
            if kind=="circular": return nx.circular_layout(G)
            if kind=="shell": return nx.shell_layout(G)
            if kind=="spectral": return nx.spectral_layout(G)
            return nx.spring_layout(G, seed=7, k=1.3/math.sqrt(max(1,len(G))), iterations=35 if len(G)>500 else 80, weight="weight")
        except Exception:
            return nx.spring_layout(G, seed=7, iterations=40)

    def change_layout(self):
        if not self.ds: return
        kind=self.sender().currentText(); self.ds.layout=kind; self.ds.pos=self.layout_for(self.ds.G,kind); self.refresh_all()

    def populate_all(self):
        ds=self.ds; nodes=sorted(ds.G.nodes()); preferred="Odysseus" if "Odysseus" in nodes else (nodes[0] if nodes else "")
        for c in [self.feat_dataset,self.msg_dataset,self.miss_dataset,self.exp_dataset]: c.blockSignals(True); c.setCurrentText(ds.name); c.blockSignals(False)
        for c in [self.feat_layout,self.msg_layout,self.exp_layout]: c.blockSignals(True); c.setCurrentText(ds.layout); c.blockSignals(False)
        for c in [self.feat_node,self.msg_node]: c.blockSignals(True); c.clear(); c.addItems(nodes); c.setCurrentText(preferred); c.blockSignals(False)
        for c in [self.feat_color,self.msg_color,self.exp_color]: c.blockSignals(True); c.clear(); c.addItems(ds.features); c.setCurrentText("Political" if "Political" in ds.features else ds.features[0]); c.blockSignals(False)
        self.populate_feature_table(); self.populate_node_list(); self.populate_missing_nodes(); self.apply_missing_preset(); self.refresh_all()

    def populate_feature_table(self):
        ds=self.ds; cols=["node_id"]+ds.features; self.feature_table.setColumnCount(len(cols)); self.feature_table.setHorizontalHeaderLabels(cols); self.feature_table.setRowCount(len(ds.X))
        for i,n in enumerate(ds.X.index):
            it=QTableWidgetItem(n); it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable); self.feature_table.setItem(i,0,it)
            for j,f in enumerate(ds.features,1): self.feature_table.setItem(i,j,QTableWidgetItem(f"{float(ds.X.loc[n,f]):.3g}"))
        self.feature_table.resizeColumnsToContents()

    def populate_story_edges(self):
        ds=self.theseus; cols=["source","relation","target","weight"]; self.story_edges.setColumnCount(4); self.story_edges.setHorizontalHeaderLabels(cols); self.story_edges.setRowCount(len(ds.edges))
        for i,(_,r) in enumerate(ds.edges.iterrows()):
            for j,c in enumerate(cols):
                it=QTableWidgetItem(str(r[c])); it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable); self.story_edges.setItem(i,j,it)
        self.story_edges.resizeColumnsToContents()

    def populate_node_list(self):
        self.node_list.clear();
        for n in sorted(self.ds.G.nodes()): self.node_list.addItem(QListWidgetItem(n))
        if "Odysseus" in self.ds.G:
            m=self.node_list.findItems("Odysseus",Qt.MatchFlag.MatchExactly)
            if m: self.node_list.setCurrentItem(m[0])

    def populate_missing_nodes(self):
        self.miss_nodes.clear()
        for n in sorted(self.ds.G.nodes()):
            it=QListWidgetItem(n); it.setFlags(it.flags()|Qt.ItemFlag.ItemIsUserCheckable); it.setCheckState(Qt.CheckState.Unchecked); self.miss_nodes.addItem(it)

    # ---------- refresh and info ----------
    def refresh_all(self): self.refresh_features(); self.refresh_message(); self.refresh_explore()
    def refresh_story(self,*_):
        kind=self.story_layout.currentText(); self.theseus.pos=self.layout_for(self.theseus.G,kind,fixed_theseus=(kind=="spring")); node=getattr(self,"_story_selected","Theseus"); self.story_canvas.draw_graph(self.theseus,node,self.story_feature.currentText(),None,"Theseus story graph", self.story_node_size.value()/100.0); self.show_info(self.theseus,node,self.story_info)
    def story_clicked(self,node): self._story_selected=node; self.refresh_story()
    def feature_clicked(self,node): self.feat_node.setCurrentText(node); self.refresh_features()
    def message_clicked(self,node): self.msg_node.setCurrentText(node); self.refresh_message()
    def explore_clicked(self,node):
        m=self.node_list.findItems(node,Qt.MatchFlag.MatchExactly)
        if m: self.node_list.setCurrentItem(m[0])
        self.refresh_explore()

    def show_info(self,ds,node,widget):
        if not node or node not in ds.G: widget.setHtml(""); return
        h=[f"<h3>{node}</h3>"]
        if node in ds.nodes.index:
            r=ds.nodes.loc[node]
            for key in ["description","type","level2","level3","gender","residence","feature_source","feature_confidence"]:
                if key in r and str(r.get(key,"")).strip() not in ["","nan","[]"]: h.append(f"<p><b>{key}:</b> {r.get(key)}</p>")
        h.append(f"<p><b>Degree:</b> {ds.G.degree(node)}</p><p><b>Features:</b></p><ul>")
        if node in ds.X.index:
            for f in ds.features:
                v=float(ds.X.loc[node,f])
                if abs(v)>1e-12: h.append(f"<li>{clean(f)}: {v:.3g}</li>")
        h.append("</ul>")
        neigh=list(ds.G.neighbors(node)); h.append(f"<p><b>Neighbors:</b> {', '.join(neigh[:30])}"+(f" ... and {len(neigh)-30} more" if len(neigh)>30 else "")+"</p>")
        widget.setHtml("\n".join(h))

    def refresh_features(self,*_):
        if not self.ds: return
        node=self.feat_node.currentText(); feat=self.feat_color.currentText(); rad=2 if self.ds.name!="mythology_full" else 1
        self.feat_canvas.draw_graph(self.ds,node,feat,rad,f"{self.ds.name}: {clean(feat)}", self.feat_node_size.value()/100.0)
        self.show_info(self.ds,node,self.feat_info)

    def avg_neighbors(self,ds,X,node):
        neigh=list(ds.G.neighbors(node)) if node in ds.G else []
        if not neigh: return pd.Series({f:0.0 for f in ds.features})
        return X.reindex(neigh)[ds.features].fillna(0).mean(axis=0)
    def one_round(self,ds,X,selfw,neighw):
        out=pd.DataFrame(index=X.index,columns=ds.features,dtype=float)
        for n in X.index: out.loc[n,ds.features]=selfw*X.loc[n,ds.features]+neighw*self.avg_neighbors(ds,X,n)
        return out.astype(float)
    def diversity(self,X): return float(X.var(axis=0).mean()) if len(X) else 0.0

    def sync_msg_weight(self): self.msg_neigh.blockSignals(True); self.msg_neigh.setValue(round(1-self.msg_self.value(),3)); self.msg_neigh.blockSignals(False); self.refresh_message()
    def refresh_message(self,*_):
        if not self.ds: return
        node=self.msg_node.currentText(); feat=self.msg_color.currentText(); rad=self.msg_radius.value() if self.msg_ego.isChecked() else None
        self.msg_canvas.draw_graph(self.ds,node,feat,rad,f"Message passing: {node}", self.msg_node_size.value()/100.0)
        if not node or node not in self.ds.X.index: return
        old=self.ds.X.loc[node,self.ds.features].astype(float); avg=self.avg_neighbors(self.ds,self.ds.X,node); a=self.msg_self.value(); b=self.msg_neigh.value(); new=a*old+b*avg
        h=[f"<h3>{node}</h3>",f"<p><b>Rule:</b> new = {a:.2f} × old + {b:.2f} × neighbor average</p>",f"<p><b>Feature diversity now:</b> {self.diversity(self.ds.X[self.ds.features]):.4f}. Lower means more over-smoothing.</p>","<table border='1' cellspacing='0' cellpadding='3'><tr><th>Feature</th><th>Old</th><th>Neighbor avg</th><th>New if run</th><th>Change</th></tr>"]
        for f in self.ds.features: h.append(f"<tr><td>{clean(f)}</td><td>{old[f]:.3g}</td><td>{avg[f]:.3g}</td><td>{new[f]:.3g}</td><td>{new[f]-old[f]:+.3g}</td></tr>")
        h.append("</table>"); self.msg_text.setHtml("\n".join(h))
        self.msg_table.setColumnCount(5); self.msg_table.setHorizontalHeaderLabels(["Feature","Current","Neighbor avg","Preview new","Change"]); self.msg_table.setRowCount(len(self.ds.features))
        for i,f in enumerate(self.ds.features):
            vals=[clean(f),old[f],avg[f],new[f],new[f]-old[f]]
            for j,v in enumerate(vals):
                it=QTableWidgetItem(str(v) if j==0 else f"{float(v):.3f}"); it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable); self.msg_table.setItem(i,j,it)
        self.msg_table.resizeColumnsToContents()
    def run_rounds(self):
        X=self.ds.X[self.ds.features].copy()
        for _ in range(self.msg_rounds.value()): X=self.one_round(self.ds,X,self.msg_self.value(),self.msg_neigh.value())
        self.ds.X.loc[:,self.ds.features]=X; self.populate_feature_table(); self.refresh_all()
    def apply_feature_edits(self):
        try:
            for i in range(self.feature_table.rowCount()):
                n=self.feature_table.item(i,0).text()
                for j,f in enumerate(self.ds.features,1): self.ds.X.loc[n,f]=float(self.feature_table.item(i,j).text())
            self.refresh_all()
        except Exception as e: QMessageBox.warning(self,"Bad feature value",str(e))
    def reset_features(self): self.ds.X=self.ds.X0.copy(); self.populate_feature_table(); self.refresh_all()

    # ---------- missing ----------
    def apply_missing_preset(self,*_):
        for i in range(self.miss_nodes.count()): self.miss_nodes.item(i).setCheckState(Qt.CheckState.Unchecked)
        if self.miss_preset.currentText()=="Odyssey teaching examples": names={"Circe","Calypso","Tiresias","Polyphemus","Nausicaa","Eumaeus","Antinous","Helios"}
        else: names=set()
        for i in range(self.miss_nodes.count()):
            if self.miss_nodes.item(i).text() in names: self.miss_nodes.item(i).setCheckState(Qt.CheckState.Checked)
    def hidden_nodes(self,ds):
        if self.miss_preset.currentText()=="Theseus story nodes": return [n for n in ["Daedalus","Aegeus","Naxos","Dionysus"] if n in ds.G]
        out=[]
        for i in range(self.miss_nodes.count()):
            it=self.miss_nodes.item(i)
            if it.checkState()==Qt.CheckState.Checked and it.text() in ds.G: out.append(it.text())
        return out
    def run_missing(self):
        ds=self.theseus if self.miss_preset.currentText()=="Theseus story nodes" else self.ds
        hidden=self.hidden_nodes(ds)
        if not hidden: QMessageBox.information(self,"No hidden nodes","Choose at least one node to hide."); return
        truth=ds.X0[ds.features].copy(); X=truth.copy(); X.loc[hidden,ds.features]=0.0
        rounds=[]; errs=[]; divs=[]; rows=[]
        for r in range(1,self.miss_rounds.value()+1):
            X=self.one_round(ds,X,self.miss_self.value(),self.miss_neigh.value()); pred=X.loc[hidden,ds.features]; real=truth.loc[hidden,ds.features]
            err=float((pred-real).abs().mean().mean()); rounds.append(r); errs.append(err); divs.append(self.diversity(X))
            for n in hidden: rows.append((r,n,float((pred.loc[n]-real.loc[n]).abs().mean())))
        self.miss_plot.plot(rounds,errs,divs)
        self.miss_table.setColumnCount(3); self.miss_table.setHorizontalHeaderLabels(["Round","Hidden node","Mean abs error"]); self.miss_table.setRowCount(len(rows))
        for i,(r,n,e) in enumerate(rows):
            for j,v in enumerate([r,n,f"{e:.4f}"]):
                it=QTableWidgetItem(str(v)); it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable); self.miss_table.setItem(i,j,it)
        self.miss_table.resizeColumnsToContents()
        bi=int(np.argmin(errs)); self.miss_text.setHtml(f"<h3>Missing-feature experiment</h3><p><b>Hidden nodes:</b> {', '.join(hidden)}</p><p><b>Best round:</b> {rounds[bi]} &nbsp; <b>mean error:</b> {errs[bi]:.4f}</p><p>If error improves at first, the graph helped reconstruct hidden features. If it worsens after more rounds, over-smoothing is flattening differences.</p><p><i>Theseus values are teacher-authored; Odyssey values come from the generated feature table.</i></p>")

    # ---------- explore ----------
    def filter_nodes(self):
        txt=self.search.text().strip().lower(); self.node_list.clear()
        for n in sorted(self.ds.G.nodes()):
            if not txt or txt in n.lower(): self.node_list.addItem(QListWidgetItem(n))
    def refresh_explore(self,*_):
        if not self.ds: return
        cur=self.node_list.currentItem(); node=cur.text() if cur else ("Odysseus" if "Odysseus" in self.ds.G else next(iter(self.ds.G.nodes()),""))
        self.exp_canvas.draw_graph(self.ds,node,self.exp_color.currentText(),self.exp_radius.value(),f"{node}: ego graph", self.exp_node_size.value()/100.0)
        self.show_info(self.ds,node,self.node_info)
        neigh=[]
        if node in self.ds.G:
            for nb in self.ds.G.neighbors(node):
                e=self.ds.G.edges[node,nb]; neigh.append((nb,e.get("relation","co-occurs"),float(e.get("weight",1.0)),self.ds.G.degree(nb)))
            neigh.sort(key=lambda x:(x[2],x[3]), reverse=True)
        self.neighbor_table.setColumnCount(4); self.neighbor_table.setHorizontalHeaderLabels(["Neighbor","Relation","Weight","Neighbor degree"]); self.neighbor_table.setRowCount(len(neigh))
        for i,row in enumerate(neigh):
            for j,v in enumerate(row):
                it=QTableWidgetItem(f"{v:.4g}" if j==2 else str(v)); it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable); self.neighbor_table.setItem(i,j,it)
        self.neighbor_table.resizeColumnsToContents()


def main():
    app=QApplication(sys.argv)
    app.setStyleSheet("""
        QTabBar::tab { padding: 8px 12px; }
        QTabBar::tab:selected { background: #F5EEF3; color: #5E0B3B; font-weight: bold; }
        QPushButton { padding: 5px 9px; }
    """)
    w=App(); w.show(); sys.exit(app.exec())
if __name__=="__main__": main()
