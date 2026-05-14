#!/usr/bin/env python3
"""
CobberHumFetcher.py

A humanities-oriented adaptation of CobberFetcher for corpus building.

Sources:
  - Library of Congress JSON API
  - Gutendex / Project Gutenberg metadata API
  - Internet Archive Advanced Search API
  - World Bank World Development Indicators API

Run:
    python CobberHumFetcher.py

Dependencies:
    pip install PyQt6 requests matplotlib
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus

import requests

from PyQt6.QtCore import Qt, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QTextEdit, QPushButton, QComboBox, QTabWidget,
    QMessageBox, QStatusBar, QSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QGroupBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class WorkerSignals(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)


def safe_join(value: Any, max_items: int = 5) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if "name" in value:
            return str(value["name"])
        return json.dumps(value, ensure_ascii=False)[:300]
    if isinstance(value, list):
        return "; ".join(safe_join(x) for x in value[:max_items] if safe_join(x))
    return str(value)


def truncate(text: str, n: int = 500) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def first_format_url(formats: Dict[str, str], preferred_words: List[str]) -> str:
    if not isinstance(formats, dict):
        return ""
    for word in preferred_words:
        for mime, url in formats.items():
            if word.lower() in mime.lower() and isinstance(url, str):
                return url
    return ""


def parse_wdi_line(line: str) -> Tuple[str, str]:
    cleaned = line.strip()
    cleaned = re.sub(r"^\\s*(country|country_code)\\s*[:=]\\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("\\t", " ")

    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",", 1)]
    elif ";" in cleaned:
        parts = [p.strip() for p in cleaned.split(";", 1)]
    else:
        parts = cleaned.split()

    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            "WDI input needs BOTH a country code and an indicator. "
            "Examples: USA, SP.POP.TOTL   or   SYR, SP.POP.TOTL"
        )
    return parts[0].upper(), parts[1].strip()


class FetcherWorker(QRunnable):
    def __init__(self, source: str, queries: List[str], max_results: int):
        super().__init__()
        self.source = source
        self.queries = queries
        self.max_results = max_results
        self.signals = WorkerSignals()

    def run(self):
        try:
            for query in self.queries:
                query = query.strip()
                if not query:
                    continue

                try:
                    self.signals.progress.emit(f"Fetching from {self.source}: {query}")
                    if self.source == "Library of Congress":
                        data = self.fetch_loc(query)
                    elif self.source == "Gutendex / Project Gutenberg":
                        data = self.fetch_gutendex(query)
                    elif self.source == "Internet Archive":
                        data = self.fetch_internet_archive(query)
                    elif self.source == "World Bank WDI":
                        country, indicator = parse_wdi_line(query)
                        data = self.fetch_wdi(country, indicator)
                    else:
                        data = {"error": f"Unknown source: {self.source}"}

                    if data.get("error"):
                        self.signals.error.emit(data["error"])
                    else:
                        self.signals.result.emit(data)

                except Exception as exc:
                    self.signals.error.emit(f"Could not process '{query}': {exc}")

        except Exception as exc:
            self.signals.error.emit(f"A critical error occurred: {exc}")
        finally:
            self.signals.finished.emit()

    def fetch_loc(self, query: str) -> Dict[str, Any]:
        url = "https://www.loc.gov/search/"
        params = {"fo": "json", "q": query, "c": str(self.max_results)}
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()
            records = []
            for item in payload.get("results", [])[: self.max_results]:
                records.append({
                    "title": safe_join(item.get("title")),
                    "date": safe_join(item.get("date")),
                    "creator": safe_join(item.get("contributor") or item.get("creator")),
                    "subjects": safe_join(item.get("subject")),
                    "description": truncate(safe_join(item.get("description")), 700),
                    "url": safe_join(item.get("url")),
                    "source_id": safe_join(item.get("id")),
                    "raw": item,
                })
            return {"type": "archive", "source": "Library of Congress", "query": query, "records": records, "raw": payload}
        except Exception as exc:
            return {"error": f"Library of Congress fetch failed for '{query}': {exc}"}

    def fetch_gutendex(self, query: str) -> Dict[str, Any]:
        url = "https://gutendex.com/books"
        params = {"search": query}
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()
            records = []
            for item in payload.get("results", [])[: self.max_results]:
                authors = "; ".join(a.get("name", "") for a in item.get("authors", []) if a.get("name"))
                formats = item.get("formats", {})
                txt_url = first_format_url(formats, ["text/plain"])
                html_url = first_format_url(formats, ["text/html"])
                records.append({
                    "title": safe_join(item.get("title")),
                    "date": "",
                    "creator": authors,
                    "subjects": safe_join(item.get("subjects")),
                    "description": f"Languages: {safe_join(item.get('languages'))}; Downloads: {item.get('download_count', '')}",
                    "url": html_url or txt_url or f"https://www.gutenberg.org/ebooks/{item.get('id')}",
                    "source_id": str(item.get("id", "")),
                    "text_url": txt_url,
                    "raw": item,
                })
            return {"type": "archive", "source": "Gutendex / Project Gutenberg", "query": query, "records": records, "raw": payload}
        except Exception as exc:
            return {"error": f"Gutendex fetch failed for '{query}': {exc}"}

    def fetch_internet_archive(self, query: str) -> Dict[str, Any]:
        """
        Internet Archive Advanced Search API.

        The IA advanced-search endpoint can be slow or return 502 on broad
        classroom queries. This version intentionally fails fast so the GUI
        does not appear to hang. It tries a simple metadata query first, then a
        plain broad query as a fallback.
        """
        url = "https://archive.org/advancedsearch.php"
        fields = ["identifier", "title", "creator", "date", "mediatype", "description", "subject"]
        headers = {"User-Agent": "CobberHumFetcher/1.2 educational app"}

        quoted = query.replace('"', "").strip()
        query_attempts = [
            f'title:("{quoted}") OR subject:("{quoted}")',
            query,
        ]

        last_error = None
        for q_attempt in query_attempts:
            params = [("q", q_attempt), ("rows", str(self.max_results)), ("page", "1"), ("output", "json")]
            for field in fields:
                params.append(("fl[]", field))

            try:
                # Short timeout by design: IA is useful, but we do not want a
                # student GUI to sit for several minutes.
                r = requests.get(url, params=params, headers=headers, timeout=(5, 12))
                r.raise_for_status()
                payload = r.json()
                docs = payload.get("response", {}).get("docs", [])
                records = []
                for item in docs[: self.max_results]:
                    identifier = safe_join(item.get("identifier"))
                    records.append({
                        "title": safe_join(item.get("title")),
                        "date": safe_join(item.get("date")),
                        "creator": safe_join(item.get("creator")),
                        "subjects": safe_join(item.get("subject")),
                        "description": truncate(safe_join(item.get("description")), 700),
                        "url": f"https://archive.org/details/{identifier}" if identifier else "",
                        "source_id": identifier,
                        "mediatype": safe_join(item.get("mediatype")),
                        "query_used": q_attempt,
                        "raw": item,
                    })
                return {"type": "archive", "source": "Internet Archive", "query": query, "records": records, "raw": payload}
            except Exception as exc:
                last_error = exc

        return {"error": f"Internet Archive is slow or unavailable for '{query}'. Try a narrower query later, or use Library of Congress/Gutendex for this lab. Last error: {last_error}"}

    def fetch_wdi(self, country: str, indicator: str) -> Dict[str, Any]:
        url = f"https://api.worldbank.org/v2/country/{quote_plus(country)}/indicator/{quote_plus(indicator)}"
        params = {"format": "json", "per_page": "20000"}
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list) or len(payload) < 2:
                return {"error": f"Unexpected World Bank response for {country}, {indicator}. Check country code and indicator code."}
            rows_raw = payload[1] or []
            rows = []
            indicator_name = ""
            country_name = country
            for item in rows_raw:
                value = item.get("value")
                if value is None:
                    continue
                try:
                    value = float(value)
                except Exception:
                    continue
                year = int(item.get("date"))
                indicator_name = item.get("indicator", {}).get("value", indicator_name)
                country_name = item.get("country", {}).get("value", country_name)
                rows.append({
                    "year": year,
                    "value": value,
                    "country": country_name,
                    "country_code": country,
                    "indicator": indicator,
                    "indicator_name": indicator_name,
                })
            rows.sort(key=lambda x: x["year"])
            if not rows:
                return {"error": f"World Bank WDI returned no numeric values for {country}, {indicator}. Check the country code and indicator code."}
            return {
                "type": "wdi", "source": "World Bank WDI", "query": f"{country}, {indicator}",
                "country": country, "country_name": country_name,
                "indicator": indicator, "indicator_name": indicator_name or indicator,
                "records": rows, "raw": payload, "meta": payload[0],
            }
        except Exception as exc:
            return {"error": f"World Bank WDI fetch failed for '{country}, {indicator}': {exc}"}


class PlotCanvas(FigureCanvas):
    def __init__(self, width=6, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)

    def plot_wdi(self, rows: List[Dict[str, Any]], title: str):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if rows:
            years = [r["year"] for r in rows]
            values = [r["value"] for r in rows]
            ax.plot(years, values, marker="o", linewidth=1.5, markersize=3)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No data returned", ha="center", va="center")
            ax.set_axis_off()
        self.figure.tight_layout()
        self.draw()


class CobberHumFetcherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cobber_maroon = QColor(108, 29, 69)
        self.lato_font = QFont("Lato")
        self.setWindowTitle("CobberHumFetcher")
        self.setGeometry(100, 100, 1350, 850)
        self.setFont(self.lato_font)
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: #f7f3e8; }}
            QLabel {{ color: #111111; }}
            QPushButton {{
                background-color: {self.cobber_maroon.name()};
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }}
            QPushButton:hover {{ background-color: #8a2a58; }}
            QTextEdit, QComboBox, QTableWidget, QSpinBox {{
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
        self.threadpool = QThreadPool()
        self.result_payloads: List[Dict[str, Any]] = []
        self.build_ui()
        self.update_placeholder()

    def build_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(330)
        left_panel.setMaximumWidth(450)

        title = QLabel("<h2>CobberHumFetcher</h2>")
        subtitle = QLabel("Build small humanities corpora by fetching public metadata, text links, and quantitative context.")
        subtitle.setWordWrap(True)
        left_layout.addWidget(title)
        left_layout.addWidget(subtitle)

        form = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Library of Congress", "Gutendex / Project Gutenberg", "Internet Archive", "World Bank WDI"])
        self.source_combo.currentTextChanged.connect(self.update_placeholder)
        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(1, 50)
        self.max_results_spin.setValue(10)
        form.addRow(QLabel("Source:"), self.source_combo)
        form.addRow(QLabel("Max results:"), self.max_results_spin)
        left_layout.addLayout(form)

        self.input_label = QLabel("Search terms / identifiers:")
        self.input_console = QTextEdit()
        left_layout.addWidget(self.input_label)
        left_layout.addWidget(self.input_console, 2)

        button_row = QHBoxLayout()
        self.fetch_btn = QPushButton("Fetch Data")
        self.clear_btn = QPushButton("Clear All")
        self.save_all_btn = QPushButton("Save All CSV")
        button_row.addWidget(self.fetch_btn)
        button_row.addWidget(self.clear_btn)
        button_row.addWidget(self.save_all_btn)
        left_layout.addLayout(button_row)

        help_box = QGroupBox("Teaching reminder")
        help_layout = QVBoxLayout(help_box)
        help_text = QLabel("A corpus is not neutral. Search terms, APIs, metadata fields, digitization, copyright, and missing records all shape what you can study.")
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        left_layout.addWidget(help_box)

        self.report_console = QTextEdit()
        self.report_console.setReadOnly(True)
        left_layout.addWidget(QLabel("Report Log:"))
        left_layout.addWidget(self.report_console, 2)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready.")
        self.fetch_btn.clicked.connect(self.start_fetching)
        self.clear_btn.clicked.connect(self.clear_all)
        self.save_all_btn.clicked.connect(self.save_all_csv)

    def update_placeholder(self):
        source = self.source_combo.currentText()
        if source == "World Bank WDI":
            self.input_label.setText("Country code and indicator, one per line:")
            self.input_console.setPlaceholderText(
                "Examples:\nUSA, SP.POP.TOTL\nSYR, SP.POP.TOTL\nSDN, SP.POP.TOTL\nUSA, NY.GDP.PCAP.CD\n\n"
                "SP.POP.TOTL = total population\nNY.GDP.PCAP.CD = GDP per capita\nSE.XPD.TOTL.GD.ZS = education expenditure % of GDP"
            )
        else:
            self.input_label.setText("Search terms, one per line:")
            examples = {
                "Library of Congress": "women suffrage\ncivil rights oral history\nschool reform\nflood photographs",
                "Gutendex / Project Gutenberg": "austen\nslavery\nreligion\neducation",
                "Internet Archive": "school yearbook\nlocal history minnesota\nadvertising age\ncivil war diary\n\nNote: Internet Archive can be slow. Narrow queries work best.",
            }
            self.input_console.setPlaceholderText("Examples:\n" + examples.get(source, ""))

    def start_fetching(self):
        lines = [ln.strip() for ln in self.input_console.toPlainText().splitlines() if ln.strip()]
        if not lines:
            QMessageBox.warning(self, "Input needed", "Please enter at least one search term or identifier.")
            return
        self.fetch_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.save_all_btn.setEnabled(False)
        self.report_console.append(f"<b>Starting fetch from {self.source_combo.currentText()}...</b>")
        worker = FetcherWorker(self.source_combo.currentText(), lines, self.max_results_spin.value())
        worker.signals.result.connect(self.add_result)
        worker.signals.error.connect(self.log_error)
        worker.signals.progress.connect(self.statusBar().showMessage)
        worker.signals.finished.connect(self.on_fetching_finished)
        self.threadpool.start(worker)

    def on_fetching_finished(self):
        self.statusBar().showMessage("Fetching complete.", 5000)
        self.fetch_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.save_all_btn.setEnabled(True)
        self.report_console.append("<b>Fetching complete.</b>")

    def log_error(self, message: str):
        self.report_console.append(f"<span style='color:red;'>Error: {message}</span>")

    def add_result(self, data: Dict[str, Any]):
        self.result_payloads.append(data)
        if data.get("type") == "wdi":
            self.create_wdi_tab(data)
        else:
            self.create_archive_tab(data)
        self.report_console.append(
            f"<span style='color:green;'>Success: {data.get('source')} | {data.get('query')} | {len(data.get('records', []))} records.</span>"
        )

    def create_archive_tab(self, data: Dict[str, Any]):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel(f"<h2>{data['source']}: {data['query']}</h2>"))
        note = QLabel("Inspect the metadata. Which fields are rich? Which are thin? Which records have dates, creators, subjects, descriptions, or text links?")
        note.setWordWrap(True)
        layout.addWidget(note)
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter, 1)

        table = QTableWidget()
        records = data.get("records", [])
        columns = ["title", "date", "creator", "subjects", "url"]
        self.populate_table(table, records, columns)
        splitter.addWidget(table)

        details = QTextEdit()
        details.setReadOnly(True)
        details.setPlainText("Select a row to inspect the raw metadata.")
        splitter.addWidget(details)

        def on_row_selected():
            row = table.currentRow()
            if 0 <= row < len(records):
                rec = records[row]
                display = {k: v for k, v in rec.items() if k != "raw"}
                display["raw"] = rec.get("raw", {})
                details.setPlainText(json.dumps(display, indent=2, ensure_ascii=False))

        table.itemSelectionChanged.connect(on_row_selected)

        button_row = QHBoxLayout()
        save_btn = QPushButton("Save This Result CSV")
        raw_btn = QPushButton("Show Full Raw JSON")
        button_row.addWidget(save_btn)
        button_row.addWidget(raw_btn)
        layout.addLayout(button_row)
        save_btn.clicked.connect(lambda: self.save_records_csv(data))
        raw_btn.clicked.connect(lambda: details.setPlainText(json.dumps(data.get("raw", {}), indent=2, ensure_ascii=False)))
        self.tabs.addTab(tab, f"{data['source'][:12]}: {truncate(data['query'], 18)}")

    def create_wdi_tab(self, data: Dict[str, Any]):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        title = f"{data.get('country_name', data.get('country'))}: {data.get('indicator_name', data.get('indicator'))}"
        layout.addWidget(QLabel(f"<h2>World Bank WDI: {title}</h2>"))
        note = QLabel("Quantitative data can provide social, historical, political, or economic context. It does not replace close reading, but it can help frame humanities questions.")
        note.setWordWrap(True)
        layout.addWidget(note)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        plot = PlotCanvas(width=6, height=4)
        plot.plot_wdi(data.get("records", []), title)
        left_layout.addWidget(plot, 3)
        meta = QTextEdit()
        meta.setReadOnly(True)
        meta.setPlainText(
            f"Country: {data.get('country_name')}\nCountry code: {data.get('country')}\n"
            f"Indicator: {data.get('indicator')}\nIndicator name: {data.get('indicator_name')}\n"
            f"Records with values: {len(data.get('records', []))}\n\n"
            "Reflection:\nWhat historical or cultural question could this time series help contextualize?\nWhat does this number fail to capture?"
        )
        left_layout.addWidget(meta, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        table = QTableWidget()
        records = data.get("records", [])
        self.populate_table(table, records, ["year", "value", "country", "indicator"])
        right_layout.addWidget(table, 3)
        save_btn = QPushButton("Save This WDI CSV")
        save_btn.clicked.connect(lambda: self.save_records_csv(data))
        right_layout.addWidget(save_btn)
        raw = QTextEdit()
        raw.setReadOnly(True)
        raw.setPlainText(json.dumps(data.get("meta", {}), indent=2, ensure_ascii=False))
        right_layout.addWidget(QLabel("API metadata:"))
        right_layout.addWidget(raw, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 2)
        self.tabs.addTab(tab, f"WDI: {data.get('country')} {data.get('indicator')}")

    def populate_table(self, table: QTableWidget, records: List[Dict[str, Any]], columns: List[str]):
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setRowCount(len(records))
        for r, record in enumerate(records):
            for c, col in enumerate(columns):
                table.setItem(r, c, QTableWidgetItem(truncate(str(record.get(col, "")), 250)))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setSortingEnabled(True)

    def save_records_csv(self, data: Dict[str, Any]):
        records = data.get("records", [])
        if not records:
            QMessageBox.information(self, "No records", "There are no records to save.")
            return
        safe_source = re.sub(r"[^A-Za-z0-9]+", "_", data.get("source", "source")).strip("_")
        safe_query = re.sub(r"[^A-Za-z0-9]+", "_", data.get("query", "query")).strip("_")[:40]
        default = f"{safe_source}_{safe_query}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", str(Path.cwd() / default), "CSV Files (*.csv)")
        if not path:
            return
        self.write_csv(Path(path), records)
        self.report_console.append(f"Saved CSV: {path}")

    def save_all_csv(self):
        if not self.result_payloads:
            QMessageBox.information(self, "No data", "There are no fetched results to save.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose folder for CSV exports", str(Path.cwd()))
        if not folder:
            return
        folder_path = Path(folder)
        for data in self.result_payloads:
            records = data.get("records", [])
            if not records:
                continue
            safe_source = re.sub(r"[^A-Za-z0-9]+", "_", data.get("source", "source")).strip("_")
            safe_query = re.sub(r"[^A-Za-z0-9]+", "_", data.get("query", "query")).strip("_")[:40]
            self.write_csv(folder_path / f"{safe_source}_{safe_query}.csv", records)
        self.report_console.append(f"Saved all CSV exports to: {folder}")

    def write_csv(self, path: Path, records: List[Dict[str, Any]]):
        keys = []
        for rec in records:
            for key in rec.keys():
                if key != "raw" and key not in keys:
                    keys.append(key)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for rec in records:
                writer.writerow({key: rec.get(key, "") for key in keys})

    def clear_all(self):
        self.tabs.clear()
        self.result_payloads.clear()
        self.report_console.clear()
        self.statusBar().showMessage("Ready.")


def main() -> int:
    app = QApplication(sys.argv)
    window = CobberHumFetcherApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
