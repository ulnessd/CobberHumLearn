#!/usr/bin/env python3
"""
clean_mill_creek_archive.py

Clean a Mill Creek generated archive by removing artifacts whose metrics status is FAIL.
This is intended for long production runs where the generator already saved all artifacts
and the metrics file identifies hard failures.

What it does
------------
1. Accepts either:
   - a run folder, or
   - a .zip file containing a run folder.

2. Reads:
   artifacts/all_artifacts.jsonl
   metrics/artifact_metrics.csv

3. Removes artifacts whose artifact_id has status == FAIL.

4. Writes a cleaned archive folder with:
   artifacts/all_artifacts.jsonl
   artifacts/<category>_artifacts.jsonl
   metrics/artifact_metrics.csv              original metrics with an added clean_action column
   metrics/artifact_metrics_clean.csv        non-FAIL rows only
   metrics/cleaning_summary.json
   metrics/removed_failed_artifacts.csv
   plus copied manifests, gazetteer, and run_plan.json when present.

5. Optionally creates a zip of the cleaned archive.

Example
-------
python clean_mill_creek_archive.py \
  --input v8_8000_production_candidate.zip \
  --out clean_v8_8000 \
  --zip-output

Or, if you have the run folder:

python clean_mill_creek_archive.py \
  --input generated/mill_creek_pilot_archive/v8_8000_production_candidate \
  --out clean_v8_8000 \
  --zip-output
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Could not parse JSONL line {line_no} in {path}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_run_root(path: Path) -> Path:
    """Find the folder containing artifacts/all_artifacts.jsonl and metrics/artifact_metrics.csv."""
    if (path / "artifacts" / "all_artifacts.jsonl").exists() and (path / "metrics" / "artifact_metrics.csv").exists():
        return path

    candidates = []
    for child in path.rglob("all_artifacts.jsonl"):
        candidate = child.parent.parent
        if (candidate / "metrics" / "artifact_metrics.csv").exists():
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find a Mill Creek run root under {path}. "
            "Expected artifacts/all_artifacts.jsonl and metrics/artifact_metrics.csv."
        )

    # Prefer the shallowest path.
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def prepare_input(input_path: Path, work_dir: Path) -> Path:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        extract_dir = work_dir / "extracted_input"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(input_path, "r") as z:
            z.extractall(extract_dir)
        return find_run_root(extract_dir)

    if input_path.is_dir():
        return find_run_root(input_path)

    raise FileNotFoundError(f"Input path not found or unsupported: {input_path}")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def zip_folder(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(folder.parent))


def clean_archive(input_path: Path, out_dir: Path, zip_output: bool = False) -> dict[str, Any]:
    work_dir = out_dir.parent / f".{out_dir.name}_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    run_root = prepare_input(input_path, work_dir)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    artifacts_path = run_root / "artifacts" / "all_artifacts.jsonl"
    metrics_path = run_root / "metrics" / "artifact_metrics.csv"

    artifacts = read_jsonl(artifacts_path)
    metrics = read_csv_dicts(metrics_path)

    metrics_by_id = {row.get("artifact_id", ""): row for row in metrics}
    fail_ids = {
        row.get("artifact_id", "")
        for row in metrics
        if row.get("status", "").upper() == "FAIL"
    }
    fail_ids.discard("")

    artifact_ids = {a.get("artifact_id", "") for a in artifacts}
    fail_ids_present = fail_ids & artifact_ids

    clean_artifacts = [a for a in artifacts if a.get("artifact_id", "") not in fail_ids]
    removed_artifacts = [a for a in artifacts if a.get("artifact_id", "") in fail_ids]

    # Copy metadata/support folders first.
    copy_if_exists(run_root / "manifests", out_dir / "manifests")
    copy_if_exists(run_root / "gazetteer", out_dir / "gazetteer")
    copy_if_exists(run_root / "run_plan.json", out_dir / "run_plan.json")

    # Write cleaned artifacts.
    write_jsonl(out_dir / "artifacts" / "all_artifacts.jsonl", clean_artifacts)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for art in clean_artifacts:
        by_category[str(art.get("category", "unknown"))].append(art)

    for category, rows in sorted(by_category.items()):
        write_jsonl(out_dir / "artifacts" / f"{category}_artifacts.jsonl", rows)

    # Mark original metrics and write clean metrics.
    metrics_with_action = []
    clean_metrics = []
    for row in metrics:
        rid = row.get("artifact_id", "")
        new_row = dict(row)
        if rid in fail_ids:
            new_row["clean_action"] = "removed_FAIL"
        else:
            new_row["clean_action"] = "kept"
            clean_metrics.append(new_row)
        metrics_with_action.append(new_row)

    metric_fields = list(metrics_with_action[0].keys()) if metrics_with_action else []
    if "clean_action" not in metric_fields:
        metric_fields.append("clean_action")

    write_csv_dicts(out_dir / "metrics" / "artifact_metrics.csv", metrics_with_action, metric_fields)
    write_csv_dicts(out_dir / "metrics" / "artifact_metrics_clean.csv", clean_metrics, metric_fields)

    # Removed artifact report.
    removed_rows = []
    for art in removed_artifacts:
        aid = art.get("artifact_id", "")
        m = metrics_by_id.get(aid, {})
        removed_rows.append({
            "artifact_id": aid,
            "category": art.get("category", ""),
            "year": art.get("year", ""),
            "source_type": art.get("source_type", ""),
            "title": art.get("title", art.get("title_seed", "")),
            "status": m.get("status", ""),
            "warning_count": m.get("warning_count", ""),
            "warnings_json": m.get("warnings_json", ""),
            "body_word_count": m.get("body_word_count", ""),
        })

    write_csv_dicts(out_dir / "metrics" / "removed_failed_artifacts.csv", removed_rows)

    # Summary.
    category_before = Counter(str(a.get("category", "unknown")) for a in artifacts)
    category_after = Counter(str(a.get("category", "unknown")) for a in clean_artifacts)
    status_counts = Counter(row.get("status", "") for row in metrics)

    summary = {
        "input": str(input_path),
        "detected_run_root": str(run_root),
        "output_folder": str(out_dir),
        "total_artifacts_before": len(artifacts),
        "total_artifacts_after": len(clean_artifacts),
        "removed_fail_artifacts": len(removed_artifacts),
        "fail_ids_in_metrics": len(fail_ids),
        "fail_ids_present_in_artifacts": len(fail_ids_present),
        "status_counts_from_metrics": dict(status_counts),
        "category_counts_before": dict(sorted(category_before.items())),
        "category_counts_after": dict(sorted(category_after.items())),
        "removed_by_category": dict(sorted(Counter(str(a.get("category", "unknown")) for a in removed_artifacts).items())),
    }

    (out_dir / "metrics" / "cleaning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Human-readable summary.
    report_lines = [
        "Mill Creek archive cleaning summary",
        "===================================",
        f"Input: {input_path}",
        f"Detected run root: {run_root}",
        f"Output: {out_dir}",
        "",
        f"Artifacts before: {len(artifacts)}",
        f"Artifacts after:  {len(clean_artifacts)}",
        f"Removed FAIL:     {len(removed_artifacts)}",
        "",
        "Status counts from metrics:",
    ]
    for key, val in sorted(status_counts.items()):
        report_lines.append(f"  {key}: {val}")

    report_lines.append("")
    report_lines.append("Removed by category:")
    for key, val in sorted(Counter(str(a.get("category", "unknown")) for a in removed_artifacts).items()):
        report_lines.append(f"  {key}: {val}")

    report_lines.append("")
    report_lines.append("Clean category counts:")
    for key, val in sorted(category_after.items()):
        report_lines.append(f"  {key}: {val}")

    (out_dir / "metrics" / "cleaning_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if zip_output:
        zip_path = out_dir.with_suffix(".zip")
        zip_folder(out_dir, zip_path)
        summary["zip_output"] = str(zip_path)

    if work_dir.exists():
        shutil.rmtree(work_dir)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove FAIL artifacts from a Mill Creek generated archive.")
    parser.add_argument("--input", required=True, type=Path, help="Input run folder or run .zip file.")
    parser.add_argument("--out", required=True, type=Path, help="Output cleaned run folder.")
    parser.add_argument("--zip-output", action="store_true", help="Also create a .zip next to the cleaned output folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = clean_archive(args.input, args.out, args.zip_output)

    print("\nMill Creek archive cleaned")
    print("--------------------------")
    print(f"Input:             {summary['input']}")
    print(f"Output folder:     {summary['output_folder']}")
    print(f"Artifacts before:  {summary['total_artifacts_before']}")
    print(f"Artifacts after:   {summary['total_artifacts_after']}")
    print(f"Removed FAIL rows: {summary['removed_fail_artifacts']}")
    print(f"Status counts:     {summary['status_counts_from_metrics']}")
    print("\nClean category counts:")
    for key, val in summary["category_counts_after"].items():
        print(f"  {key:15s} {val}")
    if "zip_output" in summary:
        print(f"\nZip output:        {summary['zip_output']}")
    print("\nReports:")
    print(f"  {Path(summary['output_folder']) / 'metrics' / 'cleaning_summary.json'}")
    print(f"  {Path(summary['output_folder']) / 'metrics' / 'cleaning_report.txt'}")
    print(f"  {Path(summary['output_folder']) / 'metrics' / 'removed_failed_artifacts.csv'}")


if __name__ == "__main__":
    main()
