#!/usr/bin/env python3
"""
GutenbergPackFetcher.py

Terminal-based Project Gutenberg / Gutendex fetcher for the humanities
similarity chapter.

It collects a generous raw pool of public-domain texts, aiming for 25 usable
items per pack, so that we can later hand-cull each pack to 8--16 classroom
examples.

Run:
    python GutenbergPackFetcher.py

Dependencies:
    pip install requests

Outputs:
    gutenberg_similarity_raw/
        all_metadata.csv
        all_records.jsonl
        fetch_report.txt
        <pack_key>/
            metadata.csv
            records.jsonl
            texts/
            excerpts/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


GUTENDEX_BASE = "https://gutendex.com/books"

PACKS: Dict[str, Dict[str, Any]] = {
    "short_stories": {
        "label": "Short Stories",
        "queries": ["short stories", "short story collection", "stories", "tales"],
        "topics": ["short stories"],
    },
    "ghost_supernatural": {
        "label": "Ghost / Supernatural Stories",
        "queries": ["ghost stories", "supernatural stories", "weird tales", "haunted"],
        "topics": ["ghost stories", "supernatural"],
    },
    "detective_mystery": {
        "label": "Detective / Mystery Stories",
        "queries": ["detective stories", "mystery stories", "detective fiction", "mystery"],
        "topics": ["detective", "mystery"],
    },
    "fairy_folk_tales": {
        "label": "Fairy Tales / Folk Tales",
        "queries": ["fairy tales", "folk tales", "folklore", "grimm", "andersen"],
        "topics": ["fairy tales", "folklore"],
    },
    "childrens_readers": {
        "label": "Children's Readers",
        "queries": ["children reader", "school reader", "children stories", "juvenile"],
        "topics": ["children", "juvenile"],
    },
    "essays_speeches": {
        "label": "Essays / Speeches",
        "queries": ["essays", "speeches", "addresses", "lectures"],
        "topics": ["essays", "speeches"],
    },
}


def slugify(text: str, max_len: int = 70) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return (text or "untitled")[:max_len].strip("_")


def safe_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> requests.Response:
    headers = {"User-Agent": "GutenbergPackFetcher/1.0 educational corpus-building script"}
    r = requests.get(url, params=params, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r


def gutendex_pages(params: Dict[str, Any], max_pages: int = 5, sleep: float = 0.25) -> Iterable[Dict[str, Any]]:
    url = GUTENDEX_BASE
    page_count = 0
    while url and page_count < max_pages:
        r = safe_get(url, params=params if url == GUTENDEX_BASE else None)
        payload = r.json()
        for item in payload.get("results", []):
            yield item
        url = payload.get("next")
        params = None
        page_count += 1
        if url and sleep > 0:
            time.sleep(sleep)


def choose_text_url(formats: Dict[str, str]) -> str:
    if not isinstance(formats, dict):
        return ""
    candidates: List[Tuple[int, str]] = []
    for mime, url in formats.items():
        if not isinstance(url, str):
            continue
        lower_mime = mime.lower()
        lower_url = url.lower()
        if "text/plain" not in lower_mime:
            continue
        if lower_url.endswith(".zip"):
            continue
        score = 0
        if "utf-8" in lower_mime or "utf-8" in lower_url:
            score += 10
        if "noimages" in lower_url:
            score += 2
        if ".txt" in lower_url:
            score += 1
        candidates.append((score, url))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]


def strip_gutenberg_boilerplate(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    start_patterns = [
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]
    end_patterns = [
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]

    for pat in start_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = text[m.end():]
            break

    for pat in end_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = text[:m.start()]
            break

    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))


def make_excerpt(text: str, max_words: int = 2000, skip_words: int = 200) -> str:
    words = re.findall(r"\S+", text)
    if len(words) <= max_words:
        return " ".join(words)
    start = min(skip_words, max(0, len(words) - max_words))
    return " ".join(words[start:start + max_words]).strip()


def authors_string(item: Dict[str, Any]) -> str:
    return "; ".join(a.get("name", "") for a in item.get("authors", []) if a.get("name"))


def subjects_string(item: Dict[str, Any]) -> str:
    return "; ".join(str(s) for s in item.get("subjects", [])[:12])


def collect_candidate_items(pack_key: str, pack_cfg: Dict[str, Any], target: int, sleep: float) -> List[Dict[str, Any]]:
    seen_ids = set()
    items: List[Dict[str, Any]] = []

    def add_item(item: Dict[str, Any], source_query: str):
        gid = item.get("id")
        if gid in seen_ids:
            return
        if "en" not in item.get("languages", []):
            return
        text_url = choose_text_url(item.get("formats", {}))
        if not text_url:
            return
        item["_source_query"] = source_query
        item["_text_url"] = text_url
        seen_ids.add(gid)
        items.append(item)

    for topic in pack_cfg.get("topics", []):
        if len(items) >= target * 3:
            break
        try:
            for item in gutendex_pages({"topic": topic}, max_pages=4, sleep=sleep):
                add_item(item, f"topic:{topic}")
                if len(items) >= target * 3:
                    break
        except Exception as exc:
            print(f"  [warn] topic search failed for {pack_key} / {topic}: {exc}")

    for query in pack_cfg.get("queries", []):
        if len(items) >= target * 3:
            break
        try:
            for item in gutendex_pages({"search": query}, max_pages=4, sleep=sleep):
                add_item(item, f"search:{query}")
                if len(items) >= target * 3:
                    break
        except Exception as exc:
            print(f"  [warn] search failed for {pack_key} / {query}: {exc}")

    return items


def download_text(url: str, timeout: int = 45) -> str:
    r = safe_get(url, timeout=timeout)
    if not r.encoding:
        r.encoding = "utf-8"
    return r.text


def write_pack_outputs(pack_dir: Path, records: List[Dict[str, Any]]) -> None:
    jsonl_path = pack_dir / "records.jsonl"
    csv_path = pack_dir / "metadata.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fields = list(records[0].keys()) if records else [
        "pack_key", "pack_label", "gutenberg_id", "title", "authors",
        "languages", "subjects", "bookshelves", "download_count",
        "source_query", "text_url", "gutenberg_url", "word_count_clean",
        "word_count_excerpt", "text_file", "excerpt_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def fetch_pack(pack_key: str, pack_cfg: Dict[str, Any], outdir: Path, target: int, max_words: int, min_words: int, sleep: float) -> List[Dict[str, Any]]:
    print(f"\n=== Pack: {pack_cfg['label']} ({pack_key}) ===")
    pack_dir = outdir / pack_key
    text_dir = pack_dir / "texts"
    excerpt_dir = pack_dir / "excerpts"
    text_dir.mkdir(parents=True, exist_ok=True)
    excerpt_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidate_items(pack_key, pack_cfg, target=target, sleep=sleep)
    print(f"  Metadata candidates with text URLs: {len(candidates)}")

    records: List[Dict[str, Any]] = []
    used_ids = set()

    for item in candidates:
        if len(records) >= target:
            break

        gid = item.get("id")
        if gid in used_ids:
            continue

        title = item.get("title", f"gutenberg_{gid}")
        text_url = item.get("_text_url", "")
        print(f"  [{len(records)+1:02d}/{target}] downloading #{gid}: {title[:70]}")

        try:
            raw_text = download_text(text_url)
            clean_text = strip_gutenberg_boilerplate(raw_text)
            wc = word_count(clean_text)
            if wc < min_words:
                print(f"      skipped: only {wc} words after cleaning")
                continue

            excerpt = make_excerpt(clean_text, max_words=max_words)
            excerpt_wc = word_count(excerpt)
            slug = slugify(f"{gid}_{title}")
            text_path = text_dir / f"{slug}.txt"
            excerpt_path = excerpt_dir / f"{slug}_excerpt.txt"

            text_path.write_text(clean_text, encoding="utf-8")
            excerpt_path.write_text(excerpt, encoding="utf-8")

            rec = {
                "pack_key": pack_key,
                "pack_label": pack_cfg["label"],
                "gutenberg_id": gid,
                "title": title,
                "authors": authors_string(item),
                "languages": ";".join(item.get("languages", [])),
                "subjects": subjects_string(item),
                "bookshelves": "; ".join(item.get("bookshelves", [])[:12]),
                "download_count": item.get("download_count", ""),
                "source_query": item.get("_source_query", ""),
                "text_url": text_url,
                "gutenberg_url": f"https://www.gutenberg.org/ebooks/{gid}",
                "word_count_clean": wc,
                "word_count_excerpt": excerpt_wc,
                "text_file": str(text_path.relative_to(outdir)),
                "excerpt_file": str(excerpt_path.relative_to(outdir)),
            }
            records.append(rec)
            used_ids.add(gid)

            if sleep > 0:
                time.sleep(sleep)

        except Exception as exc:
            print(f"      skipped: download/clean failed: {exc}")

    write_pack_outputs(pack_dir, records)
    print(f"  Kept usable texts: {len(records)}")
    return records


def write_all_outputs(outdir: Path, all_records: List[Dict[str, Any]], report_lines: List[str]) -> None:
    jsonl_path = outdir / "all_records.jsonl"
    csv_path = outdir / "all_metadata.csv"
    report_path = outdir / "fetch_report.txt"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if all_records:
        fields = list(all_records[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in all_records:
                writer.writerow(rec)

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Project Gutenberg text packs through Gutendex.")
    parser.add_argument("--outdir", default="gutenberg_similarity_raw", help="Output folder.")
    parser.add_argument("--target-per-pack", type=int, default=25, help="Usable texts to keep per pack.")
    parser.add_argument("--max-words", type=int, default=2000, help="Maximum words in classroom excerpt.")
    parser.add_argument("--min-words", type=int, default=800, help="Minimum cleaned words required to keep a text.")
    parser.add_argument("--sleep", type=float, default=0.35, help="Pause between API/download requests.")
    parser.add_argument("--packs", nargs="*", default=list(PACKS.keys()), help=f"Pack keys to fetch. Available: {', '.join(PACKS.keys())}")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    unknown = [p for p in args.packs if p not in PACKS]
    if unknown:
        raise SystemExit(f"Unknown pack(s): {unknown}\nAvailable: {', '.join(PACKS.keys())}")

    all_records: List[Dict[str, Any]] = []
    report_lines: List[str] = [
        "Gutenberg Pack Fetch Report",
        "===========================",
        f"Output folder: {outdir.resolve()}",
        f"Target per pack: {args.target_per_pack}",
        f"Excerpt max words: {args.max_words}",
        f"Minimum clean words: {args.min_words}",
        "",
    ]

    print("Gutenberg Pack Fetcher")
    print("======================")
    print(f"Output folder: {outdir.resolve()}")
    print(f"Target per pack: {args.target_per_pack}")

    for pack_key in args.packs:
        pack_cfg = PACKS[pack_key]
        records = fetch_pack(pack_key, pack_cfg, outdir, args.target_per_pack, args.max_words, args.min_words, args.sleep)
        all_records.extend(records)
        report_lines.append(f"{pack_key} | {pack_cfg['label']} | kept {len(records)}")
        for rec in records:
            report_lines.append(f"  - {rec['gutenberg_id']} | {rec['title']} | {rec['authors']} | words={rec['word_count_clean']}")
        report_lines.append("")

    write_all_outputs(outdir, all_records, report_lines)

    print("\nDone.")
    print(f"Total usable records: {len(all_records)}")
    print(f"Wrote: {outdir / 'all_metadata.csv'}")
    print(f"Wrote: {outdir / 'all_records.jsonl'}")
    print(f"Wrote: {outdir / 'fetch_report.txt'}")
    print("\nNext step: hand-cull each pack to 8--16 classroom examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
