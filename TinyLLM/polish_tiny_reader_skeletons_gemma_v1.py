#!/usr/bin/env python3
"""
polish_tiny_reader_skeletons_gemma_v1.py

Read dumb Tiny Reader skeleton stories and ask a local Gemma model, through an
OpenAI-compatible LM Studio endpoint, to lightly polish the stories.

This is intentionally NOT a free rewrite. The model is asked to preserve:
- story_id
- level
- characters
- place
- object
- object_color
- sentence count range
- level rules
- animal ontology: Spot is a dog, Puff is a cat

It writes both polished and rejected records, and it keeps the original skeleton.

Default input:
    tiny_reader_skeleton_output/tiny_reader_skeleton_clean.jsonl

Default output directory:
    tiny_reader_polished_output

Recommended first test with official Gemma 26B in LM Studio:
    python polish_tiny_reader_skeletons_gemma_v1.py \\
      --backend openai \\
      --url http://localhost:1234/v1/chat/completions \\
      --model google/gemma-4-26b-a4b \\
      --input tiny_reader_skeleton_output/tiny_reader_skeleton_clean.jsonl \\
      --max-stories 20 \\
      --workers 1

If the official model struggles, try:
    --model gemma4-26b-obliterus-custom
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


DEFAULT_OPENAI_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "google/gemma-4-26b-a4b"

HUMANS = {"Dick", "Jane", "Sally", "Tim", "Mother", "Father"}
PETS = {"Spot", "Puff"}
ALL_CHARACTERS = HUMANS | PETS
PLACES = {"home", "school", "park", "yard", "store", "garden", "street"}
OBJECTS = {"ball", "book", "kite", "apple", "toy", "basket", "hat"}
COLORS = {"red", "blue", "yellow", "green"}

PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his"}
CAUSE_WORDS = {"because", "when", "then", "after", "before", "so"}

FORBIDDEN_WORDS = {
    "coffee", "espresso", "poetry", "postmodern", "nihilistic", "bourbon",
    "whiskey", "romantic", "sexy", "violence", "blood", "death", "dead",
    "weapon", "gun", "knife", "suicide", "racism", "hate", "adult",
    "existential", "intellectual", "metaphor", "unreliable", "narrator",
    "dramatic", "mysterious", "velvet", "charming", "anxiety", "dread",
}

# This is deliberately broader than the generator's old strict list. The polisher
# should be judged mainly on level behavior and ontology, not tiny vocabulary trivia.
APPROVED_WORDS = set(w.lower() for w in [
    "Dick", "Jane", "Sally", "Tim", "Mother", "Father", "Spot", "Puff",
    "home", "school", "park", "yard", "store", "garden", "street", "tree",
    "ball", "book", "kite", "apple", "toy", "basket", "hat",
    "red", "blue", "yellow", "green",
    "a", "an", "the", "to", "and", "in", "on", "at", "with", "by", "near",
    "for", "from", "of", "is", "are", "am", "has", "have", "had",
    "go", "goes", "walk", "walks", "run", "runs", "come", "comes",
    "play", "plays", "give", "gives", "take", "takes", "find", "finds",
    "help", "helps", "look", "looks", "see", "sees", "sit", "sits",
    "stand", "stands", "jump", "jumps", "drop", "drops", "bring", "brings",
    "bark", "barks", "purr", "purrs", "wave", "waves", "smile", "smiles",
    "thank", "thanks", "share", "shares", "want", "wants", "wait", "waits",
    "sad", "happy", "kind", "good", "little", "big", "safe", "lost", "fun",
    "there", "here", "too", "again", "back", "home", "room", "door",
    "he", "she", "it", "they", "him", "her", "them", "his",
    "because", "when", "then", "after", "before", "so",
    "dog", "cat", "tail",
])

LEVEL_RULES = {
    1: {"sentence_min": 5, "sentence_max": 7, "word_max": 7, "allow_pronouns": False, "allow_cause": False, "require_problem": False},
    2: {"sentence_min": 6, "sentence_max": 8, "word_max": 8, "allow_pronouns": False, "allow_cause": False, "require_problem": False},
    3: {"sentence_min": 7, "sentence_max": 9, "word_max": 9, "allow_pronouns": True, "allow_cause": False, "require_problem": False},
    4: {"sentence_min": 7, "sentence_max": 9, "word_max": 11, "allow_pronouns": True, "allow_cause": True, "require_problem": False},
    5: {"sentence_min": 8, "sentence_max": 10, "word_max": 11, "allow_pronouns": True, "allow_cause": True, "require_problem": True},
}


def query_nvidia_smi() -> Dict[str, Any]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return {"available": False, "error": result.stderr.strip()}
        gpus = []
        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "name": parts[0],
                    "memory_used_mb": float(parts[1]),
                    "memory_total_mb": float(parts[2]),
                    "gpu_util_percent": float(parts[3]),
                    "mem_util_percent": float(parts[4]),
                })
        return {"available": True, "gpus": gpus}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def print_gpu_status(label: str) -> None:
    status = query_nvidia_smi()
    print(f"\nGPU status: {label}")
    if not status.get("available"):
        print(f"  unavailable: {status.get('error')}")
        return
    for i, gpu in enumerate(status["gpus"]):
        print(
            f"  GPU {i}: {gpu['name']} | "
            f"mem {gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f} MB | "
            f"gpu util {gpu['gpu_util_percent']:.0f}% | mem util {gpu['mem_util_percent']:.0f}%"
        )


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text.lower())


def sentence_count_words(sentence: str) -> int:
    return len(word_tokens(sentence))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def story_text(sentences: List[str]) -> str:
    return " ".join(sentences)


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def cleanup_jsonish(text: str) -> str:
    text = strip_fences(text)
    text = re.sub(r'(?m)^(\s*)[^\s\[\]\{\},"]+(?=")', r'\1', text)
    text = re.sub(r',(\s*[\]}])', r'\1', text)
    return text


def extract_json_object(text: str) -> Dict[str, Any]:
    text = cleanup_jsonish(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(cleanup_jsonish(text[start:end+1]))
        if isinstance(obj, dict):
            return obj
    raise ValueError("No JSON object found")


def prompt_for_record(record: Dict[str, Any]) -> str:
    level = int(record["level"])
    rules = LEVEL_RULES[level]
    chars = record["characters"]
    main = record["main_character"]
    other = record["friend_or_pet"]
    pet_note = ""
    if other == "Spot":
        pet_note = "Spot is a dog. Spot may bark, run, sit, look, find, bring, and play. Spot must not smile, wave, speak, ask, thank, or give objects."
    elif other == "Puff":
        pet_note = "Puff is a cat. Puff may sit, jump, look, find, come, and play. Puff must not smile, wave, speak, ask, thank, or give objects."

    pronoun_rule = (
        "Do not use pronouns at this level. Repeat names."
        if not rules["allow_pronouns"]
        else "You may use simple pronouns, but the references must be clear."
    )
    cause_rule = (
        "Do not use because, when, then, after, or before at this level."
        if not rules["allow_cause"]
        else "Use a simple time or cause connector such as because, when, or then."
    )
    problem_rule = (
        "Include a small problem and a simple resolution."
        if rules["require_problem"]
        else "Do not add danger or serious conflict."
    )

    return f"""
You are polishing a simple early-reader story for a controlled student dataset.

Return VALID JSON ONLY. No markdown. No explanation.

Task:
- Lightly polish the sentences only.
- Keep the story simple and concrete.
- Keep the same story_id, level, frame, characters, place, object, object_color.
- Do not add new characters, places, objects, or colors.
- Preserve the basic meaning of the skeleton.
- Do not make it literary, clever, adult, ironic, dramatic, or poetic.

Level rules:
- Level: {level}
- Sentence count: {rules['sentence_min']} to {rules['sentence_max']} sentences.
- Each sentence: at most {rules['word_max']} words.
- {pronoun_rule}
- {cause_rule}
- {problem_rule}

Animal rules:
- Human characters: Dick, Jane, Sally, Tim, Mother, Father.
- Spot is a dog.
- Puff is a cat.
- {pet_note}
- Pets should act like animals, not people.

Phrase-quality rules:
- Avoid "The book is here" style filler.
- Avoid repeated sentences.
- Avoid "happy faces" and "play happy".
- Use "gives the book to Tim", not "gives book to Tim".

Required metadata:
story_id: {record['story_id']}
level: {level}
frame: {record['frame']}
theme_family: {record.get('theme_family', '')}
characters: {chars}
place: {record['place']}
object: {record['object']}
object_color: {record['object_color']}

Skeleton sentences:
{json.dumps(record['sentences'], ensure_ascii=False, indent=2)}

Return exactly this JSON shape:
{{
  "story_id": "{record['story_id']}",
  "sentences": ["sentence one.", "sentence two."]
}}
""".strip()


def call_model(args, prompt: str) -> Tuple[str, Dict[str, Any]]:
    t0 = time.time()
    if args.backend == "openai":
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        resp = requests.post(args.url, json=payload, timeout=args.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
    elif args.backend == "ollama":
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_ctx": args.num_ctx,
                "num_predict": args.max_tokens,
                "repeat_penalty": 1.08,
            },
        }
        resp = requests.post(args.url, json=payload, timeout=args.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message") or {}).get("content", "")
        completion_tokens = int(data.get("eval_count") or 0)
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    elapsed = time.time() - t0
    return text.strip(), {
        "elapsed_seconds": elapsed,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "tokens_per_second": completion_tokens / elapsed if completion_tokens and elapsed else 0.0,
    }


def repair_sentences(sentences: List[str], record: Dict[str, Any]) -> List[str]:
    repaired = []
    other = record["friend_or_pet"]

    for s in sentences:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if not s:
            continue
        if not s.endswith("."):
            s += "."

        # Common repairs.
        s = s.replace("happy faces", "smiles")
        s = s.replace("play happy", "play")
        s = s.replace("gives book to", "gives the book to")
        s = s.replace("gives ball to", "gives the ball to")
        s = s.replace("gives kite to", "gives the kite to")
        s = s.replace("gives apple to", "gives the apple to")
        s = s.replace("gives toy to", "gives the toy to")
        s = s.replace("gives basket to", "gives the basket to")
        s = s.replace("gives hat to", "gives the hat to")

        if other == "Spot":
            s = s.replace("Spot smiles.", "Spot barks.")
            s = s.replace("Spot waves to", "Spot looks at")
            s = s.replace("Spot waves at", "Spot looks at")
            s = s.replace("Spot gives", "Spot finds")
        elif other == "Puff":
            s = s.replace("Puff smiles.", "Puff sits.")
            s = s.replace("Puff waves to", "Puff looks at")
            s = s.replace("Puff waves at", "Puff looks at")
            s = s.replace("Puff gives", "Puff finds")

        # Replace exact "is here" filler.
        low = s.lower()
        obj = record["object"].lower()
        color = record["object_color"].lower()
        main = record["main_character"]
        if low == f"the {obj} is here.":
            s = f"{main} sees the {obj}."
        elif low == f"the {color} {obj} is here.":
            s = f"{main} sees the {color} {obj}."

        repaired.append(s)

    # Remove consecutive duplicates.
    out = []
    prev = None
    for s in repaired:
        norm = re.sub(r"\s+", " ", s.strip().lower())
        if norm != prev:
            out.append(s)
        prev = norm
    return out


def validate_polished(sentences: List[str], record: Dict[str, Any]) -> List[str]:
    errors = []
    level = int(record["level"])
    rules = LEVEL_RULES[level]
    text = story_text(sentences)
    low_text = text.lower()
    low_words = set(word_tokens(text))

    if not (rules["sentence_min"] <= len(sentences) <= rules["sentence_max"]):
        errors.append(f"sentence count {len(sentences)} outside {rules['sentence_min']}-{rules['sentence_max']}")

    for s in sentences:
        if not isinstance(s, str):
            errors.append("non-string sentence")
            continue
        if not s.endswith("."):
            errors.append(f"sentence lacks period: {s}")
        if '"' in s or "'" in s:
            errors.append(f"dialogue/quote mark found: {s}")
        wc = sentence_count_words(s)
        if wc > rules["word_max"]:
            errors.append(f"sentence too long ({wc} words): {s}")

    # Required exact world elements.
    required_words = [
        record["main_character"].lower(),
        record["friend_or_pet"].lower(),
        record["place"].lower(),
        record["object"].lower(),
        record["object_color"].lower(),
    ]
    for w in required_words:
        if w not in low_words:
            errors.append(f"required word missing: {w}")

    if not rules["allow_pronouns"]:
        used = low_words.intersection(PRONOUNS)
        if used:
            errors.append(f"pronouns not allowed: {sorted(used)}")

    if not rules["allow_cause"]:
        used = low_words.intersection(CAUSE_WORDS)
        if used:
            errors.append(f"cause/time words not allowed: {sorted(used)}")

    if rules["allow_cause"] and level >= 4:
        if not low_words.intersection(CAUSE_WORDS):
            errors.append("level requires cause/time connector")

    if rules["require_problem"]:
        if not low_words.intersection({"lost", "sad", "drops", "drop", "helps", "finds"}):
            errors.append("level 5 needs small problem/resolution language")

    for fw in FORBIDDEN_WORDS:
        if fw in low_words:
            errors.append(f"forbidden word: {fw}")

    for pat in [
        "spot smiles", "spot waves", "spot gives", "spot thanks", "spot asks",
        "puff smiles", "puff waves", "puff gives", "puff thanks", "puff asks",
        "happy faces", "play happy",
    ]:
        if pat in low_text:
            errors.append(f"banned phrase/animal behavior: {pat}")

    # Warn-level unknown words are hard errors only if --strict-vocab is set.
    unknown = sorted(w for w in low_words if w not in APPROVED_WORDS)
    if unknown and record.get("_strict_vocab"):
        errors.append(f"unknown words: {unknown[:20]}")

    return errors


def build_attention_targets(sentences: List[str], record: Dict[str, Any]) -> List[Dict[str, str]]:
    text = story_text(sentences)
    targets = []
    if re.search(r"\bit\b", text, flags=re.IGNORECASE):
        targets.append({"token": "it", "refers_to": record["object"]})
    if re.search(r"\bthey\b|\bthem\b", text, flags=re.IGNORECASE):
        targets.append({"token": "they/them", "refers_to": "the two main characters"})
    return targets


def polish_one(args, record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    rejected_attempts = []
    prompt = prompt_for_record(record)

    for attempt in range(1, args.max_attempts + 1):
        try:
            raw, metrics = call_model(args, prompt)
            obj = extract_json_object(raw)
            if obj.get("story_id") != record["story_id"]:
                raise ValueError(f"story_id mismatch: {obj.get('story_id')} != {record['story_id']}")

            sentences = repair_sentences(obj.get("sentences", []), record)
            trial = dict(record)
            trial["_strict_vocab"] = args.strict_vocab
            errors = validate_polished(sentences, trial)

            if not errors:
                polished = dict(record)
                polished["sentences"] = sentences
                polished["plain_text"] = story_text(sentences)
                polished["source"] = "skeleton_plus_gemma_polish"
                polished["polish_status"] = "polished"
                polished["polish_model"] = args.model
                polished["polish_attempts"] = attempt
                polished["original_sentences"] = record.get("sentences", [])
                polished["word_count"] = len(word_tokens(polished["plain_text"]))
                polished["sentence_count"] = len(sentences)
                polished["unique_word_count"] = len(set(word_tokens(polished["plain_text"])))
                polished["attention_targets"] = build_attention_targets(sentences, polished)
                return polished, metrics, rejected_attempts

            rejected_attempts.append({
                "story_id": record["story_id"],
                "attempt": attempt,
                "reason": "validation_error",
                "errors": errors,
                "raw": raw,
                "candidate_sentences": sentences,
            })

        except Exception as exc:
            rejected_attempts.append({
                "story_id": record["story_id"],
                "attempt": attempt,
                "reason": "exception",
                "errors": [str(exc)],
            })

    fallback = dict(record)
    fallback["source"] = "skeleton_template"
    fallback["polish_status"] = "polish_failed_kept_skeleton"
    fallback["polish_model"] = args.model
    fallback["polish_attempts"] = args.max_attempts
    fallback["polish_errors"] = rejected_attempts[-1]["errors"] if rejected_attempts else []
    return fallback, {"elapsed_seconds": 0.0, "completion_tokens": 0, "prompt_tokens": 0, "tokens_per_second": 0.0}, rejected_attempts


def write_summary_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    fields = [
        "story_id", "level", "theme_family", "frame", "characters",
        "place", "object", "object_color", "source", "polish_status",
        "polish_model", "polish_attempts", "sentence_count",
        "word_count", "unique_word_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            row["characters"] = "|".join(r.get("characters", []))
            w.writerow(row)


def write_plain_text(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(f"### {r['story_id']} | Level {r['level']} | {r.get('theme_family','')} | {r.get('polish_status','')}\n")
            for s in r.get("sentences", []):
                f.write(s + "\n")
            f.write("\n")


def write_performance_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = ["story_id", "level", "elapsed_seconds", "completion_tokens", "prompt_tokens", "tokens_per_second", "status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def save_outputs(outdir: Path, polished: List[Dict[str, Any]], rejected: List[Dict[str, Any]], perf: List[Dict[str, Any]]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    polished_sorted = sorted(polished, key=lambda r: (int(r["level"]), r["story_id"]))
    write_jsonl(outdir / "tiny_reader_polished_clean.jsonl", polished_sorted)
    write_jsonl(outdir / "tiny_reader_polished_rejected.jsonl", rejected)
    write_summary_csv(outdir / "tiny_reader_polished_summary.csv", polished_sorted)
    write_plain_text(outdir / "tiny_reader_polished_plain_text.txt", polished_sorted)
    write_performance_csv(outdir / "tiny_reader_polished_performance.csv", perf)


def check_server(args) -> None:
    raw, metrics = call_model(args, 'Return valid JSON only: {"ok": true}')
    print("Server response:")
    print(raw[:500])
    print(f"Elapsed: {metrics['elapsed_seconds']:.2f} s")
    print(f"Completion tok/s: {metrics['tokens_per_second']:.1f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--url", default=DEFAULT_OPENAI_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", default="tiny_reader_skeleton_output/tiny_reader_skeleton_clean.jsonl")
    parser.add_argument("--outdir", default="tiny_reader_polished_output")
    parser.add_argument("--max-stories", type=int, default=0, help="0 means all stories")
    parser.add_argument("--levels", default="", help="optional comma list, e.g. 1,2,3")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--strict-vocab", action="store_true")
    parser.add_argument("--check-server", action="store_true")
    args = parser.parse_args()

    if args.backend == "ollama" and args.url == DEFAULT_OPENAI_URL:
        args.url = DEFAULT_OLLAMA_URL

    print_gpu_status("before polishing")

    if args.check_server:
        check_server(args)
        print_gpu_status("after check-server")
        return 0

    rows = load_jsonl(Path(args.input))

    if args.levels.strip():
        keep = {int(x.strip()) for x in args.levels.split(",") if x.strip()}
        rows = [r for r in rows if int(r["level"]) in keep]

    if args.max_stories and args.max_stories > 0:
        rows = rows[:args.max_stories]

    outdir = Path(args.outdir)
    polished: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    perf: List[Dict[str, Any]] = []

    print("Tiny Reader Gemma Polisher v1")
    print(f"  backend:     {args.backend}")
    print(f"  url:         {args.url}")
    print(f"  model:       {args.model}")
    print(f"  input:       {Path(args.input).resolve()}")
    print(f"  stories:     {len(rows)}")
    print(f"  workers:     {args.workers}")
    print(f"  output dir:  {outdir.resolve()}")
    print()

    start = time.time()

    if args.workers <= 1:
        for i, row in enumerate(rows, start=1):
            rec, metrics, rej = polish_one(args, row)
            polished.append(rec)
            rejected.extend(rej)
            perf.append({
                "story_id": row["story_id"],
                "level": row["level"],
                "status": rec.get("polish_status", ""),
                **metrics,
            })
            elapsed = time.time() - start
            rate = len(polished) / elapsed * 3600 if elapsed > 0 else 0.0
            if i % 5 == 0 or i == len(rows):
                ok = sum(1 for r in polished if r.get("polish_status") == "polished")
                kept = len(polished) - ok
                print(f"{i:5d}/{len(rows)} polished={ok} kept_skeleton={kept} rate={rate:.1f}/hour")
                save_outputs(outdir, polished, rejected, perf)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(polish_one, args, row): row for row in rows}
            for i, fut in enumerate(as_completed(futures), start=1):
                row = futures[fut]
                rec, metrics, rej = fut.result()
                polished.append(rec)
                rejected.extend(rej)
                perf.append({
                    "story_id": row["story_id"],
                    "level": row["level"],
                    "status": rec.get("polish_status", ""),
                    **metrics,
                })
                elapsed = time.time() - start
                rate = len(polished) / elapsed * 3600 if elapsed > 0 else 0.0
                if i % 5 == 0 or i == len(rows):
                    ok = sum(1 for r in polished if r.get("polish_status") == "polished")
                    kept = len(polished) - ok
                    print(f"{i:5d}/{len(rows)} polished={ok} kept_skeleton={kept} rate={rate:.1f}/hour")
                    save_outputs(outdir, polished, rejected, perf)

    save_outputs(outdir, polished, rejected, perf)

    elapsed = time.time() - start
    ok = sum(1 for r in polished if r.get("polish_status") == "polished")
    kept = len(polished) - ok
    total_tokens = sum(int(r.get("completion_tokens") or 0) for r in perf)
    total_time = sum(float(r.get("elapsed_seconds") or 0.0) for r in perf)
    tps = total_tokens / total_time if total_time > 0 else 0.0

    print_gpu_status("end of polishing")
    print()
    print("Polishing summary")
    print(f"  Total records:       {len(polished)}")
    print(f"  Polished:            {ok}")
    print(f"  Kept skeleton:       {kept}")
    if polished:
        print(f"  Polish success rate: {100*ok/len(polished):.1f}%")
    print(f"  Rejected attempts:   {len(rejected)}")
    print(f"  Elapsed:             {elapsed/60:.1f} min")
    if elapsed > 0:
        print(f"  Records/hour:        {len(polished)/elapsed*3600:.1f}")
    print(f"  Completion tok/s:    {tps:.1f}")
    print()
    print("Output files:")
    print(f"  {outdir / 'tiny_reader_polished_clean.jsonl'}")
    print(f"  {outdir / 'tiny_reader_polished_rejected.jsonl'}")
    print(f"  {outdir / 'tiny_reader_polished_summary.csv'}")
    print(f"  {outdir / 'tiny_reader_polished_plain_text.txt'}")
    print(f"  {outdir / 'tiny_reader_polished_performance.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
