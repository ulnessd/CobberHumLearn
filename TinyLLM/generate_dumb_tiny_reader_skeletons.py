#!/usr/bin/env python3
"""
generate_dumb_tiny_reader_skeletons.py

Create a deterministic "dumb" Tiny Reader corpus:
    1000 total stories
    200 stories at each level 1--5

This script does NOT call an LLM. It creates structurally valid, intentionally
simple story skeletons that can later be polished by Gemma or used directly as
a baseline corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


HUMANS = ["Dick", "Jane", "Sally", "Tim", "Mother", "Father"]
PETS = ["Spot", "Puff"]
PLACES = ["home", "school", "park", "yard", "store", "garden", "street"]
OBJECTS = ["ball", "book", "kite", "apple", "toy", "basket", "hat"]
COLORS = ["red", "blue", "yellow", "green"]

FRAMES = [
    "meeting a friend",
    "playing with a pet",
    "finding a lost object",
    "helping at home",
    "going to school",
    "sharing a toy",
    "walking in the park",
    "working in the garden",
    "going to the store",
    "being kind to a friend",
]

FRAME_PLACE_HINTS = {
    "going to the store": "store",
    "walking in the park": "park",
    "working in the garden": "garden",
    "going to school": "school",
    "helping at home": "home",
    "playing with a pet": "yard",
}

THEMES = [
    "friendship",
    "help and care",
    "sharing",
    "lost and found",
    "home and belonging",
    "kindness",
    "curiosity",
    "pet companionship",
    "small mistake and repair",
    "reassurance",
]

THEME_ARCS = {
    "friendship": [
        "A character meets a friend and they do a simple activity together.",
        "Two characters notice each other and spend time together.",
    ],
    "help and care": [
        "A character notices a small need and helps.",
        "A character helps another character with an object.",
    ],
    "sharing": [
        "A character has an object and shares it kindly.",
        "A character gives another character a turn.",
    ],
    "lost and found": [
        "A character looks for an object and finds it.",
        "A small object is missing, then found.",
    ],
    "home and belonging": [
        "A character returns to a safe familiar place.",
        "A character is with someone familiar at home.",
    ],
    "kindness": [
        "A small kind action makes the scene better.",
        "A character does something kind for another character.",
    ],
    "curiosity": [
        "A character notices an object and looks at it.",
        "A character finds something and pays attention to it.",
    ],
    "pet companionship": [
        "A human and a pet stay near each other.",
        "A pet notices an object and the human helps.",
    ],
    "small mistake and repair": [
        "A character drops or loses an object and the problem is fixed.",
        "A small mistake happens and someone helps repair it.",
    ],
    "reassurance": [
        "A character feels worried or sad and another character helps.",
        "A simple problem becomes safe again.",
    ],
}

LEVEL_INFO = {
    1: {"label": "names and simple actions"},
    2: {"label": "places and objects"},
    3: {"label": "pronouns and reference"},
    4: {"label": "because, when, then"},
    5: {"label": "small problem and resolution"},
}


@dataclass
class Request:
    story_id: str
    level: int
    frame: str
    theme: str
    narrative_arc: str
    main: str
    other: str
    place: str
    obj: str
    color: str


def words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text.lower())


def is_pet(name: str) -> bool:
    return name in PETS


def pronoun_for(name: str, role: str = "subject") -> str:
    if name in ["Dick", "Tim", "Father", "Spot"]:
        return "he" if role == "subject" else "him"
    if name in ["Jane", "Sally", "Mother", "Puff"]:
        return "she" if role == "subject" else "her"
    return "they"


def choose_theme(frame: str, other: str, rng: random.Random) -> str:
    if frame == "playing with a pet" or is_pet(other):
        return rng.choice(["pet companionship", "help and care", "curiosity", "kindness"])
    if frame == "finding a lost object":
        return rng.choice(["lost and found", "help and care", "small mistake and repair"])
    if frame == "sharing a toy":
        return rng.choice(["sharing", "friendship", "kindness"])
    if frame == "being kind to a friend":
        return rng.choice(["kindness", "help and care", "friendship"])
    if frame == "helping at home":
        return rng.choice(["help and care", "home and belonging", "small mistake and repair"])
    return rng.choice(THEMES)


def make_request(story_id: str, level: int, rng: random.Random) -> Request:
    frame = rng.choice(FRAMES)
    main = rng.choice(HUMANS)

    if frame == "playing with a pet":
        other = rng.choice(PETS)
    else:
        pool = [h for h in HUMANS if h != main]
        if rng.random() < 0.25:
            pool += PETS
        other = rng.choice(pool)

    place = FRAME_PLACE_HINTS.get(frame, rng.choice(PLACES))
    theme = choose_theme(frame, other, rng)
    arc = rng.choice(THEME_ARCS[theme])

    return Request(
        story_id=story_id,
        level=level,
        frame=frame,
        theme=theme,
        narrative_arc=arc,
        main=main,
        other=other,
        place=place,
        obj=rng.choice(OBJECTS),
        color=rng.choice(COLORS),
    )


def pet_action_sentence(pet: str) -> str:
    return "Spot barks." if pet == "Spot" else "Puff sits."


def level1(req: Request) -> List[str]:
    A, B, P, O, C = req.main, req.other, req.place, req.obj, req.color
    if is_pet(B):
        return [
            f"{A} goes to the {P}.",
            f"{B} is at the {P}.",
            f"{A} has a {C} {O}.",
            f"{A} sees {B}.",
            pet_action_sentence(B),
            f"{A} and {B} play.",
        ]
    return [
        f"{A} goes to the {P}.",
        f"{B} is at the {P}.",
        f"{A} has a {C} {O}.",
        f"{A} sees {B}.",
        f"{B} waves to {A}.",
        f"{A} and {B} play.",
    ]


def level2(req: Request) -> List[str]:
    A, B, P, O, C = req.main, req.other, req.place, req.obj, req.color
    if is_pet(B):
        if B == "Spot":
            return [
                f"{A} goes to the {P}.",
                "Spot is near the tree.",
                f"{A} has a {C} {O}.",
                f"Spot sees the {O}.",
                f"Spot finds the {O}.",
                "Spot barks.",
                f"{A} helps Spot.",
            ]
        return [
            f"{A} goes to the {P}.",
            "Puff is near the tree.",
            f"{A} has a {C} {O}.",
            f"Puff sees the {O}.",
            f"Puff looks at the {O}.",
            "Puff sits.",
            f"{A} helps Puff.",
        ]

    return [
        f"{A} goes to the {P}.",
        f"{B} is near the tree.",
        f"{A} has a {C} {O}.",
        f"{B} sees the {O}.",
        f"{A} gives the {O} to {B}.",
        f"{A} and {B} play.",
        f"{A} and {B} smile.",
    ]


def level3(req: Request) -> List[str]:
    A, B, P, O, C = req.main, req.other, req.place, req.obj, req.color
    subj = pronoun_for(B, "subject").capitalize()
    objp = pronoun_for(B, "object")

    if is_pet(B):
        return [
            f"{A} goes to the {P}.",
            f"{B} is near the tree.",
            f"{A} has a {C} {O}.",
            f"{A} sees {B}.",
            f"{subj} looks at the {O}.",
            f"{A} helps {B}.",
            f"They play in the {P}.",
            f"They walk home.",
        ]

    return [
        f"{A} goes to the {P}.",
        f"{B} is near the tree.",
        f"{A} has a {C} {O}.",
        f"{A} sees {B}.",
        f"{subj} waves to {A}.",
        f"{A} gives {objp} the {O}.",
        f"They play in the {P}.",
        f"They smile.",
    ]


def level4(req: Request) -> List[str]:
    A, B, P, O, C = req.main, req.other, req.place, req.obj, req.color
    subj = pronoun_for(B, "subject").capitalize()

    if is_pet(B):
        return [
            f"When {A} goes to the {P}, {B} comes too.",
            f"{A} has a {C} {O}.",
            f"{B} sees the {O}.",
            f"{subj} looks at it.",
            f"{A} helps {B} because {B} waits.",
            f"Then they play in the {P}.",
            f"{A} takes the {O} home.",
            f"{B} comes too.",
        ]

    return [
        f"When {A} goes to the {P}, {B} is there.",
        f"{A} has a {C} {O}.",
        f"{B} sees the {O}.",
        f"{subj} wants to play.",
        f"{A} shares because {B} waits.",
        f"Then they play in the {P}.",
        f"{A} takes the {O} home.",
        f"{B} smiles.",
    ]


def level5(req: Request) -> List[str]:
    A, B, P, O, C = req.main, req.other, req.place, req.obj, req.color
    subj = pronoun_for(B, "subject").capitalize()

    if is_pet(B):
        return [
            f"{A} goes to the {P}.",
            f"{B} comes too.",
            f"{A} has a {C} {O}.",
            f"{A} drops the {O}.",
            f"{A} is sad because it is lost.",
            f"{B} finds the {O}.",
            f"{subj} looks at {A}.",
            f"{A} helps {B}.",
            f"Then they play in the {P}.",
        ]

    objp = pronoun_for(A, "object")
    return [
        f"{A} goes to the {P}.",
        f"{B} is near the tree.",
        f"{A} has a {C} {O}.",
        f"{A} drops the {O}.",
        f"{A} is sad because it is lost.",
        f"{B} helps {objp} look.",
        f"{B} finds the {O}.",
        f"{A} thanks {B}.",
        f"Then they play in the {P}.",
    ]


LEVEL_GENERATORS = {
    1: level1,
    2: level2,
    3: level3,
    4: level4,
    5: level5,
}


def build_attention_targets(sentences: List[str], req: Request) -> List[Dict[str, str]]:
    text = " ".join(sentences)
    targets = []

    if re.search(r"\bit\b", text, flags=re.IGNORECASE):
        targets.append({"token": "it", "refers_to": req.obj})
    if re.search(r"\bthey\b|\bthem\b", text, flags=re.IGNORECASE):
        targets.append({"token": "they/them", "refers_to": f"{req.main} and {req.other}"})

    subj = pronoun_for(req.other, "subject")
    objp = pronoun_for(req.other, "object")
    if re.search(rf"\b{subj}\b|\b{objp}\b", text, flags=re.IGNORECASE):
        targets.append({"token": f"{subj}/{objp}", "refers_to": req.other})

    return targets


def illustration_prompt(req: Request) -> str:
    animal = ""
    if req.other == "Spot":
        animal = " Spot is a dog."
    elif req.other == "Puff":
        animal = " Puff is a cat."
    return (
        f"A simple early-reader illustration. {req.main} and {req.other} are in the "
        f"{req.place}. They have a {req.color} {req.obj}.{animal}"
    )


def validate_sentences(sentences: List[str], level: int) -> List[str]:
    errors = []
    for s in sentences:
        if not s.endswith("."):
            errors.append(f"sentence lacks period: {s}")
        if level in [1, 2]:
            low = set(words(s))
            if low.intersection({"he", "she", "it", "they", "him", "her", "them"}):
                errors.append(f"pronoun used at level {level}: {s}")
            if low.intersection({"because", "when", "then", "after", "before"}):
                errors.append(f"connector used at level {level}: {s}")
        low_text = s.lower()
        for bad in ["spot smiles", "spot waves", "puff smiles", "puff waves"]:
            if bad in low_text:
                errors.append(f"pet behavior problem: {s}")
    return errors


def make_record(req: Request) -> Dict:
    sentences = LEVEL_GENERATORS[req.level](req)
    errors = validate_sentences(sentences, req.level)
    if errors:
        raise ValueError(f"Internal skeleton validation failed for {req.story_id}: {errors}")

    plain = " ".join(sentences)
    return {
        "story_id": req.story_id,
        "level": req.level,
        "level_label": LEVEL_INFO[req.level]["label"],
        "frame": req.frame,
        "theme_family": req.theme,
        "narrative_arc": req.narrative_arc,
        "characters": [req.main, req.other],
        "main_character": req.main,
        "friend_or_pet": req.other,
        "place": req.place,
        "object": req.obj,
        "object_color": req.color,
        "illustration_prompt": illustration_prompt(req),
        "sentences": sentences,
        "plain_text": plain,
        "source": "skeleton_template",
        "polish_status": "unpolished",
        "word_count": len(words(plain)),
        "sentence_count": len(sentences),
        "unique_word_count": len(set(words(plain))),
        "attention_targets": build_attention_targets(sentences, req),
    }


def write_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, records: List[Dict]) -> None:
    fields = [
        "story_id", "level", "level_label", "theme_family", "frame", "characters",
        "place", "object", "object_color", "source", "polish_status",
        "sentence_count", "word_count", "unique_word_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            row["characters"] = "|".join(r["characters"])
            w.writerow(row)


def write_metadata_csv(path: Path, records: List[Dict]) -> None:
    fields = [
        "story_id", "level", "theme_family", "narrative_arc", "frame",
        "main_character", "friend_or_pet", "place", "object", "object_color",
        "source", "polish_status", "illustration_prompt", "attention_targets",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            row["attention_targets"] = json.dumps(r.get("attention_targets", []), ensure_ascii=False)
            w.writerow(row)


def write_plain_text(path: Path, records: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(f"### {r['story_id']} | Level {r['level']} | {r['theme_family']} | {r['frame']}\n")
            for s in r["sentences"]:
                f.write(s + "\n")
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories-per-level", type=int, default=200)
    parser.add_argument("--levels", type=str, default="1,2,3,4,5")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--outdir", type=str, default="tiny_reader_skeleton_output")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    for level in levels:
        if level not in LEVEL_GENERATORS:
            raise ValueError(f"Unknown level: {level}")

    records = []
    for level in levels:
        for i in range(1, args.stories_per_level + 1):
            story_id = f"L{level}_{i:04d}"
            req = make_request(story_id, level, rng)
            records.append(make_record(req))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    write_jsonl(outdir / "tiny_reader_skeleton_clean.jsonl", records)
    write_summary_csv(outdir / "tiny_reader_skeleton_summary.csv", records)
    write_metadata_csv(outdir / "tiny_reader_skeleton_metadata.csv", records)
    write_plain_text(outdir / "tiny_reader_skeleton_plain_text.txt", records)

    print("Dumb Tiny Reader skeleton corpus created.")
    print(f"  levels:            {levels}")
    print(f"  stories per level: {args.stories_per_level}")
    print(f"  total stories:     {len(records)}")
    print(f"  output directory:  {outdir.resolve()}")
    print()
    print("Output files:")
    print(f"  {outdir / 'tiny_reader_skeleton_clean.jsonl'}")
    print(f"  {outdir / 'tiny_reader_skeleton_summary.csv'}")
    print(f"  {outdir / 'tiny_reader_skeleton_metadata.csv'}")
    print(f"  {outdir / 'tiny_reader_skeleton_plain_text.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
