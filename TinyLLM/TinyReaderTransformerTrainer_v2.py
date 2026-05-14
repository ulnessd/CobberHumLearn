#!/usr/bin/env python3
"""
TinyReaderTransformerTrainer.py

Stage 3 program for the Tiny Reader / Dick-and-Jane language-model project.

Purpose
-------
Train a very small word-level transformer language model on the polished Tiny Reader
corpus. This is the neural comparison model for the n-gram baseline.

Default inputs from Stage 1:
    tiny_reader_model_prep/corpus_train.jsonl
    tiny_reader_model_prep/corpus_val.jsonl
    tiny_reader_model_prep/corpus_test.jsonl

Default output:
    tiny_reader_transformer_output/

Main outputs:
    tiny_transformer_model.pt
    tiny_transformer_vocab.json
    tiny_transformer_config.json
    tiny_transformer_training_log.csv
    tiny_transformer_eval.txt
    tiny_transformer_samples.txt

Example quick smoke test:
    python TinyReaderTransformerTrainer.py --epochs 2 --batch-size 64 --d-model 128 --layers 2 --heads 4

More serious first run on the fast machine:
    python TinyReaderTransformerTrainer.py \
      --epochs 20 \
      --batch-size 128 \
      --d-model 192 \
      --layers 4 \
      --heads 4 \
      --block-size 96 \
      --lr 0.001

Generate only from a saved model:
    python TinyReaderTransformerTrainer.py --generate-only \
      --prompt "father has a blue apple" \
      --num-samples 10

Design notes
------------
This is intentionally word-level rather than BPE/subword. The vocabulary is tiny
(~200 tokens), which makes the model interpretable for students and appropriate
for the Tiny Reader corpus.

The model is not meant to be a general LLM. It is a tiny language model trained
inside a controlled story world so students can see next-token prediction,
generation, and the advantage of flexible context over a fixed n-gram window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:
    raise SystemExit(
        "This script requires PyTorch.\n"
        "Install it in your environment first, for example:\n"
        "  pip install torch\n\n"
        f"Original import error: {exc}"
    )


PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Word-level tokenizer that keeps simple punctuation as tokens."""
    return re.findall(r"[A-Za-z']+|[.,!?;:]", text.lower())


def detokenize(tokens: List[str]) -> str:
    """Readable detokenization for generated Tiny Reader stories."""
    filtered = [t for t in tokens if t not in {PAD, BOS, EOS}]
    s = " ".join(filtered)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    # Capitalize sentence starts.
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
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_vocab(records: List[Dict[str, Any]], min_count: int = 1) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int]]:
    counter: Dict[str, int] = {}
    for rec in records:
        for tok in tokenize(rec.get("text", "")):
            counter[tok] = counter.get(tok, 0) + 1

    special = [PAD, BOS, EOS, UNK]
    vocab_tokens = special + sorted([tok for tok, count in counter.items() if count >= min_count])
    stoi = {tok: i for i, tok in enumerate(vocab_tokens)}
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos, counter


def encode_text(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[BOS]] + [stoi.get(tok, stoi[UNK]) for tok in tokenize(text)] + [stoi[EOS]]


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class TinyReaderDataset(torch.utils.data.Dataset):
    def __init__(self, records: List[Dict[str, Any]], stoi: Dict[str, int], block_size: int):
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.block_size = block_size
        self.pad_id = stoi[PAD]

        for rec in records:
            ids = encode_text(rec.get("text", ""), stoi)
            if len(ids) < 2:
                continue

            # For these short stories, usually one block per story is enough.
            # If a story is longer than block_size + 1, create overlapping windows.
            if len(ids) <= block_size + 1:
                x = ids[:-1]
                y = ids[1:]
                self.examples.append(self._pad_pair(x, y))
            else:
                stride = max(1, block_size // 2)
                for start in range(0, len(ids) - 1, stride):
                    chunk = ids[start:start + block_size + 1]
                    if len(chunk) < 2:
                        continue
                    x = chunk[:-1]
                    y = chunk[1:]
                    self.examples.append(self._pad_pair(x, y))
                    if start + block_size + 1 >= len(ids):
                        break

    def _pad_pair(self, x: List[int], y: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x[:self.block_size]
        y = y[:self.block_size]
        pad_len = self.block_size - len(x)
        if pad_len > 0:
            x = x + [self.pad_id] * pad_len
            y = y + [self.pad_id] * pad_len
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 96
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    pad_id: int = 0


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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        if t > self.cfg.block_size:
            idx = idx[:, -self.cfg.block_size:]
            t = idx.shape[1]
            if targets is not None:
                targets = targets[:, -self.cfg.block_size:]

        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        # Causal mask: positions cannot attend to future positions.
        causal_mask = torch.triu(torch.ones(t, t, device=idx.device, dtype=torch.bool), diagonal=1)
        pad_mask = idx.eq(self.cfg.pad_id)

        x = self.blocks(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_id,
            )
        return logits, loss


# ---------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: TinyTransformerLM, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_tokens = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, loss = model(x, y)
        total_loss += float(loss.item())
        total_batches += 1

        pred = logits.argmax(dim=-1)
        mask = y.ne(model.cfg.pad_id)
        correct += int(((pred == y) & mask).sum().item())
        total_tokens += int(mask.sum().item())

    avg_loss = total_loss / max(1, total_batches)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    acc = correct / total_tokens if total_tokens else 0.0
    return {"loss": avg_loss, "perplexity": ppl, "top1_accuracy": acc, "tokens": total_tokens}


def top_predictions(model, stoi, itos, prompt: str, device: torch.device, top_k: int = 12) -> List[Dict[str, Any]]:
    model.eval()
    ids = encode_text(prompt, stoi)
    # Remove EOS for continuation prediction.
    ids = ids[:-1]
    ids = ids[-model.cfg.block_size:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)
        last = logits[0, -1, :]
        probs = F.softmax(last, dim=-1)
        vals, inds = torch.topk(probs, k=min(top_k, probs.numel()))
    rows = []
    for rank, (v, i) in enumerate(zip(vals.tolist(), inds.tolist()), start=1):
        rows.append({"rank": rank, "token": itos[int(i)], "probability": float(v)})
    return rows


@torch.no_grad()
def generate(
    model,
    stoi,
    itos,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 20,
    seed: int = 1,
) -> str:
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    model.eval()
    ids = encode_text(prompt, stoi)[:-1]  # no EOS for prompt
    ids = ids[-model.cfg.block_size:]

    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.cfg.block_size:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        logits = logits[0, -1, :]

        # Do not sample PAD/BOS/UNK during generation.
        for bad in [PAD, BOS, UNK]:
            if bad in stoi:
                logits[stoi[bad]] = -float("inf")

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
        if itos[next_id] == EOS:
            break

    torch.random.set_rng_state(rng_state)
    return detokenize([itos[i] for i in ids])


def save_vocab(path: Path, stoi: Dict[str, int], counts: Dict[str, int]):
    payload = {
        "stoi": stoi,
        "itos": {str(i): tok for tok, i in stoi.items()},
        "counts": counts,
        "special_tokens": [PAD, BOS, EOS, UNK],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    stoi = {str(k): int(v) for k, v in payload["stoi"].items()}
    itos = {int(k): str(v) for k, v in payload["itos"].items()}
    return stoi, itos, payload


def load_model_for_generation(outdir: Path, device: torch.device):
    cfg = ModelConfig(**json.loads((outdir / "tiny_transformer_config.json").read_text(encoding="utf-8")))
    model = TinyTransformerLM(cfg).to(device)
    state = torch.load(outdir / "tiny_transformer_model.pt", map_location=device)
    model.load_state_dict(state["model_state_dict"])
    stoi, itos, _ = load_vocab(outdir / "tiny_transformer_vocab.json")
    return model, stoi, itos, cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="tiny_reader_model_prep/corpus_train.jsonl")
    parser.add_argument("--val", default="tiny_reader_model_prep/corpus_val.jsonl")
    parser.add_argument("--test", default="tiny_reader_model_prep/corpus_test.jsonl")
    parser.add_argument("--outdir", default="tiny_reader_transformer_output")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true",
                        help="Load a saved model and run the built-in full-context diagnostic prompts.")
    parser.add_argument("--prompt", default="father has a blue apple")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=80)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.generate_only or args.diagnostics_only:
        model, stoi, itos, cfg = load_model_for_generation(outdir, device)
        print(f"Loaded model from {outdir} on {device}.")

        diagnostic_prompts = [
            "father goes to the yard. mother is near the tree. father has a blue apple. he drops the",
            "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because",
            "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because it is lost. mother helps him look for",
            "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because it is lost. mother helps him look for it. mother finds the",
            "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because it is lost. mother helps him look for it. mother finds the apple. father thanks",
            "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because it is lost. mother helps him look for it. mother finds the apple. father thanks mother. then they",
            "sally goes to the park. father is near the tree. sally has a blue kite. sally sees father. he waves to sally. sally gives him the",
            "when jane goes to the store, tim is there. jane has a blue ball. tim sees the ball. he wants to",
            "tim goes to the",
            "jane goes to the yard. puff comes with her. jane has a yellow toy. she drops the",
        ]

        if args.diagnostics_only:
            prompts_to_run = diagnostic_prompts
            sample_count = 1
        else:
            prompts_to_run = [args.prompt]
            sample_count = args.num_samples

        for prompt in prompts_to_run:
            print("\n" + "=" * 78)
            print(f"Prompt: {prompt!r}")
            print("Top next-token predictions:")
            for row in top_predictions(model, stoi, itos, prompt, device, top_k=12):
                print(f"  {row['rank']:2d}. {row['token']:>12s}  p={row['probability']:.3f}")

            print("\nGenerated continuation(s):")
            for i in range(sample_count):
                print(f"\n--- sample {i+1}")
                print(generate(
                    model, stoi, itos, prompt, device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=args.seed + i,
                ))
        return 0

    train_records = read_jsonl(Path(args.train))
    val_records = read_jsonl(Path(args.val))
    test_records = read_jsonl(Path(args.test)) if Path(args.test).exists() else []

    stoi, itos, counts = build_vocab(train_records, min_count=args.min_count)
    cfg = ModelConfig(
        vocab_size=len(stoi),
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
        pad_id=stoi[PAD],
    )

    train_ds = TinyReaderDataset(train_records, stoi, args.block_size)
    val_ds = TinyReaderDataset(val_records, stoi, args.block_size)
    test_ds = TinyReaderDataset(test_records, stoi, args.block_size) if test_records else None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TinyTransformerLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print("TinyReaderTransformerTrainer")
    print("============================")
    print(f"Device:          {device}")
    print(f"Train records:   {len(train_records)}")
    print(f"Val records:     {len(val_records)}")
    print(f"Train examples:  {len(train_ds)}")
    print(f"Val examples:    {len(val_ds)}")
    print(f"Vocab size:      {len(stoi)}")
    print(f"Parameters:      {n_params:,}")
    print(f"Config:          {cfg}")
    print()

    log_path = outdir / "tiny_transformer_training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "val_loss", "val_perplexity",
            "val_top1_accuracy", "elapsed_sec"
        ])
        writer.writeheader()

        best_val = float("inf")
        best_state = None
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            batches = 0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.item())
                batches += 1

            train_loss = total_loss / max(1, batches)

            if epoch % args.eval_every == 0 or epoch == args.epochs:
                val_metrics = evaluate(model, val_loader, device)
            else:
                val_metrics = {"loss": float("nan"), "perplexity": float("nan"), "top1_accuracy": float("nan")}

            elapsed = time.time() - start_time
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_perplexity": val_metrics["perplexity"],
                "val_top1_accuracy": val_metrics["top1_accuracy"],
                "elapsed_sec": elapsed,
            }
            writer.writerow(row)
            f.flush()

            print(
                f"epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_ppl={val_metrics['perplexity']:.3f} | "
                f"val_acc={val_metrics['top1_accuracy']:.3f} | "
                f"elapsed={elapsed/60:.1f} min"
            )

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_state = {
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "stoi": stoi,
                }
                torch.save(best_state, outdir / "tiny_transformer_model.pt")

    # Save config/vocab.
    (outdir / "tiny_transformer_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    save_vocab(outdir / "tiny_transformer_vocab.json", stoi, counts)

    # Reload best model for final eval/samples.
    model, stoi, itos, cfg = load_model_for_generation(outdir, device)

    final_lines = []
    final_lines.append("Tiny Reader Transformer Evaluation")
    final_lines.append("==================================")
    final_lines.append(f"Device: {device}")
    final_lines.append(f"Parameters: {n_params:,}")
    final_lines.append(f"Best validation loss: {best_val:.4f}")
    final_lines.append("")

    val_metrics = evaluate(model, val_loader, device)
    final_lines.append("Validation:")
    final_lines.append(f"  loss: {val_metrics['loss']:.4f}")
    final_lines.append(f"  perplexity: {val_metrics['perplexity']:.3f}")
    final_lines.append(f"  top1 accuracy: {val_metrics['top1_accuracy']:.3f}")

    if test_ds is not None and len(test_ds) > 0:
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        test_metrics = evaluate(model, test_loader, device)
        final_lines.append("")
        final_lines.append("Test:")
        final_lines.append(f"  loss: {test_metrics['loss']:.4f}")
        final_lines.append(f"  perplexity: {test_metrics['perplexity']:.3f}")
        final_lines.append(f"  top1 accuracy: {test_metrics['top1_accuracy']:.3f}")

    # Full-context diagnostic prompts. Short fragments such as "he drops the"
    # can be misleading for this absolute-position transformer because they
    # place mid-story phrases at the beginning of the sequence. These prompts
    # test what we actually care about: whether the model uses earlier story
    # context to preserve objects, characters, and local narrative state.
    prompts = [
        # Object persistence after a dropped object.
        "father goes to the yard. mother is near the tree. father has a blue apple. he drops the",

        # Pronoun/object continuation after the lost-object setup.
        "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because",

        # Looking for the previously introduced object.
        "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because it is lost. mother helps him look for",

        # Finding the previously introduced object.
        "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because it is lost. mother helps him look for it. mother finds the",

        # Closing the helper sequence.
        "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because it is lost. mother helps him look for it. mother finds the apple. father thanks",

        # End-of-story phrase.
        "father goes to the yard. mother is near the tree. father has a blue apple. he drops the apple. father is sad because it is lost. mother helps him look for it. mother finds the apple. father thanks mother. then they",

        # Level 3 giving/pronoun pattern.
        "sally goes to the park. father is near the tree. sally has a blue kite. sally sees father. he waves to sally. sally gives him the",

        # Level 4 sharing/wants pattern.
        "when jane goes to the store, tim is there. jane has a blue ball. tim sees the ball. he wants to",

        # Setting continuation near story opening.
        "tim goes to the",

        # Pet-companion pattern.
        "jane goes to the yard. puff comes with her. jane has a yellow toy. she drops the",
    ]

    pred_csv = outdir / "tiny_transformer_prompt_predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "rank", "token", "probability"])
        writer.writeheader()
        for prompt in prompts:
            for row in top_predictions(model, stoi, itos, prompt, device, top_k=12):
                writer.writerow({"prompt": prompt, **row})

    sample_lines = []
    for prompt in ["father goes to the yard", "father has a blue apple", "mother helps him look for", "jane goes to the park", "then they"]:
        sample_lines.append(f"PREFIX: {prompt}")
        for i in range(args.num_samples):
            sample_lines.append(f"--- sample {i+1}")
            sample_lines.append(generate(
                model, stoi, itos, prompt, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed + i,
            ))
        sample_lines.append("")
    (outdir / "tiny_transformer_samples.txt").write_text("\n".join(sample_lines), encoding="utf-8")

    final_lines.append("")
    final_lines.append("Top predictions for canonical prompts:")
    for prompt in prompts:
        preds = top_predictions(model, stoi, itos, prompt, device, top_k=5)
        pstr = ", ".join(f"{r['token']} ({r['probability']:.2f})" for r in preds)
        final_lines.append(f"  {prompt!r} -> {pstr}")

    (outdir / "tiny_transformer_eval.txt").write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(final_lines))
    print(f"\nWrote outputs to: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
