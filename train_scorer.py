#!/usr/bin/env python3
# train_scorer.py
import argparse, json, math, os, random, re
from collections import defaultdict, Counter
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SubsetRandomSampler
from transformers import AutoTokenizer, T5EncoderModel, get_cosine_schedule_with_warmup
from statistics import mean

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Diff helpers
# ----------------------------
DIFF_HEADER_PAT = re.compile(r"^(diff --git|index |--- |\+\+\+ |@@ )")

def extract_added_text_from_unified_diff(diff: str) -> str:
    """
    Keep ONLY added lines from a unified diff (those starting with '+'),
    excluding file headers like '+++ path' and hunk headers '@@'.
    """
    if not isinstance(diff, str) or not diff:
        return ""
    added_lines = []
    for line in diff.splitlines():
        if DIFF_HEADER_PAT.match(line):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            # drop the leading '+'
            added_lines.append(line[1:])
    # Join conservatively; the downstream tokenizer will handle truncation
    return "\n".join(added_lines).strip()

# ----------------------------
# I/O
# ----------------------------
def load_generic_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().strip()
    if not text:
        return []
    if text.startswith("["):
        arr = json.loads(text)
        if isinstance(arr, list):
            return arr
    records = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records

def prepare_records(raw: List[Dict[str, Any]], diff_clip_chars: int = 2000) -> List[Dict[str, Any]]:
    out = []
    for r in raw:
        # accept either explicit y_llm or 'label'; ignore 'task'
        y = r.get("y_llm", r.get("label", None))
        try:
            y = float(y)
        except Exception:
            continue  # skip unlabeled

        commit_msg = r.get("commit_message") or ""
        file_path = r.get("file_path") or ""
        file_type = r.get("file_type") or ""

        # Critically: only use ADDED text. Prefer precomputed 'diff_added', else extract from 'diff'.
        added = r.get("diff_added")
        if not isinstance(added, str) or not added.strip():
            added = extract_added_text_from_unified_diff(r.get("diff", ""))

        if isinstance(added, str):
            added = added[:diff_clip_chars]

        # Build training text from commit message (context), file path (minimal context), and ADDED text
        pieces = []
        if commit_msg:
            pieces.append("COMMIT_MESSAGE: " + commit_msg)
        if file_path:
            pieces.append("FILE_PATH: " + file_path)
        if file_type:
            pieces.append("FILE_TYPE: " + file_type)
        if added:
            pieces.append("ADDED_TEXT: " + added)

        text = "\n\n".join(pieces) if pieces else commit_msg or file_path or ""
        rec_id = f"{r.get('commit_sha','')}:#{file_path}"
        out.append({
            "id": rec_id,
            "text": text,
            "y_llm": float(y),
            "file_type": file_type,
        })
    return out

# ----------------------------
# Dataset & Model
# ----------------------------
class JSONLDataset(Dataset):
    def __init__(self, entries, tokenizer, max_len):
        self.entries = entries
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.entries)
    def __getitem__(self, i):
        e = self.entries[i]
        toks = self.tok(
            e["text"],
            truncation=True, padding="max_length", max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": toks["input_ids"].squeeze(0),
            "attention_mask": toks["attention_mask"].squeeze(0),
            "y": torch.tensor(e["y_llm"], dtype=torch.float32),
            "meta": e["id"]
        }

def collate(b):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in b]),
        "attention_mask": torch.stack([x["attention_mask"] for x in b]),
        "y": torch.stack([x["y"] for x in b]),
        "meta": [x["meta"] for x in b],
    }

class Scorer(nn.Module):
    def __init__(self, encoder: T5EncoderModel):
        super().__init__()
        self.encoder = encoder
        hid = encoder.config.hidden_size
        self.head = nn.Linear(hid, 1)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state                              # [B,T,H]
        mask = attention_mask.unsqueeze(-1).float()            # [B,T,1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # mean pool
        logit = self.head(x).squeeze(-1)
        return logit

# ----------------------------
# Binning, split, sampler
# ----------------------------
def make_bins(vals, step: float):
    def b(v):
        v = max(0.0, min(1.0, float(v)))
        if v == 1.0:
            v = 1.0 - 1e-9
        return round(math.floor(v / step) * step, 3)
    return [b(v) for v in vals], b

def stratified_split(indices_by_bin: Dict[float, List[int]], val_frac: float, seed: int):
    rnd = random.Random(seed)
    train_idx, val_idx = [], []
    for _, idxs in indices_by_bin.items():
        n = len(idxs)
        if n == 0:
            continue
        rnd.shuffle(idxs)
        n_val = max(1, int(round(n * val_frac))) if n >= 10 else max(0, int(n >= 2))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rnd.shuffle(train_idx); rnd.shuffle(val_idx)
    return train_idx, val_idx

def build_epoch_indices(train_records, bin_step, target_per_bin, min_cap, seed):
    rnd = random.Random(seed)
    ys = [float(r["y_llm"]) for r in train_records]
    bins, _ = make_bins(ys, bin_step)
    by_bin = defaultdict(list)
    for i, bb in enumerate(bins):
        by_bin[bb].append(i)
    epoch_idx = []
    for _, idxs in by_bin.items():
        if len(idxs) <= min_cap:
            chosen = idxs[:]
        else:
            cap = min(len(idxs), target_per_bin)
            chosen = rnd.sample(idxs, cap)
        epoch_idx.extend(chosen)
    rnd.shuffle(epoch_idx)
    return epoch_idx, by_bin

# ----------------------------
# Metrics
# ----------------------------
def mae(a, b): return float(torch.mean(torch.abs(a - b)).item())

def pearsonr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.std(unbiased=False) * b.std(unbiased=False)).clamp(min=1e-8)
    return float((a * b).mean().div(denom).item())

# ----------------------------
# Train
# ----------------------------
def train(args):
    raw = load_generic_json(args.data)
    print(f"Loaded {len(raw)} raw")
    records = prepare_records(raw, diff_clip_chars=args.diff_clip)
    print(f"Prepared {len(records)} usable")

    if len(records) < 10:
        raise SystemExit("Not enough labeled records to train.")

    ys = [r["y_llm"] for r in records]
    ybins, _ = make_bins(ys, args.bin_step)
    idxs_by_bin = defaultdict(list)
    for i, bb in enumerate(ybins):
        idxs_by_bin[bb].append(i)
    print("Initial bin counts:")
    for bb in sorted(idxs_by_bin):
        print(f"  {bb:0.2f}: {len(idxs_by_bin[bb])}")

    train_idx, val_idx = stratified_split(idxs_by_bin, args.val_frac, args.seed)
    train_recs = [records[i] for i in train_idx]
    val_recs   = [records[i] for i in val_idx]
    print(f"Train size: {len(train_recs)} | Val size: {len(val_recs)}")

    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    enc = T5EncoderModel.from_pretrained(args.model_name).to(DEVICE)
    model = Scorer(enc).to(DEVICE)

    train_ds = JSONLDataset(train_recs, tok, args.max_len)
    val_ds   = JSONLDataset(val_recs, tok, args.max_len)

    bce = nn.BCEWithLogitsLoss()

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)

    best_val_mae = float("inf")
    best_path = os.path.join(args.outdir, "best.pt")
    patience = args.patience
    bad = 0

    for epoch in range(1, args.epochs + 1):
        epoch_idx, by_bin = build_epoch_indices(train_recs, args.bin_step, args.target_per_bin, args.min_cap, args.seed + epoch)
        if args.use_weighted_after:
            sub_bins = [ybins[train_idx[i]] for i in epoch_idx]
            counts = Counter(sub_bins)
            w = [min(1.0 / counts[b], args.max_weight) for b in sub_bins]
            sampler = WeightedRandomSampler(weights=torch.tensor(w, dtype=torch.double), num_samples=len(epoch_idx), replacement=True, generator=rng)
            loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)
        else:
            sampler = SubsetRandomSampler(epoch_idx, generator=rng)
            loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)

        est_steps = math.ceil(len(epoch_idx) / args.batch_size)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=max(1, int(est_steps * args.warmup_pct)), num_training_steps=max(1, est_steps))

        model.train()
        run_losses = []
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            msk = batch["attention_mask"].to(DEVICE)
            y   = batch["y"].to(DEVICE)
            logits = model(ids, msk)
            loss = bce(logits, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()
            run_losses.append(float(loss.item()))
        print(f"[epoch {epoch}] train_loss={mean(run_losses):.6f} subset_size={len(epoch_idx)}")

        model.eval()
        vlosses, vy, vp = [], [], []
        with torch.no_grad():
            vloader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
            for batch in vloader:
                ids = batch["input_ids"].to(DEVICE)
                msk = batch["attention_mask"].to(DEVICE)
                y   = batch["y"].to(DEVICE)
                logits = model(ids, msk)
                loss = bce(logits, y)
                p = torch.sigmoid(logits)
                vlosses.append(float(loss.item()))
                vy.append(y.cpu())
                vp.append(p.cpu())
        vy = torch.cat(vy, dim=0) if vy else torch.tensor([])
        vp = torch.cat(vp, dim=0) if vp else torch.tensor([])
        v_mae = mae(vp, vy) if len(vy) else float("nan")
        v_pear = pearsonr(vp, vy) if len(vy) > 1 else 0.0
        print(f"[epoch {epoch}] val_loss={mean(vlosses):.6f} val_mae={v_mae:.4f} pearson={v_pear:.3f}")

        if v_mae < best_val_mae:
            best_val_mae = v_mae
            torch.save({"model": model.state_dict()}, best_path)
            print(f"  → saved best to {best_path} (val_mae {best_val_mae:.4f})")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print(f"Best val_mae: {best_val_mae:.4f}")
    final_path = os.path.join(args.outdir, "final.pt")
    torch.save({"model": model.state_dict()}, final_path)
    print(f"Saved final to {final_path}")

def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", default="./out")
    p.add_argument("--model_name", default="Salesforce/codet5-base")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_pct", type=float, default=0.06)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_frac", type=float, default=0.1, help="keep small; use as much data for train as possible")
    p.add_argument("--bin_step", type=float, default=0.05)
    p.add_argument("--target_per_bin", type=int, default=200)
    p.add_argument("--min_cap", type=int, default=50)
    p.add_argument("--max_weight", type=float, default=2.0)
    p.add_argument("--use_weighted_after", action="store_true")
    p.add_argument("--diff_clip", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=3)
    return p

if __name__ == "__main__":
    train(args_parser().parse_args())
