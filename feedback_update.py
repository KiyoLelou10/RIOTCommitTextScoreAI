#!/usr/bin/env python3
# feedback_update.py
"""
Single-datapoint feedback update with delta accumulation.

Changes vs your version:
- Text fed to the model uses ONLY added lines from the diff (prefer 'diff_added', else extract).
- After one or few gradient steps, we compute 'delta = updated - base' for all params.
- We accumulate these deltas at --agg_path as:
      {'sum': state_dict_like, 'count': N, 'ema': state_dict_like, 'ema_decay': d}
  so future script can apply averaged / EMA deltas to the main model.
- If --save_updated is given, we write a unique file per call (no overwriting).
"""

import argparse
import json
import os
import time
import torch
import torch.nn as nn
import re
from transformers import AutoTokenizer, T5EncoderModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIFF_HEADER_PAT = re.compile(r"^(diff --git|index |--- |\+\+\+ |@@ )")

def extract_added_text_from_unified_diff(diff: str) -> str:
    if not isinstance(diff, str) or not diff:
        return ""
    added = []
    for line in diff.splitlines():
        if DIFF_HEADER_PAT.match(line):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return "\n".join(added).strip()

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_text(r: dict, clip: int = 2000) -> str:
    commit_msg = r.get("commit_message") or ""
    file_path = r.get("file_path") or ""
    file_type = r.get("file_type") or ""
    added = r.get("diff_added")
    if not isinstance(added, str) or not added.strip():
        added = extract_added_text_from_unified_diff(r.get("diff", ""))
    if isinstance(added, str):
        added = added[:clip]
    parts = []
    if commit_msg: parts.append("COMMIT_MESSAGE: " + commit_msg)
    if file_path: parts.append("FILE_PATH: " + file_path)
    if file_type: parts.append("FILE_TYPE: " + file_type)
    if added: parts.append("ADDED_TEXT: " + added)
    return "\n\n".join(parts) if parts else (commit_msg or file_path or "")

class Scorer(nn.Module):
    def __init__(self, encoder: T5EncoderModel):
        super().__init__()
        self.encoder = encoder
        hid = encoder.config.hidden_size
        self.head = nn.Linear(hid, 1)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(x).squeeze(-1)

def predict(tokenizer, model, text: str, max_len: int):
    toks = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    ids, msk = toks["input_ids"].to(DEVICE), toks["attention_mask"].to(DEVICE)
    model.eval()
    with torch.no_grad():
        logit = model(ids, msk)
        prob = torch.sigmoid(logit).item()
    return prob, ids, msk

def effective_weight(n_eff: float, w_single: float, w_max: float) -> float:
    n_eff = max(0.0, float(n_eff))
    W = 1.0 - (1.0 - float(w_single)) ** n_eff
    return float(min(W, float(w_max)))

def _state_dict_delta(sd_updated, sd_base):
    delta = {}
    for k, v in sd_updated.items():
        if k in sd_base and torch.is_tensor(v) and torch.is_tensor(sd_base[k]):
            delta[k] = (v.detach().cpu() - sd_base[k].detach().cpu())
    return delta

def _accumulate_delta(agg_path, delta_dict, ema_decay):
    """
    Maintain a file that stores:
      sum (running sum of deltas),
      count (num of deltas),
      ema (exponential moving average of deltas),
      ema_decay (stored for reference)
    """
    os.makedirs(os.path.dirname(agg_path) or ".", exist_ok=True)
    if os.path.exists(agg_path):
        blob = torch.load(agg_path, map_location="cpu")
        sum_sd = blob["sum"]
        count = int(blob["count"])
        ema_sd = blob.get("ema", None)
        old_decay = float(blob.get("ema_decay", ema_decay))
        if abs(old_decay - ema_decay) > 1e-9:
            # keep existing decay if it changed in file; minor detail
            ema_decay = old_decay
    else:
        sum_sd, count, ema_sd = {}, 0, None

    # Update running sum and count
    for k, dv in delta_dict.items():
        if k not in sum_sd:
            sum_sd[k] = dv.clone()
        else:
            sum_sd[k] += dv
    count += 1

    # Update EMA
    if ema_sd is None:
        # initialize EMA with the first delta
        ema_sd = {k: dv.clone() for k, dv in delta_dict.items()}
    else:
        d = ema_decay
        for k, dv in delta_dict.items():
            if k in ema_sd:
                ema_sd[k] = d * ema_sd[k] + (1.0 - d) * dv.clone()
            else:
                ema_sd[k] = dv.clone()

    torch.save({"sum": sum_sd, "count": count, "ema": ema_sd, "ema_decay": ema_decay}, agg_path)

def main(args):
    rec = load_json(args.single_json)
    text = build_text(rec, clip=args.diff_clip)

    if "ui_prediction" not in rec:
        raise SystemExit("feedback_update requires 'ui_prediction' in [0,1].")
    try:
        ui_p = float(rec["ui_prediction"])
    except Exception:
        raise SystemExit("'ui_prediction' must be parseable as a float in [0,1].")
    if not (0.0 <= ui_p <= 1.0):
        raise SystemExit("'ui_prediction' must be in [0,1].")

    human_val = rec.get("human_score", rec.get("label", rec.get("y_llm", None)))
    if human_val is None:
        raise SystemExit("Need a numeric human score (human_score/label/y_llm) in [0,1].")
    human_score = float(human_val)
    if not (0.0 <= human_score <= 1.0):
        raise SystemExit("Provided human score must be in [0,1].")

    feedback = (rec.get("feedback") or "").strip().lower()
    n_raters = float(rec.get("n_raters", 1.0))
    n_eff = float(rec.get("n_eff", n_raters))

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    enc = T5EncoderModel.from_pretrained(args.model_name).to(DEVICE)
    model = Scorer(enc).to(DEVICE)

    # Load base checkpoint
    base_path = args.ckpt or os.path.join(args.outdir, "best.pt")
    if not os.path.exists(base_path):
        raise SystemExit(f"Base checkpoint not found: {base_path}")
    base_sd = torch.load(base_path, map_location=DEVICE)
    model.load_state_dict(base_sd["model"])

    # Baseline prediction (also for blending)
    prob_before, ids, msk = predict(tok, model, text, args.max_len)

    if args.use_blend:
        W = effective_weight(n_eff=n_eff, w_single=args.w_single, w_max=args.w_max)
        y_target = W * human_score + (1.0 - W) * prob_before
    else:
        W = 1.0
        y_target = human_score

    bce = nn.BCEWithLogitsLoss()
    lambda_reg = args.lambda_reg_like if feedback == "like" else args.lambda_reg_dislike

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for _ in range(args.update_steps):
        logits = model(ids, msk)                  # shape: [1]
        probs = torch.sigmoid(logits)             # shape: [1]
        target = torch.tensor([y_target], dtype=torch.float32, device=DEVICE)
        loss = bce(logits.view(1), target)
        if lambda_reg > 0.0:
            loss = loss + lambda_reg * (probs - ui_p) ** 2
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

    # Compute delta vs base weights (on CPU)
    updated_sd = {"model": model.state_dict()}
    delta_dict = _state_dict_delta(updated_sd["model"], base_sd["model"])
    _accumulate_delta(args.agg_path, delta_dict, args.ema_decay)

    model.eval()
    with torch.no_grad():
        prob_after = torch.sigmoid(model(ids, msk)).item()

    out = {
        "mode": "feedback_update",
        "prob_before": prob_before,
        "ui_prediction": ui_p,
        "feedback": feedback,
        "human_score": human_score,
        "n_eff": n_eff,
        "W_used": W,
        "prob_after": prob_after,
        "delta_accumulated_to": args.agg_path,
    }
    print(json.dumps(out, ensure_ascii=False))

    if args.save_updated:
        os.makedirs(args.per_update_dir, exist_ok=True)
        stamp = int(time.time())
        out_path = os.path.join(args.per_update_dir, f"after_feedback_{stamp}.pt")
        torch.save(updated_sd, out_path)
        print(f"Saved per-call updated checkpoint to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_json", required=True, help="Path to a single feedback JSON record.")
    ap.add_argument("--model_name", default="Salesforce/codet5-base")
    ap.add_argument("--ckpt", default=None, help="Path to main/base model (e.g., ./out/best.pt).")
    ap.add_argument("--outdir", default="./out")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--diff_clip", type=int, default=2000)

    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--update_steps", type=int, default=1)

    # Accumulation settings
    ap.add_argument("--agg_path", default="./out/feedback_agg.pt", help="Where to accumulate deltas.")
    ap.add_argument("--ema_decay", type=float, default=0.9, help="EMA decay for deltas (closer to 1.0 = smoother).")
    ap.add_argument("--save_updated", action="store_true")
    ap.add_argument("--per_update_dir", default="./out/feedback_updates")

    # Blending for small-rater conservatism
    ap.add_argument("--use_blend", action="store_true", help="Blend human_score with prob_before using n_eff.")
    ap.add_argument("--w_single", type=float, default=0.1, help="Per-rater trust increment for blending.")
    ap.add_argument("--w_max", type=float, default=0.8, help="Upper cap for blending weight W.")

    # Stability regularization
    ap.add_argument("--lambda_reg_like", type=float, default=0.1)
    ap.add_argument("--lambda_reg_dislike", type=float, default=0.02)

    main(ap.parse_args())
