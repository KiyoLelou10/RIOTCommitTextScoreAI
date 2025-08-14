#!/usr/bin/env python3
# single_predict_update.py
import argparse, json, os, time, re, torch, torch.nn as nn
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

def load_json(path):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def build_text(r, diff_clip=2000):
    commit_msg = r.get("commit_message") or ""
    file_path = r.get("file_path") or ""
    file_type = r.get("file_type") or ""
    added = r.get("diff_added")
    if not isinstance(added, str) or not added.strip():
        added = extract_added_text_from_unified_diff(r.get("diff", ""))
    if isinstance(added, str):
        added = added[:diff_clip]
    parts = []
    if commit_msg: parts.append("COMMIT_MESSAGE: " + commit_msg)
    if file_path: parts.append("FILE_PATH: " + file_path)
    if file_type: parts.append("FILE_TYPE: " + file_type)
    if added: parts.append("ADDED_TEXT: " + added)
    return "\n\n".join(parts) if parts else commit_msg or file_path or ""

class Scorer(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hid = encoder.config.hidden_size
        self.head = torch.nn.Linear(hid, 1)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(x).squeeze(-1)

def predict(tok, model, text, max_len):
    toks = tok(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    ids, msk = toks["input_ids"].to(DEVICE), toks["attention_mask"].to(DEVICE)
    model.eval()
    with torch.no_grad():
        logit = model(ids, msk)
        prob = torch.sigmoid(logit).item()
    return prob, ids, msk

def _state_dict_delta(updated, base):
    d = {}
    for k, v in updated.items():
        if k in base and torch.is_tensor(v) and torch.is_tensor(base[k]):
            d[k] = (v.detach().cpu() - base[k].detach().cpu())
    return d

def _accumulate_delta(agg_path, delta_dict, ema_decay):
    os.makedirs(os.path.dirname(agg_path) or ".", exist_ok=True)
    if os.path.exists(agg_path):
        blob = torch.load(agg_path, map_location="cpu")
        sum_sd = blob["sum"]
        count = int(blob["count"])
        ema_sd = blob.get("ema", None)
        old_decay = float(blob.get("ema_decay", ema_decay))
        if abs(old_decay - ema_decay) > 1e-9:
            ema_decay = old_decay
    else:
        sum_sd, count, ema_sd = {}, 0, None

    for k, dv in delta_dict.items():
        if k not in sum_sd:
            sum_sd[k] = dv.clone()
        else:
            sum_sd[k] += dv
    count += 1

    if ema_sd is None:
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
    text = build_text(rec, diff_clip=args.diff_clip)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    enc = T5EncoderModel.from_pretrained(args.model_name).to(DEVICE)
    model = Scorer(enc).to(DEVICE)

    base_path = args.ckpt or os.path.join(args.outdir, "best.pt")
    if not os.path.exists(base_path):
        raise SystemExit(f"Base checkpoint not found: {base_path}")
    base_sd = torch.load(base_path, map_location=DEVICE)
    model.load_state_dict(base_sd["model"])

    prob_before, ids, msk = predict(tok, model, text, args.max_len)
    print(json.dumps({"mode":"predict", "prob_before": prob_before}, ensure_ascii=False))

    label = rec.get("y_llm", rec.get("label", None))
    if label is not None and args.update_steps > 0:
        y = torch.tensor([float(label)], dtype=torch.float32, device=DEVICE)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        bce = nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(args.update_steps):
            logits = model(ids, msk)
            loss = bce(logits.view(1), y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

        # accumulate delta vs base
        updated_sd = {"model": model.state_dict()}
        delta = _state_dict_delta(updated_sd["model"], base_sd["model"])
        _accumulate_delta(args.agg_path, delta, args.ema_decay)

        model.eval()
        with torch.no_grad():
            prob_after = torch.sigmoid(model(ids, msk)).item()
        print(json.dumps({"mode":"single_update", "label": float(label), "prob_after": prob_after, "delta_accumulated_to": args.agg_path}, ensure_ascii=False))

        if args.save_updated:
            os.makedirs(args.per_update_dir, exist_ok=True)
            stamp = int(time.time())
            out_path = os.path.join(args.per_update_dir, f"single_update_{stamp}.pt")
            torch.save(updated_sd, out_path)
            print(f"Saved per-call updated checkpoint to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_json", required=True)
    ap.add_argument("--model_name", default="Salesforce/codet5-base")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--outdir", default="./out")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--diff_clip", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--update_steps", type=int, default=1)

    # accumulation
    ap.add_argument("--agg_path", default="./out/single_agg.pt", help="Where to accumulate deltas.")
    ap.add_argument("--ema_decay", type=float, default=0.9, help="EMA decay for deltas.")
    ap.add_argument("--save_updated", action="store_true")
    ap.add_argument("--per_update_dir", default="./out/single_updates")

    main(ap.parse_args())
