#!/usr/bin/env python3
"""
apply_accumulated_updates.py

Apply accumulated parameter deltas to the main model.

Inputs:
  --base_ckpt: path to the CURRENT main model (e.g., ./out/best.pt)
  --out_ckpt:  path to write the UPDATED main model (can be same as base to overwrite)
  --agg_files: one or more accumulator .pt files produced by feedback_update.py / single_predict_update.py
  --use_ema:   if set, use EMA deltas from each accumulator instead of (sum/count)
  --weights:   optional per-accumulator weights, same length as --agg_files (default all 1.0)
  --scale:     global step size multiplier for the combined delta (default 1.0)
  --zero_after: if set, zero out the accumulators after applying (i.e., reset)
  --decay_after: if >0, decay the accumulators' sum/ema by this factor after applying (soft reset)

This acts like a “staged” reinforcement update without validation, as requested.
"""

import argparse
import os
import torch

def load_accumulator(path: str):
    blob = torch.load(path, map_location="cpu")
    sum_sd = blob.get("sum", None)
    count = int(blob.get("count", 0))
    ema_sd = blob.get("ema", None)
    ema_decay = float(blob.get("ema_decay", 0.9))
    return sum_sd, count, ema_sd, ema_decay

def average_state_dict(sum_sd, count):
    if not sum_sd or count <= 0:
        return {}
    avg = {}
    for k, v in sum_sd.items():
        avg[k] = v / float(count)
    return avg

def combine_deltas(delta_dicts, weights):
    out = {}
    for d, w in zip(delta_dicts, weights):
        if not d:
            continue
        for k, v in d.items():
            if k not in out:
                out[k] = v.clone() * float(w)
            else:
                out[k] += v * float(w)
    return out

def apply_delta_to_model(model_sd, delta_sd, scale: float):
    new = {}
    for k, v in model_sd.items():
        if k in delta_sd and torch.is_tensor(v) and torch.is_tensor(delta_sd[k]):
            new[k] = v + (delta_sd[k].to(v.dtype).to(v.device) * scale)
        else:
            new[k] = v
    return new

def reset_or_decay_accumulator(path: str, zero_after: bool, decay_after: float):
    if not os.path.exists(path):
        return
    blob = torch.load(path, map_location="cpu")
    if zero_after:
        blob["sum"] = {}
        blob["count"] = 0
        blob["ema"] = {}
    elif decay_after > 0.0:
        if "sum" in blob and isinstance(blob["sum"], dict):
            for k in list(blob["sum"].keys()):
                blob["sum"][k] *= float(decay_after)
        if "ema" in blob and isinstance(blob["ema"], dict):
            for k in list(blob["ema"].keys()):
                blob["ema"][k] *= float(decay_after)
        # keep count as-is (or decay to reflect effective size?). Simpler: keep.
    torch.save(blob, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", default="./out/best.pt", help="Path to current main model.")
    ap.add_argument("--out_ckpt", default="./out/best.pt", help="Path to write updated main model (can overwrite).")
    ap.add_argument("--agg_files", nargs="+", required=True, help="One or more accumulator files.")
    ap.add_argument("--use_ema", action="store_true", help="Use EMA deltas instead of mean (sum/count).")
    ap.add_argument("--weights", nargs="*", type=float, default=None, help="Optional weights per accumulator.")
    ap.add_argument("--scale", type=float, default=1.0, help="Global multiplier for combined delta.")
    ap.add_argument("--zero_after", action="store_true", help="Zero out accumulators after applying.")
    ap.add_argument("--decay_after", type=float, default=0.0, help="Decay factor for accumulators after applying (0=off).")
    args = ap.parse_args()

    if not os.path.exists(args.base_ckpt):
        raise SystemExit(f"Base checkpoint not found: {args.base_ckpt}")
    base = torch.load(args.base_ckpt, map_location="cpu")
    if "model" not in base:
        raise SystemExit("Base checkpoint missing 'model' state_dict.")

    if args.weights is None:
        weights = [1.0] * len(args.agg_files)
    else:
        if len(args.weights) != len(args.agg_files):
            raise SystemExit("--weights length must match --agg_files")
        weights = args.weights

    deltas = []
    for path in args.agg_files:
        if not os.path.exists(path):
            print(f"Warning: accumulator missing: {path}")
            deltas.append({})
            continue
        sum_sd, count, ema_sd, ema_decay = load_accumulator(path)
        use = ema_sd if args.use_ema and ema_sd is not None else average_state_dict(sum_sd, count)
        deltas.append(use)

    combined = combine_deltas(deltas, weights)
    updated_model_sd = apply_delta_to_model(base["model"], combined, scale=args.scale)
    torch.save({"model": updated_model_sd}, args.out_ckpt)
    print(f"Updated model written to {args.out_ckpt}")

    # Optional reset/decay
    for path in args.agg_files:
        reset_or_decay_accumulator(path, args.zero_after, args.decay_after)

if __name__ == "__main__":
    main()
