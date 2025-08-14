#!/usr/bin/env python3
"""
label_riot_commits_with_openai.py

Batch-label riot commits by sending batches to OpenAI and
saving the same data with a new "label" attribute (float 0..1).

Updates:
- Instruct the model to rate ONLY the quality of natural-language text ADDED in the diff
  (comments/docstrings/README/Doxygen). Ignore code semantics.
- Send a compact payload per item: file_path, file_type, (short) commit_message, and 'added_text'.
- If no added natural-language text, we still return a numeric score (0.5 default handling remains).
- Fixed resume logic.

Usage:
  export OPENAI_API_KEY="sk-..."
  python label_riot_commits_with_openai.py --input riotcommits.jsonl \
      --output riotcommitswithlabel.json --batch-size 20
"""

import argparse
import json
import os
import time
import math
import sys
import random
from typing import List, Any, Tuple, Optional
import re

try:
    from openai import OpenAI
except Exception:
    raise SystemExit(
        "OpenAI Python client not found. Install with: pip install openai"
    )

SYSTEM_PROMPT = (
    "You are a meticulous documentation reviewer.\n"
    "Your task: For EACH item, rate ONLY the quality of the NATURAL-LANGUAGE text that was ADDED in the diff.\n"
    "Added text appears as the field 'added_text'. It may include:\n"
    " - Code comments (//, /* */, #, ///, /** ... */ etc.)\n"
    " - Docstrings and inline docs\n"
    " - README / Markdown / reStructuredText / Doxygen blocks\n"
    "Ignore code logic, compilation, or runtime correctness; focus SOLELY on the writing quality.\n\n"
    "Scoring rubric (0.00–1.00):\n"
    "  • Clarity & readability\n"
    "  • Correctness & accuracy (as text)\n"
    "  • Completeness & usefulness (does it explain what/why sufficiently?)\n"
    "  • Organization & formatting (consistent style, headings, punctuation)\n"
    "  • Tone & consistency with project style\n"
    "If there is effectively no natural-language content to judge, return a sensible, neutral default (e.g., 0.50).\n\n"
    "IMPORTANT — Output format rules:\n"
    "  1) Return ONLY a single JSON array of numbers with the SAME LENGTH and ORDER as the input array.\n"
    "  2) No extra text. Example: [0.75, 0.10, 1.0]\n"
    "  3) Each number must be between 0 and 1 (you may use two decimals).\n"
)

# ----------------- helpers -----------------
DIFF_HEADER_PAT = re.compile(r"^(diff --git|index |--- |\+\+\+ |@@ )")

def load_input_file(path: str) -> Tuple[List[Any], str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return [], "jsonl"
        if text.startswith("["):
            try:
                arr = json.loads(text)
                if isinstance(arr, list):
                    return arr, "array"
            except json.JSONDecodeError:
                pass
        records = []
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse JSON on line {i+1}: {e}")
        return records, "jsonl"

def write_output_file(path: str, records: List[Any], mode: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        if mode == "jsonl":
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def chunkify(lst: List[Any], size: int):
    for i in range(0, len(lst), size):
        yield i, lst[i : i + size]

def extract_json_array_from_response(txt: str) -> List[float]:
    txt = txt.strip()
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
            return [float(x) for x in parsed]
    except Exception:
        pass
    m = re.search(r"\[.*\]", txt, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON array found in model response.")
    arr_text = m.group(0)
    parsed = json.loads(arr_text)
    if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
        return [float(x) for x in parsed]
    raise ValueError("Parsed value is not a numeric array.")

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

def to_compact_payload(batch: List[dict], commit_msg_clip: int = 300, added_clip: int = 2000) -> List[dict]:
    compact = []
    for rec in batch:
        added_text = rec.get("diff_added")
        if not isinstance(added_text, str) or not added_text.strip():
            added_text = extract_added_text_from_unified_diff(rec.get("diff", "")) or ""

        cm = (rec.get("commit_message") or "").strip()
        if len(cm) > commit_msg_clip:
            cm = cm[:commit_msg_clip] + "..."

        item = {
            "file_path": rec.get("file_path", ""),
            "file_type": rec.get("file_type", ""),
            "commit_message": cm,
            "added_text": (added_text[:added_clip] if isinstance(added_text, str) else ""),
        }
        compact.append(item)
    return compact

# ----------------- main -----------------
def label_commits(
    input_path: str,
    output_path: str,
    batch_size: int = 20,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    sleep_between_requests: float = 1.0,
    max_retries: int = 5,
):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()

    records, mode = load_input_file(input_path)
    total = len(records)
    if total == 0:
        print("No records found in input file.")
        return

    print(f"Loaded {total} records from {input_path} (mode={mode}).")

    labeled_records: List[Any] = []
    labeled_count = 0
    output_exists = os.path.exists(output_path)
    if output_exists:
        try:
            existing, out_mode = load_input_file(output_path)
            labeled_records = existing
            labeled_count = len(existing)  # fixed
            print(f"Resuming from index {labeled_count}.")
            if out_mode != mode:
                print("Warning: output format differs from input format; continuing anyway.")
        except Exception as e:
            print(f"Could not load existing output file for resume: {e}. Starting fresh.")
            labeled_records = []
            labeled_count = 0

    for idx, batch in chunkify(records[labeled_count:], batch_size):
        batch_start = labeled_count + idx
        batch_end = batch_start + len(batch)
        print(f"Processing batch {batch_start}..{batch_end-1} ({len(batch)} items)...")

        user_payload = to_compact_payload(batch)
        expected_n = len(user_payload)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"You will be given a JSON array of {expected_n} items. "
                    "Each item has: file_path, file_type, commit_message, added_text.\n"
                    "Return EXACTLY {n} numeric scores [0..1] in a JSON array, same order.\n\n"
                    "Input JSON array:\n\n".replace("{n}", str(expected_n))
                    + json.dumps(user_payload, ensure_ascii=False)
                ),
            },
        ]

        attempt = 0
        skip_batch = False
        while attempt <= max_retries:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1024,
                )
                content = resp.choices[0].message.content
                print("=== Model raw content ===")
                print(content)
                print("=== End model raw content ===\n")

                scores = extract_json_array_from_response(content)
                if len(scores) != expected_n:
                    print(f"Warning: model returned {len(scores)} scores but expected {expected_n}.")
                    if len(scores) > expected_n:
                        scores = scores[:expected_n]
                    else:
                        pad = expected_n - len(scores)
                        scores = scores + [0.5] * pad

                for rec, sc in zip(batch, scores):
                    try: v = float(sc)
                    except Exception: v = 0.5
                    v = max(0.0, min(1.0, v))
                    out_rec = dict(rec)
                    out_rec["label"] = round(v, 2)
                    labeled_records.append(out_rec)

                write_output_file(output_path, labeled_records, mode)
                time.sleep(sleep_between_requests + random.random() * 0.5)
                break

            except Exception as e:
                err_str = str(e)
                token_error_signals = [
                    "rate_limit_exceeded",
                    "Request too large",
                    "'code': 'tokens'",
                    "Requested",
                    "TPM",
                    "tokens per min",
                ]
                is_token_error = any(sig in err_str for sig in token_error_signals)
                attempt += 1
                print(f"Batch failed (attempt {attempt}/{max_retries}) — {e}")
                if is_token_error:
                    print("Token/TPM issue: will skip this batch and continue.")
                    # Save skipped in a sidecar (optional)
                    skipped_file = output_path + ".skipped.jsonl"
                    try:
                        with open(skipped_file, "a", encoding="utf-8") as sf:
                            for rec in batch:
                                sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        print(f"Saved skipped batch to {skipped_file}")
                    except Exception as save_err:
                        print(f"Could not save skipped batch: {save_err}")
                    skip_batch = True
                    break
                if attempt > max_retries:
                    raise RuntimeError(
                        f"Failed to process batch starting at {batch_start} after {max_retries} retries."
                    ) from e
                sleep_t = 2 ** (attempt - 1)
                print(f"Retrying in {sleep_t:.1f}s...")
                time.sleep(sleep_t)

        if skip_batch:
            continue

    print(f"All done. Wrote {len(labeled_records)} labeled records to {output_path}.")

def main():
    p = argparse.ArgumentParser(description="Label riot commits using OpenAI in batches.")
    p.add_argument("--input", required=True, help="Input file (JSONL or JSON array).")
    p.add_argument("--output", default="riotcommitswithlabel.json", help="Output file path.")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--max-retries", type=int, default=5)
    args = p.parse_args()

    label_commits(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        model=args.model,
        temperature=args.temperature,
        api_key=args.api_key,
        sleep_between_requests=args.sleep,
        max_retries=args.max_retries,
    )

if __name__ == "__main__":
    main()
