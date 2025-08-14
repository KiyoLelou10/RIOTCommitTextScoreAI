# RIOT Commit Doc-Quality Rater

**Purpose.**
This pipeline looks at **newly added documentation/comment text** in RIOT-OS commits (README/Markdown/RST/AsciiDoc lines, Doxygen comments, inline code comments) and assigns a **quality score in \[0,1]**. It avoids leakage from full files or code by extracting **only additions** from diffs, training a small head on CodeT5, and supporting tiny, trust-weighted **online updates** from human feedback. Accumulated micro-updates are merged into the main model (no validation gate, RL-style).

---

## Contents

1. [Architecture](#architecture)
2. [Dependencies & Setup](#dependencies--setup)
3. [JSON Schemas](#json-schemas)
4. [Scripts & Usage](#scripts--usage)

   * [1) Extract commit samples](#1-extract-commit-samples)
   * [2) Label with OpenAI](#2-label-with-openai)
   * [3) Train the scorer](#3-train-the-scorer)
   * [4) Single predict (+ optional tiny supervised step)](#4-single-predict--optional-tiny-supervised-step)
   * [5) Feedback update (trust-weighted EMA)](#5-feedback-update-trustweighted-ema)
   * [6) Apply accumulated updates to main model](#6-apply-accumulated-updates-to-main-model)
5. [Parameters Guide](#parameters-guide)
6. [How the Losses & Updates Work](#how-the-losses--updates-work)
7. [Notes, Limitations, and Next Steps](#notes-limitations-and-next-steps)
8. [Automation Plan (GitHub bot)](#automation-plan-github-bot)

---

## Architecture

```
git repo (RIOT)
   └─ commits
       └─ per-file samples ──> extract_riot_dataset.py
                                  │
                                  ▼
                         riotcommits.jsonl (one file/sample)
                                  │
                                  ▼
                        label_riot_commits_with_openai.py
                                  │  (rates ONLY added doc/comment text)
                                  ▼
                   riotcommitswithlabel.jsonl  (label ∈ [0,1])
                                  │
                                  ▼
                             train_scorer.py
                   (CodeT5 encoder + linear head, BCE on [0,1])
                                  │
                                  ├─► out/best.pt  (main model)
                                  │
         Online usage ────────────┼───────────────────────────────────────┐
                                  │                                       │
                                  ▼                                       ▼
                 single_predict_update.py                     feedback_update.py
               (optional tiny supervised step)      (tiny on-policy step from human feedback)
                                  │                                       │
                    writes unique delta_*.pt                 writes unique delta_*.pt
                         and updates EMA                         and updates EMA
                         out/delta_accum.pt                      out/delta_accum.pt
                                  └───────────────┬───────────────────────┘
                                                  ▼
                                   apply_accumulated_updates.py
                                       (debiased EMA → best.pt)
```

---

## Dependencies & Setup

### Python + packages

* **Python 3.8+**
* Install core packages:

  ```bash
  pip install torch transformers sentencepiece
  ```
* For OpenAI labeling:

  ```bash
  pip install openai
  ```
* (Optional) utilities:

  ```bash
  pip install numpy tqdm
  ```

### Get RIOT and generate commits

Clone RIOT (you can use your local checkout; this just needs `git` metadata):

```bash
git clone https://github.com/RIOT-OS/RIOT.git
```

You don’t need to build RIOT for this pipeline. The extractor only asks `git` for commit/diff metadata.

---

## JSON Schemas

### Extracted sample (one **changed file** per record)

```json
{
  "repo_name": "RIOT-OS/RIOT",
  "repo_url": "https://github.com/RIOT-OS/RIOT",
  "commit_sha": "abcd1234...",
  "pr_number": 123,
  "author": "Jane Dev",
  "author_email": "jane@example.com",
  "commit_date": "Fri May 16 19:33:04 2025 +0200",
  "commit_message": "drivers: ...",
  "file_path": "drivers/include/adt7310.h",
  "file_type": "header",
  "status": "M",
  "additions": 10,
  "deletions": 2,
  "changed_lines": 12,
  "diff": "<full unified diff for this file in this commit>",
  "added_text": "<ONLY newly added doc/comment text>"
}
```

> **One changed file ⇒ one sample**, provided the diff isn’t too small and there is actual **added doc/comment text**.

### Labeled sample

`label_riot_commits_with_openai.py` adds a single numeric label:

```json
{
  ... all fields above ...,
  "label": 0.78
}
```

### Feedback input (multi-rater friendly)

Pass either a single score and rater count:

```json
{
  "file_path": "drivers/include/adt7310.h",
  "added_text": "/* ... newly added Doxygen ... */",
  "ui_prediction": 0.64,
  "human_score": 0.75,
  "n_raters": 2,
  "feedback": "like"
}
```

Or a list of scores (the script averages them and sets `n_raters` automatically if not given):

```json
{
  "file_path": "drivers/include/adt7310.h",
  "added_text": "/* ... */",
  "ui_prediction": 0.64,
  "human_scores": [0.7, 0.8],
  "feedback": "like"
}
```

---

## Scripts & Usage

> All scripts default to using **ONLY `added_text`** (derived from diffs) to avoid training/labeling on full files or unrelated code.

### 1) Extract commit samples

```bash
python extract_riot_dataset.py \
  --repo /path/to/RIOT \
  --out riotcommits.jsonl \
  --commits 1000 \
  --min-changed 3
```

* Produces `riotcommits.jsonl`.
* Each record is one changed file with `diff` and **`added_text`** (doc/comment additions only).
* Records without added documentation/comment lines are **skipped**.

### 2) Label with OpenAI

```bash
export OPENAI_API_KEY="sk-..."
python label_riot_commits_with_openai.py \
  --input riotcommits.jsonl \
  --output riotcommitswithlabel.json \
  --batch-size 6 \
  --model gpt-4o \
  --temperature 0.0
```

* Sends **only** `{id, file_path, added_text}` to the API with a clear rubric.
* Items with empty `added_text` are defaulted to `0.5` and logged as skipped.
* Output preserves input format (JSONL or array) and adds `"label": <float>`.

### 3) Train the scorer

```bash
python train_scorer.py \
  --data riotcommitswithlabel.json \
  --outdir ./out \
  --epochs 6 \
  --batch_size 8 \
  --lr 3e-5 \
  --max_len 1024
```

* Trains a linear head on top of **CodeT5 encoder**.
* Uses **BCEWithLogitsLoss** with soft labels in \[0,1].
* Saves `out/best.pt` and `out/final.pt`.

### 4) Single predict (+ optional tiny supervised step)

```bash
python single_predict_update.py \
  --single_json some_sample.json \
  --outdir ./out \
  --ckpt ./out/best.pt \
  --update_steps 1 \
  --step_weight 1.0 \
  --ema_beta 0.9
```

* Prints a prediction for `added_text`.
* If `some_sample.json` contains a `label`, the script performs **one tiny supervised step**, then:

  * Saves a unique per-step delta `out/updates_single/delta_*.pt`.
  * Updates the shared **EMA accumulator** `out/delta_accum.pt`.

### 5) Feedback update (trust-weighted EMA)

```bash
python feedback_update.py \
  --single_json feedback.json \
  --outdir ./out \
  --ckpt ./out/best.pt \
  --use_blend --w_single 0.1 --w_max 0.8 \
  --update_steps 1 \
  --step_weight 1.0 \
  --ema_beta 0.9
```

* Uses `ui_prediction` + human score(s) to compute a **blended target**:

  * `W = 1 - (1 - w_single) ** n_eff` (more raters ⇒ higher W),
  * `y_target = W * human_score + (1 - W) * prob_before`.
* Runs a tiny on-policy step with a stability term:

  * `BCEWithLogits(z, y_target) + λ*(σ(z) - ui_prediction)^2`
  * `λ` is larger when feedback is “like” (preserve a liked pred more).
* Writes a unique delta file and updates the EMA accumulator.

### 6) Apply accumulated updates to main model

```bash
python apply_accumulated_updates.py \
  --outdir ./out \
  --ckpt best.pt \
  --agg ./out/delta_accum.pt \
  --merge_scale 1.0 \
  --clip_global_norm 10.0
```

* Loads `best.pt`, reads `delta_accum.pt` (EMA), computes **debiased** EMA:

  * `m_hat = ema / (1 - beta**t)`
* Optionally clips global norm, then **adds to weights**:

  * `θ ← θ + merge_scale * m_hat`
* Backs up previous checkpoint and overwrites `best.pt`.

---

## Parameters Guide

Common useful flags:

* `--repo` (extract): path to RIOT checkout.
* `--commits` (extract): how many recent commits to scan.
* `--batch-size` (label): items per API call.
* `--model`, `--temperature` (label): OpenAI model & sampling.
* `--epochs`, `--lr`, `--batch_size`, `--max_len` (train): the usual suspects.
* `--update_steps` (single/feedback): tiny step count (1 is typical).
* `--step_weight` (single/feedback): scales each per-step delta before EMA.
* `--ema_beta` (single/feedback): EMA decay (0.9–0.99 is common).
* `--use_blend`, `--w_single`, `--w_max` (feedback): trust weighting for multiple raters.
* `--merge_scale`, `--clip_global_norm` (apply): safety knobs when merging EMA into main model.

---

## How the Losses & Updates Work

### Training loss (supervised)

* We predict a logit `z`; probability `p = σ(z)`.
* Labels are soft in `[0,1]`. We use:

  * **BCEWithLogitsLoss**: `L = - y*log σ(z) - (1-y)*log(1-σ(z))`.
  * Works well for bounded quality scores and keeps gradients healthy near the edges.

### Tiny supervised step (single\_predict\_update.py)

* If a `label` is present, we run **one** BCE step on that item.
* Compute the parameter change `Δθ` = (after – before), scale/clip, write a unique `delta_*.pt`, and update the **EMA accumulator**:

  * `ema_t = β * ema_{t-1} + (1-β) * Δθ`.

### Feedback step (feedback\_update.py)

* Construct **blended target**:

  * `W = 1 - (1 - w_single) ** n_eff` (more raters → higher W, capped by `w_max`).
  * `y_target = W * human_score + (1 - W) * prob_before`.
* Loss:

  * `L = BCEWithLogits(z, y_target) + λ * (σ(z) - ui_prediction)^2`
  * The **stability term** discourages big moves away from the user-rated UI value, especially when `feedback == "like"` (higher λ).
* Produce `Δθ`, scale by `step_weight * W`, clip, save, update EMA.

### Merging EMA into main model (apply\_accumulated\_updates.py)

* **Debias**:

  * `m_hat = ema / (1 - β^t)` (classic bias correction).
* **Clip** (optional global norm).
* **Apply**:

  * `θ ← θ + merge_scale * m_hat`
* **Backup** old `best.pt`, then overwrite.

This “EMA of deltas” is a simple, robust **policy-improvement accumulator**: you ingest lots of tiny, noisy updates but keep the main model stable, only moving it by the long-run trend (the debiased EMA).

---

## Notes, Limitations, and Next Steps

* I couldn’t fully test this exact code end-to-end on your side. I previously worked with a very similar pipeline that **included full file before/after**, but those datasets weren’t ideal (the model could learn shortcuts from non-added content) and since I had no resources left on OpenAI I could not re label the new dataset without paying extra, this is an important part for all the OpenAI code to work, you will need to pay some money to use the API which works that you pay as you go, you can if you do not want to pay money also spend some time and divide the dataset into lets say a set of 8 samples N times (where N * 8 is the size of the dataset) and manually give an llm these along the prompt from the code to get labels. 
* This version fixes that by using **only `added_text`**. Still, there may be small integration issues to iron out (paths, environments, or corner-case diffs).
* Consider extracting the shared `extract_added_doc_text` into a small utility module to remove duplication.

---

## Automation Plan (GitHub bot)

**Goal:** after you train and ship `out/best.pt`, a bot should:

1. **On each PR or push**:

   * Run the extractor on the commit(s) touched by the PR.
   * For each changed file with doc/comment additions, run `single_predict_update.py` (no update unless a label is bundled), and **post the score** as a PR comment/status.
   * Optionally attach a rubric-based suggestion (“consider adding @param/@return”, grammar tweaks, etc.) in the PR comment (separate tool).

2. **When maintainers give feedback**:

   * Collect feedback via a PR comment command (e.g., `/docscore like 0.8`).
   * Store a tiny JSON (per file) and run `feedback_update.py` to update the EMA accumulator (`delta_accum.pt`).

3. **Weekly cron**:

   * Run `apply_accumulated_updates.py` to merge the EMA into `best.pt`.
   * Commit the updated `best.pt` (and back up automatically).

This can be built with **GitHub Actions** + a small bot (GitHub App) that listens for PR events and review comments, runs the scripts in a Python action, and manages artifacts (`out/`).
