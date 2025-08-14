#!/usr/bin/env python3
"""
extract_riot_dataset.py

Scan the last N commits in a local git repository and export
per-file records (for files with > min_changed_lines) to a JSONL file.

IMPORTANT (updated):
- We now export only the unified diff and an additional 'diff_added' field that
  contains ONLY the added lines from the diff (lines starting '+', excluding headers).
- We REMOVE full_file_before/full_file_after and DO NOT include any 'task' field.
  The downstream training will only look at the ADDED text.

Usage:
  python extract_riot_dataset.py --repo "C:/Users/asums/OneDrive/Desktop/RIOT" \
      --out out.jsonl --commits 1000 --min-changed 3

Requirements:
  - git installed
"""

import argparse
import json
import os
import re
import subprocess
from typing import List, Optional, Tuple

REPO_NAME = "RIOT-OS/RIOT"
REPO_URL = "https://github.com/RIOT-OS/RIOT"

DIFF_HEADER_PAT = re.compile(r"^(diff --git|index |--- |\+\+\+ |@@ )")

def run_git(cwd: str, *args) -> str:
    cmd = ["git"] + list(args)
    out = subprocess.check_output(cmd, cwd=cwd)
    return out.decode("utf-8", errors="replace")

def resolve_repo_root(repo_path: str) -> str:
    if repo_path.endswith(os.sep + ".git") or repo_path.endswith(".git"):
        maybe = os.path.dirname(repo_path)
        if os.path.isdir(maybe):
            return maybe
        return repo_path[:-4]
    return repo_path

def list_commits(repo_root: str, max_commits: int) -> List[str]:
    out = run_git(repo_root, "rev-list", f"--max-count={max_commits}", "HEAD")
    return [line.strip() for line in out.splitlines() if line.strip()]

def get_commit_metadata(repo_root: str, sha: str) -> dict:
    fmt = "%H%n%an%n%ae%n%ad%n%B"
    out = run_git(repo_root, "show", "-s", f"--format={fmt}", sha)
    parts = out.splitlines()
    if len(parts) < 4:
        raise RuntimeError(f"Unexpected git show output for {sha}")
    commit_sha = parts[0].strip()
    author = parts[1].strip()
    author_email = parts[2].strip()
    date = parts[3].strip()
    message = "\n".join(parts[4:]).strip()
    pr_number = None
    m = re.search(r"pull request #(\d+)", message, re.IGNORECASE)
    if not m:
        m = re.search(r"\(#(\d+)\)", message)
    if m:
        pr_number = int(m.group(1))
    return {
        "commit_sha": commit_sha,
        "author": author,
        "author_email": author_email,
        "commit_date": date,
        "commit_message": message,
        "pr_number": pr_number,
    }

def list_changed_files(repo_root: str, sha: str) -> List[Tuple[str, str]]:
    out = run_git(repo_root, "diff-tree", "--no-commit-id", "--name-status", "-r", sha)
    res = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R"):
            if len(parts) >= 3:
                res.append(("R", parts[2]))
        else:
            if len(parts) >= 2:
                res.append((status, parts[1]))
    return res

def get_diff_for_file(repo_root: str, sha: str, path: str) -> str:
    try:
        out = run_git(repo_root, "show", f"{sha}", "--", path)
        return out
    except Exception:
        return ""

def get_numstat_for_commit(repo_root: str, sha: str) -> dict:
    out = run_git(repo_root, "show", "--numstat", "--format=", sha)
    res = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ins, dele, path = parts[0], parts[1], parts[2]
        try:
            ins_i = int(ins) if ins != "-" else 0
            del_i = int(dele) if dele != "-" else 0
        except ValueError:
            ins_i, del_i = 0, 0
        res[path] = (ins_i, del_i)
    return res

def classify_file_type(path: str) -> str:
    path_lower = path.lower()
    _, ext = os.path.splitext(path_lower)
    if os.path.basename(path_lower).startswith("readme"):
        return "README"
    if ext in (".c", ".cpp", ".cc", ".rs", ".go", ".java"):
        return "source"
    if ext in (".h", ".hpp"):
        return "header"
    if ext in (".md", ".rst", ".adoc") or path_lower.endswith("doc.md") or "/doc/" in path_lower:
        return "board_doc" if "/boards/" in path_lower or "board" in path_lower else "doc"
    if ext in (".sh", ".py", ".pl", ".rb", ".ps1", ".bat"):
        return "script"
    if os.path.basename(path_lower) == "makefile" or "makefile" in os.path.basename(path_lower):
        return "Makefile"
    if os.path.basename(path_lower).lower().startswith("kconfig") or ext == ".kconfig":
        return "kconfig"
    if os.path.basename(path_lower).upper().startswith("LICENSE") or "license" in path_lower:
        return "license"
    if "/tests/" in path_lower or "/test_" in path_lower or ext in (".tst",):
        return "test"
    return "other"

def extract_added_only(diff_text: str) -> str:
    if not diff_text:
        return ""
    added = []
    for line in diff_text.splitlines():
        if DIFF_HEADER_PAT.match(line):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return "\n".join(added).strip()

def extract_dataset(repo_root: str, out_path: str, max_commits: int = 1000, min_changed_lines: int = 3):
    repo_root = resolve_repo_root(repo_root)
    if not os.path.isdir(repo_root):
        raise SystemExit(f"Repository not found at: {repo_root}")
    print(f"Using repo root: {repo_root}")

    shas = list_commits(repo_root, max_commits)
    print(f"Found {len(shas)} commits (max {max_commits})")

    records_written = 0
    num_scanned = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for sha in shas:
            num_scanned += 1
            try:
                meta = get_commit_metadata(repo_root, sha)
            except Exception as e:
                print(f"Skipping commit {sha}: failed to get metadata: {e}")
                continue

            try:
                numstat = get_numstat_for_commit(repo_root, sha)
            except Exception:
                numstat = {}

            changed = list_changed_files(repo_root, sha)
            if not changed:
                continue

            for status, path in changed:
                if path.endswith(".gitmodules"):
                    continue

                adds, dels = numstat.get(path, (0, 0))
                changed_lines = adds + dels

                if changed_lines <= min_changed_lines:
                    continue

                diff_text = get_diff_for_file(repo_root, sha, path)
                diff_added = extract_added_only(diff_text)

                ftype = classify_file_type(path)

                record = {
                    "repo_name": REPO_NAME,
                    "repo_url": REPO_URL,
                    "commit_sha": meta.get("commit_sha"),
                    "pr_number": meta.get("pr_number"),
                    "author": meta.get("author"),
                    "author_email": meta.get("author_email"),
                    "commit_date": meta.get("commit_date"),
                    "commit_message": meta.get("commit_message"),
                    "file_path": path,
                    "file_type": ftype,
                    "status": status,
                    "additions": adds,
                    "deletions": dels,
                    "changed_lines": changed_lines,
                    "diff": diff_text,
                    "diff_added": diff_added,   # <— NEW: only added lines
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

            if num_scanned % 50 == 0:
                print(f"Scanned {num_scanned}/{len(shas)} commits, wrote {records_written} records so far...")

    print(f"Done. Wrote {records_written} records to {out_path}")

def main():
    p = argparse.ArgumentParser(description="Extract per-file commit dataset from a git repo.")
    p.add_argument("--repo", required=True, help="Path to local git repo (working tree root or .git path)")
    p.add_argument("--out", required=True, help="Output JSONL file path")
    p.add_argument("--commits", type=int, default=1000, help="Number of most recent commits to scan")
    p.add_argument("--min-changed", type=int, default=3, help="Minimum changed lines (add+del) to include file")
    args = p.parse_args()

    extract_dataset(args.repo, args.out, max_commits=args.commits, min_changed_lines=args.min_changed)

if __name__ == "__main__":
    main()
