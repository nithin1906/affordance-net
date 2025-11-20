#!/usr/bin/env python3
"""
Fix image paths in affordance_crops/dataset.csv so training can find files.

It:
 - makes a backup: affordance_crops/dataset.csv.bak
 - for each row, attempts to resolve the image path using multiple strategies:
    1) as-is (relative to repo root)
    2) relative to CSV parent dir
    3) under affordance_crops/crops
    4) under affordance_crops (top-level)
    5) search by filename under provided search_roots (one-level deep-ish)
 - writes corrected CSV (same header), and prints a short report.
"""
import csv
from pathlib import Path
from collections import defaultdict

CSV = Path("affordance_crops/dataset.csv")
BACKUP = CSV.with_suffix(".csv.bak")
SEARCH_ROOTS = [Path("affordance_crops"), Path("affordance_crops/crops"), Path("affordance_dataset"), Path("affordance_dataset/images")]

def find_existing(path_candidate, repo_root):
    # Check several candidate paths. Return Path or None.
    cand = Path(path_candidate)
    # 1) if absolute or relative as given
    if cand.exists():
        return cand.resolve()
    # 2) relative to CSV parent
    p2 = (CSV.parent / cand)
    if p2.exists():
        return p2.resolve()
    # 3) relative to repo root
    p3 = (repo_root / cand)
    if p3.exists():
        return p3.resolve()
    # 4) try common search roots + candidate
    for sr in SEARCH_ROOTS:
        p = (repo_root / sr / cand.name)
        if p.exists():
            return p.resolve()
    # 5) last resort: search by filename one-level under the listed roots (not recursive heavy)
    for sr in SEARCH_ROOTS:
        root = (repo_root / sr)
        if not root.exists():
            continue
        # first check root/<filename>
        p = root / cand.name
        if p.exists():
            return p.resolve()
        # then check root/*/<filename>
        for child in root.iterdir():
            if child.is_dir():
                p2 = child / cand.name
                if p2.exists():
                    return p2.resolve()
    return None

def main():
    repo_root = Path(".").resolve()
    if not CSV.exists():
        print("ERROR: dataset.csv not found at:", CSV)
        return
    # backup
    if not BACKUP.exists():
        CSV.rename(BACKUP)
        print("Backup created at:", BACKUP)
        src = BACKUP
    else:
        # if backup already present, keep it and read CSV as-is
        print("Backup already exists:", BACKUP)
        src = CSV
    rows = []
    fixed_count = 0
    unchanged_count = 0
    missing = []
    with open(src, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if not row:
                continue
            img = row[0].strip()
            aff = row[1] if len(row) > 1 else ""
            resolved = find_existing(img, repo_root)
            if resolved:
                # make relative path relative to repo root (recommended)
                rel = resolved.relative_to(repo_root)
                rows.append([str(rel).replace("\\","/"), aff])
                if str(rel).replace("\\","/") != img.replace("\\","/"):
                    fixed_count += 1
                else:
                    unchanged_count += 1
            else:
                rows.append([img, aff])
                missing.append(img)
    # write corrected CSV (overwrite original CSV path)
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("Wrote updated CSV to:", CSV)
    print("Fixed paths:", fixed_count, "Unchanged:", unchanged_count, "Missing:", len(missing))
    if missing:
        print("Missing examples (first 20):")
        for m in missing[:20]:
            print(" -", m)

if __name__ == "__main__":
    main()
