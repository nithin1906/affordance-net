# tools/check_dataset_csv.py
import csv, os, sys
from pathlib import Path

p = Path("./affordance_crops/dataset.csv")
if not p.exists():
    print("dataset.csv not found at", p.resolve())
    sys.exit(1)

rows = []
with open(p, newline='', encoding='utf-8') as f:
    r = csv.reader(f)
    header = next(r, None)
    for i,row in enumerate(r):
        rows.append(row)
        if i < 10:
            print("row", i, row)
print("Total rows:", len(rows))

# verify image paths exist
missing = []
for row in rows:
    img = Path(row[0])
    if not img.exists():
        missing.append(str(img))
print("Missing image files:", len(missing))
if len(missing) > 0:
    print("Sample missing:", missing[:10])
