# tools/inspect_label_ids.py
from pathlib import Path
from collections import defaultdict
p = Path("affordance_dataset/labels")
found = defaultdict(int)
sample = {}
for f in p.rglob("*.txt"):
    for line in f.read_text().splitlines():
        if not line.strip(): continue
        cls = int(line.split()[0])
        found[cls]+=1
        sample.setdefault(cls, f)
print("Unique class ids and counts:")
for k in sorted(found.keys()):
    print(k, found[k], "example:", sample[k])
