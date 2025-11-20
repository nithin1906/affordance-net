# tools/fix_dataset_csv.py
# Usage: python tools/fix_dataset_csv.py
import csv
import json
from pathlib import Path
from shutil import copy2

ROOT = Path.cwd()
CSV_PATH = ROOT / "affordance_crops" / "dataset.csv"
BACKUP_PATH = ROOT / "affordance_crops" / "dataset.csv.bak"
MAP_PATH = ROOT / "affordance_dataset" / "class_to_affordance.json"

DEFAULT_AFF = "object"

def load_map():
    if MAP_PATH.exists():
        try:
            with open(MAP_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to read {MAP_PATH}: {e}")
    return None

def main():
    if not CSV_PATH.exists():
        print(f"❌ CSV not found at: {CSV_PATH}")
        return

    # backup
    try:
        copy2(CSV_PATH, BACKUP_PATH)
        print(f"🔁 Backup created at: {BACKUP_PATH}")
    except Exception as e:
        print(f"⚠️ Could not create backup: {e}")

    mapping = load_map()
    if mapping:
        print(f"🔍 Loaded affordance map: {MAP_PATH} (will attempt to map classes -> affordances)")

    rows = []
    total = 0
    fixed = 0
    mapped = 0

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # tolerate files without header
        if header and header[0].lower().strip() != "image":
            # header likely part of first row, treat it as data
            f.seek(0)
            reader = csv.reader(f)
            header = next(reader, None)

        for row in reader:
            total += 1
            if not row:
                continue
            # ensure at least two columns
            img = row[0].strip()
            aff = row[1].strip() if len(row) > 1 else ""
            # normalize path separators (use forward slash)
            img = img.replace("\\", "/")
            if not aff:
                aff_to_write = DEFAULT_AFF
                # try to map using filename/class if map exists (map expects class names -> affordances)
                if mapping:
                    # attempt: filename might include class label as prefix or folder, try simple heuristics
                    # e.g. affordance_crops/crops/<class>_...jpg or <class>/...jpg
                    filename = Path(img).name
                    parts = filename.split("_")
                    candidate = parts[0].lower() if parts else ""
                    if candidate in mapping:
                        aff_to_write = ",".join(mapping[candidate]) if isinstance(mapping[candidate], list) else str(mapping[candidate])
                        mapped += 1
                    else:
                        # fallback: if mapping has single class name match by substring
                        for k in mapping:
                            if k.lower() in filename.lower():
                                aff_to_write = ",".join(mapping[k]) if isinstance(mapping[k], list) else str(mapping[k])
                                mapped += 1
                                break
                else:
                    aff_to_write = DEFAULT_AFF
                fixed += 1
            else:
                aff_to_write = aff
            rows.append([img, aff_to_write])

    # write back
    out_path = CSV_PATH
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "affordances"])
        writer.writerows(rows)

    print(f"✅ Done. Processed {total} rows.")
    print(f"  → Fixed empty affordances: {fixed}")
    if mapping:
        print(f"  → Mapped using class_to_affordance.json: {mapped}")
    print(f"Backup: {BACKUP_PATH}")
    print(f"Updated CSV written to: {out_path}")

if __name__ == "__main__":
    main()
