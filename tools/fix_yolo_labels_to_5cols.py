# tools/fix_yolo_labels_to_5cols.py
#
# ----- FULL CORRECTED FILE -----
#
# This script will:
#  1. Read all .txt files in the `root` directory.
#  2. For each line, keep only the first 5 columns (class x y w h).
#  3. IT WILL NOT DELETE EMPTY FILES, which is crucial for
#     YOLO training (as they represent images with no objects).
#
import pathlib
import sys

# --- IMPORTANT ---
# Point this to your ORIGINAL labels directory.
# We will fix the labels "in-place".
# (Make a backup of this folder first!)
root = pathlib.Path("affordance_dataset/labels")

if not root.exists():
    print(f"Path not found: {root.resolve()}", file=sys.stderr)
    print("Please check the `root` variable in this script.", file=sys.stderr)
    raise SystemExit(f"Path not found: {root}")

fixed_count = 0
processed_count = 0
empty_count = 0

all_files = list(root.rglob("*.txt"))
if not all_files:
    print(f"No .txt files found in {root.resolve()}", file=sys.stderr)
    raise SystemExit("No label files found to process.")

print(f"Processing {len(all_files)} files in {root.resolve()}...")

for p in all_files:
    processed_count += 1
    text = p.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    new_lines = []
    changed = False

    if not text:
        # File was already empty, this is valid.
        empty_count += 1
        continue  # Do nothing

    for i, l in enumerate(text):
        if not l.strip():
            continue  # Skip empty lines within a file

        toks = l.strip().split()

        if len(toks) < 5:
            # Drop invalid labels
            changed = True
            continue

        # Truncate to 5 columns if necessary
        if len(toks) > 5:
            toks = toks[:5]
            changed = True

        # ensure class is integer and the other four are floats
        try:
            cls = int(toks[0])
            # Ensure 4 values for bounding box
            floats = [float(x) for x in toks[1:5]]
            if len(floats) != 4:
                changed = True
                continue # Skip line, invalid bbox
                
            new_lines.append(f"{cls} " + " ".join(f"{x:.6f}" for x in floats))
        except Exception:
            # Skip this line if it's malformed
            changed = True
            continue

    # --- THIS IS THE FIX ---
    # We write the file *no matter what*.
    # If `new_lines` is empty, this correctly creates an empty file.
    # We DO NOT delete the file (p.unlink()).
    
    if changed:
        p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        fixed_count += 1
    
    if not new_lines and not changed:
        # This means the original file had content, but all lines were invalid
        # and resulted in an empty file. This is a "fix".
        p.write_text("\n", encoding="utf-8") # Write empty file
        fixed_count += 1

    if not new_lines:
        empty_count += 1


print(f"--- Done ---")
print(f"Processed: {processed_count} files")
print(f"Fixed:     {fixed_count} files (had 6+ columns or invalid lines)")
print(f"Empty:     {empty_count} files (are now valid empty labels)")