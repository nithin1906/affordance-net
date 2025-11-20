from pathlib import Path

def check_folder(path):
    p = Path(path)
    if not p.exists() or not any(p.rglob("*.jpg")):
        print(f"❌ Missing or empty folder: {path}")
    else:
        print(f"✅ Found images in: {path}")

ROOT = Path(__file__).resolve().parent / "affordance_dataset"
train = ROOT / "images" / "train"
val = ROOT / "images" / "val"

check_folder(train)
check_folder(val)
