# split_dataset.py (fast skip version)
import argparse, shutil, random
from pathlib import Path

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def safe_copy(src: Path, dst: Path, skipped):
    try:
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        return True
    except PermissionError:
        with skipped.open("a", encoding="utf-8") as f:
            f.write(f"LOCKED: {src}\n")
        return False
    except Exception as e:
        with skipped.open("a", encoding="utf-8") as f:
            f.write(f"ERROR: {src} -> {e}\n")
        return False

def copy_with_matches(src_img, dst_img, labels_root, aff_root, subset, skipped):
    safe_copy(src_img, dst_img, skipped)
    base = src_img.stem

    # labels
    label_dst = labels_root / subset
    ensure_dir(label_dst)
    label_src = next((p for p in (labels_root/"train").glob(base+".*")), None)
    if label_src and label_src.suffix == ".txt":
        safe_copy(label_src, label_dst/label_src.name, skipped)
    else:
        (label_dst/(base+".txt")).write_text("")

    # affordances
    aff_dst = aff_root / subset
    ensure_dir(aff_dst)
    for p in (aff_root/"train").glob(base+".*"):
        safe_copy(p, aff_dst/p.name, skipped)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    imgs = root / "images"
    labels = root / "labels"
    aff = root / "affordances"

    src = imgs / "train"
    files = [p for p in src.glob("*") if p.suffix.lower() in [".jpg",".png",".jpeg"]]
    if not files: return print("No images found in", src)

    random.seed(args.seed)
    random.shuffle(files)
    n=len(files); n_t=int(n*args.test); n_v=int(n*args.val); n_tr=n-n_t-n_v
    print(f"{n} images → train {n_tr}, val {n_v}, test {n_t}")

    skipped = root/"split_skipped.txt"
    if skipped.exists(): skipped.unlink()

    for i,p in enumerate(files):
        subset="train" if i<n_tr else "val" if i<n_tr+n_v else "test"
        dst = imgs/subset/p.name
        copy_with_matches(p,dst,labels,aff,subset,skipped)

    print("✅ Split done. Check split_skipped.txt for locked files.")

if __name__=="__main__": main()
