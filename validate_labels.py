# validate_labels.py
# Usage: python validate_labels.py --dataset ./affordance_dataset --split train --limit 200
import argparse, os, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def read_yolo_label(lbl_path):
    items=[]
    with open(lbl_path,"r",encoding="utf-8") as f:
        for l in f:
            parts = l.strip().split()
            if not parts: continue
            cid = int(parts[0])
            try:
                cx,cy,w,h = map(float, parts[1:5])
            except:
                continue
            items.append((cid,cx,cy,w,h))
    return items

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train", choices=["train","val","test"])
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()

    ds = Path(args.dataset)
    img_dir = ds / "images" / args.split
    lbl_dir = ds / "labels" / args.split
    if not img_dir.exists():
        print("Image dir not found:", img_dir); sys.exit(1)

    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in [".jpg",".jpeg",".png"]])
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        pass

    n=0
    for imf in img_files:
        if args.limit and n>=args.limit: break
        name = imf.name
        lbl = lbl_dir / (imf.stem + ".txt")
        img = Image.open(imf).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        if lbl.exists():
            items = read_yolo_label(lbl)
            iw,ih = img.size
            for cid,cx,cy,w,h in items:
                xmin = (cx - w/2) * iw
                ymin = (cy - h/2) * ih
                xmax = (cx + w/2) * iw
                ymax = (cy + h/2) * ih
                draw.rectangle([xmin,ymin,xmax,ymax], outline=(0,255,0,255), width=3)
                text = str(cid)
                tw,th = draw.textsize(text, font=font)
                draw.rectangle([xmin, ymin-th-6, xmin+tw+6, ymin], fill=(0,255,0,160))
                draw.text((xmin+3, ymin-th-3), text, fill=(0,0,0,255), font=font)
        else:
            draw.text((10,10),"NO LABEL", fill=(255,0,0,255))
        img.show()
        input("Press Enter to continue to next image (or Ctrl+C to quit)...")
        n+=1

if __name__=="__main__":
    main()
