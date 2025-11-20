# backend/train_affordance.py
"""
Train a lightweight affordance multi-label classifier from a CSV produced by autocrop.
Saves a checkpoint containing:
{
  "model_state_dict": ...,
  "affordances": [...],
  "classes": [...],         # optional, original object class names (if known)
  "hyperparams": {...}
}

CSV format expected (header):
image,affordances

Where `image` is path to crop image (relative to repo root or absolute),
and `affordances` is a comma-separated list of affordance names (or empty).
"""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import json
import sys
import os
from collections import defaultdict

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to dataset CSV (image,affordances)")
    p.add_argument("--out", default="./backend_models/affordance_clf_multilabel.pt", help="Output checkpoint path")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val", type=float, default=0.15, help="Fraction for validation split")
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs (and best)")
    return p.parse_args()

# -------------------------
# Dataset
# -------------------------
class AffordanceDataset(Dataset):
    def __init__(self, rows: List[Tuple[Path, List[str]]], affordance_to_idx: dict, imgsz: int):
        self.rows = rows
        self.affordance_to_idx = affordance_to_idx
        self.imgsz = imgsz
        self.transform = T.Compose([
            T.Resize((imgsz, imgsz)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_path, affs = self.rows[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.zeros(len(self.affordance_to_idx), dtype=torch.float32)
        for a in affs:
            if a in self.affordance_to_idx:
                y[self.affordance_to_idx[a]] = 1.0
        return x, y

# -------------------------
# Helpers
# -------------------------
def read_csv_rows(csv_path: Path) -> List[Tuple[Path, List[str]]]:
    rows = []
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for i, row in enumerate(r):
            if not row:
                continue
            img = row[0].strip()
            affs_field = ""
            if len(row) >= 2:
                affs_field = row[1].strip()
            # support semicolon or pipe if used accidentally
            affs_field = affs_field.replace(";", ",").replace("|", ",")
            affs = [a.strip() for a in affs_field.split(",") if a.strip()]
            # if affordances empty, keep empty list (will likely map to 'object' later)
            img_path = Path(img) if Path(img).is_absolute() else (Path.cwd() / img)
            if not img_path.exists():
                # try relative to CSV folder
                alt = csv_path.parent / img
                if alt.exists():
                    img_path = alt
                else:
                    # skip missing images but warn
                    if i < 5:
                        print(f"[WARN] missing image (skipping): {img_path}")
                    continue
            rows.append((img_path, affs))
    return rows

def build_affordance_index(rows: List[Tuple[Path, List[str]]]) -> Tuple[List[str], dict]:
    set_aff = set()
    for _, affs in rows:
        if affs:
            set_aff.update(affs)
    # If empty, fallback to single 'object' affordance
    if not set_aff:
        set_aff = {"object"}
    affordances = sorted(list(set_aff))
    affordance_to_idx = {a: i for i, a in enumerate(affordances)}
    return affordances, affordance_to_idx

# -------------------------
# Model
# -------------------------
def make_model(num_affordances: int, device: torch.device):
    # use pretrained mobilenetv2 backbone for speed
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)
    # replace classifier head to match num_affordances
    in_features = model.classifier[1].in_features
    # Multi-label -> output num_affordances (use BCEWithLogitsLoss)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_affordances)
    )
    return model.to(device)

# -------------------------
# Training loop
# -------------------------
def train_loop(model, device, train_loader, val_loader, epochs, lr, out_path: Path, affordances: List[str], classes: Optional[List[str]]):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
        train_loss = total_loss / max(1, n)

        # val
        model.eval()
        val_loss = 0.0
        nval = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += float(loss.item()) * x.size(0)
                nval += x.size(0)
        val_loss = val_loss / max(1, nval)

        print(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")
        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, out_path, affordances, classes, optimizer, epoch, best=True)
            print(f"📦 Saved best model (val_loss improved): {out_path}")

        # also save periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint(model, out_path.with_suffix(f".epoch{epoch}.pt"), affordances, classes, optimizer, epoch, best=False)

    # final save
    save_checkpoint(model, out_path, affordances, classes, optimizer, epoch, best=False)
    print("Training finished. Final checkpoint saved to:", out_path)

def save_checkpoint(model, out_path: Path, affordances: List[str], classes: Optional[List[str]], optimizer, epoch: int, best: bool):
    # Save state_dict + metadata
    data = {
        "model_state_dict": model.state_dict(),
        "affordances": affordances,
        "classes": classes or [],
        "hyperparams": {"epoch": epoch},
    }
    # optionally include optimizer state
    try:
        data["optimizer_state"] = optimizer.state_dict()
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(out_path))

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device in ["cpu", "cuda"] else args.device)

    csv_path = Path(args.csv)
    print(f"Loading CSV: {csv_path}")
    rows = read_csv_rows(csv_path)
    if not rows:
        raise RuntimeError("No samples found in CSV. Check dataset.csv path and contents.")

    affordances, affordance_to_idx = build_affordance_index(rows)
    print(f"Samples: {len(rows)}, affordances: {len(affordances)}")
    print("Affordances:", affordances)

    # split dataset
    ds = AffordanceDataset(rows, affordance_to_idx, imgsz=args.imgsz)
    n = len(ds)
    nval = max(1, int(args.val * n))
    ntrain = n - nval
    train_ds, val_ds = random_split(ds, [ntrain, nval], generator=torch.Generator().manual_seed(args.seed))
    print(f"Split -> train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=min(8, os.cpu_count() or 1), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=min(8, os.cpu_count() or 1), pin_memory=True)

    # build model
    model = make_model(num_affordances=len(affordances), device=device)
    print("Model created. Params:", sum(p.numel() for p in model.parameters()))

    # train
    out_path = Path(args.out)
    train_loop(model, device, train_loader, val_loader, epochs=args.epochs, lr=args.lr, out_path=out_path, affordances=affordances, classes=None)

if __name__ == "__main__":
    main()
