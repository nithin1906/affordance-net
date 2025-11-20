#!/usr/bin/env python3
r"""
train_affordance_multilabel.py
...
(rest of your docstring)
...
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import random
import time

# -------- Dataset --------
class AffordanceDataset(Dataset):
    def __init__(self, rows: List[Tuple[str, List[str]]], root: Path, transform=None):
        r"""
        rows: list of (image_path_str, [aff1, aff2, ...])
        root: Path used to resolve relative image paths
        """
        self.root = Path(root)
        self.items = rows
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def _resolve(self, img_path: str) -> Optional[Path]:
        p = Path(img_path)
        if p.is_absolute() and p.exists():
            return p
        # try relative to root
        p2 = (self.root / p)
        if p2.exists():
            return p2
        # try with path separators normalized
        p3 = (self.root / Path(*p.parts[-2:])) if len(p.parts) >= 2 else (self.root / p.name)
        if p3.exists():
            return p3
        # try root/crops/<name> as a fallback
        p4 = self.root / "affordance_crops" / "crops" / p.name
        if p4.exists():
            return p4
        return None

    def __getitem__(self, idx):
        img_rel, affs = self.items[idx]
        p = self._resolve(img_rel)
        if p is None:
            print(f"Warning: Image not found: {img_rel} (checked relative to {self.root}). Skipping.")
            # Return a dummy tensor and empty list to be filtered by collate_fn
            return torch.empty(0), [], str(p)
        
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"Error loading image {p}: {e}. Skipping.")
            # Return a dummy tensor and empty list to be filtered by collate_fn
            return torch.empty(0), [], str(p)

        if self.transform:
            img = self.transform(img)
        return img, affs, str(p)  # return path as well for debug

# -------- Model builder --------
def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    # Use weights="DEFAULT" for modern PyTorch API
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    m = models.mobilenet_v2(weights=weights)

    in_feat = m.classifier[1].in_features if hasattr(m, "classifier") else 1280
    # replace classifier with custom head
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_feat, num_classes)
    )
    return m

# -------- Utility: read csv --------
def load_csv(csv_path: Path) -> List[Tuple[str, List[str]]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        hdr = next(r, None) # Skip header
        for line in r:
            if not line or len(line) < 2:
                continue
            img = line[0].strip()
            aff_str = line[1].strip()
            if aff_str == "":
                affs = []
            else:
                affs = [a.strip() for a in aff_str.split(",") if a.strip()]
            rows.append((img, affs))
    return rows

# -------- Training / Eval loops --------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    n = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for xb, yb, _paths in pbar:
        if xb.nelement() == 0: # Skip empty batches if collate_fn filtered everything
            continue
        xb = xb.to(device)
        yb = yb.to(device).float()
        
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        pbar.set_postfix(loss=f"{(running_loss / max(n, 1)):.4f}")
        
    return running_loss / max(n, 1)

def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    n = 0
    pbar = tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for xb, yb, _paths in pbar:
            if xb.nelement() == 0:
                continue
            xb = xb.to(device)
            yb = yb.to(device).float()
            out = model(xb)
            loss = criterion(out, yb)
            running_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            pbar.set_postfix(loss=f"{(running_loss / max(n, 1)):.4f}")

    return running_loss / max(n, 1)

#
# ----- THIS IS THE FIX: A class to wrap the collate function -----
# This correctly passes the MultiLabelBinarizer (mlb) to each
# worker process without using a broken global variable.
#
class CollateWrapper:
    def __init__(self, mlb):
        self.mlb = mlb

    def __call__(self, batch):
        # Filter out bad samples (e.g., from image loading errors)
        batch = [b for b in batch if b[0].nelement() > 0]
        if not batch:
            return torch.empty(0), torch.empty(0), []

        imgs = [b[0] for b in batch]
        paths = [b[2] for b in batch]
        imgs = torch.stack(imgs, dim=0)
        
        # Reconstruct multi-hot using self.mlb
        lbls = []
        for b in batch:
            names = b[1]
            try:
                mh = self.mlb.transform([names])[0]  # shape (n_classes,)
                lbls.append(torch.tensor(mh, dtype=torch.float32))
            except Exception as e:
                print(f"Error in collate_fn transforming {names}: {e}")
                # This should not happen if data is clean, but as a fallback:
                lbls.append(torch.zeros(len(self.mlb.classes_), dtype=torch.float32))

        lbls = torch.stack(lbls, dim=0)
        return imgs, lbls, paths
#
# ----- END OF FIX -----
#

# -------- Main --------
def main():
    # We need tqdm for the progress bars
    try:
        global tqdm
        from tqdm import tqdm
    except ImportError:
        print("tqdm not found. Please install: pip install tqdm")
        # Create a dummy tqdm function if not installed
        def tqdm(iterable, *args, **kwargs):
            return iterable

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to dataset csv (image,affordances)")
    parser.add_argument("--root", default=".", help="root folder to resolve images")
    parser.add_argument("--out", default="./backend_models/affordance_clf_multilabel.pt", help="checkpoint output path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val", type=float, default=0.15, help="validation fraction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu' or integer device id")
    parser.add_argument("--augment", action="store_true", help="use simple augmentations")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true", help="use pretrained backbone")
    args = parser.parse_args()

    # seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    csv_path = Path(args.csv)
    root = Path(args.root)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print("Loading CSV...")
    rows = load_csv(csv_path)
    if len(rows) == 0:
        raise RuntimeError("No samples found in CSV. Check dataset.csv path and contents.")

    # Build affordance list (unique affordances)
    print("Building affordance list...")
    all_affs = set()
    for _img, affs in rows:
        for a in affs:
            all_affs.add(a)
            
    if len(all_affs) == 0:
        print("Warning: No affordances found in CSV. Using 'object' as a fallback.")
        all_affs = {"object"}
        rows = [(img, ["object"]) for img, _ in rows]

    affordances = sorted(list(all_affs))
    print(f"Samples in CSV: {len(rows)}, affordances: {len(affordances)}")
    print("Affordances:", affordances)

    # prepare MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=affordances)
    mlb.fit([a for (_i, a) in rows]) # Just fit on all affordances
    
    # --- THIS IS THE FIX ---
    # Create an instance of our wrapper class
    collate_wrapper = CollateWrapper(mlb)
    # --- END FIX ---

    # transforms
    norm_tf = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    base_transforms = [
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        norm_tf,
    ]
    
    if args.augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.imgsz, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ] + base_transforms[1:]) # Add ToTensor and Normalize after augs
    else:
        train_transform = transforms.Compose(base_transforms)

    val_transform = transforms.Compose(base_transforms)

    # Filter rows to only include known affordances
    rows_for_dataset = []
    unknown_affs = set()
    for img, affs in rows:
        known_affs = [a for a in affs if a in mlb.classes_]
        for a in affs:
            if a not in mlb.classes_:
                unknown_affs.add(a)
        rows_for_dataset.append((img, known_affs)) # Keep only known affordances
    
    if unknown_affs:
        print(f"Warning: Ignored {len(unknown_affs)} unknown affordance tags: {unknown_affs}")

    dataset = AffordanceDataset(rows_for_dataset, root, transform=train_transform)
    
    # split
    n_total = len(dataset)
    if n_total < 2:
        raise RuntimeError(f"Dataset is too small ({n_total} samples). Check dataset.csv.")
        
    n_val = max(1, int(n_total * args.val))
    if n_val >= n_total:
        n_val = max(1, n_total - 1) # Need at least one sample for training
        
    n_train = n_total - n_val
    
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    # Hack to set the correct transform for validation set
    val_ds.dataset.transform = val_transform

    # Data loaders
    # --- THIS IS THE FIX ---
    # Pass our class instance as the collate_fn
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_wrapper, persistent_workers=True if args.workers > 0 else False)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_wrapper, persistent_workers=True if args.workers > 0 else False)
    # --- END FIX ---

    # device handling
    dev_str = args.device
    if dev_str.isdigit():
        device = torch.device(f"cuda:{dev_str}")
    else:
        device = torch.device("cuda" if (dev_str == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Training on device: {device}")

    # model
    num_classes = len(affordances)
    model = build_model(num_classes=num_classes, pretrained=args.pretrained).to(device)

    # loss and optimizer (BCEWithLogits for multi-label)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_path = Path(args.out)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Split -> train: {len(train_ds)}, val: {len(val_ds)}")
    print(f"Using {args.workers} workers for data loading.")
    print("Starting training...")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss = eval_one_epoch(model, val_dl, criterion, device)
        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs} — train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  time:{t1-t0:.1f}s")
        
        # save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state_dict": model.state_dict(),
                "affordances": affordances,
                "args": vars(args),
                "best_val": best_val,
            }
            try:
                torch.save(ckpt, str(best_path))
                print(f"📦 Saved best model (val_loss improved): {best_path}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

    # final save
    final_path = best_path.with_name(best_path.stem + "_final" + best_path.suffix)
    try:
        torch.save({
            "model_state_dict": model.state_dict(),
            "affordances": affordances,
            "args": vars(args),
            "best_val": best_val,
        }, str(final_path))
        print(f"Training finished. Best val_loss: {best_val}")
        print(f"Saved final model to: {final_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed/60:.2f} minutes")


if __name__ == "__main__":
    # Add this check for Windows multiprocessing safety
    main()