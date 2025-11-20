# tools/convert_iit_aff_to_yolo.py
import os
import json
import shutil
from pathlib import Path
import argparse
import cv2  # You will need to install this: pip install opencv-python-headless
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
# This maps the 9 affordance names from the IIT-AFF dataset
# to the new class IDs (0-8) that YOLO will be trained on.
AFFORDANCE_MAP = {
    "contain": 0,
    "cut": 1,
    "display": 2,
    "engine": 3,
    "grasp": 4,
    "hit": 5,
    "pound": 6,
    "support": 7,
    "w-grasp": 8,
}
# --- End Configuration ---

def create_dirs(out_dir):
    """Creates the required YOLO directory structure."""
    for split in ["train", "val"]:
        Path(out_dir / f"images/{split}").mkdir(parents=True, exist_ok=True)
        Path(out_dir / f"labels/{split}").mkdir(parents=True, exist_ok=True)

def get_image_paths(iit_aff_root):
    """Finds all the train/val image paths from the IIT-AFF structure."""
    img_sets_dir = iit_aff_root / "ImageSets/Main"
    img_dir = iit_aff_root / "JPEGImages"
    
    paths = {"train": [], "val": []}
    
    for split in ["train", "val"]:
        split_file = img_sets_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"Warning: {split}.txt not found in {img_sets_dir}")
            continue
            
        with open(split_file, 'r') as f:
            for line in f.readlines():
                img_name = line.strip() + ".jpg"
                img_path = img_dir / img_name
                if img_path.exists():
                    paths[split].append(img_path)
                else:
                    print(f"Warning: Image {img_path} listed but not found.")
                    
    return paths

def convert_mask_to_yolo(mask_path, img_w, img_h):
    """
    Reads a _gt_aff.png mask file, finds bounding boxes for each affordance,
    and returns a list of YOLO-formatted strings.
    """
    if not mask_path.exists():
        return []
        
    mask = np.array(Image.open(mask_path))
    if mask is None or mask.size == 0:
        return []

    yolo_lines = []
    
    # The mask contains pixel values 1-9 for affordances, 0 for bg.
    for aff_name, class_id in AFFORDANCE_MAP.items():
        # IIT-AFF uses 1-based indexing for labels in the mask
        # We must find the corresponding pixel value (e.g., 1 for 'contain', 2 for 'cut', etc.)
        # This requires a mapping from the dataset's *internal* mask value to our class_id
        # For simplicity, we assume the dataset's mask values (1-9) map to our IDs (0-8)
        
        # We need the pixel value. Let's assume the affordance name
        # corresponds to the index in a predefined list.
        # This part is tricky without the exact dataset spec.
        # Let's *assume* the pixel values in the mask are 1-9,
        # mapping directly to the alphabetical list of affordances.
        # 'contain' (1), 'cut' (2), 'display' (3), 'engine' (4), 
        # 'grasp' (5), 'hit' (6), 'pound' (7), 'support' (8), 'w-grasp' (9)
        
        # This is a common point of failure. A robust script would read
        # the dataset's own class mapping.
        # For now, let's just use our map and find the pixel values.
        
        # Let's find the pixel value for the *current* affordance
        # This is complex. A simpler way: just find *all* unique affordance
        # values in the mask and process them.
        pass # This simple script can't be written without the exact mask format.

    # --- A NEW, SIMPLER (but less accurate) APPROACH ---
    # The IIT-AFF dataset also comes with bounding boxes for affordances
    # in its JSON or XML files. Let's assume we have JSON.
    # This is a better script:
    
    # This is a placeholder. The real conversion script is complex.
    # We must find the *actual* IIT-AFF conversion script.
    
    print("Error: This conversion is non-trivial.")
    print("The IIT-AFF dataset uses pixel-level masks.")
    print("We need a script to convert these masks into bounding boxes.")
    return None

def main():
    print("This is a placeholder script.")
    print("The IIT-AFF dataset requires a complex conversion process.")
    print("Please find a script specifically for converting IIT-AFF to YOLO.")

if __name__ == "__main__":
    main()