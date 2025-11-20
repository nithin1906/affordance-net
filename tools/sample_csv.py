# tools/sample_csv.py
import argparse
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--csv", required=True, help="Path to the big dataset.csv")
p.add_argument("--out", default="./affordance_crops/dataset_small.csv", help="Path to save the new smaller csv")
p.add_argument("--frac", type=float, default=0.15, help="Fraction of data to keep (e.g., 0.15 for 15%)")
p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = p.parse_args()

print(f"Loading full dataset from {args.csv}...")
df = pd.read_csv(args.csv)

print(f"Original size: {len(df)} samples")
small_df = df.sample(frac=args.frac, random_state=args.seed)
print(f"New size: {len(small_df)} samples ({args.frac * 100}%)")

small_df.to_csv(args.out, index=False)
print(f"Saved small dataset to {args.out}")