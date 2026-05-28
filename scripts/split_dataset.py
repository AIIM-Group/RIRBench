"""Split preprocessed RIR datasets into train / validation / test sets (60/20/20).

Each source dataset is split independently to preserve proportional representation.

Usage:
    python scripts/split_dataset.py --input D:/data/processed --output D:/data/split
"""

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DATASETS = [
    "ACE", "AIR", "Arni", "BBC", "C4DM",
    "DetnoldSRIR", "MIT", "Motus", "MYRiAD_V2", "Palimpsest", "R3VIVAL",
]


def split_dataset(root_dirs, output_train, output_valid, output_eval,
                  valid_frac=0.2, eval_frac=0.2, seed=42):
    random.seed(seed)
    all_files = defaultdict(list)

    for root_dir in root_dirs:
        source = Path(root_dir).name
        files = sorted(Path(root_dir).rglob("*.wav"))
        all_files[source].extend(files)
        print(f"  {source}: {len(files):,} files")

    for d in (output_train, output_valid, output_eval):
        Path(d).mkdir(parents=True, exist_ok=True)

    totals = {"train": 0, "valid": 0, "eval": 0}

    for source, files in all_files.items():
        n_eval = int(len(files) * eval_frac)
        n_valid = int(len(files) * valid_frac)

        eval_subset = random.sample(files, n_eval)
        remaining = [f for f in files if f not in eval_subset]
        valid_subset = random.sample(remaining, n_valid)
        train_subset = [f for f in remaining if f not in valid_subset]

        for subset, out_dir, label in [
            (train_subset, output_train, "train"),
            (valid_subset, output_valid, "valid"),
            (eval_subset, output_eval, "eval"),
        ]:
            for src in subset:
                dst = Path(out_dir) / source / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            totals[label] += len(subset)

        print(f"  {source}: train={len(train_subset)}, valid={len(valid_subset)}, eval={len(eval_subset)}")

    total = sum(totals.values())
    print(f"\nTotal: {total:,}  |  train={totals['train']:,}  valid={totals['valid']:,}  eval={totals['eval']:,}")


def parse_args():
    p = argparse.ArgumentParser(description="Split RIR datasets 60/20/20.")
    p.add_argument("--input", required=True, help="Root directory containing per-dataset subdirectories")
    p.add_argument("--output", required=True, help="Output root; train/valid/eval subdirs are created here")
    p.add_argument("--valid_frac", type=float, default=0.2)
    p.add_argument("--eval_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dirs = [
        os.path.join(args.input, ds)
        for ds in DATASETS
        if Path(args.input, ds).exists()
    ]
    print(f"Found {len(root_dirs)} dataset directories under {args.input}")
    split_dataset(
        root_dirs,
        output_train=os.path.join(args.output, "train"),
        output_valid=os.path.join(args.output, "valid"),
        output_eval=os.path.join(args.output, "eval"),
        valid_frac=args.valid_frac,
        eval_frac=args.eval_frac,
        seed=args.seed,
    )
