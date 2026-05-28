"""Split VCTK speech corpus into train / validation / test sets (60/20/20).

Splits per speaker to maintain speaker distribution across all sets.
Only mic1 recordings are used.

Usage:
    python scripts/split_vctk.py --input D:/data/VCTK --output D:/data/VCTK_split
"""

import argparse
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def split_vctk(source_dir, output_train, output_valid, output_eval,
               valid_frac=0.2, eval_frac=0.2, seed=42):
    random.seed(seed)
    source_dir = Path(source_dir)
    all_files = defaultdict(list)

    for f in source_dir.rglob("*.flac"):
        if "mic1" in f.name.lower():
            all_files[f.parent.name].append(f)

    total = sum(len(v) for v in all_files.values())
    print(f"Found {total:,} mic1 files across {len(all_files)} speakers")

    for d in (output_train, output_valid, output_eval):
        Path(d).mkdir(parents=True, exist_ok=True)

    totals = {"train": 0, "valid": 0, "eval": 0}

    for speaker, files in all_files.items():
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
                dst = Path(out_dir) / src.relative_to(source_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            totals[label] += len(subset)

    total = sum(totals.values())
    print(f"Total: {total:,}  |  train={totals['train']:,}  valid={totals['valid']:,}  eval={totals['eval']:,}")


def parse_args():
    p = argparse.ArgumentParser(description="Split VCTK corpus 60/20/20 per speaker.")
    p.add_argument("--input", required=True, help="VCTK source directory")
    p.add_argument("--output", required=True, help="Output root; train/valid/eval subdirs are created here")
    p.add_argument("--valid_frac", type=float, default=0.2)
    p.add_argument("--eval_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_vctk(
        source_dir=args.input,
        output_train=str(Path(args.output) / "train"),
        output_valid=str(Path(args.output) / "valid"),
        output_eval=str(Path(args.output) / "eval"),
        valid_frac=args.valid_frac,
        eval_frac=args.eval_frac,
        seed=args.seed,
    )
