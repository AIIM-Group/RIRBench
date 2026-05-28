"""Sample a representative subset of 3,507 RIRs from the full Arni dataset (132,037).

Sampling strategy:
  - Panel configurations are binned into 5 groups by number of reflective panels
  - Microphones 1 and 5 (spatial edges) are preferred; sweep 3 (middle) preferred over sweep 1
  - Samples are drawn proportionally from each acoustic bin

Requires the Arni combinations_setup.csv (available in the Arni download).

Usage:
    python scripts/prepare_arni.py --input D:/Arni/raw --output D:/Arni/sampled
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_panel_configurations(combinations_csv):
    df = pd.read_csv(combinations_csv, sep=";", index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df["num_reflective"] = df.eq(0).sum(axis=1)
    df["acoustic_bin"] = pd.qcut(
        df["num_reflective"], q=5, labels=False, duplicates="drop"
    )
    print(f"Configurations: {len(df)}  |  reflective panels: "
          f"{df['num_reflective'].min()}–{df['num_reflective'].max()}")
    print(df["acoustic_bin"].value_counts().sort_index().to_string())
    return df


def sample_arni(input_dir, combinations_csv, output_size, seed=42):
    random.seed(seed)
    configs_df = analyze_panel_configurations(combinations_csv)

    all_files = list(Path(input_dir).rglob("*.wav"))
    print(f"\nTotal WAV files: {len(all_files):,}")

    configs = defaultdict(lambda: defaultdict(list))
    for f in all_files:
        parts = f.stem.split("_")
        try:
            num_comb = parts[4]
            mic = parts[6]
            configs[num_comb][mic].append(f)
        except IndexError:
            continue

    n_bins = len(configs_df["acoustic_bin"].unique())
    samples_per_bin = output_size // n_bins
    selected = []

    for bin_num in range(n_bins):
        bin_files = []
        bin_configs = configs_df[configs_df["acoustic_bin"] == bin_num].index.astype(str)

        for cfg in bin_configs:
            if cfg not in configs:
                continue
            for mic in ["1", "5"]:
                if mic not in configs[cfg]:
                    continue
                sweep_files = [f for f in configs[cfg][mic] if "sweep_3" in f.name]
                if not sweep_files:
                    sweep_files = [f for f in configs[cfg][mic] if "sweep_1" in f.name]
                if sweep_files:
                    bin_files.append(sweep_files[0])

        n_pick = min(len(bin_files), samples_per_bin)
        chosen = random.sample(bin_files, n_pick)
        selected.extend(chosen)
        print(f"Bin {bin_num}: {len(bin_files)} candidates → {n_pick} selected")

    if len(selected) > output_size:
        selected = random.sample(selected, output_size)

    print(f"\nFinal selection: {len(selected):,} files")
    return selected


def copy_files(selected, input_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(selected):
        dst = output_dir / src.relative_to(Path(input_dir).parent)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if (i + 1) % 100 == 0:
            print(f"  Copied {i + 1:,} files…")
    print(f"Done — {len(selected):,} files written to {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Sample representative Arni RIR subset.")
    p.add_argument("--input", required=True, help="Path to raw Arni directory (contains combinations_setup.csv)")
    p.add_argument("--output", required=True, help="Output directory for sampled files")
    p.add_argument("--size", type=int, default=3507, help="Number of RIRs to sample (default: 3507)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    combinations_csv = str(Path(args.input) / "combinations_setup.csv")
    selected = sample_arni(args.input, combinations_csv, args.size, seed=args.seed)
    copy_files(selected, args.input, args.output)
