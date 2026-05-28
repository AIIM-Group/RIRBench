"""Main evaluation pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from .metrics.drr import drr as _drr
from .metrics.t60 import t60 as _t60
from .metrics.mstft import MultiResolutionSTFTLoss
from .metrics.edr import edr_loss as _edr_loss
from .metrics.peak_similarity import peak_similarity as _peak_sim

ALL_METRICS = ["t60", "drr", "mstft", "edr", "peak_sim"]

_KNOWN_DATASETS = [
    "ACE", "AIR", "Arni", "BBC", "C4DM", "DetmoldSRIR",
    "MIT", "Motus", "MYRIAD_V2", "Palimpsest", "R3VIVAL",
]


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _load_rir(path: Path, sr: int = 48000) -> np.ndarray:
    data, file_sr = sf.read(str(path), always_2d=True)
    mono = data[:, 0].astype(np.float32)
    if file_sr != sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, file_sr)
        mono = resample_poly(mono, sr // g, file_sr // g).astype(np.float32)
    return mono


def _collect_files(source) -> List[Path]:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            files = sorted(p.rglob("*.wav")) + sorted(p.rglob("*.flac"))
            return files
        return [p]
    return [Path(f) for f in source]


def _dataset_tag(path: Path) -> str:
    for part in path.parts:
        for ds in _KNOWN_DATASETS:
            if ds.lower() in part.lower():
                return ds
    return "Unknown"


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

class EvaluationResults:
    """Per-sample and aggregate evaluation results.

    Attributes:
        samples:          list of dicts, one entry per RIR pair
        metrics_computed: list of metric group names that were computed
        sr:               sample rate used
    """

    def __init__(self, samples: List[dict], metrics_computed: List[str], sr: int = 48000):
        self.samples = samples
        self.metrics_computed = metrics_computed
        self.sr = sr

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def _values(self, key: str) -> List[float]:
        return [
            s[key] for s in self.samples
            if key in s and s[key] is not None and not (isinstance(s[key], float) and math.isnan(s[key]))
        ]

    def mean(self, key: str) -> float:
        v = self._values(key)
        return float(np.mean(v)) if v else float("nan")

    def std(self, key: str) -> float:
        v = self._values(key)
        return float(np.std(v)) if v else float("nan")

    def pearson(self, pred_key: str, ref_key: str) -> float:
        pred = self._values(pred_key)
        ref = self._values(ref_key)
        n = min(len(pred), len(ref))
        if n < 2:
            return float("nan")
        return float(np.corrcoef(pred[:n], ref[:n])[0, 1])

    def mse(self, key: str) -> float:
        v = self._values(key)
        return float(np.mean(np.array(v) ** 2)) if v else float("nan")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            "=== RIRBench Evaluation Results ===",
            f"Samples: {len(self.samples)}",
            "",
        ]

        def _fmt(key: str, unit: str = "") -> str:
            m, s = self.mean(key), self.std(key)
            suffix = f" {unit}" if unit else ""
            if math.isnan(m):
                return "N/A"
            return f"{m:.4f} ± {s:.4f}{suffix}"

        if "t60" in self.metrics_computed:
            lines += [
                "Temporal (T60):",
                f"  Bias:  {_fmt('t60_bias', 's')}",
                f"  MSE:   {self.mse('t60_bias'):.4f}",
                f"  ρ:     {self.pearson('t60_pred', 't60_ref'):.4f}",
                "",
            ]

        if "drr" in self.metrics_computed:
            lines += [
                "Energy (DRR):",
                f"  Bias:  {_fmt('drr_bias', 'dB')}",
                f"  MSE:   {self.mse('drr_bias'):.4f}",
                f"  ρ:     {self.pearson('drr_pred', 'drr_ref'):.4f}",
                "",
            ]

        if "mstft" in self.metrics_computed:
            lines += [
                "Spectral (MSTFT):",
                f"  Loss:  {_fmt('mstft_loss')}",
                "",
            ]

        if "edr" in self.metrics_computed and self.samples:
            center_freqs = [16, 32, 63, 125, 250, 500, 1000, 2000, 4000]
            edr_keys_all = [k for k in self.samples[0] if k.startswith("edr_loss_fft") and "log" not in k]
            if edr_keys_all:
                # per-band means (averaged over FFT sizes)
                band_rows = []
                all_band_vals = []
                for fc in center_freqs:
                    fc_keys = [k for k in edr_keys_all if f"_{fc}Hz" in k]
                    band_vals = []
                    for s in self.samples:
                        v = [s[k] for k in fc_keys if not math.isnan(s.get(k, float("nan")))]
                        if v:
                            band_vals.append(float(np.mean(v)))
                    if band_vals:
                        mse_m = float(np.mean(band_vals))
                        log_m = float(-np.log10(mse_m + 1e-10))
                        band_rows.append(f"  {fc:5d} Hz:  MSE={mse_m:.6f}  -log10={log_m:.4f}")
                        all_band_vals.extend(band_vals)
                overall_mse = float(np.mean(all_band_vals)) if all_band_vals else float("nan")
                overall_log = float(-np.log10(overall_mse + 1e-10)) if not math.isnan(overall_mse) else float("nan")
                lines += ["Spectral-Temporal (EDR):  [MSE and -log10(MSE) per octave band]"]
                lines += band_rows
                lines += [
                    f"  Overall:   MSE={overall_mse:.6f}  -log10={overall_log:.4f}",
                    "",
                ]

        if "peak_sim" in self.metrics_computed:
            lines += [
                "Peak Similarity:",
                f"  Weighted (W):   {_fmt('peak_sim_weighted_peak_match')}",
                f"  Unweighted (U): {_fmt('peak_sim_unweighted_peak_match')}",
                "",
            ]

        return "\n".join(lines)

    def to_markdown(self, model_name: str = "Model") -> str:
        """Return evaluation results formatted as a Markdown document."""
        lines = [
            f"# RIRBench Evaluation — {model_name}",
            "",
            f"**Samples:** {len(self.samples)}",
            "",
            "---",
            "",
        ]

        def _fmt(key: str, unit: str = "") -> str:
            m, s = self.mean(key), self.std(key)
            suffix = f" {unit}" if unit else ""
            return f"{m:.4f} ± {s:.4f}{suffix}" if not math.isnan(m) else "N/A"

        if "t60" in self.metrics_computed:
            lines += [
                "## Temporal (T60)",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Bias   | {_fmt('t60_bias', 's')} |",
                f"| MSE    | {self.mse('t60_bias'):.4f} |",
                f"| ρ      | {self.pearson('t60_pred', 't60_ref'):.4f} |",
                "",
            ]

        if "drr" in self.metrics_computed:
            lines += [
                "## Energy (DRR)",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Bias   | {_fmt('drr_bias', 'dB')} |",
                f"| MSE    | {self.mse('drr_bias'):.4f} |",
                f"| ρ      | {self.pearson('drr_pred', 'drr_ref'):.4f} |",
                "",
            ]

        if "mstft" in self.metrics_computed:
            lines += [
                "## Spectral (MSTFT)",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Loss   | {_fmt('mstft_loss')} |",
                "",
            ]

        if "edr" in self.metrics_computed and self.samples:
            center_freqs = [16, 32, 63, 125, 250, 500, 1000, 2000, 4000]
            edr_keys_all = [k for k in self.samples[0] if k.startswith("edr_loss_fft") and "log" not in k]
            if edr_keys_all:
                lines += [
                    "## Spectral-Temporal (EDR)",
                    "",
                    "MSE and −log₁₀(MSE) per octave band, averaged over FFT sizes [1024, 2048, 4096]:",
                    "",
                    "| Frequency Band | MSE | −log₁₀(MSE) |",
                    "|----------------|-----|--------------|",
                ]
                all_band_vals = []
                for fc in center_freqs:
                    fc_keys = [k for k in edr_keys_all if f"_{fc}Hz" in k]
                    band_vals = []
                    for s in self.samples:
                        v = [s[k] for k in fc_keys if not math.isnan(s.get(k, float("nan")))]
                        if v:
                            band_vals.append(float(np.mean(v)))
                    if band_vals:
                        mse_m = float(np.mean(band_vals))
                        log_m = float(-np.log10(mse_m + 1e-10))
                        lines.append(f"| {fc} Hz | {mse_m:.6f} | {log_m:.4f} |")
                        all_band_vals.extend(band_vals)
                if all_band_vals:
                    overall_mse = float(np.mean(all_band_vals))
                    overall_log = float(-np.log10(overall_mse + 1e-10))
                    lines += [
                        f"| **Overall** | **{overall_mse:.4f}** | **{overall_log:.4f}** |",
                        "",
                    ]

        if "peak_sim" in self.metrics_computed:
            lines += [
                "## Peak Similarity",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Weighted (W)   | {_fmt('peak_sim_weighted_peak_match')} |",
                f"| Unweighted (U) | {_fmt('peak_sim_unweighted_peak_match')} |",
                "",
                "---",
                "",
                "*Full per-sample data: `results.csv`*",
            ]

        return "\n".join(lines)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.samples)

    def to_csv(self, path: str):
        self.to_dataframe().to_csv(path, index=False)
        print(f"Results saved to {path}")

    def summary_to_csv(self, path: str) -> None:
        """Save aggregated summary statistics to a CSV file.

        Each row represents one statistic (mean, std, MSE, or Pearson correlation)
        for a given metric, so the file is easy to filter and compare across models.
        """
        import csv

        rows: List[dict] = []

        def _row(metric: str, statistic: str, value: float, unit: str = "") -> dict:
            return {"metric": metric, "statistic": statistic, "value": value, "unit": unit}

        if "t60" in self.metrics_computed:
            rows += [
                _row("T60", "bias_mean", self.mean("t60_bias"), "s"),
                _row("T60", "bias_std",  self.std("t60_bias"),  "s"),
                _row("T60", "mse",       self.mse("t60_bias"),  "s^2"),
                _row("T60", "pearson",   self.pearson("t60_pred", "t60_ref")),
            ]

        if "drr" in self.metrics_computed:
            rows += [
                _row("DRR", "bias_mean", self.mean("drr_bias"), "dB"),
                _row("DRR", "bias_std",  self.std("drr_bias"),  "dB"),
                _row("DRR", "mse",       self.mse("drr_bias"),  "dB^2"),
                _row("DRR", "pearson",   self.pearson("drr_pred", "drr_ref")),
            ]

        if "mstft" in self.metrics_computed:
            rows += [
                _row("MSTFT", "loss_mean", self.mean("mstft_loss")),
                _row("MSTFT", "loss_std",  self.std("mstft_loss")),
            ]

        if "edr" in self.metrics_computed and self.samples:
            center_freqs = [16, 32, 63, 125, 250, 500, 1000, 2000, 4000]
            edr_keys_all = [k for k in self.samples[0] if k.startswith("edr_loss_fft") and "log" not in k]
            if edr_keys_all:
                all_band_vals = []
                for fc in center_freqs:
                    fc_keys = [k for k in edr_keys_all if f"_{fc}Hz" in k]
                    band_vals = []
                    for s in self.samples:
                        v = [s[k] for k in fc_keys if not math.isnan(s.get(k, float("nan")))]
                        if v:
                            band_vals.append(float(np.mean(v)))
                    if band_vals:
                        mse_m = float(np.mean(band_vals))
                        rows.append(_row(f"EDR_{fc}Hz", "mse", mse_m))
                        all_band_vals.extend(band_vals)
                if all_band_vals:
                    rows.append(_row("EDR_overall", "mse", float(np.mean(all_band_vals))))

        if "peak_sim" in self.metrics_computed:
            rows += [
                _row("PeakSim", "weighted_mean",   self.mean("peak_sim_weighted_peak_match")),
                _row("PeakSim", "weighted_std",    self.std("peak_sim_weighted_peak_match")),
                _row("PeakSim", "unweighted_mean", self.mean("peak_sim_unweighted_peak_match")),
                _row("PeakSim", "unweighted_std",  self.std("peak_sim_unweighted_peak_match")),
            ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "statistic", "value", "unit"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary saved to {path}")

    def save(self, output_dir: Union[str, Path], model_name: str = "Model") -> None:
        """Save summary.md, summary.csv, and results.csv to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / "summary.md"
        md_path.write_text(self.to_markdown(model_name), encoding="utf-8")
        print(f"Markdown saved to {md_path}")
        self.summary_to_csv(str(out / "summary.csv"))
        self.to_csv(str(out / "results.csv"))

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Main evaluate() function
# ---------------------------------------------------------------------------

def evaluate(
    predicted_rirs: Union[str, Path, np.ndarray, list],
    reference_rirs: Union[str, Path, np.ndarray, list],
    metrics: Optional[List[str]] = None,
    sr: int = 48000,
    verbose: bool = True,
) -> EvaluationResults:
    """Evaluate predicted RIRs against reference RIRs.

    Args:
        predicted_rirs: directory path, list of paths, single ndarray, or list of ndarrays
        reference_rirs: same types as predicted_rirs; files are matched by sorted order or name
        metrics:        subset of ``["t60", "drr", "mstft", "edr", "peak_sim"]``
                        (default: all)
        sr:             target sample rate for loading; files are resampled if needed
        verbose:        show tqdm progress bar

    Returns:
        :class:`EvaluationResults`

    Example::

        results = evaluate(
            predicted_rirs="path/to/predictions/",
            reference_rirs="path/to/ground_truth/",
            metrics=["t60", "drr", "mstft", "edr", "peak_sim"],
        )
        print(results.summary())
    """
    if metrics is None:
        metrics = list(ALL_METRICS)

    invalid = set(metrics) - set(ALL_METRICS)
    if invalid:
        raise ValueError(f"Unknown metrics: {invalid}. Choose from {ALL_METRICS}.")

    # ------------------------------------------------------------------
    # Build (predicted, reference) pairs
    # ------------------------------------------------------------------
    if isinstance(predicted_rirs, np.ndarray) and isinstance(reference_rirs, np.ndarray):
        pairs = [(predicted_rirs, reference_rirs, "sample_0")]
    else:
        pred_files = _collect_files(predicted_rirs)
        ref_files = _collect_files(reference_rirs)

        # Try name-based matching first
        pred_map = {f.stem: f for f in pred_files}
        ref_map = {f.stem: f for f in ref_files}
        common = sorted(set(pred_map) & set(ref_map))
        if common:
            pairs = [(pred_map[k], ref_map[k], k) for k in common]
        else:
            # Fall back to index-based
            n = min(len(pred_files), len(ref_files))
            if n < max(len(pred_files), len(ref_files)):
                print(
                    f"Warning: {len(pred_files)} predicted vs {len(ref_files)} reference files; "
                    f"using first {n} pairs."
                )
            pairs = [(pred_files[i], ref_files[i], pred_files[i].stem) for i in range(n)]

    if not pairs:
        raise ValueError("No matching RIR pairs found.")

    # ------------------------------------------------------------------
    # Per-sample evaluation
    # ------------------------------------------------------------------
    mstft_fn = MultiResolutionSTFTLoss() if "mstft" in metrics else None

    samples = []
    for item in tqdm(pairs, desc="Evaluating", disable=not verbose):
        pred_src, ref_src, sample_id = item

        # Load
        if isinstance(pred_src, Path):
            pred_arr = _load_rir(pred_src, sr)
            ref_arr = _load_rir(ref_src, sr)
            dataset = _dataset_tag(ref_src)
        else:
            pred_arr = np.squeeze(pred_src).astype(np.float32)
            ref_arr = np.squeeze(ref_src).astype(np.float32)
            dataset = "Unknown"

        row: dict = {"id": sample_id, "dataset": dataset}

        try:
            if "t60" in metrics:
                row["t60_pred"] = _t60(pred_arr, sr)
                row["t60_ref"] = _t60(ref_arr, sr)
                row["t60_bias"] = row["t60_pred"] - row["t60_ref"]

            if "drr" in metrics:
                row["drr_pred"] = _drr(pred_arr, sr)
                row["drr_ref"] = _drr(ref_arr, sr)
                row["drr_bias"] = row["drr_pred"] - row["drr_ref"]

            if "mstft" in metrics:
                pt = torch.from_numpy(pred_arr).unsqueeze(0)
                rt = torch.from_numpy(ref_arr).unsqueeze(0)
                with torch.no_grad():
                    row["mstft_loss"] = mstft_fn(pt, rt)["total"].item()

            if "edr" in metrics:
                pt = torch.from_numpy(pred_arr).unsqueeze(0)
                rt = torch.from_numpy(ref_arr).unsqueeze(0)
                with torch.no_grad():
                    edr = _edr_loss(pt, rt, sr=sr)
                row.update(edr)

            if "peak_sim" in metrics:
                ps = _peak_sim(pred_arr, ref_arr, sr=sr)
                row.update({f"peak_sim_{k}": v for k, v in ps.items()})

        except Exception as exc:
            print(f"Warning: error on sample '{sample_id}': {exc}")

        samples.append(row)

    return EvaluationResults(samples, metrics, sr)
