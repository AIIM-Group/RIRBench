"""Radar (spider) plot for multi-metric RIR evaluation comparison."""

from __future__ import annotations

import math
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np

from ..evaluate import EvaluationResults


# (label, lower_better)
_AXES = [
    ("T60 Bias",      True),
    ("T60 MSE",       True),
    ("T60 ρ",    False),
    ("DRR Bias",      True),
    ("DRR MSE",       True),
    ("DRR ρ",    False),
    ("MSTFT Loss",    True),
    ("EDR Loss",      True),
    ("Peak-Sim\n(W)", False),
    ("Peak-Sim (U)",  False),
]

_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]

BASE_FONTSIZE = 14
LABEL_DISTANCE = 1.15


def _edr_mean(samples: list) -> float:
    edr_keys = [k for k in samples[0] if k.startswith("edr_loss_fft")] if samples else []
    vals = []
    for s in samples:
        band = [s[k] for k in edr_keys if not math.isnan(s.get(k, float("nan")))]
        if band:
            vals.append(float(np.mean(band)))
    return float(np.mean(vals)) if vals else float("nan")


def _extract_raw(results: EvaluationResults) -> list:
    """Return one raw scalar per axis (order matches _AXES)."""
    samples = results.samples

    def arr(key):
        return np.array([s.get(key, float("nan")) for s in samples], dtype=float)

    t60_pred, t60_ref = arr("t60_pred"), arr("t60_ref")
    drr_pred, drr_ref = arr("drr_pred"), arr("drr_ref")

    m_t60 = ~(np.isnan(t60_pred) | np.isnan(t60_ref))
    m_drr = ~(np.isnan(drr_pred) | np.isnan(drr_ref))

    t60_bias = float(np.mean(np.abs(t60_pred[m_t60] - t60_ref[m_t60]))) if m_t60.any() else float("nan")
    t60_mse  = float(np.mean((t60_pred[m_t60] - t60_ref[m_t60]) ** 2))  if m_t60.any() else float("nan")
    t60_rho  = results.pearson("t60_pred", "t60_ref")

    drr_bias = float(np.mean(np.abs(drr_pred[m_drr] - drr_ref[m_drr]))) if m_drr.any() else float("nan")
    drr_mse  = float(np.mean((drr_pred[m_drr] - drr_ref[m_drr]) ** 2))  if m_drr.any() else float("nan")
    drr_rho  = results.pearson("drr_pred", "drr_ref")

    mstft  = results.mean("mstft_loss")
    edr    = _edr_mean(samples)
    peak_w = results.mean("peak_sim_weighted_peak_match")
    peak_u = results.mean("peak_sim_unweighted_peak_match")

    return [t60_bias, t60_mse, t60_rho, drr_bias, drr_mse, drr_rho, mstft, edr, peak_w, peak_u]


def _normalize_column(values: np.ndarray, lower_better: bool) -> np.ndarray:
    """Normalize values across models to [0.1, 1.0] where 1.0 = best."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.zeros_like(values)
    mn, mx = valid.min(), valid.max()
    if mx == mn:
        return np.where(np.isnan(values), 0.0, np.ones_like(values))
    if lower_better:
        normed = (mx - values) / (mx - mn)
    else:
        normed = (values - mn) / (mx - mn)
    normed = 0.1 + 0.9 * np.clip(normed, 0.0, 1.0)
    return np.where(np.isnan(values), 0.0, normed)


def _draw_radar(ax, model_names, norm, labels, title):
    """Shared drawing logic for all radar functions."""
    n = len(labels)
    angles = [i / float(n) * 2 * np.pi for i in range(n)]

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    legend_lines = []
    for idx, (name, scores) in enumerate(zip(model_names, norm)):
        plot_scores = np.nan_to_num(scores, nan=0.0).tolist()
        plot_angles = angles + [angles[0]]
        plot_values = plot_scores + [plot_scores[0]]
        color = _COLORS[idx % len(_COLORS)]
        line = ax.plot(plot_angles, plot_values, color=color, linewidth=3,
                       linestyle='-', zorder=5)[0]
        ax.fill(plot_angles, plot_values, color=color, alpha=0.2, zorder=1)
        legend_lines.append(line)

    ax.set_xticks(angles)
    ax.set_xticklabels([])
    for angle, label in zip(angles, labels):
        ax.text(angle, LABEL_DISTANCE, label, fontsize=BASE_FONTSIZE,
                ha='center', va='center', weight='bold', zorder=10)

    for angle in angles:
        ax.plot([angle, angle], [0, LABEL_DISTANCE - 0.05],
                color='gray', linestyle='-', linewidth=0.5)

    for level in [0.25, 0.5, 0.75]:
        circle = plt.Circle((0, 0), level, transform=ax.transData._b,
                            fill=False, edgecolor='gray', linewidth=0.5, alpha=0.5)
        ax.add_patch(circle)

    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    for r, txt in zip([0.25, 0.5, 0.75, 1.0], ['0.25', '0.50', '0.75', '1.0']):
        ax.text(np.pi, r, txt, fontsize=BASE_FONTSIZE - 4,
                ha='center', va='center', color='black', zorder=15)

    ax.set_rlim(0, LABEL_DISTANCE + 0.1)
    ax.spines['polar'].set_visible(False)

    if title:
        ax.set_title(title, size=BASE_FONTSIZE - 1, pad=15, fontweight='bold')

    ax.legend(legend_lines, model_names,
              loc='lower right', bbox_to_anchor=(1.45, 0.02),
              fontsize=BASE_FONTSIZE, framealpha=0.9, ncol=1,
              borderpad=1, handlelength=2, handletextpad=0.8)


def radar_plot(
    results: Dict[str, EvaluationResults],
    save_path: str | None = None,
    figsize: tuple = (14, 8),
    title: str = "RIRBench Evaluation",
) -> plt.Figure:
    """Draw a radar plot comparing multiple models.

    Args:
        results:    dict mapping model names to :class:`EvaluationResults`.
                    At least two models are required — the radar normalises
                    each metric relative to the compared set.
        save_path:  if given, the figure is saved here (PDF/PNG/SVG)
        figsize:    matplotlib figure size
        title:      plot title

    Returns:
        matplotlib Figure

    Raises:
        ValueError: if fewer than two models are provided.
    """
    if len(results) < 2:
        raise ValueError(
            "radar_plot requires at least two models for meaningful comparison. "
            "Scores are normalised relative to the compared set, so a single-model "
            "radar plot carries no information."
        )

    labels = [a[0] for a in _AXES]
    model_names = list(results.keys())
    raw = np.array([_extract_raw(r) for r in results.values()])

    norm = np.zeros_like(raw)
    for j, (_, lb) in enumerate(_AXES):
        norm[:, j] = _normalize_column(raw[:, j], lb)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    _draw_radar(ax, model_names, norm, labels, title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Radar plot saved to {save_path}")

    return fig


def radar_from_raw_metrics(
    metrics: Dict[str, Dict[str, float]],
    save_path: str | None = None,
    figsize: tuple = (14, 8),
    title: str = "RIRBench Evaluation",
) -> plt.Figure:
    """Draw a radar plot from pre-computed aggregate metric values.

    Args:
        metrics:   dict mapping model names to metric dicts, e.g.::

                       {
                           "Model A": {
                               "T60 Bias": 0.01, "T60 MSE": 0.004, "T60 rho": 0.96,
                               "DRR Bias": 1.03, "DRR MSE": 23.4,  "DRR rho": 0.75,
                               "MSTFT Loss": 1.36, "EDR Loss": 0.007,
                               "Peak-Sim W": 0.77, "Peak-Sim U": 0.90,
                           },
                           "Model B": { ... },
                       }

                   At least two models are required.
        save_path: if given, figure is saved here (PDF/PNG/SVG)
        figsize:   matplotlib figure size
        title:     plot title

    Returns:
        matplotlib Figure

    Raises:
        ValueError: if fewer than two models are provided.
    """
    if len(metrics) < 2:
        raise ValueError(
            "radar_from_raw_metrics requires at least two models. "
            "Scores are normalised relative to the compared set."
        )

    _KEY_MAP = {
        "T60 Bias": 0,   "t60_bias": 0,
        "T60 MSE":  1,   "t60_mse":  1,
        "T60 rho":  2,   "T60 ρ": 2, "t60_rho": 2,
        "DRR Bias": 3,   "drr_bias": 3,
        "DRR MSE":  4,   "drr_mse":  4,
        "DRR rho":  5,   "DRR ρ": 5, "drr_rho": 5,
        "MSTFT Loss": 6, "mstft_loss": 6,
        "EDR Loss": 7,   "edr_loss": 7,
        "Peak-Sim W": 8, "Peak-Sim\n(W)": 8, "peak_sim_w": 8,
        "Peak-Sim U": 9, "Peak-Sim (U)": 9, "peak_sim_u": 9,
    }

    model_names = list(metrics.keys())
    raw = np.full((len(model_names), len(_AXES)), float("nan"))
    for m_idx, mvals in enumerate(metrics.values()):
        for key, val in mvals.items():
            a_idx = _KEY_MAP.get(key)
            if a_idx is not None:
                raw[m_idx, a_idx] = float(val)

    norm = np.zeros_like(raw)
    for j, (_, lb) in enumerate(_AXES):
        norm[:, j] = _normalize_column(raw[:, j], lb)

    labels = [a[0] for a in _AXES]
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    _draw_radar(ax, model_names, norm, labels, title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Radar plot saved to {save_path}")

    return fig
