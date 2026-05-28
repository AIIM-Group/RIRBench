"""Spectral peak similarity between predicted and reference RIRs.

Two variants are computed:
- Weighted   (W): peaks are weighted by octave band (lower frequencies weighted higher).
- Unweighted (U): all peaks contribute equally.

A peak is considered matched when its frequency error is < 0.25 octaves and
its magnitude error is < 10 dB.
"""

import numpy as np
import torch
from scipy import signal


def _octave_weight(freq: float) -> float:
    base = 31.5
    octave = np.log2(max(freq, 1e-6) / base)
    return float(2**-octave)


def peak_similarity(
    predicted,
    reference,
    sr: int = 48000,
    n_fft: int = 2048,
    prominence: float = 3.0,
) -> dict:
    """Spectral peak similarity metrics.

    Args:
        predicted:  predicted RIR (tensor or ndarray, 1-D or squeezable)
        reference:  reference RIR
        sr:         sample rate in Hz
        n_fft:      FFT size for spectral analysis
        prominence: minimum peak prominence in dB

    Returns:
        dict with keys:
            ``weighted_peak_match``    — share of reference peaks matched, octave-weighted
            ``unweighted_peak_match``  — share of reference peaks matched, unweighted
            ``freq_error_mean``        — mean weighted frequency error (octaves)
            ``freq_error_std``
            ``mag_error_mean``         — mean weighted magnitude error (dB)
            ``mag_error_std``
    """
    def _to_array(x):
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        return np.squeeze(x).astype(float)

    pred = _to_array(predicted)
    ref = _to_array(reference)

    pred /= np.max(np.abs(pred)) + 1e-8
    ref /= np.max(np.abs(ref)) + 1e-8

    ref_spec = np.abs(np.fft.rfft(ref, n=n_fft))
    pred_spec = np.abs(np.fft.rfft(pred, n=n_fft))

    ref_db = 20 * np.log10(ref_spec + 1e-8)
    pred_db = 20 * np.log10(pred_spec + 1e-8)

    ref_peaks, _ = signal.find_peaks(ref_db, prominence=prominence)
    pred_peaks, _ = signal.find_peaks(pred_db, prominence=prominence)

    freqs = np.linspace(0, sr // 2, len(ref_db))
    ref_peak_freqs = freqs[ref_peaks]
    ref_peak_mags = ref_db[ref_peaks]
    pred_peak_freqs = freqs[pred_peaks]
    pred_peak_mags = pred_db[pred_peaks]

    weights = np.array([_octave_weight(f) for f in ref_peak_freqs])
    total_weight = weights.sum() if len(weights) > 0 else 1.0

    freq_errors: list = []
    mag_errors: list = []
    matched = 0
    weighted_matched = 0.0

    for orig_f, orig_m, w in zip(ref_peak_freqs, ref_peak_mags, weights):
        if len(pred_peak_freqs) == 0:
            break
        closest = np.argmin(np.abs(pred_peak_freqs - orig_f))
        f_err = float(np.abs(np.log2(pred_peak_freqs[closest] / max(orig_f, 1e-6))))
        m_err = float(np.abs(pred_peak_mags[closest] - orig_m))

        freq_errors.append(f_err * w)
        mag_errors.append(m_err * w)

        if f_err < 0.25 and m_err < 10.0:
            matched += 1
            weighted_matched += w

    n_ref = len(ref_peak_freqs)
    return {
        "weighted_peak_match": weighted_matched / total_weight if n_ref > 0 else 0.0,
        "unweighted_peak_match": matched / n_ref if n_ref > 0 else 0.0,
        "freq_error_mean": float(np.sum(freq_errors) / total_weight) if freq_errors else float("inf"),
        "freq_error_std": float(np.std(freq_errors)) if freq_errors else float("inf"),
        "mag_error_mean": float(np.sum(mag_errors) / total_weight) if mag_errors else float("inf"),
        "mag_error_std": float(np.std(mag_errors)) if mag_errors else float("inf"),
    }
