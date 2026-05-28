import numpy as np
import torch


def t60(rir, fs: int = 48000) -> float:
    """Reverberation time T60 in seconds via backwards integration (Schroeder).

    If the energy decay curve does not reach -60 dB, T60 is extrapolated from
    the overall slope of the curve.
    """
    if torch.is_tensor(rir):
        rir = rir.cpu().numpy()
    rir = np.squeeze(rir).astype(float)

    edc = np.cumsum(rir[::-1] ** 2)[::-1]
    edc_db = 10 * np.log10(edc / (np.max(edc) + 1e-10) + 1e-10)

    below = np.where(edc_db < -60)[0]
    if len(below) > 0:
        return float(below[0] / fs)

    slope = (edc_db[-1] - edc_db[0]) / len(edc_db)
    if slope < 0:
        return float(int(-60 / slope) / fs)
    return float(len(edc_db) / fs)
