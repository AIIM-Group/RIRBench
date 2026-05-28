import numpy as np
import torch


def drr(rir, fs: int = 48000, direct_window_ms: float = 2.5) -> float:
    """Direct-to-Reverberant Ratio in dB.

    Finds the peak within the first 20 ms, sums energy in a symmetric window
    of `direct_window_ms` around it, and compares to the total reverberant tail.
    """
    if torch.is_tensor(rir):
        rir = rir.cpu().numpy()
    rir = np.squeeze(rir).astype(float)

    n_direct = int(fs * direct_window_ms / 1000)
    peak = np.argmax(np.abs(rir[: int(0.02 * fs)]))
    start = max(0, peak - n_direct // 2)
    end = min(len(rir), start + n_direct)

    e_direct = np.sum(rir[start:end] ** 2)
    e_reverb = np.sum(rir**2) - e_direct
    return float(10 * np.log10(e_direct / (e_reverb + 1e-9)))
