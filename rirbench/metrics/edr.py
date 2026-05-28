"""Energy Decay Relief (EDR) loss across nine octave bands (16 Hz – 4 kHz).

For each band the EDR is computed as the backward cumulative energy of the
STFT magnitude spectrum, normalised by the first frame, and the MSE between
the predicted and reference curves is returned.

Reference: paper Section III.D.
"""

import torch


_CENTER_FREQS = [16, 32, 63, 125, 250, 500, 1000, 2000, 4000]


def edr_loss(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    sr: int = 48000,
    fft_sizes: tuple = (1024, 2048, 4096),
) -> dict:
    """EDR loss per octave band and FFT size.

    Args:
        predicted: (B, T) or (B, 1, T) predicted RIR
        reference: (B, T) or (B, 1, T) reference RIR
        sr:        sample rate in Hz
        fft_sizes: tuple of FFT sizes

    Returns:
        dict with keys ``edr_loss_fft{size}_{freq}Hz`` for each combination,
        plus ``log_edr_loss_*`` variants (negative log10, higher = better).
    """
    def _squeeze2d(x: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(x):
            if x.dim() > 2:
                x = x.squeeze(1)
            return x.float()
        x = torch.tensor(x, dtype=torch.float32)
        return x.unsqueeze(0) if x.dim() == 1 else x

    pred = _squeeze2d(predicted)
    ref = _squeeze2d(reference)

    freq_bounds = [
        (fc, fc / 2**0.5, fc * 2**0.5) for fc in _CENTER_FREQS
    ]

    results = {}
    window = torch.hann_window(fft_sizes[0], device=pred.device)

    for fft_size in fft_sizes:
        hop = fft_size // 2
        if window.size(0) != fft_size:
            window = torch.hann_window(fft_size, device=pred.device)

        pred_stft = torch.stft(pred, fft_size, hop, fft_size, window, return_complex=True)
        ref_stft = torch.stft(ref, fft_size, hop, fft_size, window, return_complex=True)

        pred_pow = torch.abs(pred_stft) ** 2
        ref_pow = torch.abs(ref_stft) ** 2

        for fc, f_lo, f_hi in freq_bounds:
            lo_bin = int(f_lo * fft_size / sr)
            hi_bin = min(int(f_hi * fft_size / sr), pred_pow.shape[1])

            key = f"edr_loss_fft{fft_size}_{fc}Hz"
            log_key = f"log_edr_loss_fft{fft_size}_{fc}Hz"

            if hi_bin <= lo_bin or lo_bin >= pred_pow.shape[1]:
                results[key] = float("nan")
                results[log_key] = float("nan")
                continue

            pred_band = torch.sum(pred_pow[:, lo_bin:hi_bin, :], dim=1)
            ref_band = torch.sum(ref_pow[:, lo_bin:hi_bin, :], dim=1)

            pred_edr = torch.cumsum(pred_band.flip([-1]), dim=-1).flip([-1])
            ref_edr = torch.cumsum(ref_band.flip([-1]), dim=-1).flip([-1])

            pred_edr = pred_edr / (pred_edr[:, :1] + 1e-10)
            ref_edr = ref_edr / (ref_edr[:, :1] + 1e-10)

            mse = torch.mean((ref_edr - pred_edr) ** 2)
            results[key] = mse.item()
            results[log_key] = float(-torch.log10(mse + 1e-10).item())

    return results
