# STFT-based loss modules.
# Adapted from Steinmetz et al. (2021) / Hayashi (2019), MIT License.

import torch
import torch.nn.functional as F


def _stft_mag(x: torch.Tensor, fft_size: int, hop_size: int, win_length: int, window: torch.Tensor) -> torch.Tensor:
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    return torch.sqrt(torch.clamp(x_stft.real**2 + x_stft.imag**2, min=1e-8))


class _STFTLoss(torch.nn.Module):
    def __init__(self, fft_size: int, hop_size: int, win_length: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_mag = _stft_mag(x, self.fft_size, self.hop_size, self.win_length, self.window)
        y_mag = _stft_mag(y, self.fft_size, self.hop_size, self.win_length, self.window)
        sc = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
        mag = F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        return sc, mag


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi-resolution STFT loss across four window sizes (64–8192 samples).

    Used as both training loss and evaluation metric.
    """

    def __init__(
        self,
        fft_sizes=(64, 512, 2048, 8192),
        hop_sizes=(32, 256, 1024, 4096),
        win_lengths=(64, 512, 2048, 8192),
        sc_weight: float = 1.0,
        mag_weight: float = 1.0,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight
        self.losses = torch.nn.ModuleList(
            [_STFTLoss(fs, hs, wl) for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Args:
            x: predicted signal (B, T) or (B, 1, T)
            y: reference signal  (B, T) or (B, 1, T)
        Returns:
            dict with keys ``total``, ``sc_loss``, ``mag_loss``
        """
        x = x.squeeze(1)
        y = y.squeeze(1)
        sc_loss = mag_loss = 0.0
        for f in self.losses:
            sc, mag = f(x, y)
            sc_loss = sc_loss + sc
            mag_loss = mag_loss + mag
        n = len(self.losses)
        return {
            "total": (sc_loss * self.sc_weight + mag_loss * self.mag_weight) / n,
            "sc_loss": sc_loss / n,
            "mag_loss": mag_loss / n,
        }


def mstft_loss(predicted: torch.Tensor, reference: torch.Tensor) -> float:
    """Convenience wrapper — returns scalar total MSTFT loss."""
    fn = MultiResolutionSTFTLoss()
    with torch.no_grad():
        return fn(predicted, reference)["total"].item()
