"""Standard preprocessing pipeline for RIRBench.

All RIRs are processed to a common format before evaluation or training:
1. Convert to mono (first channel for stereo, W-component for B-format)
2. Remove initial delay via energy-based onset detection
3. Resample to 48 kHz using polyphase method
4. Truncate to 1 second with 50 ms linear fadeout
5. Normalize peak amplitude to 0.9
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


TARGET_SR = 48_000
TARGET_DURATION = 1.0          # seconds
FADEOUT_DURATION = 0.05        # seconds
PEAK_NORM = 0.9
ONSET_ENERGY_THRESHOLD = 1e-4  # relative to peak energy


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio and convert to mono float32.

    For stereo: takes channel 0.
    For B-format (4+ channels): takes channel 0 (W component).
    """
    data, sr = sf.read(str(path), always_2d=True)
    return data[:, 0].astype(np.float32), sr


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Polyphase resample to target sample rate."""
    if orig_sr == target_sr:
        return audio
    g = gcd(target_sr, orig_sr)
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def remove_onset_delay(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Trim leading silence before the RIR onset.

    Onset is defined as the first sample where the squared value exceeds
    ``ONSET_ENERGY_THRESHOLD`` of the peak squared value.
    """
    sq = audio**2
    threshold = ONSET_ENERGY_THRESHOLD * sq.max()
    onsets = np.where(sq > threshold)[0]
    if len(onsets) == 0:
        return audio
    return audio[onsets[0]:]


def truncate_with_fadeout(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    duration: float = TARGET_DURATION,
    fadeout: float = FADEOUT_DURATION,
) -> np.ndarray:
    """Truncate to `duration` seconds and apply a linear fadeout at the end."""
    n_target = int(sr * duration)
    if len(audio) < n_target:
        audio = np.pad(audio, (0, n_target - len(audio)))
    else:
        audio = audio[:n_target]

    n_fade = int(sr * fadeout)
    fade = np.linspace(1.0, 0.0, n_fade, dtype=np.float32)
    audio[-n_fade:] *= fade
    return audio


def peak_normalize(audio: np.ndarray, peak: float = PEAK_NORM) -> np.ndarray:
    """Normalize peak amplitude to `peak`."""
    max_val = np.max(np.abs(audio))
    if max_val < 1e-8:
        return audio
    return (audio / max_val * peak).astype(np.float32)


def preprocess(
    path: str | Path,
    target_sr: int = TARGET_SR,
    duration: float = TARGET_DURATION,
    fadeout: float = FADEOUT_DURATION,
    peak: float = PEAK_NORM,
) -> np.ndarray:
    """Full preprocessing pipeline for a single RIR file.

    Args:
        path:      path to the audio file
        target_sr: target sample rate in Hz
        duration:  output duration in seconds
        fadeout:   fadeout length in seconds
        peak:      target peak amplitude

    Returns:
        preprocessed mono float32 array of length ``int(target_sr * duration)``
    """
    audio, sr = load_mono(path)
    audio = resample(audio, sr, target_sr)
    audio = remove_onset_delay(audio, target_sr)
    audio = truncate_with_fadeout(audio, target_sr, duration, fadeout)
    audio = peak_normalize(audio, peak)
    return audio


def preprocess_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    **kwargs,
) -> list[Path]:
    """Preprocess all .wav and .flac files in `input_dir` and save to `output_dir`.

    Returns list of output paths.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.rglob("*.wav")) + sorted(input_dir.rglob("*.flac"))
    outputs = []

    for f in files:
        try:
            audio = preprocess(f, **kwargs)
            out = output_dir / (f.stem + ".wav")
            sf.write(str(out), audio, kwargs.get("target_sr", TARGET_SR))
            outputs.append(out)
        except Exception as exc:
            print(f"Warning: could not preprocess {f}: {exc}")

    return outputs
