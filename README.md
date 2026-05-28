# RIRBench

> In Tune with the Room: Advancing Blind RIR Generation Through a Comprehensive Evaluation Framework

[![Paper](https://img.shields.io/badge/Paper-EUSIPCO%202026-blue)](#citation)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

RIRBench is an evaluation framework for blind Room Impulse Response (RIR) estimation from reverberant speech. It addresses the fragmented landscape of RIR generation assessment by providing a unified, multi-metric approach, combining established acoustic parameters (T60, DRR) with advanced spectral-temporal metrics (MSTFT, EDR) and a novel peak similarity metric to capture complementary aspects of RIR quality.

## Motivation

Existing RIR generation evaluation suffers from:

- **Fragmented metrics**: different papers use different metric combinations, making cross-comparison difficult
- **Unavailable codebases**: most published models are not publicly released
- **Limited datasets**: single-dataset evaluation hides generalization issues
- **Hidden trade-offs**: single metrics miss complementary strengths and weaknesses

RIRBench addresses these by offering a standardized framework with 10 complementary metrics, 11,500 RIRs from 295 rooms across 11 public datasets, and visual comparison through radar plots.

## Features

- **10 evaluation metrics** spanning temporal, spectral, and perceptual aspects
- **11,500 RIRs from 11 public datasets** with consistent preprocessing
- **Radar plot visualization** for multi-metric comparison
- **Reference results** for FiNS and several modified variants

## Metrics

| Metric | Aspect | Description |
|---|---|---|
| T60 (Bias, MSE, ρ) | Temporal | Reverberation time estimation accuracy |
| DRR (Bias, MSE, ρ) | Energy balance | Direct-to-reverberant energy ratio |
| MSTFT Loss | Spectral | Multi-resolution STFT loss across four window sizes (64–8192 samples) |
| EDR Loss | Spectral-temporal | Energy decay across nine octave bands (16 Hz–4 kHz) |
| Peak Similarity (W, U) | Spectral | Magnitude spectrum peak accuracy, weighted and unweighted variants |

## Datasets

### Used in This Framework

The combined evaluation set comprises 11,500 RIRs from 295 rooms across 11 publicly available datasets. The following table provides details on each:

| Dataset | Description | Format | RIRs Used |
|---|---|---|---|
| [R3VIVAL](https://github.com/facebookresearch/R3VIVAL) | 272 RIRs: 30 source positions in 3 circles (1.5 m, 2 m, 3 m), 4 source positions in room corners, 1 receiver position (7-microphone array), 8 acoustic panel configurations | 48 kHz, 24-bit, 7-channel | 272 / 272 |
| [Arni](https://zenodo.org/record/6985104) | 132,037 RIRs measured with 5,342 configurations of 55 acoustic panels in the variable acoustics lab Arni at Aalto University | 44.1 kHz, 32-bit, mono | 3,507 / 132,037 |
| [Palimpsest](https://research.kent.ac.uk/sonic-palimpsest/impulse-responses/) | RIRs from historic spaces of the Historic Dockyard | 48 kHz, 24-bit, stereo | 44 / 44 |
| [BBC Maida Vale](https://www.mdpi.com/2624-599X/4/3/47) | Acoustic measurements of the BBC Maida Vale recording studios with RIRs in third-order Ambisonics format and KEMAR dummy head measurements | 48 kHz, 24-bit, mono / stereo / 32-channel | 1,746 / 1,746 |
| [Motus](https://doi.org/10.5281/zenodo.4923187) | 3,320 higher-order Ambisonics RIRs measured with an Eigenmike em32, 4 loudspeaker positions, and 830 furniture configurations in a single room at Aalto University | 48 kHz, 24-bit, 32-channel | 3,320 / 3,320 |
| [Detmold-SRIR](https://zenodo.org/records/4116247) | 600 multichannel RIRs from Detmold Konzerthaus (medium concert hall), Brahmssaal (chamber music room), and Detmold Sommertheater (theater) | 48 kHz, 24-bit, 6-channel | 301 / 600 |
| [MIT IR Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) | 271 mono-channel RIRs, each recorded in a distinct space | 32 kHz, 24-bit, mono | 270 / 270 |
| [ACE Challenge](http://www.ee.ic.ac.uk/naylor/ACEweb/index.html) | 1, 2, 3, 5, 8, 32-channel RIRs recorded across 7 rooms | 48 kHz, 16-bit, multi-channel | 14 / 14 |
| [C4DM RIR](http://isophonics.net/content/room-impulse-response-data-set) | 468 mono and Ambisonics B-format RIRs from three large University of London rooms (Great Hall, Octagon, seminar room), measured via sine sweep with Genelec 8250A speaker and omnidirectional / B-format microphones | 96 kHz, 32-bit, mono / Ambisonics B | 468 / 468 |
| [AIR](http://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/) | 344 binaural RIRs measured with a dummy head in 5 environments, including a church | 48 kHz, 24-bit, mono / multichannel | 344 / 344 |
| [MYRiAD V2](https://zenodo.org/records/7389996) | 1,214 RIRs recorded in two rooms (SONORA Audio Laboratory, Alamire Interactive Laboratory) with dummy head, behind-the-ear microphones, 5 external microphones, and two circular 12-microphone arrays | 44.1 kHz, 24-bit, mono / multichannel | 1,214 / 1,214 |

### Preprocessing

All RIRs are processed consistently:

- Amplitude normalized to 0.9 peak
- Initial delays removed via energy-based onset detection
- Resampled to 48 kHz using polyphase method
- Truncated to 1 second with 50 ms linear fadeout
- Converted to mono format (first channel for stereo, W-component for B-format)

Data is split 60/20/20 (train/validation/test) with proportional representation across all source datasets.

### Speech Data

Reverberant training and evaluation speech is generated by convolving RIRs with clean speech from the VCTK corpus. Only `mic1` recordings are used; the same 60/20/20 speaker-stratified split is applied.

| Dataset | Description | Download |
|---|---|---|
| [VCTK Corpus](https://www.kaggle.com/datasets/pratt3000/vctk-corpus) | 44,200+ recordings from 110 English speakers with diverse accents, recorded at 48 kHz | Kaggle |

### Other Available RIR Datasets

For convenience, the following table lists additional publicly available RIR datasets that were not included in the current evaluation but may be useful for related research:

| Dataset | Description | Year |
|---|---|---|
| [GTU-RIR](https://github.com/mehmetpekmezci/gtu-rir) | 15,000+ RIRs collected using a semi-automated sound recording system | 2024 |
| [MIRACLE](https://depositonce.tu-berlin.de/items/b079fd1c-999f-42cb-afd2-bcd34de6180b) | 856,128 single-channel impulse responses acquired with a planar 64-channel microphone array across four scenarios | 2023 |
| [SoundCam](https://sites.google.com/view/soundcam) | 5,000 10-channel real-world RIRs and 2,000 10-channel music recordings across three rooms with humans in different positions | 2023 |
| [BRUDEX](https://zenodo.org/records/8340195) | 18-channel binaural RIRs with 36 microphone positions and 12 source positions | 2023 |
| [dEchorate](https://zenodo.org/record/5562386) | 1,800 annotated RIRs from 6 arrays of 5 microphones, 6 sources, and 11 acoustic conditions | 2021 |
| [MeshRIR](https://sh01k.github.io/MeshRIR/) | 4,410 mono RIRs recorded on dense grids with accurate coordinates in a moderately reverberant room | 2021 |
| [BUT Reverb Database](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database) | 1,300+ mono channel RIRs recorded in 8 rooms | 2019 |
| [Multichannel Impulse Response Database](https://www.eng.biu.ac.il/gannot/downloads/) | 234 8-channel RIRs with 3 reverberation levels and different array spacings | 2014 |
| [REVERB Challenge](https://reverb2014.dereverberation.com/) | 24 8-channel RIRs from small, medium, and large rooms | 2013 |
| [RWCP Sound Scene Database](http://research.nii.ac.jp/src/en/RWCP-SSD.html) | 143 multi-channel RIRs recorded in 14 rooms | 2000 |

## Installation

Tested with Python 3.9.21.

```bash
git clone https://github.com/AIIM-Group/RIRBench
cd rir-bench
pip install -r requirements.txt
```

## Quick Start

```python
from rirbench import evaluate, radar_plot

# Evaluate your model's predicted RIRs
results = evaluate(
    predicted_rirs="path/to/predictions/",
    reference_rirs="path/to/ground_truth/",
    metrics=["t60", "drr", "mstft", "edr", "peak_sim"]
)

# Print summary
print(results.summary())

# Save summary.md, summary.csv (aggregated stats), and results.csv (per-sample data)
results.save("my_model_results/", model_name="MyModel")

# Visualize as radar plot
radar_plot(results, save_path="my_model_radar.pdf")
```

For side-by-side comparison of multiple models, pass a dict to `radar_plot`:

```python
radar_plot(
    {"FiNS": results_fins, "FiNS+EDR": results_edr},
    save_path="comparison.pdf"
)
```

## FiNS Variants

The paper introduces several modifications to the FiNS architecture by [Steinmetz et al. (2021)](https://arxiv.org/abs/2107.07503). Since this repository focuses on the evaluation framework, the FiNS variants themselves are documented here with the key code changes, rather than as redistributed implementations. The modifications described below were developed on top of the open-source FiNS reimplementation by Kyungyun Lee, available at [https://github.com/kyungyunlee/fins](https://github.com/kyungyunlee/fins).

### 1. Adapted Octave Filterbank

Standard FiNS initializes its learnable filterbank with uniform octave spacing. The Adapted Octave variant replaces this with data-driven center frequencies derived from a statistical analysis of 11,500 real RIRs (2048-sample FFT windows with 50% overlap, 3 dB prominence threshold for peak detection). The resulting initialization reduces initial spectral overlap by 12% while preserving the adaptive capacity of the learnable architecture.

```python
import numpy as np
import scipy.signal

def get_calculated_filters():
    """10 bandpass filters with data-driven center frequencies (shape: (10, 1, 1023)).

    Frequency bounds derived from statistical peak analysis of 11,500 real RIRs.
    Replaces the standard octave-spaced initialization in FiNS.
    """
    f_bounds = [
        [25.0, 48.8], [48.8, 95.2], [95.2, 185.7],
        [185.7, 362.4], [362.4, 707.1], [707.1, 1379.7],
        [1379.7, 2692.2], [2692.2, 5253.1], [5253.1, 10249.9],
        [10249.9, 20000.0],
    ]
    firs = []
    for low, high in f_bounds:
        fir = scipy.signal.firwin(
            1023, [low, high],
            pass_zero='bandpass', window='hamming', fs=48000,
        )
        firs.append(fir)
    return np.expand_dims(np.array(firs), 1)

# In FilteredNoiseShaper.__init__, replace:
#   octave_filters = get_octave_filters()
# with:
octave_filters = get_calculated_filters()
self.filter.weight.data = torch.FloatTensor(octave_filters)
```

See Section III.C of the paper for the full derivation.

### 2. Third-Octave Filterbank

The Third-Octave variant replaces the 10-filter octave bank with a 30-band third-octave filterbank following IEC 61260, providing three times higher spectral resolution.

```python
import numpy as np
import scipy.signal

def get_third_octave_filters():
    """30 one-third octave bandpass filters (shape: (30, 1, 1023)).

    IEC 61260 center frequencies from 25 Hz to 20 kHz.
    Replaces the 10-band octave filterbank in FiNS; set num_filters=30 in config.
    """
    center_freqs = [
        25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200,
        250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
        2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000,
    ]
    firs = []
    for fc in center_freqs:
        f_low = fc / 2 ** (1 / 6)
        f_high = fc * 2 ** (1 / 6)
        fir = scipy.signal.firwin(
            1023, [f_low, f_high],
            pass_zero='bandpass', window='hamming', fs=48000,
        )
        firs.append(fir)
    return np.expand_dims(np.array(firs), 1)

# In FilteredNoiseShaper.__init__ (with num_filters=30 in config):
octave_filters = get_third_octave_filters()
self.filter.weight.data = torch.FloatTensor(octave_filters)
```

### 3. MSTFT+EDR Combined Loss

The original FiNS loss uses MSTFT only (spectral log-magnitude plus spectral convergence). The MSTFT+EDR variants replace spectral convergence with an Energy Decay Relief (EDR) component, computed across nine octave bands (16 Hz–4 kHz). Three FFT sizes for the EDR computation are evaluated: 1k (1024), 2k (2048), and 4k (4096) samples.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModifiedSTFTLoss(nn.Module):
    """MSTFT log-magnitude loss combined with EDR loss across nine octave bands.

    Replaces the spectral-convergence term in the original FiNS MSTFT loss.
    edr_fft_size controls EDR resolution: 1024 (1k), 2048 (2k), or 4096 (4k).
    """

    EDR_FREQ_BOUNDS = [
        [22.3, 44.5], [44.5, 88.4], [88.4, 176.8],
        [176.8, 353.6], [353.6, 707.1], [707.1, 1414.2],
        [1414.2, 2828.4], [2828.4, 5656.8], [5656.8, 11313.6],
    ]

    def __init__(
        self,
        fft_sizes=(64, 512, 2048, 8192),
        hop_sizes=(32, 256, 1024, 4096),
        win_lengths=(64, 512, 2048, 8192),
        window="hann_window",
        mag_weight=1.0,
        edr_weight=1.0,
        edr_fft_size=1024,   # 1024 → 1k, 2048 → 2k, 4096 → 4k
        sr=48000,
    ):
        super().__init__()
        self.mag_weight = mag_weight
        self.edr_weight = edr_weight
        self.edr_fft_size = edr_fft_size
        self.edr_hop_size = edr_fft_size // 2

        self.stft_losses = nn.ModuleList([
            STFTLoss(fs, hs, wl, window)
            for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)
        ])
        self.register_buffer('edr_window', getattr(torch, window)(edr_fft_size))

        freqs = torch.linspace(0, sr / 2, edr_fft_size // 2 + 1)
        masks = [
            (freqs >= lo) & (freqs < hi)
            for lo, hi in self.EDR_FREQ_BOUNDS
            if ((freqs >= lo) & (freqs < hi)).any()
        ]
        self.register_buffer('edr_masks', torch.stack(masks))

    def _edr(self, mag_sq):
        masks = self.edr_masks.unsqueeze(0).unsqueeze(-1)
        band_energy = (mag_sq.unsqueeze(1) * masks).sum(dim=2)
        edr = torch.cumsum(band_energy.flip(-1), dim=-1).flip(-1)
        return edr / (edr[..., :1] + 1e-10)

    def forward(self, x, y):
        x, y = x.squeeze(1), y.squeeze(1)

        mag_loss = sum(f(x, y)[1] for f in self.stft_losses) / len(self.stft_losses)

        kw = dict(n_fft=self.edr_fft_size, hop_length=self.edr_hop_size,
                  win_length=self.edr_fft_size, window=self.edr_window,
                  return_complex=True)
        x_edr = self._edr(torch.abs(torch.stft(x, **kw)) ** 2)
        y_edr = self._edr(torch.abs(torch.stft(y, **kw)) ** 2)
        edr_loss = torch.mean((y_edr - x_edr) ** 2) * 200

        return {
            "total": self.mag_weight * mag_loss + self.edr_weight * edr_loss,
            "mag_loss": mag_loss,
            "edr_loss": edr_loss,
        }
```

The EDR for a single RIR is computed as:

```
EDR(t, b_k) = sum_{tau=t}^{T} |H(tau, k)|^2
```

where `H(tau, k)` is the k-th frequency bin of the STFT at time `tau`. The loss is the MSE between the predicted and ground-truth EDR over all bands. See Section III.D of the paper for details.

### Implementation Notes

The code snippets above are illustrative and not drop-in replacements. To implement the variants, integrate the relevant change into a working FiNS implementation. The paper provides all necessary numerical values (filter center frequencies, FFT sizes, weighting coefficients, training hyperparameters).

## Project Structure

```
rir-bench/
├── rirbench/                # Core framework package
│   ├── metrics/             # Metric implementations
│   ├── datasets/            # Dataset loaders, preprocessing, catalog
│   │   └── DATASETS.md      # Dataset table and Arni sampling notes
│   ├── visualization/       # Radar plots and figures
│   └── evaluate.py          # Main evaluation pipeline
├── scripts/
│   ├── split_dataset.py     # 60/20/20 split for RIR datasets
│   ├── split_vctk.py        # 60/20/20 split for VCTK speech corpus
│   └── prepare_arni.py      # Sample 3,507 RIRs from full Arni dataset
├── data/                    # Dataset metadata (audio not included)
├── results/                 # Pre-computed reference results
├── setup.py
└── requirements.txt
```

## Citation

If you use RIRBench in your research, please cite:

```bibtex
@inproceedings{stauss2026intune,
  title     = {In Tune with the Room: Advancing Blind RIR Generation
               Through a Comprehensive Evaluation Framework},
  author    = {Stau{\ss}, Sebastian and Watanabe, Hiroshi and Schmid, Thomas},
  booktitle = {Proceedings of the European Signal Processing Conference (EUSIPCO)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

Note that individual datasets retain their original licenses; please consult each dataset's documentation for usage terms.

## Contact

Sebastian Stauß — [sebastian.stauss@medizin.uni-halle.de](mailto:sebastian.stauss@medizin.uni-halle.de)

For questions, bug reports, or contributions, please open an issue on GitHub.

## Acknowledgments

This framework builds on the FiNS architecture introduced by Steinmetz et al. (2021), with code-level modifications developed on top of the open-source FiNS reimplementation by Kyungyun Lee ([https://github.com/kyungyunlee/fins](https://github.com/kyungyunlee/fins)). Full references are provided in the paper.
