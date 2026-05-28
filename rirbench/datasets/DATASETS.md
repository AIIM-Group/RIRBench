# RIRBench Dataset Catalog

## RIR Datasets (11,500 RIRs from 295 rooms)

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
| [C4DM RIR](http://isophonics.net/content/room-impulse-response-data-set) | 468 mono and Ambisonics B-format RIRs from three large University of London rooms | 96 kHz, 32-bit, mono / Ambisonics B | 468 / 468 |
| [AIR](http://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/) | 344 binaural RIRs measured with a dummy head in 5 environments, including a church | 48 kHz, 24-bit, mono / multichannel | 344 / 344 |
| [MYRiAD V2](https://zenodo.org/records/7389996) | 1,214 RIRs recorded in two rooms with dummy head, behind-the-ear microphones, 5 external microphones, and two circular 12-microphone arrays | 44.1 kHz, 24-bit, mono / multichannel | 1,214 / 1,214 |

## Speech Dataset

| Dataset | Description | Download |
|---|---|---|
| [VCTK Corpus](https://www.kaggle.com/datasets/pratt3000/vctk-corpus) | 44,200+ recordings from 110 English speakers with diverse accents, recorded at 48 kHz. Only `mic1` files used. | Kaggle |

## Data Split

All datasets use a **60/20/20 train/validation/test split**, stratified per source dataset (RIR) and per speaker (VCTK) to maintain proportional representation.

- RIR split script: `scripts/split_dataset.py`
- VCTK split script: `scripts/split_vctk.py`
- Arni sampling script: `scripts/prepare_arni.py`

## Arni Sampling Strategy

Arni contains 132,037 RIRs across 5,342 panel configurations. To select a representative subset of 3,507 RIRs:

1. Panel configurations are binned into 5 groups by number of reflective panels (acoustic condition diversity)
2. For each bin, microphones 1 and 5 (spatial edges) are sampled, preferring sweep 3 (middle sweep)
3. A final random draw reduces to the target count if needed

This ensures coverage across the full range of reverberation times present in the dataset.
