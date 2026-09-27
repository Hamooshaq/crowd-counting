# Crowd Counting with CSRNet

This repository implements a reproducible crowd-counting pipeline for the ShanghaiTech dataset. It refactors the original notebook workflow into reusable modules with:

- a clean CSRNet baseline for fair reproduction
- an advanced CSRNet variant for high-density scenes
- training, evaluation, and comparison scripts
- evaluation support for MAE, RMSE, PSNR, SSIM, and density-level analysis

## Problem

Crowd counting estimates the number of people in an image by predicting a density map whose integral is the predicted count. Dense scenes are difficult because of scale variation, occlusion, and perspective distortion.

This codebase is structured to support:

- a clearly defined technical contribution
- fair baseline vs improved-model comparison
- reproducible experiments
- evidence that the design choice helps on dense crowd scenes

The original notebook had strong baseline ingredients, but it mixed data generation, training, inference, and export logic in one place. This version separates those concerns and adds an advanced variant that is appropriate for a Stage 2 submission.

## Method and architecture

The project now has two model tracks:

- `CSRNetBaseline`: faithful baseline reproduction
- `AdvancedCSRNet`: multi-scale context aggregation + channel/spatial attention + count-consistency loss support

The advanced path targets the exact weakness highlighted in the report: underestimation on high-density scenes with heavy occlusion and scale variation.

Important: standard CSRNet already uses a dilated backend. That means a Stage 2 upgrade cannot be defended as "we only added dilated convolution." The advanced model in this repo goes beyond that with multi-scale context aggregation, attention, and count-consistency supervision.

## Folder Structure

```text
CrowdDetection/
├── configs/
│   ├── advanced_shanghaitech_a.json
│   └── baseline_shanghaitech_a.json
├── docs/
│   └── stage2_alignment.md
├── scripts/
│   ├── compare_runs.py
│   ├── evaluate.py
│   ├── generate_density_maps.py
│   └── train.py
├── src/
│   └── crowd_detection/
│       ├── config.py
│       ├── losses.py
│       ├── metrics.py
│       ├── data/
│       │   ├── density_maps.py
│       │   └── dataset.py
│       ├── engine/
│       │   └── trainer.py
│       ├── models/
│       │   ├── blocks.py
│       │   └── csrnet.py
│       └── utils/
│           ├── checkpointing.py
│           ├── io.py
│           └── reproducibility.py
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dataset Layout

Expected ShanghaiTech layout:

```text
data/ShanghaiTech/
├── part_A/
│   ├── train_data/
│   │   ├── images/
│   │   ├── ground-truth/
│   │   └── density_maps/
│   └── test_data/
│       ├── images/
│       ├── ground-truth/
│       └── density_maps/
└── part_B/
    ├── train_data/
    └── test_data/
```

If `density_maps/` does not exist, generate them first:

```bash
python scripts/generate_density_maps.py --root ./data/ShanghaiTech --part A --splits train test
```

## Training

Baseline:

```bash
python scripts/train.py --config configs/baseline_shanghaitech_a.json
```

Advanced model:

```bash
python scripts/train.py --config configs/advanced_shanghaitech_a.json
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/advanced_shanghaitech_a.json --checkpoint outputs/advanced_shanghaitech_a/checkpoints/best.pt
```

## Baseline vs Advanced Comparison

```bash
python scripts/compare_runs.py \
  --baseline outputs/baseline_shanghaitech_a \
  --advanced outputs/advanced_shanghaitech_a
```

This generates a Markdown-ready comparison table for the report.

## Evaluation outputs

- clean implementation
- advanced architecture
- reproducible metrics export
- per-sample error analysis
- density-level breakdown
- Markdown comparison tables

The repository does not publish trained weights, generated density maps, or headline scores. Run the documented evaluation against your own legally obtained ShanghaiTech copy and checkpoint to reproduce metrics.

## Dataset

ShanghaiTech is not redistributed here. Obtain it from an authorized source and place it under the layout shown below. The entire `data/` directory is ignored by Git.

## Limitations

- Results depend on the local dataset copy, preprocessing, seed, and hardware.
- The advanced model is an experimental extension and is not claimed to outperform the baseline without a completed, controlled run.
- Density-map quality metrics complement count error but should not be interpreted as direct measures of real-world deployment quality.
- No production inference service or live deployment is included.

## Project status

Research prototype. The modular training and evaluation code is available; datasets, checkpoints, and generated experiment outputs remain local.

## Design rationale

Use the baseline to establish the reference point, then argue that dense scenes fail because of:

- large scale variation
- severe head overlap
- weak long-range context in crowded regions

Then show that the advanced model helps because it adds:

- multi-scale context aggregation
- attention-driven feature recalibration
- count-consistency supervision
