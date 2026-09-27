# Stage 2 Alignment

## Problem Framing

The original report identifies the key weakness correctly: the model underestimates very dense crowds because heavy occlusion and scale variation make local-only features unreliable.

This refactor turns that observation into a Stage 2 technical claim:

> We extend CSRNet with multi-scale context aggregation and attention-based feature recalibration, then evaluate whether the modified architecture reduces high-density undercounting while preserving overall density-map quality.

## What Changed from the Original Notebook

- baseline implementation separated from data generation and inference utilities
- consistent train/eval protocol
- advanced architecture added instead of vague "improvement"
- count-consistency loss added for dense-scene counting stability
- exported comparison metrics and sample-level analysis for the report

## Stage 2 Mapping

### 1. Contribution

The advanced model is a real architectural contribution, not only hyperparameter tuning:

- multi-scale dilated context block
- channel and spatial attention
- optional count-consistency regularization

This matters because plain CSRNet already contains a dilated backend. A Stage 2 claim needs to show a stronger modification than just "adding dilation."

### 2. Related Work Positioning

The baseline remains CSRNet.

The Stage 2 story should argue that:

- CSRNet is strong but still struggles with ultra-dense scenes
- dense crowd images need broader context than a standard backend provides
- attention can suppress noisy background activations and emphasize informative crowd regions

### 3. Experiments

Minimum experiment set:

- baseline vs advanced on the same dataset split
- MAE / RMSE / PSNR / SSIM
- error breakdown across low, medium, and high density samples
- 4 to 8 qualitative density-map examples

Strong Stage 2 extension:

- repeat on ShanghaiTech Part B or JHU-CROWD++ if available
- ablation:
  - baseline
  - baseline + multi-scale context
  - baseline + multi-scale context + attention
  - baseline + multi-scale context + attention + count loss

## Report Corrections You Should Make

- do not claim higher MAE is "better" than baseline
- keep dataset split identical between code, tables, and prose
- explain why the advanced model helps high-density scenes specifically
- show comparison tables, not only standalone metrics
- explicitly separate baseline reproduction from proposed method

## Suggested Section Titles for the Updated Report

1. Introduction
2. Related Work and Baseline Positioning
3. Proposed Advanced CSRNet
4. Experimental Setup
5. Results and Ablation
6. Error Analysis in High-Density Scenes
7. Conclusion and Future Work
