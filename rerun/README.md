# Rerun of the paper configuration

This directory contains the code, configuration and outputs of the rerun of MOSAIC in the paper configuration, the code of the full-pipeline bootstrap, and the external detectors on the held-out test split.

## Layout

| Path | Content |
|---|---|
| `configs/paper_configuration.json` | Paper configuration of MOSAIC and the configuration of the trajectory baseline (paper configuration without candidate features derived from paraphrases) |
| `src/mosaic_tune.py` | Fitting code: layerwise probes, experts and cross-fitted gate. It imports the original pipeline module `evaluate_one_confidence_mosaic_v1_nested_public.py` (the same file as in `src/` of this repository), expected under `~/hallu_paper/mosaic_rebuttal/work/experiments/hallucination/` |
| `src/final_test.py` | One-shot test evaluation: fit on development, select thresholds on validation at FPR ≤ 10%, apply unchanged to test |
| `src/full_bootstrap.py` | Full-pipeline hierarchical bootstrap of MOSAIC versus the trajectory baseline |
| `src/full_bootstrap_summary.py` | Point estimates, percentile intervals and one-sided p values from the bootstrap output |
| `results/test_predictions_paper_configuration.json` | Per checkpoint: labels, fact groups, sample ids, validation thresholds, test metrics and per-example test scores |
| `results/external_baselines_test.json` | Per-checkpoint TP/FP, recall, realized FPR and accuracy of the external detectors on the test split |

## Bootstrap procedure

Replicate 0 is the point estimate on the original rows. In every other replicate, fact groups are resampled with replacement separately within the development, validation and test splits of each checkpoint. Probes, experts and gate are refitted on the resampled development rows, the FPR ≤ 10% threshold is reselected on the resampled validation rows, and both detectors are scored on the same resampled test rows. The macro difference is the equal-weight mean over the five checkpoints. `full_bootstrap_summary.py` uses only replicates in which all five checkpoints are complete.

## Running

The scripts expect the cached hidden-state features and frozen auxiliary sources under `~/hallu_paper/mosaic_rebuttal/` (see the paths at the top of each script).

```
python src/final_test.py
python src/full_bootstrap.py N_REPLICATES WORKERS
python src/full_bootstrap_summary.py ~/hallu_paper/mosaic_rebuttal/tune/full_bootstrap.jsonl
```
