# MOSAIC

Code and artifacts for MOSAIC (Multi-view Output and State Aggregation for
Identifying Confident Hallucinations) and the HalluMatch benchmark.

HalluMatch contains 1,563 confidence-matched hallucination/correct pairs
across five checkpoints. MOSAIC combines equivalent-question behavior,
layerwise internal-state trajectories, and sampling persistence. It does not
use a reference answer at inference time.

## Held-out test

Macro metrics weight the five checkpoints equally. ΔR is the recall difference
versus the internal-state trajectory.

| Method | M7 (81) | Q8 (65) | L8 (63) | Q14 (61) | MN (57) | Macro R | FPR | Acc. | ΔR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean hidden probe | 34/7 | 14/6 | 27/6 | 13/5 | 25/5 | 34.31 | 8.87 | 62.72 | −6.06 |
| Capacity-matched MLP | 37/6 | 15/7 | 29/6 | 14/5 | 27/5 | 37.02 | 8.93 | 64.04 | −3.35 |
| Behavior expert | 30/6 | 18/8 | 24/7 | 15/6 | 25/5 | 34.26 | 9.89 | 62.18 | −6.11 |
| Persistence expert | 18/5 | 13/6 | 17/5 | 12/5 | 15/4 | 23.04 | 7.71 | 57.66 | −17.33 |
| Internal-state trajectory | 40/6 | 17/7 | 31/6 | 16/5 | 29/5 | 40.37 | 8.93 | 65.72 | — |
| Flat concatenation | 42/7 | 20/9 | 33/7 | 17/6 | 32/5 | 43.80 | 10.44 | 66.68 | +3.43 |
| **MOSAIC** | **45/5** | **22/10** | **36/6** | **19/5** | **36/5** | **48.17** | **9.61** | **69.28** | **+7.80** |

The 7.80-point recall gain has a full-pipeline hierarchical-bootstrap 95% CI
of 4.28 to 11.36 (p = 0.002). The same sign of improvement appears at the 5%
and 15% transferred FPR caps and in test partial AUC (0.061 vs 0.052).

Full tables, validation ablations, cost, and controls are in
[docs/RESULTS.md](docs/RESULTS.md).

## Layout

- `src/`: feature extraction, baselines, MOSAIC, intervention, and mitigation
- `tests/`: unit tests
- `configs/model_provenance.json`: model checkpoints
- `rerun/`: rerun of the paper configuration, with test predictions, thresholds, external detectors and the full-pipeline bootstrap (see `rerun/README.md`)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU regeneration requires the official checkpoints in
`configs/model_provenance.json`.

## Verify

```bash
PYTHONPATH=src pytest -q
```

## Data

Questions come from PopQA, TriviaQA, and SimpleQA. Model outputs come from the
official Mistral, Qwen, and Llama checkpoints listed in the provenance files.
Upstream dataset and model licenses continue to apply.
