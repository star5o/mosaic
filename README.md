
# One Confidence / MOSAIC

Compact research release for confidence-matched general confident hallucination
detection across five language models.

## Scope

The positive class is HCE (high-confidence error); HCC is the matched negative
class. The public validation split is used for model and threshold selection.
The 327-pair frozen cohort is retrospective because it was previously opened
for baseline and MOSAIC-v0 evaluation. The five 4,000-row fresh generations
were not correctness-reviewed or detector-scored and are therefore not test
results.

The empirical conclusion is deliberately limited:

- MOSAIC-v1 frozen macro Accuracy / Recall / FPR: 69.28 / 48.17 / 9.61%.
- Recall gain over MOSAIC-v0: +4.83 points, 95% CI [-0.57, +10.35].
- No tested cell passed both patch and erasure causal controls.
- Conditional routing had net correction -30; uniform likelihood had +5.

See [docs/RESULTS.md](docs/RESULTS.md) for the full result boundary.

## Layout

- `src/`: feature extraction, baselines, MOSAIC, mechanism, and mitigation code.
- `tests/`: focused unit and integrity tests.
- `configs/`: frozen protocol, feature schema, thresholds, and model revisions.
- `data/`: per-sample scores, OOF metadata, baseline outputs, and reviewed cases.
- `results/`: canonical paper tables, figures, bootstrap intervals, and audits.
- `external_artifacts.json`: SHA256 records for large tensors/checkpoints omitted
  from Git.

No model weights, Hugging Face tokens, remote credentials, hidden-state tensors,
joblib checkpoints, or unreviewed fresh generations are included.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU regeneration additionally requires model access for the official revisions
listed in `configs/model_provenance.json`.

## Verify

```bash
python tools/verify_release.py
PYTHONPATH=src pytest -q
```

The verifier checks every released file against `release_manifest.json` and
recomputes the headline frozen metrics.

## Data and licensing

Questions originate from PopQA and TriviaQA. Model outputs come from the
official Mistral, Qwen, and Llama checkpoints listed in the provenance files.
Upstream dataset and model licenses continue to apply. No repository-level
license is asserted in this export; choose and add one before public release.

`codex_reviewed` is treated as `human_confirmed` throughout this project.
