# Experimental Results

Numbers below match the tables and confirmatory text in the MOSAIC paper.
Error is the positive class. Primary cells are five checkpoints, weighted
equally. Thresholds are selected on validation and then frozen for the held-out
test set.

## HalluMatch

HalluMatch contains 1,563 reviewed pairs of confident hallucinations and
correct responses, matched by checkpoint, source, and sequence confidence
inside each fact-disjoint split.

| Checkpoint | Hallucinations | Correct pool | Dev | Val | Test | Mean gap |
|---|---:|---:|---:|---:|---:|---|
| Mistral-7B | 347 | 3,057 | 191 | 75 | 81 | 1.25×10⁻⁵ |
| Qwen3-8B | 388 | 3,084 | 233 | 90 | 65 | 4.46×10⁻⁵ |
| Llama-3.1-8B | 257 | 3,911 | 141 | 53 | 63 | 6.99×10⁻⁵ |
| Qwen3-14B | 284 | 1,333 | 165 | 58 | 61 | 2.80×10⁻⁵ |
| Mistral-Nemo | 287 | 1,408 | 178 | 52 | 57 | 1.02×10⁻⁴ |
| **Total** | **1,563** | **12,793** | **908** | **328** | **327** | — |

The largest mean confidence gap is 1.02×10⁻⁴. Construction starts from 85,278
generations. Trivia-style and popularity-style sources contribute 992 and 571
pairs.

## Matching removes the confidence shortcut

After matching, every error has sequence confidence at least 0.9 and is paired
with a correct response of nearly identical confidence.

| Method | Macro recall |
|---|---:|
| Sequence confidence | 9.30% |
| Self-consistency (K=4) | 17.26% |
| Semantic entropy (K=4) | 21.22% |
| Internal-state trajectory | 39.01% |

Sequence confidence obtains 50.05% balanced accuracy on the matched task.

## Held-out test (Table 2)

Validation-selected thresholds are applied without refitting features, fusion,
or thresholds. Each cell reports TP/FP. Parentheses are the number of
hallucinations and the same number of correct controls. ΔR is versus the
internal-state trajectory.

| Method | M7 (81) | Q8 (65) | L8 (63) | Q14 (61) | MN (57) | Macro R | FPR | Acc. | ΔR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean hidden probe | 34/7 | 14/6 | 27/6 | 13/5 | 25/5 | 34.31 | 8.87 | 62.72 | −6.06 |
| Capacity-matched MLP | 37/6 | 15/7 | 29/6 | 14/5 | 27/5 | 37.02 | 8.93 | 64.04 | −3.35 |
| Behavior expert | 30/6 | 18/8 | 24/7 | 15/6 | 25/5 | 34.26 | 9.89 | 62.18 | −6.11 |
| Persistence expert | 18/5 | 13/6 | 17/5 | 12/5 | 15/4 | 23.04 | 7.71 | 57.66 | −17.33 |
| Internal-state trajectory | 40/6 | 17/7 | 31/6 | 16/5 | 29/5 | 40.37 | 8.93 | 65.72 | — |
| Flat concatenation | 42/7 | 20/9 | 33/7 | 17/6 | 32/5 | 43.80 | 10.44 | 66.68 | +3.43 |
| **MOSAIC** | **45/5** | **22/10** | **36/6** | **19/5** | **36/5** | **48.17** | **9.61** | **69.28** | **+7.80** |

MOSAIC raises macro recall from 40.37% to 48.17% at 9.61% macro FPR and 69.28%
accuracy. A 500-replicate hierarchical bootstrap that resamples development,
validation, and test facts, then repeats fitting and threshold selection, gives
a 95% CI of 4.28 to 11.36 for the 7.80-point gain and a two-sided p value of
0.002. Five independent seeds give a mean gain of 7.42 points (SD 0.91). The
point estimate is positive on all five checkpoints (4.92 to 12.28 points).

### Transferred operating points

| Validation FPR cap | Trajectory FPR / recall | MOSAIC FPR / recall | ΔR |
|---|---|---|---:|
| 5% | 4.88 / 31.62 | 5.16 / 39.02 | +7.40 |
| 10% | 8.93 / 40.37 | 9.61 / 48.17 | +7.80 |
| 15% | 14.52 / 51.08 | 14.71 / 58.46 | +7.38 |

A checkpoint-stratified bootstrap places MOSAIC macro test FPR at the 10%
target between 7.12% and 12.31%, versus 6.63% to 11.58% for the trajectory
expert. Test partial AUC over FPR from 0 to 0.10 is 0.061 for MOSAIC and 0.052
for the trajectory expert. Aggregate precision and F1 at the central
transferred threshold are 83.60% and 61.21%.

## Validation operating points

Validation is used for model and threshold selection, not as a second
confirmatory test.

At the validation threshold constrained to 10% FPR, sequence confidence,
self-consistency, semantic entropy, PALE-style contrast, the mean hidden
probe, the internal-state trajectory, and MOSAIC obtain 9.30, 17.26, 21.22,
37.72, 39.01, and 46.84% macro recall, respectively. MOSAIC at this point
obtains 84.32% precision, 46.84% recall, 8.71% FPR, 60.22% F1, and 69.07%
accuracy.

At a separate balanced validation operating point, MOSAIC reaches 73.07%
accuracy and 70.50% recall. Precision, FPR, and F1 are 74.32%, 24.36%, and
72.36%. The fixed-model grouped-bootstrap 95% CI for recall is 68.76 to 72.03.
Per-checkpoint recall is 78.0, 76.0, 65.0, 69.0, and 64.5%.

## Capacity-matched validation ablation (Table 3)

FPR ≤ 10%.

| Variant | Accuracy | Recall | FPR |
|---|---:|---:|---:|
| Mean answer probe | 62.41 | 34.18 | 9.36 |
| Last-4-layer concat. | 62.96 | 35.27 | 9.35 |
| Learned layer weights | 63.48 | 36.31 | 9.34 |
| Attention pooling | 63.81 | 36.92 | 9.30 |
| Parameter-matched MLP | 64.29 | 37.88 | 9.30 |
| Internal-state trajectory | 64.98 | 39.01 | 9.06 |
| **MOSAIC** | **69.07** | **46.84** | **8.71** |

MOSAIC is 7.83 points above the internal-state trajectory on this split.

## Inference cost (Table 4)

Auxiliary generations exclude the original answer. Latency is relative to
greedy generation, measured on an RTX PRO 6000 with the 14B checkpoint. Full
MOSAIC costs 3.12× greedy latency and peaks at 34.7 GB.

| Method | Aux. gen. | Recall | Latency |
|---|---:|---:|---|
| Self-consistency (K=4) | 4 | 17.26 | 1.72× |
| Self-consistency (K=8) | 8 | 20.34 | 2.43× |
| Semantic entropy (K=4) | 4 | 21.22 | 1.78× |
| Semantic entropy (K=8) | 8 | 25.47 | 2.54× |
| Paraphrase disagreement | 8 | 31.08 | 2.66× |
| Internal-state trajectory | 0 | 39.01 | 1.08× |
| **MOSAIC** | **8** | **46.84** | **3.12×** |

## Surface and split controls (Table 5)

Validation controls for question and fact difficulty. R is recall at FPR ≤ 10%.

| Available features | Accuracy | R |
|---|---:|---:|
| Length, digits, negation | 50.87 | 10.80 |
| Question-only lexical | 52.36 | 12.70 |
| Question-only embedding | 53.74 | 16.90 |
| Question-only prompt state | 55.12 | 20.60 |
| Answer-only embedding | 52.91 | 15.40 |
| Entity popularity/frequency | 53.26 | 14.80 |
| All nuisance features | 55.41 | 22.30 |
| **MOSAIC** | **73.07** | **46.84** |

On a stricter held-out same-fact control of 214 facts, the internal-state
trajectory obtains 39.3% recall and MOSAIC obtains 46.7%. The paired 7.4-point
gain has a 95% CI of 1.2 to 13.7.

Leave-one-checkpoint-out transfer reaches 32.6% macro recall at 10.8% FPR on
the unseen checkpoint, compared with 48.17% for checkpoint-specific training.

## Additional analyses

- Eighteen of 328 validation hallucinations satisfy the strict behavioral
  recovery rule. MOSAIC detects this subset with 95.0% equal-checkpoint macro
  recall. A uniform paraphrase vote corrects nine hallucinations without
  changing a matched correct response.
- Exploratory patching and erasure on Mistral-7B, Qwen3-8B, and Llama-3.1-8B
  give control-adjusted margins of −0.116, −2.258, and −0.202. None moves
  toward the alternative candidate.
- A score-blind audit of 100 false negatives finds 41 answers stable under
  every auxiliary generation, 30 overlapping internal-state trajectories, 18
  suppressed by the gate, and 11 within two score points of the threshold.
- With 10%, 25%, 50%, and 100% of labeled development facts, MOSAIC obtains
  35.84%, 40.91%, 44.52%, and 46.84% validation recall.
- Rebuilding the validation task at confidence 0.85 and 0.95 yields 2,041 and
  914 pairs. MOSAIC recall varies from 45.31% to 48.07%.
- Retaining the least-risky 50% of each balanced validation cell yields 74.06%
  selective accuracy.

The detector requires labeled target-checkpoint data, white-box states, and
eight auxiliary generations. The intervention results do not establish that the
predictive features provide a causal direction for correcting the answer.
