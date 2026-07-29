# One Confidence, Multiple Failure Mechanisms：机制与方法闭环报告

> 状态：**本轮工程与实验闭环已完成，但科学结论是有限且包含关键负结果。** Benchmark 任务成立，MOSAIC-v1 提供中等检测增益；现有 patch/erasure 不支持统一因果路径，conditional routing 也未形成有效缓解。本文只记录真实运行结果，不把未审核的 fresh generation 写成独立测试，不把 LaTeX 旧数字升级为未获支持的论文结论。

## 1. 范围与证据等级

本轮没有修改 benchmark、HCE/HCC 标签、BF16 confidence threshold、fact-disjoint split 或 Codex 审核标签。错误答案始终为正类。

当前证据分为三层：

1. **Public development/validation**：用于选择 Run6 配置、层数、正则化、专家与 gate 阈值。这里的数字是开发结果。
2. **Retrospective frozen test**：654 行、327 个 HCE/HCC matched pairs。它此前已用于冻结 baseline 和 MOSAIC-v0，因此不是 untouched/fresh final test；Run6 只按冻结 manifest 应用一次，未在结果出来后返调。
3. **Fresh generation-only holdout**：五模型各 4,000 行 BF16 generation 已完成，但没有 correctness screening、Codex review、matching 或 detector scoring，因此不能报告独立复现性能。

正式 frozen cell 为：

| Cell | Pairs |
|---|---:|
| Mistral-7B / TriviaQA | 81 |
| Qwen3-8B / TriviaQA | 65 |
| Llama-3.1-8B / TriviaQA | 63 |
| Qwen3-14B / PopQA | 61 |
| Mistral-Nemo / PopQA | 57 |
| **Total** | **327** |

前三个 cell 来自 TriviaQA，后两个来自 PopQA。五模型 macro 只能支撑跨模型、跨来源描述，不能直接解释参数规模效应。

## 2. 冻结方法定义

唯一冻结的 MOSAIC-v1 是 Run6：

- 方法：`learned_gate_no_phenotype`
- layer slots：16
- base/expert/gate regularization：`C=0.01`
- gate 输入：development OOF expert predictions 与允许的 behavioral/internal evidence
- threshold：每个 cell 只在 validation 冻结
- gold answer：不进入推理候选集合
- phenotype supervision：未使用

因此本轮不能把 phenotype supervision、causal internal pathway 或 STEER 写成 MOSAIC-v1 已实现且验证的组成部分。

## 3. Detection 结果

### 3.1 Public validation

| Cell | Accuracy | Recall | FPR |
|---|---:|---:|---:|
| Mistral-7B / TriviaQA | 74.00% | 57.33% | 9.33% |
| Qwen3-8B / TriviaQA | 77.22% | 64.44% | 10.00% |
| Llama-3.1-8B / TriviaQA | 63.21% | 35.85% | 9.43% |
| Qwen3-14B / PopQA | 65.52% | 36.21% | 5.17% |
| Mistral-Nemo / PopQA | 65.38% | 40.38% | 9.62% |
| **Equal-cell macro** | **69.07%** | **46.84%** | **8.71%** |

Validation pooled confusion count 为 TP/FN/FP/TN `162/166/29/299`。这些数字用于方法选择，不是独立测试性能。

### 3.2 Retrospective frozen test

| Cell | TP/FN/FP/TN | Accuracy | Recall | FPR |
|---|---:|---:|---:|---:|
| Mistral-7B / TriviaQA | 48/33/13/68 | 71.60% | 59.26% | 16.05% |
| Qwen3-8B / TriviaQA | 50/15/10/55 | 80.77% | 76.92% | 15.38% |
| Llama-3.1-8B / TriviaQA | 20/43/2/61 | 64.29% | 31.75% | 3.17% |
| Qwen3-14B / PopQA | 22/39/5/56 | 63.93% | 36.07% | 8.20% |
| Mistral-Nemo / PopQA | 21/36/3/54 | 65.79% | 36.84% | 5.26% |
| **Equal-cell macro** | - | **69.28%** | **48.17%** | **9.61%** |

Pooled confusion count 为 `161/166/33/294`，对应 Accuracy `69.57%`、Recall `49.24%`、FPR `10.09%`。Macro 与 pooled 必须分开解释。

Mistral-7B 和 Qwen3-8B 的 frozen FPR 分别为 `16.05%` 和 `15.38%`，所以不能声称 MOSAIC-v1 在每个模型上都满足 FPR 不超过 10%。

### 3.3 Baseline 与增量

| Method | Status | Accuracy | Recall | FPR |
|---|---|---:|---:|---:|
| Sequence confidence | exact | 50.16% | 7.43% | 7.10% |
| Predictive/token entropy | exact | 53.12% | 13.34% | 7.10% |
| P(True) | approximate | 59.73% | 26.75% | 7.28% |
| Self-consistency K=4 | exact for K=4 | 51.59% | 5.26% | 2.08% |
| Semantic Entropy K=4 | paper-faithful K=4 | 51.75% | 5.57% | 2.08% |
| Fixed paraphrase consistency | approximate panel | 54.44% | 10.12% | 1.23% |
| Prompt-end hidden probe | exact for position | 65.45% | 42.00% | 11.09% |
| First-answer hidden probe | exact for position | 62.09% | 33.69% | 9.52% |
| Mean-answer hidden probe | exact for position | 64.67% | 40.35% | 11.01% |
| INSIDE EigenScore | approximate | 51.63% | 11.07% | 7.82% |
| LOS-Net | approximate reproduction | 50.97% | 16.81% | 14.86% |
| Internal component | project component | 66.96% | 42.67% | 8.75% |
| MOSAIC-v0 | frozen starting point | 67.24% | 43.34% | 8.86% |
| **MOSAIC-v1 Run6** | one-shot frozen application | **69.28%** | **48.17%** | **9.61%** |

`PALE` 没有可信、对齐 frozen cohort 的实现，保持 `unavailable`，没有制造替代分数。

Paired bootstrap 结论：

- MOSAIC-v1 Recall 95% CI：`[43.06%, 53.39%]`。
- 相对 MOSAIC-v0：Recall `+4.83pp`，95% CI `[-0.57, +10.35]`，区间跨零。
- 相对 validation-preselected mean-answer probe：Recall `+7.82pp`，95% CI `[+1.98, +13.71]`。
- 预设“比最强 baseline 至少提高 10pp Recall”的目标没有达到。

因此 MOSAIC-v1 的合理定位是 **modest detector improvement**，不是稳定的方法突破。

## 4. Failure Pathway 观察

### 4.1 Behavioral recovery

在 frozen HCE 中，固定 paraphrase panel 可观察到 behavioral recovery 的样本只有 `18/327`：

| Cell | Behavioral HCE |
|---|---:|
| Mistral-7B / TriviaQA | 6 |
| Qwen3-8B / TriviaQA | 6 |
| Llama-3.1-8B / TriviaQA | 2 |
| Qwen3-14B / PopQA | 1 |
| Mistral-Nemo / PopQA | 3 |

Run6 对该小子类的 equal-cell macro Recall 为 `86.67%`。剩余 `309/327` HCE 只能标为 `unassessed_HCE`，其 macro Recall 为 `46.66%`。

这个结果支持“行为恢复是一个真实但稀少的 failure phenotype”，不支持把全部非恢复错误重命名为 persistent。

### 4.2 Internal 与 persistent 标签边界

- Frozen cohort 没有进行 causal internal/persistent phenotype assessment。
- Public validation 中有 `18/328` behavioral recoverable、`17/328` assessed persistent、`293/328` unassessed HCE。
- Public validation 中 causal internal-competitive 标签为 `0`。
- Phenotype 标签是 partial observational/causal-assessment strata，不是覆盖全部样本的机制真值。

## 5. 机制实验

Patch 与 erasure 共覆盖 4 个 cell、`19` 个 validation HCE：

| Cell | Layer | Patch control-adjusted effect | Erasure control-adjusted drop | 判决 |
|---|---:|---:|---:|---|
| Mistral-7B / TriviaQA | 28 | +0.2934 | -0.0390 | patch 正，erasure 不支持 |
| Qwen3-8B / TriviaQA | 26 | -1.0990 | -0.1191 | 不支持 |
| Llama-3.1-8B / TriviaQA | 28 | -0.2021 | -0.0028 | 不支持；candidate adoption 0/7 |
| Qwen3-14B / PopQA | 35 | +0.5100 | -0.0493 | patch 正，erasure 反向 |
| Mistral-Nemo / PopQA | - | 未运行 | 未运行 | 无证据 |

控制包括 exact no-op、random layer/direction、unrelated-fact donor、shuffled candidate、reverse patch 与 HCC control。结果为：

- patch control-adjusted 为正的 cell：Mistral-7B、Qwen3-14B。
- erasure control-adjusted 为正的 cell：0。
- 同时通过 patch 与 erasure 的 cell：0。

当前可支持的机制表述是：

> 一部分 prompt-conditioned late state 可以被局部干预扰动，internal trajectory 也保留预测信号；但现有证据不支持跨模型共享、candidate-specific、可用于 STEER 的统一因果方向。

不能声称 prompt-conditioned answer-selection circuit、internal/output mismatch causal pathway 或 ACE/LEACE 反向验证已经成立。

## 6. 缓解实验

### 6.1 Public validation

| Policy | HCE corrected | HCC damaged | Net | Coverage | Selective accuracy |
|---|---:|---:|---:|---:|---:|
| Uniform paraphrase vote | 9 | 0 | +9 | 100.00% | 51.49% |
| Uniform likelihood aggregation | 8 | 4 | +4 | 100.00% | 50.72% |
| Detector-only abstention | 0 | 29 | -29 | 72.22% | 63.65% |
| Random routing | 1 | 29 | -28 | 73.33% | 62.96% |
| MOSAIC conditional routing | 7 | 29 | -22 | 73.33% | 64.20% |

### 6.2 Retrospective frozen test

| Policy | HCE corrected | HCC damaged | Net | Coverage | Selective accuracy |
|---|---:|---:|---:|---:|---:|
| Uniform paraphrase vote | 4 | 4 | 0 | 100.00% | 50.03% |
| Uniform likelihood aggregation | 8 | 3 | **+5** | 100.00% | 50.71% |
| Detector-only abstention | 0 | 33 | -33 | 71.11% | 64.70% |
| Random routing | 2 | 33 | -31 | 71.82% | 64.36% |
| MOSAIC conditional routing | 3 | 33 | **-30** | 71.82% | 64.56% |
| Uniform steering | - | - | - | - | unavailable |

冻结 conditional decisions 在附加标签前写出并固定，decision SHA256 为 `70b8b893134ba86249141ac45969b43003e882b9fa21664285639a8eb71097ce`。因此其失败不是后验评分错误。

结论：

- `RECOVER`：uniform likelihood 只有小幅 `+5` 净修复，尚不足以构成主要方法贡献。
- `STEER`：因果控制不支持，保持 unavailable。
- `ABSTAIN`：提高 selective accuracy，但以 28.89% 的拒答和 33 个 HCC trigger 为代价。
- MOSAIC conditional routing：没有改善最终答案正确率，不能称为成功缓解。

## 7. 成本、案例与可复现性

- Frozen panel size 固定为 5。
- 五个 cell 已保存 generation、teacher-forced candidate scoring、hidden extraction 与 classifier evaluation 操作数。
- 已可靠记录的 GPU stage wall time 为每 cell `111.9-216.3s`；这不是端到端训练加推理总成本。
- peak GPU memory 没有可信的逐阶段记录，表中保持 `null`，没有补估。
- `case_studies.json` 含 20 条真实 retrospective frozen 案例：每 cell 按 sample ID 字典序固定选 2 FN + 2 FP，包含问题、BF16 原始回答、gold aliases、confidence 与审核 provenance。
- 远端与本地比较了 81 个不可替代文件，缺失和 SHA mismatch 均为 0。
- One Confidence 相关测试：`221 passed`，1 个 MLP convergence warning。
- Frozen closure verifier：`654` 行、`327` pairs，通过。
- Public-validation scope verifier：5 个 cell，通过。

关键完整性文件：

- `table_main_detection.json` SHA256：`1a1517f05dcc99299e19cf333f193607b088e71ebd380c08695060f56c7d5c1f`
- `table_mitigation.json` SHA256：`d92fb7ae88d5ccc311eadbcd7e45340356e6fcf95b12c74df788b7c346c79aa8`
- `mechanism_evidence_audit.json` SHA256：`8d38a3c2097ad9ac31dbb29f630bb50c49f6a21f231d307b49f55dbb4f13c069`
- `case_studies.json` SHA256：`fc5a6de76485362aa031c3c7a5b5abfec9690911c6bf19eb847c344398e12cb8`
- `remote_local_sync_proof.json` SHA256：`c7b61706005c4fe270316ef51ee6e7d81b1e96d70b0e7f610845d6ba59f2b447`
- `artifact_sha256.json` SHA256：`f3cb1f725e3427932186f67f6f01a66e110954a6520e423e3b386af0740affef`

## 8. Fresh holdout 隔离

五模型各 `4,000` 行 BF16 generation 已同步本地：

- Llama-3.1-8B：`75d48d78...`
- Mistral-7B：`9bff563d...`
- Mistral-Nemo：`29ad60a8...`
- Qwen3-14B：`5d2648a9...`
- Qwen3-8B：`9ba1ba56...`

本轮没有：

- 读取 fresh 模型输出的 correctness；
- 生成 fresh screening 或 Codex review；
- 构造 fresh HCE/HCC matched pairs；
- 运行 detector、hidden-state、trajectory 或 mechanism score。

此前仅行政性查看过少量 source question/gold 行，没有把模型输出与 gold 比较。该 cohort 仍可继续完成审核与匹配，但当前绝不能称为独立 final-test 结果。

## 9. 论文可以与不可以声称的内容

### 可以声称

1. Confidence-matched HCE/HCC detection 去除了“错误通常置信度更低”的主要捷径。
2. Sequence confidence 在该任务上接近失效，而 internal trajectory 保留中等预测信号。
3. Behavioral recovery 是跨五个 frozen cell 均可观察、但占比很低的子类。
4. Run6 相对 mean-answer probe 有正的配对 Recall 差异，但相对 MOSAIC-v0 的区间跨零。
5. 严格 patch/erasure 控制否定了当前统一 candidate-specific causal pathway。
6. 现有 routing 未成功，uniform likelihood 只产生小幅净修复。

### 不可以声称

1. MOSAIC-v1 在 untouched independent test 上成功。
2. MOSAIC-v1 在五个模型都满足 FPR 不超过 10%。
3. MOSAIC-v1 比最强 baseline 稳定提高至少 10pp Recall。
4. Phenotype supervision 是性能来源。
5. 两条 causal failure mechanisms 已跨模型验证。
6. STEER 或 conditional recover/steer/abstain routing 已改善最终正确率。
7. 存在跨五模型共享的 hallucination circuit。

## 10. LaTeX Placeholder 账本

`latex_v2/main.tex` 与 `latex_v2/figures/projected_detection.tex` 中的 `46.84% / 8.71%` 是真实 **public validation** 数字，不能标成 independent test。

以下仍是 unsupported 或 placeholder：

- “causal pathway validated”；
- “successful conditional routing”；
- “至少 +10pp Recall gain”；
- “independent/fresh test success”；
- frozen phenotype-specific internal/persistent Recall；
- STEER correction rate；
- peak GPU memory；
- 把 fresh 5×4,000 generation 写成已审核复现。

## 11. 最终判决

本轮已经完成可复现的工程闭环：五模型冻结检测、MOSAIC-v1 消融、bootstrap、phenotype 描述、patch/erasure 控制、缓解、成本、固定 FP/FN 案例、远端同步与哈希均已落盘。

科学上形成的是一条 **有限/负面闭环**：

- 新任务成立；
- internal trajectory 提供中等检测信号；
- behavioral recovery 是稀少但真实的子机制类型；
- 当前统一 causal internal pathway 被控制实验否定；
- MOSAIC-v1 只有 modest detection gain；
- conditional routing 和 STEER 没有成立。

因此论文当前可以写“新任务、异质 observable pathways、有限检测增益、严格机制负结果”，不能写成“机制证据导出并验证了一个显著优于 baseline 的 recover/steer/abstain 方法”。若论文必须依赖正向 causal mechanism 或 routing superiority，现有证据仍不足。
