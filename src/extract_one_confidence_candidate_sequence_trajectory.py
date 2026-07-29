#!/usr/bin/env python3
"""Extract public, gold-free candidate-versus-canonical layer trajectories.

Candidates are never discovered in this script.  They must already be frozen by
the corrected-panel NLI pipeline and recorded by the candidate-margin artifact.
For each available candidate, the script teacher-forces the complete canonical
and candidate answers under the canonical prompt, applies the model's final norm
and LM head to every transformer-block output, and records length-normalized
sequence-score trajectories.

Only development and validation records from the public benchmark are accepted.
No gold answer or locked-test content is read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import torch


ANSWER_SYSTEM = (
    "Return only the requested factual answer entity or short value. Do not write "
    "a sentence, preamble, explanation, or reasoning."
)
PUBLIC_SPLITS = ("development", "validation")
ALLOWED_CANDIDATE_SOURCES = {
    "natural_alternative_cluster": (
        "alternative_candidate_natural",
        "alternative_candidate_cluster_natural",
    ),
    "safe_panel_alternative_cluster": (
        "alternative_candidate_all_safe",
        "alternative_candidate_cluster_all_safe",
    ),
}
TRAJECTORY_SUMMARY_KEYS = (
    "early_mean_margin",
    "middle_mean_margin",
    "late_mean_margin",
    "late_minus_middle_margin",
    "trajectory_auc",
    "minimum_margin",
    "minimum_margin_layer",
    "maximum_margin",
    "maximum_margin_layer",
    "maximum_positive_slope",
    "maximum_positive_slope_to_layer",
    "maximum_negative_slope",
    "maximum_negative_slope_to_layer",
    "maximum_absolute_slope",
    "sign_flip_count",
    "candidate_preferred_layer_fraction",
    "first_candidate_preferred_layer",
    "last_candidate_preferred_layer",
    "final_margin",
    "final_candidate_preferred",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text)).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def normalized_whitespace(text: str) -> str:
    return " ".join(str(text).strip().split())


def indexed(records: Iterable[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in records:
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise ValueError(f"duplicate {name} sample_id: {sample_id}")
        output[sample_id] = row
    return output


def reject_unsafe_path(path: Path) -> None:
    lowered = str(path).casefold()
    if "invalidated" in lowered:
        raise ValueError(f"invalidated artifacts are forbidden: {path}")
    if "locked" in path.name.casefold():
        raise ValueError(f"locked artifacts are forbidden: {path}")


def require_public_artifact(payload: dict[str, Any], name: str) -> None:
    if payload.get("locked_accessed"):
        raise ValueError(f"{name} reports locked-test access")
    schema = str(payload.get("schema_version", ""))
    if "public" not in schema and name == "benchmark":
        raise ValueError(f"benchmark is not a public artifact: {schema!r}")


def benchmark_public_rows(
    benchmark: dict[str, Any], cell: str, splits: tuple[str, ...]
) -> list[dict[str, Any]]:
    if cell not in benchmark.get("cells", {}):
        raise ValueError(f"unknown benchmark cell: {cell}")
    rows: list[dict[str, Any]] = []
    for split in splits:
        if split not in PUBLIC_SPLITS:
            raise ValueError(f"non-public split requested: {split}")
        split_payload = benchmark["cells"][cell].get(split)
        if not isinstance(split_payload, dict) or "records" not in split_payload:
            raise ValueError(f"missing public split {cell}/{split}")
        for raw in split_payload["records"]:
            if raw.get("split") != split:
                raise ValueError(f"split mismatch for {raw.get('sample_id')}")
            # Deliberately copy only deployment-visible fields.  In particular,
            # gold_aliases is never accessed or propagated.
            rows.append(
                {
                    "sample_id": str(raw["sample_id"]),
                    "model": str(raw["model"]),
                    "dataset": str(raw["dataset"]),
                    "row_id": str(raw["row_id"]),
                    "fact_key_sha256": str(raw["fact_key_sha256"]),
                    "split": split,
                    "label": str(raw["class"]),
                    "confidence": float(raw["confidence"]),
                    "question": str(raw["question"]),
                    "canonical_answer": str(raw["generated_answer"]),
                }
            )
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate public benchmark rows in {cell}")
    return rows


def _same_text(left: Any, right: Any) -> bool:
    return normalized_whitespace(str(left)) == normalized_whitespace(str(right))


def build_source_records(
    benchmark: dict[str, Any],
    features: dict[str, Any],
    nli: dict[str, Any],
    margins: dict[str, Any],
    cell: str,
    splits: tuple[str, ...] = PUBLIC_SPLITS,
    candidate_only: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Join public artifacts without consulting correctness answers.

    The candidate-margin record is the sole candidate-selection authority.  The
    corrected NLI and panel artifacts are used only to prove that this candidate
    was frozen before this trajectory extraction.
    """

    require_public_artifact(benchmark, "benchmark")
    for payload, name in ((features, "features"), (nli, "nli"), (margins, "margins")):
        require_public_artifact(payload, name)
        if payload.get("cell") != cell:
            raise ValueError(f"{name} cell mismatch: {payload.get('cell')} != {cell}")
    if not features.get("complete"):
        raise ValueError("corrected-panel feature artifact is incomplete")
    answer_protocol = features.get("answer_protocol", {})
    if answer_protocol.get("system") != ANSWER_SYSTEM:
        raise ValueError("corrected panel used a different canonical answer system prompt")
    if answer_protocol.get("qwen_thinking") is not False:
        raise ValueError("corrected panel must explicitly disable Qwen thinking")
    if margins.get("gold_visible") is not False:
        raise ValueError("candidate-margin artifact is not explicitly gold-free")

    public_rows = benchmark_public_rows(benchmark, cell, splits)
    feature_rows = indexed(features.get("records", []), "corrected-panel-feature")
    nli_rows = indexed(nli.get("records", []), "corrected-panel-nli")
    margin_rows = indexed(margins.get("records", []), "candidate-margin")
    public_ids = {row["sample_id"] for row in public_rows}
    for source, name in (
        (feature_rows, "corrected-panel-feature"),
        (nli_rows, "corrected-panel-nli"),
        (margin_rows, "candidate-margin"),
    ):
        missing = public_ids - set(source)
        if missing:
            raise ValueError(f"{name} misses {len(missing)} public rows; first={min(missing)}")

    output: list[dict[str, Any]] = []
    for source_record_index, base in enumerate(public_rows):
        sample_id = base["sample_id"]
        feature = feature_rows[sample_id]
        nli_row = nli_rows[sample_id]
        margin = margin_rows[sample_id]
        expected = {
            "split": base["split"],
            "label": base["label"],
            "fact_key_sha256": base["fact_key_sha256"],
        }
        for key, value in expected.items():
            for source, name in ((feature, "features"), (nli_row, "nli"), (margin, "margins")):
                if key in source and source[key] != value:
                    raise ValueError(f"{sample_id}: {name} {key} mismatch")
        if not _same_text(feature["canonical_answer"], base["canonical_answer"]):
            raise ValueError(f"{sample_id}: canonical answer differs from benchmark")
        if not _same_text(margin["canonical_answer"], base["canonical_answer"]):
            raise ValueError(f"{sample_id}: margin canonical answer differs from benchmark")
        panel_questions = feature.get("panel_questions", [])
        panel_answers = feature.get("panel_answers", [])
        panel_token_ids = feature.get("panel_generated_token_ids", [])
        if not panel_questions or not _same_text(panel_questions[0], base["question"]):
            raise ValueError(f"{sample_id}: canonical panel question differs from benchmark")
        if not panel_answers:
            raise ValueError(f"{sample_id}: corrected panel has no answers")
        if len(panel_answers) != len(panel_token_ids):
            raise ValueError(f"{sample_id}: panel answer/token row counts differ")
        cluster_ids = nli_row.get("cluster_ids", [])
        if len(cluster_ids) != len(panel_answers) - 1:
            raise ValueError(f"{sample_id}: NLI cluster IDs do not align with four panel answers")

        candidate_available = bool(margin.get("candidate_available"))
        candidate = str(margin.get("candidate_answer") or "").strip()
        candidate_source = str(margin.get("candidate_source") or "none")
        candidate_cluster = margin.get("candidate_cluster")
        candidate_panel_index: int | None = None
        candidate_generated_ids: list[int] = []
        if candidate_available:
            if not candidate or candidate_source not in ALLOWED_CANDIDATE_SOURCES:
                raise ValueError(f"{sample_id}: invalid frozen candidate metadata")
            text_key, cluster_key = ALLOWED_CANDIDATE_SOURCES[candidate_source]
            if not _same_text(candidate, nli_row.get(text_key, "")):
                raise ValueError(f"{sample_id}: candidate is not the frozen NLI alternative")
            if candidate_cluster != nli_row.get(cluster_key):
                raise ValueError(f"{sample_id}: candidate cluster differs from frozen NLI cluster")
            if normalize(candidate) == normalize(base["canonical_answer"]):
                raise ValueError(f"{sample_id}: candidate normalizes to canonical answer")
            matching_indices = [
                index
                for index, answer in enumerate(panel_answers[1:], start=1)
                if _same_text(answer, candidate)
                and cluster_ids[index - 1] == candidate_cluster
            ]
            if not matching_indices:
                raise ValueError(f"{sample_id}: frozen candidate has no aligned panel token sequence")
            candidate_panel_index = matching_indices[0]
            candidate_generated_ids = [
                int(value) for value in panel_token_ids[candidate_panel_index]
            ]
        else:
            if candidate or candidate_source != "none" or candidate_cluster is not None:
                raise ValueError(f"{sample_id}: unavailable candidate has non-empty metadata")

        if candidate_only and not candidate_available:
            continue
        output.append(
            {
                **base,
                "source_record_index": source_record_index,
                "panel_questions": [str(value) for value in panel_questions],
                # The panel's first answer is a fresh greedy repeat and can differ
                # from the frozen benchmark answer.  It must not replace the
                # canonical benchmark answer or its standalone teacher forcing.
                "canonical_generated_token_ids": None,
                "candidate_available": candidate_available,
                "candidate_answer": candidate,
                "candidate_source": candidate_source,
                "candidate_cluster": candidate_cluster,
                "candidate_panel_index": candidate_panel_index,
                "candidate_generated_token_ids": candidate_generated_ids,
                "source_canonical_nll": margin.get("canonical_answer_nll_recomputed"),
                "source_candidate_nll": margin.get("candidate_answer_nll"),
                "source_candidate_logscore_margin": float(
                    margin.get("candidate_logscore_margin", 0.0)
                ),
            }
        )
        if limit is not None and len(output) >= limit:
            break
    return output


def chat_ids(tokenizer: Any, question: str) -> list[int]:
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": question},
    ]
    try:
        values = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        values = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    values = [int(value) for value in values]
    if not values:
        raise ValueError("canonical prompt tokenizes empty")
    return values


def aligned_answer_ids(
    tokenizer: Any,
    answer: str,
    frozen_generated_ids: list[int] | None,
    sample_id: str,
    answer_name: str,
) -> tuple[list[int], dict[str, Any]]:
    teacher_ids = [
        int(value) for value in tokenizer.encode(answer, add_special_tokens=False)
    ]
    if not teacher_ids:
        raise ValueError(f"{sample_id}: {answer_name} tokenizes empty")
    teacher_decoded = tokenizer.decode(teacher_ids, skip_special_tokens=True).strip()
    if normalized_whitespace(teacher_decoded) != normalized_whitespace(answer):
        raise ValueError(f"{sample_id}: {answer_name} teacher-token decode mismatch")
    frozen_match: bool | None = None
    frozen_decoded: str | None = None
    if frozen_generated_ids is not None:
        frozen_decoded = tokenizer.decode(
            frozen_generated_ids, skip_special_tokens=True
        ).strip()
        if normalized_whitespace(frozen_decoded) != normalized_whitespace(answer):
            raise ValueError(f"{sample_id}: {answer_name} frozen-token decode mismatch")
        # BPE/SentencePiece token IDs may legitimately depend on the preceding
        # assistant-prefix bytes.  The frozen source scorer explicitly used
        # standalone encode(answer), so trajectories must use teacher_ids too.
        # Panel IDs remain a provenance/decode audit and any difference is
        # surfaced rather than silently treated as the scoring tokenization.
        frozen_match = teacher_ids == frozen_generated_ids
    return teacher_ids, {
        "token_ids": teacher_ids,
        "token_count": len(teacher_ids),
        "decoded_text": teacher_decoded,
        "teacher_decode_matches_answer": True,
        "frozen_generated_token_ids_available": frozen_generated_ids is not None,
        "frozen_generated_decoded_text": frozen_decoded,
        "frozen_generated_decode_matches_answer": (
            True if frozen_generated_ids is not None else None
        ),
        "frozen_generated_token_ids_match": frozen_match,
        "context_dependent_tokenization_difference": frozen_match is False,
    }


def model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def padded(
    ids_list: list[list[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, ids_list))
    values = torch.full(
        (len(ids_list), width), pad_id, dtype=torch.long, device=device
    )
    mask = torch.zeros_like(values)
    for index, ids in enumerate(ids_list):
        values[index, width - len(ids) :] = torch.tensor(
            ids, dtype=torch.long, device=device
        )
        mask[index, width - len(ids) :] = 1
    return values, mask


@torch.inference_mode()
def teacher_forced_nll_batch(
    model: Any,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    answer_lengths: list[int],
    pad_id: int,
) -> list[float]:
    """Exact replay of the frozen candidate-margin scorer's batch path."""

    ids, mask = padded(sequences, pad_id, model_device(model))
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    output = []
    for index, sequence in enumerate(sequences):
        left_padding = ids.shape[1] - len(sequence)
        start = left_padding + prompt_lengths[index] - 1
        length = answer_lengths[index]
        target = ids[index, start + 1 : start + 1 + length]
        answer_logits = logits[index, start : start + length]
        output.append(
            float(
                -torch.log_softmax(answer_logits, dim=-1)
                .gather(1, target[:, None])
                .mean()
            )
        )
    return output


@torch.inference_mode()
def replay_frozen_source_batches(
    model: Any,
    tokenizer: Any,
    all_rows: list[dict[str, Any]],
    selected_sample_ids: set[str],
    source_batch_rows: int,
    replay_tolerance: float,
) -> dict[str, dict[str, Any]]:
    """Reproduce the old candidate-margin batches before comparing trajectories.

    A per-answer BF16 forward can differ numerically from the original heterogeneous
    left-padded batch.  Exact batch replay establishes prompt/token/source identity;
    the later single-row delta is recorded rather than confused with misalignment.
    """

    if source_batch_rows <= 0:
        raise ValueError("source_batch_rows must be positive")
    target_batch_starts = {
        (int(row["source_record_index"]) // source_batch_rows) * source_batch_rows
        for row in all_rows
        if row["sample_id"] in selected_sample_ids
    }
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define pad or EOS token for source replay")
    replayed: dict[str, dict[str, Any]] = {}
    for batch_start in sorted(target_batch_starts):
        batch = all_rows[batch_start : batch_start + source_batch_rows]
        sequences: list[list[int]] = []
        prompt_lengths: list[int] = []
        answer_lengths: list[int] = []
        metadata: list[tuple[str, int]] = []
        for row in batch:
            if not row["candidate_available"]:
                continue
            canonical_ids = [
                int(value)
                for value in tokenizer.encode(
                    row["canonical_answer"], add_special_tokens=False
                )
            ]
            candidate_ids = [
                int(value)
                for value in tokenizer.encode(
                    row["candidate_answer"], add_special_tokens=False
                )
            ]
            if not canonical_ids or not candidate_ids:
                raise ValueError(f"{row['sample_id']}: empty source-replay answer")
            prompts = [chat_ids(tokenizer, question) for question in row["panel_questions"]]
            metadata.append((row["sample_id"], len(prompts)))
            for prompt in prompts:
                for answer_ids in (canonical_ids, candidate_ids):
                    sequences.append(prompt + answer_ids)
                    prompt_lengths.append(len(prompt))
                    answer_lengths.append(len(answer_ids))
        values = teacher_forced_nll_batch(
            model,
            sequences,
            prompt_lengths,
            answer_lengths,
            int(pad_id),
        ) if sequences else []
        iterator = iter(values)
        for sample_id, prompt_count in metadata:
            canonical_nlls = []
            candidate_nlls = []
            for _ in range(prompt_count):
                canonical_nlls.append(float(next(iterator)))
                candidate_nlls.append(float(next(iterator)))
            row = all_rows[next(
                index for index in range(batch_start, min(batch_start + source_batch_rows, len(all_rows)))
                if all_rows[index]["sample_id"] == sample_id
            )]
            canonical_error = abs(
                canonical_nlls[0] - float(row["source_canonical_nll"])
            )
            candidate_error = abs(
                candidate_nlls[0] - float(row["source_candidate_nll"])
            )
            replay_margin = canonical_nlls[0] - candidate_nlls[0]
            margin_error = abs(
                replay_margin - float(row["source_candidate_logscore_margin"])
            )
            if max(canonical_error, candidate_error, margin_error) > replay_tolerance:
                raise RuntimeError(
                    f"{sample_id}: exact source-batch replay exceeds {replay_tolerance:.6g} "
                    f"(canonical={canonical_error:.6g}, candidate={candidate_error:.6g}, "
                    f"margin={margin_error:.6g})"
                )
            replayed[sample_id] = {
                "source_batch_start": batch_start,
                "source_batch_rows": source_batch_rows,
                "canonical_nll": canonical_nlls[0],
                "candidate_nll": candidate_nlls[0],
                "candidate_logscore_margin": replay_margin,
                "canonical_nll_abs_error": canonical_error,
                "candidate_nll_abs_error": candidate_error,
                "candidate_margin_abs_error": margin_error,
            }
    missing = selected_sample_ids - set(replayed)
    if missing:
        raise RuntimeError(f"source replay missed selected rows: {sorted(missing)[:3]}")
    return replayed


def transformer_components(model: Any) -> tuple[Any, Any, Any]:
    body = getattr(model, "model", None)
    blocks = getattr(body, "layers", None)
    final_norm = getattr(body, "norm", None)
    output_head = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings")
        else getattr(model, "lm_head", None)
    )
    if body is None or blocks is None or final_norm is None or output_head is None:
        raise ValueError("unsupported causal LM layout; expected model.layers, model.norm, lm_head")
    return blocks, final_norm, output_head


@torch.inference_mode()
def capture_answer_states(
    model: Any, prompt_ids: list[int], answer_ids: list[int]
) -> tuple[list[torch.Tensor], torch.Tensor]:
    if not prompt_ids or not answer_ids:
        raise ValueError("prompt and answer token IDs must be non-empty")
    blocks, _, _ = transformer_components(model)
    positions = list(range(len(prompt_ids) - 1, len(prompt_ids) + len(answer_ids) - 1))
    input_ids = torch.tensor(
        [prompt_ids + answer_ids], dtype=torch.long, device=model_device(model)
    )
    captured: list[torch.Tensor | None] = [None] * len(blocks)
    handles = []

    def make_hook(index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output if torch.is_tensor(output) else output[0]
            captured[index] = hidden[0, positions, :].detach()

        return hook

    for index, block in enumerate(blocks):
        handles.append(block.register_forward_hook(make_hook(index)))
    try:
        output = model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured):
        raise RuntimeError("failed to capture every transformer block output")
    actual_logits = output.logits[0, positions].detach().float()
    return [value for value in captured if value is not None], actual_logits


@torch.inference_mode()
def selected_token_logprobs(
    logits: torch.Tensor, answer_ids: list[int]
) -> torch.Tensor:
    targets = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("answer token positions and targets do not align")
    return torch.log_softmax(logits.float(), dim=-1).gather(
        1, targets[:, None]
    ).squeeze(1)


@torch.inference_mode()
def score_answer_trajectory(
    model: Any,
    prompt_ids: list[int],
    answer_ids: list[int],
    final_nll_tolerance: float,
) -> dict[str, Any]:
    states, actual_logits = capture_answer_states(model, prompt_ids, answer_ids)
    _, final_norm, output_head = transformer_components(model)
    layer_mean_logprob: list[float] = []
    layer_token_logprobs: list[list[float]] = []
    for hidden in states:
        logits = output_head(final_norm(hidden)).float()
        token_values = selected_token_logprobs(logits, answer_ids)
        layer_token_logprobs.append([float(value) for value in token_values.cpu()])
        layer_mean_logprob.append(float(token_values.mean().cpu()))
    actual_token_values = selected_token_logprobs(actual_logits, answer_ids)
    actual_mean_logprob = float(actual_token_values.mean().cpu())
    final_token_values = torch.tensor(layer_token_logprobs[-1], dtype=torch.float32)
    token_error = float(
        (final_token_values - actual_token_values.detach().cpu().float()).abs().max()
    )
    nll_error = abs(-layer_mean_logprob[-1] - (-actual_mean_logprob))
    if nll_error > final_nll_tolerance:
        raise RuntimeError(
            f"final-layer lens NLL error {nll_error:.6g} exceeds "
            f"{final_nll_tolerance:.6g}"
        )
    return {
        "layer_mean_logprob": layer_mean_logprob,
        "layer_geometric_mean_probability": [
            math.exp(min(0.0, value)) for value in layer_mean_logprob
        ],
        "layer_token_logprobs": layer_token_logprobs,
        "actual_final_token_logprobs": [
            float(value) for value in actual_token_values.cpu()
        ],
        "actual_final_mean_logprob": actual_mean_logprob,
        "actual_final_nll": -actual_mean_logprob,
        "lens_final_nll": -layer_mean_logprob[-1],
        "final_layer_nll_abs_error": nll_error,
        "final_layer_token_logprob_max_abs_error": token_error,
    }


def _regions(values: list[float]) -> tuple[list[float], list[float], list[float]]:
    if not values:
        raise ValueError("empty trajectory")
    if len(values) == 1:
        return values, values, values
    early_stop = max(1, len(values) // 3)
    middle_stop = max(early_stop + 1, (2 * len(values)) // 3)
    middle_stop = min(middle_stop, len(values) - 1)
    return values[:early_stop], values[early_stop:middle_stop], values[middle_stop:]


def sign_flip_count(values: list[float]) -> int:
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in values]
    nonzero = [value for value in signs if value]
    return sum(left != right for left, right in zip(nonzero, nonzero[1:]))


def trajectory_summary(margins: list[float]) -> dict[str, Any]:
    early, middle, late = _regions(margins)
    slopes = [right - left for left, right in zip(margins, margins[1:])]
    max_slope = max(slopes) if slopes else 0.0
    min_slope = min(slopes) if slopes else 0.0
    preferred = [index + 1 for index, value in enumerate(margins) if value > 0.0]
    return {
        "early_mean_margin": float(sum(early) / len(early)),
        "middle_mean_margin": float(sum(middle) / len(middle)),
        "late_mean_margin": float(sum(late) / len(late)),
        "late_minus_middle_margin": float(
            sum(late) / len(late) - sum(middle) / len(middle)
        ),
        "trajectory_auc": float(sum(margins) / len(margins)),
        "minimum_margin": float(min(margins)),
        "minimum_margin_layer": int(margins.index(min(margins)) + 1),
        "maximum_margin": float(max(margins)),
        "maximum_margin_layer": int(margins.index(max(margins)) + 1),
        "maximum_positive_slope": float(max_slope),
        "maximum_positive_slope_to_layer": int(slopes.index(max_slope) + 2) if slopes else None,
        "maximum_negative_slope": float(min_slope),
        "maximum_negative_slope_to_layer": int(slopes.index(min_slope) + 2) if slopes else None,
        "maximum_absolute_slope": float(max((abs(value) for value in slopes), default=0.0)),
        "sign_flip_count": int(sign_flip_count(margins)),
        "candidate_preferred_layer_fraction": float(len(preferred) / len(margins)),
        "first_candidate_preferred_layer": preferred[0] if preferred else None,
        "last_candidate_preferred_layer": preferred[-1] if preferred else None,
        "final_margin": float(margins[-1]),
        "final_candidate_preferred": bool(margins[-1] > 0.0),
    }


def flat_trajectory_features(summary: dict[str, Any] | None) -> dict[str, float | None]:
    """Return one imputer-friendly feature schema for every public sample."""

    output: dict[str, float | None] = {
        "candidate_available": 1.0 if summary is not None else 0.0
    }
    for key in TRAJECTORY_SUMMARY_KEYS:
        if summary is None:
            output[key] = None
            continue
        value = summary[key]
        # Layer indices are one-based, so zero is an unambiguous numeric sentinel
        # when a scored trajectory never gives the candidate a positive margin.
        if value is None:
            value = 0.0
        output[key] = float(value)
    return output


def score_source_record(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    final_nll_tolerance: float,
    source_replay: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "sample_id": row["sample_id"],
        "model": row["model"],
        "dataset": row["dataset"],
        "row_id": row["row_id"],
        "fact_key_sha256": row["fact_key_sha256"],
        "split": row["split"],
        "label": row["label"],
        "confidence": row["confidence"],
        "question": row["question"],
        "canonical_answer": row["canonical_answer"],
        "candidate_available": row["candidate_available"],
        "candidate_answer": row["candidate_answer"],
        "candidate_source": row["candidate_source"],
        "candidate_cluster": row["candidate_cluster"],
        "candidate_panel_index": row["candidate_panel_index"],
    }
    if not row["candidate_available"]:
        return {
            **base,
            "status": "no_frozen_nli_alternative",
            "features": flat_trajectory_features(None),
            "trajectory": None,
        }

    prompt_ids = chat_ids(tokenizer, row["question"])
    canonical_ids, canonical_alignment = aligned_answer_ids(
        tokenizer,
        row["canonical_answer"],
        row["canonical_generated_token_ids"],
        row["sample_id"],
        "canonical",
    )
    candidate_ids, candidate_alignment = aligned_answer_ids(
        tokenizer,
        row["candidate_answer"],
        row["candidate_generated_token_ids"],
        row["sample_id"],
        "candidate",
    )
    canonical = score_answer_trajectory(
        model, prompt_ids, canonical_ids, final_nll_tolerance
    )
    candidate = score_answer_trajectory(
        model, prompt_ids, candidate_ids, final_nll_tolerance
    )
    if len(canonical["layer_mean_logprob"]) != len(candidate["layer_mean_logprob"]):
        raise RuntimeError(f"{row['sample_id']}: canonical/candidate layer counts differ")
    margins = [
        float(candidate_value - canonical_value)
        for canonical_value, candidate_value in zip(
            canonical["layer_mean_logprob"], candidate["layer_mean_logprob"]
        )
    ]
    actual_margin = float(
        candidate["actual_final_mean_logprob"]
        - canonical["actual_final_mean_logprob"]
    )
    final_margin_error = abs(margins[-1] - actual_margin)
    if final_margin_error > final_nll_tolerance:
        raise RuntimeError(
            f"{row['sample_id']}: final candidate margin reconstruction error "
            f"{final_margin_error:.6g} exceeds {final_nll_tolerance:.6g}"
        )
    single_vs_source_canonical = abs(
        canonical["actual_final_nll"] - float(row["source_canonical_nll"])
    )
    single_vs_source_candidate = abs(
        candidate["actual_final_nll"] - float(row["source_candidate_nll"])
    )
    single_vs_source_margin = abs(
        actual_margin - float(row["source_candidate_logscore_margin"])
    )
    summary = trajectory_summary(margins)
    return {
        **base,
        "status": "scored",
        "prompt_token_count": len(prompt_ids),
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "token_alignment": {
            "canonical": canonical_alignment,
            "candidate": candidate_alignment,
        },
        "canonical": canonical,
        "candidate": candidate,
        "features": flat_trajectory_features(summary),
        "trajectory": {
            "candidate_minus_canonical_mean_logprob_margin": margins,
            "actual_final_candidate_minus_canonical_margin": actual_margin,
            "final_margin_abs_error": final_margin_error,
            "summary": summary,
        },
        "source_consistency": {
            "source_canonical_nll": float(row["source_canonical_nll"]),
            "source_candidate_nll": float(row["source_candidate_nll"]),
            "source_candidate_logscore_margin": float(
                row["source_candidate_logscore_margin"]
            ),
            "exact_original_batch_replay": source_replay,
            "single_forward_vs_source_canonical_nll_abs_delta": single_vs_source_canonical,
            "single_forward_vs_source_candidate_nll_abs_delta": single_vs_source_candidate,
            "single_forward_vs_source_candidate_margin_abs_delta": single_vs_source_margin,
            "single_forward_delta_interpretation": (
                "BF16 batch-layout sensitivity; prompt/token identity is established "
                "by exact_original_batch_replay"
            ),
        },
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(part.strip() for part in value.split(",") if part.strip())
    if not splits or any(split not in PUBLIC_SPLITS for split in splits):
        raise argparse.ArgumentTypeError(
            "splits must be development, validation, or development,validation"
        )
    if len(set(splits)) != len(splits):
        raise argparse.ArgumentTypeError("duplicate split")
    return splits


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--nli", type=Path, required=True)
    parser.add_argument("--candidate-margins", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=parse_splits, default=PUBLIC_SPLITS)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sanity-rows", type=int, default=20)
    parser.add_argument("--final-nll-tolerance", type=float, default=1e-4)
    parser.add_argument("--source-batch-rows", type=int, default=32)
    parser.add_argument("--source-replay-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.sanity_rows <= 0:
        parser.error("--sanity-rows must be positive")

    paths = (args.benchmark, args.features, args.nli, args.candidate_margins)
    for path in paths:
        reject_unsafe_path(path)
    benchmark, features, nli, margins = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    all_rows = build_source_records(
        benchmark,
        features,
        nli,
        margins,
        args.cell,
        splits=args.splits,
        candidate_only=False,
        limit=None,
    )
    rows = [
        row for row in all_rows if not args.candidate_only or row["candidate_available"]
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("no public rows selected")

    from transformers import AutoTokenizer

    from model_loading import load_causal_model

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    model = load_causal_model(
        args.model_path, load_in_4bit=False, torch_dtype=torch.bfloat16
    ).eval()
    if next(model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError("candidate trajectory extraction requires BF16 model weights")
    blocks, _, _ = transformer_components(model)

    selected_candidate_ids = {
        row["sample_id"] for row in rows if row["candidate_available"]
    }
    source_replay = replay_frozen_source_batches(
        model,
        tokenizer,
        all_rows,
        selected_candidate_ids,
        args.source_batch_rows,
        args.source_replay_tolerance,
    ) if selected_candidate_ids else {}

    started = time.time()
    records = []
    for index, row in enumerate(rows, start=1):
        records.append(
            score_source_record(
                model,
                tokenizer,
                row,
                args.final_nll_tolerance,
                source_replay.get(row["sample_id"], {}),
            )
        )
        print(
            f"{args.cell} {index}/{len(rows)} "
            f"scored={sum(record['status'] == 'scored' for record in records)} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    scored = [record for record in records if record["status"] == "scored"]
    sanity = scored[: args.sanity_rows]
    payload = {
        "schema_version": "one-confidence-candidate-sequence-trajectory-v1",
        "cell": args.cell,
        "locked_accessed": False,
        "gold_visible": False,
        "model_path": args.model_path,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "dtype": "bfloat16",
        "splits": list(args.splits),
        "layer_numbers": list(range(1, len(blocks) + 1)),
        "readout": (
            "final model norm plus LM head applied to each transformer block output; "
            "complete answer teacher forcing; answer-token mean log probability"
        ),
        "margin_definition": "score(frozen NLI candidate) - score(canonical generated answer)",
        "candidate_policy": (
            "candidate-margin artifact is authoritative; corrected NLI cluster and "
            "panel-generated token IDs must decode to the frozen candidate"
        ),
        "teacher_forced_tokenization_policy": (
            "standalone encode(answer), exactly matching the frozen candidate-margin "
            "scorer; context-dependent panel token IDs are provenance only"
        ),
        "flat_feature_names": ["candidate_available", *TRAJECTORY_SUMMARY_KEYS],
        "answer_system": ANSWER_SYSTEM,
        "qwen_thinking": False,
        "source_sha256": {
            "benchmark": sha256(args.benchmark),
            "features": sha256(args.features),
            "nli": sha256(args.nli),
            "candidate_margins": sha256(args.candidate_margins),
        },
        "selection": {
            "candidate_only": bool(args.candidate_only),
            "limit": args.limit,
            "selected_rows": len(rows),
            "scored_rows": len(scored),
            "no_candidate_rows": len(rows) - len(scored),
        },
        "sanity_audit": {
            "requested_rows": args.sanity_rows,
            "audited_scored_rows": len(sanity),
            "sample_ids": [record["sample_id"] for record in sanity],
            "all_teacher_and_frozen_decodes_match_answer": all(
                record["token_alignment"][answer]["teacher_decode_matches_answer"]
                and record["token_alignment"][answer][
                    "frozen_generated_decode_matches_answer"
                ]
                is not False
                for record in sanity
                for answer in ("canonical", "candidate")
            ),
            "standalone_vs_frozen_id_exact_count": sum(
                record["token_alignment"][answer][
                    "frozen_generated_token_ids_match"
                ]
                is True
                for record in sanity
                for answer in ("canonical", "candidate")
            ),
            "standalone_vs_frozen_id_mismatch_count": sum(
                record["token_alignment"][answer][
                    "context_dependent_tokenization_difference"
                ]
                for record in sanity
                for answer in ("canonical", "candidate")
            ),
            "standalone_vs_frozen_id_mismatch_sample_ids": sorted(
                {
                    record["sample_id"]
                    for record in sanity
                    for answer in ("canonical", "candidate")
                    if record["token_alignment"][answer][
                        "context_dependent_tokenization_difference"
                    ]
                }
            ),
            "max_final_layer_nll_abs_error": max(
                (
                    record[answer]["final_layer_nll_abs_error"]
                    for record in sanity
                    for answer in ("canonical", "candidate")
                ),
                default=0.0,
            ),
            "max_source_nll_or_margin_abs_error": max(
                (
                    max(
                        value
                        for key, value in record["source_consistency"][
                            "exact_original_batch_replay"
                        ].items()
                        if key.endswith("_abs_error")
                    )
                    for record in sanity
                ),
                default=0.0,
            ),
        },
        "records": records,
    }
    atomic_json(args.output, payload)
    print(f"saved {args.output} rows={len(records)} scored={len(scored)}", flush=True)


if __name__ == "__main__":
    main()
