#!/usr/bin/env python3
"""Run development-derived linear direction erasure on public mechanism rows.

The erased direction is fitted only from paired canonical/donor MLP outputs of
development HCE rows at the layer already frozen by the public patching run.
Validation is evaluation-only.  The runner rejects locked splits, answer-key
fields, stale source hashes, and any attempt to select a new layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from model_loading import load_causal_model
from run_one_confidence_mechanism_patching import (
    ALLOWED_SPLITS,
    capture_prompt,
    chat_ids,
    distribution_js,
    distribution_kl,
    forbidden_key,
    model_device,
    output_tensor,
    padded_right,
    sha256,
)


ERASURE_SEED = 2026072904


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def fit_paired_direction(
    canonical_states: list[torch.Tensor], donor_states: list[torch.Tensor]
) -> dict[str, Any]:
    """Fit an equal-row-weighted paired displacement and pooled midpoint.

    Normalizing each displacement before averaging is essential here.  The
    frozen unrelated control is a derangement of the same donor-state set, so a
    raw difference-of-means would be exactly invariant to that control and
    would not constitute an independent direction.
    """

    if len(canonical_states) != len(donor_states) or not canonical_states:
        raise ValueError("paired direction requires equal non-empty state lists")
    canonical = torch.stack([state.detach().float().cpu() for state in canonical_states])
    donor = torch.stack([state.detach().float().cpu() for state in donor_states])
    if canonical.shape != donor.shape or canonical.ndim != 2:
        raise ValueError("states must be paired rank-two tensors")
    differences = donor - canonical
    pair_norms = differences.norm(dim=1)
    if bool((pair_norms <= 1e-8).any()):
        raise ValueError("paired direction contains a degenerate displacement")
    unit_differences = differences / pair_norms[:, None]
    raw_direction = unit_differences.mean(dim=0)
    raw_norm = raw_direction.norm()
    if not torch.isfinite(raw_norm) or float(raw_norm) <= 1e-8:
        raise ValueError("mean normalized paired displacement is degenerate")
    direction = raw_direction / raw_norm
    if float((differences @ direction).mean()) < 0:
        direction = -direction
    canonical_mean = canonical.mean(dim=0)
    donor_mean = donor.mean(dim=0)
    center = (canonical_mean + donor_mean) / 2
    signed_displacements = differences @ direction
    cosines = signed_displacements / pair_norms.clamp_min(1e-12)
    return {
        "direction": direction,
        "center": center,
        "statistics": {
            "n_pairs": len(canonical_states),
            "hidden_size": int(direction.numel()),
            "raw_mean_unit_displacement_norm": float(raw_norm),
            "unweighted_mean_displacement_norm": float(
                differences.mean(dim=0).norm()
            ),
            "pair_displacement_norm_mean": float(pair_norms.mean()),
            "pair_displacement_norm_p50": float(pair_norms.median()),
            "paired_direction_cosine_mean": float(cosines.mean()),
            "positive_projection_fraction": float(
                (signed_displacements > 0).float().mean()
            ),
            "canonical_centroid_coordinate": float(
                (canonical_mean - center) @ direction
            ),
            "donor_centroid_coordinate": float((donor_mean - center) @ direction),
        },
    }


def random_unit_direction(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    return direction / direction.norm().clamp_min(1e-12)


def project_orthogonal(
    values: torch.Tensor, direction: torch.Tensor, center: torch.Tensor
) -> torch.Tensor:
    """Remove one centered linear direction from one state or a state batch."""

    original_dtype = values.dtype
    working = values.float()
    direction = direction.to(device=working.device, dtype=working.dtype)
    center = center.to(device=working.device, dtype=working.dtype)
    centered = working - center
    coefficients = centered @ direction
    projected = working - coefficients.unsqueeze(-1) * direction
    return projected.to(original_dtype)


def project_output_position(
    output: Any,
    position: int,
    direction: torch.Tensor | None = None,
    center: torch.Tensor | None = None,
    exact_noop: bool = False,
) -> Any:
    if exact_noop == (direction is not None):
        raise ValueError("choose exactly one of projection or exact no-op")
    if direction is not None and center is None:
        raise ValueError("projection requires a center")
    tensor = output_tensor(output)
    modified = tensor.clone()
    current = modified[:, position]
    replacement = (
        current.detach()
        if exact_noop
        else project_orthogonal(current, direction, center)  # type: ignore[arg-type]
    )
    modified[:, position] = replacement.to(modified.dtype)
    return modified if torch.is_tensor(output) else (modified, *output[1:])


@torch.inference_mode()
def score_answers(
    model: Any,
    mlp: Any,
    prompt_ids: list[int],
    answer_ids: dict[str, list[int]],
    pad_id: int,
    direction: torch.Tensor | None = None,
    center: torch.Tensor | None = None,
    exact_noop: bool = False,
) -> dict[str, Any]:
    """Score every complete answer sequence under an optional rank-one erasure."""

    if not answer_ids or any(not tokens for tokens in answer_ids.values()):
        raise ValueError("every scored answer must contain at least one token")
    if exact_noop and direction is not None:
        raise ValueError("direction and exact no-op are mutually exclusive")
    if direction is None and center is not None:
        raise ValueError("center without a direction is invalid")
    names = list(answer_ids)
    sequences = [prompt_ids + answer_ids[name] for name in names]
    input_ids, attention_mask = padded_right(
        sequences, pad_id, model_device(model)
    )
    handle = None
    if direction is not None or exact_noop:
        position = len(prompt_ids) - 1

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            return project_output_position(
                output,
                position,
                direction=direction,
                center=center,
                exact_noop=exact_noop,
            )

        handle = mlp.register_forward_hook(hook)
    try:
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
    finally:
        if handle is not None:
            handle.remove()
    start = len(prompt_ids) - 1
    scores: dict[str, float] = {}
    for row_index, name in enumerate(names):
        tokens = answer_ids[name]
        answer_logits = logits[row_index, start : start + len(tokens)].float()
        targets = torch.tensor(tokens, dtype=torch.long, device=answer_logits.device)
        token_log_probabilities = torch.log_softmax(answer_logits, dim=-1).gather(
            1, targets[:, None]
        )[:, 0]
        scores[name] = float(token_log_probabilities.mean())
    return {
        "scores": scores,
        "answer_start_logits": logits[0, start].detach().float().cpu(),
    }


def candidate_margin(result: dict[str, Any], candidate: str = "candidate") -> float:
    return float(result["scores"][candidate] - result["scores"]["canonical"])


def tokenize_hce_answers(
    tokenizer: Any, row: dict[str, Any], row_by_id: dict[str, dict[str, Any]]
) -> dict[str, list[int]]:
    shuffled_id = row.get("shuffled_candidate_sample_id")
    if not shuffled_id or shuffled_id not in row_by_id:
        raise ValueError(f"{row['sample_id']}: missing shuffled-candidate control")
    shuffled_answer = row_by_id[shuffled_id].get("candidate_answer")
    if not shuffled_answer:
        raise ValueError(f"{row['sample_id']}: shuffled control has no candidate")
    answers = {
        "canonical": str(row["canonical_answer"]),
        "candidate": str(row["candidate_answer"]),
        "shuffled": str(shuffled_answer),
    }
    tokenized = {
        name: list(tokenizer.encode(answer, add_special_tokens=False))
        for name, answer in answers.items()
    }
    if any(not tokens for tokens in tokenized.values()):
        raise ValueError(f"{row['sample_id']}: empty answer tokenization")
    return tokenized


def direction_metrics(
    canonical_baseline: dict[str, Any],
    donor_baseline: dict[str, Any],
    canonical_intervened: dict[str, Any],
    donor_intervened: dict[str, Any],
) -> dict[str, float]:
    baseline_js = distribution_js(
        canonical_baseline["answer_start_logits"],
        donor_baseline["answer_start_logits"],
    )
    donor_only_js = distribution_js(
        canonical_baseline["answer_start_logits"],
        donor_intervened["answer_start_logits"],
    )
    both_js = distribution_js(
        canonical_intervened["answer_start_logits"],
        donor_intervened["answer_start_logits"],
    )
    donor_margin_before = candidate_margin(donor_baseline)
    donor_margin_after = candidate_margin(donor_intervened)
    canonical_margin_before = candidate_margin(canonical_baseline)
    canonical_margin_after = candidate_margin(canonical_intervened)
    shuffled_margin_before = candidate_margin(donor_baseline, "shuffled")
    shuffled_margin_after = candidate_margin(donor_intervened, "shuffled")
    return {
        "canonical_candidate_minus_canonical_margin": canonical_margin_after,
        "canonical_candidate_margin_change": canonical_margin_after
        - canonical_margin_before,
        "donor_candidate_minus_canonical_margin": donor_margin_after,
        "donor_candidate_margin_change": donor_margin_after - donor_margin_before,
        "donor_candidate_margin_drop": donor_margin_before - donor_margin_after,
        "donor_candidate_score_change": float(
            donor_intervened["scores"]["candidate"]
            - donor_baseline["scores"]["candidate"]
        ),
        "donor_canonical_score_change": float(
            donor_intervened["scores"]["canonical"]
            - donor_baseline["scores"]["canonical"]
        ),
        "donor_shuffled_margin_change": shuffled_margin_after
        - shuffled_margin_before,
        "donor_shuffled_margin_drop": shuffled_margin_before
        - shuffled_margin_after,
        "answer_start_js_baseline": baseline_js,
        "answer_start_js_donor_erased": donor_only_js,
        "answer_start_js_both_erased": both_js,
        "answer_start_js_reduction_donor_only": baseline_js - donor_only_js,
        "answer_start_js_reduction_both": baseline_js - both_js,
    }


def evaluate_hce_rows(
    model: Any,
    tokenizer: Any,
    mlp: Any,
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    true_projection: dict[str, Any],
    unrelated_projection: dict[str, Any],
    random_directions: list[torch.Tensor],
) -> list[dict[str, Any]]:
    records = []
    pad_id = int(tokenizer.pad_token_id)
    for index, row in enumerate(rows):
        prompt = cache[row["sample_id"]]["canonical_prompt_ids"]
        donor_prompt = cache[row["sample_id"]]["donor_prompt_ids"]
        answers = tokenize_hce_answers(tokenizer, row, row_by_id)
        canonical_baseline = score_answers(model, mlp, prompt, answers, pad_id)
        donor_baseline = score_answers(model, mlp, donor_prompt, answers, pad_id)

        def intervention(direction: torch.Tensor, center: torch.Tensor) -> dict[str, Any]:
            canonical_result = score_answers(
                model, mlp, prompt, answers, pad_id, direction, center
            )
            donor_result = score_answers(
                model, mlp, donor_prompt, answers, pad_id, direction, center
            )
            return direction_metrics(
                canonical_baseline,
                donor_baseline,
                canonical_result,
                donor_result,
            )

        true_metrics = intervention(
            true_projection["direction"], true_projection["center"]
        )
        unrelated_metrics = intervention(
            unrelated_projection["direction"], unrelated_projection["center"]
        )
        random_metrics = [
            intervention(direction, true_projection["center"])
            for direction in random_directions
        ]
        canonical_noop = score_answers(
            model, mlp, prompt, answers, pad_id, exact_noop=True
        )
        donor_noop = score_answers(
            model, mlp, donor_prompt, answers, pad_id, exact_noop=True
        )
        noop_metrics = direction_metrics(
            canonical_baseline, donor_baseline, canonical_noop, donor_noop
        )
        records.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": "HCE",
                "candidate_answer": row["candidate_answer"],
                "canonical_answer": row["canonical_answer"],
                "shuffled_candidate_sample_id": row[
                    "shuffled_candidate_sample_id"
                ],
                "baseline": {
                    "canonical_candidate_minus_canonical_margin": candidate_margin(
                        canonical_baseline
                    ),
                    "donor_candidate_minus_canonical_margin": candidate_margin(
                        donor_baseline
                    ),
                    "donor_shuffled_minus_canonical_margin": candidate_margin(
                        donor_baseline, "shuffled"
                    ),
                    "answer_start_js": distribution_js(
                        canonical_baseline["answer_start_logits"],
                        donor_baseline["answer_start_logits"],
                    ),
                },
                "true_direction_erasure": true_metrics,
                "unrelated_direction_erasure": unrelated_metrics,
                "random_direction_erasures": random_metrics,
                "exact_noop": noop_metrics,
            }
        )
        print(
            json.dumps(
                {"stage": "validation_hce", "completed": index + 1, "total": len(rows)}
            ),
            flush=True,
        )
    return records


def hcc_intervention_metrics(
    baseline: dict[str, Any], intervened: dict[str, Any]
) -> dict[str, float]:
    return {
        "canonical_sequence_score": float(intervened["scores"]["canonical"]),
        "canonical_sequence_score_change": float(
            intervened["scores"]["canonical"] - baseline["scores"]["canonical"]
        ),
        "answer_start_kl_from_baseline": distribution_kl(
            baseline["answer_start_logits"], intervened["answer_start_logits"]
        ),
        "answer_start_js_from_baseline": distribution_js(
            baseline["answer_start_logits"], intervened["answer_start_logits"]
        ),
    }


def evaluate_hcc_rows(
    model: Any,
    tokenizer: Any,
    mlp: Any,
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    true_projection: dict[str, Any],
    unrelated_projection: dict[str, Any],
    random_directions: list[torch.Tensor],
) -> list[dict[str, Any]]:
    records = []
    pad_id = int(tokenizer.pad_token_id)
    for index, row in enumerate(rows):
        answer_ids = {
            "canonical": list(
                tokenizer.encode(row["canonical_answer"], add_special_tokens=False)
            )
        }
        if not answer_ids["canonical"]:
            raise ValueError(f"{row['sample_id']}: empty canonical tokenization")
        prompt = cache[row["sample_id"]]["canonical_prompt_ids"]
        baseline = score_answers(model, mlp, prompt, answer_ids, pad_id)
        true_result = score_answers(
            model,
            mlp,
            prompt,
            answer_ids,
            pad_id,
            true_projection["direction"],
            true_projection["center"],
        )
        unrelated_result = score_answers(
            model,
            mlp,
            prompt,
            answer_ids,
            pad_id,
            unrelated_projection["direction"],
            unrelated_projection["center"],
        )
        random_results = [
            score_answers(
                model,
                mlp,
                prompt,
                answer_ids,
                pad_id,
                direction,
                true_projection["center"],
            )
            for direction in random_directions
        ]
        noop = score_answers(
            model, mlp, prompt, answer_ids, pad_id, exact_noop=True
        )
        records.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": "HCC",
                "canonical_answer": row["canonical_answer"],
                "baseline_canonical_sequence_score": float(
                    baseline["scores"]["canonical"]
                ),
                "true_direction_erasure": hcc_intervention_metrics(
                    baseline, true_result
                ),
                "unrelated_direction_erasure": hcc_intervention_metrics(
                    baseline, unrelated_result
                ),
                "random_direction_erasures": [
                    hcc_intervention_metrics(baseline, result)
                    for result in random_results
                ],
                "exact_noop": hcc_intervention_metrics(baseline, noop),
            }
        )
        print(
            json.dumps(
                {"stage": "validation_hcc", "completed": index + 1, "total": len(rows)}
            ),
            flush=True,
        )
    return records


def summarize_hce(records: list[dict[str, Any]]) -> dict[str, Any]:
    true_drop = [
        float(row["true_direction_erasure"]["donor_candidate_margin_drop"])
        for row in records
    ]
    unrelated_drop = [
        float(row["unrelated_direction_erasure"]["donor_candidate_margin_drop"])
        for row in records
    ]
    shuffled_drop = [
        float(row["true_direction_erasure"]["donor_shuffled_margin_drop"])
        for row in records
    ]
    random_drop = [
        safe_mean(
            [
                float(control["donor_candidate_margin_drop"])
                for control in row["random_direction_erasures"]
            ]
        )
        or 0.0
        for row in records
    ]
    adjusted = [
        effect - max(random_value, unrelated_value, shuffled_value)
        for effect, random_value, unrelated_value, shuffled_value in zip(
            true_drop, random_drop, unrelated_drop, shuffled_drop
        )
    ]
    noop_changes = [
        abs(float(row["exact_noop"]["donor_candidate_margin_change"]))
        for row in records
    ]
    return {
        "n": len(records),
        "true_donor_candidate_margin_drop_mean": safe_mean(true_drop),
        "true_donor_candidate_margin_drop_p50": percentile(true_drop, 0.5),
        "true_donor_candidate_margin_drop_positive_fraction": safe_mean(
            [float(value > 0) for value in true_drop]
        ),
        "true_canonical_candidate_margin_change_mean": safe_mean(
            [
                float(
                    row["true_direction_erasure"][
                        "canonical_candidate_margin_change"
                    ]
                )
                for row in records
            ]
        ),
        "true_answer_start_js_reduction_donor_only_mean": safe_mean(
            [
                float(
                    row["true_direction_erasure"][
                        "answer_start_js_reduction_donor_only"
                    ]
                )
                for row in records
            ]
        ),
        "true_answer_start_js_reduction_both_mean": safe_mean(
            [
                float(
                    row["true_direction_erasure"][
                        "answer_start_js_reduction_both"
                    ]
                )
                for row in records
            ]
        ),
        "random_direction_margin_drop_mean": safe_mean(random_drop),
        "unrelated_direction_margin_drop_mean": safe_mean(unrelated_drop),
        "shuffled_candidate_margin_drop_mean": safe_mean(shuffled_drop),
        "control_adjusted_margin_drop_mean": safe_mean(adjusted),
        "control_adjusted_positive_fraction": safe_mean(
            [float(value > 0) for value in adjusted]
        ),
        "max_abs_noop_margin_change": max(noop_changes, default=0.0),
    }


def summarize_hcc(records: list[dict[str, Any]]) -> dict[str, Any]:
    true_score_changes = [
        float(row["true_direction_erasure"]["canonical_sequence_score_change"])
        for row in records
    ]
    random_score_changes = [
        safe_mean(
            [
                float(control["canonical_sequence_score_change"])
                for control in row["random_direction_erasures"]
            ]
        )
        or 0.0
        for row in records
    ]
    return {
        "n": len(records),
        "true_canonical_sequence_score_change_mean": safe_mean(true_score_changes),
        "true_canonical_sequence_score_change_p05": percentile(
            true_score_changes, 0.05
        ),
        "true_answer_start_kl_mean": safe_mean(
            [
                float(
                    row["true_direction_erasure"][
                        "answer_start_kl_from_baseline"
                    ]
                )
                for row in records
            ]
        ),
        "true_answer_start_js_mean": safe_mean(
            [
                float(
                    row["true_direction_erasure"][
                        "answer_start_js_from_baseline"
                    ]
                )
                for row in records
            ]
        ),
        "random_direction_sequence_score_change_mean": safe_mean(
            random_score_changes
        ),
        "unrelated_direction_sequence_score_change_mean": safe_mean(
            [
                float(
                    row["unrelated_direction_erasure"][
                        "canonical_sequence_score_change"
                    ]
                )
                for row in records
            ]
        ),
        "max_abs_noop_sequence_score_change": max(
            (
                abs(float(row["exact_noop"]["canonical_sequence_score_change"]))
                for row in records
            ),
            default=0.0,
        ),
    }


def public_rows(
    manifest: dict[str, Any], patch_result: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if manifest.get("locked_accessed") or patch_result.get("protocol", {}).get(
        "locked_accessed"
    ):
        raise ValueError("erasure runner accepts public inputs only")
    records = manifest.get("records", [])
    if not records or any(row.get("split") not in ALLOWED_SPLITS for row in records):
        raise ValueError("manifest contains empty or non-public records")
    development_hce = [
        row
        for row in records
        if row["split"] == "development" and row["label"] == "HCE"
    ]
    validation_hce = [
        row
        for row in records
        if row["split"] == "validation" and row["label"] == "HCE"
    ]
    validation_hcc = [
        row
        for row in records
        if row["split"] == "validation" and row["label"] == "HCC"
    ]
    if not development_hce or not validation_hce or not validation_hcc:
        raise ValueError("development HCE and validation HCE/HCC are required")
    frozen_development_ids = {
        row["sample_id"] for row in patch_result.get("development_scan_records", [])
    }
    frozen_validation_ids = {
        row["sample_id"] for row in patch_result.get("validation_hce_records", [])
    }
    if {row["sample_id"] for row in development_hce} != frozen_development_ids:
        raise ValueError("development HCE cohort differs from frozen patching run")
    if {row["sample_id"] for row in validation_hce} != frozen_validation_ids:
        raise ValueError("validation HCE cohort differs from frozen patching run")
    return development_hce, validation_hce, validation_hcc


def validate_inputs(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    patch_result: dict[str, Any],
    manifest_path: Path,
    patch_result_path: Path,
) -> dict[str, Any]:
    for path in (manifest_path, patch_result_path):
        if "invalidated" in str(path).casefold():
            raise ValueError("invalidated artifacts are forbidden")
    leaked = forbidden_key(manifest)
    if leaked is not None:
        raise ValueError(f"manifest contains forbidden answer-key field: {leaked}")
    leaked = forbidden_key(patch_result)
    if leaked is not None:
        raise ValueError(f"patch result contains forbidden answer-key field: {leaked}")
    cell = manifest.get("cell")
    if patch_result.get("protocol", {}).get("cell") != cell:
        raise ValueError("manifest and patch result cells differ")
    cell_protocol = protocol.get("cells", {}).get(cell)
    if cell_protocol is None:
        raise ValueError(f"protocol does not authorize cell {cell}")
    if sha256(manifest_path) != cell_protocol.get("manifest_sha256"):
        raise ValueError("manifest hash differs from erasure protocol")
    if sha256(patch_result_path) != cell_protocol.get("patch_result_sha256"):
        raise ValueError("patch result hash differs from erasure protocol")
    selected_layer = int(patch_result["layer_selection"]["selected_layer"])
    if selected_layer != int(cell_protocol.get("selected_layer")):
        raise ValueError("selected layer differs from frozen erasure protocol")
    public_rows(manifest, patch_result)
    return cell_protocol


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().float().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--patch-result", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction-output", type=Path, required=True)
    parser.add_argument("--random-controls", type=int, default=5)
    parser.add_argument("--seed", type=int, default=ERASURE_SEED)
    args = parser.parse_args()
    if args.random_controls < 1:
        raise ValueError("random-controls must be positive")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    patch_result = json.loads(args.patch_result.read_text(encoding="utf-8"))
    cell_protocol = validate_inputs(
        protocol, manifest, patch_result, args.manifest, args.patch_result
    )
    development_hce, validation_hce, validation_hcc = public_rows(
        manifest, patch_result
    )
    records = manifest["records"]
    row_by_id = {row["sample_id"]: row for row in records}
    if len(row_by_id) != len(records):
        raise ValueError("duplicate sample IDs in mechanism manifest")

    selected_layer = int(cell_protocol["selected_layer"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = load_causal_model(
        args.model_path, load_in_4bit=False, torch_dtype=torch.bfloat16
    ).eval()
    blocks = model.model.layers
    if selected_layer < 0 or selected_layer >= len(blocks):
        raise ValueError("frozen selected layer is outside the loaded model")
    mlp = blocks[selected_layer].mlp

    needed_ids = {
        row["sample_id"]
        for row in [*development_hce, *validation_hce, *validation_hcc]
    }
    for row in development_hce:
        unrelated_id = row.get("unrelated_sample_id")
        if not unrelated_id:
            raise ValueError(f"{row['sample_id']}: missing unrelated control")
        needed_ids.add(unrelated_id)
    missing = needed_ids - set(row_by_id)
    if missing:
        raise ValueError(f"manifest omits control rows: {sorted(missing)}")

    started = time.time()
    cache: dict[str, dict[str, Any]] = {}
    for index, sample_id in enumerate(sorted(needed_ids)):
        row = row_by_id[sample_id]
        canonical_ids = chat_ids(tokenizer, row["question"])
        donor_ids = chat_ids(tokenizer, row["donor_paraphrase"])
        cache[sample_id] = {
            "canonical_prompt_ids": canonical_ids,
            "donor_prompt_ids": donor_ids,
            "canonical": capture_prompt(
                model, [block.mlp for block in blocks], canonical_ids, [selected_layer]
            ),
            "donor": capture_prompt(
                model, [block.mlp for block in blocks], donor_ids, [selected_layer]
            ),
        }
        print(
            json.dumps(
                {
                    "stage": "capture",
                    "completed": index + 1,
                    "total": len(needed_ids),
                    "elapsed": time.time() - started,
                }
            ),
            flush=True,
        )

    canonical_states = [
        cache[row["sample_id"]]["canonical"]["states"][selected_layer]
        for row in development_hce
    ]
    donor_states = [
        cache[row["sample_id"]]["donor"]["states"][selected_layer]
        for row in development_hce
    ]
    unrelated_states = [
        cache[row["unrelated_sample_id"]]["donor"]["states"][selected_layer]
        for row in development_hce
    ]
    true_projection = fit_paired_direction(canonical_states, donor_states)
    unrelated_projection = fit_paired_direction(canonical_states, unrelated_states)
    true_unrelated_cosine = float(
        true_projection["direction"] @ unrelated_projection["direction"]
    )
    if abs(true_unrelated_cosine) >= 0.999999:
        raise RuntimeError(
            "unrelated-fact control direction is degenerate with the true direction"
        )
    random_directions = [
        random_unit_direction(
            true_projection["direction"], args.seed + 1000 + control_index
        )
        for control_index in range(args.random_controls)
    ]

    args.direction_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "one-confidence-public-linear-erasure-direction-v1",
            "cell": manifest["cell"],
            "selected_layer": selected_layer,
            "component": "MLP output at final prompt position",
            "development_sample_ids": sorted(
                row["sample_id"] for row in development_hce
            ),
            "true_direction": true_projection["direction"],
            "true_center": true_projection["center"],
            "unrelated_direction": unrelated_projection["direction"],
            "unrelated_center": unrelated_projection["center"],
            "random_directions": random_directions,
            "seed": args.seed,
        },
        args.direction_output,
    )

    hce_records = evaluate_hce_rows(
        model,
        tokenizer,
        mlp,
        validation_hce,
        row_by_id,
        cache,
        true_projection,
        unrelated_projection,
        random_directions,
    )
    hcc_records = evaluate_hcc_rows(
        model,
        tokenizer,
        mlp,
        validation_hcc,
        cache,
        true_projection,
        unrelated_projection,
        random_directions,
    )
    chat_template = tokenizer.chat_template or ""
    result = {
        "schema_version": "one-confidence-public-linear-erasure-v1",
        "protocol": {
            "cell": manifest["cell"],
            "model_path": args.model_path,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "tokenizer_revision": getattr(tokenizer, "_commit_hash", None),
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
            "dtype": "bfloat16",
            "selected_layer": selected_layer,
            "component": "MLP output at final prompt position",
            "direction_fit": "mean of unit-normalized paired donor-minus-canonical development HCE displacements",
            "projection": "rank-one centered orthogonal projection around development centroid midpoint",
            "validation_use": "evaluation only",
            "candidate_source": "frozen NLI alternative cluster; no answer key is read",
            "controls": [
                "exact no-op",
                "seeded random rank-one directions",
                "development-derived unrelated-fact direction",
                "frozen shuffled candidate",
                "validation HCC damage",
            ],
            "random_controls": args.random_controls,
            "seed": args.seed,
            "locked_accessed": False,
            "erasure_protocol_sha256": sha256(args.protocol),
            "manifest_sha256": sha256(args.manifest),
            "patch_result_sha256": sha256(args.patch_result),
            "direction_artifact": str(args.direction_output),
            "direction_artifact_sha256": sha256(args.direction_output),
            "scientific_status": "reverse validation of a development-localized public mechanism direction; not a detector and not a unique-circuit claim",
        },
        "sample_counts": {
            "development_direction_pairs": len(development_hce),
            "validation_hce": len(validation_hce),
            "validation_hcc": len(validation_hcc),
        },
        "direction_fit": {
            "true": true_projection["statistics"],
            "unrelated_control": unrelated_projection["statistics"],
            "true_direction_sha256": tensor_sha256(true_projection["direction"]),
            "true_center_sha256": tensor_sha256(true_projection["center"]),
            "unrelated_direction_sha256": tensor_sha256(
                unrelated_projection["direction"]
            ),
            "true_unrelated_direction_cosine": true_unrelated_cosine,
            "random_direction_cosines_to_true": [
                float(direction @ true_projection["direction"])
                for direction in random_directions
            ],
        },
        "validation_hce_summary": summarize_hce(hce_records),
        "validation_hce_records": hce_records,
        "validation_hcc_summary": summarize_hcc(hcc_records),
        "validation_hcc_records": hcc_records,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cell": manifest["cell"],
                "selected_layer": selected_layer,
                "validation_hce": len(hce_records),
                "validation_hcc": len(hcc_records),
                "output": str(args.output),
                "elapsed": time.time() - started,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
