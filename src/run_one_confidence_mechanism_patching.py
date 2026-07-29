#!/usr/bin/env python3
"""Run gold-free late-MLP patching on public One Confidence splits.

Layer selection is performed only on recoverable development HCE rows.  The
selected layer and a seeded random-layer control are then evaluated on public
validation HCE rows and correct-to-correct HCC controls.  Candidate answers are
read exclusively from the frozen NLI alternative clusters in the input
manifest; this runner never reads benchmark answer keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from extract_one_confidence_natural_paraphrase_features import ANSWER_SYSTEM
from model_loading import load_causal_model


ALLOWED_SPLITS = {"development", "validation"}


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbidden_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if "gold" in str(key).casefold():
                return str(key)
            nested = forbidden_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = forbidden_key(item)
            if nested is not None:
                return nested
    return None


def chat_ids(tokenizer: Any, question: str) -> list[int]:
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": question},
    ]
    try:
        return list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError:
        return list(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )


def output_tensor(output: Any) -> torch.Tensor:
    return output if torch.is_tensor(output) else output[0]


def replace_position(output: Any, position: int, replacement: torch.Tensor) -> Any:
    tensor = output_tensor(output)
    modified = tensor.clone()
    modified[:, position] = replacement.to(device=modified.device, dtype=modified.dtype)
    return modified if torch.is_tensor(output) else (modified, *output[1:])


def model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    return (
        torch.device(device) if device is not None else next(model.parameters()).device
    )


def padded_right(
    sequences: list[list[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(
            sequence, dtype=torch.long, device=device
        )
        mask[index, : len(sequence)] = 1
    return ids, mask


@torch.inference_mode()
def capture_prompt(
    model: Any,
    mlps: list[Any],
    token_ids: list[int],
    layers: list[int],
) -> dict[str, Any]:
    ids = torch.tensor(token_ids, dtype=torch.long, device=model_device(model))[None]
    captured: dict[int, torch.Tensor | None] = {layer: None for layer in layers}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured[layer] = output_tensor(output)[0, -1].detach().cpu()

        return hook

    for layer in layers:
        handles.append(mlps[layer].register_forward_hook(make_hook(layer)))
    try:
        logits = model(input_ids=ids, use_cache=False).logits[0, -1]
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captured.values()):
        raise RuntimeError("failed to capture every requested MLP output")
    return {
        "states": {layer: captured[layer] for layer in layers},
        "answer_start_logits": logits.detach().float().cpu(),
    }


@torch.inference_mode()
def score_answers(
    model: Any,
    mlp: Any,
    prompt_ids: list[int],
    answer_ids: dict[str, list[int]],
    pad_id: int,
    replacement: torch.Tensor | None = None,
    exact_noop: bool = False,
) -> dict[str, Any]:
    if not answer_ids or any(not values for values in answer_ids.values()):
        raise ValueError("every scored answer must contain at least one token")
    names = list(answer_ids)
    sequences = [prompt_ids + answer_ids[name] for name in names]
    ids, mask = padded_right(sequences, pad_id, model_device(model))
    if replacement is not None and exact_noop:
        raise ValueError("replacement and exact_noop are mutually exclusive")
    handle = None
    if replacement is not None or exact_noop:
        position = len(prompt_ids) - 1

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if exact_noop:
                tensor = output_tensor(output)
                self_value = tensor[:, position].detach()
                return replace_position(output, position, self_value)
            return replace_position(output, position, replacement)

        handle = mlp.register_forward_hook(hook)
    try:
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    finally:
        if handle is not None:
            handle.remove()
    scores = {}
    start = len(prompt_ids) - 1
    for row_index, name in enumerate(names):
        tokens = answer_ids[name]
        answer_logits = logits[row_index, start : start + len(tokens)].float()
        targets = torch.tensor(tokens, dtype=torch.long, device=answer_logits.device)
        scores[name] = float(
            torch.log_softmax(answer_logits, dim=-1).gather(1, targets[:, None]).mean()
        )
    return {
        "scores": scores,
        "answer_start_logits": logits[0, start].detach().float().cpu(),
    }


def random_replacement(
    recipient: torch.Tensor, donor: torch.Tensor, seed: int
) -> torch.Tensor:
    displacement = donor.float() - recipient.float()
    norm = displacement.norm()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    direction = torch.randn(recipient.shape, generator=generator)
    direction = direction / direction.norm().clamp_min(1e-12)
    return (recipient.float() + direction * norm).to(recipient.dtype)


def distribution_js(left_logits: torch.Tensor, right_logits: torch.Tensor) -> float:
    left = torch.softmax(left_logits.float(), dim=-1)
    right = torch.softmax(right_logits.float(), dim=-1)
    middle = (left + right) / 2
    left_kl = (
        left * (left.clamp_min(1e-30).log() - middle.clamp_min(1e-30).log())
    ).sum()
    right_kl = (
        right * (right.clamp_min(1e-30).log() - middle.clamp_min(1e-30).log())
    ).sum()
    return float((left_kl + right_kl) / 2)


def distribution_kl(base_logits: torch.Tensor, shifted_logits: torch.Tensor) -> float:
    base = torch.softmax(base_logits.float(), dim=-1)
    shifted = torch.softmax(shifted_logits.float(), dim=-1)
    return float(
        (base * (base.clamp_min(1e-30).log() - shifted.clamp_min(1e-30).log())).sum()
    )


def margin(result: dict[str, Any], candidate: str = "candidate") -> float:
    return float(result["scores"][candidate] - result["scores"]["canonical"])


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def deterministic_subset(
    rows: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: row["sample_id"])
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    rng = random.Random(seed)
    return sorted(rng.sample(ordered, limit), key=lambda row: row["sample_id"])


def summarize_hce(
    records: list[dict[str, Any]], layers: list[int]
) -> list[dict[str, Any]]:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for row in record["layers"]:
            by_layer[int(row["layer"])].append(row)
    summary = []
    for layer in layers:
        rows = by_layer[layer]
        aligned = [float(row["aligned_effect"]) for row in rows]
        random_effect = [float(row["random_direction_effect_mean"]) for row in rows]
        unrelated = [float(row["unrelated_effect"]) for row in rows]
        shuffled = [float(row["shuffled_candidate_effect"]) for row in rows]
        primary = [
            value - max(random_value, unrelated_value, shuffled_value)
            for value, random_value, unrelated_value, shuffled_value in zip(
                aligned, random_effect, unrelated, shuffled
            )
        ]
        summary.append(
            {
                "layer": layer,
                "n": len(rows),
                "aligned_effect_mean": safe_mean(aligned),
                "aligned_positive_fraction": safe_mean(
                    [float(value > 0) for value in aligned]
                ),
                "reverse_effect_mean": safe_mean(
                    [float(row["reverse_effect"]) for row in rows]
                ),
                "bidirectional_fraction": safe_mean(
                    [
                        float(row["aligned_effect"] > 0 and row["reverse_effect"] < 0)
                        for row in rows
                    ]
                ),
                "random_direction_effect_mean": safe_mean(random_effect),
                "unrelated_effect_mean": safe_mean(unrelated),
                "shuffled_candidate_effect_mean": safe_mean(shuffled),
                "aligned_minus_random_mean": safe_mean(
                    [left - right for left, right in zip(aligned, random_effect)]
                ),
                "aligned_minus_unrelated_mean": safe_mean(
                    [left - right for left, right in zip(aligned, unrelated)]
                ),
                "aligned_minus_shuffled_mean": safe_mean(
                    [left - right for left, right in zip(aligned, shuffled)]
                ),
                "primary_control_adjusted_effect_mean": safe_mean(primary),
                "answer_start_js_reduction_mean": safe_mean(
                    [float(row["answer_start_js_reduction"]) for row in rows]
                ),
                "max_abs_noop_effect": max(
                    (abs(float(row["noop_effect"])) for row in rows), default=0.0
                ),
            }
        )
    return summary


def summarize_hcc(
    records: list[dict[str, Any]], layers: list[int]
) -> list[dict[str, Any]]:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for row in record["layers"]:
            by_layer[int(row["layer"])].append(row)
    output = []
    for layer in layers:
        rows = by_layer[layer]
        output.append(
            {
                "layer": layer,
                "n": len(rows),
                "aligned_canonical_score_change_mean": safe_mean(
                    [float(row["aligned_canonical_score_change"]) for row in rows]
                ),
                "aligned_answer_start_kl_mean": safe_mean(
                    [float(row["aligned_answer_start_kl"]) for row in rows]
                ),
                "random_direction_score_change_mean": safe_mean(
                    [float(row["random_direction_score_change_mean"]) for row in rows]
                ),
                "unrelated_score_change_mean": safe_mean(
                    [float(row["unrelated_canonical_score_change"]) for row in rows]
                ),
                "max_abs_noop_score_change": max(
                    (abs(float(row["noop_canonical_score_change"])) for row in rows),
                    default=0.0,
                ),
            }
        )
    return output


def tokenize_answers(
    tokenizer: Any, row: dict[str, Any], row_by_id: dict[str, dict[str, Any]]
) -> dict[str, list[int]]:
    answers = {
        "canonical": str(row["canonical_answer"]),
        "candidate": str(row["candidate_answer"]),
    }
    shuffled_id = row.get("shuffled_candidate_sample_id")
    if not shuffled_id:
        raise ValueError(f"{row['sample_id']}: missing shuffled candidate control")
    answers["shuffled"] = str(row_by_id[shuffled_id]["candidate_answer"])
    tokenized = {
        name: list(tokenizer.encode(answer, add_special_tokens=False))
        for name, answer in answers.items()
    }
    if any(not values for values in tokenized.values()):
        raise ValueError(f"{row['sample_id']}: empty answer tokenization")
    return tokenized


def evaluate_hce(
    model: Any,
    tokenizer: Any,
    mlps: list[Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    layers: list[int],
    random_controls: int,
    seed: int,
) -> list[dict[str, Any]]:
    records = []
    pad_id = int(tokenizer.pad_token_id)
    for row_index, row in enumerate(rows):
        canonical_prompt = cache[row["sample_id"]]["canonical_prompt_ids"]
        donor_prompt = cache[row["sample_id"]]["donor_prompt_ids"]
        answers = tokenize_answers(tokenizer, row, row_by_id)
        base = score_answers(model, mlps[0], canonical_prompt, answers, pad_id)
        donor_base = score_answers(model, mlps[0], donor_prompt, answers, pad_id)
        base_margin = margin(base)
        donor_margin = margin(donor_base)
        base_shuffled_margin = margin(base, "shuffled")
        baseline_js = distribution_js(
            base["answer_start_logits"], donor_base["answer_start_logits"]
        )
        unrelated_id = row.get("unrelated_sample_id")
        if unrelated_id is None:
            raise ValueError(f"{row['sample_id']}: missing unrelated-fact control")
        layer_rows = []
        for layer in layers:
            canonical_state = cache[row["sample_id"]]["canonical"]["states"][layer]
            donor_state = cache[row["sample_id"]]["donor"]["states"][layer]
            unrelated_state = cache[unrelated_id]["donor"]["states"][layer]
            aligned = score_answers(
                model, mlps[layer], canonical_prompt, answers, pad_id, donor_state
            )
            reverse = score_answers(
                model, mlps[layer], donor_prompt, answers, pad_id, canonical_state
            )
            noop = score_answers(
                model,
                mlps[layer],
                canonical_prompt,
                answers,
                pad_id,
                exact_noop=True,
            )
            unrelated = score_answers(
                model, mlps[layer], canonical_prompt, answers, pad_id, unrelated_state
            )
            random_effects = []
            for control_index in range(random_controls):
                replacement = random_replacement(
                    canonical_state,
                    donor_state,
                    seed + row_index * 100000 + layer * 100 + control_index,
                )
                random_result = score_answers(
                    model,
                    mlps[layer],
                    canonical_prompt,
                    answers,
                    pad_id,
                    replacement,
                )
                random_effects.append(margin(random_result) - base_margin)
            aligned_effect = margin(aligned) - base_margin
            layer_rows.append(
                {
                    "layer": layer,
                    "baseline_candidate_minus_canonical_margin": base_margin,
                    "donor_candidate_minus_canonical_margin": donor_margin,
                    "aligned_candidate_minus_canonical_margin": margin(aligned),
                    "aligned_effect": aligned_effect,
                    "reverse_effect": margin(reverse) - donor_margin,
                    "noop_effect": margin(noop) - base_margin,
                    "unrelated_effect": margin(unrelated) - base_margin,
                    "random_direction_effects": random_effects,
                    "random_direction_effect_mean": safe_mean(random_effects),
                    "shuffled_candidate_effect": margin(aligned, "shuffled")
                    - base_shuffled_margin,
                    "answer_start_js_baseline": baseline_js,
                    "answer_start_js_patched_to_donor": distribution_js(
                        aligned["answer_start_logits"],
                        donor_base["answer_start_logits"],
                    ),
                    "answer_start_js_reduction": baseline_js
                    - distribution_js(
                        aligned["answer_start_logits"],
                        donor_base["answer_start_logits"],
                    ),
                    "answer_start_mi_reduction": baseline_js
                    - distribution_js(
                        aligned["answer_start_logits"],
                        donor_base["answer_start_logits"],
                    ),
                    "aligned_answer_start_top_token": int(
                        aligned["answer_start_logits"].argmax()
                    ),
                    "baseline_answer_start_top_token": int(
                        base["answer_start_logits"].argmax()
                    ),
                    "donor_answer_start_top_token": int(
                        donor_base["answer_start_logits"].argmax()
                    ),
                    "mlp_displacement_norm": float(
                        (donor_state.float() - canonical_state.float()).norm()
                    ),
                }
            )
        records.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": "HCE",
                "candidate_answer": row["candidate_answer"],
                "canonical_answer": row["canonical_answer"],
                "donor_answer": row["donor_answer"],
                "unrelated_sample_id": unrelated_id,
                "shuffled_candidate_sample_id": row["shuffled_candidate_sample_id"],
                "layers": layer_rows,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "hce",
                    "completed": row_index + 1,
                    "total": len(rows),
                }
            ),
            flush=True,
        )
    return records


def evaluate_hcc(
    model: Any,
    tokenizer: Any,
    mlps: list[Any],
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    layers: list[int],
    random_controls: int,
    seed: int,
) -> list[dict[str, Any]]:
    records = []
    pad_id = int(tokenizer.pad_token_id)
    for row_index, row in enumerate(rows):
        answer_ids = {
            "canonical": list(
                tokenizer.encode(row["canonical_answer"], add_special_tokens=False)
            )
        }
        canonical_prompt = cache[row["sample_id"]]["canonical_prompt_ids"]
        donor_prompt = cache[row["sample_id"]]["donor_prompt_ids"]
        base = score_answers(model, mlps[0], canonical_prompt, answer_ids, pad_id)
        donor_base = score_answers(model, mlps[0], donor_prompt, answer_ids, pad_id)
        unrelated_id = row.get("unrelated_sample_id")
        if unrelated_id is None:
            raise ValueError(f"{row['sample_id']}: missing unrelated-fact control")
        layer_rows = []
        for layer in layers:
            canonical_state = cache[row["sample_id"]]["canonical"]["states"][layer]
            donor_state = cache[row["sample_id"]]["donor"]["states"][layer]
            unrelated_state = cache[unrelated_id]["donor"]["states"][layer]
            aligned = score_answers(
                model, mlps[layer], canonical_prompt, answer_ids, pad_id, donor_state
            )
            reverse = score_answers(
                model, mlps[layer], donor_prompt, answer_ids, pad_id, canonical_state
            )
            noop = score_answers(
                model,
                mlps[layer],
                canonical_prompt,
                answer_ids,
                pad_id,
                exact_noop=True,
            )
            unrelated = score_answers(
                model,
                mlps[layer],
                canonical_prompt,
                answer_ids,
                pad_id,
                unrelated_state,
            )
            random_changes = []
            for control_index in range(random_controls):
                replacement = random_replacement(
                    canonical_state,
                    donor_state,
                    seed + 900000000 + row_index * 100000 + layer * 100 + control_index,
                )
                random_result = score_answers(
                    model,
                    mlps[layer],
                    canonical_prompt,
                    answer_ids,
                    pad_id,
                    replacement,
                )
                random_changes.append(
                    random_result["scores"]["canonical"] - base["scores"]["canonical"]
                )
            layer_rows.append(
                {
                    "layer": layer,
                    "aligned_canonical_score_change": aligned["scores"]["canonical"]
                    - base["scores"]["canonical"],
                    "reverse_canonical_score_change": reverse["scores"]["canonical"]
                    - donor_base["scores"]["canonical"],
                    "noop_canonical_score_change": noop["scores"]["canonical"]
                    - base["scores"]["canonical"],
                    "unrelated_canonical_score_change": unrelated["scores"]["canonical"]
                    - base["scores"]["canonical"],
                    "random_direction_score_changes": random_changes,
                    "random_direction_score_change_mean": safe_mean(random_changes),
                    "aligned_answer_start_kl": distribution_kl(
                        base["answer_start_logits"], aligned["answer_start_logits"]
                    ),
                    "aligned_answer_start_js_to_donor_reduction": distribution_js(
                        base["answer_start_logits"], donor_base["answer_start_logits"]
                    )
                    - distribution_js(
                        aligned["answer_start_logits"],
                        donor_base["answer_start_logits"],
                    ),
                    "mlp_displacement_norm": float(
                        (donor_state.float() - canonical_state.float()).norm()
                    ),
                }
            )
        records.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": "HCC",
                "canonical_answer": row["canonical_answer"],
                "donor_answer": row["donor_answer"],
                "unrelated_sample_id": unrelated_id,
                "layers": layer_rows,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "hcc",
                    "completed": row_index + 1,
                    "total": len(rows),
                }
            ),
            flush=True,
        )
    return records


@torch.inference_mode()
def generate_answer(
    model: Any,
    tokenizer: Any,
    mlp: Any,
    prompt_ids: list[int],
    replacement: torch.Tensor | None,
    max_new_tokens: int,
) -> str:
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=model_device(model))[None]
    handle = None
    if replacement is not None:
        patched = False

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            nonlocal patched
            if patched:
                return output
            patched = True
            return replace_position(output, len(prompt_ids) - 1, replacement)

        handle = mlp.register_forward_hook(hook)
    try:
        generated = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0, len(prompt_ids) :]
    finally:
        if handle is not None:
            handle.remove()
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def add_validation_generations(
    model: Any,
    tokenizer: Any,
    mlps: list[Any],
    rows: list[dict[str, Any]],
    row_by_id: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    selected_layer: int,
    random_layer: int,
    seed: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        sample_id = row["sample_id"]
        canonical_state = cache[sample_id]["canonical"]["states"][selected_layer]
        donor_state = cache[sample_id]["donor"]["states"][selected_layer]
        random_layer_donor = cache[sample_id]["donor"]["states"][random_layer]
        unrelated_id = row["unrelated_sample_id"]
        unrelated_state = cache[unrelated_id]["donor"]["states"][selected_layer]
        random_state = random_replacement(
            canonical_state, donor_state, seed + 700000000 + index
        )
        prompt = cache[sample_id]["canonical_prompt_ids"]
        donor_prompt = cache[sample_id]["donor_prompt_ids"]
        generations = {
            "canonical_unpatched": generate_answer(
                model, tokenizer, mlps[selected_layer], prompt, None, max_new_tokens
            ),
            "donor_unpatched": generate_answer(
                model,
                tokenizer,
                mlps[selected_layer],
                donor_prompt,
                None,
                max_new_tokens,
            ),
            "aligned_selected_layer": generate_answer(
                model,
                tokenizer,
                mlps[selected_layer],
                prompt,
                donor_state,
                max_new_tokens,
            ),
            "aligned_random_layer": generate_answer(
                model,
                tokenizer,
                mlps[random_layer],
                prompt,
                random_layer_donor,
                max_new_tokens,
            ),
            "norm_matched_random_direction": generate_answer(
                model,
                tokenizer,
                mlps[selected_layer],
                prompt,
                random_state,
                max_new_tokens,
            ),
            "unrelated_fact_donor": generate_answer(
                model,
                tokenizer,
                mlps[selected_layer],
                prompt,
                unrelated_state,
                max_new_tokens,
            ),
            "reverse_selected_layer": generate_answer(
                model,
                tokenizer,
                mlps[selected_layer],
                donor_prompt,
                canonical_state,
                max_new_tokens,
            ),
        }
        candidate = row.get("candidate_answer")
        output.append(
            {
                "sample_id": sample_id,
                "split": row["split"],
                "label": row["label"],
                "canonical_answer": row["canonical_answer"],
                "candidate_answer": candidate,
                "generations": generations,
                "canonical_exact_match": {
                    name: normalize(answer) == normalize(row["canonical_answer"])
                    for name, answer in generations.items()
                },
                "candidate_exact_match": {
                    name: bool(candidate)
                    and normalize(answer) == normalize(str(candidate))
                    for name, answer in generations.items()
                },
            }
        )
        print(
            json.dumps(
                {
                    "stage": "generation",
                    "completed": index + 1,
                    "total": len(rows),
                }
            ),
            flush=True,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--random-controls", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026072903)
    parser.add_argument("--hcc-controls-per-split", type=int, default=40)
    parser.add_argument("--limit-hce-per-split", type=int, default=0)
    parser.add_argument("--generate-selected", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--allow-unverified-manifest", action="store_true")
    args = parser.parse_args()
    if args.random_controls < 1:
        raise ValueError("random-controls must be positive")
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    if len(set(layers)) < 2:
        raise ValueError(
            "at least two distinct layers are needed for random-layer control"
        )
    layers = sorted(set(layers))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("locked_accessed"):
        raise ValueError("patch runner accepts public manifests only")
    if (
        manifest.get("question_equivalence") == "unverified diagnostic override"
        and not args.allow_unverified_manifest
    ):
        raise ValueError(
            "refusing unverified rewrites; provide an audited manifest or explicitly mark this run diagnostic"
        )
    leaked = forbidden_key(manifest)
    if leaked is not None:
        raise ValueError(f"manifest contains a forbidden answer-key field: {leaked}")
    records = manifest["records"]
    if any(row["split"] not in ALLOWED_SPLITS for row in records):
        raise ValueError("manifest contains a non-public split")
    row_by_id = {row["sample_id"]: row for row in records}
    if len(row_by_id) != len(records):
        raise ValueError("duplicate sample IDs in patch manifest")

    selected: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for split_index, split in enumerate(("development", "validation")):
        hce = [
            row for row in records if row["split"] == split and row["label"] == "HCE"
        ]
        hcc = [
            row for row in records if row["split"] == split and row["label"] == "HCC"
        ]
        selected[(split, "HCE")] = deterministic_subset(
            hce, args.limit_hce_per_split, args.seed + split_index
        )
        selected[(split, "HCC")] = deterministic_subset(
            hcc, args.hcc_controls_per_split, args.seed + 100 + split_index
        )
    if not selected[("development", "HCE")]:
        raise ValueError("no development HCE rows are available for layer selection")
    if not selected[("validation", "HCE")]:
        raise ValueError("no validation HCE rows are available for confirmation")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = load_causal_model(
        args.model_path, load_in_4bit=False, torch_dtype=torch.bfloat16
    ).eval()
    blocks = model.model.layers
    if min(layers) < 0 or max(layers) >= len(blocks):
        raise ValueError(f"layers must be within 0..{len(blocks) - 1}")
    mlps = [block.mlp for block in blocks]
    chat_template = tokenizer.chat_template or ""

    needed_ids = {row["sample_id"] for rows in selected.values() for row in rows}
    for sample_id in list(needed_ids):
        row = row_by_id[sample_id]
        for field in ("unrelated_sample_id", "shuffled_candidate_sample_id"):
            if row.get(field):
                needed_ids.add(row[field])
    missing = needed_ids - set(row_by_id)
    if missing:
        raise ValueError(
            f"control references are missing from manifest: {sorted(missing)}"
        )

    cache: dict[str, dict[str, Any]] = {}
    started = time.time()
    for index, sample_id in enumerate(sorted(needed_ids)):
        row = row_by_id[sample_id]
        canonical_ids = chat_ids(tokenizer, row["question"])
        donor_ids = chat_ids(tokenizer, row["donor_paraphrase"])
        cache[sample_id] = {
            "canonical_prompt_ids": canonical_ids,
            "donor_prompt_ids": donor_ids,
            "canonical": capture_prompt(model, mlps, canonical_ids, layers),
            "donor": capture_prompt(model, mlps, donor_ids, layers),
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

    development_scan = evaluate_hce(
        model,
        tokenizer,
        mlps,
        selected[("development", "HCE")],
        row_by_id,
        cache,
        layers,
        args.random_controls,
        args.seed,
    )
    development_summary = summarize_hce(development_scan, layers)
    selected_row = max(
        development_summary,
        key=lambda row: (
            float(row["primary_control_adjusted_effect_mean"]),
            -int(row["layer"]),
        ),
    )
    selected_layer = int(selected_row["layer"])
    candidate_random_layers = [layer for layer in layers if layer != selected_layer]
    random_layer = random.Random(args.seed + 500000000).choice(candidate_random_layers)

    validation_hce = evaluate_hce(
        model,
        tokenizer,
        mlps,
        selected[("validation", "HCE")],
        row_by_id,
        cache,
        [selected_layer, random_layer],
        args.random_controls,
        args.seed + 10000000,
    )
    hcc_rows = [
        *selected[("development", "HCC")],
        *selected[("validation", "HCC")],
    ]
    hcc_controls = evaluate_hcc(
        model,
        tokenizer,
        mlps,
        hcc_rows,
        cache,
        [selected_layer, random_layer],
        args.random_controls,
        args.seed,
    )
    generations = []
    if args.generate_selected:
        generations = add_validation_generations(
            model,
            tokenizer,
            mlps,
            [
                *selected[("validation", "HCE")],
                *selected[("validation", "HCC")],
            ],
            row_by_id,
            cache,
            selected_layer,
            random_layer,
            args.seed,
            args.max_new_tokens,
        )

    result = {
        "schema_version": "one-confidence-public-late-mlp-patching-v1",
        "protocol": {
            "cell": manifest["cell"],
            "model_path": args.model_path,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "tokenizer_revision": getattr(tokenizer, "_commit_hash", None),
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
            "dtype": "bfloat16",
            "component": "MLP output at the final prompt position",
            "candidate_source": "NLI alternative cluster; no answer key is read",
            "selection": "maximize development mean aligned candidate-margin effect minus strongest per-row random-direction, unrelated-donor, or shuffled-candidate control",
            "confirmation": "frozen selected layer on public validation",
            "controls": [
                "exact no-op",
                "seeded random layer",
                "norm-matched random direction",
                "unrelated-fact donor",
                "shuffled candidate",
                "reverse patch",
                "HCC correct-to-correct patch",
            ],
            "answer_start_mi_note": "with the canonical/donor two-prompt panel, equal-weight MI equals JS",
            "random_controls": args.random_controls,
            "seed": args.seed,
            "locked_accessed": False,
            "manifest_sha256": sha256(args.manifest),
            "scientific_status": "development layer selection plus public validation confirmation; whole-MLP-output transport is localization evidence, not a unique circuit claim",
        },
        "sample_counts": {
            f"{split}_{label.lower()}": len(selected[(split, label)])
            for split in ("development", "validation")
            for label in ("HCE", "HCC")
        },
        "layer_selection": {
            "selected_layer": selected_layer,
            "random_control_layer": random_layer,
            "development_selected_row": selected_row,
        },
        "development_scan_summary": development_summary,
        "development_scan_records": development_scan,
        "validation_hce_summary": summarize_hce(
            validation_hce, [selected_layer, random_layer]
        ),
        "validation_hce_records": validation_hce,
        "hcc_control_summary": {
            split: summarize_hcc(
                [row for row in hcc_controls if row["split"] == split],
                [selected_layer, random_layer],
            )
            for split in ("development", "validation")
        },
        "hcc_control_records": hcc_controls,
        "validation_generations": generations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_layer": selected_layer,
                "random_control_layer": random_layer,
                "validation_hce": len(validation_hce),
                "hcc_controls": len(hcc_controls),
                "elapsed": time.time() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
