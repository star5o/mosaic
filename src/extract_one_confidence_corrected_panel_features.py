#!/usr/bin/env python3
"""Extract gold-free features for an audited public question panel.

The input panel is produced by ``audit_one_confidence_question_equivalence.py``.
Rejected model rewrites are replaced by deterministic wrappers that retain the
source question verbatim.  Natural-recovery features are computed separately so
wrapper fallbacks cannot be mistaken for natural paraphrase evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from extract_one_confidence_natural_paraphrase_features import (
    ANSWER_SYSTEM,
    chat_ids,
    entropy,
    js_divergence,
    normalize,
    padded,
    teacher_forced_nll,
)
from model_loading import load_causal_model
from one_confidence_frozen_source_access import (
    load_rows,
    require_authorized_argument,
    require_same_access,
)


PUBLIC_SPLITS = ("development", "validation")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def indexed(records: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in records:
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise ValueError(f"duplicate {name} sample_id: {sample_id}")
        output[sample_id] = row
    return output


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def variance(values: list[float]) -> float:
    if not values:
        return 0.0
    center = mean(values)
    return mean([(value - center) ** 2 for value in values])


def generated_answers_with_metadata(
    model: Any,
    tokenizer: Any,
    prompts: list[list[int]],
    pad_id: int,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[list[str], list[float], list[list[int]], list[list[float]], list[str]]:
    ids, mask = padded(prompts, pad_id, device)
    width = ids.shape[1]
    output = model.generate(
        ids,
        attention_mask=mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    generated = output.sequences[:, width:]
    eos_values = (
        [tokenizer.eos_token_id]
        if isinstance(tokenizer.eos_token_id, int)
        else tokenizer.eos_token_id or []
    )
    eos_ids = {int(value) for value in eos_values}
    answers: list[str] = []
    confidences: list[float] = []
    token_ids: list[list[int]] = []
    token_logprobs: list[list[float]] = []
    finish_reasons: list[str] = []
    for row_index in range(generated.shape[0]):
        content_ids = []
        logps = []
        finish_reason = "length"
        for step, logits in enumerate(output.scores):
            token = int(generated[row_index, step])
            if token in eos_ids:
                finish_reason = "eos"
                break
            if token == pad_id:
                finish_reason = "pad"
                break
            content_ids.append(token)
            logps.append(
                float(torch.log_softmax(logits[row_index].float(), dim=-1)[token])
            )
        token_ids.append(content_ids)
        token_logprobs.append(logps)
        answers.append(tokenizer.decode(content_ids, skip_special_tokens=True).strip())
        confidences.append(math.exp(sum(logps) / len(logps)) if logps else 0.0)
        finish_reasons.append(finish_reason)
    return answers, confidences, token_ids, token_logprobs, finish_reasons


def feature_payload(
    *,
    cell: str,
    model_path: str,
    benchmark_path: Path,
    audit_path: Path,
    records: list[dict[str, Any]],
    complete: bool,
    started_at: float,
    locked_accessed: bool = False,
    authorization_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "one-confidence-corrected-panel-features-v2",
        "cell": cell,
        "model_path": model_path,
        "dtype": "bfloat16",
        "locked_accessed": locked_accessed,
        "source_authorization_id": authorization_id,
        "gold_visible": False,
        "complete": complete,
        "source_sha256": {
            "benchmark": sha256(benchmark_path),
            "question_equivalence_audit": sha256(audit_path),
        },
        "answer_protocol": {
            "system": ANSWER_SYSTEM,
            "decoding": "greedy",
            "qwen_thinking": False,
            "panel_size_including_canonical": 5,
        },
        "feature_semantics": {
            "safe_panel": "canonical plus four audited rewrites or verbatim-question wrappers",
            "natural_subset": "canonical plus only accepted model rewrites; wrappers excluded",
            "canonical_answer": "the model answer frozen in the benchmark, never a gold answer",
        },
        "elapsed_seconds": time.time() - started_at,
        "records": records,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--question-audit", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--frozen-source-authorization", type=Path)
    args = parser.parse_args()

    benchmark, selected_rows, locked_accessed, authorization_id = load_rows(
        benchmark_path=args.benchmark,
        cell=args.cell,
        authorization_path=args.frozen_source_authorization,
        stage="corrected_panel_features",
        output_path=args.output,
    )
    if locked_accessed and args.limit is not None:
        raise ValueError("frozen source generation forbids --limit")
    if locked_accessed and args.resume:
        raise ValueError("frozen one-shot extraction forbids --resume")
    require_authorized_argument(
        args.frozen_source_authorization,
        cell=args.cell,
        name="model_path",
        value=args.model_path,
    )
    audit = json.loads(args.question_audit.read_text(encoding="utf-8"))
    require_same_access(
        audit,
        frozen=locked_accessed,
        authorization_id=authorization_id,
        source_name="question equivalence audit",
    )
    corrected = audit.get("corrected_panel_manifest", {})
    require_same_access(
        corrected,
        frozen=locked_accessed,
        authorization_id=authorization_id,
        source_name="corrected question panel",
    )
    if corrected.get("contains_labels") is not False:
        raise ValueError("corrected panel must be label-free")
    if audit.get("cell") != args.cell or corrected.get("cell") != args.cell:
        raise ValueError("cell mismatch across corrected-panel inputs")

    benchmark_rows = {
        str(row["sample_id"]): (split, row) for split, row in selected_rows
    }
    panels = indexed(corrected["records"], "corrected-panel")
    if set(panels) != set(benchmark_rows):
        raise ValueError("corrected panel must cover exactly the public benchmark rows")
    ordered_ids = [
        str(row["sample_id"])
        for _, row in selected_rows
    ]
    if args.limit is not None:
        ordered_ids = ordered_ids[: args.limit]

    existing_records: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("cell") != args.cell:
            raise ValueError("resume artifact belongs to a different cell")
        if existing.get("source_sha256") != {
            "benchmark": sha256(args.benchmark),
            "question_equivalence_audit": sha256(args.question_audit),
        }:
            raise ValueError("resume artifact source hashes changed")
        existing_records = list(existing.get("records", []))
    existing_by_id = indexed(existing_records, "resume")
    pending_ids = [sample_id for sample_id in ordered_ids if sample_id not in existing_by_id]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_causal_model(
        args.model_path, load_in_4bit=False, torch_dtype=torch.bfloat16
    ).eval()
    device = next(model.parameters()).device
    pad_id = int(tokenizer.pad_token_id)
    records = list(existing_records)
    started_at = time.time()

    for offset in range(0, len(pending_ids), args.batch_rows):
        batch_ids = pending_ids[offset : offset + args.batch_rows]
        prompt_groups: list[list[list[int]]] = []
        flat_prompts: list[list[int]] = []
        for sample_id in batch_ids:
            panel_questions = panels[sample_id]["panel_questions"]
            if not isinstance(panel_questions, list) or len(panel_questions) != 5:
                raise ValueError(f"{sample_id}: corrected panel must contain five questions")
            prompts = [chat_ids(tokenizer, ANSWER_SYSTEM, question) for question in panel_questions]
            prompt_groups.append(prompts)
            flat_prompts.extend(prompts)

        prompt_ids, prompt_mask = padded(flat_prompts, pad_id, device)
        start_logits = model(
            input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=False
        ).logits[:, -1].float()
        (
            answers,
            confidences,
            generated_token_ids,
            generated_token_logprobs,
            finish_reasons,
        ) = generated_answers_with_metadata(
            model,
            tokenizer,
            flat_prompts,
            pad_id,
            device,
            args.max_answer_tokens,
        )

        sequences: list[list[int]] = []
        prompt_lengths: list[int] = []
        answer_lengths: list[int] = []
        for sample_id, prompts in zip(batch_ids, prompt_groups):
            _, benchmark_row = benchmark_rows[sample_id]
            answer_ids = tokenizer.encode(
                benchmark_row["generated_answer"], add_special_tokens=False
            )
            if not answer_ids:
                raise ValueError(f"{sample_id}: frozen generated answer tokenizes empty")
            for prompt in prompts:
                sequences.append(prompt + answer_ids)
                prompt_lengths.append(len(prompt))
                answer_lengths.append(len(answer_ids))
        canonical_nlls = teacher_forced_nll(
            model, sequences, prompt_lengths, answer_lengths, pad_id, device
        )

        for local_index, sample_id in enumerate(batch_ids):
            split, benchmark_row = benchmark_rows[sample_id]
            panel = panels[sample_id]
            base = local_index * 5
            row_logits = start_logits[base : base + 5]
            distributions = torch.softmax(row_logits, dim=-1)
            row_answers = answers[base : base + 5]
            row_confidences = confidences[base : base + 5]
            row_generated_token_ids = generated_token_ids[base : base + 5]
            row_generated_token_logprobs = generated_token_logprobs[base : base + 5]
            row_finish_reasons = finish_reasons[base : base + 5]
            row_valid_answers = [
                bool(answer.strip()) and reason != "length"
                for answer, reason in zip(row_answers, row_finish_reasons)
            ]
            row_nlls = canonical_nlls[base : base + 5]
            natural_mask = [bool(value) for value in panel["natural_paraphrase_mask"]]
            if len(natural_mask) != 5 or not natural_mask[0]:
                raise ValueError(f"{sample_id}: invalid natural paraphrase mask")
            natural_indices = [index for index in range(1, 5) if natural_mask[index]]

            canonical_js = [
                float(js_divergence(distributions[0], distributions[index]))
                for index in range(1, 5)
            ]
            natural_js = [canonical_js[index - 1] for index in natural_indices]
            all_pairwise = [
                float(js_divergence(distributions[left], distributions[right]))
                for left in range(5)
                for right in range(left + 1, 5)
            ]
            natural_panel_indices = [0, *natural_indices]
            natural_pairwise = [
                float(js_divergence(distributions[left], distributions[right]))
                for local, left in enumerate(natural_panel_indices)
                for right in natural_panel_indices[local + 1 :]
            ]
            top_probabilities, top_ids = distributions.max(dim=-1)
            _, top_counts = torch.unique(top_ids, return_counts=True)
            changes = [row_nlls[index] - row_nlls[0] for index in range(1, 5)]
            natural_changes = [changes[index - 1] for index in natural_indices]
            canonical_key = normalize(benchmark_row["generated_answer"])
            alternative_answers = [
                answer
                for answer, valid in zip(row_answers[1:], row_valid_answers[1:])
                if valid and normalize(answer) and normalize(answer) != canonical_key
            ]
            exact_counts = Counter(normalize(answer) for answer in alternative_answers)
            exact_candidate_key, exact_support = (
                exact_counts.most_common(1)[0] if exact_counts else ("", 0)
            )
            exact_candidate = next(
                (answer for answer in alternative_answers if normalize(answer) == exact_candidate_key),
                "",
            )
            natural_answers = [row_answers[index] for index in natural_indices]

            query_budget_features: dict[str, float] = {}
            for panel_size in (2, 4, 5):
                suffix = f"k{panel_size}"
                selected_indices = list(range(panel_size))
                selected_distributions = distributions[selected_indices]
                selected_canonical_js = canonical_js[: panel_size - 1]
                selected_pairwise_js = [
                    float(js_divergence(distributions[left], distributions[right]))
                    for local, left in enumerate(selected_indices)
                    for right in selected_indices[local + 1 :]
                ]
                selected_top_ids = top_ids[selected_indices]
                _, selected_top_counts = torch.unique(
                    selected_top_ids, return_counts=True
                )
                selected_changes = changes[: panel_size - 1]
                selected_answers = row_answers[1:panel_size]
                selected_confidences = row_confidences[1:panel_size]
                query_budget_features.update(
                    {
                        f"answer_start_mi_{suffix}": float(
                            entropy(selected_distributions.mean(dim=0))
                            - entropy(selected_distributions).mean()
                        ),
                        f"canonical_panel_js_mean_{suffix}": mean(
                            selected_canonical_js
                        ),
                        f"canonical_panel_js_max_{suffix}": max(
                            selected_canonical_js
                        ),
                        f"canonical_panel_js_variance_{suffix}": variance(
                            selected_canonical_js
                        ),
                        f"all_pairwise_js_mean_{suffix}": mean(
                            selected_pairwise_js
                        ),
                        f"answer_start_top1_disagreement_{suffix}": 1.0
                        - float(selected_top_counts.max()) / panel_size,
                        f"answer_start_unique_top1_{suffix}": float(
                            selected_top_counts.numel()
                        ),
                        f"cross_prompt_nll_signed_mean_{suffix}": mean(
                            selected_changes
                        ),
                        f"cross_prompt_nll_absolute_mean_{suffix}": mean(
                            [abs(value) for value in selected_changes]
                        ),
                        f"cross_prompt_nll_max_{suffix}": max(selected_changes),
                        f"canonical_cluster_retention_{suffix}": mean(
                            [
                                float(normalize(answer) == canonical_key)
                                for answer in selected_answers
                            ]
                        ),
                        f"string_disagreement_{suffix}": mean(
                            [
                                float(normalize(answer) != canonical_key)
                                for answer in selected_answers
                            ]
                        ),
                        f"paraphrase_confidence_gain_{suffix}": float(
                            max(selected_confidences) - benchmark_row["confidence"]
                        ),
                    }
                )

            record = {
                    "sample_id": sample_id,
                    "split": split,
                    "fact_key_sha256": benchmark_row["fact_key_sha256"],
                    "panel_questions": panel["panel_questions"],
                    "panel_question_sha256": panel["panel_question_sha256"],
                    "natural_paraphrase_mask": natural_mask,
                    "accepted_model_rewrite_count": int(panel["accepted_model_rewrite_count"]),
                    "fallback_count": int(panel["fallback_count"]),
                    "canonical_answer": benchmark_row["generated_answer"],
                    "panel_answers": row_answers,
                    "panel_answer_confidences": row_confidences,
                    "panel_generated_token_ids": row_generated_token_ids,
                    "panel_generated_token_logprobs": row_generated_token_logprobs,
                    "panel_finish_reasons": row_finish_reasons,
                    "panel_answer_valid_mask": row_valid_answers,
                    "panel_canonical_answer_nlls": [
                        float(value) for value in row_nlls
                    ],
                    "answer_start_canonical_js_per_rewrite": canonical_js,
                    "answer_start_pairwise_js": [
                        {
                            "left": left,
                            "right": right,
                            "js": float(
                                js_divergence(
                                    distributions[left], distributions[right]
                                )
                            ),
                        }
                        for left in range(5)
                        for right in range(left + 1, 5)
                    ],
                    "answer_start_top_token_ids": [
                        int(value) for value in top_ids.tolist()
                    ],
                    "answer_start_top_probabilities": [
                        float(value) for value in top_probabilities.tolist()
                    ],
                    "exact_candidate_answer": exact_candidate,
                    "exact_candidate_support": int(exact_support),
                    "features": query_budget_features
                    | {
                        "answer_start_mi_all_safe": float(
                            entropy(distributions.mean(dim=0)) - entropy(distributions).mean()
                        ),
                        "canonical_panel_js_mean_all_safe": mean(canonical_js),
                        "canonical_panel_js_max_all_safe": max(canonical_js),
                        "canonical_panel_js_variance_all_safe": variance(canonical_js),
                        "all_pairwise_js_mean_all_safe": mean(all_pairwise),
                        "canonical_panel_js_mean_natural": mean(natural_js),
                        "canonical_panel_js_max_natural": max(natural_js) if natural_js else 0.0,
                        "all_pairwise_js_mean_natural": mean(natural_pairwise),
                        "natural_paraphrase_count": float(len(natural_indices)),
                        "answer_start_top1_disagreement_all_safe": 1.0
                        - float(top_counts.max()) / 5.0,
                        "answer_start_unique_top1_all_safe": float(top_counts.numel()),
                        "answer_start_top1_probability_range_all_safe": float(
                            top_probabilities.max() - top_probabilities.min()
                        ),
                        "canonical_answer_nll": float(row_nlls[0]),
                        "cross_prompt_nll_signed_mean_all_safe": mean(changes),
                        "cross_prompt_nll_absolute_mean_all_safe": mean(
                            [abs(value) for value in changes]
                        ),
                        "cross_prompt_nll_max_all_safe": max(changes),
                        "cross_prompt_nll_signed_mean_natural": mean(natural_changes),
                        "cross_prompt_nll_absolute_mean_natural": mean(
                            [abs(value) for value in natural_changes]
                        ),
                        "canonical_cluster_retention_all_safe": mean(
                            [
                                float(normalize(answer) == canonical_key)
                                for answer in row_answers[1:]
                            ]
                        ),
                        "string_disagreement_all_safe": mean(
                            [
                                float(normalize(answer) != canonical_key)
                                for answer in row_answers[1:]
                            ]
                        ),
                        "string_disagreement_natural": mean(
                            [
                                float(normalize(answer) != canonical_key)
                                for answer in natural_answers
                            ]
                        ),
                        "largest_exact_alternative_cluster_all_safe": float(
                            exact_support / 4
                        ),
                        "paraphrase_confidence_gain_all_safe": float(
                            max(row_confidences[1:]) - benchmark_row["confidence"]
                        ),
                        "canonical_greedy_repeat_exact": float(
                            normalize(row_answers[0]) == canonical_key
                        ),
                        "canonical_greedy_repeat_valid": float(row_valid_answers[0]),
                        "canonical_greedy_repeat_truncated": float(
                            row_finish_reasons[0] == "length"
                        ),
                        "canonical_greedy_repeat_confidence_delta": float(
                            row_confidences[0] - benchmark_row["confidence"]
                        ),
                    },
                }
            if not locked_accessed:
                record["label"] = benchmark_row["class"]
            records.append(record)

        completed = len(existing_records) + min(offset + len(batch_ids), len(pending_ids))
        print(
            json.dumps(
                {
                    "cell": args.cell,
                    "completed": completed,
                    "target": len(ordered_ids),
                    "elapsed_seconds": time.time() - started_at,
                }
            ),
            flush=True,
        )
        if args.checkpoint_every > 0 and (
            (offset // args.batch_rows + 1) % args.checkpoint_every == 0
        ):
            atomic_json(
                args.output,
                feature_payload(
                    cell=args.cell,
                    model_path=args.model_path,
                    benchmark_path=args.benchmark,
                    audit_path=args.question_audit,
                    records=records,
                    complete=False,
                    started_at=started_at,
                    locked_accessed=locked_accessed,
                    authorization_id=authorization_id,
                ),
            )

    by_id = indexed(records, "output")
    final_records = [by_id[sample_id] for sample_id in ordered_ids]
    atomic_json(
        args.output,
        feature_payload(
            cell=args.cell,
            model_path=args.model_path,
            benchmark_path=args.benchmark,
            audit_path=args.question_audit,
            records=final_records,
            complete=True,
            started_at=started_at,
            locked_accessed=locked_accessed,
            authorization_id=authorization_id,
        ),
    )


if __name__ == "__main__":
    main()
