#!/usr/bin/env python3
"""Score model-proposed NLI candidates under the canonical public prompt.

The candidate is selected without a gold answer: natural recovery candidates
take precedence, followed by the largest alternative cluster from the audited
safe panel.  Scores are length-normalized teacher-forced sequence log scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from extract_one_confidence_natural_paraphrase_features import (
    ANSWER_SYSTEM,
    chat_ids,
    normalize,
    teacher_forced_nll,
)
from model_loading import load_causal_model
from one_confidence_frozen_source_access import (
    load_rows,
    require_authorized_argument,
    require_same_access,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(records: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in records:
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise ValueError(f"duplicate {name} sample_id: {sample_id}")
        output[sample_id] = row
    return output


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--nli", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=32)
    parser.add_argument("--frozen-source-authorization", type=Path)
    args = parser.parse_args()

    benchmark, selected_rows, locked_accessed, authorization_id = load_rows(
        benchmark_path=args.benchmark,
        cell=args.cell,
        authorization_path=args.frozen_source_authorization,
        stage="candidate_margins",
        output_path=args.output,
    )
    require_authorized_argument(
        args.frozen_source_authorization,
        cell=args.cell,
        name="model_path",
        value=args.model_path,
    )
    features = json.loads(args.features.read_text(encoding="utf-8"))
    nli = json.loads(args.nli.read_text(encoding="utf-8"))
    require_same_access(
        features,
        frozen=locked_accessed,
        authorization_id=authorization_id,
        source_name="corrected panel features",
    )
    require_same_access(
        nli,
        frozen=locked_accessed,
        authorization_id=authorization_id,
        source_name="corrected panel NLI",
    )
    if not features.get("complete"):
        raise ValueError("candidate margin scorer requires complete corrected-panel features")
    if features.get("cell") != args.cell or nli.get("cell") != args.cell:
        raise ValueError("cell mismatch across candidate margin inputs")

    benchmark_rows = {
        str(row["sample_id"]): row
        for _, row in selected_rows
    }
    feature_rows = indexed(features["records"], "corrected-panel-feature")
    nli_rows = indexed(nli["records"], "corrected-panel-nli")
    if set(feature_rows) != set(nli_rows) or set(feature_rows) != set(benchmark_rows):
        raise ValueError("candidate margin inputs must cover identical public rows")

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

    records = []
    sample_ids = list(feature_rows)
    for offset in range(0, len(sample_ids), args.batch_rows):
        batch_ids = sample_ids[offset : offset + args.batch_rows]
        sequences: list[list[int]] = []
        prompt_lengths: list[int] = []
        answer_lengths: list[int] = []
        metadata = []
        for sample_id in batch_ids:
            feature_row = feature_rows[sample_id]
            nli_row = nli_rows[sample_id]
            canonical = str(feature_row["canonical_answer"])
            natural_candidate = str(
                nli_row.get("alternative_candidate_natural") or ""
            ).strip()
            safe_candidate = str(
                nli_row.get("alternative_candidate_all_safe") or ""
            ).strip()
            if natural_candidate and normalize(natural_candidate) != normalize(canonical):
                candidate = natural_candidate
                candidate_source = "natural_alternative_cluster"
                candidate_cluster = nli_row.get(
                    "alternative_candidate_cluster_natural"
                )
            elif safe_candidate and normalize(safe_candidate) != normalize(canonical):
                candidate = safe_candidate
                candidate_source = "safe_panel_alternative_cluster"
                candidate_cluster = nli_row.get(
                    "alternative_candidate_cluster_all_safe"
                )
            else:
                candidate = ""
                candidate_source = "none"
                candidate_cluster = None
            support_positions = [
                position
                for position, cluster_id in enumerate(nli_row["cluster_ids"])
                if candidate_cluster is not None
                and cluster_id == candidate_cluster
                and feature_row["natural_paraphrase_mask"][position + 1]
                and feature_row["panel_answer_valid_mask"][position + 1]
            ]
            metadata.append(
                (
                    sample_id,
                    canonical,
                    candidate,
                    candidate_source,
                    candidate_cluster,
                    support_positions,
                    bool(nli_row["behavioral_recoverable"]),
                )
            )
            if not candidate:
                continue
            prompts = [
                chat_ids(tokenizer, ANSWER_SYSTEM, question)
                for question in feature_row["panel_questions"]
            ]
            for prompt in prompts:
                for answer in (canonical, candidate):
                    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                    if not answer_ids:
                        raise ValueError(f"{sample_id}: candidate sequence tokenizes empty")
                    sequences.append(prompt + answer_ids)
                    prompt_lengths.append(len(prompt))
                    answer_lengths.append(len(answer_ids))

        nll_values = (
            teacher_forced_nll(
                model,
                sequences,
                prompt_lengths,
                answer_lengths,
                pad_id,
                device,
            )
            if sequences
            else []
        )
        nll_iterator = iter(nll_values)
        for (
            sample_id,
            canonical,
            candidate,
            candidate_source,
            candidate_cluster,
            support_positions,
            nli_behavioral_recoverable,
        ) in metadata:
            if candidate:
                candidate_nlls = []
                canonical_nlls = []
                for _ in range(5):
                    canonical_nlls.append(float(next(nll_iterator)))
                    candidate_nlls.append(float(next(nll_iterator)))
                canonical_nll = canonical_nlls[0]
                candidate_nll = candidate_nlls[0]
                candidate_logscore_margin = canonical_nll - candidate_nll
                candidate_minus_canonical_nll = candidate_nll - canonical_nll
                panel_candidate_logscore_margins = [
                    canonical_value - candidate_value
                    for canonical_value, candidate_value in zip(
                        canonical_nlls, candidate_nlls
                    )
                ]
                support_margins = [
                    panel_candidate_logscore_margins[position + 1]
                    for position in support_positions
                ]
                support_margin_mean = (
                    sum(support_margins) / len(support_margins)
                    if support_margins
                    else 0.0
                )
            else:
                canonical_nll = None
                candidate_nll = None
                candidate_logscore_margin = 0.0
                candidate_minus_canonical_nll = 0.0
                canonical_nlls = []
                candidate_nlls = []
                panel_candidate_logscore_margins = []
                support_margins = []
                support_margin_mean = 0.0
            record = {
                    "sample_id": sample_id,
                    "split": feature_rows[sample_id]["split"],
                    "fact_key_sha256": feature_rows[sample_id]["fact_key_sha256"],
                    "canonical_answer": canonical,
                    "candidate_answer": candidate,
                    "candidate_source": candidate_source,
                    "candidate_cluster": candidate_cluster,
                    "candidate_available": bool(candidate),
                    "canonical_answer_nll_recomputed": canonical_nll,
                    "candidate_answer_nll": candidate_nll,
                    "candidate_logscore_margin": candidate_logscore_margin,
                    "candidate_minus_canonical_nll": candidate_minus_canonical_nll,
                    "panel_canonical_answer_nlls": canonical_nlls,
                    "panel_candidate_answer_nlls": candidate_nlls,
                    "panel_candidate_logscore_margins": panel_candidate_logscore_margins,
                    "natural_support_panel_indices": support_positions,
                    "natural_support_candidate_logscore_margins": support_margins,
                    "support_prompt_candidate_logscore_margin_mean": support_margin_mean,
                    "nli_behavioral_recoverable": nli_behavioral_recoverable,
                    "behavioral_recoverable_margin_pass": bool(
                        nli_behavioral_recoverable and support_margin_mean >= 0.0
                    ),
                }
            if not locked_accessed:
                record["label"] = feature_rows[sample_id]["label"]
            records.append(record)
        print(f"{min(offset + len(batch_ids), len(sample_ids))}/{len(sample_ids)}", flush=True)

    payload = {
        "schema_version": "one-confidence-nli-candidate-margins-v1",
        "cell": args.cell,
        "locked_accessed": locked_accessed,
        "source_authorization_id": authorization_id,
        "gold_visible": False,
        "model_path": args.model_path,
        "dtype": "bfloat16",
        "score_definition": {
            "sequence_score": "negative mean teacher-forced answer-token NLL",
            "candidate_logscore_margin": "score(candidate) - score(canonical) = canonical_nll - candidate_nll",
            "candidate_priority": [
                "natural alternative cluster",
                "safe-panel alternative cluster"
            ],
            "behavioral_recoverability_margin": {
                "threshold": 0.0,
                "aggregation": "mean over accepted natural rewrites in the candidate cluster",
                "labels_used_to_choose_threshold": False
            }
        },
        "source_sha256": {
            "benchmark": sha256(args.benchmark),
            "features": sha256(args.features),
            "nli": sha256(args.nli),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
