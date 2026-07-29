#!/usr/bin/env python3
"""NLI cluster answers from an audited public question panel.

Natural recovery requires two accepted model rewrites to support the same
non-original answer cluster. Deterministic wrapper fallbacks remain useful for
safe-panel stability features but never count toward natural recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from one_confidence_frozen_source_access import (
    load_rows,
    require_authorized_argument,
    require_same_access,
)
from score_one_confidence_sampling import NLI, clusters


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


def entropy_from_counts(counts: list[int], denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    probabilities = [count / denominator for count in counts if count]
    return -sum(value * math.log(value) for value in probabilities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--nli-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-source-authorization", type=Path)
    args = parser.parse_args()

    benchmark, selected_rows, locked_accessed, authorization_id = load_rows(
        benchmark_path=args.benchmark,
        cell=args.cell,
        authorization_path=args.frozen_source_authorization,
        stage="corrected_panel_nli",
        output_path=args.output,
    )
    require_authorized_argument(
        args.frozen_source_authorization,
        cell=args.cell,
        name="nli_model",
        value=args.nli_model,
    )
    features = json.loads(args.features.read_text(encoding="utf-8"))
    require_same_access(
        features,
        frozen=locked_accessed,
        authorization_id=authorization_id,
        source_name="corrected panel features",
    )
    if not features.get("complete"):
        raise ValueError("corrected-panel NLI requires a complete feature artifact")
    if features.get("cell") != args.cell:
        raise ValueError("feature artifact belongs to another cell")
    benchmark_rows = {
        str(row["sample_id"]): row
        for _, row in selected_rows
    }
    feature_rows = indexed(features["records"], "corrected-panel-feature")
    if set(feature_rows) != set(benchmark_rows):
        raise ValueError("features must cover exactly the public benchmark rows")

    nli = NLI(args.nli_model)
    records = []
    for index, sample_id in enumerate(feature_rows, 1):
        row = feature_rows[sample_id]
        question = benchmark_rows[sample_id]["question"]
        canonical = row["canonical_answer"]
        answers = row["panel_answers"][1:]
        natural_mask = [bool(value) for value in row["natural_paraphrase_mask"]][1:]
        valid_mask = [bool(value) for value in row["panel_answer_valid_mask"]][1:]
        if len(answers) != 4 or len(natural_mask) != 4 or len(valid_mask) != 4:
            raise ValueError(f"{sample_id}: expected four non-canonical panel answers")

        valid_positions = [position for position, valid in enumerate(valid_mask) if valid]
        valid_answers = [answers[position] for position in valid_positions]
        equivalent_mask = [False] * 4
        for position in valid_positions:
            equivalent_mask[position] = nli.equivalent(
                question, canonical, answers[position]
            )
        valid_cluster_ids, representatives = (
            clusters(nli, question, valid_answers) if valid_answers else ([], [])
        )
        cluster_ids = [-1] * 4
        for position, cluster_id in zip(valid_positions, valid_cluster_ids):
            cluster_ids[position] = cluster_id
        counts = Counter(valid_cluster_ids)
        alternative_clusters = []
        for cluster_id, count in counts.items():
            representative = representatives[cluster_id]
            if not nli.equivalent(question, canonical, representative):
                natural_count = sum(
                    cluster_ids[position] == cluster_id and natural_mask[position]
                    for position in range(4)
                )
                alternative_clusters.append(
                    (count, natural_count, representative, cluster_id)
                )
        alternative_clusters.sort(key=lambda value: (-value[0], -value[1], value[2]))
        if alternative_clusters:
            all_count, all_natural_count, all_candidate, all_cluster = alternative_clusters[0]
        else:
            all_count, all_natural_count, all_candidate, all_cluster = 0, 0, "", None

        natural_candidates = sorted(
            alternative_clusters,
            key=lambda value: (-value[1], -value[0], value[2]),
        )
        if natural_candidates and natural_candidates[0][1] > 0:
            natural_all_count, natural_count, natural_candidate, natural_cluster = natural_candidates[0]
        else:
            natural_all_count, natural_count, natural_candidate, natural_cluster = 0, 0, "", None

        natural_positions = [
            position
            for position, keep in enumerate(natural_mask)
            if keep and valid_mask[position]
        ]
        natural_original_support = (
            sum(equivalent_mask[position] for position in natural_positions)
            / len(natural_positions)
            if natural_positions
            else 0.0
        )
        natural_cluster_counts = Counter(cluster_ids[position] for position in natural_positions)
        record = {
                "sample_id": sample_id,
                "split": row["split"],
                "fact_key_sha256": row["fact_key_sha256"],
                "canonical_equivalent_mask": equivalent_mask,
                "panel_answer_valid_mask": valid_mask,
                "cluster_ids": cluster_ids,
                "cluster_representatives": representatives,
                "valid_safe_panel_answer_count": len(valid_positions),
                "original_support_all_safe": (
                    sum(equivalent_mask) / len(valid_positions)
                    if valid_positions
                    else 0.0
                ),
                "semantic_disagreement_all_safe": (
                    1 - sum(equivalent_mask) / len(valid_positions)
                    if valid_positions
                    else 0.0
                ),
                "semantic_entropy_all_safe": entropy_from_counts(
                    list(counts.values()), len(valid_positions)
                ),
                "original_support_natural": natural_original_support,
                "semantic_disagreement_natural": (
                    1 - natural_original_support if natural_positions else 0.0
                ),
                "semantic_entropy_natural": entropy_from_counts(
                    list(natural_cluster_counts.values()), len(natural_positions)
                ),
                "natural_paraphrase_count": len(natural_positions),
                "alternative_candidate_all_safe": all_candidate,
                "alternative_candidate_cluster_all_safe": all_cluster,
                "alternative_support_all_safe": (
                    all_count / len(valid_positions) if valid_positions else 0.0
                ),
                "alternative_natural_support_for_all_safe_candidate": all_natural_count,
                "alternative_candidate_natural": natural_candidate,
                "alternative_candidate_cluster_natural": natural_cluster,
                "alternative_support_count_natural": natural_count,
                "alternative_support_fraction_natural": (
                    natural_count / len(natural_positions) if natural_positions else 0.0
                ),
                "alternative_total_support_for_natural_candidate": natural_all_count,
                "safe_panel_alternative_supported": all_count >= 2,
                "behavioral_recoverable": natural_count >= 2,
            }
        if not locked_accessed:
            record["label"] = row["label"]
        records.append(record)
        if index % 50 == 0 or index == len(feature_rows):
            print(f"{index}/{len(feature_rows)}", flush=True)

    payload = {
        "schema_version": "one-confidence-corrected-panel-nli-v1",
        "cell": args.cell,
        "locked_accessed": locked_accessed,
        "source_authorization_id": authorization_id,
        "nli_model": args.nli_model,
        "clustering": "greedy bidirectional entailment",
        "candidate_protocol": {
            "all_safe": "largest non-original semantic cluster among four audited rewrites/wrappers",
            "natural": "largest non-original semantic cluster counted only on accepted model rewrites",
            "behavioral_recoverable": "at least two accepted natural rewrites support the same non-original cluster",
            "fallback_wrappers_count_as_natural": False,
        },
        "source_sha256": {
            "features": sha256(args.features),
            "benchmark": sha256(args.benchmark),
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
