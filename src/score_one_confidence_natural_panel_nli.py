#!/usr/bin/env python3
"""NLI semantic clustering for the fixed natural-paraphrase answer panel."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from score_one_confidence_sampling import NLI, clusters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--nli-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    feature_payload = json.loads(args.features.read_text(encoding="utf-8"))
    if feature_payload.get("locked_accessed"):
        raise ValueError("natural-panel NLI scorer accepts public features only")
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    questions = {
        row["sample_id"]: row["question"]
        for split in ("development", "validation")
        for row in benchmark["cells"][args.cell][split]["records"]
    }
    nli = NLI(args.nli_model)
    records = []
    for index, row in enumerate(feature_payload["records"], 1):
        question = questions[row["sample_id"]]
        canonical = row["canonical_answer"]
        paraphrase_answers = row["panel_answers"][1:]
        original_support = sum(nli.equivalent(question, canonical, answer) for answer in paraphrase_answers) / 4
        cluster_ids, representatives = clusters(nli, question, paraphrase_answers)
        counts = Counter(cluster_ids)
        alternative_clusters = []
        for cluster_id, count in counts.items():
            representative = representatives[cluster_id]
            if not nli.equivalent(question, canonical, representative):
                alternative_clusters.append((count, representative, cluster_id))
        alternative_clusters.sort(reverse=True)
        largest_count, candidate, candidate_cluster = alternative_clusters[0] if alternative_clusters else (0, "", None)
        cluster_probabilities = [count / 4 for count in counts.values()]
        semantic_entropy = -sum(value * math.log(value) for value in cluster_probabilities)
        records.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": row["label"],
                "original_support": original_support,
                "semantic_disagreement": 1 - original_support,
                "semantic_entropy": semantic_entropy,
                "cluster_ids": cluster_ids,
                "cluster_representatives": representatives,
                "largest_alternative_cluster": largest_count / 4,
                "alternative_candidate": candidate,
                "alternative_candidate_cluster": candidate_cluster,
                "behavioral_recoverable": largest_count >= 2,
            }
        )
        if index % 50 == 0 or index == len(feature_payload["records"]):
            print(f"{index}/{len(feature_payload['records'])}", flush=True)

    payload = {
        "schema_version": "one-confidence-natural-panel-nli-v1",
        "cell": args.cell,
        "locked_accessed": False,
        "nli_model": "microsoft/deberta-large-mnli",
        "clustering": "greedy bidirectional entailment",
        "phenotype_rule": "at least two of four paraphrases support the same non-original semantic cluster",
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
