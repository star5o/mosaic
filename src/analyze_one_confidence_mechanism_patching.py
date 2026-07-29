#!/usr/bin/env python3
"""Summarize frozen public late-MLP patching with row bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def interval(values: list[float], seed: int, reps: int) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean(rng.choice(array, size=len(array), replace=True)))
        for _ in range(reps)
    ]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def metric(values: list[float], seed: int, reps: int) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)) if values else None,
        "ci95": interval(values, seed, reps),
        "positive_fraction": float(np.mean(np.asarray(values) > 0)) if values else None,
    }


def layer_rows(records: list[dict[str, Any]], layer: int) -> list[dict[str, Any]]:
    return [
        next(item for item in record["layers"] if int(item["layer"]) == layer)
        for record in records
    ]


def summarize_hce(
    records: list[dict[str, Any]], layer: int, seed: int, reps: int
) -> dict[str, Any]:
    rows = layer_rows(records, layer)
    aligned = [float(row["aligned_effect"]) for row in rows]
    random_control = [float(row["random_direction_effect_mean"]) for row in rows]
    unrelated = [float(row["unrelated_effect"]) for row in rows]
    shuffled = [float(row["shuffled_candidate_effect"]) for row in rows]
    reverse = [float(row["reverse_effect"]) for row in rows]
    primary = [
        value - max(random_value, unrelated_value, shuffled_value)
        for value, random_value, unrelated_value, shuffled_value in zip(
            aligned, random_control, unrelated, shuffled
        )
    ]
    return {
        "n": len(rows),
        "aligned_effect": metric(aligned, seed, reps),
        "aligned_minus_random": metric(
            [left - right for left, right in zip(aligned, random_control)],
            seed + 1,
            reps,
        ),
        "aligned_minus_unrelated": metric(
            [left - right for left, right in zip(aligned, unrelated)],
            seed + 2,
            reps,
        ),
        "aligned_minus_shuffled": metric(
            [left - right for left, right in zip(aligned, shuffled)],
            seed + 3,
            reps,
        ),
        "primary_control_adjusted": metric(primary, seed + 4, reps),
        "reverse_effect": {
            **metric(reverse, seed + 5, reps),
            "negative_fraction": (
                float(np.mean(np.asarray(reverse) < 0)) if reverse else None
            ),
        },
        "bidirectional_fraction": (
            float(
                np.mean(
                    [left > 0 and right < 0 for left, right in zip(aligned, reverse)]
                )
            )
            if rows
            else None
        ),
        "answer_start_js_reduction": metric(
            [float(row["answer_start_js_reduction"]) for row in rows],
            seed + 6,
            reps,
        ),
        "max_abs_noop": max(
            (abs(float(row["noop_effect"])) for row in rows), default=0.0
        ),
    }


def summarize_hcc(
    records: list[dict[str, Any]], layer: int, seed: int, reps: int
) -> dict[str, Any]:
    rows = layer_rows(records, layer)
    return {
        "n": len(rows),
        "canonical_score_change": metric(
            [float(row["aligned_canonical_score_change"]) for row in rows],
            seed,
            reps,
        ),
        "answer_start_kl": metric(
            [float(row["aligned_answer_start_kl"]) for row in rows],
            seed + 1,
            reps,
        ),
        "max_abs_noop": max(
            (abs(float(row["noop_canonical_score_change"])) for row in rows),
            default=0.0,
        ),
    }


def generation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hce = [row for row in rows if row["label"] == "HCE"]
    hcc = [row for row in rows if row["label"] == "HCC"]
    return {
        "hce_n": len(hce),
        "aligned_candidate_exact": sum(
            row["candidate_exact_match"]["aligned_selected_layer"] for row in hce
        ),
        "aligned_changed_from_canonical": sum(
            not row["canonical_exact_match"]["aligned_selected_layer"] for row in hce
        ),
        "random_layer_candidate_exact": sum(
            row["candidate_exact_match"]["aligned_random_layer"] for row in hce
        ),
        "unrelated_candidate_exact": sum(
            row["candidate_exact_match"]["unrelated_fact_donor"] for row in hce
        ),
        "hcc_n": len(hcc),
        "hcc_aligned_damaged": sum(
            row["canonical_exact_match"]["canonical_unpatched"]
            and not row["canonical_exact_match"]["aligned_selected_layer"]
            for row in hcc
        ),
        "hcc_random_layer_damaged": sum(
            row["canonical_exact_match"]["canonical_unpatched"]
            and not row["canonical_exact_match"]["aligned_random_layer"]
            for row in hcc
        ),
        "hcc_unrelated_damaged": sum(
            row["canonical_exact_match"]["canonical_unpatched"]
            and not row["canonical_exact_match"]["unrelated_fact_donor"]
            for row in hcc
        ),
    }


def summarize(payload: dict[str, Any], seed: int, reps: int) -> dict[str, Any]:
    selected_layer = int(payload["layer_selection"]["selected_layer"])
    random_layer = int(payload["layer_selection"]["random_control_layer"])
    hcc_records = payload["hcc_control_records"]
    return {
        "cell": payload["protocol"]["cell"],
        "selected_layer": selected_layer,
        "random_control_layer": random_layer,
        "development_selected": summarize_hce(
            payload["development_scan_records"], selected_layer, seed, reps
        ),
        "validation_selected": summarize_hce(
            payload["validation_hce_records"], selected_layer, seed + 100, reps
        ),
        "validation_random_layer": summarize_hce(
            payload["validation_hce_records"], random_layer, seed + 200, reps
        ),
        "validation_hcc": summarize_hcc(
            [row for row in hcc_records if row["split"] == "validation"],
            selected_layer,
            seed + 300,
            reps,
        ),
        "generation": generation_summary(payload["validation_generations"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026072903)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()
    summaries = [
        summarize(
            json.loads(path.read_text(encoding="utf-8")),
            args.seed + index * 1000,
            args.bootstrap_reps,
        )
        for index, path in enumerate(args.inputs)
    ]
    result = {
        "schema_version": "one-confidence-public-mechanism-patching-analysis-v1",
        "positive_endpoint": "candidate-minus-canonical sequence margin",
        "candidate_uses_answer_key": False,
        "locked_accessed": False,
        "bootstrap": "row bootstrap; intervals are descriptive because validation n is very small",
        "models": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
