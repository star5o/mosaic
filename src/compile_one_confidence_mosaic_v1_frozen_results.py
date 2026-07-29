#!/usr/bin/env python3
"""Compile the one-shot Run6 frozen result and pair-preserving bootstrap CIs.

This script is evaluation-only. It joins the class strata stored in the frozen
application manifests with already-written predictions. It never fits a model,
selects a threshold, or reads a public development/validation score to make a
new choice.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CELLS = (
    ("mistral7|TriviaQA", "mistral7-triviaqa"),
    ("qwen8|TriviaQA", "qwen8-triviaqa"),
    ("llama31|TriviaQA", "llama31-triviaqa"),
    ("qwen14|PopQA", "qwen14-popqa"),
    ("nemo12|PopQA", "nemo12-popqa"),
)
RUN6_METHOD = "learned_gate_no_phenotype"
FROZEN_MATRIX_METHODS = (
    "sequence_confidence",
    "predictive_token_entropy",
    "p_true",
    "self_consistency",
    "semantic_entropy",
    "paraphrase_consistency",
    "prompt_end_hidden_linear_probe",
    "first_answer_hidden_linear_probe",
    "mean_answer_hidden_linear_probe",
    "inside_eigenscore",
    "los_net",
    "behavioral_component",
    "internal_component",
    "persistence_component",
    "behavioral_plus_internal_average",
    "flat_feature_concatenation",
    "minimal_gated_mosaic",
)
METHODS = (
    "mosaic_v1_run6",
    *FROZEN_MATRIX_METHODS,
)
BOOTSTRAP_METHODS = (
    "mosaic_v1_run6",
    "minimal_gated_mosaic",
    "mean_answer_hidden_linear_probe",
)
UNAVAILABLE_METHODS = {
    "pale": {
        "display_name": "PALE",
        "status": "unavailable_no_frozen_faithful_implementation",
        "reason": "No reliable frozen PALE implementation or locked score exists; no substitute score was fabricated.",
    }
}
METHOD_DISPLAY_NAMES = {
    "mosaic_v1_run6": "MOSAIC-v1 Run6",
    "minimal_gated_mosaic": "MOSAIC-v0",
    "pale": "PALE",
}
METHOD_ROLES = {
    "mosaic_v1_run6": "candidate_method",
    "minimal_gated_mosaic": "candidate_method_v0",
    "behavioral_component": "component",
    "internal_component": "component",
    "persistence_component": "component",
    "behavioral_plus_internal_average": "ablation_or_combination",
    "flat_feature_concatenation": "ablation_or_combination",
    "pale": "published_baseline_unavailable",
}
POINT_METRICS = ("accuracy", "precision", "recall", "fpr", "f1", "auroc")
BOOTSTRAP_METRICS = ("accuracy", "precision", "recall", "fpr", "f1")
ALLOWED_APPLICATION_RECORD_FIELDS = {
    "sample_id",
    "fact_key_sha256",
    "split",
    "pair_id",
    "class_stratum",
    "model",
    "dataset",
}


@dataclass(frozen=True)
class CellEvaluation:
    cell: str
    slug: str
    pair_ids: tuple[str, ...]
    unit_pair_indices: tuple[tuple[int, ...], ...]
    thresholds: dict[str, float]
    predictions: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]
    statuses: dict[str, str]
    source_sha256: dict[str, str]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def safe_divide(numerator: Any, denominator: Any) -> Any:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    result = np.zeros(np.broadcast_shapes(numerator_array.shape, denominator_array.shape))
    np.divide(numerator_array, denominator_array, out=result, where=denominator_array != 0)
    return float(result) if result.ndim == 0 else result


def metrics_from_counts(tp: Any, fn: Any, fp: Any, tn: Any) -> dict[str, Any]:
    total = np.asarray(tp) + np.asarray(fn) + np.asarray(fp) + np.asarray(tn)
    accuracy = safe_divide(np.asarray(tp) + np.asarray(tn), total)
    precision = safe_divide(tp, np.asarray(tp) + np.asarray(fp))
    recall = safe_divide(tp, np.asarray(tp) + np.asarray(fn))
    fpr = safe_divide(fp, np.asarray(fp) + np.asarray(tn))
    f1 = safe_divide(2 * np.asarray(precision) * np.asarray(recall), np.asarray(precision) + np.asarray(recall))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
    }


def integer_counts(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    return {
        "TP": int(np.sum((labels == 1) & predictions)),
        "FN": int(np.sum((labels == 1) & ~predictions)),
        "FP": int(np.sum((labels == 0) & predictions)),
        "TN": int(np.sum((labels == 0) & ~predictions)),
    }


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_scores = np.asarray(scores, dtype=float)[np.asarray(labels) == 1]
    negative_scores = np.asarray(scores, dtype=float)[np.asarray(labels) == 0]
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return 0.0
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def point_result(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, Any]:
    counts = integer_counts(labels, predictions)
    result = {
        **counts,
        **metrics_from_counts(counts["TP"], counts["FN"], counts["FP"], counts["TN"]),
    }
    if scores is not None:
        result["auroc"] = binary_auroc(labels, scores)
    return result


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def fact_pair_units(pair_rows: list[dict[str, dict[str, Any]]]) -> tuple[tuple[int, ...], ...]:
    """Return connected pair units, merging pairs that share a fact hash."""

    union = UnionFind(len(pair_rows))
    fact_owner: dict[str, int] = {}
    for pair_index, pair in enumerate(pair_rows):
        for class_name in ("HCE", "HCC"):
            fact_key = str(pair[class_name]["fact_key_sha256"])
            if fact_key in fact_owner:
                union.union(pair_index, fact_owner[fact_key])
            else:
                fact_owner[fact_key] = pair_index
    grouped: dict[int, list[int]] = {}
    for pair_index in range(len(pair_rows)):
        grouped.setdefault(union.find(pair_index), []).append(pair_index)
    return tuple(tuple(values) for _, values in sorted(grouped.items()))


def validate_no_forbidden_path(path: Path) -> None:
    if "invalidated" in str(path).casefold():
        raise ValueError(f"invalidated artifact is forbidden: {path}")


def load_cell(
    *,
    cell: str,
    slug: str,
    application_root: Path,
    score_root: Path,
    baseline_root: Path,
    application_authorization: dict[str, Any],
    application_authorization_sha256: str,
) -> CellEvaluation:
    application_path = application_root / f"{slug}-frozen-application.json"
    score_path = score_root / f"{slug}-frozen-scores.json"
    baseline_path = baseline_root / f"{slug}-locked-frozen-matrix.json"
    for path in (application_path, score_path, baseline_path):
        validate_no_forbidden_path(path)

    application = read_json(application_path)
    score_payload = read_json(score_path)
    baseline = read_json(baseline_path)
    authorized_cell = application_authorization.get("cells", {}).get(cell)
    if not isinstance(authorized_cell, Mapping):
        raise ValueError(f"application authorization lacks cell: {cell}")
    if sha256_file(application_path) != authorized_cell.get("application_manifest_sha256"):
        raise ValueError(f"application manifest SHA mismatch: {cell}")
    if application.get("cell") != cell or application.get("cohort_kind") != "frozen_test":
        raise ValueError(f"invalid frozen application identity: {cell}")
    if application.get("locked_accessed") is not True:
        raise ValueError(f"application does not declare frozen access: {cell}")
    if application.get("selected_method") != RUN6_METHOD:
        raise ValueError(f"application selected method differs from Run6: {cell}")
    records = application.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"application has no rows: {cell}")
    for record in records:
        if set(record) - ALLOWED_APPLICATION_RECORD_FIELDS:
            raise ValueError(f"application row exposes a forbidden field: {cell}")
        if record.get("class_stratum") not in ("HCE", "HCC"):
            raise ValueError(f"invalid class stratum: {cell}")
    ids = [str(record["sample_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate application sample ID: {cell}")
    by_id = {str(record["sample_id"]): record for record in records}

    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        pair = pairs.setdefault(str(record["pair_id"]), {})
        class_name = str(record["class_stratum"])
        if class_name in pair:
            raise ValueError(f"duplicate class within pair: {cell}/{record['pair_id']}")
        pair[class_name] = record
    if any(set(pair) != {"HCE", "HCC"} for pair in pairs.values()):
        raise ValueError(f"each pair must contain one HCE and one HCC: {cell}")
    ordered_pair_ids = tuple(sorted(pairs))
    pair_rows = [pairs[pair_id] for pair_id in ordered_pair_ids]
    units = fact_pair_units(pair_rows)
    pair_sample_ids = [
        (str(pair["HCE"]["sample_id"]), str(pair["HCC"]["sample_id"]))
        for pair in pair_rows
    ]

    if score_payload.get("cell") != cell or score_payload.get("selected_method") != RUN6_METHOD:
        raise ValueError(f"Run6 score identity mismatch: {cell}")
    required_false = (
        "refit_performed",
        "threshold_search_performed",
        "layer_selection_performed",
        "class_stratum_used_for_prediction",
    )
    if any(score_payload.get(field) is not False for field in required_false):
        raise ValueError(f"Run6 score violates frozen prediction guarantees: {cell}")
    if score_payload.get("application_authorization_sha256") != application_authorization_sha256:
        raise ValueError(f"Run6 score authorization SHA mismatch: {cell}")
    if score_payload.get("application_manifest_sha256") != sha256_file(application_path):
        raise ValueError(f"Run6 score application SHA mismatch: {cell}")
    run6 = score_payload.get("methods", {}).get(RUN6_METHOD)
    if not isinstance(run6, Mapping):
        raise ValueError(f"Run6 score lacks selected method: {cell}")
    run6_threshold = float(run6["frozen_validation_operating_point"]["threshold"])
    run6_records = run6.get("records", [])
    run6_by_id = {str(record["sample_id"]): record for record in run6_records}
    if len(run6_by_id) != len(run6_records) or set(run6_by_id) != set(by_id):
        raise ValueError(f"Run6 score alignment failure: {cell}")
    run6_scores = np.asarray(
        [[float(run6_by_id[hce]["score"]), float(run6_by_id[hcc]["score"])] for hce, hcc in pair_sample_ids]
    )
    run6_predictions = run6_scores >= run6_threshold
    declared_run6_predictions = np.asarray(
        [
            [bool(run6_by_id[hce]["predicted_hce"]), bool(run6_by_id[hcc]["predicted_hce"])]
            for hce, hcc in pair_sample_ids
        ]
    )
    if not np.array_equal(run6_predictions, declared_run6_predictions):
        raise ValueError(f"Run6 declared predictions differ from frozen threshold: {cell}")

    if baseline.get("cell") != cell or baseline.get("locked_threshold_selected") is not False:
        raise ValueError(f"baseline matrix is not frozen: {cell}")
    predictions = {"mosaic_v1_run6": run6_predictions}
    scores = {"mosaic_v1_run6": run6_scores}
    thresholds = {"mosaic_v1_run6": run6_threshold}
    statuses = {"mosaic_v1_run6": "frozen_run6_learned_gate_no_phenotype"}
    labels = np.asarray([[1, 0]] * len(pair_rows), dtype=int)
    flat_labels = labels.reshape(-1)
    if set(baseline.get("methods", {})) != set(FROZEN_MATRIX_METHODS):
        raise ValueError(f"baseline matrix method set differs from the frozen 17-method protocol: {cell}")
    for baseline_name in FROZEN_MATRIX_METHODS:
        item = baseline.get("methods", {}).get(baseline_name)
        if not isinstance(item, Mapping):
            raise ValueError(f"baseline method missing: {cell}/{baseline_name}")
        item_records = item.get("scores", [])
        item_by_id = {str(record["sample_id"]): record for record in item_records}
        if len(item_by_id) != len(item_records) or set(item_by_id) != set(by_id):
            raise ValueError(f"baseline score alignment failure: {cell}/{baseline_name}")
        for sample_id, record in item_by_id.items():
            if record.get("label") != by_id[sample_id]["class_stratum"]:
                raise ValueError(f"baseline label mismatch: {cell}/{baseline_name}/{sample_id}")
        threshold = float(item["metrics"]["threshold"])
        method_scores = np.asarray(
            [[float(item_by_id[hce]["score"]), float(item_by_id[hcc]["score"])] for hce, hcc in pair_sample_ids]
        )
        method_predictions = method_scores >= threshold
        recomputed = point_result(
            flat_labels,
            method_predictions.reshape(-1),
            method_scores.reshape(-1),
        )
        for key in ("TP", "FN", "FP", "TN"):
            if recomputed[key] != int(item["metrics"][key]):
                raise ValueError(f"baseline metric mismatch: {cell}/{baseline_name}/{key}")
        for key in POINT_METRICS:
            if abs(float(recomputed[key]) - float(item["metrics"][key])) > 1e-12:
                raise ValueError(f"baseline metric mismatch: {cell}/{baseline_name}/{key}")
        predictions[baseline_name] = method_predictions
        scores[baseline_name] = method_scores
        thresholds[baseline_name] = threshold
        statuses[baseline_name] = str(item.get("status", "unknown"))

    return CellEvaluation(
        cell=cell,
        slug=slug,
        pair_ids=ordered_pair_ids,
        unit_pair_indices=units,
        thresholds=thresholds,
        predictions=predictions,
        scores=scores,
        statuses=statuses,
        source_sha256={
            str(application_path): sha256_file(application_path),
            str(score_path): sha256_file(score_path),
            str(baseline_path): sha256_file(baseline_path),
        },
    )


def point_metrics(cell: CellEvaluation) -> dict[str, dict[str, Any]]:
    labels = np.asarray([[1, 0]] * len(cell.pair_ids), dtype=int).reshape(-1)
    return {
        method: {
            "display_name": METHOD_DISPLAY_NAMES.get(method, method),
            "method_role": METHOD_ROLES.get(method, "published_or_standard_baseline"),
            "threshold": float(cell.thresholds[method]),
            "status": cell.statuses[method],
            **point_result(
                labels,
                cell.predictions[method].reshape(-1),
                cell.scores[method].reshape(-1),
            ),
        }
        for method in METHODS
    }


def bootstrap_cell_counts(
    cell: CellEvaluation,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, dict[str, np.ndarray]]:
    unit_sizes = np.asarray([len(unit) for unit in cell.unit_pair_indices], dtype=int)
    selected_units = rng.integers(
        0,
        len(cell.unit_pair_indices),
        size=(draws, len(cell.unit_pair_indices)),
    )
    positive_totals = unit_sizes[selected_units].sum(axis=1)
    output: dict[str, dict[str, np.ndarray]] = {}
    for method in BOOTSTRAP_METHODS:
        pair_predictions = cell.predictions[method]
        unit_tp = np.asarray(
            [int(pair_predictions[list(unit), 0].sum()) for unit in cell.unit_pair_indices]
        )
        unit_fp = np.asarray(
            [int(pair_predictions[list(unit), 1].sum()) for unit in cell.unit_pair_indices]
        )
        tp = unit_tp[selected_units].sum(axis=1)
        fp = unit_fp[selected_units].sum(axis=1)
        output[method] = {
            "TP": tp,
            "FN": positive_totals - tp,
            "FP": fp,
            "TN": positive_totals - fp,
        }
    return output


def metric_arrays(counts: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return metrics_from_counts(counts["TP"], counts["FN"], counts["FP"], counts["TN"])


def interval(point: float, values: np.ndarray) -> dict[str, Any]:
    return {
        "point": float(point),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def method_interval_payload(
    point: Mapping[str, float],
    values: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        metric: interval(float(point[metric]), values[metric])
        for metric in BOOTSTRAP_METRICS
    }


def difference_interval_payload(
    left_point: Mapping[str, float],
    right_point: Mapping[str, float],
    left_values: Mapping[str, np.ndarray],
    right_values: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        metric: interval(
            float(left_point[metric]) - float(right_point[metric]),
            left_values[metric] - right_values[metric],
        )
        for metric in BOOTSTRAP_METRICS
    }


def write_main_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope",
        "cell",
        "method",
        "display_name",
        "method_role",
        "status",
        "reason",
        "threshold",
        "TP",
        "FN",
        "FP",
        "TN",
        *POINT_METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_bootstrap_csv(path: Path, payload: Mapping[str, Any]) -> None:
    rows = []
    scopes = [
        *(('cell', cell, value) for cell, value in payload["cells"].items()),
        ("macro_equal_cell", "macro_equal_cell", payload["macro_equal_cell_average"]),
        ("pooled", "pooled", payload["pooled_supplement"]),
    ]
    for scope, cell, value in scopes:
        for method, metrics in value["methods"].items():
            for metric, result in metrics.items():
                rows.append(
                    {
                        "scope": scope,
                        "cell": cell,
                        "quantity_type": "method",
                        "item": method,
                        "metric": metric,
                        "point": result["point"],
                        "ci_low": result["ci95"][0],
                        "ci_high": result["ci95"][1],
                        "draws": payload["draws"],
                    }
                )
        for comparison, metrics in value["paired_differences"].items():
            for metric, result in metrics.items():
                rows.append(
                    {
                        "scope": scope,
                        "cell": cell,
                        "quantity_type": "paired_difference",
                        "item": comparison,
                        "metric": metric,
                        "point": result["point"],
                        "ci_low": result["ci95"][0],
                        "ci_high": result["ci95"][1],
                        "draws": payload["draws"],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scope",
                "cell",
                "quantity_type",
                "item",
                "metric",
                "point",
                "ci_low",
                "ci_high",
                "draws",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def compile_results(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (
        args.application_root,
        args.score_root,
        args.baseline_root,
        args.application_authorization,
    ):
        validate_no_forbidden_path(path)
    authorization = read_json(args.application_authorization)
    if authorization.get("authorized") is not True or authorization.get("one_shot") is not True:
        raise ValueError("application authorization is not an authorized one-shot artifact")
    if authorization.get("purpose") != "one_confidence_apply_frozen_test_once":
        raise ValueError("unexpected application authorization purpose")
    authorization_sha = sha256_file(args.application_authorization)
    if set(authorization.get("cells", {})) != {cell for cell, _ in CELLS}:
        raise ValueError("application authorization must contain exactly five frozen cells")

    cells = [
        load_cell(
            cell=cell,
            slug=slug,
            application_root=args.application_root,
            score_root=args.score_root,
            baseline_root=args.baseline_root,
            application_authorization=authorization,
            application_authorization_sha256=authorization_sha,
        )
        for cell, slug in CELLS
    ]
    point_by_cell = {cell.cell: point_metrics(cell) for cell in cells}

    def unavailable_result(_scope: str) -> dict[str, Any]:
        definition = UNAVAILABLE_METHODS["pale"]
        return {
            "display_name": definition["display_name"],
            "method_role": METHOD_ROLES["pale"],
            "status": definition["status"],
            "reason": definition["reason"],
            "threshold": None,
            "TP": None,
            "FN": None,
            "FP": None,
            "TN": None,
            **{metric: None for metric in POINT_METRICS},
        }

    macro = {
        method: {
            "display_name": METHOD_DISPLAY_NAMES.get(method, method),
            "method_role": METHOD_ROLES.get(method, "published_or_standard_baseline"),
            "status": "equal_cell_average",
            **{
                metric: float(np.mean([point_by_cell[cell.cell][method][metric] for cell in cells]))
                for metric in POINT_METRICS
            },
        }
        for method in METHODS
    }
    pooled = {}
    for method in METHODS:
        counts = {
            key: int(sum(point_by_cell[cell.cell][method][key] for cell in cells))
            for key in ("TP", "FN", "FP", "TN")
        }
        pooled[method] = {
            "display_name": METHOD_DISPLAY_NAMES.get(method, method),
            "method_role": METHOD_ROLES.get(method, "published_or_standard_baseline"),
            "status": "pooled_supplement",
            **counts,
            **metrics_from_counts(counts["TP"], counts["FN"], counts["FP"], counts["TN"]),
            "auroc": binary_auroc(
                np.concatenate(
                    [np.asarray([[1, 0]] * len(cell.pair_ids), dtype=int).reshape(-1) for cell in cells]
                ),
                np.concatenate([cell.scores[method].reshape(-1) for cell in cells]),
            ),
        }
    cell_method_tables = {
        cell.cell: {
            **point_by_cell[cell.cell],
            "pale": unavailable_result("cell"),
        }
        for cell in cells
    }
    macro["pale"] = unavailable_result("macro_equal_cell")
    pooled["pale"] = unavailable_result("pooled_supplement")

    source_sha = {str(args.application_authorization): authorization_sha}
    for cell in cells:
        source_sha.update(cell.source_sha256)
    main_payload = {
        "schema_version": "one-confidence-mosaic-v1-run6-frozen-results-v2",
        "positive_class": "HCE",
        "selected_method": RUN6_METHOD,
        "available_scored_methods": list(METHODS),
        "unavailable_methods": UNAVAILABLE_METHODS,
        "locked_application_authorization_id": authorization["authorization_id"],
        "locked_application_authorization_sha256": authorization_sha,
        "operating_point": "validation-frozen threshold; no locked retuning",
        "refit_performed": False,
        "threshold_search_performed": False,
        "layer_selection_performed": False,
        "cells": {
            cell.cell: {
                "pairs": len(cell.pair_ids),
                "rows": 2 * len(cell.pair_ids),
                "fact_pair_bootstrap_units": len(cell.unit_pair_indices),
                "max_pairs_per_bootstrap_unit": max(map(len, cell.unit_pair_indices)),
                "methods": cell_method_tables[cell.cell],
            }
            for cell in cells
        },
        "macro_equal_cell_average": macro,
        "pooled_supplement": pooled,
        "source_sha256": dict(sorted(source_sha.items())),
    }

    rng = np.random.default_rng(args.seed)
    count_draws = {
        cell.cell: bootstrap_cell_counts(cell, rng, args.draws) for cell in cells
    }
    metric_draws = {
        cell.cell: {
            method: metric_arrays(count_draws[cell.cell][method])
            for method in BOOTSTRAP_METHODS
        }
        for cell in cells
    }

    cell_bootstrap = {}
    for cell in cells:
        cell_name = cell.cell
        cell_bootstrap[cell_name] = {
            "pairs": len(cell.pair_ids),
            "fact_pair_units": len(cell.unit_pair_indices),
            "max_pairs_per_unit": max(map(len, cell.unit_pair_indices)),
            "methods": {
                method: method_interval_payload(
                    point_by_cell[cell_name][method], metric_draws[cell_name][method]
                )
                for method in BOOTSTRAP_METHODS
            },
            "paired_differences": {
                "mosaic_v1_run6_minus_mosaic_v0": difference_interval_payload(
                    point_by_cell[cell_name]["mosaic_v1_run6"],
                    point_by_cell[cell_name]["minimal_gated_mosaic"],
                    metric_draws[cell_name]["mosaic_v1_run6"],
                    metric_draws[cell_name]["minimal_gated_mosaic"],
                ),
                "mosaic_v1_run6_minus_mean_answer_hidden_linear_probe": difference_interval_payload(
                    point_by_cell[cell_name]["mosaic_v1_run6"],
                    point_by_cell[cell_name]["mean_answer_hidden_linear_probe"],
                    metric_draws[cell_name]["mosaic_v1_run6"],
                    metric_draws[cell_name]["mean_answer_hidden_linear_probe"],
                ),
            },
        }

    macro_draws = {
        method: {
            metric: np.mean(
                np.stack([metric_draws[cell.cell][method][metric] for cell in cells]),
                axis=0,
            )
            for metric in BOOTSTRAP_METRICS
        }
        for method in BOOTSTRAP_METHODS
    }
    pooled_count_draws = {
        method: {
            key: np.sum(
                np.stack([count_draws[cell.cell][method][key] for cell in cells]),
                axis=0,
            )
            for key in ("TP", "FN", "FP", "TN")
        }
        for method in BOOTSTRAP_METHODS
    }
    pooled_draws = {
        method: metric_arrays(pooled_count_draws[method])
        for method in BOOTSTRAP_METHODS
    }

    bootstrap_payload = {
        "schema_version": "one-confidence-mosaic-v1-run6-pair-bootstrap-v1",
        "positive_class": "HCE",
        "draws": args.draws,
        "seed": args.seed,
        "ci": "percentile 95% interval",
        "resampling_unit": "connected fact/question groups while preserving complete matched HCE/HCC pairs",
        "paired_comparison": True,
        "thresholds_frozen": True,
        "cells": cell_bootstrap,
        "macro_equal_cell_average": {
            "methods": {
                method: method_interval_payload(macro[method], macro_draws[method])
                for method in BOOTSTRAP_METHODS
            },
            "paired_differences": {
                "mosaic_v1_run6_minus_mosaic_v0": difference_interval_payload(
                    macro["mosaic_v1_run6"],
                    macro["minimal_gated_mosaic"],
                    macro_draws["mosaic_v1_run6"],
                    macro_draws["minimal_gated_mosaic"],
                ),
                "mosaic_v1_run6_minus_mean_answer_hidden_linear_probe": difference_interval_payload(
                    macro["mosaic_v1_run6"],
                    macro["mean_answer_hidden_linear_probe"],
                    macro_draws["mosaic_v1_run6"],
                    macro_draws["mean_answer_hidden_linear_probe"],
                ),
            },
        },
        "pooled_supplement": {
            "methods": {
                method: method_interval_payload(pooled[method], pooled_draws[method])
                for method in BOOTSTRAP_METHODS
            },
            "paired_differences": {
                "mosaic_v1_run6_minus_mosaic_v0": difference_interval_payload(
                    pooled["mosaic_v1_run6"],
                    pooled["minimal_gated_mosaic"],
                    pooled_draws["mosaic_v1_run6"],
                    pooled_draws["minimal_gated_mosaic"],
                ),
                "mosaic_v1_run6_minus_mean_answer_hidden_linear_probe": difference_interval_payload(
                    pooled["mosaic_v1_run6"],
                    pooled["mean_answer_hidden_linear_probe"],
                    pooled_draws["mosaic_v1_run6"],
                    pooled_draws["mean_answer_hidden_linear_probe"],
                ),
            },
        },
        "source_sha256": dict(sorted(source_sha.items())),
    }
    return main_payload, bootstrap_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--application-authorization", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-json", type=Path, required=True)
    parser.add_argument("--bootstrap-csv", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026072912)
    args = parser.parse_args()
    if args.draws <= 0:
        raise ValueError("--draws must be positive")
    main_payload, bootstrap_payload = compile_results(args)
    atomic_json(args.output_json, main_payload)
    write_main_csv(
        args.output_csv,
        [
            {
                "scope": "cell",
                "cell": cell,
                "method": method,
                **result,
            }
            for cell, cell_payload in main_payload["cells"].items()
            for method, result in cell_payload["methods"].items()
        ]
        + [
            {
                "scope": "macro_equal_cell",
                "cell": "macro_equal_cell",
                "method": method,
                "status": "equal_cell_average",
                "threshold": "",
                "TP": "",
                "FN": "",
                "FP": "",
                "TN": "",
                **result,
            }
            for method, result in main_payload["macro_equal_cell_average"].items()
        ]
        + [
            {
                "scope": "pooled",
                "cell": "pooled",
                "method": method,
                "status": "pooled_supplement",
                "threshold": "",
                **result,
            }
            for method, result in main_payload["pooled_supplement"].items()
        ],
    )
    atomic_json(args.bootstrap_json, bootstrap_payload)
    write_bootstrap_csv(args.bootstrap_csv, bootstrap_payload)
    print(
        json.dumps(
            {
                "macro_mosaic_v1": main_payload["macro_equal_cell_average"]["mosaic_v1_run6"],
                "macro_difference_vs_mosaic_v0": bootstrap_payload["macro_equal_cell_average"]["paired_differences"]["mosaic_v1_run6_minus_mosaic_v0"],
                "macro_difference_vs_mean_answer_probe": bootstrap_payload["macro_equal_cell_average"]["paired_differences"]["mosaic_v1_run6_minus_mean_answer_hidden_linear_probe"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
