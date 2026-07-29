#!/usr/bin/env python3
"""Combine validation ablations with the sole authorized frozen Run6 result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compile_ablation(
    validation_path: Path, frozen_detection_path: Path
) -> dict[str, Any]:
    validation = load_json(validation_path)
    frozen = load_json(frozen_detection_path)
    validation_rows = validation["rows"]
    if any(row.get("split") != "validation" for row in validation_rows):
        raise ValueError("ablation source contains a non-validation row")
    frozen_rows = []
    for cell, payload in frozen["cells"].items():
        method = payload["methods"]["mosaic_v1_run6"]
        frozen_rows.append(
            {
                "aggregation": "cell",
                "cell": cell,
                "split": "retrospective_frozen_test",
                "method": "learned_gate_no_phenotype",
                "status": method["status"],
                "TP": method["TP"],
                "FN": method["FN"],
                "FP": method["FP"],
                "TN": method["TN"],
                "accuracy": method["accuracy"],
                "precision": method["precision"],
                "recall": method["recall"],
                "fpr": method["fpr"],
                "f1": method["f1"],
                "threshold": method["threshold"],
                "eligible": True,
                "auroc_supplementary": None,
            }
        )
    macro = frozen["macro_equal_cell_average"]["mosaic_v1_run6"]
    frozen_rows.append(
        {
            "aggregation": "macro_equal_cell",
            "cell": "macro_equal_cell",
            "split": "retrospective_frozen_test",
            "method": "learned_gate_no_phenotype",
            "status": "equal_cell_average",
            "TP": None,
            "FN": None,
            "FP": None,
            "TN": None,
            "accuracy": macro["accuracy"],
            "precision": macro["precision"],
            "recall": macro["recall"],
            "fpr": macro["fpr"],
            "f1": macro["f1"],
            "threshold": None,
            "eligible": True,
            "auroc_supplementary": None,
        }
    )
    return {
        "schema_version": "one-confidence-mosaic-v1-ablation-delivery-v1",
        "positive_class": "HCE",
        "fresh_accessed": False,
        "validation_scope": {
            "rows": validation_rows,
            "selection_role": "all ablations and hyperparameter comparisons",
        },
        "retrospective_frozen_test_scope": {
            "rows": frozen_rows,
            "application_count": 1,
            "selected_method_only": True,
            "refit_performed": False,
            "threshold_search_performed": False,
            "unrun_ablation_note": (
                "No other ablation was applied after unlock; a complete frozen-test "
                "ablation table would violate the one-shot Run6 application freeze."
            ),
        },
        "source_sha256": {
            validation_path.as_posix(): sha256(validation_path),
            frozen_detection_path.as_posix(): sha256(frozen_detection_path),
        },
    }


def flatten_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scope in ("validation_scope", "retrospective_frozen_test_scope"):
        for source_row in payload[scope]["rows"]:
            row = dict(source_row)
            row["result_scope"] = scope.removesuffix("_scope")
            rows.append(row)
    return rows


def write_outputs(payload: dict[str, Any], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rows = flatten_rows(payload)
    fields = [
        "result_scope",
        "aggregation",
        "cell",
        "split",
        "method",
        "status",
        "eligible",
        "threshold",
        "TP",
        "FN",
        "FP",
        "TN",
        "accuracy",
        "precision",
        "recall",
        "fpr",
        "f1",
        "auroc_supplementary",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--frozen-detection", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = compile_ablation(args.validation, args.frozen_detection)
    write_outputs(payload, args.output_json, args.output_csv)
    print(
        json.dumps(
            {
                "output_json_sha256": sha256(args.output_json),
                "output_csv_sha256": sha256(args.output_csv),
                "validation_rows": len(payload["validation_scope"]["rows"]),
                "frozen_rows": len(payload["retrospective_frozen_test_scope"]["rows"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
