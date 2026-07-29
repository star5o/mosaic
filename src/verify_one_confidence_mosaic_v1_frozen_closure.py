#!/usr/bin/env python3
"""Verify canonical retrospective-frozen MOSAIC-v1 closure artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


EXPECTED_ROWS = {
    "mistral7|TriviaQA": 162,
    "qwen8|TriviaQA": 130,
    "llama31|TriviaQA": 126,
    "qwen14|PopQA": 122,
    "nemo12|PopQA": 114,
}
FORBIDDEN_FRESH_TOKENS = ("fresh_holdout", "fresh_method_holdout")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def assert_close(left: float | None, right: float | None, context: str) -> None:
    if left is None or right is None:
        if left is not right:
            raise ValueError(f"{context}: null mismatch")
        return
    if abs(float(left) - float(right)) > 1e-12:
        raise ValueError(f"{context}: {left} != {right}")


def verify(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "frozen_closure_artifact_sha256.json"
    phenotype_path = output_dir / "phenotype_labels_frozen_test.json"
    table_path = output_dir / "table_detection_by_phenotype.json"
    risk_path = output_dir / "figure_risk_coverage.json"
    cases_path = output_dir / "case_studies.json"
    mitigation_path = output_dir / "table_mitigation.json"
    for path in (
        manifest_path,
        phenotype_path,
        table_path,
        risk_path,
        cases_path,
        mitigation_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = read_json(manifest_path)
    for raw_path, expected_sha in manifest.get("outputs", {}).items():
        path = Path(raw_path)
        if sha256_file(path) != expected_sha:
            raise ValueError(f"frozen closure output SHA mismatch: {path}")
    for raw_path in manifest.get("sources", {}):
        lowered = raw_path.casefold()
        if any(token in lowered for token in FORBIDDEN_FRESH_TOKENS):
            raise ValueError(f"fresh source leaked into frozen closure: {raw_path}")
    if manifest.get("fresh_correctness_accessed") is not False:
        raise ValueError("frozen closure manifest must declare no fresh correctness access")

    phenotypes = read_json(phenotype_path)
    records = phenotypes.get("records")
    if not isinstance(records, list) or len(records) != sum(EXPECTED_ROWS.values()):
        raise ValueError("frozen phenotype row count mismatch")
    ids = [str(row["sample_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate frozen phenotype sample ID")
    by_cell = {cell: [row for row in records if row["cell"] == cell] for cell in EXPECTED_ROWS}
    for cell, expected in EXPECTED_ROWS.items():
        rows = by_cell[cell]
        if len(rows) != expected:
            raise ValueError(f"frozen phenotype cell row count mismatch: {cell}")
        hce = [row for row in rows if row["label"] == "HCE"]
        hcc = [row for row in rows if row["label"] == "HCC"]
        if len(hce) != len(hcc) or len(rows) != 2 * len(hce):
            raise ValueError(f"frozen phenotype class balance mismatch: {cell}")
        if any(row["internal_competitive_assessed"] for row in rows):
            raise ValueError(f"post-hoc internal phenotype detected: {cell}")
        if any(row["assessed_persistent"] for row in rows):
            raise ValueError(f"post-hoc persistent phenotype detected: {cell}")

    table = read_json(table_path)
    if table.get("phenotype_status") != "observational_and_partial_not_causal":
        raise ValueError("frozen phenotype table overstates causal status")
    table_rows = table.get("rows")
    if not isinstance(table_rows, list):
        raise ValueError("frozen phenotype table rows missing")
    for cell, rows in by_cell.items():
        cell_table = {
            row["phenotype"]: row
            for row in table_rows
            if row["aggregation"] == "cell" and row["cell"] == cell
        }
        for phenotype in ("behavioral_recoverable", "unassessed_HCE"):
            subset = [row for row in rows if row["phenotype"] == phenotype]
            tp = sum(bool(row["predicted_hce"]) for row in subset)
            expected_recall = tp / len(subset) if subset else None
            if int(cell_table[phenotype]["count"]) != len(subset):
                raise ValueError(f"phenotype count mismatch: {cell}/{phenotype}")
            assert_close(cell_table[phenotype]["recall"], expected_recall, f"recall {cell}/{phenotype}")
        hcc = [row for row in rows if row["label"] == "HCC"]
        fp = sum(bool(row["predicted_hce"]) for row in hcc)
        assert_close(cell_table["HCC_control"]["fpr"], fp / len(hcc), f"FPR {cell}")
        for phenotype in ("internal_competitive", "assessed_persistent"):
            if cell_table[phenotype]["count"] != 0 or cell_table[phenotype]["recall"] is not None:
                raise ValueError(f"unassessed phenotype received a result: {cell}/{phenotype}")

    risk = read_json(risk_path)
    risk_rows = risk.get("rows")
    if not isinstance(risk_rows, list):
        raise ValueError("risk-coverage rows missing")
    for cell in EXPECTED_ROWS:
        rows = sorted(
            (
                row
                for row in risk_rows
                if row["aggregation"] == "cell" and row["cell"] == cell
            ),
            key=lambda row: row["answer_coverage"],
        )
        if [row["answer_coverage"] for row in rows] != [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            raise ValueError(f"risk-coverage grid mismatch: {cell}")
        if any(rows[index]["answered"] > rows[index + 1]["answered"] for index in range(len(rows) - 1)):
            raise ValueError(f"risk-coverage answered count is not monotonic: {cell}")
        assert_close(rows[-1]["selective_accuracy"], 0.5, f"full coverage accuracy {cell}")

    cases = read_json(cases_path)
    if cases.get("fresh_correctness_accessed") is not False:
        raise ValueError("case studies declare fresh correctness access")
    if cases.get("retrospective_frozen_only") is not True:
        raise ValueError("case studies are not explicitly retrospective-only")
    case_rows = cases.get("rows")
    if not isinstance(case_rows, list) or len(case_rows) != 20:
        raise ValueError("expected two FP and two FN cases per frozen cell")
    for cell in EXPECTED_ROWS:
        rows = [row for row in case_rows if row["cell"] == cell]
        if sum(row["outcome"] == "FP" for row in rows) != 2:
            raise ValueError(f"frozen FP case count mismatch: {cell}")
        if sum(row["outcome"] == "FN" for row in rows) != 2:
            raise ValueError(f"frozen FN case count mismatch: {cell}")
    for row in case_rows:
        if not row.get("question") or not row.get("extracted_answer") or not row.get("gold_aliases"):
            raise ValueError(f"real frozen case content missing: {row.get('sample_id')}")
        if str(row.get("dtype", "")).casefold() not in {"bf16", "bfloat16", "torch.bfloat16"}:
            raise ValueError(f"non-BF16 case study: {row.get('sample_id')}")

    mitigation = read_json(mitigation_path)
    if mitigation.get("schema_version") != "one-confidence-mosaic-v1-mitigation-delivery-v1":
        raise ValueError("canonical mitigation delivery schema mismatch")
    frozen_scope = mitigation.get("retrospective_frozen_test_scope")
    if not isinstance(frozen_scope, Mapping) or frozen_scope.get("decisions_frozen_before_labels") is not True:
        raise ValueError("canonical mitigation lacks label-blind frozen decision proof")
    if "unavailable" not in str(mitigation.get("steering_status", "")).casefold():
        raise ValueError("canonical mitigation must keep STEER unavailable")

    return {
        "schema_version": "one-confidence-mosaic-v1-frozen-closure-integrity-v1",
        "status": "passed",
        "retrospective_frozen_rows": len(records),
        "retrospective_frozen_pairs": len(records) // 2,
        "cells": EXPECTED_ROWS,
        "phenotype_status": table["phenotype_status"],
        "case_rows": len(case_rows),
        "fresh_correctness_accessed": False,
        "steering_available": False,
        "verified_sha256": {
            str(path): sha256_file(path)
            for path in (manifest_path, phenotype_path, table_path, risk_path, cases_path, mitigation_path)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.output_dir)
    atomic_json(args.report, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": report["retrospective_frozen_rows"],
                "pairs": report["retrospective_frozen_pairs"],
                "report": str(args.report),
                "sha256": sha256_file(args.report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
