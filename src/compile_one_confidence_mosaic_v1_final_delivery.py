#!/usr/bin/env python3
"""Compile canonical mitigation and case-study delivery artifacts.

This compiler does not fit, score, route, or select samples.  It only combines
the already-scoped public validation table with the pre-label-frozen Run6 test
outcomes and preserves the deterministic public case-study selection under an
explicitly public-scoped filename.  The canonical frozen ``case_studies.json``
is compiled independently and is never overwritten here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def compile_mitigation(
    public_path: Path,
    frozen_path: Path,
    decision_freeze_path: Path,
) -> dict[str, Any]:
    public = read_json(public_path)
    frozen = read_json(frozen_path)
    decision_freeze = read_json(decision_freeze_path)
    if public.get("public_only") is not True or public.get("locked_accessed") is not False:
        raise ValueError("public mitigation source has the wrong scope")
    if frozen.get("locked_accessed") is not True:
        raise ValueError("frozen mitigation source is not marked locked")
    if decision_freeze.get("decision_stage_is_label_blind") is not True:
        raise ValueError("frozen mitigation decisions were not label-blind")
    if frozen.get("decision_sha256") != decision_freeze.get("decision_sha256"):
        raise ValueError("frozen mitigation table/decision freeze SHA mismatch")
    if public.get("risk_method") != "learned_gate_no_phenotype":
        raise ValueError("public mitigation source is not Run6")
    if frozen.get("selected_method") != "learned_gate_no_phenotype":
        raise ValueError("frozen mitigation source is not Run6")
    return {
        "schema_version": "one-confidence-mosaic-v1-mitigation-delivery-v1",
        "positive_class": "HCE",
        "selected_method": "learned_gate_no_phenotype",
        "selection_and_routing_frozen_on_public_validation": True,
        "public_validation_scope": {
            "role": "method development and validation",
            "source_path": str(public_path),
            "source_sha256": sha256_file(public_path),
            "rows": public["rows"],
        },
        "retrospective_frozen_test_scope": {
            "role": "single authorized Run6 application; no post-test retuning",
            "source_path": str(frozen_path),
            "source_sha256": sha256_file(frozen_path),
            "decision_freeze_path": str(decision_freeze_path),
            "decision_freeze_sha256": sha256_file(decision_freeze_path),
            "decision_sha256": decision_freeze["decision_sha256"],
            "decisions_frozen_before_labels": True,
            "rows": frozen["rows"],
        },
        "steering_status": "unavailable; no full-cohort frozen generations and causal controls do not support steering",
    }


def mitigation_csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for scope_key in ("public_validation_scope", "retrospective_frozen_test_scope"):
        for row in payload[scope_key]["rows"]:
            output.append({"result_scope": scope_key, **row})
    return output


def write_union_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compile_case_studies(source_path: Path) -> dict[str, Any]:
    source = read_json(source_path)
    if source.get("selection_is_deterministic") is not True:
        raise ValueError("case-study source is not deterministically selected")
    rows = source.get("rows")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("expected two TP/FN/FP/TN cases for each of five cells")
    expected_outcomes = {"TP", "FN", "FP", "TN"}
    for row in rows:
        if row.get("outcome") not in expected_outcomes:
            raise ValueError("unexpected case-study outcome")
        if row.get("selection_rule") != "first two sample IDs per cell and confusion outcome":
            raise ValueError("case-study selection rule changed")
    return {
        "schema_version": "one-confidence-mosaic-v1-case-studies-delivery-v1",
        "public_only": True,
        "locked_accessed": False,
        "selection_role": "fixed qualitative error analysis; never used for method selection",
        "selection_rule": "first two sample IDs per cell and confusion outcome",
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--frozen-mitigation-dir", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.result_dir
    public_path = result_dir / "table_mitigation_run6_public_validation.json"
    frozen_path = args.frozen_mitigation_dir / "table_mitigation_locked_run6.json"
    decision_freeze_path = args.frozen_mitigation_dir / "mitigation_decision_freeze_run6.json"
    case_source_path = result_dir / "case_studies_public_validation.json"

    mitigation = compile_mitigation(public_path, frozen_path, decision_freeze_path)
    mitigation_json = result_dir / "table_mitigation.json"
    mitigation_csv = result_dir / "table_mitigation.csv"
    write_json(mitigation_json, mitigation)
    write_union_csv(mitigation_csv, mitigation_csv_rows(mitigation))

    cases = compile_case_studies(case_source_path)
    case_path = result_dir / "case_studies_public_validation_delivery.json"
    write_json(case_path, cases)
    print(
        json.dumps(
            {
                "table_mitigation.json": sha256_file(mitigation_json),
                "table_mitigation.csv": sha256_file(mitigation_csv),
                "case_studies_public_validation_delivery.json": sha256_file(case_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
