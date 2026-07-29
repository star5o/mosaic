#!/usr/bin/env python3
"""Sanitize legacy locked artifacts for the frozen Run6 application.

The earlier baseline run already extracted BF16 hidden states and K=4 sampling
signals for the exact 654-row cohort. Those files incorrectly carried public
metadata and supervision fields. This command is authorization-gated and writes
new one-shot artifacts containing only inference-time inputs and identifiers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from one_confidence_frozen_source_access import load_rows, sha256_file


def _stage_input(
    authorization_path: Path,
    benchmark_path: Path,
    cell: str,
    stage: str,
    output_path: Path,
    input_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, dict[str, Any]]:
    benchmark, selected, frozen, authorization_id = load_rows(
        benchmark_path=benchmark_path,
        cell=cell,
        authorization_path=authorization_path,
        stage=stage,
        output_path=output_path,
    )
    if not frozen or not authorization_id:
        raise ValueError("auxiliary source materialization requires frozen authorization")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    stage_spec = authorization["cells"][cell]["stages"][stage]
    declared = stage_spec.get("input", {})
    if str(input_path) != str(declared.get("path")):
        raise ValueError(f"input path differs from authorization for {cell}/{stage}")
    if sha256_file(input_path) != declared.get("sha256"):
        raise ValueError(f"input SHA256 differs from authorization for {cell}/{stage}")
    rows = [row for _, row in selected]
    return benchmark, rows, authorization_id, stage_spec


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sanitize_internal(
    authorization: Path,
    benchmark: Path,
    cell: str,
    source: Path,
    output: Path,
) -> dict[str, Any]:
    _, rows, authorization_id, stage_spec = _stage_input(
        authorization, benchmark, cell, "sanitized_internal_states", output, source
    )
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    expected_ids = [str(row["sample_id"]) for row in rows]
    source_ids = [str(value) for value in artifact.get("row_ids", [])]
    if len(source_ids) != len(set(source_ids)) or set(source_ids) != set(expected_ids):
        raise ValueError("legacy hidden-state IDs differ from the authorized cohort")
    if artifact.get("cell") != cell:
        raise ValueError("legacy hidden-state cell mismatch")
    if str(artifact.get("model_revision")) != str(stage_spec.get("model_revision")):
        raise ValueError("legacy hidden-state model revision mismatch")
    if str(artifact.get("dtype")) != str(stage_spec.get("dtype")):
        raise ValueError("legacy hidden-state dtype mismatch")
    order = {sample_id: index for index, sample_id in enumerate(source_ids)}
    indices = [order[sample_id] for sample_id in expected_ids]
    states = {
        position: tensor[indices].contiguous()
        for position, tensor in artifact.get("states", {}).items()
    }
    if set(states) != {"prompt_end", "first_answer", "mean_answer"}:
        raise ValueError("legacy hidden-state artifact lacks a required position")
    payload = {
        "schema_version": "one-confidence-frozen-internal-states-v1",
        "cell": cell,
        "locked_accessed": True,
        "source_authorization_id": authorization_id,
        "model_revision": str(artifact["model_revision"]),
        "dtype": str(artifact["dtype"]),
        "row_ids": expected_ids,
        "layer_numbers": [int(value) for value in artifact["layer_numbers"]],
        "states": states,
        "source_sha256": {str(source): sha256_file(source)},
        "legacy_metadata_correction": {
            "source_declared_locked_accessed": artifact.get("locked_accessed"),
            "labels_removed": "labels" in artifact,
            "token_entropy_records_removed": "token_entropy_records" in artifact,
        },
    }
    _atomic_torch(output, payload)
    return payload


def sanitize_persistence(
    authorization: Path,
    benchmark: Path,
    cell: str,
    source: Path,
    output: Path,
) -> dict[str, Any]:
    _, rows, authorization_id, _stage_spec = _stage_input(
        authorization, benchmark, cell, "sanitized_persistence", output, source
    )
    artifact = json.loads(source.read_text(encoding="utf-8"))
    keyed = {str(row["sample_id"]): row for row in artifact.get("records", [])}
    expected_ids = [str(row["sample_id"]) for row in rows]
    if len(keyed) != len(artifact.get("records", [])) or set(keyed) != set(expected_ids):
        raise ValueError("legacy persistence IDs differ from the authorized cohort")
    records = [
        {
            "sample_id": sample_id,
            "split": "locked_test",
            "self_consistency_risk": float(keyed[sample_id]["self_consistency_risk"]),
            "semantic_entropy_risk": float(keyed[sample_id]["semantic_entropy_risk"]),
            "original_support": float(keyed[sample_id]["original_support"]),
        }
        for sample_id in expected_ids
    ]
    payload = {
        "schema_version": "one-confidence-frozen-persistence-features-v1",
        "cell": cell,
        "locked_accessed": True,
        "source_authorization_id": authorization_id,
        "contains_labels": False,
        "K": artifact.get("K"),
        "records": records,
        "source_sha256": {str(source): sha256_file(source)},
        "legacy_metadata_correction": {
            "labels_removed": any("label" in row for row in artifact.get("records", [])),
            "cluster_text_removed": True,
        },
    }
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("sanitize-internal", "sanitize-persistence"):
        item = subparsers.add_parser(name)
        item.add_argument("--authorization", type=Path, required=True)
        item.add_argument("--benchmark", type=Path, required=True)
        item.add_argument("--cell", required=True)
        item.add_argument("--source", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    function = sanitize_internal if args.command == "sanitize-internal" else sanitize_persistence
    payload = function(
        args.authorization, args.benchmark, args.cell, args.source, args.output
    )
    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "cell": payload["cell"],
                "locked_accessed": payload["locked_accessed"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
