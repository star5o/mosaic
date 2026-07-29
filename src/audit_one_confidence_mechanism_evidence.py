#!/usr/bin/env python3
"""Compile a conservative audit of the public MOSAIC mechanism evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALL_CELLS = (
    "mistral7|TriviaQA",
    "qwen8|TriviaQA",
    "llama31|TriviaQA",
    "qwen14|PopQA",
    "nemo12|PopQA",
)


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


def compile_audit(result_dir: Path) -> dict[str, Any]:
    paths = {
        "verdict": result_dir / "mechanism_control_verdict.json",
        "controls": result_dir / "table_mechanism_controls.json",
        "patch": result_dir / "mechanism_patch_results.json",
        "erasure": result_dir / "mechanism_erasure_results.json",
        "phenotypes": result_dir / "phenotype_labels.json",
        "phenotype_detection": result_dir / "table_detection_by_phenotype_legacy_public_validation.json",
    }
    payloads = {name: read_json(path) for name, path in paths.items()}
    verdict = payloads["verdict"]
    controls = payloads["controls"]
    patch = payloads["patch"]
    erasure = payloads["erasure"]
    phenotypes = payloads["phenotypes"]
    phenotype_detection = payloads["phenotype_detection"]

    if len(controls.get("rows", [])) != 167:
        raise ValueError("canonical mechanism control table must contain 167 aggregate rows")
    for name, payload in (("patch", patch), ("erasure", erasure)):
        if payload.get("aggregate_rows_only") is not True:
            raise ValueError(f"{name} summary is not aggregate-only")
        if payload.get("locked_accessed") is not False or payload.get("fresh_accessed") is not False:
            raise ValueError(f"{name} summary accessed a frozen cohort")
        if payload.get("per_sample_records_read") is not False:
            raise ValueError(f"{name} summary unexpectedly read per-sample records")
        if payload.get("source_table_sha256") != sha256_file(paths["controls"]):
            raise ValueError(f"{name} source table SHA mismatch")
    if phenotypes.get("public_only") is not True or phenotypes.get("locked_accessed") is not False:
        raise ValueError("phenotype labels are not public-only")

    cell_rows = []
    verdict_cells = verdict.get("cells", {})
    for cell in ALL_CELLS:
        cell_verdict = verdict_cells.get(cell)
        if cell_verdict is None:
            cell_rows.append(
                {
                    "cell": cell,
                    "tested": False,
                    "selected_layer": None,
                    "patch_validation_hce": 0,
                    "patch_control_adjusted_effect": None,
                    "patch_control_adjusted_positive": False,
                    "erasure_validation_hce": 0,
                    "erasure_control_adjusted_drop": None,
                    "erasure_control_adjusted_positive": False,
                    "bidirectional_support": False,
                    "verdict": "not_run",
                }
            )
            continue
        patch_validation = cell_verdict["patch_validation"]
        erasure_validation = cell_verdict["erasure_validation"]
        bidirectional = bool(cell_verdict["bidirectional_control_adjusted_support"])
        cell_rows.append(
            {
                "cell": cell,
                "tested": True,
                "selected_layer": cell_verdict["selected_layer"],
                "patch_validation_hce": patch_validation["n_hce"],
                "patch_control_adjusted_effect": patch_validation["control_adjusted_effect_mean"],
                "patch_control_adjusted_positive": patch_validation["control_adjusted_mean_positive"],
                "erasure_validation_hce": erasure_validation["n_hce"],
                "erasure_control_adjusted_drop": erasure_validation["control_adjusted_margin_drop_mean"],
                "erasure_control_adjusted_positive": erasure_validation["control_adjusted_mean_positive"],
                "bidirectional_support": bidirectional,
                "verdict": "bidirectional_support" if bidirectional else "no_bidirectional_support",
            }
        )

    validation_rows = [
        row
        for row in phenotype_detection.get("rows", [])
        if row.get("aggregation") == "cell" and row.get("split") == "validation"
    ]
    phenotype_counts = Counter()
    phenotype_by_cell: dict[str, dict[str, int]] = {cell: {} for cell in ALL_CELLS}
    for row in validation_rows:
        phenotype = str(row["phenotype"])
        count = int(row["count"])
        phenotype_counts[phenotype] += count
        phenotype_by_cell[str(row["cell"])][phenotype] = count
    hce_count = sum(
        phenotype_counts[name]
        for name in ("behavioral_recoverable", "internal_competitive", "assessed_persistent", "unassessed_HCE")
    )
    expected_hce = 328
    if hce_count != expected_hce:
        raise ValueError(f"public validation HCE phenotype count mismatch: {hce_count} != {expected_hce}")

    macro_detection = {
        str(row["phenotype"]): {
            "eligible_cells": int(str(row["cell"]).split("_")[1]),
            "recall_or_fpr": row.get("recall", row.get("fpr")),
        }
        for row in phenotype_detection.get("rows", [])
        if row.get("aggregation") == "macro"
    }
    summary = verdict["summary"]
    if summary.get("bidirectional_control_adjusted_positive_cells"):
        raise ValueError("canonical verdict unexpectedly claims bidirectional support")
    return {
        "schema_version": "one-confidence-mosaic-v1-mechanism-evidence-audit-v1",
        "scope": "public development/validation mechanism analysis only",
        "public_only": True,
        "locked_accessed": False,
        "fresh_accessed": False,
        "controls": {
            "aggregate_rows": len(controls["rows"]),
            "required_controls_present": [
                "exact_noop",
                "random_layer_true_patch",
                "random_direction",
                "unrelated_fact",
                "shuffled_candidate",
                "reverse_patch",
                "HCC_DAMAGE",
            ],
        },
        "cell_verdicts": cell_rows,
        "mechanism_summary": {
            "tested_cells": sum(row["tested"] for row in cell_rows),
            "patch_control_adjusted_positive_cells": [
                row["cell"] for row in cell_rows if row["patch_control_adjusted_positive"]
            ],
            "erasure_control_adjusted_positive_cells": [
                row["cell"] for row in cell_rows if row["erasure_control_adjusted_positive"]
            ],
            "bidirectional_supported_cells": [
                row["cell"] for row in cell_rows if row["bidirectional_support"]
            ],
            "validation_hce_with_patch_and_erasure": sum(
                row["patch_validation_hce"] for row in cell_rows
            ),
            "shared_candidate_specific_mechanism_supported": False,
            "verdict": summary["verdict"],
        },
        "phenotype_coverage_public_validation": {
            "total_hce": hce_count,
            "counts": dict(phenotype_counts),
            "fractions_of_hce": {
                name: phenotype_counts[name] / hce_count
                for name in ("behavioral_recoverable", "internal_competitive", "assessed_persistent", "unassessed_HCE")
            },
            "by_cell": phenotype_by_cell,
            "overlap": phenotypes["overlap"],
            "detector_macro_descriptive": macro_detection,
            "interpretation": "Phenotypes are partial observational/causal-assessment strata. Unassessed HCE must not be relabeled persistent.",
        },
        "supported_claims": [
            "Some HCE show fixed-panel behavioral recovery or disagreement.",
            "Late-state interventions can perturb candidate competition in limited public subsets.",
            "Run6 internal trajectory features retain observational detection signal after confidence matching.",
            "The tested rank-one candidate direction fails bidirectional causal validation.",
        ],
        "unsupported_claims": [
            "A shared prompt-conditioned answer-selection circuit is validated across models.",
            "Internal/output mismatch is a causally established pathway rather than an observational detector signal.",
            "ACE/LEACE-style erasure validates the proposed direction.",
            "Current mechanism evidence supports full-cohort STEER mitigation.",
        ],
        "source_sha256": {str(path): sha256_file(path) for path in paths.values()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compile_audit(args.result_dir)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "mechanism_summary": payload["mechanism_summary"],
                "phenotype_counts": payload["phenotype_coverage_public_validation"]["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
