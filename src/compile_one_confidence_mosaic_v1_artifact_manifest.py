#!/usr/bin/env python3
"""Build the canonical MOSAIC-v1 closure artifact and isolation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_OUTPUTS = (
    "protocol.json",
    "feature_manifest.json",
    "phenotype_labels.json",
    "behavioral_features.json",
    "internal_features.json",
    "mechanism_patch_results.json",
    "mechanism_erasure_results.json",
    "validation_search.json",
    "final_thresholds.json",
    "table_main_detection.csv",
    "table_main_detection.json",
    "table_detection_by_phenotype.csv",
    "table_detection_by_phenotype.json",
    "table_mechanism_controls.csv",
    "table_mechanism_controls.json",
    "table_ablation.csv",
    "table_ablation.json",
    "table_mitigation.csv",
    "table_mitigation.json",
    "table_cost.csv",
    "table_cost.json",
    "figure_layer_patching.csv",
    "figure_layer_patching.json",
    "figure_risk_coverage.csv",
    "figure_risk_coverage.json",
    "case_studies.json",
    "model_checkpoints/index.json",
    "frozen_test_bootstrap.csv",
    "frozen_test_bootstrap.json",
    "frozen_results_compile_manifest.json",
)

FRESH_GENERATION_GLOB = "fresh_holdout_generations/*-generation.jsonl"
FRESH_ALLOWED_NAMES = {
    "fresh_method_holdout_protocol_preregistered.json",
    "fresh_method_holdout_source.json",
}
FRESH_FORBIDDEN_NAME_PARTS = (
    "screen",
    "review",
    "correctness",
    "matched",
    "detector",
    "hidden",
    "semantic_entropy",
    "trajectory_score",
    "probe_score",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def ensure_required(output_dir: Path, required: Iterable[str]) -> list[Path]:
    paths = [output_dir / relative for relative in required]
    missing = [path.as_posix() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required artifacts: " + ", ".join(missing))
    return paths


def fresh_isolation(output_dir: Path) -> dict[str, Any]:
    generation_paths = sorted(output_dir.glob(FRESH_GENERATION_GLOB))
    fresh_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and "invalidated" not in path.parts
        and ("fresh" in path.name.lower() or "fresh_holdout_generations" in path.parts)
    )
    forbidden = []
    for path in fresh_paths:
        relative = path.relative_to(output_dir)
        if path.name in FRESH_ALLOWED_NAMES:
            continue
        if relative.parts and relative.parts[0] == "fresh_holdout_generations":
            if path.suffix in {".jsonl", ".json", ".log"}:
                continue
        lowered = path.name.lower()
        if any(part in lowered for part in FRESH_FORBIDDEN_NAME_PARTS):
            forbidden.append(relative.as_posix())
    if forbidden:
        raise ValueError(f"fresh holdout has forbidden derived artifacts: {forbidden}")
    return {
        "generation_file_count": len(generation_paths),
        "generation_files": {
            path.relative_to(output_dir).as_posix(): {
                "rows": count_lines(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in generation_paths
        },
        "forbidden_derived_artifacts": forbidden,
        "correctness_screening_performed": False,
        "codex_review_performed": False,
        "matching_performed": False,
        "detector_scoring_performed": False,
        "note": (
            "Only source selection, BF16 generation, row counts, and byte hashes are "
            "represented. No model correctness labels were materialized or consumed."
        ),
    }


def build_manifest(
    project_root: Path,
    output_dir: Path,
    required: Iterable[str] = REQUIRED_OUTPUTS,
) -> dict[str, Any]:
    required_paths = ensure_required(output_dir, required)
    delivery_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path.name != "artifact_sha256.json"
        and "invalidated" not in path.parts
        and "__pycache__" not in path.parts
    )
    code_paths = sorted(
        path
        for path in (project_root / "experiments/hallucination").glob(
            "*one_confidence*mosaic_v1*.py"
        )
        if path.is_file()
    )
    all_paths = sorted(set(delivery_paths + code_paths))
    return {
        "schema_version": "one-confidence-mosaic-v1-closure-artifact-manifest-v1",
        "project_root": project_root.as_posix(),
        "output_dir": output_dir.relative_to(project_root).as_posix(),
        "positive_class": "HCE",
        "selected_run": "run6_16layer_strong_regularization",
        "selected_method": "learned_gate_no_phenotype",
        "result_scope": {
            "development_and_validation": "method selection and ablation",
            "retrospective_frozen_test": "baseline, Run6 detection, and frozen mitigation",
            "fresh_holdout": "generation-only; no correctness review or detector access",
        },
        "required_artifacts": {
            path.relative_to(output_dir).as_posix(): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in required_paths
        },
        "artifacts": {
            path.relative_to(project_root).as_posix(): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in all_paths
        },
        "fresh_holdout_isolation": fresh_isolation(output_dir),
        "invalidated_paths_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(args.project_root.resolve(), args.output_dir.resolve())
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "sha256": sha256(args.output),
                "artifact_count": len(payload["artifacts"]),
                "fresh_generation_files": payload["fresh_holdout_isolation"][
                    "generation_file_count"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
