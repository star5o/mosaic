#!/usr/bin/env python3
"""Compile measured and protocol-level MOSAIC-v1 frozen-test costs.

The frozen source pipeline did not record peak GPU memory for every stage. This
compiler therefore reports measured values only where an artifact or log
contains them and leaves unavailable quantities null rather than estimating
them from model size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


CELLS = (
    "mistral7-triviaqa",
    "qwen8-triviaqa",
    "llama31-triviaqa",
    "qwen14-popqa",
    "nemo12-popqa",
)

CELL_NAMES = {
    "mistral7-triviaqa": "mistral7|TriviaQA",
    "qwen8-triviaqa": "qwen8|TriviaQA",
    "llama31-triviaqa": "llama31|TriviaQA",
    "qwen14-popqa": "qwen14|PopQA",
    "nemo12-popqa": "nemo12|PopQA",
}

PIPELINE_FIELDS = (
    "cell",
    "n_rows",
    "panel_size",
    "rewrite_generation_queries",
    "initial_panel_answer_generations",
    "corrected_panel_answer_generations",
    "self_consistency_sampling_generations",
    "canonical_hidden_forward_passes",
    "canonical_teacher_forced_sequence_scores",
    "candidate_teacher_forced_sequence_scores",
    "risk_classifier_evaluations",
    "corrected_panel_generated_answer_tokens",
    "mean_corrected_panel_answer_tokens",
    "natural_rewrite_stage_wall_time_seconds",
    "corrected_panel_stage_wall_time_seconds",
    "measured_gpu_stage_wall_time_seconds",
    "peak_gpu_memory_mb",
    "peak_gpu_memory_status",
)

POLICY_FIELDS = (
    "model",
    "dataset",
    "split",
    "method",
    "available",
    "n",
    "keep_count",
    "recover_count",
    "abstain_count",
    "high_risk_count",
    "paraphrase_generation_queries",
    "self_consistency_sampling_generations",
    "canonical_hidden_forward_passes",
    "teacher_forced_sequence_scores",
    "semantic_cluster_rows",
    "risk_classifier_evaluations",
    "additional_answer_generation_queries",
    "actual_steering_generation_rows",
    "candidate_adoption_count",
    "wall_time_seconds",
    "peak_gpu_memory_mb",
    "cost_status",
)

CSV_FIELDS = ("row_type",) + tuple(dict.fromkeys(PIPELINE_FIELDS + POLICY_FIELDS))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def last_elapsed_seconds(path: Path) -> float | None:
    elapsed: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.search(
            r"elapsed(?:_seconds)?[\"']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            line,
        )
        if match:
            elapsed = float(match.group(1))
    return elapsed


def non_null_count(values: list[Any]) -> int:
    return sum(value is not None for value in values)


def compile_cell(source_dir: Path, cell_slug: str) -> tuple[dict[str, Any], dict[str, str]]:
    natural_path = source_dir / f"{cell_slug}-natural-paraphrase.json"
    natural_log = source_dir / f"{cell_slug}-natural-paraphrase.log"
    panel_path = source_dir / f"{cell_slug}-corrected-panel-features.json"
    candidate_path = source_dir / f"{cell_slug}-candidate-margins.json"
    persistence_path = source_dir / f"{cell_slug}-persistence.json"
    internal_path = source_dir / f"{cell_slug}-internal-states.pt"

    natural = load_json(natural_path)
    panel = load_json(panel_path)
    candidate = load_json(candidate_path)
    persistence = load_json(persistence_path)
    n_rows = len(panel["records"])
    expected_ids = {row["sample_id"] for row in panel["records"]}
    for name, payload in (
        ("natural", natural),
        ("candidate", candidate),
        ("persistence", persistence),
    ):
        ids = {row["sample_id"] for row in payload["records"]}
        if ids != expected_ids:
            raise ValueError(f"{cell_slug}: {name} sample IDs do not align")
    if not internal_path.exists():
        raise FileNotFoundError(internal_path)

    panel_sizes = {len(row["panel_answers"]) for row in panel["records"]}
    if len(panel_sizes) != 1:
        raise ValueError(f"{cell_slug}: inconsistent corrected panel sizes {panel_sizes}")
    panel_size = panel_sizes.pop()
    initial_panel_sizes = {len(row["panel_answers"]) for row in natural["records"]}
    if initial_panel_sizes != {panel_size}:
        raise ValueError(
            f"{cell_slug}: initial panel sizes {initial_panel_sizes} differ from {panel_size}"
        )

    answer_token_count = sum(
        len(token_ids)
        for row in panel["records"]
        for token_ids in row["panel_generated_token_ids"]
    )
    corrected_generations = n_rows * panel_size
    canonical_scores = sum(
        non_null_count(row["panel_canonical_answer_nlls"]) for row in candidate["records"]
    )
    candidate_scores = sum(
        non_null_count(row["panel_candidate_answer_nlls"]) for row in candidate["records"]
    )
    persistence_k = int(persistence["K"])
    natural_elapsed = last_elapsed_seconds(natural_log)
    panel_elapsed = panel.get("elapsed_seconds")
    measured_wall = None
    if natural_elapsed is not None and panel_elapsed is not None:
        measured_wall = float(natural_elapsed) + float(panel_elapsed)

    row = {
        "cell": CELL_NAMES[cell_slug],
        "n_rows": n_rows,
        "panel_size": panel_size,
        "rewrite_generation_queries": n_rows,
        "initial_panel_answer_generations": n_rows * panel_size,
        "corrected_panel_answer_generations": corrected_generations,
        "self_consistency_sampling_generations": n_rows * persistence_k,
        "canonical_hidden_forward_passes": n_rows,
        "canonical_teacher_forced_sequence_scores": canonical_scores,
        "candidate_teacher_forced_sequence_scores": candidate_scores,
        "risk_classifier_evaluations": n_rows,
        "corrected_panel_generated_answer_tokens": answer_token_count,
        "mean_corrected_panel_answer_tokens": (
            answer_token_count / corrected_generations if corrected_generations else None
        ),
        "natural_rewrite_stage_wall_time_seconds": natural_elapsed,
        "corrected_panel_stage_wall_time_seconds": panel_elapsed,
        "measured_gpu_stage_wall_time_seconds": measured_wall,
        "peak_gpu_memory_mb": None,
        "peak_gpu_memory_status": "unmeasured; no trustworthy per-stage record",
    }
    sources = {
        path.as_posix(): sha256(path)
        for path in (
            natural_path,
            natural_log,
            panel_path,
            candidate_path,
            persistence_path,
            internal_path,
        )
    }
    return row, sources


def compile_cost(source_dir: Path, mitigation_cost_path: Path | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for cell_slug in CELLS:
        row, cell_sources = compile_cell(source_dir, cell_slug)
        rows.append(row)
        sources.update(cell_sources)
    mitigation_rows: list[dict[str, Any]] = []
    if mitigation_cost_path is not None:
        mitigation = load_json(mitigation_cost_path)
        mitigation_rows = mitigation["rows"]
        if any(row.get("split") != "locked_test" for row in mitigation_rows):
            raise ValueError("mitigation cost contains a non-locked-test row")
        sources[mitigation_cost_path.as_posix()] = sha256(mitigation_cost_path)
    combined_rows = [dict(row_type="pipeline", **row) for row in rows]
    combined_rows.extend(dict(row_type="mitigation_policy", **row) for row in mitigation_rows)
    return {
        "schema_version": "one-confidence-mosaic-v1-frozen-cost-v1",
        "result_scope": "retrospective_frozen_test",
        "positive_class": "HCE",
        "locked_accessed": True,
        "fresh_accessed": False,
        "method": "learned_gate_no_phenotype",
        "selected_run": "run6_16layer_strong_regularization",
        "cost_definition": {
            "counts": "realized source-pipeline operations on the 654 frozen rows",
            "wall_time": (
                "sum of recorded natural rewrite/initial-panel time and corrected-panel "
                "generation time; candidate scoring, hidden extraction, K=4 sampling, NLI, "
                "and CPU classifier time were not consistently timed and are excluded"
            ),
            "mean_tokens": "answer-content tokens in the final corrected five-prompt panel",
            "peak_memory": "not recorded; intentionally not estimated",
        },
        "pipeline_rows": rows,
        "mitigation_policy_rows": mitigation_rows,
        "rows": combined_rows,
        "source_sha256": sources,
    }


def write_outputs(payload: dict[str, Any], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(payload["rows"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--mitigation-cost", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    payload = compile_cost(args.source_dir, args.mitigation_cost)
    write_outputs(payload, args.output_json, args.output_csv)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "output_json_sha256": sha256(args.output_json),
                "output_csv": args.output_csv.as_posix(),
                "output_csv_sha256": sha256(args.output_csv),
                "pipeline_rows": len(payload["pipeline_rows"]),
                "mitigation_policy_rows": len(payload["mitigation_policy_rows"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
