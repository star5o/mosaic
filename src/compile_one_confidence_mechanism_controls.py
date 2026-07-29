#!/usr/bin/env python3
"""Compile public patching/erasure controls into paper-ready tidy artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


TABLE_FIELDS = [
    "cell",
    "model",
    "dataset",
    "experiment",
    "split",
    "population",
    "layer",
    "layer_role",
    "control",
    "n",
    "effect_metric",
    "effect_mean",
    "positive_fraction",
    "answer_start_metric",
    "answer_start_mean",
    "control_adjusted_metric",
    "control_adjusted_mean",
    "max_abs_noop_effect",
    "source_path",
    "source_sha256",
]


FIGURE_FIELDS = [
    "cell",
    "model",
    "dataset",
    "split",
    "layer",
    "layer_role",
    "n",
    "aligned_candidate_margin_effect_mean",
    "aligned_positive_fraction",
    "random_direction_effect_mean",
    "unrelated_fact_effect_mean",
    "shuffled_candidate_effect_mean",
    "control_adjusted_effect_mean",
    "answer_start_js_reduction_mean",
    "max_abs_noop_effect",
    "source_path",
    "source_sha256",
]


DEFAULT_STEMS = (
    "mistral7-triviaqa",
    "qwen8-triviaqa",
    "qwen14-popqa",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_cell(cell: str) -> tuple[str, str]:
    if "|" not in cell:
        raise ValueError(f"invalid cell name: {cell}")
    return tuple(cell.split("|", 1))  # type: ignore[return-value]


def load_public(path: Path) -> dict[str, Any]:
    if "invalidated" in str(path).casefold():
        raise ValueError("invalidated artifacts are forbidden")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol", {}).get("locked_accessed") is not False:
        raise ValueError(f"{path}: expected an explicitly public result")
    return payload


def row_base(
    cell: str,
    experiment: str,
    split: str,
    population: str,
    layer: int,
    layer_role: str,
    control: str,
    n: int,
    source_path: Path,
) -> dict[str, Any]:
    model, dataset = split_cell(cell)
    return {
        "cell": cell,
        "model": model,
        "dataset": dataset,
        "experiment": experiment,
        "split": split,
        "population": population,
        "layer": layer,
        "layer_role": layer_role,
        "control": control,
        "n": n,
        "effect_metric": None,
        "effect_mean": None,
        "positive_fraction": None,
        "answer_start_metric": None,
        "answer_start_mean": None,
        "control_adjusted_metric": None,
        "control_adjusted_mean": None,
        "max_abs_noop_effect": None,
        "source_path": str(source_path),
        "source_sha256": sha256(source_path),
    }


def patch_hce_rows(
    payload: dict[str, Any], source_path: Path, split: str
) -> list[dict[str, Any]]:
    cell = payload["protocol"]["cell"]
    selected_layer = int(payload["layer_selection"]["selected_layer"])
    random_layer = int(payload["layer_selection"]["random_control_layer"])
    summary = (
        payload["development_scan_summary"]
        if split == "development"
        else payload["validation_hce_summary"]
    )
    by_layer = {int(row["layer"]): row for row in summary}
    selected = by_layer[selected_layer]
    n = int(selected["n"])
    controls = [
        (
            "true_patch",
            selected["aligned_effect_mean"],
            selected["aligned_positive_fraction"],
        ),
        ("random_direction", selected["random_direction_effect_mean"], None),
        ("unrelated_fact", selected["unrelated_effect_mean"], None),
        ("shuffled_candidate", selected["shuffled_candidate_effect_mean"], None),
    ]
    output = []
    for control, value, positive_fraction in controls:
        row = row_base(
            cell,
            "late_mlp_patch",
            split,
            "HCE",
            selected_layer,
            "selected_layer",
            control,
            n,
            source_path,
        )
        row.update(
            {
                "effect_metric": "candidate_minus_canonical_margin_change",
                "effect_mean": value,
                "positive_fraction": positive_fraction,
            }
        )
        if control == "true_patch":
            row.update(
                {
                    "answer_start_metric": "canonical_to_donor_js_reduction",
                    "answer_start_mean": selected[
                        "answer_start_js_reduction_mean"
                    ],
                    "control_adjusted_metric": "true_minus_strongest_per_row_control",
                    "control_adjusted_mean": selected[
                        "primary_control_adjusted_effect_mean"
                    ],
                }
            )
        output.append(row)
    noop = row_base(
        cell,
        "late_mlp_patch",
        split,
        "HCE",
        selected_layer,
        "selected_layer",
        "exact_noop",
        n,
        source_path,
    )
    noop.update(
        {
            "effect_metric": "candidate_minus_canonical_margin_change",
            "max_abs_noop_effect": selected["max_abs_noop_effect"],
        }
    )
    output.append(noop)
    if random_layer in by_layer:
        random_layer_row = by_layer[random_layer]
        control = row_base(
            cell,
            "late_mlp_patch",
            split,
            "HCE",
            random_layer,
            "random_control_layer",
            "random_layer_true_patch",
            int(random_layer_row["n"]),
            source_path,
        )
        control.update(
            {
                "effect_metric": "candidate_minus_canonical_margin_change",
                "effect_mean": random_layer_row["aligned_effect_mean"],
                "positive_fraction": random_layer_row[
                    "aligned_positive_fraction"
                ],
                "answer_start_metric": "canonical_to_donor_js_reduction",
                "answer_start_mean": random_layer_row[
                    "answer_start_js_reduction_mean"
                ],
            }
        )
        output.append(control)
    return output


def patch_hcc_rows(
    payload: dict[str, Any], source_path: Path, split: str
) -> list[dict[str, Any]]:
    cell = payload["protocol"]["cell"]
    selected_layer = int(payload["layer_selection"]["selected_layer"])
    random_layer = int(payload["layer_selection"]["random_control_layer"])
    by_layer = {
        int(row["layer"]): row
        for row in payload["hcc_control_summary"][split]
    }
    selected = by_layer[selected_layer]
    n = int(selected["n"])
    controls = [
        ("true_patch", selected["aligned_canonical_score_change_mean"]),
        ("random_direction", selected["random_direction_score_change_mean"]),
        ("unrelated_fact", selected["unrelated_score_change_mean"]),
    ]
    output = []
    for control, value in controls:
        row = row_base(
            cell,
            "late_mlp_patch",
            split,
            "HCC_DAMAGE",
            selected_layer,
            "selected_layer",
            control,
            n,
            source_path,
        )
        row.update(
            {
                "effect_metric": "canonical_sequence_score_change",
                "effect_mean": value,
            }
        )
        if control == "true_patch":
            row.update(
                {
                    "answer_start_metric": "answer_start_kl_from_baseline",
                    "answer_start_mean": selected[
                        "aligned_answer_start_kl_mean"
                    ],
                }
            )
        output.append(row)
    noop = row_base(
        cell,
        "late_mlp_patch",
        split,
        "HCC_DAMAGE",
        selected_layer,
        "selected_layer",
        "exact_noop",
        n,
        source_path,
    )
    noop.update(
        {
            "effect_metric": "canonical_sequence_score_change",
            "max_abs_noop_effect": selected["max_abs_noop_score_change"],
        }
    )
    output.append(noop)
    if random_layer in by_layer:
        random_selected = by_layer[random_layer]
        control = row_base(
            cell,
            "late_mlp_patch",
            split,
            "HCC_DAMAGE",
            random_layer,
            "random_control_layer",
            "random_layer_true_patch",
            int(random_selected["n"]),
            source_path,
        )
        control.update(
            {
                "effect_metric": "canonical_sequence_score_change",
                "effect_mean": random_selected[
                    "aligned_canonical_score_change_mean"
                ],
                "answer_start_metric": "answer_start_kl_from_baseline",
                "answer_start_mean": random_selected[
                    "aligned_answer_start_kl_mean"
                ],
            }
        )
        output.append(control)
    return output


def erasure_rows(
    payload: dict[str, Any], source_path: Path
) -> list[dict[str, Any]]:
    cell = payload["protocol"]["cell"]
    layer = int(payload["protocol"]["selected_layer"])
    hce = payload["validation_hce_summary"]
    hce_controls = [
        (
            "true_erasure",
            hce["true_donor_candidate_margin_drop_mean"],
            hce["true_donor_candidate_margin_drop_positive_fraction"],
        ),
        ("random_direction", hce["random_direction_margin_drop_mean"], None),
        ("unrelated_fact", hce["unrelated_direction_margin_drop_mean"], None),
        ("shuffled_candidate", hce["shuffled_candidate_margin_drop_mean"], None),
    ]
    output = []
    for control, value, positive_fraction in hce_controls:
        row = row_base(
            cell,
            "rank_one_erasure",
            "validation",
            "HCE",
            layer,
            "frozen_selected_layer",
            control,
            int(hce["n"]),
            source_path,
        )
        row.update(
            {
                "effect_metric": "donor_candidate_minus_canonical_margin_drop",
                "effect_mean": value,
                "positive_fraction": positive_fraction,
            }
        )
        if control == "true_erasure":
            row.update(
                {
                    "answer_start_metric": "canonical_to_donor_js_reduction_donor_only",
                    "answer_start_mean": hce[
                        "true_answer_start_js_reduction_donor_only_mean"
                    ],
                    "control_adjusted_metric": "true_minus_strongest_per_row_control",
                    "control_adjusted_mean": hce[
                        "control_adjusted_margin_drop_mean"
                    ],
                }
            )
        output.append(row)
    noop = row_base(
        cell,
        "rank_one_erasure",
        "validation",
        "HCE",
        layer,
        "frozen_selected_layer",
        "exact_noop",
        int(hce["n"]),
        source_path,
    )
    noop.update(
        {
            "effect_metric": "donor_candidate_minus_canonical_margin_drop",
            "max_abs_noop_effect": hce["max_abs_noop_margin_change"],
        }
    )
    output.append(noop)

    hcc = payload["validation_hcc_summary"]
    hcc_controls = [
        ("true_erasure", hcc["true_canonical_sequence_score_change_mean"]),
        (
            "random_direction",
            hcc["random_direction_sequence_score_change_mean"],
        ),
        (
            "unrelated_fact",
            hcc["unrelated_direction_sequence_score_change_mean"],
        ),
    ]
    for control, value in hcc_controls:
        row = row_base(
            cell,
            "rank_one_erasure",
            "validation",
            "HCC_DAMAGE",
            layer,
            "frozen_selected_layer",
            control,
            int(hcc["n"]),
            source_path,
        )
        row.update(
            {
                "effect_metric": "canonical_sequence_score_change",
                "effect_mean": value,
            }
        )
        if control == "true_erasure":
            row.update(
                {
                    "answer_start_metric": "answer_start_js_from_baseline",
                    "answer_start_mean": hcc["true_answer_start_js_mean"],
                }
            )
        output.append(row)
    noop = row_base(
        cell,
        "rank_one_erasure",
        "validation",
        "HCC_DAMAGE",
        layer,
        "frozen_selected_layer",
        "exact_noop",
        int(hcc["n"]),
        source_path,
    )
    noop.update(
        {
            "effect_metric": "canonical_sequence_score_change",
            "max_abs_noop_effect": hcc[
                "max_abs_noop_sequence_score_change"
            ],
        }
    )
    output.append(noop)
    return output


def compile_table(
    patch_inputs: list[tuple[dict[str, Any], Path]],
    erasure_inputs: list[tuple[dict[str, Any], Path]],
) -> list[dict[str, Any]]:
    rows = []
    for payload, path in patch_inputs:
        for split in ("development", "validation"):
            rows.extend(patch_hce_rows(payload, path, split))
            rows.extend(patch_hcc_rows(payload, path, split))
    for payload, path in erasure_inputs:
        rows.extend(erasure_rows(payload, path))
    return sorted(
        rows,
        key=lambda row: (
            row["cell"],
            row["experiment"],
            row["split"],
            row["population"],
            row["layer"],
            row["control"],
        ),
    )


def compile_layer_figure(
    patch_inputs: list[tuple[dict[str, Any], Path]]
) -> list[dict[str, Any]]:
    output = []
    for payload, path in patch_inputs:
        cell = payload["protocol"]["cell"]
        model, dataset = split_cell(cell)
        selected_layer = int(payload["layer_selection"]["selected_layer"])
        random_layer = int(payload["layer_selection"]["random_control_layer"])
        for split, summary in (
            ("development", payload["development_scan_summary"]),
            ("validation", payload["validation_hce_summary"]),
        ):
            for row in summary:
                layer = int(row["layer"])
                if layer == selected_layer:
                    role = "selected_layer"
                elif layer == random_layer:
                    role = "random_control_layer"
                else:
                    role = "scanned_layer"
                output.append(
                    {
                        "cell": cell,
                        "model": model,
                        "dataset": dataset,
                        "split": split,
                        "layer": layer,
                        "layer_role": role,
                        "n": int(row["n"]),
                        "aligned_candidate_margin_effect_mean": row[
                            "aligned_effect_mean"
                        ],
                        "aligned_positive_fraction": row[
                            "aligned_positive_fraction"
                        ],
                        "random_direction_effect_mean": row[
                            "random_direction_effect_mean"
                        ],
                        "unrelated_fact_effect_mean": row[
                            "unrelated_effect_mean"
                        ],
                        "shuffled_candidate_effect_mean": row[
                            "shuffled_candidate_effect_mean"
                        ],
                        "control_adjusted_effect_mean": row[
                            "primary_control_adjusted_effect_mean"
                        ],
                        "answer_start_js_reduction_mean": row[
                            "answer_start_js_reduction_mean"
                        ],
                        "max_abs_noop_effect": row["max_abs_noop_effect"],
                        "source_path": str(path),
                        "source_sha256": sha256(path),
                    }
                )
    return sorted(
        output, key=lambda row: (row["cell"], row["split"], row["layer"])
    )


def compile_verdict(
    patch_inputs: list[tuple[dict[str, Any], Path]],
    erasure_inputs: list[tuple[dict[str, Any], Path]],
) -> dict[str, Any]:
    patches = {payload["protocol"]["cell"]: payload for payload, _ in patch_inputs}
    erasures = {
        payload["protocol"]["cell"]: payload for payload, _ in erasure_inputs
    }
    if set(patches) != set(erasures):
        raise ValueError("patch and erasure cell sets differ")
    cells = {}
    patch_positive_cells = []
    erasure_positive_cells = []
    bidirectional_cells = []
    for cell in sorted(patches):
        patch = patches[cell]
        erasure = erasures[cell]
        selected_layer = int(patch["layer_selection"]["selected_layer"])
        patch_row = next(
            row
            for row in patch["validation_hce_summary"]
            if int(row["layer"]) == selected_layer
        )
        erase_row = erasure["validation_hce_summary"]
        patch_adjusted = float(patch_row["primary_control_adjusted_effect_mean"])
        erasure_adjusted = float(erase_row["control_adjusted_margin_drop_mean"])
        patch_positive = patch_adjusted > 0
        erasure_positive = erasure_adjusted > 0
        if patch_positive:
            patch_positive_cells.append(cell)
        if erasure_positive:
            erasure_positive_cells.append(cell)
        if patch_positive and erasure_positive:
            bidirectional_cells.append(cell)
        cells[cell] = {
            "selected_layer": selected_layer,
            "patch_validation": {
                "n_hce": int(patch_row["n"]),
                "true_candidate_margin_effect_mean": patch_row[
                    "aligned_effect_mean"
                ],
                "random_direction_effect_mean": patch_row[
                    "random_direction_effect_mean"
                ],
                "unrelated_fact_effect_mean": patch_row[
                    "unrelated_effect_mean"
                ],
                "shuffled_candidate_effect_mean": patch_row[
                    "shuffled_candidate_effect_mean"
                ],
                "control_adjusted_effect_mean": patch_adjusted,
                "control_adjusted_mean_positive": patch_positive,
                "hcc_n": int(
                    next(
                        row
                        for row in patch["hcc_control_summary"]["validation"]
                        if int(row["layer"]) == selected_layer
                    )["n"]
                ),
            },
            "erasure_validation": {
                "n_hce": int(erase_row["n"]),
                "true_candidate_margin_drop_mean": erase_row[
                    "true_donor_candidate_margin_drop_mean"
                ],
                "random_direction_margin_drop_mean": erase_row[
                    "random_direction_margin_drop_mean"
                ],
                "unrelated_fact_margin_drop_mean": erase_row[
                    "unrelated_direction_margin_drop_mean"
                ],
                "shuffled_candidate_margin_drop_mean": erase_row[
                    "shuffled_candidate_margin_drop_mean"
                ],
                "control_adjusted_margin_drop_mean": erasure_adjusted,
                "control_adjusted_mean_positive": erasure_positive,
                "control_adjusted_positive_fraction": erase_row[
                    "control_adjusted_positive_fraction"
                ],
                "hcc_n": int(erasure["validation_hcc_summary"]["n"]),
                "hcc_canonical_sequence_score_change_mean": erasure[
                    "validation_hcc_summary"
                ]["true_canonical_sequence_score_change_mean"],
            },
            "bidirectional_control_adjusted_support": patch_positive
            and erasure_positive,
            "small_validation_mechanism_subset": int(patch_row["n"]) < 10,
        }
    if bidirectional_cells:
        bidirectional_text = (
            "Control-adjusted patch and erasure means are both positive for "
            f"{', '.join(bidirectional_cells)}."
        )
    else:
        bidirectional_text = (
            "No cell has both a positive control-adjusted patch mean and a "
            "positive control-adjusted erasure mean."
        )
    small_cells = [
        cell
        for cell, payload in cells.items()
        if payload["small_validation_mechanism_subset"]
    ]
    small_subset_text = (
        " All evaluated mechanism cohorts are smaller than 10 HCE rows."
        if len(small_cells) == len(cells)
        else (
            f" Small validation mechanism subsets: {', '.join(small_cells)}."
            if small_cells
            else ""
        )
    )
    verdict_text = (
        f"Patch control-adjusted-positive cells: "
        f"{', '.join(patch_positive_cells) or 'none'}. "
        f"Erasure control-adjusted-positive cells: "
        f"{', '.join(erasure_positive_cells) or 'none'}. "
        f"{bidirectional_text}{small_subset_text} These public subset results "
        "do not establish a shared or universal hallucination circuit."
    )
    return {
        "schema_version": "one-confidence-public-mechanism-control-verdict-v1",
        "criterion": "A result is control-adjusted-positive only when its frozen validation mean true effect exceeds the strongest per-row random, unrelated, or shuffled control. This is descriptive, not a significance test.",
        "cells": cells,
        "summary": {
            "patch_control_adjusted_positive_cells": patch_positive_cells,
            "erasure_control_adjusted_positive_cells": erasure_positive_cells,
            "bidirectional_control_adjusted_positive_cells": bidirectional_cells,
            "shared_rank_one_bidirectional_mechanism_supported": len(
                bidirectional_cells
            )
            == len(cells),
            "verdict": verdict_text,
            "locked_accessed": False,
        },
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--stems",
        nargs="+",
        default=list(DEFAULT_STEMS),
        help="Public patch/erasure filename stems to compile.",
    )
    args = parser.parse_args()
    result_dir = args.result_dir
    patch_inputs = []
    erasure_inputs = []
    for stem in args.stems:
        patch_path = result_dir / f"{stem}-public-mechanism-patching.json"
        erasure_path = result_dir / f"{stem}-public-mechanism-erasure.json"
        patch_inputs.append((load_public(patch_path), patch_path))
        erasure_inputs.append((load_public(erasure_path), erasure_path))
    table = compile_table(patch_inputs, erasure_inputs)
    figure = compile_layer_figure(patch_inputs)
    verdict = compile_verdict(patch_inputs, erasure_inputs)
    write_csv(result_dir / "table_mechanism_controls.csv", TABLE_FIELDS, table)
    write_json(
        result_dir / "table_mechanism_controls.json",
        {
            "schema_version": "one-confidence-public-mechanism-controls-v1",
            "metric_direction_note": "Positive patch effect means candidate support increased. Positive erasure drop means donor candidate support decreased. HCC score changes near zero indicate lower damage.",
            "rows": table,
        },
    )
    write_csv(result_dir / "figure_layer_patching.csv", FIGURE_FIELDS, figure)
    write_json(
        result_dir / "figure_layer_patching.json",
        {
            "schema_version": "one-confidence-public-layer-patching-figure-v1",
            "note": "Only actually run patch layers are present. Erasure used one frozen selected layer and is reported in table_mechanism_controls instead of a fabricated layer curve.",
            "rows": figure,
        },
    )
    write_json(result_dir / "mechanism_control_verdict.json", verdict)
    print(
        json.dumps(
            {
                "table_rows": len(table),
                "figure_rows": len(figure),
                "cells": sorted(verdict["cells"]),
                "verdict": verdict["summary"]["verdict"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
