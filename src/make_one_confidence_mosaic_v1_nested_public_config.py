#!/usr/bin/env python3
"""Write the unified five-cell public MOSAIC-v1 nested-evaluation config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CELLS = (
    ("mistral7|TriviaQA", "mistral7-triviaqa"),
    ("qwen8|TriviaQA", "qwen8-triviaqa"),
    ("llama31|TriviaQA", "llama31-triviaqa"),
    ("qwen14|PopQA", "qwen14-popqa"),
    ("nemo12|PopQA", "nemo12-popqa"),
)
BEHAVIORAL_FIELDS = (
    "answer_start_mi_all_safe",
    "canonical_panel_js_mean_all_safe",
    "canonical_panel_js_max_all_safe",
    "canonical_panel_js_variance_all_safe",
    "all_pairwise_js_mean_all_safe",
    "canonical_panel_js_mean_natural",
    "canonical_panel_js_max_natural",
    "all_pairwise_js_mean_natural",
    "natural_paraphrase_count",
    "answer_start_top1_disagreement_all_safe",
    "answer_start_unique_top1_all_safe",
    "answer_start_top1_probability_range_all_safe",
    "canonical_answer_nll",
    "cross_prompt_nll_signed_mean_all_safe",
    "cross_prompt_nll_absolute_mean_all_safe",
    "cross_prompt_nll_max_all_safe",
    "cross_prompt_nll_signed_mean_natural",
    "cross_prompt_nll_absolute_mean_natural",
    "canonical_cluster_retention_all_safe",
    "string_disagreement_all_safe",
    "string_disagreement_natural",
    "largest_exact_alternative_cluster_all_safe",
    "paraphrase_confidence_gain_all_safe",
    "canonical_greedy_repeat_exact",
    "canonical_greedy_repeat_valid",
    "canonical_greedy_repeat_truncated",
    "canonical_greedy_repeat_confidence_delta",
)
NLI_FIELDS = (
    "valid_safe_panel_answer_count",
    "original_support_all_safe",
    "semantic_disagreement_all_safe",
    "semantic_entropy_all_safe",
    "original_support_natural",
    "semantic_disagreement_natural",
    "semantic_entropy_natural",
    "natural_paraphrase_count",
    "alternative_support_all_safe",
    "alternative_natural_support_for_all_safe_candidate",
    "alternative_support_count_natural",
    "alternative_support_fraction_natural",
    "alternative_total_support_for_natural_candidate",
    "safe_panel_alternative_supported",
    "behavioral_recoverable",
)
CANDIDATE_FIELDS = (
    "candidate_available",
    "candidate_logscore_margin",
    "candidate_minus_canonical_nll",
    "support_prompt_candidate_logscore_margin_mean",
    "nli_behavioral_recoverable",
    "behavioral_recoverable_margin_pass",
)
SAMPLING_FIELDS = (
    "self_consistency_risk",
    "semantic_entropy_risk",
    "original_support",
)
PHENOTYPE_EVIDENCE_FIELDS = (
    "natural_support_count",
    "support_margin",
    "reliable_natural_candidate",
    "causal_assessed",
    "causal_internal_support",
    "true_effect",
    "random_control_effect",
    "unrelated_control_effect",
    "shuffled_control_effect",
    "reverse_effect",
    "validation_confirmed",
)
CANDIDATE_TRAJECTORY_FIELDS = (
    "early_mean_margin",
    "middle_mean_margin",
    "late_mean_margin",
    "late_minus_middle_margin",
    "trajectory_auc",
    "minimum_margin",
    "minimum_margin_layer",
    "maximum_margin",
    "maximum_margin_layer",
    "maximum_positive_slope",
    "maximum_positive_slope_to_layer",
    "maximum_negative_slope",
    "maximum_negative_slope_to_layer",
    "maximum_absolute_slope",
    "sign_flip_count",
    "candidate_preferred_layer_fraction",
    "first_candidate_preferred_layer",
    "last_candidate_preferred_layer",
    "final_margin",
    "final_candidate_preferred",
)
TRAJECTORY_GATE_EVIDENCE = (
    "maximum_margin",
    "late_minus_middle_margin",
    "final_margin",
    "sign_flip_count",
    "candidate_preferred_layer_fraction",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-candidate-trajectory",
        action="store_true",
        help="write the run2 structure with frozen candidate sequence-trajectory inputs",
    )
    parser.add_argument(
        "--causal-negative-phenotypes",
        action="store_true",
        help=(
            "replace only the Mistral7/Qwen8 phenotype sources with the frozen "
            "v2 causal-negative labels"
        ),
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    mosaic = repo / "experiments/hallucination/results/one_confidence_mosaic_v1"
    five_model = repo / "experiments/hallucination/results/one_confidence_five_model_eval"
    cells = []
    portable = lambda path: str(path.relative_to(repo))
    for cell, stem in CELLS:
        phenotype_dir = "strict_partial_phenotypes"
        if args.causal_negative_phenotypes and cell in {
            "mistral7|TriviaQA",
            "qwen8|TriviaQA",
        }:
            phenotype_dir = "strict_partial_phenotypes_v2_causal_negative"
        cell_config = {
            "cell": cell,
            "internal_states": portable(five_model / f"{stem}-public-features.pt"),
            "feature_sources": {
                "behavioral": {
                    "path": portable(mosaic / f"{stem}-public-corrected-panel-features.json"),
                    "fields": {name: f"features.{name}" for name in BEHAVIORAL_FIELDS},
                },
                "nli": {
                    "path": portable(mosaic / f"{stem}-public-corrected-panel-nli.json"),
                    "fields": {name: name for name in NLI_FIELDS},
                },
                "candidate": {
                    "path": portable(mosaic / f"{stem}-public-nli-candidate-margins.json"),
                    "fields": {name: name for name in CANDIDATE_FIELDS},
                },
                "persistence": {
                    "path": portable(five_model / f"{stem}-sampling-scores-k4.json"),
                    "fields": {name: name for name in SAMPLING_FIELDS},
                },
            },
            "feature_groups": {
                "behavioral": ["behavioral.*", "nli.*"],
                "candidate": ["candidate.*"],
                "persistence": ["persistence.*"],
                "gate_evidence": [
                    "nli.behavioral_recoverable",
                    "nli.alternative_support_count_natural",
                    "nli.alternative_support_fraction_natural",
                    "nli.semantic_disagreement_natural",
                    "candidate.*",
                ],
            },
            "routing": {
                "behavioral_indicator": "candidate.behavioral_recoverable_margin_pass",
                "behavioral_threshold": 0.5,
                "candidate_available": "candidate.candidate_available",
                "candidate_available_threshold": 0.5,
                "candidate_margin": "candidate.candidate_logscore_margin",
                "candidate_margin_threshold": 0.0,
            },
            "phenotype_auxiliary": {
                "path": portable(
                    mosaic
                    / phenotype_dir
                    / f"{stem}-public-strict-partial-phenotypes.json"
                ),
                "field": "phenotype",
                "mapping": {"behavioral": 0, "internal": 1, "persistent": 2},
                "ambiguous_values": ["ambiguous", "unlabeled"],
                "apply_to_classes": ["HCE"],
                "strict_rules": True,
                "evidence_fields": {
                    name: f"evidence.{name}" for name in PHENOTYPE_EVIDENCE_FIELDS
                },
            },
        }
        if args.include_candidate_trajectory:
            cell_config["feature_sources"]["candidate_trajectory"] = {
                "path": portable(
                    mosaic / f"{stem}-public-candidate-sequence-trajectory.json"
                ),
                "fields": {
                    name: f"features.{name}" for name in CANDIDATE_TRAJECTORY_FIELDS
                },
            }
            cell_config["feature_groups"]["candidate"].append("candidate_trajectory.*")
            cell_config["feature_groups"]["gate_evidence"].extend(
                f"candidate_trajectory.{name}" for name in TRAJECTORY_GATE_EVIDENCE
            )
        cells.append(cell_config)
    if args.include_candidate_trajectory and args.causal_negative_phenotypes:
        raise ValueError(
            "candidate trajectory and causal-negative phenotype variants must be "
            "generated as separate single-change runs"
        )
    payload = {
        "schema_version": "one-confidence-mosaic-v1-group-nested-config-v1",
        "seed": 2026072913,
        "outer_folds": 5,
        "base_inner_folds": 4,
        "gate_inner_folds": 4,
        "gate_base_inner_folds": 3,
        "fixed_layer_count": 8,
        "base_probe_c": 0.01,
        "expert_c": 0.1,
        "gate_c": 0.1,
        "phenotype_gate_hidden": 8,
        "phenotype_gate_epochs": 300,
        "phenotype_gate_lr": 0.02,
        "phenotype_gate_l2": 0.001,
        "phenotype_aux_weight": 0.3,
        "max_fpr": 0.1,
        "serialize_crossfit_training_models": True,
        "joblib_compress": 3,
        "configuration_variant": (
            "run2_candidate_sequence_trajectory"
            if args.include_candidate_trajectory
            else (
                "run8_8layer_causal_negative_phenotypes"
                if args.causal_negative_phenotypes
                else "run1_strict_nested_structural"
            )
        ),
        "cells": cells,
    }
    if args.causal_negative_phenotypes:
        base_config = (
            repo
            / "experiments/hallucination/configs/one_confidence_mosaic_v1_nested_public.json"
        )
        payload["derivation"] = {
            "base_config": portable(base_config),
            "base_sha256": hashlib.sha256(base_config.read_bytes()).hexdigest(),
            "selection_scope": "public development/validation only",
            "single_change": (
                "replace only Mistral7/Qwen8 phenotype sources with frozen v2 "
                "causal-negative labels"
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
