import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch

from evaluate_one_confidence_mosaic_v1_nested_public import (
    CONFIG_SCHEMA_VERSION,
    CellData,
    assert_public_path,
    benchmark_rows,
    crossfit_stack_predictions,
    enumerate_thresholds,
    evaluate_cell,
    grouped_folds,
    load_cell_data,
    make_expert_estimator,
    validate_feature_field,
    validate_global_config,
)
from build_one_confidence_strict_partial_phenotypes import classify


def synthetic_cell() -> CellData:
    rng = np.random.default_rng(7)
    group_count = 48
    rows_per_group = 2
    groups = np.repeat([hashlib.sha256(f"fact-{index}".encode()).hexdigest() for index in range(group_count)], rows_per_group)
    labels_by_group = np.asarray([index % 2 for index in range(group_count)], dtype=int)
    labels = np.repeat(labels_by_group, rows_per_group)
    row_count = len(labels)
    signal = labels[:, None, None] * 0.35
    states = {
        position: rng.normal(size=(row_count, 3, 6)).astype(np.float32) + signal
        for position in ("prompt_end", "first_answer", "mean_answer")
    }
    features = np.column_stack(
        (
            labels + rng.normal(scale=0.7, size=row_count),
            rng.normal(size=row_count),
            rng.normal(size=row_count),
            np.ones(row_count),
            labels + rng.normal(scale=1.0, size=row_count),
            rng.normal(size=row_count),
            (labels + rng.normal(scale=0.8, size=row_count) > 0.5).astype(float),
            rng.normal(size=row_count),
        )
    )
    names = [
        "behavior.answer_start_js",
        "behavior.cross_nll",
        "candidate.margin",
        "candidate.available",
        "persistence.semantic_entropy",
        "persistence.self_consistency",
        "nli.recoverable",
        "nli.disagreement",
    ]
    return CellData(
        cell="model|Dataset",
        sample_ids=[f"sample-{index}" for index in range(row_count)],
        splits=np.asarray(["development"] * row_count, dtype=object),
        labels=labels,
        groups=np.asarray(groups, dtype=object),
        states=states,
        selected_layer_indices=[0, 1, 2],
        selected_layer_numbers=[1, 2, 3],
        feature_matrix=features,
        feature_names=names,
        behavioral_indices=[0, 1, 6, 7],
        candidate_indices=[2, 3],
        persistence_indices=[4, 5],
        evidence_indices=[2, 3, 6, 7],
        routing={
            "behavioral_indicator": "nli.recoverable",
            "candidate_available": "candidate.available",
            "candidate_margin": "candidate.margin",
            "feature_name_to_index": {name: index for index, name in enumerate(names)},
        },
        phenotype_labels=np.where(labels == 1, np.arange(row_count) % 3, -1),
        phenotype_names=["behavioral", "internal", "persistent"],
        source_hashes={},
        model_revision="synthetic",
        dtype="float32",
    )


def nested_config() -> dict:
    return {
        "outer_folds": 3,
        "base_inner_folds": 2,
        "gate_inner_folds": 2,
        "gate_base_inner_folds": 2,
        "base_probe_c": 0.01,
        "expert_c": 0.1,
        "gate_c": 0.1,
        "phenotype_gate_hidden": 6,
        "phenotype_gate_epochs": 15,
        "phenotype_gate_lr": 0.02,
        "phenotype_gate_l2": 0.001,
        "phenotype_aux_weight": 0.3,
        "max_fpr": 0.1,
        "serialize_crossfit_training_models": True,
        "joblib_compress": 0,
    }


def assert_all_fold_maps_are_group_disjoint(value):
    if isinstance(value, dict):
        if "train_fact_key_sha256" in value:
            assert set(value["train_fact_key_sha256"]).isdisjoint(value["holdout_fact_key_sha256"])
        for child in value.values():
            assert_all_fold_maps_are_group_disjoint(child)
    elif isinstance(value, list):
        for child in value:
            assert_all_fold_maps_are_group_disjoint(child)


def test_grouped_folds_keep_repeated_fact_rows_together():
    data = synthetic_cell()
    folds = grouped_folds(data.labels, data.groups, 5, 3, "test")
    for train, holdout in folds:
        assert set(data.groups[train]).isdisjoint(data.groups[holdout])
    held_out = np.concatenate([holdout for _, holdout in folds])
    assert sorted(held_out.tolist()) == list(range(len(data.labels)))


def test_strict_nested_stack_produces_finite_outer_oof_and_auditable_maps():
    data = synthetic_cell()
    scores, outer_map, training = crossfit_stack_predictions(
        data,
        nested_config(),
        seed=11,
        retain_training_models=False,
    )
    assert set(scores) == {
        "behavioral_only",
        "internal_only",
        "persistence_only",
        "simple_average",
        "flat_feature_concatenation",
        "hard_rule_routing",
        "learned_gate_no_phenotype",
        "mosaic_v1_phenotype_auxiliary_gate",
    }
    assert all(np.isfinite(values).all() for values in scores.values())
    assert_all_fold_maps_are_group_disjoint(outer_map)
    assert_all_fold_maps_are_group_disjoint(training[0]["nested_fold_maps"])


@pytest.mark.parametrize(
    "kind, expected_step, fit_locally",
    [
        ("logistic", "logisticregression", True),
        # The local macOS Python 3.13 sklearn wheel segfaults inside HGB's
        # native binning code.  The Blackwell Linux environment runs the
        # actual fit smoke test before any experiment is launched.
        ("hist_gradient_boosting", "histgradientboostingclassifier", False),
        ("mlp_two_layer", "mlpclassifier", True),
    ],
)
def test_expert_classifier_options_are_fitted_and_finite(kind, expected_step, fit_locally):
    rng = np.random.default_rng(19)
    features = rng.normal(size=(64, 9))
    labels = np.asarray([0, 1] * 32, dtype=int)
    config = {
        "expert_classifier": kind,
        "expert_c": 0.1,
        "expert_hgb_max_iter": 10,
        "expert_mlp_max_iter": 20,
        "expert_mlp_hidden": 8,
        "expert_mlp_second_hidden": 4,
    }
    estimator = make_expert_estimator(config, 0.1, seed=23)
    assert expected_step in estimator.named_steps
    if fit_locally:
        estimator.fit(features, labels)
        assert np.isfinite(estimator.predict_proba(features)).all()


def test_evaluate_cell_serializes_complete_models_and_no_locked_fields(tmp_path):
    data = synthetic_cell()
    validation_groups = {hashlib.sha256(f"val-{index}".encode()).hexdigest() for index in range(8)}
    validation_indices = np.flatnonzero(np.isin(data.groups, list(validation_groups)))
    assert len(validation_indices) == 0
    # Move four groups from each class to validation while preserving fact isolation.
    chosen = []
    for label in (0, 1):
        label_groups = list(dict.fromkeys(data.groups[data.labels == label]))
        chosen.extend(label_groups[:4])
    mask = np.isin(data.groups, chosen)
    data.splits[mask] = "validation"
    result = evaluate_cell(data, nested_config(), tmp_path, seed=13)
    assert result["development_rows"] > 0
    assert result["validation_rows"] > 0

    cell_dir = tmp_path / "model-dataset"
    checkpoint = joblib.load(cell_dir / "checkpoint.joblib")
    assert checkpoint["locked_accessed"] is False
    final_model = checkpoint["validation_model_fit_on_development_only"]
    assert final_model["base_full"]["models"]
    assert final_model["experts"]["models"]
    assert set(final_model["gates"]["models"]) == {"no_phenotype", "full"}
    assert final_model["gates"]["models"]["full"]["kind"] == "phenotype_auxiliary_mixture_gate_v1"
    assert final_model["gates"]["models"]["full"]["training"]["phenotype_labeled_rows"] > 0
    first_probe = next(iter(final_model["base_full"]["models"].values()))
    assert "standardscaler" in first_probe.named_steps
    assert "logisticregression" in first_probe.named_steps
    assert "expert_oof_mean" in final_model["normalization"]

    fold_map = json.loads((cell_dir / "fold_map.json").read_text())
    assert fold_map["locked_accessed"] is False
    assert_all_fold_maps_are_group_disjoint(fold_map)
    thresholds = json.loads((cell_dir / "threshold_candidates.json").read_text())
    assert thresholds["methods"]["mosaic_v1_phenotype_auxiliary_gate"]["validation_candidates"]
    scores = json.loads((cell_dir / "public_scores.json").read_text())
    assert all(
        row["split"] != "locked"
        for row in scores["methods"]["mosaic_v1_phenotype_auxiliary_gate"]["scores"]
    )


def test_load_cell_data_uses_expanded_fact_hash_and_parameterized_sources(tmp_path):
    cell = "model|Dataset"
    sample_ids = ["a", "b", "c", "d"]
    classes = ["HCE", "HCC", "HCE", "HCC"]
    splits = ["development", "development", "validation", "validation"]
    facts = [hashlib.sha256(f"fact-{index}".encode()).hexdigest() for index in range(4)]
    benchmark = {
        "cells": {
            cell: {
                split: {
                    "records": [
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "class": label,
                            "fact_key_sha256": fact,
                        }
                        for sample_id, label, row_split, fact in zip(sample_ids, classes, splits, facts)
                        if row_split == split
                    ]
                }
                for split in ("development", "validation")
            }
        }
    }
    benchmark_path = tmp_path / "expanded-public.json"
    benchmark_path.write_text(json.dumps(benchmark))
    benchmark_hash = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    internal_path = tmp_path / "public-features.pt"
    torch.save(
        {
            "cell": cell,
            "locked_accessed": False,
            "benchmark_sha256": benchmark_hash,
            "row_ids": sample_ids,
            "splits": splits,
            "labels": classes,
            "layer_numbers": [1, 2],
            "states": {
                position: torch.randn(4, 2, 3)
                for position in ("prompt_end", "first_answer", "mean_answer")
            },
            "model_revision": "revision",
            "dtype": "bfloat16",
        },
        internal_path,
    )
    source_path = tmp_path / "corrected-behavior.json"
    source_path.write_text(
        json.dumps(
            {
                "cell": cell,
                "locked_accessed": False,
                "records": [
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "label": label,
                        "features": {"js": index / 10, "margin": index, "available": True},
                    }
                    for index, (sample_id, split, label) in enumerate(zip(sample_ids, splits, classes))
                ],
            }
        )
    )
    phenotype_path = tmp_path / "phenotype-labels.json"
    phenotype_path.write_text(
        json.dumps(
            {
                "cell": cell,
                "locked_accessed": False,
                "records": [
                    {
                        "sample_id": sample_id,
                        "phenotype": "behavioral" if label == "HCE" else "unlabeled",
                        "evidence": {
                            "natural_support_count": 2,
                            "support_margin": 0.2,
                            "reliable_natural_candidate": True,
                            "causal_assessed": False,
                            "causal_internal_support": False,
                            "true_effect": 0.0,
                            "random_control_effect": 0.0,
                            "unrelated_control_effect": 0.0,
                            "shuffled_control_effect": 0.0,
                            "reverse_effect": 0.0,
                            "validation_confirmed": False,
                        },
                    }
                    for sample_id, label in zip(sample_ids, classes)
                ],
            }
        )
    )
    cell_config = {
        "cell": cell,
        "internal_states": str(internal_path),
        "feature_sources": {
            "corrected": {
                "path": str(source_path),
                "fields": {
                    "js": "features.js",
                    "margin": "features.margin",
                    "available": "features.available",
                },
            }
        },
        "feature_groups": {
            "behavioral": ["corrected.js"],
            "candidate": ["corrected.margin", "corrected.available"],
            "persistence": ["corrected.js"],
            "gate_evidence": ["corrected.*"],
        },
        "routing": {
            "candidate_available": "corrected.available",
            "candidate_margin": "corrected.margin",
        },
        "phenotype_auxiliary": {
            "path": str(phenotype_path),
            "field": "phenotype",
            "mapping": {"behavioral": 0, "internal": 1, "persistent": 2},
            "ambiguous_values": ["unlabeled", "ambiguous"],
            "apply_to_classes": ["HCE"],
            "strict_rules": True,
            "evidence_fields": {
                "natural_support_count": "evidence.natural_support_count",
                "support_margin": "evidence.support_margin",
                "reliable_natural_candidate": "evidence.reliable_natural_candidate",
                "causal_assessed": "evidence.causal_assessed",
                "causal_internal_support": "evidence.causal_internal_support",
                "true_effect": "evidence.true_effect",
                "random_control_effect": "evidence.random_control_effect",
                "unrelated_control_effect": "evidence.unrelated_control_effect",
                "shuffled_control_effect": "evidence.shuffled_control_effect",
                "reverse_effect": "evidence.reverse_effect",
                "validation_confirmed": "evidence.validation_confirmed",
            },
        },
    }
    loaded = load_cell_data(benchmark, benchmark_path, cell_config, fixed_layer_count=2)
    assert loaded.groups.tolist() == facts
    assert loaded.feature_names == ["corrected.js", "corrected.margin", "corrected.available"]
    assert loaded.candidate_indices == [1, 2]


def test_public_guards_reject_locked_invalidated_and_supervision_fields(tmp_path):
    with pytest.raises(ValueError, match="forbidden path"):
        assert_public_path(tmp_path / "fresh_locked_test.json")
    with pytest.raises(ValueError, match="forbidden path"):
        assert_public_path(tmp_path / "features.invalidated.json")
    with pytest.raises(ValueError, match="forbidden supervision"):
        validate_feature_field("correctness", "features.score")
    with pytest.raises(ValueError, match="forbidden supervision"):
        validate_feature_field("risk", "gold.answer_probability")


def test_benchmark_requires_fact_key_sha256():
    benchmark = {
        "cells": {
            "cell": {
                "development": {"records": [{"sample_id": "a", "split": "development", "class": "HCE"}]},
                "validation": {"records": []},
            }
        }
    }
    with pytest.raises(ValueError, match="fact_key_sha256"):
        benchmark_rows(benchmark, "cell")


def test_threshold_candidates_include_reject_none_and_obey_fpr():
    labels = np.asarray([1, 1, 0, 0])
    scores = np.asarray([0.8, 0.2, 0.7, 0.1])
    candidates, selected = enumerate_thresholds(labels, scores, max_fpr=0.1)
    assert candidates
    assert selected["fpr"] <= 0.1
    assert selected["recall"] == 0.5


def test_unified_config_requires_exactly_five_cells():
    config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "seed": 1,
        "outer_folds": 5,
        "base_inner_folds": 4,
        "gate_inner_folds": 4,
        "gate_base_inner_folds": 3,
        "fixed_layer_count": 8,
        "base_probe_c": 0.01,
        "expert_c": 0.1,
        "gate_c": 0.1,
        "phenotype_gate_hidden": 8,
        "phenotype_gate_epochs": 100,
        "phenotype_gate_lr": 0.02,
        "phenotype_gate_l2": 0.001,
        "phenotype_aux_weight": 0.3,
        "max_fpr": 0.1,
        "cells": [{"cell": "only-one"}],
    }
    with pytest.raises(ValueError, match="exactly five cells"):
        validate_global_config(config)


def test_strict_partial_phenotypes_do_not_use_nll_proxy_or_assume_persistence():
    base = {
        "natural_support_count": 0,
        "support_margin": -2.0,
        "reliable_natural_candidate": False,
        "causal_assessed": False,
        "causal_internal_support": False,
        "true_effect": 0.0,
        "random_control_effect": 0.0,
        "unrelated_control_effect": 0.0,
        "shuffled_control_effect": 0.0,
        "reverse_effect": 0.0,
        "validation_confirmed": False,
    }
    assert classify("HCE", base) == "ambiguous"
    assert classify("HCC", {**base, "natural_support_count": 4, "support_margin": 2.0}) == "unlabeled"
    assert classify(
        "HCE",
        {**base, "natural_support_count": 2, "support_margin": 0.0, "reliable_natural_candidate": True},
    ) == "behavioral"
    assert classify(
        "HCE",
        {
            **base,
            "causal_assessed": True,
            "causal_internal_support": True,
            "validation_confirmed": True,
            "true_effect": 0.5,
            "random_control_effect": 0.1,
            "unrelated_control_effect": 0.2,
            "shuffled_control_effect": 0.0,
            "reverse_effect": -0.2,
        },
    ) == "internal"
    assert classify("HCE", {**base, "causal_assessed": True}) == "persistent"
