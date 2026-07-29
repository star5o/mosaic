import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

import apply_one_confidence_mosaic_v1_frozen_checkpoint as frozen_application_module
from apply_one_confidence_mosaic_v1_frozen_checkpoint import (
    APPLICATION_SCHEMA,
    apply_frozen_checkpoint,
)
from evaluate_one_confidence_mosaic_v1_nested_public import (
    METHOD_NAMES,
    CellData,
    fit_stack_model,
    predict_stack,
    sha256_file,
)


def make_cell() -> CellData:
    rng = np.random.default_rng(19)
    group_count = 36
    rows_per_group = 2
    groups = np.repeat(
        [hashlib.sha256(f"application-fact-{index}".encode()).hexdigest() for index in range(group_count)],
        rows_per_group,
    )
    labels = np.repeat(np.asarray([index % 2 for index in range(group_count)]), rows_per_group)
    row_count = len(labels)
    states = {
        position: (
            rng.normal(size=(row_count, 3, 5)).astype(np.float32)
            + labels[:, None, None] * 0.2
        )
        for position in ("prompt_end", "first_answer", "mean_answer")
    }
    feature_names = [
        "behavior.answer_start_js",
        "behavior.cross_nll",
        "candidate.margin",
        "candidate.available",
        "persistence.semantic_entropy",
        "persistence.self_consistency",
        "nli.recoverable",
        "nli.disagreement",
    ]
    features = np.column_stack(
        (
            labels + rng.normal(scale=0.6, size=row_count),
            rng.normal(size=row_count),
            rng.normal(size=row_count),
            np.ones(row_count),
            labels + rng.normal(size=row_count),
            rng.normal(size=row_count),
            (labels + rng.normal(scale=0.7, size=row_count) > 0.5).astype(float),
            rng.normal(size=row_count),
        )
    )
    return CellData(
        cell="synthetic|PublicQA",
        sample_ids=[f"row-{index}" for index in range(row_count)],
        splits=np.asarray(["development"] * row_count, dtype=object),
        labels=labels.astype(int),
        groups=np.asarray(groups, dtype=object),
        states=states,
        selected_layer_indices=[0, 1, 2],
        selected_layer_numbers=[1, 2, 3],
        feature_matrix=features,
        feature_names=feature_names,
        behavioral_indices=[0, 1, 6, 7],
        candidate_indices=[2, 3],
        persistence_indices=[4, 5],
        evidence_indices=[2, 3, 6, 7],
        routing={
            "behavioral_indicator": "nli.recoverable",
            "candidate_available": "candidate.available",
            "candidate_margin": "candidate.margin",
            "feature_name_to_index": {name: index for index, name in enumerate(feature_names)},
        },
        phenotype_labels=np.where(labels == 1, np.arange(row_count) % 3, -1),
        phenotype_names=["behavioral", "internal", "persistent"],
        source_hashes={},
        model_revision="synthetic-revision",
        dtype="bfloat16",
    )


def training_config() -> dict:
    return {
        "base_inner_folds": 2,
        "gate_inner_folds": 2,
        "gate_base_inner_folds": 2,
        "base_probe_c": 0.01,
        "expert_c": 0.1,
        "gate_c": 0.1,
        "phenotype_gate_hidden": 5,
        "phenotype_gate_epochs": 10,
        "phenotype_gate_lr": 0.02,
        "phenotype_gate_l2": 0.001,
        "phenotype_aux_weight": 0.3,
    }


def write_frozen_fixture(tmp_path: Path) -> tuple[CellData, dict[str, Path]]:
    data = make_cell()
    model, _fold_map = fit_stack_model(
        data,
        training_config(),
        seed=23,
        context="synthetic/application",
        retain_training_models=False,
    )
    checkpoint = {
        "schema_version": "synthetic-checkpoint",
        "cell": data.cell,
        "locked_accessed": False,
        "model_revision": data.model_revision,
        "dtype": data.dtype,
        "selected_layer_indices": data.selected_layer_indices,
        "selected_layer_numbers": data.selected_layer_numbers,
        "validation_model_fit_on_development_only": model,
    }
    checkpoint_path = tmp_path / "checkpoint.joblib"
    joblib.dump(checkpoint, checkpoint_path)
    thresholds_path = tmp_path / "thresholds.json"
    thresholds_path.write_text(
        json.dumps(
            {
                "cell": data.cell,
                "methods": {
                    name: {
                        "validation_selected_for_future_test": {
                            "threshold": 0.5,
                            "recall": 0.0,
                            "fpr": 0.0,
                        }
                    }
                    for name in METHOD_NAMES
                },
            }
        )
    )
    internal_path = tmp_path / "application-public-features.pt"
    torch.save(
        {
            "cell": data.cell,
            "locked_accessed": False,
            "model_revision": data.model_revision,
            "dtype": data.dtype,
            "row_ids": data.sample_ids,
            "layer_numbers": data.selected_layer_numbers,
            "states": {
                position: torch.from_numpy(values)
                for position, values in data.states.items()
            },
        },
        internal_path,
    )
    source_path = tmp_path / "application-public-features.json"
    source_path.write_text(
        json.dumps(
            {
                "cell": data.cell,
                "locked_accessed": False,
                "records": [
                    {
                        "sample_id": sample_id,
                        "features": {
                            "behavior": {
                                "answer_start_js": float(data.feature_matrix[index, 0]),
                                "cross_nll": float(data.feature_matrix[index, 1]),
                            },
                            "candidate": {
                                "margin": float(data.feature_matrix[index, 2]),
                                "available": float(data.feature_matrix[index, 3]),
                            },
                            "persistence": {
                                "semantic_entropy": float(data.feature_matrix[index, 4]),
                                "self_consistency": float(data.feature_matrix[index, 5]),
                            },
                            "nli": {
                                "recoverable": float(data.feature_matrix[index, 6]),
                                "disagreement": float(data.feature_matrix[index, 7]),
                            },
                        },
                    }
                    for index, sample_id in enumerate(data.sample_ids)
                ],
            }
        )
    )
    model_schema = model["feature_schema"]
    feature_schema_sha = hashlib.sha256(
        json.dumps(model_schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    application = {
        "schema_version": APPLICATION_SCHEMA,
        "cohort_kind": "public",
        "cell": data.cell,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "thresholds_sha256": sha256_file(thresholds_path),
        "model_revision": data.model_revision,
        "dtype": data.dtype,
        "selected_layer_numbers": data.selected_layer_numbers,
        "feature_schema_sha256": feature_schema_sha,
        "records": [
            {
                "sample_id": sample_id,
                "fact_key_sha256": str(group),
                "split": "public_application",
            }
            for sample_id, group in zip(data.sample_ids, data.groups)
        ],
        "internal_states": {
            "path": str(internal_path),
            "sha256": sha256_file(internal_path),
        },
        "feature_sources": {
            "behavior": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "fields": {
                    "answer_start_js": "features.behavior.answer_start_js",
                    "cross_nll": "features.behavior.cross_nll",
                },
            },
            "candidate": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "fields": {
                    "margin": "features.candidate.margin",
                    "available": "features.candidate.available",
                },
            },
            "persistence": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "fields": {
                    "semantic_entropy": "features.persistence.semantic_entropy",
                    "self_consistency": "features.persistence.self_consistency",
                },
            },
            "nli": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "fields": {
                    "recoverable": "features.nli.recoverable",
                    "disagreement": "features.nli.disagreement",
                },
            },
        },
        "feature_groups": {
            "behavioral": ["behavior.*", "nli.*"],
            "candidate": ["candidate.*"],
            "persistence": ["persistence.*"],
            "gate_evidence": ["candidate.*", "nli.*"],
        },
        "routing": {
            "behavioral_indicator": "nli.recoverable",
            "candidate_available": "candidate.available",
            "candidate_margin": "candidate.margin",
        },
    }
    application_path = tmp_path / "application.json"
    application_path.write_text(json.dumps(application))
    return data, {
        "checkpoint": checkpoint_path,
        "thresholds": thresholds_path,
        "internal": internal_path,
        "source": source_path,
        "application": application_path,
        "output": tmp_path / "scores.json",
    }


def test_frozen_application_matches_direct_prediction_and_never_refits(tmp_path):
    data, paths = write_frozen_fixture(tmp_path)
    checkpoint = joblib.load(paths["checkpoint"])
    expected = predict_stack(checkpoint["validation_model_fit_on_development_only"], data)
    with patch.object(LogisticRegression, "fit", side_effect=AssertionError("refit attempted")):
        payload = apply_frozen_checkpoint(
            paths["checkpoint"],
            paths["thresholds"],
            paths["application"],
            paths["output"],
        )
    assert payload["refit_performed"] is False
    assert payload["threshold_search_performed"] is False
    assert set(payload["methods"]) == set(METHOD_NAMES)
    for name in METHOD_NAMES:
        actual = np.asarray([row["score"] for row in payload["methods"][name]["records"]])
        np.testing.assert_allclose(actual, expected[name], rtol=0, atol=1e-6)
        assert payload["methods"][name]["frozen_validation_operating_point"]["threshold"] == 0.5
    assert "label" not in json.dumps(payload)


def test_frozen_application_source_exposes_no_fit_call():
    source = Path(frozen_application_module.__file__).read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "fit_transform(" not in source


def test_frozen_application_rejects_tampered_source_sha(tmp_path):
    _data, paths = write_frozen_fixture(tmp_path)
    application = json.loads(paths["application"].read_text())
    application["feature_sources"]["behavior"]["sha256"] = "0" * 64
    paths["application"].write_text(json.dumps(application))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        apply_frozen_checkpoint(
            paths["checkpoint"], paths["thresholds"], paths["application"], paths["output"]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_revision", "other-revision", "model_revision"),
        ("dtype", "float16", "dtype"),
        ("selected_layer_numbers", [1, 3], "selected layers"),
        ("feature_schema_sha256", "0" * 64, "feature_schema_sha256"),
    ],
)
def test_frozen_application_rejects_structural_mismatch(tmp_path, field, value, message):
    _data, paths = write_frozen_fixture(tmp_path)
    application = json.loads(paths["application"].read_text())
    application[field] = value
    paths["application"].write_text(json.dumps(application))
    with pytest.raises(ValueError, match=message):
        apply_frozen_checkpoint(
            paths["checkpoint"], paths["thresholds"], paths["application"], paths["output"]
        )


def test_public_mode_rejects_locked_path_before_opening_it(tmp_path):
    _data, paths = write_frozen_fixture(tmp_path)
    application = json.loads(paths["application"].read_text())
    application["feature_sources"]["behavior"]["path"] = str(tmp_path / "fresh_locked_rows.json")
    paths["application"].write_text(json.dumps(application))
    with pytest.raises(ValueError, match="forbidden paths"):
        apply_frozen_checkpoint(
            paths["checkpoint"], paths["thresholds"], paths["application"], paths["output"]
        )
