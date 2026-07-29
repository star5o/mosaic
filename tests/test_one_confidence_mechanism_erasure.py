from __future__ import annotations

import hashlib
import json

import pytest
import torch

from run_one_confidence_mechanism_erasure import (
    fit_paired_direction,
    project_orthogonal,
    public_rows,
    random_unit_direction,
    summarize_hce,
    summarize_hcc,
    validate_inputs,
)


def test_paired_direction_is_oriented_and_centroid_separation_is_erased():
    canonical = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 3.0])]
    donor = [torch.tensor([2.0, 1.0]), torch.tensor([2.0, 3.0])]
    fitted = fit_paired_direction(canonical, donor)
    assert torch.allclose(fitted["direction"], torch.tensor([1.0, 0.0]))
    assert fitted["statistics"]["positive_projection_fraction"] == 1.0
    canonical_projected = project_orthogonal(
        torch.stack(canonical), fitted["direction"], fitted["center"]
    )
    donor_projected = project_orthogonal(
        torch.stack(donor), fitted["direction"], fitted["center"]
    )
    assert torch.allclose(canonical_projected.mean(0), donor_projected.mean(0))


def test_paired_direction_changes_when_donor_pairing_is_deranged():
    canonical = [torch.tensor([0.0, 0.0]), torch.tensor([100.0, 0.0])]
    donor = [torch.tensor([1.0, 0.0]), torch.tensor([100.0, 1.0])]
    aligned = fit_paired_direction(canonical, donor)["direction"]
    deranged = fit_paired_direction(canonical, list(reversed(donor)))["direction"]
    assert abs(float(aligned @ deranged)) < 0.9


def test_projection_removes_direction_and_is_idempotent():
    direction = torch.tensor([1.0, 0.0, 0.0])
    center = torch.tensor([2.0, 3.0, 4.0])
    values = torch.tensor([[9.0, 5.0, 7.0], [-1.0, 8.0, 0.0]])
    projected = project_orthogonal(values, direction, center)
    assert torch.allclose((projected - center) @ direction, torch.zeros(2))
    assert torch.allclose(
        project_orthogonal(projected, direction, center), projected
    )


def test_random_direction_is_seeded_and_unit_norm():
    reference = torch.zeros(16)
    left = random_unit_direction(reference, 17)
    right = random_unit_direction(reference, 17)
    other = random_unit_direction(reference, 18)
    assert torch.equal(left, right)
    assert not torch.equal(left, other)
    assert torch.allclose(left.norm(), torch.tensor(1.0), atol=1e-6)


def fixture_inputs():
    records = [
        {
            "sample_id": "dev-hce",
            "split": "development",
            "label": "HCE",
            "unrelated_sample_id": "val-hce",
        },
        {
            "sample_id": "val-hce",
            "split": "validation",
            "label": "HCE",
        },
        {
            "sample_id": "val-hcc",
            "split": "validation",
            "label": "HCC",
        },
    ]
    manifest = {"cell": "model|data", "locked_accessed": False, "records": records}
    patch = {
        "protocol": {"cell": "model|data", "locked_accessed": False},
        "layer_selection": {"selected_layer": 7},
        "development_scan_records": [{"sample_id": "dev-hce"}],
        "validation_hce_records": [{"sample_id": "val-hce"}],
    }
    return manifest, patch


def test_public_rows_are_frozen_to_patch_hce_cohorts():
    manifest, patch = fixture_inputs()
    development, validation_hce, validation_hcc = public_rows(manifest, patch)
    assert [row["sample_id"] for row in development] == ["dev-hce"]
    assert [row["sample_id"] for row in validation_hce] == ["val-hce"]
    assert [row["sample_id"] for row in validation_hcc] == ["val-hcc"]
    patch["validation_hce_records"] = []
    with pytest.raises(ValueError, match="validation HCE cohort"):
        public_rows(manifest, patch)


def test_validate_inputs_checks_hash_and_frozen_layer(tmp_path):
    manifest, patch = fixture_inputs()
    manifest_path = tmp_path / "manifest.json"
    patch_path = tmp_path / "patch.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    patch_path.write_text(json.dumps(patch), encoding="utf-8")
    protocol = {
        "cells": {
            "model|data": {
                "selected_layer": 7,
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "patch_result_sha256": hashlib.sha256(
                    patch_path.read_bytes()
                ).hexdigest(),
            }
        }
    }
    selected = validate_inputs(
        protocol, manifest, patch, manifest_path, patch_path
    )
    assert selected["selected_layer"] == 7
    protocol["cells"]["model|data"]["selected_layer"] = 8
    with pytest.raises(ValueError, match="selected layer"):
        validate_inputs(protocol, manifest, patch, manifest_path, patch_path)


def test_validate_inputs_rejects_answer_key_fields(tmp_path):
    manifest, patch = fixture_inputs()
    manifest["records"][0]["gold_answer"] = "forbidden"
    manifest_path = tmp_path / "manifest.json"
    patch_path = tmp_path / "patch.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    patch_path.write_text(json.dumps(patch), encoding="utf-8")
    protocol = {"cells": {"model|data": {}}}
    with pytest.raises(ValueError, match="answer-key"):
        validate_inputs(protocol, manifest, patch, manifest_path, patch_path)


def test_summaries_keep_true_and_all_controls_separate():
    hce = [
        {
            "true_direction_erasure": {
                "donor_candidate_margin_drop": 2.0,
                "canonical_candidate_margin_change": 0.5,
                "answer_start_js_reduction_donor_only": 0.2,
                "answer_start_js_reduction_both": 0.3,
                "donor_shuffled_margin_drop": 0.25,
            },
            "unrelated_direction_erasure": {"donor_candidate_margin_drop": 0.5},
            "random_direction_erasures": [
                {"donor_candidate_margin_drop": 0.25},
                {"donor_candidate_margin_drop": 0.75},
            ],
            "exact_noop": {"donor_candidate_margin_change": 0.0},
        }
    ]
    hce_summary = summarize_hce(hce)
    assert hce_summary["control_adjusted_margin_drop_mean"] == 1.5
    assert hce_summary["max_abs_noop_margin_change"] == 0.0

    hcc = [
        {
            "true_direction_erasure": {
                "canonical_sequence_score_change": -0.2,
                "answer_start_kl_from_baseline": 0.1,
                "answer_start_js_from_baseline": 0.05,
            },
            "unrelated_direction_erasure": {
                "canonical_sequence_score_change": -0.4
            },
            "random_direction_erasures": [
                {"canonical_sequence_score_change": -0.1},
                {"canonical_sequence_score_change": 0.1},
            ],
            "exact_noop": {"canonical_sequence_score_change": 0.0},
        }
    ]
    hcc_summary = summarize_hcc(hcc)
    assert hcc_summary["true_canonical_sequence_score_change_mean"] == -0.2
    assert hcc_summary["random_direction_sequence_score_change_mean"] == 0.0
    assert hcc_summary["max_abs_noop_sequence_score_change"] == 0.0
