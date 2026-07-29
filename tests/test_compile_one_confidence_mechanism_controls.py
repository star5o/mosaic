from __future__ import annotations

import json

import pytest

from compile_one_confidence_mechanism_controls import (
    compile_layer_figure,
    compile_table,
    compile_verdict,
    load_public,
)


def patch_fixture(cell: str = "model|Dataset"):
    selected = {
        "layer": 8,
        "n": 4,
        "aligned_effect_mean": 2.0,
        "aligned_positive_fraction": 0.75,
        "random_direction_effect_mean": 0.5,
        "unrelated_effect_mean": 0.25,
        "shuffled_candidate_effect_mean": 1.0,
        "primary_control_adjusted_effect_mean": 0.75,
        "answer_start_js_reduction_mean": 0.2,
        "max_abs_noop_effect": 0.0,
    }
    random_layer = {
        **selected,
        "layer": 2,
        "aligned_effect_mean": 0.1,
        "primary_control_adjusted_effect_mean": -0.9,
    }
    hcc_selected = {
        "layer": 8,
        "n": 10,
        "aligned_canonical_score_change_mean": -0.01,
        "aligned_answer_start_kl_mean": 0.02,
        "random_direction_score_change_mean": 0.0,
        "unrelated_score_change_mean": -0.03,
        "max_abs_noop_score_change": 0.0,
    }
    hcc_random = {**hcc_selected, "layer": 2}
    return {
        "protocol": {"cell": cell, "locked_accessed": False},
        "layer_selection": {"selected_layer": 8, "random_control_layer": 2},
        "development_scan_summary": [random_layer, selected],
        "validation_hce_summary": [selected, random_layer],
        "hcc_control_summary": {
            "development": [hcc_selected, hcc_random],
            "validation": [hcc_selected, hcc_random],
        },
    }


def erasure_fixture(cell: str = "model|Dataset"):
    return {
        "protocol": {
            "cell": cell,
            "locked_accessed": False,
            "selected_layer": 8,
        },
        "validation_hce_summary": {
            "n": 4,
            "true_donor_candidate_margin_drop_mean": -0.2,
            "true_donor_candidate_margin_drop_positive_fraction": 0.25,
            "true_answer_start_js_reduction_donor_only_mean": 0.01,
            "random_direction_margin_drop_mean": 0.0,
            "unrelated_direction_margin_drop_mean": 0.3,
            "shuffled_candidate_margin_drop_mean": 0.1,
            "control_adjusted_margin_drop_mean": -0.5,
            "control_adjusted_positive_fraction": 0.0,
            "max_abs_noop_margin_change": 0.0,
        },
        "validation_hcc_summary": {
            "n": 12,
            "true_canonical_sequence_score_change_mean": -0.01,
            "true_answer_start_js_mean": 0.001,
            "random_direction_sequence_score_change_mean": 0.0,
            "unrelated_direction_sequence_score_change_mean": -0.02,
            "max_abs_noop_sequence_score_change": 0.0,
        },
    }


def write_payload(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_table_separates_patch_erasure_controls_and_hcc(tmp_path):
    patch_path = write_payload(tmp_path, "patch.json", patch_fixture())
    erase_path = write_payload(tmp_path, "erase.json", erasure_fixture())
    rows = compile_table(
        [(patch_fixture(), patch_path)], [(erasure_fixture(), erase_path)]
    )
    controls = {
        (row["experiment"], row["population"], row["control"]) for row in rows
    }
    assert ("late_mlp_patch", "HCE", "true_patch") in controls
    assert ("late_mlp_patch", "HCE", "random_layer_true_patch") in controls
    assert ("rank_one_erasure", "HCE", "true_erasure") in controls
    assert ("rank_one_erasure", "HCE", "shuffled_candidate") in controls
    assert ("rank_one_erasure", "HCC_DAMAGE", "exact_noop") in controls
    true_erasure = next(
        row
        for row in rows
        if row["experiment"] == "rank_one_erasure"
        and row["population"] == "HCE"
        and row["control"] == "true_erasure"
    )
    assert true_erasure["effect_metric"].endswith("margin_drop")
    assert true_erasure["control_adjusted_mean"] == -0.5


def test_layer_figure_contains_only_actually_run_patch_layers(tmp_path):
    path = write_payload(tmp_path, "patch.json", patch_fixture())
    rows = compile_layer_figure([(patch_fixture(), path)])
    assert len(rows) == 4
    assert {row["split"] for row in rows} == {"development", "validation"}
    assert {row["layer"] for row in rows} == {2, 8}
    selected = next(
        row
        for row in rows
        if row["split"] == "validation" and row["layer"] == 8
    )
    assert selected["layer_role"] == "selected_layer"
    assert selected["control_adjusted_effect_mean"] == 0.75


def test_verdict_requires_both_control_adjusted_directions(tmp_path):
    patch_path = write_payload(tmp_path, "patch.json", patch_fixture())
    erase_path = write_payload(tmp_path, "erase.json", erasure_fixture())
    verdict = compile_verdict(
        [(patch_fixture(), patch_path)], [(erasure_fixture(), erase_path)]
    )
    cell = verdict["cells"]["model|Dataset"]
    assert cell["patch_validation"]["control_adjusted_mean_positive"] is True
    assert cell["erasure_validation"]["control_adjusted_mean_positive"] is False
    assert cell["bidirectional_control_adjusted_support"] is False
    assert verdict["summary"]["shared_rank_one_bidirectional_mechanism_supported"] is False
    assert "No cell has both" in verdict["summary"]["verdict"]


def test_verdict_is_generated_from_actual_cells_not_hardcoded_models(tmp_path):
    positive_erasure = erasure_fixture("qwen14|PopQA")
    positive_erasure["validation_hce_summary"][
        "control_adjusted_margin_drop_mean"
    ] = 0.4
    patch = patch_fixture("qwen14|PopQA")
    patch_path = write_payload(tmp_path, "qwen14-patch.json", patch)
    erase_path = write_payload(tmp_path, "qwen14-erase.json", positive_erasure)
    verdict = compile_verdict(
        [(patch, patch_path)], [(positive_erasure, erase_path)]
    )
    summary = verdict["summary"]
    assert summary["bidirectional_control_adjusted_positive_cells"] == [
        "qwen14|PopQA"
    ]
    assert "qwen14|PopQA" in summary["verdict"]
    assert "Mistral" not in summary["verdict"]


def test_load_public_rejects_locked_and_invalidated(tmp_path):
    locked = write_payload(
        tmp_path,
        "locked.json",
        {"protocol": {"locked_accessed": True}},
    )
    with pytest.raises(ValueError, match="public"):
        load_public(locked)
    invalidated = write_payload(
        tmp_path,
        "invalidated-result.json",
        {"protocol": {"locked_accessed": False}},
    )
    with pytest.raises(ValueError, match="invalidated"):
        load_public(invalidated)
