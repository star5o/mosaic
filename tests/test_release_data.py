
from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_frozen_headline_metrics_and_counts():
    table = load("results/final/table_main_detection.json")
    run6 = table["macro_equal_cell_average"]["mosaic_v1_run6"]
    assert run6["accuracy"] == pytest.approx(0.6927675664805374)
    assert run6["recall"] == pytest.approx(0.4816720939240352)
    assert run6["fpr"] == pytest.approx(0.09613696096296039)
    pooled = table["pooled_supplement"]["mosaic_v1_run6"]
    assert tuple(pooled[key] for key in ("TP", "FN", "FP", "TN")) == (161, 166, 33, 294)


def test_mechanism_and_mitigation_verdicts_remain_negative():
    audit = load("results/final/mechanism_evidence_audit.json")
    assert audit["mechanism_summary"]["validation_hce_with_patch_and_erasure"] == 19
    assert audit["mechanism_summary"]["bidirectional_supported_cells"] == []
    mitigation = load("results/final/table_mitigation.json")
    frozen = mitigation["retrospective_frozen_test_scope"]["rows"]
    macro = {
        row["method"]: row
        for row in frozen
        if row["aggregation"] == "macro"
    }
    assert macro["mosaic_conditional_routing"]["net_correction"] == -30
    assert macro["uniform_likelihood_aggregation"]["net_correction"] == 5
    assert macro["uniform_steering"]["available"] is False


def test_release_excludes_fresh_evaluation_and_large_tensors():
    external = load("external_artifacts.json")
    assert external["included_in_git"] is False
    assert external["fresh_holdout"]["correctness_reviewed"] is False
    assert external["fresh_holdout"]["detector_scored"] is False
    assert not list(ROOT.rglob("*.joblib"))
    assert not list(ROOT.rglob("*.pt"))
