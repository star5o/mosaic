from __future__ import annotations

import pytest

from evaluate_one_confidence_frozen_mitigation_run6 import (
    assert_label_blind,
    build_policy_rows,
    macro_rows,
    summarize_outcomes,
)


def source_payload(cell: str, record: dict) -> dict:
    return {
        "cell": cell,
        "locked_accessed": True,
        "records": [record],
    }


def test_build_policy_rows_uses_frozen_validation_threshold_without_labels():
    cell = "mistral7|TriviaQA"
    panel = source_payload(
        cell,
        {
            "sample_id": "row-1",
            "canonical_answer": "Wrong",
            "panel_answers": ["Wrong", "Right", "Right", "Wrong", "Right"],
        },
    )
    nli = source_payload(
        cell,
        {
            "sample_id": "row-1",
            "panel_answer_valid_mask": [True, True, True, True],
            "cluster_ids": [0, 0, 1, 0],
            "cluster_representatives": ["Right", "Wrong"],
            "canonical_equivalent_mask": [False, False, True, False],
            "alternative_candidate_natural": "Right",
            "alternative_support_count_natural": 3,
        },
    )
    margin = source_payload(
        cell,
        {
            "sample_id": "row-1",
            "canonical_answer": "Wrong",
            "candidate_answer": "Right",
            "candidate_available": True,
            "panel_canonical_answer_nlls": [2.0] * 5,
            "panel_candidate_answer_nlls": [1.0] * 5,
        },
    )
    score = {
        "cell": cell,
        "locked_accessed": True,
        "selected_method": "learned_gate_no_phenotype",
        "methods": {
            "learned_gate_no_phenotype": {
                "frozen_validation_operating_point": {"threshold": 0.7},
                "records": [
                    {
                        "sample_id": "row-1",
                        "pair_id": "pair-1",
                        "fact_key_sha256": "a" * 64,
                        "score": 0.8,
                    }
                ],
            }
        },
    }
    rows, metadata = build_policy_rows(cell, panel, nli, margin, score)
    assert rows[0].high_risk
    assert rows[0].reliable_candidate
    assert rows[0].risk_threshold == 0.7
    assert not {"label", "class", "gold_aliases"}.intersection(metadata["row-1"])


def test_label_blind_guard_rejects_targets():
    assert_label_blind({"cells": {"x": [{"sample_id": "row-1"}]}})
    with pytest.raises(ValueError, match="gold_aliases"):
        assert_label_blind({"gold_aliases": ["Right"]})


def test_frozen_outcomes_use_precomputed_actions():
    decision_rows = [
        {
            "sample_id": "hce",
            "pair_id": "pair-1",
            "risk_threshold": 0.7,
            "decisions": {
                "mosaic_conditional_routing": {
                    "action": "RECOVER",
                    "output_answer": "Right",
                    "triggered": True,
                    "reason": "frozen",
                }
            },
        },
        {
            "sample_id": "hcc",
            "pair_id": "pair-1",
            "risk_threshold": 0.7,
            "decisions": {
                "mosaic_conditional_routing": {
                    "action": "KEEP",
                    "output_answer": "Correct",
                    "triggered": False,
                    "reason": "frozen",
                }
            },
        },
    ]
    runtime = {
        "hce": {"class": "HCE", "gold_aliases": ["Right"]},
        "hcc": {"class": "HCC", "gold_aliases": ["Correct"]},
    }
    summary, records = summarize_outcomes(
        "mistral7|TriviaQA",
        "mosaic_conditional_routing",
        decision_rows,
        runtime,
    )
    assert summary["hce_corrected"] == 1
    assert summary["hcc_damaged"] == 0
    assert summary["net_correction"] == 1
    assert all(row["output_correct"] for row in records)


def test_macro_is_equal_cell_for_rates_and_pooled_for_counts():
    rows = []
    for cell, recall in (("mistral7|TriviaQA", 0.2), ("qwen8|TriviaQA", 0.8)):
        rows.append(
            {
                "aggregation": "cell",
                "cell": cell,
                "method": "keep",
                "available": True,
                "coverage": 1.0,
                "selective_accuracy": 0.5,
                "recall": recall,
                "fpr": 0.1,
                "n": 2,
                "hce": 1,
                "hcc": 1,
                "keep_count": 2,
                "recover_count": 0,
                "abstain_count": 0,
                "high_risk_count": 0,
                "hce_corrected": 0,
                "hcc_damaged": 0,
                "net_correction": 0,
                "answered": 2,
                "correct_answered": 1,
                "trigger_tp": 0,
                "trigger_fn": 1,
                "trigger_fp": 0,
                "trigger_tn": 1,
                "candidate_adoption_count": None,
                "actual_generation_rows": None,
            }
        )
    keep = next(row for row in macro_rows(rows) if row["method"] == "keep")
    assert keep["recall"] == pytest.approx(0.5)
    assert keep["n"] == 4
