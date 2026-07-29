from __future__ import annotations

import json

import pytest

from evaluate_one_confidence_public_mitigation import (
    EvaluationTarget,
    PolicyInput,
    conditional_policy,
    cost_action_row,
    detector_abstention_policy,
    evaluate_decisions,
    macro_rows,
    random_recovery_ids,
    random_routing_policy,
    read_public_json,
    unavailable_steering_row,
    uniform_likelihood_policy,
    uniform_vote_policy,
)


def policy_row(**overrides):
    values = {
        "sample_id": "row-1",
        "split": "validation",
        "canonical_answer": "Wrong",
        "safe_panel_answers": ("Right", "Right", "Wrong", "Right"),
        "safe_panel_valid_mask": (True, True, True, True),
        "cluster_ids": (0, 0, 1, 0),
        "cluster_representatives": ("Right", "Wrong"),
        "canonical_equivalent_mask": (False, False, True, False),
        "natural_candidate": "Right",
        "natural_support_count": 3,
        "candidate_answer": "Right",
        "candidate_available": True,
        "panel_canonical_nlls": (2.0, 2.0, 2.0, 2.0, 2.0),
        "panel_candidate_nlls": (1.0, 1.0, 1.0, 1.0, 1.0),
        "risk_score": 0.8,
        "risk_threshold": 0.7,
    }
    values.update(overrides)
    return PolicyInput(**values)


def test_uniform_vote_includes_canonical_vote_but_recovers_clear_majority():
    decision = uniform_vote_policy(policy_row())
    assert decision.action == "RECOVER"
    assert decision.answer == "Right"
    tied = policy_row(
        safe_panel_answers=("Right", "Right", "Wrong", "Wrong"),
        cluster_ids=(0, 0, 1, 1),
        canonical_equivalent_mask=(False, False, True, True),
    )
    assert uniform_vote_policy(tied).action == "KEEP"


def test_uniform_likelihood_uses_all_frozen_prompt_nlls():
    assert uniform_likelihood_policy(policy_row()).action == "RECOVER"
    canonical_better = policy_row(
        panel_canonical_nlls=(0.1,) * 5,
        panel_candidate_nlls=(0.2,) * 5,
    )
    assert uniform_likelihood_policy(canonical_better).action == "KEEP"


def test_conditional_routing_is_fixed_by_risk_and_support_three():
    assert conditional_policy(policy_row()).action == "RECOVER"
    insufficient = policy_row(natural_support_count=2)
    assert conditional_policy(insufficient).action == "ABSTAIN"
    low_risk = policy_row(risk_score=0.6)
    assert conditional_policy(low_risk).action == "KEEP"
    assert detector_abstention_policy(policy_row()).action == "ABSTAIN"


def test_random_routing_preserves_conditional_recover_count():
    rows = [
        policy_row(sample_id=f"row-{index}", natural_support_count=3 if index < 2 else 1)
        for index in range(5)
    ]
    recovery_ids = random_recovery_ids(rows, "model|data", "validation", 7)
    assert len(recovery_ids) == 2
    decisions = [random_routing_policy(row, recovery_ids) for row in rows]
    assert sum(decision.action == "RECOVER" for decision in decisions) == 2
    assert all(decision.triggered for decision in decisions)


def test_policy_decisions_do_not_depend_on_labels_or_gold_aliases():
    row = policy_row()
    decision = conditional_policy(row)
    left = EvaluationTarget("HCE", ("Right",))
    right = EvaluationTarget("HCC", ("Something Else",))
    assert conditional_policy(row) == decision
    left_summary, _ = evaluate_decisions([row], {row.sample_id: left}, {row.sample_id: decision})
    right_summary, _ = evaluate_decisions([row], {row.sample_id: right}, {row.sample_id: decision})
    assert left_summary["recover_count"] == right_summary["recover_count"] == 1
    assert left_summary["hce_corrected"] == 1
    assert right_summary["hcc_damaged"] == 1


def test_metrics_report_hce_corrected_hcc_damage_and_trigger_rates():
    hce = policy_row(sample_id="hce")
    hcc = policy_row(sample_id="hcc", canonical_answer="Correct", natural_candidate="Wrong")
    targets = {
        "hce": EvaluationTarget("HCE", ("Right",)),
        "hcc": EvaluationTarget("HCC", ("Correct",)),
    }
    decisions = {
        "hce": conditional_policy(hce),
        "hcc": conditional_policy(hcc),
    }
    summary, _ = evaluate_decisions([hce, hcc], targets, decisions)
    assert summary["hce_corrected"] == 1
    assert summary["hcc_damaged"] == 1
    assert summary["net_correction"] == 0
    assert summary["recall"] == 1.0
    assert summary["fpr"] == 1.0


def test_public_loader_rejects_locked_invalidated_and_gold_visible(tmp_path):
    locked = tmp_path / "locked.json"
    locked.write_text(json.dumps({"locked_accessed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden path"):
        read_public_json(locked)
    invalidated = tmp_path / "invalidated.json"
    invalidated.write_text(json.dumps({"locked_accessed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden path"):
        read_public_json(invalidated)
    exposed = tmp_path / "public.json"
    exposed.write_text(json.dumps({"gold_visible": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="answer keys"):
        read_public_json(exposed)


def test_uniform_steering_zero_adoption_is_unavailable_not_fabricated():
    payload = {
        "validation_generations": [
            {
                "split": "validation",
                "candidate_exact_match": {"aligned_selected_layer": False},
            }
        ]
    }
    row = unavailable_steering_row("model|Dataset", "validation", payload)
    assert row["available"] is False
    assert row["candidate_adoption_count"] == 0
    assert row["actual_generation_rows"] == 1
    assert row["hce_corrected"] is None
    assert "candidate_adoption_zero" in row["availability_reason"]


def test_macro_uses_equal_cell_rate_mean_and_pooled_counts():
    rows = []
    for index, recall in enumerate((0.2, 0.8)):
        rows.append(
            {
                "split": "validation",
                "method": "keep",
                "available": True,
                "n": 10,
                "hce": 5,
                "hcc": 5,
                "keep_count": 10,
                "recover_count": 0,
                "abstain_count": 0,
                "high_risk_count": 0,
                "hce_corrected": index,
                "hcc_damaged": 0,
                "net_correction": index,
                "answered": 10,
                "coverage": 1.0,
                "correct_answered": 5,
                "selective_accuracy": 0.5,
                "trigger_tp": int(recall * 5),
                "trigger_fn": 5 - int(recall * 5),
                "trigger_fp": 0,
                "trigger_tn": 5,
                "recall": recall,
                "fpr": 0.0,
                "candidate_adoption_count": None,
                "actual_generation_rows": None,
            }
        )
    macro = next(
        row
        for row in macro_rows(rows)
        if row["split"] == "validation" and row["method"] == "keep"
    )
    assert macro["n"] == 20
    assert macro["recall"] == 0.5
    assert macro["metric_aggregation"].startswith("equal_cell_mean")


def test_cost_table_reports_protocol_operations_without_invented_runtime():
    rows = [policy_row(sample_id="a"), policy_row(sample_id="b")]
    summary = {
        "cell": "model|Dataset",
        "model": "model",
        "dataset": "Dataset",
        "split": "validation",
        "method": "mosaic_conditional_routing",
        "available": True,
        "n": 2,
        "keep_count": 1,
        "recover_count": 1,
        "abstain_count": 0,
        "high_risk_count": 1,
        "actual_generation_rows": None,
        "candidate_adoption_count": None,
    }
    cost = cost_action_row(summary, rows, self_consistency_k=4)
    assert cost["paraphrase_generation_queries"] == 8
    assert cost["self_consistency_sampling_generations"] == 8
    assert cost["canonical_hidden_forward_passes"] == 2
    assert cost["teacher_forced_sequence_scores"] == 20
    assert cost["wall_time_seconds"] is None
    assert cost["peak_gpu_memory_mb"] is None
