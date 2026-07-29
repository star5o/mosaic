from __future__ import annotations

import json

import torch

from build_one_confidence_mechanism_patch_manifest import (
    compile_manifest,
    contains_forbidden_key,
    load_audit,
)
from run_one_confidence_mechanism_patching import (
    random_replacement,
    summarize_hce,
)


def fixture_payloads():
    rows = []
    natural = []
    nli = []
    for index, label in enumerate(("HCE", "HCE", "HCC", "HCC")):
        sample_id = f"model|Dataset|{index}"
        rows.append(
            {
                "sample_id": sample_id,
                "split": "development" if index < 2 else "validation",
                "class": label,
                "fact_key_sha256": f"fact-{index}",
                "question": f"Question {index}?",
            }
        )
        canonical = f"canonical-{index}"
        answers = (
            [f"candidate-{index}", f"candidate-{index}", "other-a", "other-b"]
            if label == "HCE"
            else [canonical, canonical, "other-a", "other-b"]
        )
        natural.append(
            {
                "sample_id": sample_id,
                "split": rows[-1]["split"],
                "label": label,
                "paraphrases": [f"P{item} {index}" for item in range(4)],
                "canonical_answer": canonical,
                "panel_answers": [canonical, *answers],
                "panel_answer_confidences": [0.9, 0.8, 0.95, 0.7, 0.6],
            }
        )
        nli.append(
            {
                "sample_id": sample_id,
                "split": rows[-1]["split"],
                "label": label,
                "cluster_ids": [0, 0, 1, 2],
                "alternative_candidate": f"candidate-{index}" if label == "HCE" else "",
                "alternative_candidate_cluster": 0 if label == "HCE" else None,
            }
        )
    benchmark = {
        "cells": {
            "model|Dataset": {
                "development": {"records": rows[:2]},
                "validation": {"records": rows[2:]},
            }
        }
    }
    natural_payload = {
        "cell": "model|Dataset",
        "locked_accessed": False,
        "records": natural,
    }
    nli_payload = {
        "cell": "model|Dataset",
        "locked_accessed": False,
        "records": nli,
    }
    audit = {
        row["sample_id"]: {
            "sample_id": row["sample_id"],
            "equivalent_mask": [True, True, True, True],
            "review_status": "codex_reviewed",
        }
        for row in rows
    }
    return benchmark, natural_payload, nli_payload, audit


def test_manifest_uses_nli_candidate_and_audited_donor_without_gold():
    benchmark, natural, nli, audit = fixture_payloads()
    rows, counts = compile_manifest(
        benchmark, natural, nli, "model|Dataset", audit, False
    )
    assert len(rows) == 4
    assert counts["included_total"] == 4
    first = rows[0]
    assert first["candidate_answer"] == "candidate-0"
    assert first["donor_panel_index"] == 1
    assert first["candidate_support_panel_indices"] == [0, 1]
    assert all(row["question_equivalence_status"] == "codex_reviewed" for row in rows)
    assert contains_forbidden_key(rows) is None


def test_manifest_refuses_unreviewed_rewrites_by_default():
    benchmark, natural, nli, _ = fixture_payloads()
    rows, counts = compile_manifest(benchmark, natural, nli, "model|Dataset", {}, False)
    assert rows == []
    assert counts["skip_missing_question_equivalence_audit"] == 4


def test_rule_audit_accepts_five_element_mask_without_claiming_human_review(
    tmp_path,
):
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sample_id": "row-1",
                        "natural_paraphrase_mask": [True, True, False, True, False],
                        "review_status": "deterministic_rule_audited_v1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    audit = load_audit(path)
    assert audit["row-1"]["equivalent_mask"] == [True, False, True, False]
    assert audit["row-1"]["review_status"] == "deterministic_rule_audited_v1"


def test_unverified_override_is_explicit_and_controls_are_deranged():
    benchmark, natural, nli, _ = fixture_payloads()
    rows, _ = compile_manifest(benchmark, natural, nli, "model|Dataset", {}, True)
    hce = [row for row in rows if row["label"] == "HCE"]
    assert len(hce) == 2
    assert all(
        row["question_equivalence_status"] == "unverified_diagnostic" for row in rows
    )
    assert all(row["unrelated_sample_id"] != row["sample_id"] for row in hce)
    assert all(row["shuffled_candidate_sample_id"] != row["sample_id"] for row in hce)


def test_norm_matched_random_direction_preserves_displacement_norm():
    recipient = torch.tensor([1.0, 2.0, 3.0])
    donor = torch.tensor([3.0, 4.0, 7.0])
    replacement = random_replacement(recipient, donor, 7)
    assert torch.allclose(
        (replacement - recipient).norm(),
        (donor - recipient).norm(),
        atol=1e-5,
    )


def test_hce_layer_summary_uses_strongest_control():
    records = [
        {
            "layers": [
                {
                    "layer": 4,
                    "aligned_effect": 2.0,
                    "reverse_effect": -1.0,
                    "random_direction_effect_mean": 0.5,
                    "unrelated_effect": 0.25,
                    "shuffled_candidate_effect": 1.0,
                    "answer_start_js_reduction": 0.1,
                    "noop_effect": 0.0,
                }
            ]
        }
    ]
    summary = summarize_hce(records, [4])[0]
    assert summary["primary_control_adjusted_effect_mean"] == 1.0
    assert summary["bidirectional_fraction"] == 1.0
