#!/usr/bin/env python3
"""Tests for the deterministic public-only question-equivalence audit."""

from __future__ import annotations

import copy

import pytest

from audit_one_confidence_question_equivalence import (
    audit_rewrite,
    compile_audit,
    fallback_text,
    standalone_panel_payload,
    stable_sha256,
)


def test_accepts_target_and_constraints_preserving_rewrite() -> None:
    source = "What country connected the Bushehr plant to its national grid in Sep 2011?"
    rewrite = "Which country connected the Bushehr plant to the national grid in September 2011?"
    result = audit_rewrite(source, rewrite)
    assert result["accepted"]
    assert result["diagnostics"]["source_answer_target"] == "country"
    assert result["diagnostics"]["rewrite_answer_target"] == "country"


def test_rejects_answer_target_drift() -> None:
    source = "What country connected the Bushehr plant to its national grid in Sep 2011?"
    rewrite = "What was the name of the power station connected to a national grid in September 2011?"
    result = audit_rewrite(source, rewrite)
    assert not result["accepted"]
    assert "answer_target_mismatch:country->facility" in result["reasons"]


def test_rejects_count_to_binary_drift() -> None:
    result = audit_rewrite(
        "How many countries border the Sargasso Sea?",
        "Is the Sargasso Sea bordered by a specific number of countries?",
    )
    assert not result["accepted"]
    assert "answer_target_mismatch:count->binary" in result["reasons"]


def test_rejects_dropped_name_and_temporal_constraint() -> None:
    result = audit_rewrite(
        "Who was the first chairman of the British Rail Board in the 1960s?",
        "Who directed railway reductions in Britain?",
    )
    assert not result["accepted"]
    assert any(reason.startswith("missing_named_anchors:") for reason in result["reasons"])
    assert "missing_constraint_operators:first" in result["reasons"]
    assert any(reason.startswith("missing_numeric_constraints:") for reason in result["reasons"])


def test_rejects_changed_relation_even_when_entity_anchor_survives() -> None:
    result = audit_rewrite(
        "Where is the ancient game of senet traditionally played?",
        "Where did the traditional practice of playing senet originate?",
    )
    assert not result["accepted"]
    assert "added_semantic_axes:origin" in result["reasons"]


def test_sentence_initial_entity_is_a_protected_anchor() -> None:
    result = audit_rewrite(
        "Cassiterite is a principal ore of which metal?",
        "What is the principal mineral from which a widely used metal is extracted?",
    )
    assert not result["accepted"]
    assert any(reason.startswith("missing_named_anchors:cassiterite") for reason in result["reasons"])


def test_fallback_is_unique_and_contains_source_verbatim() -> None:
    source = "Which country is this asking about?"
    values = [fallback_text(source, index) for index in range(4)]
    assert len(set(values)) == 4
    assert all(source in value for value in values)


def fixture_payloads() -> tuple[dict, dict]:
    records = []
    for index, split in enumerate(("development", "validation")):
        records.append(
            {
                "sample_id": f"cell|row{index}",
                "split": split,
                "class": "HCE" if index == 0 else "HCC",
                "question": "How many countries border the Sargasso Sea?",
            }
        )
    benchmark = {
        "cells": {
            "cell": {
                "development": {"records": [records[0]]},
                "validation": {"records": [records[1]]},
            }
        }
    }
    features = {
        "cell": "cell",
        "locked_accessed": False,
        "records": [
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "label": row["class"],
                "rewrite_status": "model_four_unique",
                "rewrite_raw_output": "raw",
                "paraphrases": [
                    "What is the number of countries bordering the Sargasso Sea?",
                    "Is the Sargasso Sea bordered by a specific number of countries?",
                    "How many countries have a border with the Sargasso Sea?",
                    "What total count of countries borders the Sargasso Sea?",
                ],
            }
            for row in records
        ],
    }
    return benchmark, features


def test_compile_is_deterministic_and_label_blind() -> None:
    benchmark, features = fixture_payloads()
    first = compile_audit(benchmark, features, "cell")
    second = compile_audit(copy.deepcopy(benchmark), copy.deepcopy(features), "cell")
    assert stable_sha256(first) == stable_sha256(second)
    assert first["locked_accessed"] is False
    assert first["summary"]["records"] == 2
    assert first["records"][0]["fallback_count"] == 1
    assert first["records"][0]["rewrites"][1]["selected_source"] == "deterministic_wrapper_fallback"
    assert first["records"][0]["source_question"] in first["records"][0]["rewrites"][1]["selected_text"]
    panel = first["corrected_panel_manifest"]
    assert panel["contains_labels"] is False
    assert "label" not in panel["records"][0]
    assert len(panel["records"][0]["panel_questions"]) == 5
    assert panel["records"][0]["panel_questions"][0] == first["records"][0]["source_question"]

    relabeled_benchmark = copy.deepcopy(benchmark)
    relabeled_features = copy.deepcopy(features)
    relabeled_benchmark["cells"]["cell"]["development"]["records"][0]["class"] = "HCC"
    relabeled_features["records"][0]["label"] = "HCC"
    assert stable_sha256(first) == stable_sha256(compile_audit(relabeled_benchmark, relabeled_features, "cell"))


def test_standalone_panel_is_label_free_and_bound_to_audit() -> None:
    benchmark, features = fixture_payloads()
    audit = compile_audit(benchmark, features, "cell")
    audit["input_provenance"] = {"benchmark_sha256": "b", "features_sha256": "f"}
    audit["artifact_sha256_excluding_this_field"] = stable_sha256(audit)
    panel = standalone_panel_payload(audit)
    assert panel["contains_labels"] is False
    assert panel["audit_artifact_sha256"] == audit["artifact_sha256_excluding_this_field"]
    assert all("label" not in row and "class" not in row for row in panel["records"])
    expected_hash = panel.pop("artifact_sha256_excluding_this_field")
    assert stable_sha256(panel) == expected_hash


def test_rejects_locked_or_nonpublic_feature_input() -> None:
    benchmark, features = fixture_payloads()
    features["locked_accessed"] = True
    with pytest.raises(ValueError, match="locked_accessed=false"):
        compile_audit(benchmark, features, "cell")

    benchmark, features = fixture_payloads()
    features["records"][0]["split"] = "locked_test"
    with pytest.raises(ValueError, match="non-public split"):
        compile_audit(benchmark, features, "cell")


def test_requires_exact_public_id_coverage() -> None:
    benchmark, features = fixture_payloads()
    features["records"].pop()
    with pytest.raises(ValueError, match="exactly the public"):
        compile_audit(benchmark, features, "cell")
