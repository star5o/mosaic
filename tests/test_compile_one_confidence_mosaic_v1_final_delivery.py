from __future__ import annotations

import json

import pytest

from compile_one_confidence_mosaic_v1_final_delivery import (
    compile_case_studies,
    compile_mitigation,
    mitigation_csv_rows,
)


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compile_mitigation_preserves_public_and_frozen_scopes(tmp_path):
    public = tmp_path / "public.json"
    frozen = tmp_path / "frozen.json"
    decision = tmp_path / "decision.json"
    write_json(
        public,
        {
            "public_only": True,
            "locked_accessed": False,
            "risk_method": "learned_gate_no_phenotype",
            "rows": [{"method": "keep", "split": "validation"}],
        },
    )
    write_json(
        frozen,
        {
            "locked_accessed": True,
            "selected_method": "learned_gate_no_phenotype",
            "decision_sha256": "abc",
            "rows": [{"method": "keep", "split": "locked_test"}],
        },
    )
    write_json(
        decision,
        {
            "decision_stage_is_label_blind": True,
            "decision_sha256": "abc",
        },
    )
    payload = compile_mitigation(public, frozen, decision)
    assert payload["retrospective_frozen_test_scope"]["decisions_frozen_before_labels"]
    rows = mitigation_csv_rows(payload)
    assert {row["result_scope"] for row in rows} == {
        "public_validation_scope",
        "retrospective_frozen_test_scope",
    }


def test_compile_mitigation_rejects_decision_sha_mismatch(tmp_path):
    public = tmp_path / "public.json"
    frozen = tmp_path / "frozen.json"
    decision = tmp_path / "decision.json"
    write_json(
        public,
        {
            "public_only": True,
            "locked_accessed": False,
            "risk_method": "learned_gate_no_phenotype",
            "rows": [],
        },
    )
    write_json(
        frozen,
        {
            "locked_accessed": True,
            "selected_method": "learned_gate_no_phenotype",
            "decision_sha256": "left",
            "rows": [],
        },
    )
    write_json(
        decision,
        {"decision_stage_is_label_blind": True, "decision_sha256": "right"},
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        compile_mitigation(public, frozen, decision)


def test_case_studies_require_fixed_two_per_outcome_and_cell(tmp_path):
    source = tmp_path / "cases.json"
    rows = []
    for cell_index in range(5):
        for outcome in ("TP", "FN", "FP", "TN"):
            for case_index in range(2):
                rows.append(
                    {
                        "cell": f"cell-{cell_index}",
                        "sample_id": f"{cell_index}-{outcome}-{case_index}",
                        "outcome": outcome,
                        "selection_rule": "first two sample IDs per cell and confusion outcome",
                    }
                )
    write_json(source, {"selection_is_deterministic": True, "rows": rows})
    payload = compile_case_studies(source)
    assert payload["public_only"]
    assert len(payload["rows"]) == 40
