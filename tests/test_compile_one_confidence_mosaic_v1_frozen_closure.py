from pathlib import Path

from compile_one_confidence_mosaic_v1_frozen_closure import (
    archive_if_needed,
    confusion_outcome,
    macro_risk_rows,
    phenotype_for,
    risk_coverage_rows,
    select_case_rows,
    summarize_phenotype_rows,
)


def _records():
    return [
        {
            "sample_id": "cell|data|a",
            "pair_id": "p1",
            "fact_key_sha256": "a" * 64,
            "label": "HCE",
            "phenotype": "behavioral_recoverable",
            "predicted_hce": True,
            "detector_score": 0.9,
            "outcome": "TP",
        },
        {
            "sample_id": "cell|data|b",
            "pair_id": "p1",
            "fact_key_sha256": "b" * 64,
            "label": "HCC",
            "phenotype": "HCC_control",
            "predicted_hce": True,
            "detector_score": 0.8,
            "outcome": "FP",
        },
        {
            "sample_id": "cell|data|c",
            "pair_id": "p2",
            "fact_key_sha256": "c" * 64,
            "label": "HCE",
            "phenotype": "unassessed_HCE",
            "predicted_hce": False,
            "detector_score": 0.1,
            "outcome": "FN",
        },
        {
            "sample_id": "cell|data|d",
            "pair_id": "p2",
            "fact_key_sha256": "d" * 64,
            "label": "HCC",
            "phenotype": "HCC_control",
            "predicted_hce": False,
            "detector_score": 0.2,
            "outcome": "TN",
        },
    ]


def test_phenotype_direction_and_confusion_direction():
    assert phenotype_for("HCE", True) == "behavioral_recoverable"
    assert phenotype_for("HCE", False) == "unassessed_HCE"
    assert phenotype_for("HCC", True) == "HCC_control"
    assert confusion_outcome("HCE", True) == "TP"
    assert confusion_outcome("HCE", False) == "FN"
    assert confusion_outcome("HCC", True) == "FP"
    assert confusion_outcome("HCC", False) == "TN"


def test_unassessed_internal_phenotypes_are_not_inferred():
    rows = summarize_phenotype_rows("cell|data", _records())
    by_name = {row["phenotype"]: row for row in rows}
    assert by_name["internal_competitive"]["count"] == 0
    assert by_name["internal_competitive"]["recall"] is None
    assert by_name["assessed_persistent"]["status"] == "not_assessed_on_frozen_cohort"
    assert by_name["behavioral_recoverable"]["recall"] == 1.0
    assert by_name["unassessed_HCE"]["recall"] == 0.0
    assert by_name["HCC_control"]["fpr"] == 0.5


def test_risk_coverage_answers_lowest_risk_first():
    rows = risk_coverage_rows("cell|data", _records(), (0.5, 1.0))
    assert rows[0]["answered"] == 2
    assert rows[0]["hce_answered"] == 1
    assert rows[0]["hcc_answered"] == 1
    assert rows[0]["selective_accuracy"] == 0.5
    macro = macro_risk_rows(rows)
    assert macro[0]["selective_accuracy"] == 0.5


def test_fixed_case_list_contains_frozen_real_case_fields():
    review = {
        "c": {
            "failure_type": "plausible fabrication",
            "quality_category": "strict_hce",
            "review_status": "codex_reviewed",
            "review_reason": "verified factual error",
            "final_correct": False,
            "question": "Which city?",
            "raw_model_output": "Wrong city",
            "extracted_answer": "Wrong city",
            "gold_aliases": ["Right city"],
            "sequence_confidence": 0.95,
            "model_provenance": {
                "model_id": "official/model",
                "model_revision": "revision",
                "dtype": "bfloat16",
                "chat_template_sha256": "c" * 64,
            },
            "source_path": "review.json",
            "source_sha256": "f" * 64,
        },
        "b": {
            "failure_type": None,
            "quality_category": "hcc",
            "review_status": "codex_reviewed",
            "review_reason": "alias match",
            "final_correct": True,
            "question": "Which person?",
            "raw_model_output": "Right person",
            "extracted_answer": "Right person",
            "gold_aliases": ["Right person"],
            "sequence_confidence": 0.96,
            "model_provenance": {
                "model_id": "official/model",
                "model_revision": "revision",
                "dtype": "bfloat16",
                "chat_template_sha256": "d" * 64,
            },
            "source_path": "review.json",
            "source_sha256": "f" * 64,
        },
    }
    rows = select_case_rows(
        "cell|data", _records(), review, 1, "a" * 64, "b" * 64
    )
    assert {row["outcome"] for row in rows} == {"FP", "FN"}
    fn = next(row for row in rows if row["outcome"] == "FN")
    assert fn["failure_type"] == "plausible fabrication"
    assert fn["question"] == "Which city?"
    assert fn["extracted_answer"] == "Wrong city"
    assert fn["gold_aliases"] == ["Right city"]
    assert fn["dtype"] == "bfloat16"


def test_archive_preserves_exact_bytes(tmp_path: Path):
    source = tmp_path / "current.json"
    archive = tmp_path / "legacy.json"
    source.write_bytes(b"legacy\n")
    record = archive_if_needed(source, archive)
    assert record is not None
    assert archive.read_bytes() == b"legacy\n"
    assert record["source_sha256"] == record["archive_sha256"]


def test_archive_is_idempotent_after_canonical_changes(tmp_path: Path):
    source = tmp_path / "current.json"
    archive = tmp_path / "legacy.json"
    source.write_bytes(b"legacy\n")
    archive_if_needed(source, archive)
    source.write_bytes(b"frozen\n")
    record = archive_if_needed(source, archive)
    assert archive.read_bytes() == b"legacy\n"
    assert record is not None and record["archive_preexisted"] is True
