#!/usr/bin/env python3
"""Compile retrospective frozen-cohort phenotype and closure artifacts.

This compiler is evaluation-only. It joins the already-frozen Run6 scores to
identifier-only application manifests and uses only the preregistered
behavioral phenotype flag. No internal/causal phenotype is inferred when the
frozen cohort lacks a causal assessment. Review files are consulted only to
attach a coarse failure type to the deterministic FP/FN ID list; question and
answer text are never exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


SELECTED_METHOD = "learned_gate_no_phenotype"
CELLS = (
    ("mistral7|TriviaQA", "mistral7-triviaqa"),
    ("qwen8|TriviaQA", "qwen8-triviaqa"),
    ("llama31|TriviaQA", "llama31-triviaqa"),
    ("qwen14|PopQA", "qwen14-popqa"),
    ("nemo12|PopQA", "nemo12-popqa"),
)
REVIEW_FILES = {
    "mistral7|TriviaQA": (
        "mistral-triviaqa-bf16-codex-reviewed.json",
        "mistral-triviaqa-expansion-bf16-codex-reviewed.json",
    ),
    "qwen8|TriviaQA": (
        "qwen8-triviaqa-bf16-codex-reviewed.json",
        "qwen8-triviaqa-expansion-bf16-codex-reviewed.json",
    ),
    "llama31|TriviaQA": (
        "llama31-triviaqa-bf16-codex-reviewed.json",
        "llama31-triviaqa-expansion-bf16-codex-reviewed.json",
    ),
    "qwen14|PopQA": (
        "qwen14-popqa-bf16-codex-reviewed.json",
        "qwen14-popqa-large-expansion-bf16-codex-reviewed.json",
    ),
    "nemo12|PopQA": (
        "nemo-popqa-bf16-codex-reviewed.json",
        "nemo-popqa-large-expansion-bf16-codex-reviewed.json",
    ),
}
FORBIDDEN_SOURCE_TOKENS = ("invalidated", "fresh_holdout", "fresh_method_holdout")
PHENOTYPES = (
    "behavioral_recoverable",
    "internal_competitive",
    "assessed_persistent",
    "unassessed_HCE",
    "HCC_control",
)
TABLE_FIELDS = (
    "aggregation",
    "cell",
    "split",
    "phenotype",
    "status",
    "count",
    "TP",
    "FN",
    "recall",
    "FP",
    "TN",
    "fpr",
    "interpretation",
)
RISK_FIELDS = (
    "aggregation",
    "cell",
    "split",
    "answer_coverage",
    "actual_coverage",
    "answered",
    "abstained",
    "hce_answered",
    "hcc_answered",
    "selective_accuracy",
    "selective_error_rate",
    "metric_aggregation",
    "selection_rule",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def reject_forbidden_source(path: Path) -> None:
    lowered = str(path.resolve(strict=False)).casefold()
    if any(token in lowered for token in FORBIDDEN_SOURCE_TOKENS):
        raise ValueError(f"forbidden source path: {path}")


def verify_declared(path: Path, declared: str, context: str) -> str:
    actual = sha256_file(path)
    if actual != declared:
        raise ValueError(f"{context} SHA256 mismatch: {actual} != {declared}")
    return actual


def keyed_records(payload: Mapping[str, Any], source: Path) -> dict[str, dict[str, Any]]:
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError(f"records list missing: {source}")
    output = {str(row["sample_id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate sample IDs: {source}")
    return output


def phenotype_for(label: str, behavioral_recoverable: bool) -> str:
    if label == "HCC":
        return "HCC_control"
    if label != "HCE":
        raise ValueError(f"unexpected class stratum: {label}")
    return "behavioral_recoverable" if behavioral_recoverable else "unassessed_HCE"


def confusion_outcome(label: str, predicted_hce: bool) -> str:
    if label == "HCE":
        return "TP" if predicted_hce else "FN"
    if label == "HCC":
        return "FP" if predicted_hce else "TN"
    raise ValueError(f"unexpected class stratum: {label}")


def summarize_phenotype_rows(cell: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for phenotype in PHENOTYPES:
        subset = [row for row in records if row["phenotype"] == phenotype]
        if phenotype == "HCC_control":
            fp = sum(bool(row["predicted_hce"]) for row in subset)
            output.append(
                {
                    "aggregation": "cell",
                    "cell": cell,
                    "split": "retrospective_frozen_test",
                    "phenotype": phenotype,
                    "status": "observed_control",
                    "count": len(subset),
                    "TP": None,
                    "FN": None,
                    "recall": None,
                    "FP": fp,
                    "TN": len(subset) - fp,
                    "fpr": fp / len(subset) if subset else None,
                    "interpretation": "matched high-confidence-correct control",
                }
            )
            continue
        if phenotype in {"internal_competitive", "assessed_persistent"}:
            output.append(
                {
                    "aggregation": "cell",
                    "cell": cell,
                    "split": "retrospective_frozen_test",
                    "phenotype": phenotype,
                    "status": "not_assessed_on_frozen_cohort",
                    "count": 0,
                    "TP": None,
                    "FN": None,
                    "recall": None,
                    "FP": None,
                    "TN": None,
                    "fpr": None,
                    "interpretation": "no causal phenotype was inferred post hoc",
                }
            )
            continue
        tp = sum(bool(row["predicted_hce"]) for row in subset)
        interpretation = (
            "observational behavioral-recovery HCE stratum"
            if phenotype == "behavioral_recoverable"
            else "observational residual HCE stratum; may contain persistent or unmeasured internal failures"
        )
        output.append(
            {
                "aggregation": "cell",
                "cell": cell,
                "split": "retrospective_frozen_test",
                "phenotype": phenotype,
                "status": "observed" if subset else "empty",
                "count": len(subset),
                "TP": tp,
                "FN": len(subset) - tp,
                "recall": tp / len(subset) if subset else None,
                "FP": None,
                "TN": None,
                "fpr": None,
                "interpretation": interpretation,
            }
        )
    return output


def macro_phenotype_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for phenotype in PHENOTYPES:
        rows = [row for row in cell_rows if row["phenotype"] == phenotype]
        eligible = [
            row
            for row in rows
            if row["fpr" if phenotype == "HCC_control" else "recall"] is not None
        ]
        metric = "fpr" if phenotype == "HCC_control" else "recall"
        output.append(
            {
                "aggregation": "macro",
                "cell": f"macro_{len(eligible)}_eligible_cells",
                "split": "retrospective_frozen_test",
                "phenotype": phenotype,
                "status": "equal_cell_mean" if eligible else "not_assessed_or_empty",
                "count": sum(int(row["count"]) for row in rows),
                "TP": None,
                "FN": None,
                "recall": mean(float(row["recall"]) for row in eligible)
                if metric == "recall" and eligible
                else None,
                "FP": None,
                "TN": None,
                "fpr": mean(float(row["fpr"]) for row in eligible)
                if metric == "fpr" and eligible
                else None,
                "interpretation": "equal-cell mean across non-empty, actually assessed strata",
            }
        )
    return output


def risk_coverage_rows(
    cell: str,
    records: list[dict[str, Any]],
    coverages: tuple[float, ...],
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: (float(row["detector_score"]), row["sample_id"]))
    output = []
    for coverage in coverages:
        answered_count = int(round(len(ordered) * coverage))
        answered = ordered[:answered_count]
        hcc_answered = sum(row["label"] == "HCC" for row in answered)
        hce_answered = len(answered) - hcc_answered
        actual_coverage = len(answered) / len(ordered) if ordered else None
        selective_accuracy = hcc_answered / len(answered) if answered else None
        output.append(
            {
                "aggregation": "cell",
                "cell": cell,
                "split": "retrospective_frozen_test",
                "answer_coverage": coverage,
                "actual_coverage": actual_coverage,
                "answered": len(answered),
                "abstained": len(ordered) - len(answered),
                "hce_answered": hce_answered,
                "hcc_answered": hcc_answered,
                "selective_accuracy": selective_accuracy,
                "selective_error_rate": 1.0 - selective_accuracy
                if selective_accuracy is not None
                else None,
                "metric_aggregation": "cell",
                "selection_rule": "answer the lowest frozen Run6-risk rows; ties break by sample_id",
            }
        )
    return output


def macro_risk_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for coverage in sorted({float(row["answer_coverage"]) for row in cell_rows}):
        rows = [row for row in cell_rows if float(row["answer_coverage"]) == coverage]
        selective = [float(row["selective_accuracy"]) for row in rows if row["selective_accuracy"] is not None]
        actual = [float(row["actual_coverage"]) for row in rows if row["actual_coverage"] is not None]
        output.append(
            {
                "aggregation": "macro",
                "cell": f"macro_{len(rows)}_cells",
                "split": "retrospective_frozen_test",
                "answer_coverage": coverage,
                "actual_coverage": mean(actual) if actual else None,
                "answered": sum(int(row["answered"]) for row in rows),
                "abstained": sum(int(row["abstained"]) for row in rows),
                "hce_answered": sum(int(row["hce_answered"]) for row in rows),
                "hcc_answered": sum(int(row["hcc_answered"]) for row in rows),
                "selective_accuracy": mean(selective) if selective else None,
                "selective_error_rate": 1.0 - mean(selective) if selective else None,
                "metric_aggregation": "equal-cell mean for rates; pooled sum for counts",
                "selection_rule": "per-cell lowest frozen Run6 risk followed by equal-cell macro",
            }
        )
    return output


def review_row_id(sample_id: str) -> str:
    parts = sample_id.split("|", 2)
    if len(parts) != 3 or not parts[2]:
        raise ValueError(f"cannot derive review row ID from sample ID: {sample_id}")
    return parts[2]


def load_review_index(
    cell: str,
    review_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    index: dict[str, dict[str, Any]] = {}
    hashes = {}
    for filename in REVIEW_FILES[cell]:
        path = review_root / filename
        reject_forbidden_source(path)
        payload = read_json(path)
        hashes[str(path)] = sha256_file(path)
        rows = payload.get("records")
        if not isinstance(rows, list):
            raise ValueError(f"review records missing: {path}")
        for row in rows:
            row_id = str(row.get("row_id", ""))
            if not row_id:
                raise ValueError(f"review row lacks row_id: {path}")
            compact = {
                "failure_type": row.get("failure_type"),
                "quality_category": row.get("quality_category"),
                "review_status": row.get("review_status"),
                "review_reason": row.get("review_reason"),
                "final_correct": row.get("final_correct"),
                "question": row.get("question"),
                "raw_model_output": row.get("raw_model_output"),
                "extracted_answer": row.get("answer"),
                "gold_aliases": row.get("gold_answers"),
                "sequence_confidence": row.get("mean_token_probability"),
                "model_provenance": row.get("model_provenance"),
                "source_path": str(path),
                "source_sha256": hashes[str(path)],
            }
            previous = index.get(row_id)
            if previous is not None:
                comparable = (
                    "failure_type",
                    "quality_category",
                    "review_status",
                    "review_reason",
                    "final_correct",
                    "question",
                    "raw_model_output",
                    "extracted_answer",
                    "gold_aliases",
                    "sequence_confidence",
                    "model_provenance",
                )
                if any(previous[key] != compact[key] for key in comparable):
                    raise ValueError(f"conflicting review metadata for {cell}/{row_id}")
                continue
            index[row_id] = compact
    return index, hashes


def select_case_rows(
    cell: str,
    records: list[dict[str, Any]],
    review_index: Mapping[str, Mapping[str, Any]],
    per_outcome: int,
    application_sha: str,
    score_sha: str,
) -> list[dict[str, Any]]:
    output = []
    for outcome in ("FN", "FP"):
        selected = sorted(
            (row for row in records if row["outcome"] == outcome),
            key=lambda row: row["sample_id"],
        )[:per_outcome]
        for row in selected:
            review_id = review_row_id(row["sample_id"])
            if review_id not in review_index:
                raise ValueError(f"review metadata missing for frozen case: {cell}/{review_id}")
            review = review_index[review_id]
            expected_correct = outcome == "FP"
            if review.get("final_correct") is not expected_correct:
                raise ValueError(
                    f"frozen case correctness disagrees with review: {cell}/{review_id}"
                )
            provenance = review.get("model_provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError(f"model provenance missing for frozen case: {cell}/{review_id}")
            dtype = str(provenance.get("dtype", ""))
            if dtype.casefold() not in {"bfloat16", "bf16", "torch.bfloat16"}:
                raise ValueError(f"frozen case is not BF16: {cell}/{review_id}/{dtype}")
            gold_aliases = review.get("gold_aliases")
            if not isinstance(gold_aliases, list) or not gold_aliases:
                raise ValueError(f"gold aliases missing for frozen case: {cell}/{review_id}")
            if outcome == "FN":
                failure_type = str(
                    review.get("failure_type")
                    or review.get("quality_category")
                    or "other_factual_error_untyped"
                )
            else:
                failure_type = "high_confidence_correct_control"
            output.append(
                {
                    "cell": cell,
                    "sample_id": row["sample_id"],
                    "pair_id": row["pair_id"],
                    "fact_key_sha256": row["fact_key_sha256"],
                    "class_stratum": row["label"],
                    "outcome": outcome,
                    "failure_type": failure_type,
                    "question": str(review.get("question") or ""),
                    "raw_model_output": str(review.get("raw_model_output") or ""),
                    "extracted_answer": str(review.get("extracted_answer") or ""),
                    "gold_aliases": [str(value) for value in gold_aliases],
                    "primary_gold": str(gold_aliases[0]),
                    "sequence_confidence": float(review["sequence_confidence"]),
                    "confidence_definition": "geometric mean probability over answer-content tokens only",
                    "review_status": review.get("review_status"),
                    "review_reason": review.get("review_reason"),
                    "model_id": provenance.get("model_id"),
                    "model_revision": provenance.get("model_revision"),
                    "dtype": dtype,
                    "chat_template_sha256": provenance.get("chat_template_sha256"),
                    "selection_rule": f"lexicographically first {per_outcome} sample IDs per cell and FP/FN outcome",
                    "detector_method": SELECTED_METHOD,
                    "application_manifest_sha256": application_sha,
                    "frozen_score_sha256": score_sha,
                    "review_source_path": review.get("source_path"),
                    "review_source_sha256": review.get("source_sha256"),
                }
            )
    return output


def load_cell_records(
    cell: str,
    slug: str,
    application_root: Path,
    score_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    application_path = application_root / f"{slug}-frozen-application.json"
    score_path = score_root / f"{slug}-frozen-scores.json"
    for path in (application_path, score_path):
        reject_forbidden_source(path)
    application = read_json(application_path)
    score = read_json(score_path)
    application_sha = sha256_file(application_path)
    score_sha = sha256_file(score_path)
    if application.get("cell") != cell or application.get("cohort_kind") != "frozen_test":
        raise ValueError(f"application identity mismatch: {cell}")
    if application.get("selected_method") != SELECTED_METHOD:
        raise ValueError(f"application selected method mismatch: {cell}")
    if score.get("cell") != cell or score.get("selected_method") != SELECTED_METHOD:
        raise ValueError(f"score identity mismatch: {cell}")
    if score.get("application_manifest_sha256") != application_sha:
        raise ValueError(f"score/application SHA mismatch: {cell}")
    for field in (
        "refit_performed",
        "threshold_search_performed",
        "layer_selection_performed",
        "class_stratum_used_for_prediction",
    ):
        if score.get(field) is not False:
            raise ValueError(f"frozen score guarantee violated: {cell}/{field}")

    application_records = application.get("records")
    if not isinstance(application_records, list) or not application_records:
        raise ValueError(f"application records missing: {cell}")
    application_by_id = {str(row["sample_id"]): row for row in application_records}
    if len(application_by_id) != len(application_records):
        raise ValueError(f"duplicate application sample IDs: {cell}")
    pairs: dict[str, list[str]] = defaultdict(list)
    for row in application_records:
        pairs[str(row["pair_id"])].append(str(row["class_stratum"]))
    if any(sorted(classes) != ["HCC", "HCE"] for classes in pairs.values()):
        raise ValueError(f"malformed frozen matched pair: {cell}")

    method = score.get("methods", {}).get(SELECTED_METHOD)
    if not isinstance(method, Mapping):
        raise ValueError(f"selected Run6 score missing: {cell}")
    threshold = float(method["frozen_validation_operating_point"]["threshold"])
    score_records = method.get("records")
    if not isinstance(score_records, list):
        raise ValueError(f"score records missing: {cell}")
    score_by_id = {str(row["sample_id"]): row for row in score_records}
    if len(score_by_id) != len(score_records) or set(score_by_id) != set(application_by_id):
        raise ValueError(f"score/application sample alignment failure: {cell}")

    candidate_spec = application.get("feature_sources", {}).get("candidate")
    if not isinstance(candidate_spec, Mapping):
        raise ValueError(f"candidate source missing from application: {cell}")
    candidate_path = Path(str(candidate_spec["path"]))
    reject_forbidden_source(candidate_path)
    candidate_sha = verify_declared(
        candidate_path,
        str(candidate_spec["sha256"]),
        f"{cell} candidate source",
    )
    candidate_payload = read_json(candidate_path)
    if candidate_payload.get("cell") != cell or candidate_payload.get("locked_accessed") is not True:
        raise ValueError(f"candidate source identity mismatch: {cell}")
    candidate_by_id = keyed_records(candidate_payload, candidate_path)
    if set(candidate_by_id) != set(application_by_id):
        raise ValueError(f"candidate/application sample alignment failure: {cell}")

    records = []
    for sample_id in sorted(application_by_id):
        application_row = application_by_id[sample_id]
        score_row = score_by_id[sample_id]
        candidate_row = candidate_by_id[sample_id]
        value = float(score_row["score"])
        predicted = value >= threshold
        if bool(score_row["predicted_hce"]) != predicted:
            raise ValueError(f"declared prediction differs from frozen threshold: {cell}/{sample_id}")
        label = str(application_row["class_stratum"])
        behavioral = bool(candidate_row.get("behavioral_recoverable_margin_pass"))
        records.append(
            {
                "cell": cell,
                "sample_id": sample_id,
                "pair_id": str(application_row["pair_id"]),
                "fact_key_sha256": str(application_row["fact_key_sha256"]),
                "split": "retrospective_frozen_test",
                "label": label,
                "phenotype": phenotype_for(label, behavioral),
                "behavioral_recoverable_margin_pass": behavioral,
                "internal_competitive_assessed": False,
                "assessed_persistent": False,
                "detector_method": SELECTED_METHOD,
                "detector_score": value,
                "detector_threshold": threshold,
                "predicted_hce": predicted,
                "outcome": confusion_outcome(label, predicted),
            }
        )
    return records, {
        str(application_path): application_sha,
        str(score_path): score_sha,
        str(candidate_path): candidate_sha,
    }


def archive_if_needed(path: Path, archive_path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    current_sha = sha256_file(path)
    if archive_path.exists():
        archive_sha = sha256_file(archive_path)
    else:
        archive_path.write_bytes(path.read_bytes())
        archive_sha = sha256_file(archive_path)
    return {
        "source_path": str(path),
        "source_sha256": current_sha,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha,
        "archive_preexisted": archive_path.exists() and archive_sha != current_sha,
    }


def compile_closure(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.application_root, args.score_root, args.review_root):
        reject_forbidden_source(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    phenotype_rows = []
    risk_rows = []
    case_rows = []
    source_hashes: dict[str, str] = {}
    review_hashes: dict[str, str] = {}

    for cell, slug in CELLS:
        records, hashes = load_cell_records(
            cell, slug, args.application_root, args.score_root
        )
        source_hashes.update(hashes)
        all_records.extend(records)
        phenotype_rows.extend(summarize_phenotype_rows(cell, records))
        risk_rows.extend(
            risk_coverage_rows(cell, records, tuple(args.coverages))
        )
        review_index, cell_review_hashes = load_review_index(cell, args.review_root)
        review_hashes.update(cell_review_hashes)
        application_sha = hashes[str(args.application_root / f"{slug}-frozen-application.json")]
        score_sha = hashes[str(args.score_root / f"{slug}-frozen-scores.json")]
        case_rows.extend(
            select_case_rows(
                cell,
                records,
                review_index,
                args.case_per_outcome,
                application_sha,
                score_sha,
            )
        )

    phenotype_rows.extend(macro_phenotype_rows(phenotype_rows))
    risk_rows.extend(macro_risk_rows(risk_rows))
    overlap = {
        "behavioral_and_internal_competitive": 0,
        "behavioral_and_assessed_persistent": 0,
        "internal_competitive_and_assessed_persistent": 0,
        "note": "internal and persistent causal phenotypes were not assessed on this retrospective frozen cohort",
    }
    phenotype_payload = {
        "schema_version": "one-confidence-mosaic-v1-frozen-observational-phenotypes-v1",
        "result_scope": "retrospective_frozen_test",
        "locked_accessed": True,
        "selected_method": SELECTED_METHOD,
        "causal_claim_allowed": False,
        "definitions": {
            "behavioral_recoverable": "the preregistered natural-paraphrase support and candidate-margin flag passes",
            "internal_competitive": "not assessed on this frozen cohort; no label is inferred from detector scores",
            "assessed_persistent": "not assessed on this frozen cohort; absence of behavioral recovery is insufficient",
            "unassessed_HCE": "strict HCE without the frozen behavioral-recovery flag; may include persistent or unmeasured internal failures",
            "HCC_control": "matched high-confidence-correct control",
        },
        "evidence_availability": {
            "behavioral_observational": True,
            "internal_causal": False,
            "persistent_causal_negative": False,
        },
        "overlap": overlap,
        "records": all_records,
        "source_sha256": dict(sorted(source_hashes.items())),
    }
    table_payload = {
        "schema_version": "one-confidence-mosaic-v1-frozen-detection-by-phenotype-v1",
        "result_scope": "retrospective_frozen_test",
        "selected_method": SELECTED_METHOD,
        "phenotype_status": "observational_and_partial_not_causal",
        "overlap": overlap,
        "rows": phenotype_rows,
    }
    risk_payload = {
        "schema_version": "one-confidence-mosaic-v1-frozen-risk-coverage-v1",
        "result_scope": "retrospective_frozen_test",
        "selected_method": SELECTED_METHOD,
        "rows": risk_rows,
    }
    case_payload = {
        "schema_version": "one-confidence-mosaic-v1-frozen-fp-fn-case-ids-v1",
        "result_scope": "retrospective_frozen_test",
        "selected_method": SELECTED_METHOD,
        "selection_is_deterministic": True,
        "contains_question_or_answer_text": True,
        "retrospective_frozen_only": True,
        "fresh_correctness_accessed": False,
        "rows": case_rows,
        "review_source_sha256": dict(sorted(review_hashes.items())),
    }

    archives = []
    for current, archive in (
        (
            args.output_dir / "table_detection_by_phenotype.json",
            args.output_dir / "table_detection_by_phenotype_legacy_public_validation.json",
        ),
        (
            args.output_dir / "table_detection_by_phenotype.csv",
            args.output_dir / "table_detection_by_phenotype_legacy_public_validation.csv",
        ),
        (
            args.output_dir / "figure_risk_coverage.json",
            args.output_dir / "figure_risk_coverage_legacy_public_validation.json",
        ),
        (
            args.output_dir / "figure_risk_coverage.csv",
            args.output_dir / "figure_risk_coverage_legacy_public_validation.csv",
        ),
    ):
        archived = archive_if_needed(current, archive)
        if archived is not None:
            archives.append(archived)

    outputs = {
        "phenotype_labels_frozen_test.json": phenotype_payload,
        "table_detection_by_phenotype.json": table_payload,
        "figure_risk_coverage.json": risk_payload,
        "case_studies.json": case_payload,
    }
    for name, payload in outputs.items():
        atomic_json(args.output_dir / name, payload)
    write_csv(args.output_dir / "table_detection_by_phenotype.csv", phenotype_rows, TABLE_FIELDS)
    write_csv(args.output_dir / "figure_risk_coverage.csv", risk_rows, RISK_FIELDS)

    output_paths = [
        args.output_dir / "phenotype_labels_frozen_test.json",
        args.output_dir / "table_detection_by_phenotype.json",
        args.output_dir / "table_detection_by_phenotype.csv",
        args.output_dir / "figure_risk_coverage.json",
        args.output_dir / "figure_risk_coverage.csv",
        args.output_dir / "case_studies.json",
    ]
    manifest = {
        "schema_version": "one-confidence-mosaic-v1-frozen-closure-manifest-v1",
        "result_scope": "retrospective_frozen_test",
        "selected_method": SELECTED_METHOD,
        "fresh_correctness_accessed": False,
        "archives": archives,
        "sources": dict(sorted({**source_hashes, **review_hashes}.items())),
        "outputs": {str(path): sha256_file(path) for path in output_paths},
    }
    atomic_json(args.output_dir / "frozen_closure_artifact_sha256.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-per-outcome", type=int, default=2)
    parser.add_argument(
        "--coverages",
        type=float,
        nargs="+",
        default=(0.50, 0.60, 0.70, 0.80, 0.90, 1.00),
    )
    args = parser.parse_args()
    if args.case_per_outcome <= 0:
        raise ValueError("--case-per-outcome must be positive")
    if any(value <= 0.0 or value > 1.0 for value in args.coverages):
        raise ValueError("coverages must be in (0, 1]")
    manifest = compile_closure(args)
    print(
        json.dumps(
            {
                "status": "compiled",
                "outputs": manifest["outputs"],
                "fresh_correctness_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
