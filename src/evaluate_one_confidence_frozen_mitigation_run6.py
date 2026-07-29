#!/usr/bin/env python3
"""Freeze label-blind Run6 mitigation decisions, then score them once.

The ``decide`` stage reads only frozen feature artifacts and already-frozen
Run6 risk scores.  It writes decisions plus a SHA256 freeze record without
opening the runtime benchmark.  The separate ``evaluate`` stage verifies that
freeze record before it opens class labels and gold aliases for outcome
scoring.  This keeps action routing independent of correctness labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from evaluate_one_confidence_public_mitigation import (
    COST_FIELDS,
    METHODS,
    SUMMARY_FIELDS,
    Decision,
    PolicyInput,
    conditional_policy,
    detector_abstention_policy,
    keep_policy,
    random_recovery_ids,
    random_routing_policy,
    uniform_likelihood_policy,
    uniform_vote_policy,
)


AUTH_SCHEMA = "one-confidence-mosaic-v1-run6-mitigation-authorization-v1"
DECISION_SCHEMA = "one-confidence-mosaic-v1-run6-frozen-mitigation-decisions-v1"
FREEZE_SCHEMA = "one-confidence-mosaic-v1-run6-mitigation-decision-freeze-v1"
OUTCOME_SCHEMA = "one-confidence-mosaic-v1-run6-frozen-mitigation-outcomes-v1"
SELECTED_METHOD = "learned_gate_no_phenotype"
DEFAULT_SEED = 2026072905
CELLS = (
    "mistral7|TriviaQA",
    "qwen8|TriviaQA",
    "llama31|TriviaQA",
    "qwen14|PopQA",
    "nemo12|PopQA",
)
STEMS = {
    "mistral7|TriviaQA": "mistral7-triviaqa",
    "qwen8|TriviaQA": "qwen8-triviaqa",
    "llama31|TriviaQA": "llama31-triviaqa",
    "qwen14|PopQA": "qwen14-popqa",
    "nemo12|PopQA": "nemo12-popqa",
}
POLICY_METHODS = METHODS[:-1]
FORBIDDEN_DECISION_KEYS = {
    "class",
    "class_stratum",
    "gold",
    "gold_aliases",
    "label",
    "question",
    "source_final_correct",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_authorization(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("schema_version") != AUTH_SCHEMA:
        raise ValueError(f"authorization schema must be {AUTH_SCHEMA}")
    if payload.get("authorized") is not True:
        raise PermissionError("Run6 mitigation is not authorized")
    if payload.get("locked_access_acknowledged") is not True:
        raise PermissionError("locked access is not acknowledged")
    if payload.get("decision_stage_is_label_blind") is not True:
        raise PermissionError("authorization must require label-blind decisions")
    if payload.get("selected_method") != SELECTED_METHOD:
        raise ValueError("authorization selected method differs from Run6 freeze")
    if tuple(payload.get("cells", {}).keys()) != CELLS:
        raise ValueError("authorization cells/order differ from the frozen cohort")
    return payload


def verify_declared(path: Path, declared_sha: str, name: str) -> str:
    actual = sha256_file(path)
    if actual != declared_sha:
        raise ValueError(f"{name} SHA256 mismatch: {actual} != {declared_sha}")
    return actual


def keyed_records(payload: dict[str, Any], source: Path) -> dict[str, dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"records missing: {source}")
    keyed = {str(row["sample_id"]): row for row in records}
    if len(keyed) != len(records):
        raise ValueError(f"duplicate sample IDs: {source}")
    return keyed


def assert_label_blind(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_DECISION_KEYS.intersection(value)
        if forbidden:
            raise ValueError(f"label-blind decision artifact contains {sorted(forbidden)} at {location}")
        for key, child in value.items():
            assert_label_blind(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_label_blind(child, f"{location}[{index}]")


def split_cell(cell: str) -> tuple[str, str]:
    return tuple(cell.split("|", 1))  # type: ignore[return-value]


def build_policy_rows(
    cell: str,
    panel_payload: dict[str, Any],
    nli_payload: dict[str, Any],
    margin_payload: dict[str, Any],
    score_payload: dict[str, Any],
) -> tuple[list[PolicyInput], dict[str, dict[str, Any]]]:
    for name, payload in (
        ("panel", panel_payload),
        ("nli", nli_payload),
        ("margin", margin_payload),
        ("score", score_payload),
    ):
        if payload.get("cell") != cell:
            raise ValueError(f"{cell}: {name} cell mismatch")
        if payload.get("locked_accessed") is not True:
            raise ValueError(f"{cell}: {name} is not a frozen source")
    if score_payload.get("selected_method") != SELECTED_METHOD:
        raise ValueError(f"{cell}: score method differs from Run6 freeze")
    method_payload = score_payload.get("methods", {}).get(SELECTED_METHOD)
    if not isinstance(method_payload, dict):
        raise ValueError(f"{cell}: selected score payload missing")
    threshold = float(method_payload["frozen_validation_operating_point"]["threshold"])
    score_records = method_payload.get("records")
    if not isinstance(score_records, list):
        raise ValueError(f"{cell}: frozen score records missing")
    score_by_id = {str(row["sample_id"]): row for row in score_records}
    if len(score_by_id) != len(score_records):
        raise ValueError(f"{cell}: duplicate score IDs")

    panel_by_id = keyed_records(panel_payload, Path(f"{cell}:panel"))
    nli_by_id = keyed_records(nli_payload, Path(f"{cell}:nli"))
    margin_by_id = keyed_records(margin_payload, Path(f"{cell}:margin"))
    identifiers = set(score_by_id)
    if set(panel_by_id) != identifiers or set(nli_by_id) != identifiers or set(margin_by_id) != identifiers:
        raise ValueError(f"{cell}: source sample IDs are not identical")

    rows: list[PolicyInput] = []
    metadata: dict[str, dict[str, Any]] = {}
    for score_record in score_records:
        sample_id = str(score_record["sample_id"])
        panel = panel_by_id[sample_id]
        nli = nli_by_id[sample_id]
        margin = margin_by_id[sample_id]
        canonical_answer = str(panel["canonical_answer"])
        if canonical_answer != str(margin["canonical_answer"]):
            raise ValueError(f"{cell}/{sample_id}: canonical answer mismatch")
        panel_answers = tuple(str(value) for value in panel["panel_answers"])
        if len(panel_answers) != 5:
            raise ValueError(f"{cell}/{sample_id}: expected canonical plus four panel answers")
        row = PolicyInput(
            sample_id=sample_id,
            split="locked_test",
            canonical_answer=canonical_answer,
            safe_panel_answers=panel_answers[1:],
            safe_panel_valid_mask=tuple(bool(value) for value in nli["panel_answer_valid_mask"]),
            cluster_ids=tuple(int(value) for value in nli["cluster_ids"]),
            cluster_representatives=tuple(str(value) for value in nli["cluster_representatives"]),
            canonical_equivalent_mask=tuple(bool(value) for value in nli["canonical_equivalent_mask"]),
            natural_candidate=str(nli.get("alternative_candidate_natural") or ""),
            natural_support_count=int(nli.get("alternative_support_count_natural") or 0),
            candidate_answer=str(margin.get("candidate_answer") or ""),
            candidate_available=bool(margin.get("candidate_available")),
            panel_canonical_nlls=tuple(float(value) for value in margin.get("panel_canonical_answer_nlls", [])),
            panel_candidate_nlls=tuple(float(value) for value in margin.get("panel_candidate_answer_nlls", [])),
            risk_score=float(score_record["score"]),
            risk_threshold=threshold,
        )
        rows.append(row)
        metadata[sample_id] = {
            "sample_id": sample_id,
            "pair_id": str(score_record["pair_id"]),
            "fact_key_sha256": str(score_record["fact_key_sha256"]),
            "risk_score": row.risk_score,
            "risk_threshold": row.risk_threshold,
            "high_risk": row.high_risk,
            "reliable_candidate": row.reliable_candidate,
            "canonical_answer": row.canonical_answer,
            "natural_candidate": row.natural_candidate,
            "natural_support_count": row.natural_support_count,
            "candidate_answer": row.candidate_answer,
            "candidate_available": row.candidate_available,
            "panel_canonical_nll_count": len(row.panel_canonical_nlls),
            "panel_candidate_nll_count": len(row.panel_candidate_nlls),
        }
    return rows, metadata


def freeze_decisions(authorization_path: Path) -> dict[str, Any]:
    authorization = validate_authorization(authorization_path)
    output_path = Path(authorization["outputs"]["decisions"])
    freeze_path = Path(authorization["outputs"]["decision_freeze"])
    if output_path.exists() or freeze_path.exists():
        raise FileExistsError("one-shot Run6 mitigation decision output already exists")
    cells_output: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    seed = int(authorization["seed"])

    for cell in CELLS:
        specification = authorization["cells"][cell]
        payloads: dict[str, dict[str, Any]] = {}
        for name in ("panel", "nli", "margin", "score"):
            path = Path(specification[name]["path"])
            source_hashes[f"{cell}:{name}"] = verify_declared(
                path, str(specification[name]["sha256"]), f"{cell}:{name}"
            )
            payloads[name] = read_json(path)
        rows, metadata = build_policy_rows(
            cell,
            payloads["panel"],
            payloads["nli"],
            payloads["margin"],
            payloads["score"],
        )
        recovery_ids = random_recovery_ids(rows, cell, "locked_test", seed)
        policies: dict[str, Callable[[PolicyInput], Decision]] = {
            "keep": keep_policy,
            "uniform_paraphrase_vote": uniform_vote_policy,
            "uniform_likelihood_aggregation": uniform_likelihood_policy,
            "detector_only_abstention": detector_abstention_policy,
            "random_routing": lambda row, ids=recovery_ids: random_routing_policy(row, ids),
            "mosaic_conditional_routing": conditional_policy,
        }
        records = []
        for row in rows:
            decisions = {}
            for method, policy in policies.items():
                decision = policy(row)
                decisions[method] = {
                    "action": decision.action,
                    "output_answer": decision.answer,
                    "triggered": decision.triggered,
                    "reason": decision.reason,
                }
            records.append({**metadata[row.sample_id], "decisions": decisions})
        cells_output[cell] = {
            "row_count": len(records),
            "pair_count": len(records) // 2,
            "records": records,
        }

    decision_payload = {
        "schema_version": DECISION_SCHEMA,
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": sha256_file(authorization_path),
        "decision_stage_is_label_blind": True,
        "locked_runtime_opened": False,
        "selected_method": SELECTED_METHOD,
        "seed": seed,
        "routing_protocol": authorization["routing_protocol"],
        "source_sha256": source_hashes,
        "cells": cells_output,
    }
    assert_label_blind(decision_payload)
    atomic_json(output_path, decision_payload)
    decision_sha = sha256_file(output_path)
    freeze_payload = {
        "schema_version": FREEZE_SCHEMA,
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": sha256_file(authorization_path),
        "decision_path": str(output_path),
        "decision_sha256": decision_sha,
        "decision_stage_is_label_blind": True,
        "locked_runtime_opened_before_decision_freeze": False,
        "row_count": sum(cell["row_count"] for cell in cells_output.values()),
        "pair_count": sum(cell["pair_count"] for cell in cells_output.values()),
    }
    atomic_json(freeze_path, freeze_payload)
    return freeze_payload


def normalize(text: str) -> str:
    import re
    import unicodedata

    value = unicodedata.normalize("NFKD", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def alias_match(answer: str, aliases: list[str]) -> bool:
    normalized = normalize(answer)
    return bool(normalized) and normalized in {normalize(alias) for alias in aliases}


def summarize_outcomes(
    cell: str,
    method: str,
    decision_rows: list[dict[str, Any]],
    runtime_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = []
    for row in decision_rows:
        sample_id = str(row["sample_id"])
        target = runtime_by_id[sample_id]
        decision = row["decisions"][method]
        label = str(target["class"])
        action = str(decision["action"])
        answer = decision.get("output_answer")
        if action == "KEEP":
            output_correct = label == "HCC"
        elif action == "RECOVER" and answer is not None:
            output_correct = alias_match(str(answer), list(target["gold_aliases"]))
        else:
            output_correct = False
        records.append(
            {
                "sample_id": sample_id,
                "pair_id": row["pair_id"],
                "label": label,
                "method": method,
                "action": action,
                "triggered": bool(decision["triggered"]),
                "output_correct": output_correct,
                "reason": decision["reason"],
            }
        )
    hce = [row for row in records if row["label"] == "HCE"]
    hcc = [row for row in records if row["label"] == "HCC"]
    answered = [row for row in records if row["action"] != "ABSTAIN"]
    tp = sum(row["triggered"] for row in hce)
    fp = sum(row["triggered"] for row in hcc)
    hce_corrected = sum(row["output_correct"] for row in hce)
    hcc_damaged = sum(not row["output_correct"] for row in hcc)
    model, dataset = split_cell(cell)
    summary = {
        "aggregation": "cell",
        "cell": cell,
        "model": model,
        "dataset": dataset,
        "split": "locked_test",
        "method": method,
        "available": True,
        "availability_reason": "evaluated_from_prelabel_frozen_decisions",
        "metric_aggregation": "cell",
        "macro_cells": None,
        "n": len(records),
        "hce": len(hce),
        "hcc": len(hcc),
        "keep_count": sum(row["action"] == "KEEP" for row in records),
        "recover_count": sum(row["action"] == "RECOVER" for row in records),
        "abstain_count": sum(row["action"] == "ABSTAIN" for row in records),
        "high_risk_count": sum(row["triggered"] for row in records),
        "hce_corrected": hce_corrected,
        "hcc_damaged": hcc_damaged,
        "net_correction": hce_corrected - hcc_damaged,
        "answered": len(answered),
        "coverage": len(answered) / len(records),
        "correct_answered": sum(row["output_correct"] for row in answered),
        "selective_accuracy": (
            sum(row["output_correct"] for row in answered) / len(answered)
            if answered
            else None
        ),
        "trigger_tp": tp,
        "trigger_fn": len(hce) - tp,
        "trigger_fp": fp,
        "trigger_tn": len(hcc) - fp,
        "recall": tp / len(hce),
        "fpr": fp / len(hcc),
        "risk_threshold": (
            decision_rows[0]["risk_threshold"]
            if method in {"detector_only_abstention", "random_routing", "mosaic_conditional_routing"}
            else None
        ),
        "candidate_adoption_count": None,
        "actual_generation_rows": None,
    }
    return summary, records


def unavailable_steering(cell: str, n: int) -> dict[str, Any]:
    model, dataset = split_cell(cell)
    return {
        "aggregation": "cell",
        "cell": cell,
        "model": model,
        "dataset": dataset,
        "split": "locked_test",
        "method": "uniform_steering",
        "available": False,
        "availability_reason": "no_frozen_full_cohort_steering_generation; mechanism_controls_do_not_support_steering",
        "metric_aggregation": "unavailable",
        "macro_cells": None,
        **{field: None for field in SUMMARY_FIELDS[10:]},
        "n": n,
    }


def macro_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    rate_fields = ("coverage", "selective_accuracy", "recall", "fpr")
    count_fields = (
        "n", "hce", "hcc", "keep_count", "recover_count", "abstain_count",
        "high_risk_count", "hce_corrected", "hcc_damaged", "net_correction",
        "answered", "correct_answered", "trigger_tp", "trigger_fn", "trigger_fp",
        "trigger_tn", "candidate_adoption_count", "actual_generation_rows",
    )
    for method in METHODS:
        selected = [row for row in rows if row["aggregation"] == "cell" and row["method"] == method]
        available = bool(selected) and all(row["available"] for row in selected)
        macro = {
            "aggregation": "macro",
            "cell": "macro_5_cells",
            "model": "macro",
            "dataset": "mixed",
            "split": "locked_test",
            "method": method,
            "available": available,
            "availability_reason": "all_cells_evaluated" if available else "one_or_more_cells_unavailable",
            "metric_aggregation": "equal_cell_mean_for_rates; pooled_sum_for_counts",
            "macro_cells": len(selected),
            "risk_threshold": None,
        }
        for field in count_fields:
            values = [row[field] for row in selected if row.get(field) is not None]
            macro[field] = sum(values) if values else None
        for field in rate_fields:
            values = [float(row[field]) for row in selected if row.get(field) is not None]
            macro[field] = sum(values) / len(values) if available and values else None
        output.append(macro)
    return output


def cost_row(summary: dict[str, Any], decision_rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    method = str(summary["method"])
    n = len(decision_rows)
    panel_methods = {
        "uniform_paraphrase_vote", "uniform_likelihood_aggregation",
        "detector_only_abstention", "random_routing", "mosaic_conditional_routing",
    }
    detector_methods = {"detector_only_abstention", "random_routing", "mosaic_conditional_routing"}
    likelihood_methods = {"uniform_likelihood_aggregation", *detector_methods}
    teacher_forced = sum(
        int(row["panel_canonical_nll_count"]) + int(row["panel_candidate_nll_count"])
        for row in decision_rows
    )
    return {
        "cell": summary["cell"],
        "model": summary["model"],
        "dataset": summary["dataset"],
        "split": "locked_test",
        "method": method,
        "available": summary["available"],
        "n": n,
        "keep_count": summary.get("keep_count"),
        "recover_count": summary.get("recover_count"),
        "abstain_count": summary.get("abstain_count"),
        "high_risk_count": summary.get("high_risk_count"),
        "paraphrase_generation_queries": 4 * n if method in panel_methods else 0,
        "self_consistency_sampling_generations": k * n if method in detector_methods else 0,
        "canonical_hidden_forward_passes": n if method in detector_methods else 0,
        "teacher_forced_sequence_scores": teacher_forced if method in likelihood_methods else 0,
        "semantic_cluster_rows": n if method in panel_methods else 0,
        "risk_classifier_evaluations": n if method in detector_methods else 0,
        "additional_answer_generation_queries": 0,
        "actual_steering_generation_rows": None,
        "candidate_adoption_count": None,
        "wall_time_seconds": None,
        "peak_gpu_memory_mb": None,
        "cost_status": "protocol operation counts only; wall time and peak memory were not measured for frozen offline application",
    }


def evaluate_frozen(authorization_path: Path) -> dict[str, Any]:
    authorization = validate_authorization(authorization_path)
    decisions_path = Path(authorization["outputs"]["decisions"])
    freeze_path = Path(authorization["outputs"]["decision_freeze"])
    table_path = Path(authorization["outputs"]["table"])
    csv_path = Path(authorization["outputs"]["table_csv"])
    outcomes_path = Path(authorization["outputs"]["outcomes"])
    cost_path = Path(authorization["outputs"]["cost"])
    cost_csv_path = Path(authorization["outputs"]["cost_csv"])
    hash_path = Path(authorization["outputs"]["artifact_sha256"])
    for path in (table_path, csv_path, outcomes_path, cost_path, cost_csv_path, hash_path):
        if path.exists():
            raise FileExistsError(f"one-shot Run6 mitigation output exists: {path}")

    freeze = read_json(freeze_path)
    if freeze.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("invalid decision freeze schema")
    if freeze.get("decision_stage_is_label_blind") is not True:
        raise ValueError("decision freeze is not label-blind")
    verify_declared(decisions_path, str(freeze["decision_sha256"]), "frozen decisions")
    decisions = read_json(decisions_path)
    if decisions.get("schema_version") != DECISION_SCHEMA:
        raise ValueError("invalid decision schema")
    assert_label_blind(decisions)

    runtime_spec = authorization["runtime_benchmark"]
    runtime_path = Path(runtime_spec["path"])
    runtime_sha = verify_declared(runtime_path, str(runtime_spec["sha256"]), "runtime benchmark")
    runtime = read_json(runtime_path)
    summaries: list[dict[str, Any]] = []
    outcome_cells: dict[str, Any] = {}
    cost_rows: list[dict[str, Any]] = []

    for cell in CELLS:
        decision_rows = decisions["cells"][cell]["records"]
        runtime_rows = runtime["cells"][cell]["locked"]["records"]
        runtime_by_id = {str(row["sample_id"]): row for row in runtime_rows}
        if len(runtime_by_id) != len(runtime_rows):
            raise ValueError(f"{cell}: duplicate runtime IDs")
        if set(runtime_by_id) != {str(row["sample_id"]) for row in decision_rows}:
            raise ValueError(f"{cell}: decision/runtime IDs differ")
        outcome_cells[cell] = {}
        for method in POLICY_METHODS:
            summary, records = summarize_outcomes(cell, method, decision_rows, runtime_by_id)
            summaries.append(summary)
            cost_rows.append(cost_row(summary, decision_rows, int(authorization["self_consistency_k"])))
            outcome_cells[cell][method] = records
        unavailable = unavailable_steering(cell, len(decision_rows))
        summaries.append(unavailable)
        cost_rows.append(cost_row(unavailable, decision_rows, int(authorization["self_consistency_k"])))
        outcome_cells[cell]["uniform_steering"] = {
            "available": False,
            "reason": unavailable["availability_reason"],
        }

    summaries.extend(macro_rows(summaries))
    table_payload = {
        "schema_version": "one-confidence-mosaic-v1-run6-frozen-mitigation-table-v1",
        "locked_accessed": True,
        "decision_sha256": sha256_file(decisions_path),
        "runtime_sha256": runtime_sha,
        "selected_method": SELECTED_METHOD,
        "rows": summaries,
    }
    outcomes_payload = {
        "schema_version": OUTCOME_SCHEMA,
        "locked_accessed": True,
        "decision_sha256": sha256_file(decisions_path),
        "evaluation_only_after_decision_freeze": True,
        "cells": outcome_cells,
    }
    cost_payload = {
        "schema_version": "one-confidence-mosaic-v1-run6-frozen-mitigation-cost-v1",
        "cost_scope": "protocol operation counts only; no inferred wall time or memory",
        "rows": cost_rows,
    }
    atomic_json(table_path, table_payload)
    write_csv(csv_path, summaries, SUMMARY_FIELDS)
    atomic_json(outcomes_path, outcomes_payload)
    atomic_json(cost_path, cost_payload)
    write_csv(cost_csv_path, cost_rows, COST_FIELDS)
    output_hashes = {
        str(path): sha256_file(path)
        for path in (decisions_path, freeze_path, table_path, csv_path, outcomes_path, cost_path, cost_csv_path)
    }
    artifact_payload = {
        "schema_version": "one-confidence-mosaic-v1-run6-frozen-mitigation-artifacts-v1",
        "authorization_path": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "runtime_benchmark_sha256": runtime_sha,
        "outputs": output_hashes,
    }
    atomic_json(hash_path, artifact_payload)
    return table_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("decide", "evaluate"):
        subparser = subparsers.add_parser(stage)
        subparser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "decide":
        result = freeze_decisions(args.authorization)
    else:
        result = evaluate_frozen(args.authorization)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
