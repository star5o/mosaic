#!/usr/bin/env python3
"""Evaluate public-only mitigation policies from frozen One Confidence artifacts.

Routing functions receive no labels or answer keys.  Benchmark labels and gold
aliases are attached only after a decision has been made, for outcome scoring.
The evaluator has no locked mode and refuses paths containing ``locked`` or
``invalidated``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_CELLS = (
    "mistral7|TriviaQA",
    "qwen8|TriviaQA",
    "llama31|TriviaQA",
    "qwen14|PopQA",
    "nemo12|PopQA",
)
METHODS = (
    "keep",
    "uniform_paraphrase_vote",
    "uniform_likelihood_aggregation",
    "detector_only_abstention",
    "random_routing",
    "mosaic_conditional_routing",
    "uniform_steering",
)
FORBIDDEN_PATH_TOKENS = ("locked", "invalidated")
SUMMARY_FIELDS = [
    "aggregation",
    "cell",
    "model",
    "dataset",
    "split",
    "method",
    "available",
    "availability_reason",
    "metric_aggregation",
    "macro_cells",
    "n",
    "hce",
    "hcc",
    "keep_count",
    "recover_count",
    "abstain_count",
    "high_risk_count",
    "hce_corrected",
    "hcc_damaged",
    "net_correction",
    "answered",
    "coverage",
    "correct_answered",
    "selective_accuracy",
    "trigger_tp",
    "trigger_fn",
    "trigger_fp",
    "trigger_tn",
    "recall",
    "fpr",
    "risk_threshold",
    "candidate_adoption_count",
    "actual_generation_rows",
]
COST_FIELDS = [
    "cell",
    "model",
    "dataset",
    "split",
    "method",
    "available",
    "n",
    "keep_count",
    "recover_count",
    "abstain_count",
    "high_risk_count",
    "paraphrase_generation_queries",
    "self_consistency_sampling_generations",
    "canonical_hidden_forward_passes",
    "teacher_forced_sequence_scores",
    "semantic_cluster_rows",
    "risk_classifier_evaluations",
    "additional_answer_generation_queries",
    "actual_steering_generation_rows",
    "candidate_adoption_count",
    "wall_time_seconds",
    "peak_gpu_memory_mb",
    "cost_status",
]


@dataclass(frozen=True)
class PolicyInput:
    sample_id: str
    split: str
    canonical_answer: str
    safe_panel_answers: tuple[str, ...]
    safe_panel_valid_mask: tuple[bool, ...]
    cluster_ids: tuple[int, ...]
    cluster_representatives: tuple[str, ...]
    canonical_equivalent_mask: tuple[bool, ...]
    natural_candidate: str
    natural_support_count: int
    candidate_answer: str
    candidate_available: bool
    panel_canonical_nlls: tuple[float, ...]
    panel_candidate_nlls: tuple[float, ...]
    risk_score: float
    risk_threshold: float

    @property
    def high_risk(self) -> bool:
        return self.risk_score >= self.risk_threshold

    @property
    def reliable_candidate(self) -> bool:
        return bool(self.natural_candidate) and self.natural_support_count >= 3


@dataclass(frozen=True)
class EvaluationTarget:
    label: str
    gold_aliases: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    action: str
    answer: str | None
    triggered: bool
    reason: str


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_public_path(path: Path) -> None:
    lowered = str(path.resolve(strict=False)).casefold()
    if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"public mitigation refuses forbidden path: {path}")


def read_public_json(path: Path, allow_benchmark: bool = False) -> dict[str, Any]:
    assert_public_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("locked_accessed") or payload.get("protocol", {}).get(
        "locked_accessed"
    ):
        raise ValueError(f"artifact declares locked access: {path}")
    if not allow_benchmark and payload.get("gold_visible") is True:
        raise ValueError(f"feature artifact exposes answer keys: {path}")
    return payload


def split_cell(cell: str) -> tuple[str, str]:
    if "|" not in cell:
        raise ValueError(f"invalid cell: {cell}")
    model, dataset = cell.split("|", 1)
    return model, dataset


def cell_stem(cell: str) -> str:
    return cell.replace("|", "-").casefold()


def by_sample_id(payload: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: records list missing")
    output = {str(row["sample_id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"{path}: duplicate sample IDs")
    return output


def decision_from_answer(answer: str, canonical: str, reason: str) -> Decision:
    if not answer or normalize(answer) == normalize(canonical):
        return Decision("KEEP", canonical, False, reason)
    return Decision("RECOVER", answer, True, reason)


def keep_policy(row: PolicyInput) -> Decision:
    return Decision("KEEP", row.canonical_answer, False, "keep_baseline")


def uniform_vote_policy(row: PolicyInput) -> Decision:
    if not (
        len(row.safe_panel_answers)
        == len(row.safe_panel_valid_mask)
        == len(row.cluster_ids)
        == len(row.canonical_equivalent_mask)
    ):
        raise ValueError(f"{row.sample_id}: malformed semantic panel")
    counts: dict[int, int] = {}
    canonical_clusters = set()
    representatives: dict[int, str] = {}
    for answer, valid, cluster, equivalent in zip(
        row.safe_panel_answers,
        row.safe_panel_valid_mask,
        row.cluster_ids,
        row.canonical_equivalent_mask,
    ):
        if not valid:
            continue
        counts[cluster] = counts.get(cluster, 0) + 1
        representatives.setdefault(cluster, answer)
        if equivalent:
            canonical_clusters.add(cluster)
    if canonical_clusters:
        canonical_cluster = max(
            canonical_clusters, key=lambda value: (counts.get(value, 0), -value)
        )
    else:
        canonical_cluster = -1
    counts[canonical_cluster] = counts.get(canonical_cluster, 0) + 1
    if not counts:
        return keep_policy(row)
    maximum = max(counts.values())
    winners = {cluster for cluster, count in counts.items() if count == maximum}
    if canonical_cluster in winners:
        return Decision(
            "KEEP", row.canonical_answer, False, "uniform_vote_canonical_tie_or_win"
        )
    winner = min(winners)
    if 0 <= winner < len(row.cluster_representatives):
        answer = row.cluster_representatives[winner]
    else:
        answer = representatives.get(winner, "")
    return decision_from_answer(answer, row.canonical_answer, "uniform_vote_alternative")


def finite_mean(values: tuple[float, ...]) -> float | None:
    if not values or not all(math.isfinite(value) for value in values):
        return None
    return sum(values) / len(values)


def uniform_likelihood_policy(row: PolicyInput) -> Decision:
    if not row.candidate_available or not row.candidate_answer:
        return Decision(
            "KEEP", row.canonical_answer, False, "likelihood_candidate_unavailable"
        )
    canonical_nll = finite_mean(row.panel_canonical_nlls)
    candidate_nll = finite_mean(row.panel_candidate_nlls)
    if canonical_nll is None or candidate_nll is None:
        return Decision(
            "KEEP", row.canonical_answer, False, "likelihood_panel_incomplete"
        )
    if candidate_nll < canonical_nll:
        return decision_from_answer(
            row.candidate_answer,
            row.canonical_answer,
            "uniform_likelihood_candidate_lower_mean_nll",
        )
    return Decision(
        "KEEP", row.canonical_answer, False, "uniform_likelihood_canonical_lower_mean_nll"
    )


def detector_abstention_policy(row: PolicyInput) -> Decision:
    if row.high_risk:
        return Decision("ABSTAIN", None, True, "nested_detector_high_risk")
    return Decision("KEEP", row.canonical_answer, False, "nested_detector_low_risk")


def conditional_policy(row: PolicyInput) -> Decision:
    if not row.high_risk:
        return Decision("KEEP", row.canonical_answer, False, "mosaic_low_risk")
    if row.reliable_candidate:
        decision = decision_from_answer(
            row.natural_candidate,
            row.canonical_answer,
            "mosaic_high_risk_natural_support_at_least_3",
        )
        return Decision(decision.action, decision.answer, True, decision.reason)
    return Decision(
        "ABSTAIN", None, True, "mosaic_high_risk_without_reliable_candidate"
    )


def random_recovery_ids(
    rows: list[PolicyInput], cell: str, split: str, seed: int
) -> set[str]:
    high_risk = [row for row in rows if row.high_risk]
    required = sum(row.reliable_candidate for row in high_risk)
    available = [
        row
        for row in high_risk
        if row.candidate_available and bool(row.candidate_answer)
    ]
    if required > len(available):
        raise ValueError("random routing cannot preserve conditional recovery count")
    rng = random.Random(
        int.from_bytes(
            hashlib.sha256(f"{seed}|{cell}|{split}".encode("utf-8")).digest()[:8],
            "big",
        )
    )
    return {row.sample_id for row in rng.sample(available, required)}


def random_routing_policy(row: PolicyInput, recovery_ids: set[str]) -> Decision:
    if not row.high_risk:
        return Decision("KEEP", row.canonical_answer, False, "random_route_low_risk")
    if row.sample_id in recovery_ids:
        decision = decision_from_answer(
            row.candidate_answer,
            row.canonical_answer,
            "random_route_recover_count_matched",
        )
        return Decision(decision.action, decision.answer, True, decision.reason)
    return Decision("ABSTAIN", None, True, "random_route_abstain")


def alias_match(answer: str, aliases: tuple[str, ...]) -> bool:
    normalized = normalize(answer)
    return bool(normalized) and normalized in {normalize(alias) for alias in aliases}


def outcome_correct(
    decision: Decision, row: PolicyInput, target: EvaluationTarget
) -> bool:
    if decision.answer is None:
        return False
    if normalize(decision.answer) == normalize(row.canonical_answer):
        return target.label == "HCC"
    return alias_match(decision.answer, target.gold_aliases)


def evaluate_decisions(
    rows: list[PolicyInput],
    targets: dict[str, EvaluationTarget],
    decisions: dict[str, Decision],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = []
    for row in rows:
        target = targets[row.sample_id]
        decision = decisions[row.sample_id]
        correct = outcome_correct(decision, row, target)
        records.append(
            {
                "sample_id": row.sample_id,
                "split": row.split,
                "label": target.label,
                "action": decision.action,
                "triggered": decision.triggered,
                "reason": decision.reason,
                "canonical_answer": row.canonical_answer,
                "output_answer": decision.answer,
                "output_correct": correct,
                "risk_score": row.risk_score,
                "risk_threshold": row.risk_threshold,
                "natural_support_count": row.natural_support_count,
                "reliable_candidate": row.reliable_candidate,
            }
        )
    hce = [row for row in records if row["label"] == "HCE"]
    hcc = [row for row in records if row["label"] == "HCC"]
    answered = [row for row in records if row["output_answer"] is not None]
    correct_answered = sum(bool(row["output_correct"]) for row in answered)
    hce_corrected = sum(bool(row["output_correct"]) for row in hce)
    hcc_damaged = sum(not bool(row["output_correct"]) for row in hcc)
    tp = sum(bool(row["triggered"]) for row in hce)
    fp = sum(bool(row["triggered"]) for row in hcc)
    summary = {
        "n": len(records),
        "hce": len(hce),
        "hcc": len(hcc),
        "keep_count": sum(row["action"] == "KEEP" for row in records),
        "recover_count": sum(row["action"] == "RECOVER" for row in records),
        "abstain_count": sum(row["action"] == "ABSTAIN" for row in records),
        "high_risk_count": sum(bool(row["triggered"]) for row in records),
        "hce_corrected": hce_corrected,
        "hcc_damaged": hcc_damaged,
        "net_correction": hce_corrected - hcc_damaged,
        "answered": len(answered),
        "coverage": len(answered) / len(records) if records else None,
        "correct_answered": correct_answered,
        "selective_accuracy": correct_answered / len(answered) if answered else None,
        "trigger_tp": tp,
        "trigger_fn": len(hce) - tp,
        "trigger_fp": fp,
        "trigger_tn": len(hcc) - fp,
        "recall": tp / len(hce) if hce else None,
        "fpr": fp / len(hcc) if hcc else None,
    }
    return summary, records


def nested_scores_for_split(
    payload: dict[str, Any], method: str, split: str
) -> tuple[float, dict[str, float]]:
    methods = payload.get("methods", {})
    if method not in methods:
        raise ValueError(f"nested score file omits method {method}")
    method_payload = methods[method]
    summary_key = "development_oof" if split == "development" else "validation"
    threshold = float(method_payload[summary_key]["threshold"])
    scores = {
        str(row["sample_id"]): float(row["score"])
        for row in method_payload["scores"]
        if row["split"] == split
    }
    return threshold, scores


def build_policy_rows(
    benchmark_rows: list[dict[str, Any]],
    panel_by_id: dict[str, dict[str, Any]],
    nli_by_id: dict[str, dict[str, Any]],
    margin_by_id: dict[str, dict[str, Any]],
    nested_scores: dict[str, float],
    threshold: float,
) -> tuple[list[PolicyInput], dict[str, EvaluationTarget]]:
    output = []
    targets = {}
    for benchmark_row in benchmark_rows:
        sample_id = str(benchmark_row["sample_id"])
        if sample_id not in panel_by_id or sample_id not in nli_by_id or sample_id not in margin_by_id:
            raise ValueError(f"{sample_id}: feature artifacts are incomplete")
        if sample_id not in nested_scores:
            raise ValueError(f"{sample_id}: nested score missing")
        panel = panel_by_id[sample_id]
        nli = nli_by_id[sample_id]
        margin = margin_by_id[sample_id]
        if str(panel["split"]) != benchmark_row["split"]:
            raise ValueError(f"{sample_id}: split mismatch")
        safe_answers = tuple(str(value) for value in panel["panel_answers"][1:])
        row = PolicyInput(
            sample_id=sample_id,
            split=str(benchmark_row["split"]),
            canonical_answer=str(benchmark_row["generated_answer"]),
            safe_panel_answers=safe_answers,
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
            risk_score=float(nested_scores[sample_id]),
            risk_threshold=threshold,
        )
        output.append(row)
        targets[sample_id] = EvaluationTarget(
            label=str(benchmark_row["class"]),
            gold_aliases=tuple(str(value) for value in benchmark_row["gold_aliases"]),
        )
    if set(nested_scores) != {row.sample_id for row in output}:
        raise ValueError("nested score IDs differ from the benchmark split")
    return output, targets


def summarize_policy(
    cell: str,
    split: str,
    method: str,
    summary: dict[str, Any],
    threshold: float | None,
) -> dict[str, Any]:
    model, dataset = split_cell(cell)
    return {
        "aggregation": "cell",
        "cell": cell,
        "model": model,
        "dataset": dataset,
        "split": split,
        "method": method,
        "available": True,
        "availability_reason": "evaluated",
        "metric_aggregation": "cell",
        "macro_cells": None,
        **summary,
        "risk_threshold": threshold,
        "candidate_adoption_count": None,
        "actual_generation_rows": None,
    }


def unavailable_steering_row(
    cell: str,
    split: str,
    generation_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    model, dataset = split_cell(cell)
    generations = [] if generation_payload is None else generation_payload.get("validation_generations", [])
    split_generations = [row for row in generations if row.get("split") == split]
    adoption = sum(
        bool(row.get("candidate_exact_match", {}).get("aligned_selected_layer"))
        for row in split_generations
    )
    if not split_generations:
        reason = "no_actual_patch_generations_for_split"
    elif adoption == 0:
        reason = "actual_patch_generation_candidate_adoption_zero"
    else:
        reason = "partial_mechanism_subset_not_full_public_cohort"
    return {
        "aggregation": "cell",
        "cell": cell,
        "model": model,
        "dataset": dataset,
        "split": split,
        "method": "uniform_steering",
        "available": False,
        "availability_reason": reason,
        "metric_aggregation": "unavailable",
        "macro_cells": None,
        **{field: None for field in SUMMARY_FIELDS[10:30]},
        "risk_threshold": None,
        "candidate_adoption_count": adoption,
        "actual_generation_rows": len(split_generations),
    }


def macro_rows(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    metric_fields = (
        "coverage",
        "selective_accuracy",
        "recall",
        "fpr",
    )
    count_fields = (
        "n",
        "hce",
        "hcc",
        "keep_count",
        "recover_count",
        "abstain_count",
        "high_risk_count",
        "hce_corrected",
        "hcc_damaged",
        "net_correction",
        "answered",
        "correct_answered",
        "trigger_tp",
        "trigger_fn",
        "trigger_fp",
        "trigger_tn",
        "candidate_adoption_count",
        "actual_generation_rows",
    )
    for split in ("development", "validation"):
        for method in METHODS:
            rows = [
                row
                for row in cell_rows
                if row["split"] == split and row["method"] == method
            ]
            if not rows:
                continue
            available = all(bool(row["available"]) for row in rows)
            macro = {
                "aggregation": "macro",
                "cell": f"macro_{len(rows)}_cells",
                "model": "macro",
                "dataset": "mixed",
                "split": split,
                "method": method,
                "available": available,
                "availability_reason": "all_cells_evaluated" if available else "one_or_more_cells_unavailable",
                "metric_aggregation": "equal_cell_mean_for_rates; pooled_sum_for_counts",
                "macro_cells": len(rows),
                "risk_threshold": None,
            }
            for field in count_fields:
                values = [row[field] for row in rows if row.get(field) is not None]
                macro[field] = sum(values) if values else None
            for field in metric_fields:
                values = [float(row[field]) for row in rows if row.get(field) is not None]
                macro[field] = sum(values) / len(values) if available and values else None
            output.append(macro)
    return output


def cost_action_row(
    summary: dict[str, Any],
    rows: list[PolicyInput],
    self_consistency_k: int,
) -> dict[str, Any]:
    method = str(summary["method"])
    n = len(rows)
    panel_methods = {
        "uniform_paraphrase_vote",
        "uniform_likelihood_aggregation",
        "detector_only_abstention",
        "random_routing",
        "mosaic_conditional_routing",
    }
    detector_methods = {
        "detector_only_abstention",
        "random_routing",
        "mosaic_conditional_routing",
    }
    likelihood_methods = {
        "uniform_likelihood_aggregation",
        *detector_methods,
    }
    teacher_forced = sum(
        len(row.panel_canonical_nlls) + len(row.panel_candidate_nlls)
        for row in rows
    )
    return {
        "cell": summary["cell"],
        "model": summary["model"],
        "dataset": summary["dataset"],
        "split": summary["split"],
        "method": method,
        "available": summary["available"],
        "n": summary.get("n"),
        "keep_count": summary.get("keep_count"),
        "recover_count": summary.get("recover_count"),
        "abstain_count": summary.get("abstain_count"),
        "high_risk_count": summary.get("high_risk_count"),
        "paraphrase_generation_queries": 4 * n if method in panel_methods else 0,
        "self_consistency_sampling_generations": self_consistency_k * n
        if method in detector_methods
        else 0,
        "canonical_hidden_forward_passes": n if method in detector_methods else 0,
        "teacher_forced_sequence_scores": teacher_forced
        if method in likelihood_methods
        else 0,
        "semantic_cluster_rows": n if method in panel_methods else 0,
        "risk_classifier_evaluations": n if method in detector_methods else 0,
        "additional_answer_generation_queries": 0,
        "actual_steering_generation_rows": summary.get("actual_generation_rows"),
        "candidate_adoption_count": summary.get("candidate_adoption_count"),
        "wall_time_seconds": None,
        "peak_gpu_memory_mb": None,
        "cost_status": "protocol_operation_counts; wall time and peak memory were not measured in this offline evaluator",
    }


def parse_overrides(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError("nested-score overrides must use cell=path")
        cell, raw_path = value.split("=", 1)
        output[cell] = Path(raw_path)
    return output


def resolve_nested_score(
    root: Path, cell: str, overrides: dict[str, Path]
) -> Path:
    if cell in overrides:
        return overrides[cell]
    stem = cell_stem(cell)
    candidates = (
        root / stem / "public_scores.json",
        root / f"{stem}-public-scores.json",
        root / f"{stem}-nested-public-scores.json",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no nested public score file found for {cell} under {root}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_rows_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--nested-score-root", type=Path, required=True)
    parser.add_argument("--nested-score", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--risk-method", default="mosaic_v1_full")
    parser.add_argument("--seed", type=int, default=2026072905)
    parser.add_argument("--self-consistency-k", type=int, default=4)
    parser.add_argument("--cell", action="append", default=[])
    args = parser.parse_args()
    if args.self_consistency_k < 0:
        raise ValueError("self-consistency-k must be non-negative")
    for path in (args.benchmark, args.artifact_dir, args.nested_score_root, args.output_dir):
        assert_public_path(path)
    cells = tuple(args.cell) if args.cell else DEFAULT_CELLS
    benchmark = read_public_json(args.benchmark, allow_benchmark=True)
    overrides = parse_overrides(args.nested_score)
    summaries = []
    cost_rows = []
    all_decisions: dict[str, Any] = {}
    source_hashes = {"benchmark": sha256(args.benchmark)}

    for cell in cells:
        if cell not in benchmark.get("cells", {}):
            raise ValueError(f"benchmark omits cell {cell}")
        stem = cell_stem(cell)
        paths = {
            "panel": args.artifact_dir / f"{stem}-public-corrected-panel-features.json",
            "nli": args.artifact_dir / f"{stem}-public-corrected-panel-nli.json",
            "margin": args.artifact_dir / f"{stem}-public-nli-candidate-margins.json",
            "nested": resolve_nested_score(args.nested_score_root, cell, overrides),
        }
        payloads = {
            name: read_public_json(path) for name, path in paths.items()
        }
        for name, payload in payloads.items():
            if payload.get("cell") != cell:
                raise ValueError(f"{paths[name]}: cell mismatch")
            source_hashes[f"{cell}:{name}"] = sha256(paths[name])
        panel_by_id = by_sample_id(payloads["panel"], paths["panel"])
        nli_by_id = by_sample_id(payloads["nli"], paths["nli"])
        margin_by_id = by_sample_id(payloads["margin"], paths["margin"])
        steering_path = args.artifact_dir / f"{stem}-public-mechanism-patching.json"
        steering_payload = None
        if steering_path.exists():
            steering_payload = read_public_json(steering_path)
            source_hashes[f"{cell}:steering"] = sha256(steering_path)
        all_decisions[cell] = {}

        for split in ("development", "validation"):
            threshold, scores = nested_scores_for_split(
                payloads["nested"], args.risk_method, split
            )
            benchmark_rows = benchmark["cells"][cell][split]["records"]
            rows, targets = build_policy_rows(
                benchmark_rows,
                panel_by_id,
                nli_by_id,
                margin_by_id,
                scores,
                threshold,
            )
            recovery_ids = random_recovery_ids(rows, cell, split, args.seed)
            policies: dict[str, Callable[[PolicyInput], Decision]] = {
                "keep": keep_policy,
                "uniform_paraphrase_vote": uniform_vote_policy,
                "uniform_likelihood_aggregation": uniform_likelihood_policy,
                "detector_only_abstention": detector_abstention_policy,
                "random_routing": lambda row, ids=recovery_ids: random_routing_policy(row, ids),
                "mosaic_conditional_routing": conditional_policy,
            }
            split_decisions = {}
            for method, policy in policies.items():
                decisions = {row.sample_id: policy(row) for row in rows}
                summary, decision_records = evaluate_decisions(rows, targets, decisions)
                risk_threshold = threshold if method in {
                    "detector_only_abstention",
                    "random_routing",
                    "mosaic_conditional_routing",
                } else None
                summaries.append(
                    summarize_policy(cell, split, method, summary, risk_threshold)
                )
                cost_rows.append(
                    cost_action_row(
                        summaries[-1], rows, args.self_consistency_k
                    )
                )
                split_decisions[method] = decision_records
            summaries.append(
                unavailable_steering_row(cell, split, steering_payload)
            )
            cost_rows.append(
                cost_action_row(summaries[-1], rows, args.self_consistency_k)
            )
            split_decisions["uniform_steering"] = {
                "available": False,
                "reason": summaries[-1]["availability_reason"],
                "candidate_adoption_count": summaries[-1]["candidate_adoption_count"],
                "actual_generation_rows": summaries[-1]["actual_generation_rows"],
            }
            all_decisions[cell][split] = split_decisions

    summaries.extend(macro_rows(summaries))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_json = {
        "schema_version": "one-confidence-public-mitigation-table-v1",
        "public_only": True,
        "locked_accessed": False,
        "risk_method": args.risk_method,
        "routing_protocol": {
            "recover": "natural NLI alternative support count >= 3",
            "conditional": "low risk KEEP; high risk with reliable candidate RECOVER; other high risk ABSTAIN",
            "random": "same high-risk set and same RECOVER count as conditional, with RECOVER IDs sampled from high-risk candidate-available rows",
            "uniform_vote": "canonical plus four safe-panel semantic-cluster votes; canonical wins ties",
            "uniform_likelihood": "candidate replaces canonical only when its mean NLL across the five frozen prompts is lower",
            "trigger_metrics": "Recall/FPR use non-KEEP actions for uniform policies and the nested high-risk predicate for detector-routed policies",
            "evaluation_only": "labels and gold aliases are attached only after policy decisions",
        },
        "rows": summaries,
    }
    decisions_json = {
        "schema_version": "one-confidence-public-mitigation-decisions-v1",
        "public_only": True,
        "locked_accessed": False,
        "cells": all_decisions,
    }
    table_path = args.output_dir / "table_mitigation.json"
    csv_path = args.output_dir / "table_mitigation.csv"
    decisions_path = args.output_dir / "mitigation_public_decisions.json"
    validation_rows = [row for row in summaries if row["split"] == "validation"]
    development_rows = [row for row in summaries if row["split"] == "development"]
    validation_json_path = args.output_dir / "table_mitigation_public_validation.json"
    validation_csv_path = args.output_dir / "table_mitigation_public_validation.csv"
    development_json_path = args.output_dir / "table_mitigation_public_development.json"
    development_csv_path = args.output_dir / "table_mitigation_public_development.csv"
    validation_cost_rows = [row for row in cost_rows if row["split"] == "validation"]
    cost_json_path = args.output_dir / "table_mitigation_cost_actions_public_validation.json"
    cost_csv_path = args.output_dir / "table_mitigation_cost_actions_public_validation.csv"
    table_path.write_text(
        json.dumps(table_json, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    decisions_path.write_text(
        json.dumps(decisions_json, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, summaries)
    validation_json_path.write_text(
        json.dumps(
            {
                "schema_version": "one-confidence-public-mitigation-validation-v1",
                "risk_method": args.risk_method,
                "public_only": True,
                "locked_accessed": False,
                "rows": validation_rows,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_rows_csv(validation_csv_path, SUMMARY_FIELDS, validation_rows)
    development_json_path.write_text(
        json.dumps(
            {
                "schema_version": "one-confidence-public-mitigation-development-v1",
                "risk_method": args.risk_method,
                "public_only": True,
                "locked_accessed": False,
                "rows": development_rows,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_rows_csv(development_csv_path, SUMMARY_FIELDS, development_rows)
    cost_json_path.write_text(
        json.dumps(
            {
                "schema_version": "one-confidence-public-mitigation-cost-actions-v1",
                "cost_scope": "protocol operation counts only; no inferred wall time or memory",
                "self_consistency_k": args.self_consistency_k,
                "rows": validation_cost_rows,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    write_rows_csv(cost_csv_path, COST_FIELDS, validation_cost_rows)
    artifact_hashes = {
        "sources": source_hashes,
        "outputs": {
            table_path.name: sha256(table_path),
            csv_path.name: sha256(csv_path),
            decisions_path.name: sha256(decisions_path),
            validation_json_path.name: sha256(validation_json_path),
            validation_csv_path.name: sha256(validation_csv_path),
            development_json_path.name: sha256(development_json_path),
            development_csv_path.name: sha256(development_csv_path),
            cost_json_path.name: sha256(cost_json_path),
            cost_csv_path.name: sha256(cost_csv_path),
        },
    }
    hash_path = args.output_dir / "mitigation_artifact_sha256.json"
    hash_path.write_text(
        json.dumps(artifact_hashes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cells": list(cells),
                "summary_rows": len(summaries),
                "risk_method": args.risk_method,
                "outputs": artifact_hashes["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
