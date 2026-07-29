#!/usr/bin/env python3
"""Compile a gold-free public manifest for paraphrase-conditioned patching.

The compiler deliberately separates candidate construction from model execution.
Only development and validation records are accepted.  An audited four-element
question-equivalence mask is required by default because the generated rewrite
panel is not itself evidence that a paraphrase preserves the question target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_SPLITS = {"development", "validation"}
ALLOWED_REVIEW = {
    "codex_reviewed",
    "human_confirmed",
    "deterministic_rule_audited_v1",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def indexed(records: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in records:
        sample_id = str(row["sample_id"])
        if sample_id in result:
            raise ValueError(f"duplicate {name} sample_id: {sample_id}")
        result[sample_id] = row
    return result


def contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).casefold()
            if "gold" in lowered:
                return str(key)
            nested = contains_forbidden_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = contains_forbidden_key(item)
            if nested is not None:
                return nested
    return None


def load_audit(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("rewrite audit must contain a records list")
    audit = indexed(records, "rewrite-audit")
    for sample_id, row in audit.items():
        mask = row.get("equivalent_mask")
        if mask is None:
            natural_mask = row.get("natural_paraphrase_mask")
            if isinstance(natural_mask, list) and len(natural_mask) == 5:
                mask = natural_mask[1:]
                row["equivalent_mask"] = mask
        if (
            not isinstance(mask, list)
            or len(mask) != 4
            or not all(isinstance(value, bool) for value in mask)
        ):
            raise ValueError(f"{sample_id}: equivalent_mask must contain four booleans")
        if row.get("review_status") not in ALLOWED_REVIEW:
            raise ValueError(
                f"{sample_id}: unsupported question-equivalence audit status"
            )
    return audit


def choose_highest_confidence(indices: list[int], natural: dict[str, Any]) -> int:
    confidences = natural["panel_answer_confidences"]
    return min(indices, key=lambda index: (-float(confidences[index + 1]), index))


def assign_controls(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["label"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: row["sample_id"])
        if len(group) < 2:
            for row in group:
                row["unrelated_sample_id"] = None
                row["shuffled_candidate_sample_id"] = None
            continue
        for index, row in enumerate(group):
            unrelated = next(
                (
                    group[(index + offset) % len(group)]
                    for offset in range(1, len(group))
                    if group[(index + offset) % len(group)]["fact_key_sha256"]
                    != row["fact_key_sha256"]
                ),
                None,
            )
            row["unrelated_sample_id"] = (
                unrelated["sample_id"] if unrelated is not None else None
            )
            row["shuffled_candidate_sample_id"] = None
            if row["label"] != "HCE":
                continue
            shuffled = next(
                (
                    group[(index + offset) % len(group)]
                    for offset in range(1, len(group))
                    if normalize(
                        group[(index + offset) % len(group)]["candidate_answer"]
                    )
                    != normalize(row["candidate_answer"])
                ),
                None,
            )
            row["shuffled_candidate_sample_id"] = (
                shuffled["sample_id"] if shuffled is not None else None
            )


def compile_manifest(
    benchmark: dict[str, Any],
    natural_payload: dict[str, Any],
    nli_payload: dict[str, Any],
    cell: str,
    audit: dict[str, dict[str, Any]],
    allow_unverified: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if natural_payload.get("locked_accessed") or nli_payload.get("locked_accessed"):
        raise ValueError("public patch manifest cannot consume locked artifacts")
    if natural_payload.get("cell") != cell or nli_payload.get("cell") != cell:
        raise ValueError("cell mismatch across inputs")
    natural = indexed(natural_payload["records"], "natural-panel")
    nli = indexed(nli_payload["records"], "natural-panel-nli")
    benchmark_rows = {
        str(row["sample_id"]): row
        for split in ("development", "validation")
        for row in benchmark["cells"][cell][split]["records"]
    }
    if set(natural) != set(nli) or set(natural) != set(benchmark_rows):
        raise ValueError(
            "benchmark, natural-panel, and NLI sample IDs must match exactly"
        )

    counts: Counter[str] = Counter()
    output = []
    for sample_id in sorted(benchmark_rows):
        base = benchmark_rows[sample_id]
        natural_row = natural[sample_id]
        nli_row = nli[sample_id]
        split = str(base["split"])
        label = str(base["class"])
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"forbidden split in public manifest: {split}")
        if label not in {"HCE", "HCC"}:
            counts["skip_non_main_class"] += 1
            continue
        if natural_row["split"] != split or nli_row["split"] != split:
            raise ValueError(f"{sample_id}: split mismatch")
        if natural_row["label"] != label or nli_row["label"] != label:
            raise ValueError(f"{sample_id}: label mismatch")

        audit_row = audit.get(sample_id)
        if audit_row is None:
            if not allow_unverified:
                counts["skip_missing_question_equivalence_audit"] += 1
                continue
            equivalent_mask = [True] * 4
            equivalence_status = "unverified_diagnostic"
        else:
            equivalent_mask = [bool(value) for value in audit_row["equivalent_mask"]]
            equivalence_status = str(audit_row["review_status"])

        paraphrases = natural_row["paraphrases"]
        answers = natural_row["panel_answers"][1:]
        if len(paraphrases) != 4 or len(answers) != 4:
            raise ValueError(f"{sample_id}: expected four paraphrases and answers")

        candidate_answer = None
        nli_alternative_representative = None
        candidate_cluster = None
        support_indices: list[int] = []
        answer_equivalence_source = None
        if label == "HCE":
            nli_alternative_representative = str(
                nli_row.get("alternative_candidate") or ""
            ).strip()
            candidate_cluster = nli_row.get("alternative_candidate_cluster")
            if not nli_alternative_representative or candidate_cluster is None:
                counts["skip_hce_without_nli_candidate"] += 1
                continue
            if normalize(nli_alternative_representative) == normalize(
                natural_row["canonical_answer"]
            ):
                counts["skip_hce_candidate_equals_canonical"] += 1
                continue
            support_indices = [
                index
                for index, cluster_id in enumerate(nli_row["cluster_ids"])
                if cluster_id == candidate_cluster and equivalent_mask[index]
            ]
            if len(support_indices) < 2:
                counts["skip_hce_without_two_audited_supports"] += 1
                continue
            answer_equivalence_source = "nli_alternative_cluster"
        else:
            optional_mask = nli_row.get("canonical_equivalent_mask")
            if optional_mask is not None:
                if not isinstance(optional_mask, list) or len(optional_mask) != 4:
                    raise ValueError(f"{sample_id}: invalid canonical_equivalent_mask")
                canonical_mask = [bool(value) for value in optional_mask]
                answer_equivalence_source = "nli_canonical_equivalent_mask"
            else:
                canonical_key = normalize(natural_row["canonical_answer"])
                canonical_mask = [
                    normalize(answer) == canonical_key for answer in answers
                ]
                answer_equivalence_source = "normalized_exact_conservative"
            support_indices = [
                index
                for index in range(4)
                if equivalent_mask[index] and canonical_mask[index]
            ]
            if not support_indices:
                counts["skip_hcc_without_correct_to_correct_donor"] += 1
                continue

        donor_index = choose_highest_confidence(support_indices, natural_row)
        if label == "HCE":
            # The NLI representative can originate from an excluded rewrite.
            # Use an answer actually emitted by an audited equivalent donor.
            candidate_answer = str(answers[donor_index])
            if normalize(candidate_answer) == normalize(
                natural_row["canonical_answer"]
            ):
                counts["skip_hce_selected_candidate_equals_canonical"] += 1
                continue
        output.append(
            {
                "sample_id": sample_id,
                "split": split,
                "label": label,
                "fact_key_sha256": str(base["fact_key_sha256"]),
                "question": str(base["question"]),
                "canonical_answer": str(natural_row["canonical_answer"]),
                "donor_paraphrase": str(paraphrases[donor_index]),
                "donor_answer": str(answers[donor_index]),
                "donor_panel_index": donor_index,
                "candidate_answer": candidate_answer,
                "candidate_cluster": candidate_cluster,
                "nli_alternative_representative": nli_alternative_representative,
                "candidate_support_panel_indices": support_indices,
                "question_equivalence_status": equivalence_status,
                "answer_equivalence_source": answer_equivalence_source,
            }
        )
        counts[f"include_{split}_{label.lower()}"] += 1
    assign_controls(output)
    counts["included_total"] = len(output)
    return output, dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--natural-features", type=Path, required=True)
    parser.add_argument("--nli", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--rewrite-audit", type=Path)
    parser.add_argument("--allow-unverified-rewrites", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    natural = json.loads(args.natural_features.read_text(encoding="utf-8"))
    nli = json.loads(args.nli.read_text(encoding="utf-8"))
    audit = load_audit(args.rewrite_audit)
    rows, counts = compile_manifest(
        benchmark,
        natural,
        nli,
        args.cell,
        audit,
        args.allow_unverified_rewrites,
    )
    payload = {
        "schema_version": "one-confidence-mechanism-patch-manifest-v1",
        "cell": args.cell,
        "locked_accessed": False,
        "candidate_protocol": "largest NLI alternative cluster with at least two question-equivalent paraphrase supports",
        "hcc_control_protocol": "question-equivalent paraphrase whose answer is equivalent to the canonical answer",
        "question_equivalence": (
            "audited masks"
            if args.rewrite_audit is not None
            else "unverified diagnostic override"
        ),
        "source_sha256": {
            "benchmark": sha256(args.benchmark),
            "natural_features": sha256(args.natural_features),
            "nli": sha256(args.nli),
            "rewrite_audit": sha256(args.rewrite_audit) if args.rewrite_audit else None,
        },
        "selection_counts": counts,
        "records": rows,
    }
    forbidden = contains_forbidden_key(payload)
    if forbidden is not None:
        raise AssertionError(f"forbidden key leaked into patch manifest: {forbidden}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cell": args.cell, **counts}, indent=2))


if __name__ == "__main__":
    main()
