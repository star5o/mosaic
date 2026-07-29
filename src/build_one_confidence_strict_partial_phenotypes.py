#!/usr/bin/env python3
"""Build strict public-only partial phenotype labels for MOSAIC-v1.

Behavioral labels come only from corrected natural paraphrase support and a
non-negative candidate margin.  Internal and persistent labels require an
optional row-level causal artifact.  Without causal assessment, nonbehavioral
HCE rows remain ambiguous rather than being treated as persistent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_payload(path: Path) -> dict:
    lowered = str(path.resolve(strict=False)).casefold()
    if "locked" in lowered or "invalidated" in lowered:
        raise ValueError(f"strict phenotype builder refuses forbidden path: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("locked_accessed"):
        raise ValueError(f"artifact declares locked access: {path}")
    return payload


def keyed(payload: dict, source: Path) -> dict[str, dict]:
    rows = payload.get("records", [])
    output = {row["sample_id"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate sample ID in {source}")
    return output


def causal_evidence(row: dict | None) -> dict:
    if row is None:
        return {
            "causal_assessed": False,
            "causal_internal_support": False,
            "true_effect": 0.0,
            "random_control_effect": 0.0,
            "unrelated_control_effect": 0.0,
            "shuffled_control_effect": 0.0,
            "reverse_effect": 0.0,
            "validation_confirmed": False,
        }
    required = (
        "causal_assessed",
        "causal_internal_support",
        "true_effect",
        "random_control_effect",
        "unrelated_control_effect",
        "shuffled_control_effect",
        "reverse_effect",
        "validation_confirmed",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"causal phenotype row lacks fields: {missing}")
    return {name: row[name] for name in required}


def classify(label: str, evidence: dict) -> str:
    if label != "HCE":
        return "unlabeled"
    behavioral = (
        evidence["natural_support_count"] >= 2
        and evidence["support_margin"] >= 0.0
        and evidence["reliable_natural_candidate"]
    )
    if behavioral:
        return "behavioral"
    causal = (
        evidence["causal_assessed"]
        and evidence["causal_internal_support"]
        and evidence["validation_confirmed"]
        and evidence["true_effect"]
        > max(
            evidence["random_control_effect"],
            evidence["unrelated_control_effect"],
            evidence["shuffled_control_effect"],
        )
        and evidence["reverse_effect"] < 0.0
    )
    if causal:
        return "internal"
    if (
        not evidence["reliable_natural_candidate"]
        and evidence["causal_assessed"]
        and not evidence["causal_internal_support"]
    ):
        return "persistent"
    return "ambiguous"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nli", type=Path, required=True)
    parser.add_argument("--candidate-margins", type=Path, required=True)
    parser.add_argument("--causal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nli_payload = public_payload(args.nli)
    candidate_payload = public_payload(args.candidate_margins)
    causal_payload = public_payload(args.causal) if args.causal else None
    cells = {nli_payload.get("cell"), candidate_payload.get("cell")}
    if causal_payload is not None:
        cells.add(causal_payload.get("cell"))
    if len(cells) != 1:
        raise ValueError(f"phenotype inputs belong to different cells: {cells}")
    nli = keyed(nli_payload, args.nli)
    candidate = keyed(candidate_payload, args.candidate_margins)
    causal = keyed(causal_payload, args.causal) if causal_payload is not None else {}
    if set(nli) != set(candidate):
        raise ValueError("corrected NLI and candidate-margin row IDs differ")
    if causal and not set(causal).issubset(set(nli)):
        raise ValueError("causal artifact contains rows outside the public cohort")

    records = []
    counts: dict[str, int] = {}
    for sample_id, nli_row in nli.items():
        candidate_row = candidate[sample_id]
        if nli_row.get("split") != candidate_row.get("split") or nli_row.get("label") != candidate_row.get("label"):
            raise ValueError(f"split/label mismatch for {sample_id}")
        natural_support = int(nli_row["alternative_support_count_natural"])
        support_margin = float(candidate_row["support_prompt_candidate_logscore_margin_mean"])
        reliable_natural = bool(
            candidate_row["candidate_available"]
            and natural_support >= 2
            and support_margin >= 0.0
            and candidate_row["behavioral_recoverable_margin_pass"]
        )
        evidence = {
            "natural_support_count": natural_support,
            "support_margin": support_margin,
            "reliable_natural_candidate": reliable_natural,
            **causal_evidence(causal.get(sample_id)),
        }
        phenotype = classify(str(nli_row["label"]), evidence)
        counts[phenotype] = counts.get(phenotype, 0) + 1
        records.append(
            {
                "sample_id": sample_id,
                "split": nli_row["split"],
                "label": nli_row["label"],
                "fact_key_sha256": nli_row.get("fact_key_sha256"),
                "phenotype": phenotype,
                "evidence": evidence,
            }
        )

    output = {
        "schema_version": "one-confidence-strict-partial-phenotypes-v1",
        "cell": next(iter(cells)),
        "locked_accessed": False,
        "hcc_phenotype_loss": "excluded",
        "rules": {
            "behavioral": "corrected natural support >=2 and support margin >=0",
            "internal": "causal assessed and validation-confirmed true effect exceeds random/unrelated/shuffled, reverse<0",
            "persistent": "no reliable natural candidate and causal assessment finds no internal support",
            "ambiguous": "all other HCE rows",
            "forbidden_proxy": "NLL<=1 is not used",
        },
        "source_sha256": {
            str(args.nli): sha256(args.nli),
            str(args.candidate_margins): sha256(args.candidate_margins),
            **({str(args.causal): sha256(args.causal)} if args.causal else {}),
        },
        "counts": counts,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
