#!/usr/bin/env python3
"""Apply a frozen MOSAIC-v1 checkpoint without fitting or threshold search.

The default public mode rejects locked/invalidated paths.  A future frozen-test
application must provide a separately hashed authorization manifest before any
cohort artifact is opened.  This implementation only transforms inputs with
already-fitted preprocessors/probes/experts/gates and calls predict functions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import torch

from evaluate_one_confidence_mosaic_v1_nested_public import (
    METHOD_NAMES,
    POSITIONS,
    CellData,
    atomic_json,
    dotted_value,
    numeric_value,
    predict_stack,
    sha256_file,
    validate_feature_field,
)


APPLICATION_SCHEMA = "one-confidence-mosaic-v1-frozen-application-v1"
OUTPUT_SCHEMA = "one-confidence-mosaic-v1-frozen-scores-v1"
FREEZE_SCHEMA = "one-confidence-mosaic-v1-evaluation-freeze-v1"
FORBIDDEN_PATH_TOKENS = ("locked", "invalidated")


def _contains_forbidden_path_token(path: Path) -> bool:
    candidates = (str(path), str(path.resolve(strict=False)))
    return any(
        token in candidate.casefold()
        for candidate in candidates
        for token in FORBIDDEN_PATH_TOKENS
    )


def _read_json_unchecked(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_application_authorization(
    application_path: Path,
    application: dict,
    checkpoint_path: Path,
    thresholds_path: Path,
    freeze_manifest_path: Path | None,
) -> dict | None:
    cohort_kind = application.get("cohort_kind")
    if cohort_kind == "public":
        paths = [application_path, checkpoint_path, thresholds_path]
        paths.extend(Path(spec["path"]) for spec in application.get("feature_sources", {}).values())
        paths.append(Path(application["internal_states"]["path"]))
        forbidden = [str(path) for path in paths if _contains_forbidden_path_token(path)]
        if forbidden:
            raise ValueError(f"public application refuses forbidden paths: {forbidden}")
        if freeze_manifest_path is not None:
            raise ValueError("public application must not provide a frozen-test authorization")
        return None
    if cohort_kind != "frozen_test":
        raise ValueError("cohort_kind must be public or frozen_test")
    if freeze_manifest_path is None:
        raise ValueError("frozen_test application requires --evaluation-freeze-manifest")
    if _contains_forbidden_path_token(freeze_manifest_path):
        raise ValueError("evaluation freeze manifest path itself must remain public")
    freeze = _read_json_unchecked(freeze_manifest_path)
    if freeze.get("schema_version") != FREEZE_SCHEMA or not freeze.get("authorized"):
        raise ValueError("invalid or unauthorized evaluation freeze manifest")
    expected = {
        "application_manifest_sha256": sha256_file(application_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "thresholds_sha256": sha256_file(thresholds_path),
    }
    mismatches = {
        key: (freeze.get(key), value)
        for key, value in expected.items()
        if freeze.get(key) != value
    }
    if mismatches:
        raise ValueError(f"evaluation freeze hash mismatch: {mismatches}")
    return freeze


def _keyed_records(payload: dict, source: Path, expected_ids: set[str]) -> dict[str, dict]:
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError(f"feature source lacks records list: {source}")
    keyed = {row.get("sample_id"): row for row in rows}
    if None in keyed or len(keyed) != len(rows):
        raise ValueError(f"duplicate/missing sample ID in {source}")
    if set(keyed) != expected_ids:
        missing = sorted(expected_ids - set(keyed))[:5]
        extra = sorted(set(keyed) - expected_ids)[:5]
        raise ValueError(f"application source alignment failure: missing={missing}, extra={extra}")
    return keyed


def _verify_declared_sha(path: Path, declared: str, source: str) -> str:
    actual = sha256_file(path)
    if actual != declared:
        raise ValueError(f"{source} SHA256 mismatch: {actual} != {declared}")
    return actual


def _resolve_feature_names(
    names: list[str], requested: list[str], group: str
) -> list[str]:
    output = []
    for name in requested:
        if name.endswith(".*"):
            prefix = name[:-1]
            matches = [candidate for candidate in names if candidate.startswith(prefix)]
            if not matches:
                raise KeyError(f"{group} wildcard matched no application features: {name}")
            output.extend(matches)
        elif name in names:
            output.append(name)
        else:
            raise KeyError(f"{group} references unavailable application feature: {name}")
    output = list(dict.fromkeys(output))
    if not output:
        raise ValueError(f"application feature group is empty: {group}")
    return output


def load_frozen_application_data(
    application: dict,
    checkpoint: dict,
    permit_locked_metadata: bool,
) -> tuple[CellData, dict[str, str]]:
    if application.get("schema_version") != APPLICATION_SCHEMA:
        raise ValueError(f"application schema must be {APPLICATION_SCHEMA}")
    if application.get("cell") != checkpoint.get("cell"):
        raise ValueError("application/checkpoint cell mismatch")
    records = application.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("application manifest requires non-empty records")
    sample_ids = [row.get("sample_id") for row in records]
    if None in sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("application records contain duplicate/missing sample IDs")
    expected_ids = set(sample_ids)
    allowed_record_fields = {
        "sample_id",
        "fact_key_sha256",
        "split",
        "pair_id",
        "class_stratum",
        "model",
        "dataset",
    }
    for row in records:
        forbidden = sorted(set(row) - allowed_record_fields)
        if forbidden:
            raise ValueError(
                f"application records may contain identifiers only; forbidden fields: {forbidden}"
            )
    groups = np.asarray([row.get("fact_key_sha256", "") for row in records], dtype=object)
    if any(not isinstance(value, str) or not value for value in groups):
        raise ValueError("application records require a non-empty fact_key_sha256 identifier")

    internal_spec = application["internal_states"]
    internal_path = Path(internal_spec["path"])
    internal_sha = _verify_declared_sha(
        internal_path, str(internal_spec["sha256"]), "internal state artifact"
    )
    artifact = torch.load(internal_path, map_location="cpu", weights_only=False)
    if artifact.get("locked_accessed") and not permit_locked_metadata:
        raise ValueError("public application received internal states marked locked_accessed")
    if artifact.get("cell") != checkpoint["cell"]:
        raise ValueError("internal-state/checkpoint cell mismatch")
    if str(artifact.get("model_revision")) != str(checkpoint["model_revision"]):
        raise ValueError("internal state model revision differs from frozen checkpoint")
    if str(artifact.get("dtype")) != str(checkpoint["dtype"]):
        raise ValueError("internal state dtype differs from frozen checkpoint")
    if application.get("model_revision") != checkpoint["model_revision"]:
        raise ValueError("application model_revision differs from frozen checkpoint")
    if application.get("dtype") != checkpoint["dtype"]:
        raise ValueError("application dtype differs from frozen checkpoint")
    selected_indices = [int(value) for value in checkpoint["selected_layer_indices"]]
    selected_numbers = [int(value) for value in checkpoint["selected_layer_numbers"]]
    application_layers = [int(value) for value in application.get("selected_layer_numbers", [])]
    if application_layers != selected_numbers:
        raise ValueError("application selected layers differ from frozen checkpoint")
    layer_numbers = artifact.get("layer_numbers", [])
    if max(selected_indices, default=-1) >= len(layer_numbers):
        raise ValueError("internal state artifact lacks a frozen selected layer")
    if [int(layer_numbers[index]) for index in selected_indices] != selected_numbers:
        raise ValueError("internal state layer numbering differs from frozen checkpoint")
    internal_ids = list(artifact.get("row_ids", []))
    if len(internal_ids) != len(set(internal_ids)) or set(internal_ids) != expected_ids:
        raise ValueError("internal state row IDs differ from application manifest")
    internal_index = {sample_id: index for index, sample_id in enumerate(internal_ids)}
    order = [internal_index[sample_id] for sample_id in sample_ids]
    states = {}
    for position in POSITIONS:
        tensor = artifact["states"][position][order][:, selected_indices]
        if tensor.ndim != 3:
            raise ValueError(f"invalid state tensor shape for {position}: {tuple(tensor.shape)}")
        states[position] = tensor.float().numpy()

    source_hashes = {str(internal_path): internal_sha}
    feature_columns: dict[str, np.ndarray] = {}
    feature_sources = application.get("feature_sources", {})
    if not feature_sources:
        raise ValueError("application has no explicit feature_sources")
    for source_name, spec in feature_sources.items():
        path = Path(spec["path"])
        actual_sha = _verify_declared_sha(path, str(spec["sha256"]), source_name)
        source_hashes[str(path)] = actual_sha
        payload = _read_json_unchecked(path)
        if payload.get("locked_accessed") and not permit_locked_metadata:
            raise ValueError(f"public application source marked locked_accessed: {path}")
        if payload.get("cell") not in (None, checkpoint["cell"]):
            raise ValueError(f"feature source cell mismatch: {path}")
        keyed = _keyed_records(payload, path, expected_ids)
        fields = spec.get("fields", {})
        if not fields:
            raise ValueError(f"feature source has no explicit fields: {source_name}")
        for local_name, dotted_path in fields.items():
            validate_feature_field(local_name, dotted_path)
            qualified = f"{source_name}.{local_name}"
            if qualified in feature_columns:
                raise ValueError(f"duplicate application feature: {qualified}")
            feature_columns[qualified] = np.asarray(
                [
                    numeric_value(
                        dotted_value(keyed[sample_id], dotted_path),
                        f"{path}:{dotted_path}",
                    )
                    for sample_id in sample_ids
                ],
                dtype=float,
            )
    model_schema = checkpoint["validation_model_fit_on_development_only"]["feature_schema"]
    expected_feature_names = list(
        dict.fromkeys(
            [
                *model_schema["behavioral"],
                *model_schema["candidate"],
                *model_schema["persistence"],
                *model_schema["gate_evidence"],
            ]
        )
    )
    if set(feature_columns) != set(expected_feature_names):
        missing = sorted(set(expected_feature_names) - set(feature_columns))
        extra = sorted(set(feature_columns) - set(expected_feature_names))
        raise ValueError(
            "application feature columns differ from frozen checkpoint: "
            f"missing={missing}, extra={extra}"
        )
    feature_names = expected_feature_names
    matrix = np.column_stack([feature_columns[name] for name in feature_names])
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    group_config = application.get("feature_groups", {})
    required_groups = ("behavioral", "candidate", "persistence", "gate_evidence")
    resolved = {
        name: _resolve_feature_names(feature_names, group_config.get(name, []), name)
        for name in required_groups
    }
    routing = dict(application.get("routing", {}))
    for key in ("behavioral_indicator", "candidate_available", "candidate_margin"):
        value = routing.get(key)
        if value is not None and value not in name_to_index:
            raise ValueError(f"routing references unavailable feature: {value}")
    routing["feature_name_to_index"] = name_to_index

    # Preserve non-feature metadata (currently phenotype_names) while replacing
    # only the groups reconstructed from the frozen application sources.
    application_model_schema = dict(model_schema)
    application_model_schema.update(
        {
            "base": model_schema["base"],
            "behavioral": resolved["behavioral"],
            "candidate": resolved["candidate"],
            "persistence": resolved["persistence"],
            "gate_evidence": resolved["gate_evidence"],
        }
    )
    if application_model_schema != model_schema:
        raise ValueError("application feature schema differs from frozen checkpoint")
    schema_hash = hashlib.sha256(
        json.dumps(model_schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if application.get("feature_schema_sha256") != schema_hash:
        raise ValueError("application feature_schema_sha256 differs from frozen checkpoint")

    data = CellData(
        cell=checkpoint["cell"],
        sample_ids=sample_ids,
        splits=np.asarray([str(row.get("split", "application")) for row in records], dtype=object),
        labels=np.zeros(len(records), dtype=int),
        groups=groups,
        states=states,
        selected_layer_indices=selected_indices,
        selected_layer_numbers=selected_numbers,
        feature_matrix=matrix,
        feature_names=feature_names,
        behavioral_indices=[name_to_index[name] for name in resolved["behavioral"]],
        candidate_indices=[name_to_index[name] for name in resolved["candidate"]],
        persistence_indices=[name_to_index[name] for name in resolved["persistence"]],
        evidence_indices=[name_to_index[name] for name in resolved["gate_evidence"]],
        routing=routing,
        phenotype_labels=np.full(len(records), -1, dtype=int),
        phenotype_names=list(model_schema.get("phenotype_names", ("behavioral", "internal", "persistent"))),
        source_hashes=source_hashes,
        model_revision=str(checkpoint["model_revision"]),
        dtype=str(checkpoint["dtype"]),
    )
    return data, source_hashes


def load_frozen_thresholds(
    thresholds_path: Path,
    checkpoint: dict,
) -> dict[str, dict]:
    payload = _read_json_unchecked(thresholds_path)
    if payload.get("cell") != checkpoint.get("cell"):
        raise ValueError("threshold/checkpoint cell mismatch")
    methods = payload.get("methods", {})
    if set(methods) != set(METHOD_NAMES):
        raise ValueError("frozen threshold methods differ from checkpoint methods")
    output = {}
    for name in METHOD_NAMES:
        operating_point = methods[name].get("validation_selected_for_future_test")
        if not isinstance(operating_point, Mapping) or "threshold" not in operating_point:
            raise ValueError(f"missing frozen validation threshold for {name}")
        output[name] = dict(operating_point)
    return output


def apply_frozen_checkpoint(
    checkpoint_path: Path,
    thresholds_path: Path,
    application_path: Path,
    output_path: Path,
    freeze_manifest_path: Path | None = None,
) -> dict:
    application = _read_json_unchecked(application_path)
    if application.get("schema_version") != APPLICATION_SCHEMA:
        raise ValueError(f"application schema must be {APPLICATION_SCHEMA}")
    if application.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("application checkpoint SHA256 does not match")
    if application.get("thresholds_sha256") != sha256_file(thresholds_path):
        raise ValueError("application threshold SHA256 does not match")
    freeze = validate_application_authorization(
        application_path,
        application,
        checkpoint_path,
        thresholds_path,
        freeze_manifest_path,
    )
    checkpoint = joblib.load(checkpoint_path)
    if checkpoint.get("locked_accessed"):
        raise ValueError("training checkpoint unexpectedly declares locked access")
    frozen_thresholds = load_frozen_thresholds(thresholds_path, checkpoint)
    data, source_hashes = load_frozen_application_data(
        application,
        checkpoint,
        permit_locked_metadata=freeze is not None,
    )
    scores = predict_stack(checkpoint["validation_model_fit_on_development_only"], data)
    if set(scores) != set(METHOD_NAMES):
        raise AssertionError("frozen model returned an unexpected method set")
    method_payload = {}
    for name in METHOD_NAMES:
        threshold = float(frozen_thresholds[name]["threshold"])
        values = np.asarray(scores[name], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"frozen prediction contains non-finite scores for {name}")
        method_payload[name] = {
            "frozen_validation_operating_point": frozen_thresholds[name],
            "records": [
                {
                    "sample_id": sample_id,
                    "fact_key_sha256": str(group),
                    "score": float(score),
                    "predicted_hce": bool(score >= threshold),
                }
                for sample_id, group, score in zip(data.sample_ids, data.groups, values)
            ],
        }
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "cell": checkpoint["cell"],
        "cohort_kind": application["cohort_kind"],
        "positive_class": "HCE",
        "refit_performed": False,
        "threshold_search_performed": False,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "thresholds_sha256": sha256_file(thresholds_path),
        "application_manifest_sha256": sha256_file(application_path),
        "evaluation_freeze_manifest_sha256": (
            sha256_file(freeze_manifest_path) if freeze_manifest_path is not None else None
        ),
        "source_sha256": source_hashes,
        "methods": method_payload,
    }
    atomic_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--application-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-freeze-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    apply_frozen_checkpoint(
        args.checkpoint,
        args.thresholds,
        args.application_manifest,
        args.output,
        args.evaluation_freeze_manifest,
    )


if __name__ == "__main__":
    main()
