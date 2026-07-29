#!/usr/bin/env python3
"""Authorize read-only feature extraction on the already opened frozen cohort.

Public feature scripts import this module so their mathematical definitions stay
identical. The only added behavior is access control and explicit provenance:
an authorization artifact is read and validated before the runtime benchmark is
opened, every stage has a predeclared output path, and frozen outputs are marked
as such rather than being disguised as public development/validation rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


AUTH_SCHEMA = "one-confidence-mosaic-v1-frozen-source-generation-authorization-v1"
RUNTIME_SCHEMA = "one-confidence-locked-runtime-v1"
PUBLIC_SPLITS = ("development", "validation")
FROZEN_SPLIT = "locked_test"
FORBIDDEN_UNAUTHORIZED_PATH_TOKENS = ("locked", "frozen")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def load_authorization(path: Path) -> dict[str, Any]:
    """Read and validate the public control artifact before any frozen source."""

    payload = _read_json(path)
    if payload.get("schema_version") != AUTH_SCHEMA:
        raise ValueError(f"authorization schema must be {AUTH_SCHEMA}")
    if payload.get("authorized") is not True:
        raise PermissionError("frozen source generation is not authorized")
    if payload.get("locked_access_acknowledged") is not True:
        raise PermissionError("authorization lacks locked-access acknowledgement")
    if payload.get("one_shot") is not True:
        raise PermissionError("frozen source authorization must be one-shot")
    if payload.get("purpose") != "one_confidence_generate_frozen_sources_once":
        raise ValueError("unexpected frozen source authorization purpose")
    if not payload.get("authorization_id"):
        raise ValueError("frozen source authorization requires authorization_id")
    return payload


def authorized_stage(
    authorization: dict[str, Any],
    *,
    cell: str,
    stage: str,
    output_path: Path,
) -> dict[str, Any]:
    """Validate one predeclared one-shot stage without opening its data inputs."""

    cell_auth = authorization.get("cells", {}).get(cell)
    if not isinstance(cell_auth, dict):
        raise KeyError(f"cell is not authorized: {cell}")
    stage_auth = cell_auth.get("stages", {}).get(stage)
    if not isinstance(stage_auth, dict):
        raise KeyError(f"stage is not authorized for {cell}: {stage}")
    if str(output_path) != str(stage_auth.get("output_path")):
        raise ValueError(f"output path differs from authorization for {cell}/{stage}")
    if output_path.exists():
        raise FileExistsError(f"one-shot frozen source output already exists: {output_path}")
    return stage_auth


def load_rows(
    *,
    benchmark_path: Path,
    cell: str,
    authorization_path: Path | None,
    stage: str,
    output_path: Path,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]], bool, str | None]:
    """Load public rows or an explicitly authorized frozen cell."""

    authorization = None
    if authorization_path is not None:
        authorization = load_authorization(authorization_path)
    elif any(
        token in str(benchmark_path).casefold()
        for token in FORBIDDEN_UNAUTHORIZED_PATH_TOKENS
    ):
        # Refuse before open/read/stat/hash. Public callers must use the public
        # challenge path; frozen paths always require the control artifact.
        raise PermissionError("frozen runtime access requires --frozen-source-authorization")

    benchmark = _read_json(benchmark_path)
    frozen = benchmark.get("schema_version") == RUNTIME_SCHEMA
    if not frozen:
        if authorization is not None:
            raise ValueError("frozen authorization cannot be used with a public benchmark")
        if cell not in benchmark.get("cells", {}):
            raise KeyError(f"unknown benchmark cell: {cell}")
        rows = [
            (split_name, row)
            for split_name in PUBLIC_SPLITS
            for row in benchmark["cells"][cell][split_name]["records"]
        ]
        return benchmark, rows, False, None

    if authorization is None:
        raise PermissionError("frozen runtime access requires --frozen-source-authorization")
    declared_runtime = authorization.get("runtime_benchmark", {})
    if str(benchmark_path) != str(declared_runtime.get("path")):
        raise ValueError("runtime benchmark path differs from authorization")
    if sha256_file(benchmark_path) != declared_runtime.get("sha256"):
        raise ValueError("runtime benchmark SHA256 differs from authorization")
    cell_auth = authorization.get("cells", {}).get(cell)
    if not isinstance(cell_auth, dict):
        raise KeyError(f"cell is not authorized: {cell}")
    authorized_stage(
        authorization,
        cell=cell,
        stage=stage,
        output_path=output_path,
    )
    rows = [
        (FROZEN_SPLIT, row)
        for row in benchmark.get("cells", {}).get(cell, {}).get("locked", {}).get("records", [])
    ]
    expected = int(cell_auth.get("expected_rows", -1))
    if len(rows) != expected:
        raise ValueError(f"authorized row count mismatch for {cell}: {len(rows)} != {expected}")
    return benchmark, rows, True, str(authorization["authorization_id"])


def require_same_access(
    payload: dict[str, Any],
    *,
    frozen: bool,
    authorization_id: str | None,
    source_name: str,
) -> None:
    if bool(payload.get("locked_accessed")) != frozen:
        raise ValueError(f"{source_name} access mode differs from benchmark")
    if frozen and payload.get("source_authorization_id") != authorization_id:
        raise ValueError(f"{source_name} authorization ID differs from benchmark run")


def require_authorized_argument(
    authorization_path: Path | None,
    *,
    cell: str,
    name: str,
    value: str,
) -> None:
    """Freeze model/NLI paths for an authorized source-generation run."""

    if authorization_path is None:
        return
    authorization = load_authorization(authorization_path)
    cell_auth = authorization.get("cells", {}).get(cell)
    if not isinstance(cell_auth, dict):
        raise KeyError(f"cell is not authorized: {cell}")
    expected = cell_auth.get("runtime_arguments", {}).get(name)
    if str(value) != str(expected):
        raise ValueError(f"{name} differs from authorization for {cell}")
