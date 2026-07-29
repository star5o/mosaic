import hashlib
import json
from pathlib import Path

import pytest
import torch

import one_confidence_frozen_source_access as access
from materialize_one_confidence_frozen_run6_aux_sources import (
    sanitize_internal,
    sanitize_persistence,
)


CELL = "model|Dataset"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_payload() -> dict:
    rows = [
        {
            "sample_id": f"{CELL}|{index}",
            "class": "HCE" if index else "HCC",
            "fact_key_sha256": str(index) * 64,
        }
        for index in range(2)
    ]
    return {
        "schema_version": access.RUNTIME_SCHEMA,
        "cells": {CELL: {"locked": {"records": rows}}},
    }


def authorization_payload(runtime: Path, stages: dict) -> dict:
    return {
        "schema_version": access.AUTH_SCHEMA,
        "purpose": "one_confidence_generate_frozen_sources_once",
        "authorized": True,
        "locked_access_acknowledged": True,
        "one_shot": True,
        "authorization_id": "test-authorization",
        "runtime_benchmark": {"path": str(runtime), "sha256": sha256(runtime)},
        "cells": {
            CELL: {
                "expected_rows": 2,
                "runtime_arguments": {
                    "model_path": "/models/model",
                    "nli_model": "/models/nli",
                },
                "stages": stages,
            }
        },
    }


def test_unauthorized_locked_path_is_rejected_before_read(monkeypatch, tmp_path):
    opened = []

    def fail_read(path):
        opened.append(path)
        raise AssertionError("locked path was opened")

    monkeypatch.setattr(access, "_read_json", fail_read)
    with pytest.raises(PermissionError):
        access.load_rows(
            benchmark_path=tmp_path / "locked_runtime_benchmark.json",
            cell=CELL,
            authorization_path=None,
            stage="stage",
            output_path=tmp_path / "output.json",
        )
    assert opened == []


def test_authorized_rows_require_exact_hash_stage_and_output(tmp_path):
    runtime = tmp_path / "locked_runtime_benchmark.json"
    output = tmp_path / "result.json"
    write_json(runtime, runtime_payload())
    authorization = tmp_path / "authorization.json"
    write_json(
        authorization,
        authorization_payload(runtime, {"stage": {"output_path": str(output)}}),
    )
    _, rows, frozen, authorization_id = access.load_rows(
        benchmark_path=runtime,
        cell=CELL,
        authorization_path=authorization,
        stage="stage",
        output_path=output,
    )
    assert frozen is True
    assert authorization_id == "test-authorization"
    assert [row[0] for row in rows] == ["locked_test", "locked_test"]
    access.require_authorized_argument(
        authorization,
        cell=CELL,
        name="model_path",
        value="/models/model",
    )
    with pytest.raises(ValueError, match="model_path differs"):
        access.require_authorized_argument(
            authorization,
            cell=CELL,
            name="model_path",
            value="/models/other",
        )
    with pytest.raises(ValueError, match="output path differs"):
        access.load_rows(
            benchmark_path=runtime,
            cell=CELL,
            authorization_path=authorization,
            stage="stage",
            output_path=tmp_path / "wrong.json",
        )


def test_auxiliary_sanitizers_remove_supervision(tmp_path):
    runtime = tmp_path / "locked_runtime_benchmark.json"
    write_json(runtime, runtime_payload())
    internal_source = tmp_path / "legacy-hidden.pt"
    persistence_source = tmp_path / "legacy-persistence.json"
    internal_output = tmp_path / "internal-output.pt"
    persistence_output = tmp_path / "persistence-output.json"
    torch.save(
        {
            "schema_version": "legacy",
            "cell": CELL,
            "model_revision": "revision",
            "dtype": "bfloat16",
            "locked_accessed": False,
            "row_ids": [f"{CELL}|0", f"{CELL}|1"],
            "labels": ["HCC", "HCE"],
            "splits": ["locked", "locked"],
            "token_entropy_records": [{"label": "HCC"}, {"label": "HCE"}],
            "layer_numbers": [1, 2],
            "states": {
                position: torch.arange(12).reshape(2, 2, 3).float()
                for position in ("prompt_end", "first_answer", "mean_answer")
            },
        },
        internal_source,
    )
    write_json(
        persistence_source,
        {
            "K": 4,
            "records": [
                {
                    "sample_id": f"{CELL}|{index}",
                    "label": "HCE" if index else "HCC",
                    "self_consistency_risk": 0.25 * index,
                    "semantic_entropy_risk": 0.5 * index,
                    "original_support": 1.0 - 0.25 * index,
                    "cluster_representatives": ["secret"],
                }
                for index in range(2)
            ],
        },
    )
    stages = {
        "sanitized_internal_states": {
            "output_path": str(internal_output),
            "input": {"path": str(internal_source), "sha256": sha256(internal_source)},
            "model_revision": "revision",
            "dtype": "bfloat16",
        },
        "sanitized_persistence": {
            "output_path": str(persistence_output),
            "input": {
                "path": str(persistence_source),
                "sha256": sha256(persistence_source),
            },
        },
    }
    authorization = tmp_path / "authorization.json"
    write_json(authorization, authorization_payload(runtime, stages))

    internal = sanitize_internal(
        authorization, runtime, CELL, internal_source, internal_output
    )
    persistence = sanitize_persistence(
        authorization, runtime, CELL, persistence_source, persistence_output
    )
    assert internal["locked_accessed"] is True
    assert "labels" not in internal and "splits" not in internal
    assert "token_entropy_records" not in internal
    assert persistence["locked_accessed"] is True
    assert persistence["contains_labels"] is False
    assert all("label" not in row for row in persistence["records"])
    assert all("cluster_representatives" not in row for row in persistence["records"])
