from __future__ import annotations

import json
from pathlib import Path

import compile_one_confidence_mosaic_v1_ablation_delivery as module


def test_compile_ablation_keeps_test_selection_frozen(tmp_path: Path) -> None:
    validation_path = tmp_path / "validation.json"
    frozen_path = tmp_path / "frozen.json"
    validation_path.write_text(
        json.dumps({"rows": [{"split": "validation", "method": "behavioral_only"}]}),
        encoding="utf-8",
    )
    frozen_path.write_text(
        json.dumps(
            {
                "cells": {
                    "m|D": {
                        "methods": {
                            "mosaic_v1_run6": {
                                "status": "frozen",
                                "TP": 1,
                                "FN": 1,
                                "FP": 0,
                                "TN": 2,
                                "accuracy": 0.75,
                                "precision": 1.0,
                                "recall": 0.5,
                                "fpr": 0.0,
                                "f1": 2 / 3,
                                "threshold": 0.6,
                            }
                        }
                    }
                },
                "macro_equal_cell_average": {
                    "mosaic_v1_run6": {
                        "accuracy": 0.75,
                        "precision": 1.0,
                        "recall": 0.5,
                        "fpr": 0.0,
                        "f1": 2 / 3,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = module.compile_ablation(validation_path, frozen_path)

    assert len(result["validation_scope"]["rows"]) == 1
    frozen = result["retrospective_frozen_test_scope"]
    assert frozen["selected_method_only"] is True
    assert frozen["threshold_search_performed"] is False
    assert [row["method"] for row in frozen["rows"]] == [
        "learned_gate_no_phenotype",
        "learned_gate_no_phenotype",
    ]


def test_compile_ablation_rejects_nonvalidation_ablation_row(tmp_path: Path) -> None:
    validation_path = tmp_path / "validation.json"
    frozen_path = tmp_path / "frozen.json"
    validation_path.write_text(
        json.dumps({"rows": [{"split": "locked_test", "method": "bad"}]}),
        encoding="utf-8",
    )
    frozen_path.write_text("{}", encoding="utf-8")

    try:
        module.compile_ablation(validation_path, frozen_path)
    except ValueError as error:
        assert "non-validation" in str(error)
    else:
        raise AssertionError("expected a non-validation row to be rejected")
