#!/usr/bin/env python3
"""Unified public-split evaluation for One Confidence five-model cells.

This module never reads the locked index. It materializes per-sample scores and
selects an operating threshold on validation at maximum recall under 10% FPR.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


MAIN_CELLS = (
    "mistral7|TriviaQA",
    "qwen8|TriviaQA",
    "llama31|TriviaQA",
    "qwen14|PopQA",
    "nemo12|PopQA",
)


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = scores >= threshold
    tp = int(np.sum((labels == 1) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    fp = int(np.sum((labels == 0) & predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    return {
        "threshold": float(threshold), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.10) -> dict:
    # Above-max is included so the valid reject-none operating point always exists.
    candidates = np.r_[np.nextafter(scores.max(), math.inf), np.unique(scores)]
    valid = [metrics(labels, scores, value) for value in candidates]
    valid = [row for row in valid if row["fpr"] <= max_fpr + 1e-12]
    return max(valid, key=lambda row: (row["recall"], row["accuracy"], row["threshold"]))


def rows_for(split: dict) -> tuple[list[dict], np.ndarray]:
    rows = split["records"]
    labels = np.asarray([1 if row["class"] == "HCE" else 0 for row in rows], dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError("each public split must contain HCE and HCC")
    return rows, labels


def sequence_confidence_risk(row: dict) -> float:
    # Error is positive, so lower answer confidence means greater risk.
    return -float(row["confidence"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    result = {
        "schema_version": "one-confidence-five-model-public-eval-v1",
        "positive_class": "HCE",
        "locked_accessed": False,
        "operating_point": "validation maximum recall subject to FPR <= 0.10",
        "methods": {},
    }
    method = {"implementation": "exact", "score_direction": "higher_is_more_error", "cells": {}}
    for cell_name in MAIN_CELLS:
        cell = benchmark["cells"][cell_name]
        dev_rows, dev_labels = rows_for(cell["development"])
        val_rows, val_labels = rows_for(cell["validation"])
        dev_scores = np.asarray([sequence_confidence_risk(row) for row in dev_rows])
        val_scores = np.asarray([sequence_confidence_risk(row) for row in val_rows])
        selected = choose_threshold(val_labels, val_scores)
        method["cells"][cell_name] = {
            "development_auroc_supplementary": float(roc_auc_score(dev_labels, dev_scores)),
            "validation": selected,
            "scores": [
                {"sample_id": row["sample_id"], "split": "development", "label": row["class"], "score": float(score)}
                for row, score in zip(dev_rows, dev_scores)
            ] + [
                {"sample_id": row["sample_id"], "split": "validation", "label": row["class"], "score": float(score)}
                for row, score in zip(val_rows, val_scores)
            ],
        }
    result["methods"]["sequence_confidence"] = method
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
