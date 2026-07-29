#!/usr/bin/env python3
"""Build gold-free OOF layer-trajectory features from cached public states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


POSITIONS = ("prompt_end", "first_answer", "mean_answer")


def estimator():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.01,
            class_weight="balanced",
            max_iter=2000,
            random_state=2026072910,
            solver="liblinear",
        ),
    )


def fixed_layer_indices(count: int, requested: int = 8) -> list[int]:
    if count <= requested:
        return list(range(count))
    return sorted({int(round(value)) for value in np.linspace(0, count - 1, requested)})


def oof_and_validation_scores(
    states: np.ndarray,
    labels: np.ndarray,
    dev_mask: np.ndarray,
    validation_mask: np.ndarray,
) -> np.ndarray:
    scores = np.full(len(labels), np.nan)
    dev_indices = np.flatnonzero(dev_mask)
    folds = StratifiedKFold(5, shuffle=True, random_state=2026072910)
    for train_local, holdout_local in folds.split(states[dev_mask], labels[dev_mask]):
        train = dev_indices[train_local]
        holdout = dev_indices[holdout_local]
        scores[holdout] = estimator().fit(states[train], labels[train]).predict_proba(states[holdout])[:, 1]
    model = estimator().fit(states[dev_mask], labels[dev_mask])
    scores[validation_mask] = model.predict_proba(states[validation_mask])[:, 1]
    if np.isnan(scores).any():
        raise ValueError("trajectory score contains unfilled rows")
    return scores


def trajectory_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    width = values.shape[0]
    early = values[: max(1, width // 3)]
    middle = values[max(1, width // 3) : max(2, 2 * width // 3)]
    late = values[max(2, 2 * width // 3) :]
    slopes = np.diff(values)
    centered = values >= 0.5
    return {
        f"{prefix}_early_mean": float(np.mean(early)),
        f"{prefix}_middle_mean": float(np.mean(middle)),
        f"{prefix}_late_mean": float(np.mean(late)),
        f"{prefix}_maximum": float(np.max(values)),
        f"{prefix}_maximum_layer_fraction": float(np.argmax(values) / max(1, width - 1)),
        f"{prefix}_maximum_slope": float(np.max(slopes)) if len(slopes) else 0.0,
        f"{prefix}_minimum_slope": float(np.min(slopes)) if len(slopes) else 0.0,
        f"{prefix}_late_minus_middle": float(np.mean(late) - np.mean(middle)),
        f"{prefix}_trajectory_auc": float(np.mean(values)),
        f"{prefix}_flip_count": float(np.sum(centered[1:] != centered[:-1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-layers", type=int, default=8)
    args = parser.parse_args()

    artifact = torch.load(args.features, map_location="cpu", weights_only=False)
    if artifact.get("locked_accessed"):
        raise ValueError("internal trajectory builder accepts public features only")
    labels = np.asarray([1 if label == "HCE" else 0 for label in artifact["labels"]], dtype=int)
    dev_mask = np.asarray([split == "development" for split in artifact["splits"]])
    validation_mask = np.asarray([split == "validation" for split in artifact["splits"]])
    layer_indices = fixed_layer_indices(len(artifact["layer_numbers"]), args.fixed_layers)
    layer_numbers = [artifact["layer_numbers"][index] for index in layer_indices]

    score_trajectories: dict[str, np.ndarray] = {}
    norm_trajectories: dict[str, np.ndarray] = {}
    change_trajectories: dict[str, np.ndarray] = {}
    for position in POSITIONS:
        states = artifact["states"][position][:, layer_indices].float()
        per_layer = []
        for local_layer in range(states.shape[1]):
            per_layer.append(
                oof_and_validation_scores(
                    states[:, local_layer].numpy(), labels, dev_mask, validation_mask
                )
            )
        score_trajectories[position] = np.column_stack(per_layer)
        norm_trajectories[position] = states.norm(dim=-1).numpy()
        change = torch.zeros(states.shape[:2], dtype=torch.float32)
        change[:, 1:] = (states[:, 1:] - states[:, :-1]).norm(dim=-1)
        change_trajectories[position] = change.numpy()
        print(f"{artifact['cell']} {position} complete", flush=True)

    records = []
    for index, sample_id in enumerate(artifact["row_ids"]):
        feature_values: dict[str, float] = {}
        late_scores = []
        maxima = []
        for position in POSITIONS:
            trajectory = score_trajectories[position][index]
            feature_values.update(trajectory_summary(trajectory, position))
            feature_values.update(trajectory_summary(norm_trajectories[position][index], f"{position}_norm"))
            feature_values.update(trajectory_summary(change_trajectories[position][index], f"{position}_layer_change"))
            late_scores.append(feature_values[f"{position}_late_mean"])
            maxima.append(feature_values[f"{position}_maximum"])
        feature_values["position_late_score_range"] = float(max(late_scores) - min(late_scores))
        feature_values["position_late_score_std"] = float(np.std(late_scores))
        feature_values["position_max_score_range"] = float(max(maxima) - min(maxima))
        records.append(
            {
                "sample_id": sample_id,
                "split": artifact["splits"][index],
                "label": artifact["labels"][index],
                "features": feature_values,
                "probe_trajectories": {
                    position: [float(value) for value in score_trajectories[position][index]]
                    for position in POSITIONS
                },
            }
        )

    payload = {
        "schema_version": "one-confidence-internal-trajectory-v1",
        "cell": artifact["cell"],
        "locked_accessed": False,
        "source_feature_path": str(args.features),
        "source_model_revision": artifact["model_revision"],
        "fixed_layer_numbers": layer_numbers,
        "positions": POSITIONS,
        "probe": {
            "model": "standardized logistic regression",
            "C": 0.01,
            "development_predictions": "five-fold out-of-fold",
            "validation_predictions": "fit on complete development only",
            "layer_selection": "eight depth-uniform layers fixed before validation",
        },
        "gold_visible": "labels_only",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
