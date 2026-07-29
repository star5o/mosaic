#!/usr/bin/env python3
"""Apply a frozen LOS-Net checkpoint to locked LOS input features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_one_confidence_five_model import metrics
from fit_one_confidence_los_net import LOSNet, predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    features = torch.load(args.features, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    architecture = metadata["architecture"]

    if features["cell"] != metadata["cell"]:
        raise ValueError("LOS checkpoint and locked features belong to different cells")
    if features["topk"] != architecture["topk"] or features["vocab_size"] != architecture["vocab"]:
        raise ValueError("LOS feature dimensions do not match the frozen checkpoint")

    maxlen = int(architecture["maxlen"])
    tds = features["sorted_tds_normalized"][:, :maxlen]
    atp = features["normalized_atp"][:, :maxlen]
    rank = features["atp_r"][:, :maxlen]
    truncated_rows = int((features["sorted_tds_normalized"].shape[1] > maxlen) * len(features["row_ids"]))

    model = LOSNet(
        topk=architecture["topk"],
        vocab=architecture["vocab"],
        maxlen=maxlen,
        hidden=architecture["hidden"],
        heads=architecture["heads"],
        layers=architecture["layers"],
        dropout=architecture["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    labels = np.asarray([1 if value == "HCE" else 0 for value in features["labels"]], dtype=int)
    dummy_labels = torch.tensor(labels, dtype=torch.float32)
    data = (tds, atp, rank, dummy_labels)
    indices = np.arange(len(labels))
    scores = predict(model, data, indices, device)
    threshold = float(metadata["validation"]["threshold"])

    payload = {
        "schema_version": "one-confidence-los-net-locked-v1",
        "cell": features["cell"],
        "implementation_status": metadata["implementation_status"],
        "checkpoint_validation_threshold": threshold,
        "locked_threshold_selected": False,
        "sequence_length_policy": "truncate_to_frozen_public_maxlen",
        "frozen_maxlen": maxlen,
        "locked_raw_maxlen": int(features["sorted_tds_normalized"].shape[1]),
        "truncated_rows": truncated_rows,
        "metrics": metrics(labels, scores, threshold),
        "records": [
            {"sample_id": sample_id, "split": "locked", "label": label, "score": float(score)}
            for sample_id, label, score in zip(features["row_ids"], features["labels"], scores)
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
