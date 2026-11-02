"""Metrics of the external detectors on the held-out test split, from the per-example scores in results/.

Thresholds were selected on validation at FPR <= 10% and are applied unchanged. A row is flagged when its score is
at or above the threshold. Macro metrics weight the five checkpoints equally, accuracy is the mean of recall and 1 - FPR.
"""
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    data = json.loads((RESULTS / "external_baselines_scores.json").read_text())
    for method, cells in data["methods"].items():
        recall, fpr, tp_fp = [], [], []
        for cell in cells.values():
            labels = np.array(cell["labels"])
            flagged = np.array(cell["scores"]) >= cell["threshold"]
            recall.append(flagged[labels == 1].mean())
            fpr.append(flagged[labels == 0].mean())
            tp_fp.append(f"{flagged[labels == 1].sum()}/{flagged[labels == 0].sum()}")
        r, f = 100 * np.mean(recall), 100 * np.mean(fpr)
        print(f"{method:32s} {' '.join(tp_fp):34s} R={r:.2f} FPR={f:.2f} Acc={(r + 100 - f) / 2:.2f}")


if __name__ == "__main__":
    main()
