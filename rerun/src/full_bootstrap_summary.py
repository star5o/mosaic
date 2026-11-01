"""Summarize full_bootstrap.jsonl: replicate 0 = point estimate on the original rows; replicates 1..N give the
percentile 95% CI of the macro difference and a one-sided bootstrap p-value (share of replicates with delta <= 0).
Only replicates with all five checkpoints present are used."""
import json, sys
from collections import defaultdict
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1])]
by = defaultdict(dict)
for r in rows:
    by[r["rep"]][r["cell"]] = r
cells = sorted(by[0])
full = sorted(k for k, v in by.items() if len(v) == len(cells))
systems = ("trajectory", "mosaic", "internal_plus_behavior")


def macro(rep, s, m="recall"):
    return 100 * np.mean([by[rep][c][s][m] for c in cells])


print("point estimate (original rows):")
for s in systems:
    print(f"  {s:10s} R={macro(0, s):.2f} FPR={macro(0, s, 'fpr'):.2f} AUROC={macro(0, s, 'auroc') / 100:.3f}  per-cell R:",
          " / ".join(f"{100 * by[0][c][s]['recall']:.1f}" for c in cells))
reps = [k for k in full if k > 0]
print(f"replicates used: {len(reps)}; cells: {cells}")
for a, b in (("mosaic", "trajectory"), ("internal_plus_behavior", "trajectory"), ("mosaic", "internal_plus_behavior")):
    d = np.array([macro(k, a) - macro(k, b) for k in reps])
    print(f"  {a} - {b}: point {macro(0, a) - macro(0, b):+.2f}, 95% CI [{np.percentile(d, 2.5):+.2f}, "
          f"{np.percentile(d, 97.5):+.2f}], bootstrap mean {d.mean():+.2f}, one-sided p = {(d <= 0).mean():.3f}")
