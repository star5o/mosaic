"""Full-pipeline hierarchical bootstrap (paper Sec. 4.4): MOSAIC vs the candidate-free trajectory baseline.

Replicate 0 uses the original rows (point estimate). Replicate r >= 1 resamples fact groups with replacement
separately within development, validation and test of every checkpoint, refits probes, experts and gate on the
resampled development rows, reselects the FPR <= 10% threshold on the resampled validation rows, and scores the
resampled test rows. Both systems see the same resampled rows. Output: one JSON line per (cell, replicate).
Usage: python full_bootstrap.py N_REPLICATES WORKERS
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "hallu_paper/mosaic_rebuttal/work"))
import mosaic_tune as mt
import final_test as ft

HOME = Path.home() / "hallu_paper/mosaic_rebuttal"
OUT = HOME / "tune/full_bootstrap.jsonl"
RUN6 = json.loads((Path(__file__).resolve().parents[1] / "configs/paper_configuration.json").read_text())["systems"]["paper_configuration"]["cfg"]
SYSTEMS = {"trajectory": (dict(RUN6, internal_candidate=False), "internal"), "mosaic": (RUN6, "mosaic"),
           "internal_plus_behavior": (RUN6, "internal_plus_behavior")}
CONFIG_CELLS = {c["cell"]: c for c in json.loads(mt.RUN6_CONFIG.read_text())["cells"]}
_DATA = {}


def data(cell):
    if cell not in _DATA:
        _DATA.clear()  # one checkpoint in memory per worker (12 workers x 5 cached checkpoints exhausted RAM)
        _DATA[cell] = ft.combined(cell, CONFIG_CELLS[cell])
    return _DATA[cell]


def resample(d, split, rng):
    idx = np.where(d["splits"] == split)[0]
    if rng is None:
        return idx
    groups = d["groups"][idx]
    keys = np.unique(groups)
    by = {k: idx[groups == k] for k in keys}
    return np.concatenate([by[k] for k in rng.choice(keys, len(keys))])


def block(a):
    cell, reps = a
    return [job((cell, r)) for r in reps]


def job(a):
    cell, rep = a
    d = data(cell)
    rng = None if rep == 0 else np.random.default_rng([20261101, rep, mt.CELLS.index(cell)])
    dev, val, test = (resample(d, s, rng) for s in ("development", "validation", "locked"))
    target = np.concatenate([val, test])
    out = {"cell": cell, "rep": rep}
    fits = {}
    for name, (cfg, method) in SYSTEMS.items():
        key = json.dumps(cfg, sort_keys=True)
        if key not in fits:
            fits[key] = mt.fit_predict(d, cfg, dev, target, seed=2026103101)
        s = fits[key][method]
        row = ft.evaluate((d["labels"][val], s[:len(val)]), (d["labels"][test], s[len(val):]))
        out[name] = {"recall": row["recall"], "fpr": row["fpr"], "auroc": row["auroc"]}
    return out


if __name__ == "__main__":
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    done = set()
    if OUT.exists():
        done = {(r["cell"], r["rep"]) for r in map(json.loads, OUT.read_text().splitlines())}
    jobs = []
    for start in range(0, n, 10):  # blocks of 10 replicates per checkpoint, replicate blocks in order
        for c in mt.CELLS:
            reps = [r for r in range(start, min(start + 10, n)) if (c, r) not in done]
            if reps:
                jobs.append((c, reps))
    with Pool(workers, maxtasksperchild=2) as p, OUT.open("a") as f:  # recycle workers: fit_predict leaks memory
        for res in p.imap_unordered(block, jobs):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
