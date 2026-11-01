"""One-shot evaluation of the paper configuration on the held-out test split.

Protocol (paper Sec. 4.4): fit on development, select per-checkpoint thresholds on validation (max recall subject
to FPR <= 10%), apply unchanged to test. The configuration is read from configs/paper_configuration.json.
Test-split features: canonical hidden states (*-locked-features.pt), Run6 frozen auxiliary sources
(panel / NLI / candidate / persistence, same fields as the public config) and template-panel features.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HOME = Path.home() / "hallu_paper/mosaic_rebuttal"
sys.path.insert(0, str(HOME / "work"))
import mosaic_tune as mt  # noqa: E402

orig = mt.orig
CONFIG = Path(__file__).resolve().parents[1] / "configs/paper_configuration.json"
RESULT = HOME / "tune/FINAL_TEST_RESULT.json"
G = HOME / "gpu1_cache"
SOURCES = {"behavioral": "corrected-panel-features", "nli": "corrected-panel-nli", "candidate": "candidate-margins",
           "persistence": "persistence"}


def test_rows(cell: str, public: dict, config_cell: dict) -> dict:
    tag = mt.TAGS[cell]
    text = json.loads((HOME / "locked_text/confidence_matched_challenge_locked_text.json").read_text())
    records = text["cells"][cell]["locked"]["records"]
    payload = torch.load(G / f"one_confidence_five_model_eval/{tag}-locked-features.pt", map_location="cpu",
                         weights_only=False)
    order = {sid: i for i, sid in enumerate(payload["row_ids"])}
    ids = [r["sample_id"] for r in records]
    take = [order[sid] for sid in ids]
    states = {p: payload["states"][p][take].float().numpy().astype(np.float16) for p in orig.POSITIONS}
    sources = {}
    for group, stem in SOURCES.items():
        data = json.loads((G / f"one_confidence_mosaic_v1_frozen_sources_run6/{tag}-{stem}.json").read_text())
        sources[group] = {r["sample_id"]: r for r in data["records"]}
    columns = []
    for name in public["feature_names"]:
        group, field = str(name).split(".", 1)
        dotted = config_cell["feature_sources"][group]["fields"][field]
        columns.append([orig.numeric_value(orig.dotted_value(sources[group][sid], dotted), f"{group}.{field}")
                        for sid in ids])
    controls, _ = orig.label_free_state_controls(SimpleNamespace(states={k: v.astype(np.float32)
                                                                          for k, v in states.items()}))
    ptrue = json.loads((G / f"one_confidence_five_model_eval/{tag}-locked-ptrue.json").read_text())["records"]
    index = json.loads((G / "one_confidence_benchmark_v2_bf16_expanded/fresh_locked_test_index.json").read_text())
    strata = {r["sample_id"]: r["confidence_stratum"] for r in index["records"]}
    extra = {"ptrue": mt.ptrue_features(ptrue, ids), "stratum": np.array([strata[s] for s in ids])}
    verify = HOME / f"verify_states/{tag}-locked.npz"
    if verify.exists():
        v = np.load(verify, allow_pickle=True)
        assert v["sample_ids"].tolist() == ids
        extra.update(state_verify=v["state_verify"], verify_odds=v["verify_odds"])
    tpl = np.load(HOME / f"template_panel/{tag}-locked.npz", allow_pickle=True)
    assert tpl["sample_ids"].tolist() == ids
    names = [str(n) for n in tpl["names"]]
    return {
        "sample_ids": np.array(ids), "splits": np.array(["locked"] * len(ids)),
        "labels": np.array([1 if r["class"] == "HCE" else 0 for r in records]),
        "groups": np.array([r["fact_key_sha256"] for r in records]),
        "features": np.array(columns, dtype=float).T, "controls": controls,
        "tpl_basic": tpl["features"][:, [i for i, n in enumerate(names) if n.startswith("tpl.")]],
        "tpl_xq": tpl["features"][:, [i for i, n in enumerate(names) if n.startswith("tplxq.")]],
        **{f"state_{p}": states[p] for p in orig.POSITIONS}, **extra,
    }


def combined(cell: str, config_cell: dict) -> dict:
    public = dict(mt.load(cell, "cached"))
    test = test_rows(cell, public, config_cell)
    out = {"_key": f"{cell}-final-test"}
    for key in ("sample_ids", "splits", "labels", "groups", "features", "controls", "tpl_basic", "tpl_xq",
                *[f"state_{p}" for p in orig.POSITIONS], "ptrue", "stratum", "state_verify", "verify_odds"):
        if key in public and key in test:
            out[key] = np.concatenate([public[key], test[key]])
    for key in ("feature_names", "behavioral_idx", "candidate_idx", "persistence_idx", "evidence_idx",
                "layer_numbers", "verify_slot_index"):
        out[key] = public[key]
    return out


def evaluate(threshold_from: tuple, test: tuple) -> dict:
    (yv, sv), (yt, st) = threshold_from, test
    _, selected = orig.enumerate_thresholds(yv, sv, 0.10)
    row = orig.metric_row(yt, st, selected["threshold"])
    m = mt.metrics(yt, st)
    row["pauc_raw_0_0.1"], row["auroc"] = m["pauc"], m["auroc"]
    row["val_threshold_row"] = {k: selected[k] for k in ("threshold", "recall", "fpr")}
    return row


PARTIAL = HOME / "tune/FINAL_TEST_PARTIAL.json"


def rehearsal_split(d):
    """Rehearsal only: no locked rows. Validation facts are split in half; thresholds come from one half and the
    other half plays the test role, so the whole code path runs without touching the test split."""
    val = np.where(d["splits"] == "validation")[0]
    facts = np.unique(d["groups"][val])
    rng = np.random.default_rng(0)
    rng.shuffle(facts)
    first = set(facts[: len(facts) // 2].tolist())
    in_first = np.array([g in first for g in d["groups"][val]])
    return val[in_first], val[~in_first]


def main() -> None:
    rehearse = "--rehearse" in sys.argv
    if rehearse:
        config_systems = json.loads(Path(sys.argv[sys.argv.index("--rehearse") + 1]).read_text())
    else:
        config_systems = json.loads(CONFIG.read_text())
    config = json.loads(mt.RUN6_CONFIG.read_text())
    config_cells = {c["cell"]: c for c in config["cells"]}
    # A crash must not force a second look at finished cells: completed cells are kept in PARTIAL and reused.
    results = json.loads(PARTIAL.read_text()) if (PARTIAL.exists() and not rehearse) else {"configuration": config_systems, "cells": {}, "scores": {}}
    for cell in mt.CELLS:
        if cell in results["cells"]:
            print(cell, "already scored (partial file); reused", flush=True)
            continue
        if rehearse:
            d = dict(mt.load(cell, "cached"))
            dev = np.where(d["splits"] == "development")[0]
            val, test = rehearsal_split(d)
        else:
            d = combined(cell, config_cells[cell])
            dev = np.where(d["splits"] == "development")[0]
            val = np.where(d["splits"] == "validation")[0]
            test = np.where(d["splits"] == "locked")[0]
        target = np.concatenate([val, test])
        cell_out, cell_scores = {}, {}
        for name, entry in config_systems["systems"].items():
            # A system may bag several cross-fitting seeds: per-method logits are averaged over `inner_seeds`.
            runs = [mt.fit_predict(d, dict(entry["cfg"], inner_seed=k), dev, target,
                                   seed=2026103101 + 1009 * k) for k in entry.get("inner_seeds", [0])]
            for method in entry["methods"]:
                logits = [np.log(np.clip(r[method], 1e-6, 1 - 1e-6) / (1 - np.clip(r[method], 1e-6, 1 - 1e-6))) for r in runs]
                s = 1 / (1 + np.exp(-np.mean(logits, axis=0)))
                sv, st = s[:len(val)], s[len(val):]
                key = f"{name}:{method}"
                cell_out[key] = evaluate((d["labels"][val], sv), (d["labels"][test], st))
                cell_scores[key] = st.tolist()
        results["cells"][cell] = cell_out
        results["scores"][cell] = {"labels": d["labels"][test].tolist(), "groups": d["groups"][test].tolist(),
                                   "sample_ids": d["sample_ids"][test].tolist(), **cell_scores}
        if not rehearse:
            PARTIAL.write_text(json.dumps(results))
        print(cell, "scored", flush=True)
    keys = next(iter(results["cells"].values())).keys()
    results["macro"] = {k: {m: float(np.mean([results["cells"][c][k][m] for c in mt.CELLS]))
                            for m in ("recall", "fpr", "accuracy", "pauc_raw_0_0.1", "auroc")} for k in keys}
    if rehearse:
        print("REHEARSAL (validation halves, no test rows):")
    else:
        RESULT.write_text(json.dumps(results, indent=1))
    for k, v in results["macro"].items():
        print(f"{k:28s} R={100 * v['recall']:.2f} FPR={100 * v['fpr']:.2f} AUROC={v['auroc']:.4f} pAUC={v['pauc_raw_0_0.1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
