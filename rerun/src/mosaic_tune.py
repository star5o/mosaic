"""MOSAIC reconstruction and tuning on public (development/validation) data only.

The held-out test split is never loaded: data come from the original `load_cell_data`, which rejects any path
containing "locked". Every configuration is scored on validation after fitting on development, and the top
configurations are re-scored with repeated fact-grouped cross-validation over development+validation.

Usage (from ~/hallu_paper/mosaic_rebuttal/work):
  python mosaic_tune.py prepare --source cached|a100
  python mosaic_tune.py search --hours 12 --workers 8
  python mosaic_tune.py stage2 --top 30
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WORK = Path.home() / "hallu_paper/mosaic_rebuttal/work"
H = WORK / "experiments/hallucination"
sys.path.insert(0, str(H))
import evaluate_one_confidence_mosaic_v1_nested_public as orig  # noqa: E402

BENCHMARK = H / "results/one_confidence_benchmark_v2_bf16_expanded/confidence_matched_challenge_public.json"
RUN6_CONFIG = H / "configs/one_confidence_mosaic_v1_nested_public_run6_16layer_strong_regularization.json"
SOURCES = {
    "cached": "experiments/hallucination/results/one_confidence_five_model_eval/{tag}-public-features.pt",
    "a100": str(Path.home() / "hallu_paper/mosaic_rebuttal/reextract_a100/{tag}-public-features.pt"),
}
TAGS = {
    "mistral7|TriviaQA": "mistral7-triviaqa", "qwen8|TriviaQA": "qwen8-triviaqa",
    "llama31|TriviaQA": "llama31-triviaqa", "qwen14|PopQA": "qwen14-popqa", "nemo12|PopQA": "nemo12-popqa",
}
CELLS = list(TAGS)
OUT = Path.home() / "hallu_paper/mosaic_rebuttal/tune"
CACHE = OUT / "cache"
MAX_FPR = 0.10


# ---------------------------------------------------------------- data

def prepare(source: str) -> None:
    """Load every cell with all layers via the original loader and store a compact npz."""
    os.chdir(WORK)
    benchmark = json.loads(BENCHMARK.read_bytes())
    config = json.loads(RUN6_CONFIG.read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    for cell_config in config["cells"]:
        cell = cell_config["cell"]
        cell_config = json.loads(json.dumps(cell_config))
        cell_config["internal_states"] = SOURCES[source].format(tag=TAGS[cell])
        data = orig.load_cell_data(benchmark, BENCHMARK, cell_config, 10_000)
        assert not any(s not in ("development", "validation") for s in data.splits)
        controls, control_names = orig.label_free_state_controls(data)
        names = data.feature_names
        np.savez(
            CACHE / f"{TAGS[cell]}-{source}.npz",
            sample_ids=np.array(data.sample_ids), splits=data.splits, labels=data.labels,
            groups=data.groups.astype(str), layer_numbers=np.array(data.selected_layer_numbers),
            **{f"state_{p}": data.states[p].astype(np.float16) for p in orig.POSITIONS},
            features=data.feature_matrix, feature_names=np.array(names),
            behavioral_idx=np.array(data.behavioral_indices), candidate_idx=np.array(data.candidate_indices),
            persistence_idx=np.array(data.persistence_indices), evidence_idx=np.array(data.evidence_indices),
            controls=controls, control_names=np.array(control_names),
        )
        print(cell, source, data.states["mean_answer"].shape, flush=True)


_DATA: dict[tuple[str, str], dict] = {}


_PANEL: dict[str, dict] = {}


def panel_states(cell: str) -> dict:
    """Equivalent-question states of the original answer: (n, V, 16 slots, dim) per position, NaN = no variant."""
    if cell not in _PANEL:
        z = np.load(Path.home() / f"hallu_paper/mosaic_rebuttal/panel_states/{TAGS[cell]}-public.npz", allow_pickle=True)
        _PANEL[cell] = {k: z[k] for k in z.files}
    return _PANEL[cell]


def transfer_column(d, cfg, position, slot_index, train, test, fold_list, seed):
    """Probe transfer: error probe fit on canonical states, applied to equivalent-question states.

    Returns per-row [mean shift, max shift, spread] of the probe logit across variants relative to the canonical
    question; training rows come from fold models that did not see them (cross-fitted), test rows from the
    full-train model. Cached on disk like probe_column.
    """
    key = hashlib.sha1(json.dumps(["xfer", d["_key"], position, slot_index, cfg["pca"], cfg["probe_c"],
                                   cfg.get("probe_type", "logistic"), cfg["base_folds"], seed]).encode()
                       + train.tobytes() + test.tobytes()).hexdigest()
    path = CACHE / "xfer" / f"{key}.npz"
    if path.exists():
        try:
            z = np.load(path)
            return z["oof"], z["test"]
        except Exception:
            pass
    cell = d["_key"].rsplit("-", 1)[0]
    panel = panel_states(cell)
    slot = layer_slots(d, 16)[slot_index]
    x = state(d, position, slot)
    variants = panel[f"state_{position}"][:, :, slot_index, :].astype(np.float32)   # (n, V, dim)
    y = d["labels"]

    def shift(model, rows):
        logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        base = logit(model.predict_proba(x[rows])[:, 1])
        v = variants[rows]
        valid = ~np.isnan(v[..., 0])
        flat = np.nan_to_num(v.reshape(-1, v.shape[-1]))
        scores = logit(model.predict_proba(flat)[:, 1]).reshape(v.shape[0], v.shape[1])
        diff = np.where(valid, scores - base[:, None], np.nan)
        out = np.column_stack((np.nanmean(diff, 1), np.nanmax(diff, 1), np.nanstd(diff, 1)))
        return np.nan_to_num(out)

    seed_ = seed + 131 * slot_index + 7919 * POSITION_SEED[position]
    oof = np.zeros((len(train), 3))
    for k, (tr, ho) in enumerate(fold_list):
        model = probe(cfg, seed_ + k, len(tr)).fit(x[train[tr]], y[train[tr]])
        oof[ho] = shift(model, train[ho])
    model = probe(cfg, seed_, len(train)).fit(x[train], y[train])
    test_out = shift(model, test)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, oof=oof, test=test_out)
    os.replace(tmp, path)
    return oof, test_out


def transfer_features(d, cfg, train, test, seed):
    fold_list = folds(d["labels"][train], d["groups"][train], cfg["base_folds"], seed, "base")
    tr_parts, te_parts = [], []
    for position in ("first_answer", "mean_answer"):
        cols = [transfer_column(d, cfg, position, s, train, test, fold_list, seed) for s in range(16)]
        oof = np.stack([c[0] for c in cols], axis=1)    # (n_train, 16, 3)
        tst = np.stack([c[1] for c in cols], axis=1)
        for stat in range(3):
            tr_parts.append(orig.trajectory_summary(oof[:, :, stat]))
            te_parts.append(orig.trajectory_summary(tst[:, :, stat]))
    return np.column_stack(tr_parts), np.column_stack(te_parts)


def load(cell: str, source: str) -> dict:
    key = (cell, source)
    if key not in _DATA:
        z = np.load(CACHE / f"{TAGS[cell]}-{source}.npz", allow_pickle=True)  # own cache files
        _DATA[key] = {k: z[k] for k in z.files}
        _DATA[key]["_key"] = f"{cell}-{source}"
        xq_path = Path.home() / f"hallu_paper/mosaic_rebuttal/panel_states/{TAGS[cell]}-xq-features.npz"
        if xq_path.exists():
            xq = np.load(xq_path, allow_pickle=True)
            assert (xq["sample_ids"] == _DATA[key]["sample_ids"]).all()
            _DATA[key]["xq"] = xq["features"]
        tpl_path = Path.home() / f"hallu_paper/mosaic_rebuttal/template_panel/{TAGS[cell]}-public.npz"
        if tpl_path.exists():
            tpl = np.load(tpl_path, allow_pickle=True)
            assert (tpl["sample_ids"] == _DATA[key]["sample_ids"]).all()
            tpl_names = [str(n) for n in tpl["names"]]
            _DATA[key]["tpl_basic"] = tpl["features"][:, [i for i, n in enumerate(tpl_names) if n.startswith("tpl.")]]
            _DATA[key]["tpl_xq"] = tpl["features"][:, [i for i, n in enumerate(tpl_names) if n.startswith("tplxq.")]]
        records = {r["sample_id"]: r for c in json.loads(BENCHMARK.read_bytes())["cells"].values()
                   for split in ("development", "validation") for r in c[split]["records"]}
        _DATA[key]["stratum"] = np.array([records[s]["confidence_stratum"] if s in records else -1
                                          for s in _DATA[key]["sample_ids"]])
        verify_path = Path.home() / f"hallu_paper/mosaic_rebuttal/verify_states/{TAGS[cell]}-public.npz"
        if verify_path.exists():
            v = np.load(verify_path, allow_pickle=True)
            assert (v["sample_ids"] == _DATA[key]["sample_ids"]).all()
            _DATA[key]["state_verify"], _DATA[key]["verify_odds"] = v["state_verify"], v["verify_odds"]
            _DATA[key]["verify_slot_index"] = {int(a): i for i, a in enumerate(v["slots"])}
        lens_path = Path.home() / f"hallu_paper/mosaic_rebuttal/lens_attention/{TAGS[cell]}-public.npz"
        if lens_path.exists():
            v = np.load(lens_path, allow_pickle=True)
            assert (v["sample_ids"] == _DATA[key]["sample_ids"]).all()
            _DATA[key]["lens"] = v["features"]
        tok_path = Path.home() / f"hallu_paper/mosaic_rebuttal/token_states/{TAGS[cell]}-public.npz"
        if tok_path.exists():
            v = np.load(tok_path, allow_pickle=True)
            assert (v["sample_ids"] == _DATA[key]["sample_ids"]).all()
            _DATA[key]["tok_states"], _DATA[key]["tok_mask"] = v["states"], v["mask"]
        perturb_path = Path.home() / f"hallu_paper/mosaic_rebuttal/perturb/{TAGS[cell]}-public.npz"
        if perturb_path.exists():
            v = np.load(perturb_path, allow_pickle=True)
            assert (v["sample_ids"] == _DATA[key]["sample_ids"]).all()
            _DATA[key]["perturb"] = v["features"]
        ptrue_path = Path.home() / f"hallu_paper/mosaic_rebuttal/gpu1_cache/one_confidence_five_model_eval/{TAGS[cell]}-ptrue.json"
        if ptrue_path.exists():
            _DATA[key]["ptrue"] = ptrue_features(json.loads(ptrue_path.read_text())["records"], _DATA[key]["sample_ids"])
    return _DATA[key]


def ptrue_features(records: list, sample_ids) -> np.ndarray:
    """Self-verification (one teacher-forced judge prompt): log-odds of 'True' and the True/False log-prob gap."""
    by_id = {r["sample_id"]: r for r in records}
    p = np.clip([by_id[s]["p_true"] for s in sample_ids], 1e-9, 1 - 1e-9)
    gap = [by_id[s]["true_logp"] - by_id[s]["false_logp"] for s in sample_ids]
    return np.column_stack((np.log(p / (1 - p)), gap))


# ---------------------------------------------------------------- model pieces

def logistic(c: float, seed: int, scale: bool = True):
    steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
    if scale:
        steps.append(StandardScaler())
    steps.append(LogisticRegression(C=c, class_weight="balanced", max_iter=3000, solver="liblinear",
                                    random_state=seed))
    return make_pipeline(*steps)


def probe(cfg: dict, seed: int, n_train: int):
    steps = [StandardScaler()]
    if cfg["pca"]:
        steps.append(PCA(n_components=min(cfg["pca"], n_train - 1), random_state=seed))
    if cfg.get("probe_type", "logistic") == "lda":
        steps.append(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    else:
        steps.append(LogisticRegression(C=cfg["probe_c"], class_weight="balanced", max_iter=3000,
                                        solver="liblinear", random_state=seed))
    return make_pipeline(*steps)


# State-level cross-position contrasts: how the representation changes from reading the question to answering.
DERIVED_POSITIONS = {"delta_first": ("first_answer", "prompt_end"), "delta_mean": ("mean_answer", "prompt_end")}
POSITION_SEED = {name: i for i, name in enumerate((*orig.POSITIONS, *DERIVED_POSITIONS, "verify", "tokmax", "tokmean"))}


def state(d: dict, position: str, slot: int) -> np.ndarray:
    if position == "verify":  # self-verification states, stored at the 16 fixed slots only
        return d["state_verify"][:, d["verify_slot_index"][slot] if "verify_slot_index" in d else slot, :].astype(np.float32)
    if position in DERIVED_POSITIONS:
        a, b = DERIVED_POSITIONS[position]
        return d[f"state_{a}"][:, slot, :].astype(np.float32) - d[f"state_{b}"][:, slot, :].astype(np.float32)
    return d[f"state_{position}"][:, slot, :].astype(np.float32)


def folds(labels, groups, k, seed, context):
    return orig.grouped_folds(labels, groups, k, seed, context)


def layer_slots(d: dict, n: int) -> list[int]:
    return orig.fixed_layer_indices(d["state_mean_answer"].shape[1], n)


AUX_DIR = Path.home() / "hallu_paper/mosaic_rebuttal/aux_pool"
_AUX: dict[tuple, dict] = {}
AUX_LABELS = {"hc": {"HCE": 1, "HCC": 0}, "hcl": {"HCE": 1, "HCC": 0, "LCE": 1, "LCC": 0},
              "hce_lcc": {"HCE": 1, "HCC": 0, "LCC": 0}}


def aux_pool(d: dict, cfg: dict) -> dict:
    """Unmatched public rows of the same checkpoint used as extra probe training data (see extract_aux_pool.py).

    `aux`: which classes enter (hc = extra confident errors/corrects in every confidence stratum; hcl = plus the
    low-confidence ones); `aux_scope`: same dataset only or all datasets; `aux_hcc_ratio`: per (dataset, stratum)
    at most this many HCC per HCE, so the extra correct answers follow the confidence profile of the errors.
    Rows already in the matched benchmark are dropped.
    """
    key = (d["_key"], cfg["aux"], cfg.get("aux_scope", "same"), cfg.get("aux_hcc_ratio"), bool(cfg.get("pre_pca")))
    if key not in _AUX:
        cell = d["_key"].rsplit("-", 1)[0]
        z = np.load(AUX_DIR / f"{TAGS[cell]}.npz", allow_pickle=True)  # lazy: states are read only when indexed
        labels = AUX_LABELS[cfg["aux"]]
        scope = np.ones(len(z["class"]), bool)
        if cfg.get("aux_scope", "same") == "same":
            scope = z["dataset"] == cell.split("|")[1]
        keep = scope & np.isin(z["class"], list(labels)) & ~np.isin(z["sample_id"], d["sample_ids"])
        ratio = cfg.get("aux_hcc_ratio")
        if ratio:  # HCE counted before dropping matched rows (most HCE are already in the matched pairs)
            rng = np.random.default_rng(0)
            for ds in np.unique(z["dataset"]):
                for s in np.unique(z["stratum"][z["stratum"] >= 0]):
                    here = keep & (z["dataset"] == ds) & (z["stratum"] == s)
                    hcc = np.where(here & (z["class"] == "HCC"))[0]
                    cap = int(ratio * (scope & (z["dataset"] == ds) & (z["stratum"] == s) & (z["class"] == "HCE")).sum())
                    if len(hcc) > cap:
                        keep[rng.choice(hcc, len(hcc) - cap, replace=False)] = False
        idx = np.where(keep)[0]
        meta = {"idx": idx, "labels": np.array([labels[c] for c in z["class"][idx]]), "groups": z["fact_key_sha256"][idx],
                "splits": z["split"][idx], "slots": [int(s) for s in z["slots"]], "stratum": z["stratum"][idx]}
        if not cfg.get("pre_pca"):
            meta.update({f"state_{p}": z[f"state_{p}"][idx] for p in orig.POSITIONS})
            verify_path = Path.home() / f"hallu_paper/mosaic_rebuttal/verify_states/{TAGS[cell]}-aux.npz"
            if verify_path.exists():
                v = np.load(verify_path, allow_pickle=True)
                assert (v["sample_ids"] == z["sample_id"]).all()
                meta["state_verify"] = v["state_verify"][idx]
        _AUX[key] = meta
    return _AUX[key]


def aux_rows(d: dict, aux: dict, fit: np.ndarray) -> np.ndarray:
    """Auxiliary rows usable when fitting on matched rows `fit`: same splits as `fit`, and no fact shared with
    any matched row outside `fit` (held-out fold or evaluation rows)."""
    outside = np.ones(len(d["labels"]), bool)
    outside[fit] = False
    banned = set(d["groups"][outside].tolist())
    allowed = np.isin(aux["splits"], np.unique(d["splits"][fit]))
    return np.where(allowed & ~np.isin(aux["groups"], list(banned)))[0]


_AUX_RAW: dict = {}


def basis(d: dict, position: str, slot: int, dim: int):
    """Label-free PCA basis for one (position, layer slot), fitted once on the checkpoint's whole auxiliary pool
    (no matched or test rows, labels unused); probes then work in this `dim`-dimensional space."""
    cell = d["_key"].rsplit("-", 1)[0] if not d["_key"].endswith("-final-test") else d["_key"][:-len("-final-test")]
    index = layer_slots(d, 16).index(int(slot))
    path = CACHE / "basis" / f"{TAGS[cell]}-{position}-{index}-{dim}.npz"
    if not path.exists():
        if cell not in _AUX_RAW:
            z = np.load(AUX_DIR / f"{TAGS[cell]}.npz", allow_pickle=True)
            raw = {f"state_{p}": z[f"state_{p}"] for p in orig.POSITIONS}
            verify = Path.home() / f"hallu_paper/mosaic_rebuttal/verify_states/{TAGS[cell]}-aux.npz"
            if verify.exists():
                raw["state_verify"] = np.load(verify)["state_verify"]
            _AUX_RAW.clear()
            _AUX_RAW[cell] = raw
        x = state(_AUX_RAW[cell], position, index)
        pca = PCA(n_components=dim, svd_solver="randomized", random_state=0).fit(x)
        mean, comps = pca.mean_.astype(np.float32), pca.components_.astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, mean=mean, comps=comps, aux_proj=((x - mean) @ comps.T).astype(np.float32))
        os.replace(tmp, path)
    z = np.load(path)
    return z["mean"], z["comps"], z["aux_proj"]


def projected(d, cfg, position, slot, aux=None):
    """States of matched rows (and auxiliary rows) for one probe, optionally in the pre-PCA space."""
    x = state(d, position, slot)
    if cfg.get("pre_pca"):  # auxiliary rows come pre-projected (aux["idx"] indexes the full pool file)
        mean, comps, aux_proj = basis(d, position, slot, cfg["pre_pca"])
        return (x - mean) @ comps.T, None if aux is None else aux_proj[aux["idx"]]
    return x, None if aux is None else state(aux, position, aux["slots"].index(int(slot)))


def token_probe_column(d, cfg, position, slot, train, test, fold_list, seed):
    """Multi-instance probe over the first answer tokens: every answer token is a training instance carrying its
    row's label; a row's risk is the max (tokmax) or mean (tokmean) of its tokens' probe scores."""
    key = hashlib.sha1(json.dumps(["tok", d["_key"], int(slot), cfg["pca"], cfg.get("probe_type"), cfg["base_folds"],
                                   seed]).encode() + train.tobytes() + test.tobytes()).hexdigest()
    path = CACHE / "tokprobes" / f"{key}.npz"
    agg = "max" if position == "tokmax" else "mean"
    if path.exists():
        try:
            z = np.load(path)
            return z[f"oof_{agg}"], z[f"full_{agg}"]
        except Exception:
            pass
    index = layer_slots(d, 16).index(int(slot))
    x = d["tok_states"][:, index].astype(np.float32)            # (n, T, dim)
    mask = d["tok_mask"]
    y = d["labels"]
    s0 = seed + 97 * int(slot) + 7919 * POSITION_SEED["tokmax"]

    def fit(rows, s):
        m = mask[rows]
        model = probe(cfg, s, int(m.sum())).fit(x[rows][m], np.repeat(y[rows], m.sum(1)))

        def score(eval_rows):
            p = model.predict_proba(x[eval_rows].reshape(-1, x.shape[-1]))[:, 1].reshape(len(eval_rows), -1)
            em = mask[eval_rows]
            return np.where(em, p, -np.inf).max(1), np.where(em, p, 0).sum(1) / em.sum(1)
        return score

    out = {"oof_max": np.zeros(len(train)), "oof_mean": np.zeros(len(train))}
    for k, (tr, ho) in enumerate(fold_list):
        out["oof_max"][ho], out["oof_mean"][ho] = fit(train[tr], s0 + k)(train[ho])
    out["full_max"], out["full_mean"] = fit(train, s0)(test)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, path)
    return out[f"oof_{agg}"], out[f"full_{agg}"]


def probe_column(d, cfg, position, slot, train, test, fold_list, seed):
    """OOF scores on `train` and full-fit scores on `test` for one (position, layer) probe, cached on disk.

    The result depends only on the data, the split, the probe settings and the seed, so every configuration
    that shares them reuses the same file.
    """
    if position in ("tokmax", "tokmean"):
        return token_probe_column(d, cfg, position, slot, train, test, fold_list, seed)
    probe_key = [d["_key"], position, int(slot), cfg["pca"], cfg["probe_c"], cfg["base_folds"], seed]
    if cfg.get("probe_type", "logistic") != "logistic":  # logistic keys stay compatible with earlier cache files
        probe_key = probe_key + [cfg["probe_type"]] if cfg["probe_type"] != "lda" else probe_key[:4] + probe_key[5:] + ["lda"]
    if cfg.get("aux"):
        probe_key += ["aux", cfg["aux"], cfg.get("aux_scope", "same"), cfg.get("aux_hcc_ratio")]
    if cfg.get("stratum_center"):
        probe_key += ["stratum_center"]
    if cfg.get("pre_pca"):
        probe_key += ["pre_pca", cfg["pre_pca"]]
    key = hashlib.sha1(json.dumps(probe_key).encode() + train.tobytes() + test.tobytes()).hexdigest()
    path = CACHE / "probes" / f"{key}.npz"
    fold_avg = cfg.get("probe_fold_avg", False)
    if path.exists():
        try:
            z = np.load(path)
            if not fold_avg:
                return z["oof"], z["full"]
            if "foldavg" in z.files:
                return z["oof"], z["foldavg"]
        except Exception:
            pass
    y = d["labels"]
    probe_seed = seed + 97 * int(slot) + 7919 * POSITION_SEED[position]
    aux = aux_pool(d, cfg) if cfg.get("aux") else None
    x, ax = projected(d, cfg, position, slot, aux)
    xs, ys, ss = [x], [y], [d["stratum"] if cfg.get("stratum_center") else None]
    if aux is not None:
        xs.append(ax); ys.append(aux["labels"]); ss.append(aux["stratum"])

    def fit(rows, s):
        """Probe on matched rows (+ usable auxiliary rows). With stratum_center, every row is centred on the mean
        of its confidence stratum among the fitting rows (confidence-conditional probe: the label-independent
        shift tied to the confidence level is removed); returns a scoring function for matched rows."""
        parts = [(0, rows)]
        if cfg.get("aux"):
            parts.append((1, aux_rows(d, aux, rows)))
        fx = np.concatenate([xs[i][r] for i, r in parts])
        fy = np.concatenate([ys[i][r] for i, r in parts])
        if not cfg.get("stratum_center"):
            model = probe(cfg, s, len(fx)).fit(fx, fy)
            return lambda r: model.predict_proba(x[r])[:, 1]
        fs = np.concatenate([ss[i][r] for i, r in parts])
        means = {v: fx[fs == v].mean(0) for v in np.unique(fs)}
        overall = fx.mean(0)
        shift = lambda strata: np.stack([means.get(v, overall) for v in strata])
        model = probe(cfg, s, len(fx)).fit(fx - shift(fs), fy)
        return lambda r: model.predict_proba(x[r] - shift(d["stratum"][r]))[:, 1]

    oof = np.zeros(len(train))
    foldavg = np.zeros(len(test))
    for k, (tr, ho) in enumerate(fold_list):
        score = fit(train[tr], probe_seed + k)
        oof[ho] = score(train[ho])
        foldavg += score(test) / len(fold_list)
    full = fit(train, probe_seed)(test)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, oof=oof, full=full, foldavg=foldavg)
    os.replace(tmp, path)
    # Cross-fit averaging: test scores from the same fold models that produced the OOF training scores.
    return oof, (foldavg if fold_avg else full)


def prior_column(d, cfg, position, slot, train, test, fold_list, seed):
    """General error direction learned on the auxiliary pool only (errors vs correct answers at any confidence,
    every dataset of the checkpoint), applied to matched rows. Training rows get scores from fits that exclude
    the facts of their base fold, evaluation rows from a fit that excludes every non-training fact."""
    pcfg = dict(cfg, aux="hcl", aux_scope="all", aux_hcc_ratio=None)
    key = hashlib.sha1(json.dumps(["prior", d["_key"], position, int(slot), cfg["pca"], cfg.get("probe_type"),
                                   cfg["base_folds"], seed, cfg.get("pre_pca")]).encode() + train.tobytes() + test.tobytes()).hexdigest()
    path = CACHE / "prior" / f"{key}.npz"
    if path.exists():
        try:
            z = np.load(path)
            return z["oof"], z["full"]
        except Exception:
            pass
    aux = aux_pool(d, pcfg)
    x, ax = projected(d, cfg, position, slot, aux)
    ay = aux["labels"]
    s0 = seed + 97 * int(slot) + 7919 * POSITION_SEED[position] + 5

    def fit(rows):
        extra = aux_rows(d, aux, rows)
        return probe(cfg, s0, len(extra)).fit(ax[extra], ay[extra])

    oof = np.zeros(len(train))
    for tr, ho in fold_list:
        oof[ho] = fit(train[tr]).predict_proba(x[train[ho]])[:, 1]
    full = fit(train).predict_proba(x[test])[:, 1]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, oof=oof, full=full)
    os.replace(tmp, path)
    return oof, full


def internal_features(raw: np.ndarray, cfg: dict) -> np.ndarray:
    if cfg.get("aux_prior"):
        cfg = dict(cfg, positions=list(cfg["positions"]) * 2, aux_prior=False)
    width = raw.shape[1] // len(cfg["positions"])
    parts, late, peak = [], [], []
    values = np.clip(raw, 1e-6, 1 - 1e-6)
    if cfg["probe_logit"]:
        values = np.log(values / (1 - values))
    for p in range(len(cfg["positions"])):
        trajectory = values[:, p * width:(p + 1) * width]
        summary = orig.trajectory_summary(trajectory)
        parts.append(summary)
        late.append(summary[:, 2])
        peak.append(summary[:, 3])
        if cfg["raw_slots"]:
            parts.append(trajectory)
    late, peak = np.column_stack(late), np.column_stack(peak)
    parts.append(np.column_stack((late.max(1) - late.min(1), late.std(1), peak.max(1) - peak.min(1))))
    return np.column_stack(parts)


def base_crossfit(d, cfg, train, test, seed):
    """OOF probe scores for `train` rows and full-fit probe scores for `test` rows."""
    fold_list = folds(d["labels"][train], d["groups"][train], cfg["base_folds"], seed, "base")
    columns = [probe_column(d, cfg, position, slot, train, test, fold_list, seed)
               for position in cfg["positions"] for slot in layer_slots(d, cfg["layers"])]
    oof, out = np.column_stack([c[0] for c in columns]), np.column_stack([c[1] for c in columns])
    if cfg.get("aux_prior"):  # appended as extra positions of the internal trajectory features
        prior = [prior_column(d, cfg, position, slot, train, test, fold_list, seed)
                 for position in cfg["positions"] for slot in layer_slots(d, cfg["layers"])]
        oof = np.column_stack([oof] + [c[0] for c in prior])
        out = np.column_stack([out] + [c[1] for c in prior])
    return oof, out


def behavioral_matrix(d, cfg, rows):
    f = d["features"]
    names = d["feature_names"]
    behavioral_idx = d["behavioral_idx"]
    if cfg["behavior_set"] == "no_natural":
        behavioral_idx = np.array([i for i in behavioral_idx if "natural" not in names[i]])
    elif cfg["behavior_set"] == "no_nli":
        behavioral_idx = np.array([i for i in behavioral_idx if not names[i].startswith("nli.")])
    behavioral = f[rows][:, behavioral_idx]
    source = cfg.get("behavior_source", "panel")
    if source == "template":  # generation-free: fixed wrappers only, one teacher-forced pass each
        behavioral = d["tpl_basic"][rows]
    elif source == "both":
        behavioral = np.column_stack((behavioral, d["tpl_basic"][rows]))
    if cfg.get("verify") in ("behavioral", "both"):
        behavioral = np.column_stack((behavioral, d["ptrue"][rows]))
    if cfg.get("verify2"):  # first-token True/False log-odds under both judge prompts and their disagreement
        o = d["verify_odds"][rows]
        behavioral = np.column_stack((behavioral, o, o[:, 0] - o[:, 1]))
    if cfg["behavior_candidate"] and source != "template":
        behavioral = np.column_stack((behavioral, f[rows][:, d["candidate_idx"]]))
    return behavioral


_BSTATS: dict = {}


def behavioral_z(d, cfg, rows):
    """Behavioral features standardized with the checkpoint's own public-row statistics (label-free), so that
    checkpoints share one feature scale for the pooled behavioral expert."""
    key = (d["_key"], cfg.get("behavior_source"), cfg["behavior_set"], cfg["behavior_candidate"], cfg.get("verify"), cfg.get("verify2"))
    if key not in _BSTATS:
        public = np.where(np.isin(d["splits"], ("development", "validation")))[0]
        b = behavioral_matrix(d, cfg, public).astype(float)
        _BSTATS[key] = (np.nanmean(b, 0), np.nanstd(b, 0) + 1e-9)
    mu, sd = _BSTATS[key]
    return (behavioral_matrix(d, cfg, rows) - mu) / sd


def pooled_behavioral(d, cfg, fit_rows):
    """Other checkpoints' public rows for the behavioral expert: same splits as `fit_rows`, no fact shared with
    any of this checkpoint's rows outside `fit_rows`."""
    outside = np.ones(len(d["labels"]), bool)
    outside[fit_rows] = False
    banned = list(set(d["groups"][outside].tolist()))
    splits = np.unique(d["splits"][fit_rows])
    xs, ys = [], []
    for cell in CELLS:
        o = load(cell, cfg["source"])
        if d["_key"].startswith(cell + "-"):
            continue
        rows = np.where(np.isin(o["splits"], splits) & ~np.isin(o["groups"], banned))[0]
        xs.append(behavioral_z(o, cfg, rows)); ys.append(o["labels"][rows])
    return np.concatenate(xs), np.concatenate(ys)


def expert_matrices(d, cfg, rows, probe_raw):
    f = d["features"]
    behavioral = behavioral_z(d, cfg, rows) if cfg.get("pool_behavior") else behavioral_matrix(d, cfg, rows)
    internal = [internal_features(probe_raw, cfg)]
    if cfg["internal_candidate"]:
        internal.append(f[rows][:, d["candidate_idx"]])
    if cfg["internal_controls"]:
        internal.append(d["controls"][rows])
    if cfg.get("verify") in ("internal", "both"):
        internal.append(d["ptrue"][rows])
    persistence = f[rows][:, d["persistence_idx"]]
    if cfg.get("xq", "none") != "none":
        if cfg.get("xq_source", "panel") == "template":
            blocks = d["tpl_xq"][rows].reshape(len(rows), 9, 16)
            xq = (d["tpl_xq"][rows] if cfg.get("xq_reduce") == "raw"
                  else np.column_stack([orig.trajectory_summary(blocks[:, b, :]) for b in range(9)]))
        else:
            xq = xq_features(d["xq"][rows], cfg)
        if cfg["xq"] in ("internal", "both"):
            internal.append(xq)
        if cfg["xq"] in ("behavioral", "both"):
            behavioral = np.column_stack((behavioral, xq))
    return {"behavioral": behavioral, "internal": np.column_stack(internal), "persistence": persistence}


def xq_features(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Representation stability under equivalent questions: 3 positions x 4 statistics x 16 slots."""
    if cfg.get("xq_reduce", "summary") == "raw":
        return x
    blocks = x.reshape(len(x), 12, 16)
    return np.column_stack([orig.trajectory_summary(blocks[:, b, :]) for b in range(12)])


LENS_CHANNELS = {"all": [0, 1, 2, 3, 4, 5, 6, 7], "lens": [0, 1, 2, 3], "attn": [4, 5, 6, 7]}
VIEWS = ("behavioral", "internal", "persistence")
COMBOS = {"mosaic": VIEWS, "I+B": ("internal", "behavioral"), "I+P": ("internal", "persistence"),
          "B+P": ("behavioral", "persistence"), "internal_plus_behavior": ("internal", "behavioral")}
EVIDENCE_COMBOS = ("mosaic", "internal_plus_behavior")


def shrink_gate_fit(base, extra, y, lam):
    """Logistic gate with the anchor view's logit `base` and extra evidence logits `extra`; the L2 penalty `lam`
    acts on the extra-evidence weights only, so lam -> inf falls back to the anchor view. Newton/IRLS, class-balanced."""
    x = np.column_stack((np.ones(len(y)), base, extra))
    w = np.zeros(x.shape[1])
    sw = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))
    pen = np.r_[0.0, 1e-6, np.full(extra.shape[1], lam)] * len(y)
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
        grad = x.T @ (sw * (p - y)) + pen * w
        hess = (x * (sw * p * (1 - p))[:, None]).T @ x + np.diag(pen + 1e-9)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def shrink_gate(base_tr, extra_tr, y, groups, base_te, extra_te, seed, lams=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 1e6)):
    """Choose the extra-evidence penalty by fact-grouped inner CV (AUROC of the gate's held-out scores), refit."""
    inner = folds(y, groups, 5, seed + 13, "shrink")
    best, best_lam = -1.0, lams[-1]
    for lam in lams:
        s = np.zeros(len(y))
        for tr, ho in inner:
            w = shrink_gate_fit(base_tr[tr], extra_tr[tr], y[tr], lam)
            s[ho] = np.column_stack((np.ones(len(ho)), base_tr[ho], extra_tr[ho])) @ w
        auc = metrics(y, s)["auroc"]
        if auc > best + 1e-9:
            best, best_lam = auc, lam
    w = shrink_gate_fit(base_tr, extra_tr, y, best_lam)
    return np.column_stack((np.ones(len(base_te)), base_te, extra_te)) @ w, best_lam


def gate_input(scores: np.ndarray, cfg: dict) -> np.ndarray:
    scores = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(scores / (1 - scores)) if cfg["gate_logit"] else scores


def fit_predict(d, cfg, train, test, seed):
    """Full pipeline: train on `train` rows, return per-method scores on `test` rows."""
    labels = d["labels"]
    base_oof, base_test = base_crossfit(d, cfg, train, test, seed)
    x_train = expert_matrices(d, cfg, train, base_oof)
    x_test = expert_matrices(d, cfg, test, base_test)
    if cfg.get("union_preset") is not None:
        # Second probe family (e.g. the paper's logistic trajectory); the regularized internal expert weighs both.
        other = dict(cfg, **PROBE_PRESETS[cfg["union_preset"]])
        other_oof, other_test = base_crossfit(d, other, train, test, seed)
        x_train["internal"] = np.column_stack((x_train["internal"], internal_features(other_oof, other)))
        x_test["internal"] = np.column_stack((x_test["internal"], internal_features(other_test, other)))
    if cfg.get("verify_view"):
        # Self-verification view (replaces persistence): trajectory of probes on the states at the judge prompt's
        # answer position, plus the judge's own True/False log-odds (P(True) prompt and the paraphrased judge).
        vcfg = dict(cfg, positions=["verify"], aux=None, aux_prior=False, pre_pca=None, stratum_center=False)
        v_oof, v_test = base_crossfit(d, vcfg, train, test, seed)
        x_train["persistence"] = np.column_stack((internal_features(v_oof, vcfg), d["ptrue"][train], d["verify_odds"][train]))
        x_test["persistence"] = np.column_stack((internal_features(v_test, vcfg), d["ptrue"][test], d["verify_odds"][test]))
    if cfg.get("lens_view"):
        # Generation-free mechanistic view (replaces persistence): logit-lens answer emergence and attention to the
        # question across depth, per channel summarized as a trajectory (optionally with the raw 16 slots).
        chans = LENS_CHANNELS[cfg["lens_view"]]
        def lens_matrix(rows):
            x = d["lens"][rows][:, chans, :]
            parts = [orig.trajectory_summary(x[:, c, :]) for c in range(len(chans))]
            if cfg.get("lens_raw"):
                parts.append(x.reshape(len(rows), -1))
            return np.column_stack(parts)
        x_train["persistence"], x_test["persistence"] = lens_matrix(train), lens_matrix(test)
        if cfg.get("lens_into"):  # or append to an existing view instead
            x_train[cfg["lens_into"]] = np.column_stack((x_train[cfg["lens_into"]], x_train["persistence"]))
            x_test[cfg["lens_into"]] = np.column_stack((x_test[cfg["lens_into"]], x_test["persistence"]))
    if cfg.get("perturb"):
        # Intervention evidence: answer log-prob drop / first-token flips under residual-stream noise.
        target = "persistence" if cfg["perturb"] == "view" else cfg["perturb"]
        if cfg["perturb"] == "view" and not cfg.get("lens_view"):
            x_train["persistence"], x_test["persistence"] = d["perturb"][train], d["perturb"][test]
        else:
            x_train[target] = np.column_stack((x_train[target], d["perturb"][train]))
            x_test[target] = np.column_stack((x_test[target], d["perturb"][test]))
    if cfg.get("verify_in_behavior"):
        # Self-verification states as behavioral evidence: compact trajectory summary (no raw slots) of the
        # judge-position probes, appended to the behavioral expert next to P(True).
        vcfg = dict(cfg, positions=["verify"], aux=None, aux_prior=False, pre_pca=None, stratum_center=False,
                    raw_slots=cfg["verify_in_behavior"] == "raw")
        v_oof, v_test = base_crossfit(d, vcfg, train, test, seed)
        x_train["behavioral"] = np.column_stack((x_train["behavioral"], internal_features(v_oof, vcfg)))
        x_test["behavioral"] = np.column_stack((x_test["behavioral"], internal_features(v_test, vcfg)))
    if cfg.get("xfer", "none") != "none":
        t_train, t_test = transfer_features(d, cfg, train, test, seed)
        view = "behavioral" if cfg["xfer"] == "behavioral" else "internal"
        x_train[view] = np.column_stack((x_train[view], t_train))
        x_test[view] = np.column_stack((x_test[view], t_test))
    out = {}
    experts_test = {}

    boost_w = None
    if cfg.get("boost"):
        # Complementary experts: rows the internal expert gets wrong (cross-fitted within the training rows) are
        # up-weighted when fitting the other experts, so they learn what the internal view misses.
        p_int = np.zeros(len(train))
        for k, (tr, ho) in enumerate(folds(labels[train], d["groups"][train], cfg["gate_folds"], seed + 7, "gate")):
            p_int[ho] = logistic(cfg["expert_c"], seed + k).fit(x_train["internal"][tr], labels[train][tr]).predict_proba(
                x_train["internal"][ho])[:, 1]
        boost_w = 1 + cfg["boost"] * np.abs(labels[train] - p_int)

    def expert(view, idx, s):
        """Expert for `view` fit on train[idx]; the pooled behavioral expert adds other checkpoints' rows."""
        x, y = x_train[view][idx], labels[train][idx]
        if view == "persistence" and cfg.get("p_model") == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0,
                                                  min_samples_leaf=20, class_weight="balanced", random_state=s).fit(
                x, y, sample_weight=None if boost_w is None else boost_w[idx])
        if boost_w is not None and view != "internal":
            return logistic(cfg["expert_c"], s).fit(x, y, logisticregression__sample_weight=boost_w[idx])
        if view == "behavioral" and cfg.get("pool_behavior"):
            px, py = pooled_behavioral(d, cfg, train[idx])
            w = np.r_[np.ones(len(y)), np.full(len(py), cfg.get("pool_weight", 1.0))]
            return logistic(cfg["expert_c"], s).fit(np.concatenate([x, px]), np.r_[y, py], logisticregression__sample_weight=w)
        return logistic(cfg["expert_c"], s).fit(x, y)

    everything = np.arange(len(train))
    for view in VIEWS:
        model = expert(view, everything, seed)
        experts_test[view] = model.predict_proba(x_test[view])[:, 1]
    # OOF expert scores for the gate (fact-grouped, same depth as the paper's cross-fit).
    # Repeated fact-grouped cross-fitting: average OOF expert scores over several fold draws (less gate noise).
    oof = {view: np.zeros(len(train)) for view in VIEWS}
    fold_test = {view: np.zeros(len(test)) for view in VIEWS}
    repeats = cfg.get("gate_repeats", 1)
    for r in range(repeats):
        gate_folds = folds(labels[train], d["groups"][train], cfg["gate_folds"], seed + 7 + 101 * r, "gate")
        for k, (tr, ho) in enumerate(gate_folds):
            for view in VIEWS:
                model = expert(view, tr, seed + k)
                oof[view][ho] += model.predict_proba(x_train[view][ho])[:, 1] / repeats
                fold_test[view] += model.predict_proba(x_test[view])[:, 1] / (repeats * len(gate_folds))
    if cfg.get("expert_fold_avg", False):
        experts_test = fold_test  # same models (and score distribution) as the gate's training inputs
    for view in VIEWS:
        out[view] = experts_test[view]
    evidence_train = d["features"][train][:, d["evidence_idx"]]
    evidence_test = d["features"][test][:, d["evidence_idx"]]
    for name, views in COMBOS.items():
        g_train = gate_input(np.column_stack([oof[v] for v in views]), cfg)
        g_test = gate_input(np.column_stack([experts_test[v] for v in views]), cfg)
        with_evidence = name == "internal_plus_behavior" or (cfg["gate_evidence"] and name == "mosaic")
        if with_evidence:
            g_train = np.column_stack((g_train, evidence_train))
            g_test = np.column_stack((g_test, evidence_test))
        gate = logistic(cfg["gate_c"], seed, scale=with_evidence).fit(g_train, labels[train])
        out[name] = gate.predict_proba(g_test)[:, 1]
    if cfg.get("shrink_gate"):
        # Evidence-shrinkage gate: internal view as anchor, extra views' weights shrunk by an inner-CV penalty.
        lg = lambda v: gate_input(v, {"gate_logit": True})
        for name, extra_views in (("I+B_shrink", ("behavioral",)), ("mosaic_shrink", ("behavioral", "persistence"))):
            s_te, lam = shrink_gate(lg(oof["internal"]), np.column_stack([lg(oof[v]) for v in extra_views]),
                                    labels[train], d["groups"][train], lg(experts_test["internal"]),
                                    np.column_stack([lg(experts_test[v]) for v in extra_views]), seed)
            out[name] = 1 / (1 + np.exp(-np.clip(s_te, -30, 30)))
    if cfg.get("gate_agree"):
        # Agreement-aware gate for I+B: standardized expert logits plus their product, absolute gap, max and min,
        # so the gate can weigh the views differently when they disagree on a row.
        def agree(scores, ref):
            zs = []
            for v in ("internal", "behavioral"):
                lo = gate_input(ref[v], {"gate_logit": True})
                zs.append((gate_input(scores[v], {"gate_logit": True}) - lo.mean()) / (lo.std() + 1e-9))
            zi, zb = zs
            return np.column_stack((zi, zb, zi * zb, np.abs(zi - zb), np.maximum(zi, zb), np.minimum(zi, zb)))
        a_train, a_test = agree(oof, oof), agree(experts_test, oof)
        gate = logistic(cfg["gate_c"], seed).fit(a_train, labels[train])
        out["I+B_agree"] = gate.predict_proba(a_test)[:, 1]
        gate = logistic(cfg["gate_c"], seed).fit(np.column_stack((a_train, evidence_train)), labels[train])
        out["I+B_agree_ev"] = gate.predict_proba(np.column_stack((a_test, evidence_test)))[:, 1]
    # Untrained fusion: each expert's logit standardized with its OOF training distribution, then averaged.
    z = {}
    for view in VIEWS:
        lo = gate_input(oof[view], {"gate_logit": True})
        z[view] = (gate_input(experts_test[view], {"gate_logit": True}) - lo.mean()) / (lo.std() + 1e-9)
    out["zavg_IB"] = (z["internal"] + z["behavioral"]) / 2
    out["zavg_IP"] = (z["internal"] + z["persistence"]) / 2
    out["zavg_IBP"] = (z["internal"] + z["behavioral"] + z["persistence"]) / 3
    flat = logistic(cfg["expert_c"], seed).fit(np.column_stack([x_train[v] for v in VIEWS]), labels[train])
    out["flat"] = flat.predict_proba(np.column_stack([x_test[v] for v in VIEWS]))[:, 1]
    return out


# ---------------------------------------------------------------- metrics

def roc(y, s):
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    idx = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[idx]
    fps = 1 + idx - tps
    return np.r_[0, fps] / max((1 - y).sum(), 1), np.r_[0, tps] / max(y.sum(), 1)


def metrics(y, s):
    fpr, tpr = roc(y, s)
    stop = np.searchsorted(fpr, MAX_FPR, side="right")
    x = np.append(fpr[:stop], MAX_FPR)
    t = np.append(tpr[:stop], np.interp(MAX_FPR, fpr, tpr))
    return {"r10": float(tpr[fpr <= MAX_FPR + 1e-12].max()), "pauc": float(np.trapezoid(t, x)),
            "auroc": float(np.trapezoid(tpr, fpr))}


def macro(per_cell: dict) -> dict:
    methods = next(iter(per_cell.values())).keys()
    return {m: {k: float(np.mean([per_cell[c][m][k] for c in per_cell])) for k in ("r10", "pauc", "auroc")}
            for m in methods}


# ---------------------------------------------------------------- evaluation protocols

def eval_validation(cfg, seed=2026103101):
    per_cell = {}
    for cell in CELLS:
        d = load(cell, cfg["source"])
        train = np.where(d["splits"] == "development")[0]
        test = np.where(d["splits"] == "validation")[0]
        scores = fit_predict(d, cfg, train, test, seed)
        per_cell[cell] = {m: metrics(d["labels"][test], s) for m, s in scores.items()}
    return {"per_cell": per_cell, "macro": macro(per_cell)}


def eval_cv_cell(cfg, cell, repeats=1, k=5, seed=2026103102, keep_scores=False):
    d = load(cell, cfg["source"])
    rows = np.arange(len(d["labels"]))
    acc, kept = [], None
    for r in range(repeats):
        outer = folds(d["labels"], d["groups"], k, seed + r, f"outer{r}")
        pooled = {}
        for tr, ho in outer:
            scores = fit_predict(d, cfg, rows[tr], rows[ho], seed + 31 * r + 1009 * cfg.get("inner_seed", 0))
            for m, s in scores.items():
                pooled.setdefault(m, np.zeros(len(rows)))[ho] = s
        acc.append({m: metrics(d["labels"], s) for m, s in pooled.items()})
        if keep_scores and r == 0:
            kept = {"labels": d["labels"].tolist(), "groups": d["groups"].tolist(),
                    **{m: s.tolist() for m, s in pooled.items() }}
    per_cell = {m: {key: float(np.mean([a[m][key] for a in acc])) for key in ("r10", "pauc", "auroc")} for m in acc[0]}
    return per_cell, kept


def eval_cv(cfg, repeats=1, k=5, seed=2026103102, keep_scores=False):
    """Repeated fact-grouped CV over development+validation (stability check for top configs)."""
    per_cell = {}
    scores_out = {}
    for cell in CELLS:
        per_cell[cell], scores_out[cell] = eval_cv_cell(cfg, cell, repeats, k, seed, keep_scores)
    result = {"per_cell": per_cell, "macro": macro(per_cell)}
    if keep_scores:
        result["scores"] = scores_out
    return result


def _cv_cell_job(job):
    cfg, cell, seed = job
    try:
        return cfg, cell, seed, eval_cv_cell(cfg, cell, seed=seed, keep_scores=True), None
    except Exception:
        return cfg, cell, seed, None, traceback.format_exc()[-1500:]


def cvgrid(cfgs: list, seeds: list, workers: int) -> None:
    """Same as cvmanual, parallel over (config, cell, seed); writes one stage2_manual record per (config, seed)."""
    import multiprocessing as mp
    jobs = [(cfg, cell, seed) for cfg in cfgs for seed in seeds for cell in CELLS]
    jobs.sort(key=lambda j: CELLS.index(j[1]) in (3, 4), reverse=True)  # big cells first
    done: dict = {}
    with mp.get_context("fork").Pool(workers, maxtasksperchild=1) as pool, (OUT / "stage2_manual.jsonl").open("a") as handle:
        for cfg, cell, seed, result, error in pool.imap_unordered(_cv_cell_job, jobs):
            if error:
                print("ERROR", cfg_id(cfg), cell, error, flush=True)
                continue
            entry = done.setdefault((cfg_id(cfg), seed), {"cfg": cfg, "per_cell": {}, "scores": {}})
            entry["per_cell"][cell], entry["scores"][cell] = result
            if len(entry["per_cell"]) == len(CELLS):
                per_cell = {c: entry["per_cell"][c] for c in CELLS}
                cv = {"per_cell": per_cell, "macro": macro(per_cell), "scores": {c: entry["scores"][c] for c in CELLS}}
                handle.write(json.dumps({"id": cfg_id(cfg), "cfg": cfg, "seed": seed, "cv": cv}) + "\n"); handle.flush()
                m = cv["macro"]
                print(time.strftime("%H:%M:%S"), cfg_id(cfg), cfg.get("label", ""), f"seed={seed}",
                      {k: round(v["r10"], 4) for k, v in m.items()},
                      "| I+B per cell", [round(per_cell[c]["I+B"]["r10"], 3) for c in CELLS], flush=True)


# ---------------------------------------------------------------- search

SPACE = {
    "layers": [8, 16, 32],
    "probe_type": ["logistic", "logistic", "lda"],
    "positions": [("prompt_end", "first_answer", "mean_answer"), ("first_answer", "mean_answer"),
                  ("prompt_end", "first_answer", "mean_answer", "delta_first"),
                  ("first_answer", "mean_answer", "delta_first", "delta_mean"),
                  ("prompt_end", "mean_answer")],
    "pca": [0, 32, 64, 128, 256],
    "probe_c": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    "probe_logit": [False, True],
    "raw_slots": [False, True],
    "internal_candidate": [True, False],
    "internal_controls": [False, True],
    "behavior_set": ["all", "no_natural", "no_nli"],
    "behavior_candidate": [False, True],
    "expert_c": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    "gate_c": [0.01, 0.03, 0.1, 0.3, 1.0, 10.0],
    "gate_logit": [False, True],
    "gate_evidence": [False, True],
    "base_folds": [4, 5],
    "gate_folds": [4, 5],
    "xq": ["none", "internal", "behavioral", "both"],
    "gate_repeats": [1, 3, 5],
    "behavior_source": ["panel", "template", "both"],
    "probe_fold_avg": [False, True],
    "expert_fold_avg": [False, True],
    "xq_source": ["panel", "template"],
    "xq_reduce": ["summary", "raw"],
}
RUN6_LIKE = {"layers": 16, "positions": ("prompt_end", "first_answer", "mean_answer"), "pca": 0, "probe_c": 0.01,
             "probe_logit": False, "raw_slots": True, "internal_candidate": True, "internal_controls": True,
             "behavior_set": "all", "behavior_candidate": False, "expert_c": 0.01, "gate_c": 0.01,
             "gate_logit": False, "gate_evidence": False, "base_folds": 4, "gate_folds": 4}


def cfg_id(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def xq_ready() -> bool:
    return all((Path.home() / f"hallu_paper/mosaic_rebuttal/panel_states/{t}-xq-features.npz").exists()
               for t in TAGS.values())


def tpl_ready() -> bool:
    return all((Path.home() / f"hallu_paper/mosaic_rebuttal/template_panel/{t}-public.npz").exists()
               for t in TAGS.values())


def sample_cfg(rng, sources):
    cfg = {k: rng.choice(v) for k, v in SPACE.items()}
    cfg["source"] = rng.choice(sources)
    if not xq_ready():
        cfg["xq"] = "none"
    if not tpl_ready():
        cfg["behavior_source"] = "panel"
        cfg["xq_source"] = "panel"
    if cfg["xq"] == "none":
        cfg.pop("xq_reduce")
    return cfg


def run_one(cfg):
    start = time.time()
    try:
        result = eval_validation(cfg)
        return {"id": cfg_id(cfg), "cfg": cfg, "val": result, "secs": time.time() - start}
    except Exception:
        return {"id": cfg_id(cfg), "cfg": cfg, "error": traceback.format_exc()[-2000:], "secs": time.time() - start}


def available_sources():
    return [s for s in SOURCES if all((CACHE / f"{TAGS[c]}-{s}.npz").exists() for c in CELLS)]


def search(hours: float, workers: int, seed: int) -> None:
    import multiprocessing as mp
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "stage1.jsonl"
    done = set()
    if log.exists():
        done = {json.loads(line)["id"] for line in log.open()}
    deadline = time.time() + hours * 3600
    rng = random.Random(seed + len(done))
    for source in available_sources():  # loaded once in the parent, shared copy-on-write by forked workers
        for cell in CELLS:
            load(cell, source)

    def jobs():
        for source in available_sources():
            cfg = dict(RUN6_LIKE, source=source)
            if cfg_id(cfg) not in done:
                yield cfg
        while time.time() < deadline:
            # Half explore (random), half exploit (mutate 1-3 settings of a current top-5 configuration).
            cfg = sample_cfg(rng, available_sources())
            top = leaderboard("mosaic", top=5) if log.exists() and log.stat().st_size else []
            if top and rng.random() < 0.5:
                cfg = dict(rng.choice(top)["cfg"])
                for key in rng.sample(sorted(SPACE), rng.randint(1, 3)):
                    cfg[key] = rng.choice(SPACE[key])
                if cfg.get("xq", "none") == "none":
                    cfg.pop("xq_reduce", None)
                elif "xq_reduce" not in cfg:
                    cfg["xq_reduce"] = rng.choice(SPACE["xq_reduce"])
                cfg["positions"] = tuple(cfg["positions"])
            if cfg_id(cfg) not in done:
                done.add(cfg_id(cfg))
                yield cfg

    with mp.get_context("fork").Pool(workers, maxtasksperchild=20) as pool, log.open("a") as handle:
        for record in pool.imap_unordered(run_one, jobs()):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if "val" in record:
                m = record["val"]["macro"]
                print(f"{time.strftime('%H:%M:%S')} {record['id']} {record['cfg']['source']:6s} "
                      f"mosaic r10={m['mosaic']['r10']:.4f} pauc={m['mosaic']['pauc']:.4f} "
                      f"internal r10={m['internal']['r10']:.4f} secs={record['secs']:.0f}", flush=True)
            else:
                print("ERROR", record["id"], record["error"].splitlines()[-1], flush=True)
            if time.time() > deadline:
                break


def leaderboard(method="mosaic", key="r10", top=20):
    rows = [json.loads(l) for name in ("stage1.jsonl", "stage1_manual.jsonl") if (OUT / name).exists()
            for l in (OUT / name).open()]
    rows = [r for r in rows if "val" in r]
    rows.sort(key=lambda r: (-r["val"]["macro"][method][key], -r["val"]["macro"][method]["pauc"]))
    return rows[:top]


def stage2(top: int, workers: int) -> None:
    import multiprocessing as mp
    candidates = {r["id"]: r["cfg"] for r in leaderboard("mosaic", top=top)}
    candidates.update({r["id"]: r["cfg"] for r in leaderboard("internal", top=max(5, top // 3))})
    log = OUT / "stage2.jsonl"
    done = {json.loads(l)["id"] for l in log.open()} if log.exists() else set()
    todo = [c for i, c in candidates.items() if i not in done]
    with mp.get_context("fork").Pool(workers) as pool, log.open("a") as handle:
        for cfg, result in zip(todo, pool.imap(eval_cv, todo)):
            handle.write(json.dumps({"id": cfg_id(cfg), "cfg": cfg, "cv": result}) + "\n")
            handle.flush()
            m = result["macro"]
            print(f"stage2 {cfg_id(cfg)} mosaic r10={m['mosaic']['r10']:.4f} internal r10={m['internal']['r10']:.4f}",
                  flush=True)


PROBE_PRESETS = [  # internal-view probe settings whose CV probe columns get cached once and then reused
    {"layers": 16, "probe_type": "logistic", "positions": ("prompt_end", "first_answer", "mean_answer"), "pca": 0,
     "probe_c": 0.01, "base_folds": 4, "source": "cached"},
    {"layers": 16, "probe_type": "lda", "positions": ("first_answer", "mean_answer", "delta_first", "delta_mean"),
     "pca": 64, "probe_c": 0.001, "base_folds": 4, "source": "cached"},
    {"layers": 16, "probe_type": "logistic", "positions": ("prompt_end", "first_answer", "mean_answer", "delta_first"),
     "pca": 0, "probe_c": 0.01, "base_folds": 4, "source": "cached"},
]


def search_cv(hours: float, workers: int, seed: int) -> None:
    """CV-objective search: probe settings fixed to PROBE_PRESETS, everything downstream sampled."""
    import multiprocessing as mp
    log = OUT / "stage2_search.jsonl"
    done = {json.loads(l)["id"] for l in log.open()} if log.exists() else set()
    rng = random.Random(seed + len(done))
    deadline = time.time() + hours * 3600
    for cell in CELLS:
        load(cell, "cached")

    def jobs():
        while time.time() < deadline:
            cfg = sample_cfg(rng, ["cached"])
            cfg.update(rng.choice(PROBE_PRESETS))
            if cfg_id(cfg) not in done:
                done.add(cfg_id(cfg))
                yield cfg

    def run(cfg):
        try:
            return cfg, eval_cv(cfg), None
        except Exception:
            return cfg, None, traceback.format_exc()[-1500:]

    with mp.get_context("fork").Pool(workers, maxtasksperchild=10) as pool, log.open("a") as handle:
        for cfg, result, error in pool.imap_unordered(_run_cv, jobs()):
            if error:
                print("ERROR", cfg_id(cfg), error.splitlines()[-1], flush=True)
                continue
            handle.write(json.dumps({"id": cfg_id(cfg), "cfg": cfg, "cv": result}) + "\n"); handle.flush()
            m = result["macro"]
            print(f"{time.strftime('%H:%M:%S')} CVS {cfg_id(cfg)} probe={cfg['probe_type']}/{len(cfg['positions'])} "
                  f"mosaic r10={m['mosaic']['r10']:.4f} internal r10={m['internal']['r10']:.4f} "
                  f"flat={m['flat']['r10']:.4f} I+B={m['I+B']['r10']:.4f}", flush=True)
            if time.time() > deadline:
                break


def _run_cv(cfg):
    try:
        return cfg, eval_cv(cfg), None
    except Exception:
        return cfg, None, traceback.format_exc()[-1500:]


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--source", required=True, choices=list(SOURCES))
    s = sub.add_parser("search"); s.add_argument("--hours", type=float, default=12); s.add_argument("--workers", type=int, default=8)
    s.add_argument("--seed", type=int, default=20261031)
    t = sub.add_parser("stage2"); t.add_argument("--top", type=int, default=30); t.add_argument("--workers", type=int, default=8)
    sc = sub.add_parser("search_cv"); sc.add_argument("--hours", type=float, default=7); sc.add_argument("--workers", type=int, default=4)
    sc.add_argument("--seed", type=int, default=20261032)
    b = sub.add_parser("board"); b.add_argument("--method", default="mosaic"); b.add_argument("--top", type=int, default=15)
    one = sub.add_parser("one"); one.add_argument("--source", default="cached")
    cvm = sub.add_parser("cvmanual"); cvm.add_argument("--cfgs", type=Path, required=True); cvm.add_argument("--workers", type=int, default=3)
    cvm.add_argument("--seed", type=int, default=2026103102)
    man = sub.add_parser("manual"); man.add_argument("--cfgs", type=Path, required=True); man.add_argument("--workers", type=int, default=2)
    grid = sub.add_parser("cvgrid"); grid.add_argument("--cfgs", type=Path, required=True)
    grid.add_argument("--workers", type=int, default=10); grid.add_argument("--seeds", type=int, nargs="+", default=[2026103102])
    bs = sub.add_parser("bases"); bs.add_argument("--cell", required=True); bs.add_argument("--dim", type=int, default=256)
    bs.add_argument("--positions", nargs="+", default=["prompt_end", "first_answer", "mean_answer", "delta_first", "delta_mean"])
    args = parser.parse_args()
    if args.cmd == "bases":
        d = load(args.cell, "cached")
        for position in args.positions:
            for slot in layer_slots(d, 16):
                basis(d, position, slot, args.dim)
            print(args.cell, position, "bases done", flush=True)
    if args.cmd == "cvgrid":
        cvgrid([json.loads(l) for l in args.cfgs.open()], args.seeds, args.workers)
    if args.cmd == "prepare":
        prepare(args.source)
    elif args.cmd == "search":
        search(args.hours, args.workers, args.seed)
    elif args.cmd == "search_cv":
        search_cv(args.hours, args.workers, args.seed)
    elif args.cmd == "stage2":
        stage2(args.top, args.workers)
    elif args.cmd == "board":
        for r in leaderboard(args.method, top=args.top):
            m = r["val"]["macro"]
            print(r["id"], r["cfg"]["source"], {k: round(v["r10"], 4) for k, v in m.items()})
    elif args.cmd == "manual":
        import multiprocessing as mp
        cfgs = [json.loads(l) for l in args.cfgs.open()]
        with mp.get_context("fork").Pool(args.workers) as pool, (OUT / "stage1_manual.jsonl").open("a") as handle:
            for record in pool.imap_unordered(run_one, cfgs):
                handle.write(json.dumps(record) + "\n"); handle.flush()
                m = record.get("val", {}).get("macro")
                print(record["id"], record["cfg"].get("behavior_source"), record["cfg"].get("xq"),
                      record["cfg"].get("xq_source"),
                      {k: round(v["r10"], 4) for k, v in m.items()} if m else record.get("error", "")[-300:],
                      flush=True)
    elif args.cmd == "cvmanual":
        import multiprocessing as mp
        cfgs = [json.loads(l) for l in args.cfgs.open()]
        with mp.get_context("fork").Pool(args.workers) as pool, (OUT / "stage2_manual.jsonl").open("a") as handle:
            evaluate = functools.partial(eval_cv, seed=args.seed, keep_scores=True)
            for cfg, result in zip(cfgs, pool.imap(evaluate, cfgs)):
                handle.write(json.dumps({"id": cfg_id(cfg), "cfg": cfg, "seed": args.seed, "cv": result}) + "\n"); handle.flush()
                m = result["macro"]
                print(cfg_id(cfg), cfg.get("label", ""), f"seed={args.seed}", {k: round(v["r10"], 4) for k, v in m.items()}, flush=True)
    elif args.cmd == "one":
        cfg = dict(RUN6_LIKE, source=args.source)
        start = time.time()
        result = eval_validation(cfg)
        print(json.dumps(result["macro"], indent=1), f"{time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
