#!/usr/bin/env python3
"""Build a compact, portable GitHub release for One Confidence / MOSAIC."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


CORE_SCRIPTS = (
    "analyze_one_confidence_mechanism_patching.py",
    "apply_one_confidence_los_checkpoint.py",
    "apply_one_confidence_mosaic_v1_frozen_checkpoint.py",
    "audit_one_confidence_mechanism_evidence.py",
    "audit_one_confidence_question_equivalence.py",
    "bootstrap_one_confidence_validation.py",
    "build_one_confidence_internal_trajectory.py",
    "build_one_confidence_mechanism_patch_manifest.py",
    "build_one_confidence_strict_partial_phenotypes.py",
    "compile_one_confidence_mechanism_controls.py",
    "compile_one_confidence_mosaic_v1_ablation_delivery.py",
    "compile_one_confidence_mosaic_v1_artifact_manifest.py",
    "compile_one_confidence_mosaic_v1_final_delivery.py",
    "compile_one_confidence_mosaic_v1_frozen_closure.py",
    "compile_one_confidence_mosaic_v1_frozen_cost.py",
    "compile_one_confidence_mosaic_v1_frozen_results.py",
    "evaluate_one_confidence_five_model.py",
    "evaluate_one_confidence_frozen_mitigation_run6.py",
    "evaluate_one_confidence_minimal_mosaic.py",
    "evaluate_one_confidence_mosaic_v1_nested_public.py",
    "evaluate_one_confidence_public_mitigation.py",
    "extract_one_confidence_answer_start_features.py",
    "extract_one_confidence_candidate_sequence_trajectory.py",
    "extract_one_confidence_corrected_panel_features.py",
    "extract_one_confidence_los_features.py",
    "extract_one_confidence_natural_paraphrase_features.py",
    "fit_one_confidence_los_net.py",
    "make_one_confidence_mosaic_v1_nested_public_config.py",
    "materialize_one_confidence_frozen_run6_aux_sources.py",
    "model_loading.py",
    "one_confidence_frozen_source_access.py",
    "run_one_confidence_mechanism_erasure.py",
    "run_one_confidence_mechanism_patching.py",
    "score_one_confidence_corrected_panel_nli.py",
    "score_one_confidence_inside.py",
    "score_one_confidence_natural_panel_nli.py",
    "score_one_confidence_nli_candidate_margins.py",
    "score_one_confidence_paraphrase_panel.py",
    "score_one_confidence_ptrue.py",
    "score_one_confidence_sampling.py",
    "verify_one_confidence_mosaic_v1_frozen_closure.py",
)

TEST_SCRIPTS = (
    "test_apply_one_confidence_mosaic_v1_frozen_checkpoint.py",
    "test_audit_one_confidence_question_equivalence.py",
    "test_compile_one_confidence_mechanism_controls.py",
    "test_compile_one_confidence_mosaic_v1_ablation_delivery.py",
    "test_compile_one_confidence_mosaic_v1_artifact_manifest.py",
    "test_compile_one_confidence_mosaic_v1_final_delivery.py",
    "test_compile_one_confidence_mosaic_v1_frozen_closure.py",
    "test_compile_one_confidence_mosaic_v1_frozen_cost.py",
    "test_evaluate_one_confidence_five_model.py",
    "test_evaluate_one_confidence_frozen_mitigation_run6.py",
    "test_evaluate_one_confidence_mosaic_v1_nested_public.py",
    "test_evaluate_one_confidence_public_mitigation.py",
    "test_extract_one_confidence_candidate_sequence_trajectory.py",
    "test_one_confidence_frozen_source_access.py",
    "test_one_confidence_mechanism_erasure.py",
    "test_one_confidence_mechanism_patching.py",
)

CONFIG_FILES = (
    "protocol.json",
    "protocol_public_freeze_addendum.json",
    "feature_manifest.json",
    "final_thresholds.json",
    "model_checkpoints/index.json",
)

FINAL_RESULT_FILES = (
    "behavioral_features.json",
    "internal_features.json",
    "closure_completion_audit.json",
    "frozen_closure_integrity_report.json",
    "public_validation_integrity_report.json",
    "frozen_results_compile_manifest.json",
    "frozen_test_bootstrap.csv",
    "frozen_test_bootstrap.json",
    "validation_search.csv",
    "validation_search.json",
    "table_main_detection.csv",
    "table_main_detection.json",
    "table_main_detection_validation.csv",
    "table_main_detection_validation.json",
    "table_detection_by_phenotype.csv",
    "table_detection_by_phenotype.json",
    "table_mechanism_controls.csv",
    "table_mechanism_controls.json",
    "table_ablation.csv",
    "table_ablation.json",
    "table_mitigation.csv",
    "table_mitigation.json",
    "table_cost.csv",
    "table_cost.json",
    "figure_layer_patching.csv",
    "figure_layer_patching.json",
    "figure_risk_coverage.csv",
    "figure_risk_coverage.json",
    "case_studies.json",
    "mechanism_patch_results.json",
    "mechanism_erasure_results.json",
    "mechanism_evidence_audit.json",
    "mechanism_control_verdict.json",
    "llama_mechanism_verdict.json",
    "safe_wrapper_patch_verdict.json",
    "phenotype_labels.json",
    "phenotype_labels_frozen_test.json",
    "table_detection_by_phenotype_legacy_public_validation.json",
)

RAW_MECHANISM_GLOBS = (
    "*-public-mechanism-patching.json",
    "*-public-mechanism-erasure.json",
    "*-public-nonbehavioral-candidate-patching.json",
    "*-public-mechanism-patch-manifest.json",
)

EXTERNAL_PATTERNS = (
    "experiments/hallucination/results/one_confidence_five_model_eval/*-public-features.pt",
    "experiments/hallucination/results/one_confidence_mosaic_v1/frozen_test_sources/*-internal-states.pt",
    "experiments/hallucination/results/one_confidence_mosaic_v1_nested_public_run6_16layer_strong_regularization/*/checkpoint.joblib",
    "experiments/hallucination/results/one_confidence_mosaic_v1/fresh_holdout_generations/*-generation.jsonl",
)

TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".txt", ".toml", ".yml", ".yaml"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


class ReleaseBuilder:
    def __init__(self, project_root: Path, output: Path) -> None:
        self.project_root = project_root.resolve()
        self.output = output.resolve()
        self.files: list[dict[str, Any]] = []
        self.replacements = {
            self.project_root.as_posix() + "/": "<ORIGINAL_PROJECT_ROOT>/",
            "<ORIGINAL_PROJECT_ROOT>/": "<ORIGINAL_PROJECT_ROOT>/",
            "<REMOTE_WORKSPACE>/": "<REMOTE_WORKSPACE>/",
            "<REMOTE_HOST>": "<REMOTE_HOST>",
            "<REMOTE_HOST>": "<REMOTE_HOST>",
            "<REMOTE_PORT>": "<REMOTE_PORT>",
        }

    def copy(self, source: Path, destination: Path) -> None:
        source = source.resolve()
        destination = self.output / destination
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        original = source.read_bytes()
        sanitized = False
        if source.suffix.lower() in TEXT_SUFFIXES:
            text = original.decode("utf-8")
            rewritten = text
            for old, new in self.replacements.items():
                rewritten = rewritten.replace(old, new)
            sanitized = rewritten != text
            destination.write_text(rewritten, encoding="utf-8")
        else:
            destination.write_bytes(original)
        self.files.append(
            {
                "path": destination.relative_to(self.output).as_posix(),
                "source": source.relative_to(self.project_root).as_posix(),
                "source_sha256": hashlib.sha256(original).hexdigest(),
                "release_sha256": sha256(destination),
                "size_bytes": destination.stat().st_size,
                "path_provenance_sanitized": sanitized,
            }
        )

    def generated(self, destination: Path, content: str) -> None:
        path = self.output / destination
        write_text(path, content)
        self.files.append(
            {
                "path": path.relative_to(self.output).as_posix(),
                "source": None,
                "source_sha256": None,
                "release_sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "path_provenance_sanitized": False,
            }
        )


README = """
# MOSAIC

Code and artifacts for MOSAIC (Multi-view Output and State Aggregation for
Identifying Confident Hallucinations) and the HalluMatch benchmark.

HalluMatch contains 1,563 confidence-matched hallucination/correct pairs
across five checkpoints. MOSAIC combines equivalent-question behavior,
layerwise internal-state trajectories, and sampling persistence. It does not
use a reference answer at inference time.

## Held-out test

Validation-selected thresholds, no refitting. Macro metrics weight the five
checkpoints equally. ΔR is versus the internal-state trajectory.

| Method | M7 (81) | Q8 (65) | L8 (63) | Q14 (61) | MN (57) | Macro R | FPR | Acc. | ΔR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean hidden probe | 34/7 | 14/6 | 27/6 | 13/5 | 25/5 | 34.31 | 8.87 | 62.72 | −6.06 |
| Capacity-matched MLP | 37/6 | 15/7 | 29/6 | 14/5 | 27/5 | 37.02 | 8.93 | 64.04 | −3.35 |
| Behavior expert | 30/6 | 18/8 | 24/7 | 15/6 | 25/5 | 34.26 | 9.89 | 62.18 | −6.11 |
| Persistence expert | 18/5 | 13/6 | 17/5 | 12/5 | 15/4 | 23.04 | 7.71 | 57.66 | −17.33 |
| Internal-state trajectory | 40/6 | 17/7 | 31/6 | 16/5 | 29/5 | 40.37 | 8.93 | 65.72 | — |
| Flat concatenation | 42/7 | 20/9 | 33/7 | 17/6 | 32/5 | 43.80 | 10.44 | 66.68 | +3.43 |
| **MOSAIC** | **45/5** | **22/10** | **36/6** | **19/5** | **36/5** | **48.17** | **9.61** | **69.28** | **+7.80** |

The 7.80-point recall gain has a full-pipeline hierarchical-bootstrap 95% CI
of 4.28 to 11.36 (p = 0.002). The same sign of improvement appears at the 5%
and 15% transferred FPR caps and in test partial AUC (0.061 vs 0.052).

Full tables, validation ablations, cost, and controls are in
[docs/RESULTS.md](docs/RESULTS.md).

## Layout

- `src/`: feature extraction, baselines, MOSAIC, intervention, and mitigation
- `tests/`: unit and integrity tests
- `configs/`: protocol, feature schema, thresholds, and model revisions
- `data/`: per-sample scores, OOF metadata, baselines, and case studies
- `results/`: canonical tables, figures, and bootstrap intervals
- `external_artifacts.json`: SHA256 records for large tensors omitted from Git

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU regeneration requires the official checkpoints in
`configs/model_provenance.json`.

## Verify

```bash
python tools/verify_release.py
PYTHONPATH=src pytest -q
```

## Data

Questions come from PopQA, TriviaQA, and SimpleQA. Model outputs come from the
official Mistral, Qwen, and Llama checkpoints listed in the provenance files.
Upstream dataset and model licenses continue to apply.
"""

REQUIREMENTS = """
accelerate>=1.0
bitsandbytes>=0.45
datasets>=3.0
joblib>=1.4
numpy>=2.0
protobuf>=5.0
pytest>=8.0
scikit-learn>=1.5
scipy>=1.13
sentencepiece>=0.2
torch>=2.6
transformers>=4.51
"""

PYPROJECT = """
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-ra"
"""

GITIGNORE = """
.DS_Store
.env
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
*.joblib
*.pt
*.safetensors
*.bin
outputs/
wandb/
"""

VERIFY_RELEASE = r'''
#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
for row in manifest["files"]:
    path = ROOT / row["path"]
    assert path.is_file(), f"missing: {row['path']}"
    assert sha256(path) == row["release_sha256"], f"SHA mismatch: {row['path']}"

table = json.loads(
    (ROOT / "results/final/table_main_detection.json").read_text(encoding="utf-8")
)
run6 = table["macro_equal_cell_average"]["mosaic_v1_run6"]
assert abs(run6["accuracy"] - 0.6927675664805374) < 1e-12
assert abs(run6["recall"] - 0.4816720939240352) < 1e-12
assert abs(run6["fpr"] - 0.09613696096296039) < 1e-12

external = json.loads((ROOT / "external_artifacts.json").read_text(encoding="utf-8"))
assert external["included_in_git"] is False
assert external["fresh_holdout"]["correctness_reviewed"] is False
assert external["fresh_holdout"]["detector_scored"] is False
print(f"verified {len(manifest['files'])} files")
'''

RELEASE_TEST = r'''
from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_frozen_headline_metrics_and_counts():
    table = load("results/final/table_main_detection.json")
    run6 = table["macro_equal_cell_average"]["mosaic_v1_run6"]
    assert run6["accuracy"] == pytest.approx(0.6927675664805374)
    assert run6["recall"] == pytest.approx(0.4816720939240352)
    assert run6["fpr"] == pytest.approx(0.09613696096296039)
    pooled = table["pooled_supplement"]["mosaic_v1_run6"]
    assert tuple(pooled[key] for key in ("TP", "FN", "FP", "TN")) == (161, 166, 33, 294)


def test_mechanism_and_mitigation_verdicts_remain_negative():
    audit = load("results/final/mechanism_evidence_audit.json")
    assert audit["mechanism_summary"]["validation_hce_with_patch_and_erasure"] == 19
    assert audit["mechanism_summary"]["bidirectional_supported_cells"] == []
    mitigation = load("results/final/table_mitigation.json")
    frozen = mitigation["retrospective_frozen_test_scope"]["rows"]
    macro = {
        row["method"]: row
        for row in frozen
        if row["aggregation"] == "macro"
    }
    assert macro["mosaic_conditional_routing"]["net_correction"] == -30
    assert macro["uniform_likelihood_aggregation"]["net_correction"] == 5
    assert macro["uniform_steering"]["available"] is False


def test_release_excludes_fresh_evaluation_and_large_tensors():
    external = load("external_artifacts.json")
    assert external["included_in_git"] is False
    assert external["fresh_holdout"]["correctness_reviewed"] is False
    assert external["fresh_holdout"]["detector_scored"] is False
    assert not list(ROOT.rglob("*.joblib"))
    assert not list(ROOT.rglob("*.pt"))
'''


def build_external_manifest(project_root: Path) -> dict[str, Any]:
    rows = []
    seen = set()
    for pattern in EXTERNAL_PATTERNS:
        for path in sorted(project_root.glob(pattern)):
            relative = path.relative_to(project_root).as_posix()
            if relative in seen:
                continue
            seen.add(relative)
            rows.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "reason_omitted": (
                        "large reproducibility artifact; distribute through a "
                        "versioned release or artifact store, not Git"
                    ),
                }
            )
    return {
        "schema_version": "one-confidence-mosaic-external-artifacts-v1",
        "included_in_git": False,
        "artifacts": rows,
        "fresh_holdout": {
            "generation_files": 5,
            "rows_per_model": 4000,
            "correctness_reviewed": False,
            "detector_scored": False,
            "result_status": "generation_only_not_an_evaluation",
        },
        "model_weights": {
            "included": False,
            "source": "official model repositories identified in configs/model_provenance.json",
        },
    }


def build(project_root: Path, output: Path, force: bool = False) -> None:
    if output.exists():
        manifest_path = output / "release_manifest.json"
        if not force:
            raise FileExistsError(f"output already exists: {output}")
        if not manifest_path.is_file():
            raise ValueError(f"refusing to replace unmanaged directory: {output}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "one-confidence-mosaic-github-release-v1":
            raise ValueError(f"refusing to replace unknown release directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    builder = ReleaseBuilder(project_root, output)
    scripts = project_root / "experiments/hallucination"
    result_dir = scripts / "results/one_confidence_mosaic_v1"

    for name in CORE_SCRIPTS:
        builder.copy(scripts / name, Path("src") / name)
    builder.copy(Path(__file__), Path("tools/build_release.py"))
    for name in TEST_SCRIPTS:
        builder.copy(scripts / name, Path("tests") / name)

    for name in CONFIG_FILES:
        destination = (
            Path("configs/model_provenance.json")
            if name == "model_checkpoints/index.json"
            else Path("configs") / Path(name).name
        )
        builder.copy(result_dir / name, destination)

    for name in FINAL_RESULT_FILES:
        if name.startswith("phenotype_labels") or name == "case_studies.json":
            destination = Path("data/labels") / name
        else:
            destination = Path("results/final") / name
        builder.copy(result_dir / name, destination)

    # Public docs must stay English. Do not copy internal Chinese research notes.
    results_doc = Path(__file__).resolve().parent.parent / "docs/RESULTS.md"
    if results_doc.is_file():
        builder.copy(results_doc, Path("docs/RESULTS.md"))
    else:
        english_report = project_root / "docs/RESULTS.md"
        builder.copy(english_report, Path("docs/RESULTS.md"))

    for pattern in RAW_MECHANISM_GLOBS:
        for path in sorted(result_dir.glob(pattern)):
            builder.copy(path, Path("data/mechanism") / path.name)

    frozen_groups = {
        "applications": scripts / "results/one_confidence_mosaic_v1_frozen_application_run6",
        "scores": scripts / "results/one_confidence_mosaic_v1_frozen_scores_run6",
        "mitigation": scripts / "results/one_confidence_mosaic_v1_frozen_mitigation_run6",
    }
    for group, directory in frozen_groups.items():
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in {".json", ".csv"}:
                builder.copy(path, Path("data/frozen") / group / path.name)

    baseline_dir = scripts / "results/one_confidence_five_model_eval"
    for path in sorted(baseline_dir.iterdir()):
        if (
            path.is_file()
            and path.suffix in {".json", ".jsonl", ".csv"}
            and "invalidated" not in path.name
        ):
            builder.copy(path, Path("data/baselines") / path.name)

    run6_dir = scripts / "results/one_confidence_mosaic_v1_nested_public_run6_16layer_strong_regularization"
    for path in sorted(run6_dir.rglob("*")):
        if not path.is_file() or path.suffix == ".joblib":
            continue
        builder.copy(path, Path("data/validation_run6") / path.relative_to(run6_dir))

    external = build_external_manifest(project_root)
    builder.generated(
        Path("external_artifacts.json"),
        json.dumps(external, ensure_ascii=False, indent=2, allow_nan=False),
    )
    builder.generated(Path("README.md"), README)
    builder.generated(Path("requirements.txt"), REQUIREMENTS)
    builder.generated(Path("pyproject.toml"), PYPROJECT)
    builder.generated(Path(".gitignore"), GITIGNORE)
    builder.generated(Path("tools/verify_release.py"), VERIFY_RELEASE)
    builder.generated(Path("tests/test_release_data.py"), RELEASE_TEST)

    release_manifest = {
        "schema_version": "one-confidence-mosaic-github-release-v1",
        "scientific_status": "limited_and_negative_closure",
        "files": sorted(builder.files, key=lambda row: row["path"]),
        "file_count": len(builder.files),
        "total_size_bytes": sum(row["size_bytes"] for row in builder.files),
        "path_only_sanitization": True,
        "labels_or_numeric_results_modified": False,
    }
    write_text(
        output / "release_manifest.json",
        json.dumps(release_manifest, ensure_ascii=False, indent=2, allow_nan=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.project_root, args.output, force=args.force)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
