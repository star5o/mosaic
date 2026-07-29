
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
