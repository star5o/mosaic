from __future__ import annotations

from pathlib import Path

import compile_one_confidence_mosaic_v1_artifact_manifest as module


def test_ensure_required_rejects_missing_file(tmp_path: Path) -> None:
    try:
        module.ensure_required(tmp_path, ("missing.json",))
    except FileNotFoundError as error:
        assert "missing.json" in str(error)
    else:
        raise AssertionError("missing required artifact was accepted")


def test_fresh_isolation_counts_generations_without_derived_scores(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh_holdout_generations"
    fresh.mkdir()
    (fresh / "model-generation.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (fresh / "model-run.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "fresh_method_holdout_protocol_preregistered.json").write_text(
        "{}\n", encoding="utf-8"
    )
    result = module.fresh_isolation(tmp_path)
    assert result["generation_file_count"] == 1
    assert next(iter(result["generation_files"].values()))["rows"] == 2
    assert result["detector_scoring_performed"] is False


def test_fresh_isolation_rejects_detector_artifact(tmp_path: Path) -> None:
    (tmp_path / "fresh_detector_scores.json").write_text("{}\n", encoding="utf-8")
    try:
        module.fresh_isolation(tmp_path)
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("fresh detector artifact was accepted")
