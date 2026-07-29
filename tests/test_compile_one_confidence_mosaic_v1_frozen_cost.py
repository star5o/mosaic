from __future__ import annotations

import json
from pathlib import Path

import compile_one_confidence_mosaic_v1_frozen_cost as module


def test_last_elapsed_seconds_reads_both_log_formats(tmp_path: Path) -> None:
    path = tmp_path / "run.log"
    path.write_text(
        '8/10 elapsed=1.5\n{"elapsed_seconds": 2.75}\n', encoding="utf-8"
    )
    assert module.last_elapsed_seconds(path) == 2.75


def test_compile_cell_counts_realized_operations(tmp_path: Path, monkeypatch) -> None:
    slug = "toy-data"
    monkeypatch.setitem(module.CELL_NAMES, slug, "toy|Data")
    records = [
        {
            "sample_id": "a",
            "panel_answers": ["x", "y"],
            "panel_generated_token_ids": [[1], [2, 3]],
        },
        {
            "sample_id": "b",
            "panel_answers": ["x", "z"],
            "panel_generated_token_ids": [[4, 5], [6]],
        },
    ]
    (tmp_path / f"{slug}-natural-paraphrase.json").write_text(
        json.dumps(
            {
                "records": [
                    {"sample_id": "a", "panel_answers": ["x", "y"]},
                    {"sample_id": "b", "panel_answers": ["x", "z"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / f"{slug}-natural-paraphrase.log").write_text(
        "2/2 elapsed=3.0\n", encoding="utf-8"
    )
    (tmp_path / f"{slug}-corrected-panel-features.json").write_text(
        json.dumps({"elapsed_seconds": 4.0, "records": records}), encoding="utf-8"
    )
    (tmp_path / f"{slug}-candidate-margins.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sample_id": sample_id,
                        "panel_canonical_answer_nlls": [0.1, 0.2],
                        "panel_candidate_answer_nlls": [None, 0.3],
                    }
                    for sample_id in ("a", "b")
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / f"{slug}-persistence.json").write_text(
        json.dumps(
            {"K": 4, "records": [{"sample_id": "a"}, {"sample_id": "b"}]}
        ),
        encoding="utf-8",
    )
    (tmp_path / f"{slug}-internal-states.pt").write_bytes(b"fixture")

    row, sources = module.compile_cell(tmp_path, slug)

    assert row["n_rows"] == 2
    assert row["corrected_panel_answer_generations"] == 4
    assert row["self_consistency_sampling_generations"] == 8
    assert row["canonical_teacher_forced_sequence_scores"] == 4
    assert row["candidate_teacher_forced_sequence_scores"] == 2
    assert row["corrected_panel_generated_answer_tokens"] == 6
    assert row["mean_corrected_panel_answer_tokens"] == 1.5
    assert row["measured_gpu_stage_wall_time_seconds"] == 7.0
    assert len(sources) == 6


def test_compile_cost_rejects_public_policy_cost(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(module, "CELLS", ())
    mitigation_path = tmp_path / "cost.json"
    mitigation_path.write_text(
        json.dumps({"rows": [{"split": "validation", "method": "keep"}]}),
        encoding="utf-8",
    )
    try:
        module.compile_cost(tmp_path, mitigation_path)
    except ValueError as error:
        assert "non-locked-test" in str(error)
    else:
        raise AssertionError("expected validation policy cost to be rejected")
