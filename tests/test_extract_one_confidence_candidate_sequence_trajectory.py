from types import SimpleNamespace

import pytest
import torch

from extract_one_confidence_candidate_sequence_trajectory import (
    ANSWER_SYSTEM,
    aligned_answer_ids,
    build_source_records,
    flat_trajectory_features,
    score_answer_trajectory,
    trajectory_summary,
)


class TinyBlock(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = torch.nn.Linear(width, width, bias=False)

    def forward(self, hidden):
        return (hidden + 0.05 * self.linear(hidden),)


class TinyBody(torch.nn.Module):
    def __init__(self, vocab: int, width: int, layers: int):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, width)
        self.layers = torch.nn.ModuleList([TinyBlock(width) for _ in range(layers)])
        self.norm = torch.nn.LayerNorm(width)


class TinyLM(torch.nn.Module):
    def __init__(self, vocab: int = 17, width: int = 8, layers: int = 4):
        super().__init__()
        self.model = TinyBody(vocab, width, layers)
        self.lm_head = torch.nn.Linear(width, vocab, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, use_cache=False):
        del use_cache
        hidden = self.model.embed_tokens(input_ids)
        for block in self.model.layers:
            hidden = block(hidden)[0]
        logits = self.lm_head(self.model.norm(hidden))
        return SimpleNamespace(logits=logits)


class TinyTokenizer:
    def __init__(self):
        self.mapping = {"canonical": [3, 4], "candidate": [5, 6]}
        self.reverse = {tuple(value): key for key, value in self.mapping.items()}
        self.reverse[(3, 3)] = "canonical"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(self.mapping[text])

    def decode(self, values, skip_special_tokens=True):
        assert skip_special_tokens is True
        return self.reverse[tuple(values)]


def public_payloads(count: int = 25):
    benchmark_records = []
    feature_records = []
    nli_records = []
    margin_records = []
    for index in range(count):
        sample_id = f"mistral7|TriviaQA|{index}"
        base = {
            "sample_id": sample_id,
            "model": "mistral7",
            "dataset": "TriviaQA",
            "row_id": str(index),
            "fact_key_sha256": f"fact-{index}",
            "split": "development",
            "class": "HCE" if index % 2 else "HCC",
            "confidence": 0.9,
            "question": f"Question {index}?",
            "generated_answer": "canonical",
            "gold_aliases": ["must-not-be-propagated"],
        }
        benchmark_records.append(base)
        feature_records.append(
            {
                "sample_id": sample_id,
                "split": "development",
                "label": base["class"],
                "fact_key_sha256": f"fact-{index}",
                "canonical_answer": "canonical",
                "panel_questions": [base["question"], "p1", "p2", "p3", "p4"],
                "panel_answers": ["canonical", "candidate", "canonical", "canonical", "canonical"],
                "panel_generated_token_ids": [[3, 4], [5, 6], [3, 4], [3, 4], [3, 4]],
            }
        )
        nli_records.append(
            {
                "sample_id": sample_id,
                "split": "development",
                "label": base["class"],
                "fact_key_sha256": f"fact-{index}",
                "cluster_ids": [1, 0, 0, 0],
                "alternative_candidate_natural": "candidate",
                "alternative_candidate_cluster_natural": 1,
                "alternative_candidate_all_safe": "candidate",
                "alternative_candidate_cluster_all_safe": 1,
            }
        )
        margin_records.append(
            {
                "sample_id": sample_id,
                "split": "development",
                "label": base["class"],
                "fact_key_sha256": f"fact-{index}",
                "canonical_answer": "canonical",
                "candidate_available": True,
                "candidate_answer": "candidate",
                "candidate_source": "natural_alternative_cluster",
                "candidate_cluster": 1,
                "canonical_answer_nll_recomputed": 1.0,
                "candidate_answer_nll": 0.8,
                "candidate_logscore_margin": 0.2,
            }
        )
    benchmark = {
        "schema_version": "one-confidence-matched-challenge-public-v1",
        "cells": {
            "mistral7|TriviaQA": {
                "development": {"records": benchmark_records},
                "validation": {"records": []},
            }
        },
    }
    features = {
        "schema_version": "one-confidence-corrected-panel-features-v1",
        "cell": "mistral7|TriviaQA",
        "locked_accessed": False,
        "complete": True,
        "answer_protocol": {
            "system": ANSWER_SYSTEM,
            "qwen_thinking": False,
        },
        "records": feature_records,
    }
    nli = {
        "schema_version": "one-confidence-corrected-panel-nli-v1",
        "cell": "mistral7|TriviaQA",
        "locked_accessed": False,
        "records": nli_records,
    }
    margins = {
        "schema_version": "one-confidence-nli-candidate-margins-v1",
        "cell": "mistral7|TriviaQA",
        "locked_accessed": False,
        "gold_visible": False,
        "records": margin_records,
    }
    return benchmark, features, nli, margins


def test_public_join_limits_sanity_to_20_and_never_propagates_gold():
    rows = build_source_records(
        *public_payloads(),
        "mistral7|TriviaQA",
        splits=("development",),
        candidate_only=True,
        limit=20,
    )
    assert len(rows) == 20
    assert all("gold_aliases" not in row for row in rows)
    assert all(row["candidate_source"] == "natural_alternative_cluster" for row in rows)


def test_public_join_rejects_candidate_not_frozen_by_nli():
    benchmark, features, nli, margins = public_payloads(1)
    margins["records"][0]["candidate_answer"] = "invented after seeing labels"
    with pytest.raises(ValueError, match="frozen NLI alternative"):
        build_source_records(
            benchmark, features, nli, margins, "mistral7|TriviaQA"
        )


def test_answer_token_alignment_is_exact():
    tokenizer = TinyTokenizer()
    ids, audit = aligned_answer_ids(
        tokenizer, "canonical", [3, 4], "sample", "canonical"
    )
    assert ids == [3, 4]
    assert audit["frozen_generated_token_ids_match"] is True
    _, canonical_without_generation_ids = aligned_answer_ids(
        tokenizer, "canonical", None, "sample", "canonical"
    )
    assert canonical_without_generation_ids["teacher_decode_matches_answer"] is True
    assert canonical_without_generation_ids["frozen_generated_token_ids_match"] is None
    mismatch_ids, mismatch = aligned_answer_ids(
        tokenizer, "canonical", [3, 3], "sample", "canonical"
    )
    assert mismatch_ids == [3, 4]
    assert mismatch["frozen_generated_token_ids_match"] is False
    assert mismatch["context_dependent_tokenization_difference"] is True


def test_final_layer_sequence_nll_matches_actual_model_nll():
    torch.manual_seed(7)
    model = TinyLM().eval()
    result = score_answer_trajectory(
        model, prompt_ids=[1, 2, 7], answer_ids=[3, 4], final_nll_tolerance=1e-6
    )
    assert len(result["layer_mean_logprob"]) == 4
    assert result["final_layer_nll_abs_error"] == pytest.approx(0.0, abs=1e-7)
    assert result["lens_final_nll"] == pytest.approx(result["actual_final_nll"])
    assert len(result["layer_token_logprobs"][-1]) == 2


def test_trajectory_summary_reports_regions_slopes_and_flips():
    result = trajectory_summary([-0.4, -0.2, 0.3, 0.1, -0.1, 0.2])
    assert result["sign_flip_count"] == 3
    assert result["maximum_positive_slope"] == pytest.approx(0.5)
    assert result["maximum_positive_slope_to_layer"] == 3
    assert result["final_candidate_preferred"] is True
    assert result["early_mean_margin"] == pytest.approx(-0.3)


def test_flat_features_cover_all_rows_and_are_numeric_when_scored():
    missing = flat_trajectory_features(None)
    assert missing["candidate_available"] == 0.0
    assert all(
        value is None
        for key, value in missing.items()
        if key != "candidate_available"
    )

    summary = trajectory_summary([-0.4, -0.2, -0.1, -0.05])
    scored = flat_trajectory_features(summary)
    assert scored["candidate_available"] == 1.0
    assert set(scored) == set(missing)
    assert all(isinstance(value, float) for value in scored.values())
    assert scored["first_candidate_preferred_layer"] == 0.0
