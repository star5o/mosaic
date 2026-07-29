#!/usr/bin/env python3
"""Extract fixed-prompt answer-start and cross-NLL features on public rows.

The generated answer and the model's own probabilities are the only inference
inputs. Gold labels are copied solely for evaluation bookkeeping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model_loading import load_causal_model


SYSTEM = "Return only the requested factual answer entity or short value. Do not write a sentence, preamble, explanation, or reasoning."
TEMPLATES = (
    "{q}",
    "Give the shortest factual answer to this question: {q}",
    "Identify only the entity or value requested here: {q}",
    "Without explanation, answer this question: {q}",
)


def prompt_ids(tokenizer, text: str) -> list[int]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    try:
        return list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
    except TypeError:
        return list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def padded(ids_list: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(ids) for ids in ids_list)
    input_ids = torch.full((len(ids_list), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(input_ids)
    for index, ids in enumerate(ids_list):
        input_ids[index, width - len(ids) :] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[index, width - len(ids) :] = 1
    return input_ids, mask


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    mixture = (left + right) / 2
    return entropy(mixture) - (entropy(left) + entropy(right)) / 2


def stable_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=8)
    args = parser.parse_args()

    benchmark_bytes = args.benchmark.read_bytes()
    benchmark = json.loads(benchmark_bytes)
    cell = benchmark["cells"][args.cell]
    rows = [
        (split, row)
        for split in ("development", "validation")
        for row in cell[split]["records"]
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_causal_model(args.model_path, load_in_4bit=False, torch_dtype=torch.bfloat16).eval()
    device = next(model.parameters()).device
    pad_id = int(tokenizer.pad_token_id)
    records: list[dict] = []
    started = time.time()

    for offset in range(0, len(rows), args.batch_rows):
        batch = rows[offset : offset + args.batch_rows]
        all_prompt_ids: list[list[int]] = []
        all_teacher_ids: list[list[int]] = []
        answer_lengths: list[int] = []
        for _, row in batch:
            question = row["question"]
            canonical = prompt_ids(tokenizer, TEMPLATES[0].format(q=question))
            variants = [prompt_ids(tokenizer, template.format(q=question)) for template in TEMPLATES]
            answer = tokenizer.encode(row["generated_answer"], add_special_tokens=False)
            if not answer:
                raise ValueError(f"empty answer token sequence: {row['sample_id']}")
            all_prompt_ids.extend(variants)
            all_teacher_ids.extend([variant + answer for variant in variants])
            answer_lengths.append(len(answer))

        prompt_input, prompt_mask = padded(all_prompt_ids, pad_id, device)
        teacher_input, teacher_mask = padded(all_teacher_ids, pad_id, device)
        prompt_logits = model(input_ids=prompt_input, attention_mask=prompt_mask, use_cache=False).logits[:, -1].float()
        teacher_logits = model(input_ids=teacher_input, attention_mask=teacher_mask, use_cache=False).logits.float()

        for local, (split, row) in enumerate(batch):
            start = local * len(TEMPLATES)
            end = start + len(TEMPLATES)
            distributions = torch.softmax(prompt_logits[start:end], dim=-1)
            top_probs, top_ids = distributions.topk(min(128, distributions.shape[-1]), dim=-1)
            top1 = top_ids[:, 0]
            _, top1_frequencies = torch.unique(top1, return_counts=True)
            top1_counts = top1_frequencies.numel()
            pairwise = [
                float(js_divergence(distributions[left], distributions[right]))
                for left in range(len(TEMPLATES))
                for right in range(left + 1, len(TEMPLATES))
            ]
            mean_distribution = distributions.mean(dim=0)
            answer_start_mi = float(entropy(mean_distribution) - entropy(distributions).mean())

            answer = tokenizer.encode(row["generated_answer"], add_special_tokens=False)
            nlls = []
            for variant in range(len(TEMPLATES)):
                item_index = start + variant
                sequence = teacher_logits[item_index]
                prompt_len = len(all_teacher_ids[item_index]) - len(answer)
                left_padding = teacher_input.shape[1] - len(all_teacher_ids[item_index])
                answer_start = left_padding + prompt_len - 1
                answer_logits = sequence[answer_start : answer_start + len(answer)]
                target = torch.tensor(answer, dtype=torch.long, device=device)
                nlls.append(float(-torch.log_softmax(answer_logits, dim=-1).gather(1, target[:, None]).mean()))
            canonical_nll = nlls[0]
            variant_changes = [value - canonical_nll for value in nlls[1:]]

            records.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "label": row["class"],
                    "canonical_answer": row["generated_answer"],
                    "answer_token_count": len(answer),
                    "answer_start_top1_ids": [int(value) for value in top1.tolist()],
                    "answer_start_top1_probabilities": [float(value) for value in top_probs[:, 0].tolist()],
                    "answer_start_top1_disagreement": 1.0 - float(top1_frequencies.max()) / len(TEMPLATES),
                    "answer_start_unique_top1_tokens": int(top1_counts),
                    "answer_start_mean_pairwise_js": float(sum(pairwise) / len(pairwise)),
                    "answer_start_max_pairwise_js": float(max(pairwise)),
                    "answer_start_js_mi": answer_start_mi,
                    "answer_start_mi_fixed2": float(entropy(distributions[:2].mean(dim=0)) - entropy(distributions[:2]).mean()),
                    "answer_start_mi_fixed3": float(entropy(distributions[:3].mean(dim=0)) - entropy(distributions[:3]).mean()),
                    "answer_start_mi_fixed4": float(entropy(distributions.mean(dim=0)) - entropy(distributions).mean()),
                    "canonical_answer_nll": canonical_nll,
                    "paraphrase_answer_nlls": nlls[1:],
                    "signed_cross_prompt_nll_change": float(sum(variant_changes) / len(variant_changes)),
                    "absolute_cross_prompt_nll_change": float(sum(abs(value) for value in variant_changes) / len(variant_changes)),
                    "max_cross_prompt_nll_change": float(max(variant_changes)),
                    "topk_ids": [[int(value) for value in row_ids] for row_ids in top_ids.tolist()],
                    "topk_probabilities": [[float(value) for value in row_probs] for row_probs in top_probs.tolist()],
                }
            )
        print(f"{min(offset + len(batch), len(rows))}/{len(rows)} elapsed={time.time() - started:.1f}", flush=True)

    payload = {
        "schema_version": "one-confidence-answer-start-features-v1",
        "cell": args.cell,
        "model_path": args.model_path,
        "dtype": "bfloat16",
        "benchmark_sha256": stable_hash(args.benchmark),
        "locked_accessed": False,
        "templates": TEMPLATES,
        "gold_visible": "labels_only",
        "feature_definitions": {
            "answer_start_js_mi": "entropy(mean of five answer-start distributions) minus mean entropy",
            "cross_prompt_nll": "length-normalized teacher-forced NLL of the canonical generated answer",
            "candidate_source": "canonical model answer and fixed prompt-panel model outputs only",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
