#!/usr/bin/env python3
"""Generate a fixed natural-paraphrase panel and extract gold-free features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model_loading import load_causal_model
from one_confidence_frozen_source_access import load_rows, require_authorized_argument


ANSWER_SYSTEM = "Return only the requested factual answer entity or short value. Do not write a sentence, preamble, explanation, or reasoning."
REWRITE_SYSTEM = "Rewrite questions without answering them or adding facts."
REWRITE_INSTRUCTION = (
    "Rewrite the question below as four natural, semantically equivalent questions. "
    "Preserve every factual constraint and do not reveal or guess the answer. "
    "Return exactly four lines beginning P1:, P2:, P3:, and P4:.\n\nQuestion: {q}"
)
FALLBACK_TEMPLATES = (
    "Give the shortest factual answer to this question: {q}",
    "Identify only the entity or value requested here: {q}",
    "Without explanation, answer this question: {q}",
    "Please answer this factual question directly: {q}",
)


def chat_ids(tokenizer, system: str, user: str) -> list[int]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
    except TypeError:
        return list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def padded(ids_list: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, ids_list))
    values = torch.full((len(ids_list), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(values)
    for index, ids in enumerate(ids_list):
        values[index, width - len(ids) :] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[index, width - len(ids) :] = 1
    return values, mask


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def parse_paraphrases(text: str, question: str) -> tuple[list[str], str]:
    found = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:P?[1-4]|[-*])\s*[:.)-]?\s*", "", line).strip()
        if cleaned and normalize(cleaned) != normalize(question):
            found.append(cleaned)
    deduplicated = []
    seen = set()
    for value in found:
        key = normalize(value)
        if key and key not in seen:
            seen.add(key)
            deduplicated.append(value)
    status = "model_four_unique" if len(deduplicated) >= 4 else "fallback_completed"
    for template in FALLBACK_TEMPLATES:
        if len(deduplicated) >= 4:
            break
        candidate = template.format(q=question)
        if normalize(candidate) not in seen:
            deduplicated.append(candidate)
            seen.add(normalize(candidate))
    return deduplicated[:4], status


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    midpoint = (left + right) / 2
    return entropy(midpoint) - (entropy(left) + entropy(right)) / 2


def generated_answers(
    model,
    tokenizer,
    prompts: list[list[int]],
    pad_id: int,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[list[str], list[float]]:
    ids, mask = padded(prompts, pad_id, device)
    width = ids.shape[1]
    output = model.generate(
        ids,
        attention_mask=mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    generated = output.sequences[:, width:]
    answers = [tokenizer.decode(row, skip_special_tokens=True).strip() for row in generated]
    eos_ids = {int(value) for value in ([tokenizer.eos_token_id] if isinstance(tokenizer.eos_token_id, int) else tokenizer.eos_token_id or [])}
    confidences = []
    for row_index in range(generated.shape[0]):
        logps = []
        for step, logits in enumerate(output.scores):
            token = int(generated[row_index, step])
            if token in eos_ids:
                break
            logps.append(float(torch.log_softmax(logits[row_index].float(), dim=-1)[token]))
        confidences.append(math.exp(sum(logps) / len(logps)) if logps else 0.0)
    return answers, confidences


def teacher_forced_nll(
    model,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    answer_lengths: list[int],
    pad_id: int,
    device: torch.device,
) -> list[float]:
    ids, mask = padded(sequences, pad_id, device)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    output = []
    for index, sequence in enumerate(sequences):
        left_padding = ids.shape[1] - len(sequence)
        start = left_padding + prompt_lengths[index] - 1
        length = answer_lengths[index]
        target = ids[index, start + 1 : start + 1 + length]
        answer_logits = logits[index, start : start + length]
        output.append(float(-torch.log_softmax(answer_logits, dim=-1).gather(1, target[:, None]).mean()))
    return output


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument("--max-rewrite-tokens", type=int, default=160)
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frozen-source-authorization", type=Path)
    args = parser.parse_args()

    benchmark, rows, locked_accessed, authorization_id = load_rows(
        benchmark_path=args.benchmark,
        cell=args.cell,
        authorization_path=args.frozen_source_authorization,
        stage="raw_natural_paraphrase",
        output_path=args.output,
    )
    if locked_accessed and args.limit is not None:
        raise ValueError("frozen source generation forbids --limit")
    require_authorized_argument(
        args.frozen_source_authorization,
        cell=args.cell,
        name="model_path",
        value=args.model_path,
    )
    raw = args.benchmark.read_bytes()
    if args.limit is not None:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_causal_model(args.model_path, load_in_4bit=False, torch_dtype=torch.bfloat16).eval()
    device = next(model.parameters()).device
    pad_id = int(tokenizer.pad_token_id)
    records = []
    start_time = time.time()

    for offset in range(0, len(rows), args.batch_rows):
        batch = rows[offset : offset + args.batch_rows]
        rewrite_prompts = [
            chat_ids(tokenizer, REWRITE_SYSTEM, REWRITE_INSTRUCTION.format(q=row["question"]))
            for _, row in batch
        ]
        rewrite_texts, _ = generated_answers(
            model, tokenizer, rewrite_prompts, pad_id, device, args.max_rewrite_tokens
        )
        panels = [parse_paraphrases(text, row["question"]) for text, (_, row) in zip(rewrite_texts, batch)]

        answer_prompts = []
        prompt_groups = []
        for (_, row), (paraphrases, _) in zip(batch, panels):
            prompts = [chat_ids(tokenizer, ANSWER_SYSTEM, row["question"])] + [
                chat_ids(tokenizer, ANSWER_SYSTEM, paraphrase) for paraphrase in paraphrases
            ]
            prompt_groups.append(prompts)
            answer_prompts.extend(prompts)

        prompt_ids, prompt_mask = padded(answer_prompts, pad_id, device)
        answer_start_logits = model(input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=False).logits[:, -1].float()
        panel_answers, panel_confidences = generated_answers(
            model, tokenizer, answer_prompts, pad_id, device, args.max_answer_tokens
        )

        canonical_sequences = []
        canonical_prompt_lengths = []
        canonical_answer_lengths = []
        for (_, row), prompts in zip(batch, prompt_groups):
            answer_ids = tokenizer.encode(row["generated_answer"], add_special_tokens=False)
            for prompt in prompts:
                canonical_sequences.append(prompt + answer_ids)
                canonical_prompt_lengths.append(len(prompt))
                canonical_answer_lengths.append(len(answer_ids))
        canonical_nlls = teacher_forced_nll(
            model,
            canonical_sequences,
            canonical_prompt_lengths,
            canonical_answer_lengths,
            pad_id,
            device,
        )

        candidate_sequences = []
        candidate_prompt_lengths = []
        candidate_answer_lengths = []
        candidate_meta = []
        for local, ((_, row), (paraphrases, status)) in enumerate(zip(batch, panels)):
            panel_slice = panel_answers[local * 5 : (local + 1) * 5]
            paraphrase_answers = panel_slice[1:]
            canonical_key = normalize(row["generated_answer"])
            alternatives = [answer for answer in paraphrase_answers if normalize(answer) and normalize(answer) != canonical_key]
            counts = Counter(normalize(answer) for answer in alternatives)
            if counts:
                candidate_key, support = counts.most_common(1)[0]
                candidate = next(answer for answer in alternatives if normalize(answer) == candidate_key)
                candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
            else:
                candidate = ""
                candidate_ids = []
                support = 0
            candidate_meta.append((candidate, support, counts, status, paraphrases, panel_slice))
            if candidate_ids:
                canonical_prompt = prompt_groups[local][0]
                candidate_sequences.append(canonical_prompt + candidate_ids)
                candidate_prompt_lengths.append(len(canonical_prompt))
                candidate_answer_lengths.append(len(candidate_ids))
        candidate_nll_values = teacher_forced_nll(
            model,
            candidate_sequences,
            candidate_prompt_lengths,
            candidate_answer_lengths,
            pad_id,
            device,
        ) if candidate_sequences else []
        candidate_nll_iterator = iter(candidate_nll_values)

        for local, (split, row) in enumerate(batch):
            base = local * 5
            distributions = torch.softmax(answer_start_logits[base : base + 5], dim=-1)
            canonical_js = [float(js_divergence(distributions[0], distributions[index])) for index in range(1, 5)]
            all_pairwise = [
                float(js_divergence(distributions[left], distributions[right]))
                for left in range(5)
                for right in range(left + 1, 5)
            ]
            top_probs, top_ids = distributions.max(dim=-1)
            _, frequencies = torch.unique(top_ids, return_counts=True)
            row_nlls = canonical_nlls[base : base + 5]
            changes = [value - row_nlls[0] for value in row_nlls[1:]]
            candidate, support, counts, status, paraphrases, panel_slice = candidate_meta[local]
            candidate_nll = next(candidate_nll_iterator) if candidate else None
            candidate_margin = candidate_nll - row_nlls[0] if candidate_nll is not None else None
            confidences = panel_confidences[base : base + 5]

            record = {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "rewrite_raw_output": rewrite_texts[local],
                    "rewrite_status": status,
                    "paraphrases": paraphrases,
                    "canonical_answer": row["generated_answer"],
                    "panel_answers": panel_slice,
                    "panel_answer_confidences": confidences,
                    "candidate_answer": candidate,
                    "candidate_support": int(support),
                    "features": {
                        "answer_start_mi": float(entropy(distributions.mean(dim=0)) - entropy(distributions).mean()),
                        "canonical_panel_js_mean": float(sum(canonical_js) / len(canonical_js)),
                        "canonical_panel_js_max": float(max(canonical_js)),
                        "canonical_panel_js_variance": float(torch.tensor(canonical_js).var(unbiased=False)),
                        "all_pairwise_js_mean": float(sum(all_pairwise) / len(all_pairwise)),
                        "answer_start_top1_disagreement": 1.0 - float(frequencies.max()) / 5.0,
                        "answer_start_unique_top1": float(frequencies.numel()),
                        "answer_start_top1_probability_range": float(top_probs.max() - top_probs.min()),
                        "canonical_answer_nll": float(row_nlls[0]),
                        "cross_prompt_nll_signed_mean": float(sum(changes) / len(changes)),
                        "cross_prompt_nll_absolute_mean": float(sum(abs(value) for value in changes) / len(changes)),
                        "cross_prompt_nll_max": float(max(changes)),
                        "canonical_cluster_retention": float(sum(normalize(answer) == normalize(row["generated_answer"]) for answer in panel_slice[1:]) / 4),
                        "string_disagreement": float(sum(normalize(answer) != normalize(row["generated_answer"]) for answer in panel_slice[1:]) / 4),
                        "largest_alternative_cluster": float(support / 4),
                        "top_two_alternative_margin": float((counts.most_common(2)[0][1] - (counts.most_common(2)[1][1] if len(counts) > 1 else 0)) / 4) if counts else 0.0,
                        "paraphrase_confidence_gain": float(max(confidences[1:]) - row["confidence"]),
                        "canonical_candidate_nll_margin": float(candidate_margin) if candidate_margin is not None else 0.0,
                        "candidate_available": float(bool(candidate)),
                    },
                }
            if not locked_accessed:
                record["label"] = row["class"]
            records.append(record)
        print(f"{min(offset + len(batch), len(rows))}/{len(rows)} elapsed={time.time() - start_time:.1f}", flush=True)

    payload = {
        "schema_version": "one-confidence-natural-paraphrase-features-v1",
        "cell": args.cell,
        "model_path": args.model_path,
        "dtype": "bfloat16",
        "benchmark_sha256": hashlib.sha256(raw).hexdigest(),
        "locked_accessed": locked_accessed,
        "source_authorization_id": authorization_id,
        "rewrite_protocol": {
            "system": REWRITE_SYSTEM,
            "instruction": REWRITE_INSTRUCTION,
            "decoding": "greedy",
            "panel_size": 4,
            "fallback_templates": FALLBACK_TEMPLATES,
        },
        "answer_protocol": {"system": ANSWER_SYSTEM, "decoding": "greedy", "qwen_thinking": False},
        "gold_visible": False if locked_accessed else "labels_only",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
