"""Shared causal-model loading with optional 8-bit quantization."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def load_causal_model(
    path: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    torch_dtype: torch.dtype = torch.float16,
):
    kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    if load_in_8bit and load_in_4bit:
        raise ValueError("choose only one quantization mode")
    if load_in_4bit:
        kwargs.update({
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            ),
            "device_map": {"": 0},
        })
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    elif load_in_8bit:
        kwargs.update({
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "device_map": {"": 0},
        })
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs).cuda()
    model.eval()
    model.config.use_cache = False
    return model
