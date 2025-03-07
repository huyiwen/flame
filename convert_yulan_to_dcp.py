# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from safetensors import safe_open

from torchtitan.logging import init_logger, logger


@torch.inference_mode()
def convert_hf_weights(model: str, checkpoint: str):
    logger.info(f"Loading model from {model}")
    state_dict = {}
    with safe_open(f"{model}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if len(state_dict[k].shape) == 0:
                state_dict[k] = state_dict[k].view(1)
    state_dict['pad.weight'] = torch.zeros(1, 62)
    pad = torch.zeros(8, state_dict['model.embed_tokens.weight'].shape[1])
    torch.nn.init.normal_(pad, mean=0.0, std=0.00005)
    state_dict['model.embed_tokens.weight'] = torch.cat(
        [state_dict['model.embed_tokens.weight'], pad], dim=0)
    state_dict['lm_head.weight'] = torch.cat(
        [state_dict['lm_head.weight'], pad], dim=0)
    print(state_dict['model.embed_tokens.weight'].shape)
    # model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, state_dict=state_dict, torch_dtype=torch.bfloat16)
    # state_dict = model.state_dict()

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint,
                                                     thread_count=8)
    DCP.save({"model": state_dict}, storage_writer=storage_writer)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(
        description="Convert huggingface-style model weights to DCP format.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    convert_hf_weights(args.model, args.checkpoint)
    # /data/flame/.venv/bin/python /data/flame/convert_hf_to_dcp.py --model /data/metadata/output/miniyulan-2B-final-stage16-nooptim/checkpoint-155587 --checkpoint /data/metadata/output/miniyulan-2B-final-stage16-tt/step-0
    # /data/flame/.venv/bin/python /data/flame/convert_hf_to_dcp.py --model /data/metadata/reference-models/YuLan-Mini-Pub --checkpoint /data/metadata/output/YuLan-Mini-Pub-tt/step-0
