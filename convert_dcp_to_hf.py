# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import io
import math
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import timedelta

import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          LlamaConfig)

from torchtitan.logging import init_logger, logger


def get_target_state_dict(source_state_dict, method="rebalanced"):

    alpha_mapping = {
        ".embed_tokens": ".embed_tokens_alpha",
        ".q_proj": ".q_proj_alpha",
        ".k_proj": ".k_proj_alpha",
        ".v_proj": ".v_proj_alpha",
        ".o_proj": ".o_proj_alpha",
        ".mlp.down_proj": ".down_proj_alpha",
        ".mlp.gate_proj": ".gate_up_proj_alpha",
        ".mlp.up_proj": ".gate_up_proj_alpha",
        ".input_layernorm": ".input_layernorm_alpha",
        ".post_attention_layernorm": ".post_attention_layernorm_alpha",
        ".norm": ".norm_alpha",
        "lm_head": "lm_head_alpha"
    }

    count = defaultdict(int)
    target_state_dict = {}
    for key, value in tqdm(source_state_dict.items()):
        if "alpha" in key:
            if "gate_up" in key:
                count[key] += 2
            else:
                count[key] += 1

        elif ".weight" in key:
            alpha_key = None
            done = False
            orig_key_norm = value.norm()
            for k, v in alpha_mapping.items():
                if k + ".weight" in key:
                    alpha_key = key.replace(k + ".weight", v)
                    if alpha_key not in source_state_dict:
                        logger.info(
                            f"Not found: {key} -> {alpha_key}. Just copying the weights without multiplying alpha."
                        )
                        target_state_dict[key] = value
                        done = True
                        break

                    alpha = source_state_dict[alpha_key]

                    if method == "rebalanced":
                        target_state_dict[key] = (value * alpha).to(
                            torch.bfloat16)
                        done = True
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    count[alpha_key] -= 1
                    key_norm = target_state_dict[key].norm()
                    alpha_norm = source_state_dict[alpha_key].item()
                    logger.info(
                        f">>> {key}, {orig_key_norm}, {alpha_norm}, {alpha_key}, {key_norm}"
                    )
                    break

            if not done and key not in {'pad.weight'}:
                raise ValueError(f"Not found {key}")
        else:
            target_state_dict[key] = value

    for key, value in count.items():
        if value != 0:
            logger.info("\033[91mNot found: " + key + " " + str(value) +
                        "\033[0m")

    return target_state_dict


@torch.inference_mode()
def save_pretrained(checkpoint: str, path: str, config: str, tokenizer: str):
    if os.path.exists(path):
        logger.info(f"Removing the existing directory {path}")
        shutil.rmtree(path)
    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)
    orig_config = None

    logger.info(f"Saving the config to {path}")
    if config.model_type == "yulanmini":
        orig_config = config
        config = LlamaConfig(
            attention_bias=True,
            attention_dropout=config.attention_dropout,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            head_dim=config.hidden_size // config.num_attention_heads,
            hidden_act=config.hidden_act,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            mlp_bias=False,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            pretraining_tp=1,
            rms_norm_eps=config.rms_norm_eps,
            rope_scaling=None,
            rope_theta=config.rope_theta,
            tie_word_embeddings=False,
            torch_dtype=torch.float32,
            use_cache=True,
            vocab_size=config.vocab_size,
        )
    config.save_pretrained(path)
    logger.info(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer,
                                              trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {path}")
    tokenizer.save_pretrained(path)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)
        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        # torch.load now with default weights_only=True will work
        state_dict = torch.load(checkpoint_path, map_location='cpu')['model']

    if orig_config is not None:
        # step 1: merge alpha weights
        state_dict = get_target_state_dict(state_dict)
        scale_depth = orig_config.hidden_states_shrink * math.sqrt(
            orig_config.num_hidden_layers)
        if not hasattr(orig_config, "scale_depth"):
            orig_config.scale_depth = scale_depth
        else:
            assert math.isclose(
                orig_config.scale_depth, scale_depth,
                rel_tol=1e-5), f"{orig_config.scale_depth} != {scale_depth}"

        state_dict["model.embed_tokens.weight"] = state_dict[
            "model.embed_tokens.weight"] * orig_config.scale_emb
        for i in range(orig_config.num_hidden_layers):
            state_dict[
                f"model.layers.{i}.self_attn.o_proj.bias"] = torch.zeros(
                    (orig_config.hidden_size, ),
                    dtype=state_dict[f"model.layers.{i}.mlp.down_proj.weight"].
                    dtype)
            state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = state_dict[
                f"model.layers.{i}.self_attn.o_proj.weight"] * orig_config.scale_depth / math.sqrt(
                    orig_config.num_hidden_layers)
            state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.down_proj.weight"] * orig_config.scale_depth / math.sqrt(
                    orig_config.num_hidden_layers)

    logger.info(f"Initializing the model from config\n{config}")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    logger.info(model)
    logger.info("Loading state dict from the checkpoint")

    model.load_state_dict(state_dict)

    logger.info(f"Saving the model to {path}")
    model.save_pretrained(path)


#  /data/flame/.venv/bin/python /data/flame/convert_dcp_to_hf.py --checkpoint /data/flame/exp/yulanmini-16/batch32.seqlen4096.gas1.lr0.001/checkpoint/step-4096 --path /data/flame/exp/yulanmini-16/batch32.seqlen4096.gas1.lr0.001/huggingface/step-4096 --config /data/metadata/output/miniyulan-2B-final-stage16-nooptim/checkpoint-155587 --tokenizer /data/metadata/output/miniyulan-2B-final-stage16-nooptim/checkpoint-155587
# /data/flame/.venv/bin/python /data/flame/convert_dcp_to_hf.py --checkpoint /data/flame/exp/yulanmini-16/batch32.seqlen4096.gas1.lr0.001/checkpoint/step-2048

if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(
        "Convert DCP format model weights to huggingface-style.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=
        "/data/metadata/output/miniyulan-2B-final-stage16-nooptim/checkpoint-155587"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=
        "/data/metadata/output/miniyulan-2B-final-stage16-nooptim/checkpoint-155587"
    )
    args = parser.parse_args()
    if args.path is None:
        if "checkpoint" in args.checkpoint:
            exp = "-".join(
                args.checkpoint.split("/checkpoint/")[0].split('/')[-2:])
            args.path = args.checkpoint.replace("/checkpoint/",
                                                f"/huggingface-{exp}/")
        else:
            raise ValueError("Please specify the path to save the model.")
    save_pretrained(args.checkpoint, args.path, args.config, args.tokenizer)
