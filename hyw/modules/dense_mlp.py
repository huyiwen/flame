# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from torch.distributed.tensor import (DeviceMesh, DTensor, Placement,
                                      Replicate, Shard, distribute_module)
from torch.distributed.tensor.parallel import ParallelStyle

from fla.modules.activations import swiglu, swiglu_linear


if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


def forward_block(weight: torch.Tensor, block_indices: torch.Tensor, x: torch.Tensor, block_size: int = 16, hidden_ratio: int = 4):
    
    bsz, _ = block_indices.shape
    intermediate_size, hidden_size = weight.shape
    block_x_num = hidden_size // block_size
    block_y_num = intermediate_size // block_size

    results = torch.zeros(bsz, intermediate_size, dtype=x.dtype, device=x.device)

    for i in range(block_x_num):
        for j in range(block_x_num):
            
            # load cur_x
            cur_x = []
            good_k = []
            for k in range(bsz):
                idx = (j - i + block_x_num) % block_x_num
                if block_indices[k].item() == idx:
                    cur_x.append(x[k, j*block_size:(j+1)*block_size])
                    good_k.append(k)
            if len(cur_x) == 0:
                continue
            cur_x = torch.stack(cur_x, dim=0)

            cur_results = F.linear(cur_x, weight[hidden_ratio*i*block_size:hidden_ratio*(i+1)*block_size, j*block_size:(j+1)*block_size])
            for ki, k in enumerate(good_k):
                results[k, hidden_ratio*i*block_size:hidden_ratio*(i+1)*block_size] = cur_results[ki]

    return results


def get_updated_expert_bias(tokens_per_expert, expert_bias, expert_bias_update_rate, update_type="recentered"):
    """Update expert bias for biased expert routing. See https://arxiv.org/abs/2408.15664v1#

    Args:
        tokens_per_expert (torch.Tensor): The number of tokens assigned to each expert.
        expert_bias (torch.Tensor): The bias for each expert.
        expert_bias_udpate_rate (float): The update rate for the expert bias.
    """
    with torch.no_grad():
        # All Reduce Across TPxCPxDP group
        # torch.distributed.all_reduce(
        #     tokens_per_expert,
        #     group=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
        # )
        average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]
        offset = average_tokens - tokens_per_expert
        if update_type == "recentered":
            updated_expert_bias = expert_bias + torch.sign(offset - offset.mean(dim=-1)) * expert_bias_update_rate
        else:
            updated_expert_bias = expert_bias + torch.sign(offset) * expert_bias_update_rate
        return updated_expert_bias


class DenseGatedMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
        fuse_swiglu: bool = True
    ) -> DenseGatedMLP:
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.fuse_swiglu = fuse_swiglu

        if hidden_act != 'swish':
            raise ValueError(f'Unsupported hidden_act: {hidden_act}')

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if self.fuse_swiglu:
            self.swiglu_linear = SwiGLULinear()

        self.block_size = 32
        self.block_x_num = hidden_size // self.block_size
        self.block_y_num = intermediate_size // self.block_size
        self.router = nn.Linear(hidden_size, self.block_x_num)

        self.register_buffer(
            "expert_bias", torch.zeros(self.block_x_num, dtype=torch.float32)
        )
        self.register_buffer(
            "local_tokens_per_expert", torch.zeros(self.block_x_num, dtype=torch.float32)
        )

        self.block_logger = None

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Unpack[Any]
    ) -> torch.Tensor:
        self._maintain_float32_expert_bias()

        logits = F.linear(x, self.router.weight)
        scores = torch.sigmoid(logits) + self.expert_bias
        block_indices = scores.topk(1, dim=-1)[1]
        topk_map = torch.zeros_like(logits).int().scatter(1, block_indices, 1).bool()

        gate = forward_block(self.gate_proj.weight, block_indices, x)
        y = forward_block(self.up_proj.weight, block_indices, x)
        self.block_logger("6_gate", gate)
        self.block_logger("6_y", y)
        return forward_block(self.down_proj.weight, block_indices, swiglu(gate, y))


class SwiGLULinear(nn.Module):

    def forward(self, x, y, weight, bias):
        return swiglu_linear(x, y, weight, bias)


class SwiGLULinearParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.desired_input_layouts = (Shard(-1),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        x, y, weight, bias = inputs
        if not isinstance(x, DTensor):
            x = DTensor.from_local(x, device_mesh, input_layouts, run_check=False)
        if x.placements != desired_input_layouts:
            x = x.redistribute(placements=desired_input_layouts, async_op=True)

        if not isinstance(y, DTensor):
            y = DTensor.from_local(y, device_mesh, input_layouts, run_check=False)
        if y.placements != desired_input_layouts:
            y = y.redistribute(placements=desired_input_layouts, async_op=True)

        if not isinstance(weight, DTensor):
            weight = DTensor.from_local(weight, device_mesh, (Shard(1),))

        if bias is not None and not isinstance(bias, DTensor):
            bias = DTensor.from_local(bias, device_mesh, (Replicate(),))

        return x, y, weight, bias

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            output_fn=partial(self._prepare_output_fn, self.output_layouts, self.use_local_output)
        )
