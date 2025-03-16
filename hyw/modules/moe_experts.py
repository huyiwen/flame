from typing import List, Optional, Union

import torch
from torch import nn

from fla.modules.activations import swiglu, swiglu_grouped_linear
from fla.modules.mlp import GatedMLP
from fla.ops.group_gemm.tgroup_gemm import grouped_gemm_with_grad


class SwiGLUGroupedLinear(nn.Module):

    def forward(self, x, y, weight, group_sizes):
        return swiglu_grouped_linear(x, y, weight, group_sizes)


class GroupedExperts(nn.Module):
    """This class implements the grouped experts layer used in Mixture of Experts. Each expert
    is a variant of the Gated Linear Units network. See more details in https://arxiv.org/pdf/2002.05202.

    Args:
        hidden_size (int): Input dimension.
        intermediate_size (int): Output dimension.
        num_experts (int): Number of experts in this grouped experts layer. Default is 1.
        swiglu (bool): Whether to use gated linear unit. Default is True.
        activation (nn.Module): Activation function to use. Default is F.silu.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
        fuse_swiglu: bool = True,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.num_experts = num_experts

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
        self.fuse_swiglu = False
        self.initializer_range = initializer_range

        if hidden_act != 'swish':
            raise ValueError(f'Unsupported hidden_act: {hidden_act}')

        self.gate_up_proj = nn.Linear(self.hidden_size, self.num_experts * self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.num_experts * self.intermediate_size, self.hidden_size, bias=False)
        if self.fuse_swiglu:
            self.swiglu_grouped_linear = SwiGLUGroupedLinear()

    def forward(
        self,
        x: torch.Tensor,
        group_sizes: Union[torch.Tensor, List[int]],
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape (grouped tokens, hidden_size).

        Returns:
            torch.Tensor: with shape (grouped tokens, hidden_size).
        """
        if isinstance(group_sizes, list):
            group_sizes = torch.tensor(group_sizes, device=x.device, dtype=torch.int32)

        # Ensure correct dtype
        if group_sizes.dtype != torch.int32:
            group_sizes = group_sizes.to(torch.int32)

        if group_sizes.device != x.device:
            group_sizes = group_sizes.to(x.device)

        gate, y = grouped_gemm_with_grad(x, self.gate_up_proj.weight, group_sizes).chunk(2, dim=-1)
        # print(gate, y, flush=True)
        if self.fuse_swiglu:
            return self.swiglu_grouped_linear(gate, y, self.down_proj.weight, group_sizes)
        else:
            return grouped_gemm_with_grad(swiglu(gate.contiguous(), y.contiguous()), self.down_proj.weight, group_sizes)
