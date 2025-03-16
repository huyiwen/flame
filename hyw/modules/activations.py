from typing import List, Union

import torch

from fla.utils import autocast_custom_bwd, autocast_custom_fwd
from fla.modules.activations import swiglu_fwd, swiglu_fwd_torch
from hyw.ops.group_gemm.tgroup_gemm import grouped_gemm
from hyw.ops.group_gemm.tgroup_gemm_backwards import grouped_gemm_backward


@torch.compile
def swiglu_fwdbwd_grouped_torch(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, dout: torch.Tensor, group_sizes: torch.Tensor):
    dtype = x.dtype
    x, y = x.float(), y.float()
    x_sigmoid = x.sigmoid()
    x_swish = x * x_sigmoid
    z = x_swish * y
    g, dw = grouped_gemm_backward(dout, z.to(dout.dtype), w, group_sizes)
    g = g.float()
    dx = x_sigmoid * (1 + x * (1.0 - x_sigmoid)) * g * y
    dy = x_swish * g

    return dx.to(dtype), dy.to(dtype), dw.to(dtype)


class SwiGLUGroupedLinearFunction(torch.autograd.Function):
    r"""
    Swish-Gated Linear Unit (SwiGLU) function followed by a Grouped linear transformation.

    .. math::
        \text{SwiGLULinear}(x_i, y_i, W_i) = (swish(x_i) * y_i) W_i

    This simple wrap discards the intermediate results of SwiGLU(x, y) to save memory.
    """

    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, group_sizes: torch.Tensor):
        with torch.no_grad():
            if torch.compiler.is_compiling() or isinstance(x, torch.distributed.tensor.DTensor):
                z = swiglu_fwd_torch(x, y)
            else:
                z = swiglu_fwd(x, y)
        out = grouped_gemm(z, weight, group_sizes, True)
        # We don't store z, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(x, y, weight, group_sizes)
        return out

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, dout: torch.Tensor, *args):
        x, y, weight, group_sizes = ctx.saved_tensors
        with torch.no_grad():
            dx, dy, dw = swiglu_fwdbwd_grouped_torch(x, y, weight, dout, group_sizes)
        return dx, dy, dw, None



def swiglu_grouped_linear(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, group_sizes: Union[List[int], torch.Tensor]):
    # Convert group_sizes to tensor if it's a list
    if isinstance(group_sizes, list):
        group_sizes = torch.tensor(group_sizes, device=x.device, dtype=torch.int32)

    # Ensure correct dtype
    if group_sizes.dtype != torch.int32:
        group_sizes = group_sizes.to(torch.int32)

    return SwiGLUGroupedLinearFunction.apply(x, y, weight, group_sizes)

