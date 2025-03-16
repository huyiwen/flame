from functools import partial
from typing import Callable, Optional, Tuple

import torch
import transformer_engine as te
from torch import nn
from megatron.core.tensor_parallel.utils import divide
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl


class TEGroupedLinear(te.pytorch.GroupedLinear):
    """
    Wrapper for the Transformer-Engine's `GroupedLinear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().
    """

    def __init__(
        self,
        num_gemms: int,
        input_size: int,
        output_size: int,
        initializer_range: float,
        parallel_mode: Optional[str],
        is_expert: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        expert_model_parallel_size: int = 1,
    ):

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = False

        extra_kwargs = {"device": torch.cuda.current_device(), "params_dtype": torch.float32, "ub_name": tp_comm_buffer_name}

        self.expert_parallel = expert_model_parallel_size > 1
        if is_expert:
            extra_kwargs["rng_tracker_name"] = 'expert-parallel-rng'

        super().__init__(
            num_gemms=num_gemms,
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=False,
            fuse_wgrad_accumulation=True,
            tp_group=None,
            tp_size=1,
            get_rng_state_tracker=None,
            init_method=partial(
                torch.nn.init.trunc_normal_, std=initializer_range, a=-3*initializer_range, b=3*initializer_range
            ),
            bias=False,
            return_bias=False,
            parallel_mode=None,
            **extra_kwargs,
        )

        for param in self.parameters():
            setattr(param, 'allreduce', not (is_expert and self.expert_parallel))

    def forward(self, x, m_splits):
        """Forward."""
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        out = super().forward(x, m_splits, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        return out


class TEGroupedMLP(nn.Module):
    """An efficient implementation of the Experts layer using TE's GroupedLinear.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        gated_linear_unit: bool = True,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.input_size = hidden_size

        # Double the output width with gated linear unit, see https://arxiv.org/pdf/2002.05202.pdf
        if gated_linear_unit:
            intermediate_size *= 2

        self.linear_fc1 = TEGroupedLinear(
            num_gemms=self.num_local_experts,
            input_size=self.input_size,
            output_size=intermediate_size,
            is_expert=True,
            parallel_mode="row",
            tp_comm_buffer_name='fc1',
            initializer_range=initializer_range,
        )

        self.activation_func = partial(bias_swiglu_impl, bias=None)

        self.linear_fc2 = TEGroupedLinear(
            num_gemms=self.num_local_experts,
            input_size=self.input_size,
            output_size=intermediate_size,
            is_expert=True,
            parallel_mode="column",
            tp_comm_buffer_name='fc2',
            initializer_range=initializer_range,
        )

    def forward(
        self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of TEGroupedMLP

        Args:
            permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
            local experts.
            tokens_per_expert (torch.Tensor): The number of tokens per expert.

        Return:
            output (torch.Tensor): The output of the local experts.
        """
        tokens_per_expert = tokens_per_expert.tolist()

        intermediate_parallel = self.linear_fc1(
            permuted_local_hidden_states, tokens_per_expert
        )

        intermediate_parallel = self.activation_func(intermediate_parallel)

        output = self.linear_fc2(intermediate_parallel, tokens_per_expert)

        return output
