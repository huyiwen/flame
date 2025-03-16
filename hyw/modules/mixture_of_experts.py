from typing import Any, Callable, Literal, Optional, Tuple, Union, Unpack

import torch
import torch.nn.functional as F
from torch import nn

from fla.modules.moe_experts import GroupedExperts
from fla.modules.moe_mcore_experts import TEGroupedMLP
from fla.modules.moe_router import TopKRouter
from fla.modules.moe_token_dispatcher import MoEAllGatherTokenDispatcher


@torch.no_grad
def _calculate_max_vio(
    tokens_per_expert: torch.Tensor,
    expected_load: float,
) -> float:
    """Calculate the average of MaxVio_batch across all layers."""

    # Maximal expert load per token in MoE layer
    max_load = tokens_per_expert.max(dim=-1).values
    max_vio = (max_load - expected_load) / expected_load
    return max_vio.mean().item()


def load_balancing_loss_func(
    gate_logits: Tuple[torch.Tensor],
    num_experts: int,
    top_k: int,
    attention_mask: Optional[torch.Tensor] = None,
    global_batch: bool = False,
) -> Tuple[Union[torch.Tensor, Literal[0]], float]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0, 0

    compute_device = gate_logits[0].device
    num_hidden_layers = len(gate_logits)
    concatenated_gate_logits = torch.cat(
        [layer_gate.to(compute_device) for layer_gate in gate_logits],
        dim=0,
    )

    # [L x B x S, E]
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits,
                                                  dim=-1)

    # selected experts for each token (across L & B & S)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (attention_mask[None, :, :, None, None].expand(
            (num_hidden_layers, batch_size, sequence_length, top_k,
             num_experts)).reshape(-1, top_k, num_experts).to(compute_device))

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None].expand(
                (num_hidden_layers, batch_size, sequence_length,
                 num_experts)).reshape(-1, num_experts).to(compute_device))

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask,
            dim=0) / torch.sum(router_per_expert_attention_mask, dim=0)

    # Global load-balancing loss: http://arxiv.org/abs/2501.11873
    if global_batch:
        torch.distributed.all_reduce(tokens_per_expert, op=torch.distributed.ReduceOp.AVG)

    max_vio = _calculate_max_vio(tokens_per_expert, expected_load=1/num_experts)

    overall_loss = torch.sum(tokens_per_expert *
                             router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts, max_vio


class SparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_local_experts: int,
        num_experts_per_tok: int,
        expert_capacity_factor: Optional[float],
        routing_type: Literal["aux_loss", "bias"] = "aux_loss",
        router_jitter_noise: float = 0,
        n_shared_experts: int = 0,
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        router_topk_scaling_factor: Optional[float] = None,
        pad_expert_input_to_capacity: bool = False,  # need ablation
        aux_loss_coeff: float = 0,
        z_loss_coeff: float = 0.001,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.num_experts = num_local_experts
        self.n_shared_experts = n_shared_experts

        if routing_type == "aux_loss":
            assert aux_loss_coeff > 0
            router_kwargs = dict(
                routing_type="aux_loss",
                score_function="softmax",
                router_enable_expert_bias=False,
                router_pre_softmax=True,
                aux_loss_coeff=aux_loss_coeff,
            )
        elif routing_type == "bias":
            router_kwargs = dict(
                routing_type="none",
                score_function="sigmoid",
                router_enable_expert_bias=True,
            )

        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
            router_jitter_noise=router_jitter_noise,
            expert_capacity_factor=expert_capacity_factor,
            pad_expert_input_to_capacity=pad_expert_input_to_capacity,
            token_drop_policy="probs",
            z_loss_coeff=z_loss_coeff,
            router_num_groups=n_group,
            router_group_topk=topk_group,
            router_topk_scaling_factor=router_topk_scaling_factor,
            initializer_range=initializer_range,
            **router_kwargs,
        )

        self.token_dispatcher = MoEAllGatherTokenDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=list(range(num_local_experts)),
            num_experts_per_tok=num_experts_per_tok,
        )
        expert_kwargs = dict(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            initializer_range=initializer_range,
        )
        self.experts = GroupedExperts(num_local_experts, **expert_kwargs)
        self.shared_experts = GroupedExperts(
            n_shared_experts, **expert_kwargs
        ) if self.n_shared_experts > 0 else None

        self.shared_expert_overlap = False

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[Any]) -> torch.Tensor:
        probs, routing_map = self.router(hidden_states)
        # print(f"{probs.shape} {routing_map.shape}\n{probs}\n{routing_map}", flush=True)

        dispatched_input, tokens_per_expert = self.token_dispatcher.token_permutation(
            hidden_states, probs, routing_map
        )
        # print(f"{dispatched_input.shape} {tokens_per_expert}", flush=True)
        expert_output = self.experts(dispatched_input, tokens_per_expert)
        output = self.token_dispatcher.token_unpermutation(expert_output)

        if self.shared_experts is not None and not self.shared_expert_overlap:
            # if shared_expert_overlap is True, the expert calculation happens in
            # the token_dispatcher to overlap communications and computations
            output = output + self.shared_experts(hidden_states)
        return output, probs
