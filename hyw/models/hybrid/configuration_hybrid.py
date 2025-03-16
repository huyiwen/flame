# -*- coding: utf-8 -*-

from typing import Literal, Optional

from transformers.configuration_utils import PretrainedConfig


class HybridConfig(PretrainedConfig):

    model_type = 'hybrid'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int = None,
        qkv_bias: bool = False,
        window_size: Optional[int] = None,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: int = 2048,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        vocab_size: int = 32000,
        # native sparse attention
        nsa: bool = False,
        compress_block_size: Optional[int] = None,
        selection_block_size: Optional[int] = None,
        num_selected_blocks: Optional[int] = None,
        # mixture of experts
        topk_method: Literal['noaux_tc', 'aux'] = "aux",
        num_local_experts: int = 1,
        num_experts_per_tok: int = 1,
        router_jitter_noise: float = 0,
        router_aux_loss_coef: float = 0.001,
        global_batch_aux: bool = False,
        expert_capacity_factor: Optional[float] = None,
        # DeepSeek V3
        n_shared_experts: int = 0,
        update_rate: float = 1e-5,
        routed_scaling_factor: float = 2.5,  # copy from deepseek v3
        router_z_loss_coef: float = 0.001,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        n_routed_experts: Optional[int] = None,  # experts in a group
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_bias = qkv_bias
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.vocab_size = vocab_size

        # native sparse attention
        self.nsa = nsa
        self.compress_block_size = compress_block_size
        self.selection_block_size = selection_block_size
        self.num_selected_blocks = num_selected_blocks

        # mixture-of-experts
        self.num_local_experts = num_local_experts
        self.topk_method = topk_method
        self.update_rate = update_rate
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.expert_capacity_factor = expert_capacity_factor
        self.router_z_loss_coef = router_z_loss_coef
        assert num_experts_per_tok <= num_local_experts
        if n_routed_experts is None:
            n_routed_experts = num_local_experts // n_group
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.router_jitter_noise = router_jitter_noise
        self.router_aux_loss_coef = router_aux_loss_coef
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.global_batch_aux = global_batch_aux

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
