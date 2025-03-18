# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from hyw.models.hyw_xfmr.configuration_transformer import TransformerConfig
from hyw.models.hyw_xfmr.modeling_transformer import (
    TransformerForCausalLM, TransformerModel)

AutoConfig.register(TransformerConfig.model_type, TransformerConfig)
AutoModel.register(TransformerConfig, TransformerModel)
AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM)


__all__ = ['TransformerConfig', 'TransformerForCausalLM', 'TransformerModel']
