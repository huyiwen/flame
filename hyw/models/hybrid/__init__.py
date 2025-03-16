# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from hyw.models.hybrid.configuration_hybrid import HybridConfig
from hyw.models.hybrid.modeling_hybrid import HybridForCausalLM, HybridModel

AutoConfig.register(HybridConfig.model_type, HybridConfig)
AutoModel.register(HybridConfig, HybridModel)
AutoModelForCausalLM.register(HybridConfig, HybridForCausalLM)


__all__ = ['HybridConfig', 'HybridForCausalLM', 'HybridModel']
