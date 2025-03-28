# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from hyw.models.hyw_dense.configuration_dense import DenseConfig
from hyw.models.hyw_dense.modeling_dense import (
    DenseForCausalLM, DenseModel)

AutoConfig.register(DenseConfig.model_type, DenseConfig)
AutoModel.register(DenseConfig, DenseModel)
AutoModelForCausalLM.register(DenseConfig, DenseForCausalLM)


__all__ = ['DenseConfig', 'DenseForCausalLM', 'DenseModel']
