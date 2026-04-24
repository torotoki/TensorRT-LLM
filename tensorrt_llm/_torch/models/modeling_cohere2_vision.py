from typing import Optional

import torch
from torch import nn
from transformers import (AutoProcessor, AutoTokenizer, Cohere2VisionConfig, PreTrainedModel, PretrainedConfig)
from transformers.activations import ACT2FN

from ...inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ContentFormat,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_procesor,
)

from .modeling_auto import AutoModelForCausalLM
from .modeling_utils import ModelConfig, filter_weights, register_auto_model

class Cohere2InputProcessor(BaseMultimodalInputProcessor,
                             BaseMultimodalDummyInputsBuilder):
    
    def __init__(self,
                 model_path: str,
                 config: PretrainedConfig,
                 tokenizer: AutoTokenizer,
                 trust_remote_code: bool = True,
                 **kwargs):
        super().__init__(model_path=model_path,
                         config=config,
                         tokenizer=tokenizer,
                         trust_remote_code=trust_remote_code,
                         **kwargs)
        self._config = config
        self._tokenizer = tokenizer
        self._model_path = model_path
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=self.use_fast
        )
        self._dtype = self.config.torch_dtype

@register_auto_model("Cohere2ForConditionalGeneration")
@register_input_processor(
    Cohere2InputProcessor,
    model_type="cohere2_vision",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": "<START_OF_IMG>"},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        content_format=ContentFormat.STRING,
    ),
)
class Cohere2VisionModel(PreTrainedModel):
    def __init__(self, model_config: ModelConfig[Cohere2VisionConfig]):
        pass

