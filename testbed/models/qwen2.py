from functools import lru_cache
import itertools
from typing import Any, Dict, List, Optional, Union
from PIL.Image import Image
import warnings
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)

from .model_base import ModelBase
import torch
import torch_npu


class Qwen2(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoProcessor,
        model_class=AutoModelForCausalLM,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

    @property
    def default_prompt_template(self):
        @lru_cache
        def warn_once(msg):
            warnings.warn(msg)

        if self.model_name.startswith("Qwen2-") or self.model_name.startswith(
            "Qwen1.5-"
        ):
            # fmt: off
            return (
                "{% if messages[0]['role'].lower() in ['instruction', 'system'] %}"
                    "{{ '<|im_start|>' + messages[0]['role'] + '\n' + messages[0]['content'] + '<|im_end|>\n'}}"
                    "{% set messages = messages[1:] %}"
                "{% else %}"
                    "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
                "{% endif %}"
                "{% set first_role = messages[0]['role'] %}"
                "{% set ns = namespace(generation_role='assistant') %}"
                "{% for message in messages %}"
                    "{% if loop.last or loop.nextitem['role'] == first_role %}"
                        "{% set ns.generation_role = message['role'] %}"
                    "{% endif %}"
                    "{{ '<|im_start|>' + message['role'] + '\n'}}"
                    "{% if 'content' in message %}"
                        "{{ message['content'] + '<|im_end|>' + '\n' }}"
                    "{% endif %}" 
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                    "{{ '<|im_start|>' + ns.generation_role + '\n' }}"
                "{% endif %}"
            )
            # fmt: on
        else:
            warn_once(
                f"The model {self.model_name} is not in official qwen1.5/qwen2/qwen2.5 collections, "
                "see https://huggingface.co/Qwen for more information. "
                "Please either customize your own prompt template for this model, "
                "or set `model_name` to select a default prompt template."
            )
            return super().default_prompt_template
