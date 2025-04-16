from functools import lru_cache
import itertools
from typing import Any, Dict, List, Optional, Union
from PIL.Image import Image
import warnings
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)

from .model_base import ModelBase
import torch
import torch_npu


HF_LLAVA = {
    "llava-1.5": ["llava-1.5-7b-hf", "llava-1.5-13b-hf", "bakLlava-v1-hf"],
    "llava-interleave": [
        "llava-interleave-qwen-0.5b-hf",
        "llava-interleave-qwen-7b-hf",
        "llava-interleave-qwen-7b-dpo-hf",
    ],
}


class LLaVa(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoProcessor,
        model_class=LlavaForConditionalGeneration,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            support_models=list(name for v in HF_LLAVA.values() for name in v),
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )
        self.processor.patch_size = self.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = (
            self.config.vision_feature_select_strategy
        )

    @property
    def default_prompt_template(self):
        # see https://huggingface.co/docs/transformers/main/model_doc/llava
        # make sure you download from hf official llava, otherwise you should use customize your own prompt template,
        @lru_cache
        def warn_once(msg):
            warnings.warn(msg)

        if self.model_name in HF_LLAVA["llava-1.5"]:
            # modified from https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/main/chat_template.json
            # fmt: off
            return (
                "{% if messages[0]['role'].lower() in ['instruction', 'system'] %}"
                    "{{ messages[0]['content'] + '\n' }}"
                    "{% set messages = messages[1:] %}"
                "{% endif %}"
                "{% set first_role = messages[0]['role'] %}"
                "{% set ns = namespace(generation_role='ASSISTANT') %}"
                "{% for message in messages %}"
                    "{% if loop.last or loop.nextitem['role'] == first_role %}"
                        "{% set ns.generation_role = message['role'] %}"
                    "{% endif %}"
                    "{{ message['role'].upper() }}"
                    "{% if 'content' in message %}"
                        "{{ ': ' }}"
                        "{# Render all images first #}"
                        "{% for line in message['content'] | selectattr('type', 'equalto', 'image')%}"
                            "{{ '<image>\n' }}"
                        "{% endfor %}"
                        "{# Render all text next #}"
                        "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
                            "{% if message['role'] != ns.generation_role %}"
                                "{{ content['text'] + ' ' }}"
                            "{% else %}"
                                "{% generation %}"
                                    "{{ content['text'] + ' '}}"
                                "{% endgeneration %}"
                            "{% endif %}"
                        "{% endfor %}"
                    "{% else %}"
                        "{{ ':' }}"
                    "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                    "{{ ns.generation_role.upper() + ':' }}"
                "{% endif %}"
            )
            # fmt: on
        elif self.model_name in HF_LLAVA["llava-interleave"]:
            # modified from https://huggingface.co/llava-hf/llava-interleave-qwen-0.5b-hf/blob/main/chat_template.json
            # fmt: off
            return (
                "{% if messages[0]['role'].lower() in ['instruction', 'system'] %}"
                    "{{ '<|im_start|>' + messages[0]['role'] + '\n' + messages[0]['content'] + '<|im_end|>\n' }}"
                    "{% set messages = messages[1:] %}"
                "{% endif %}"
                "{% set first_role = messages[0]['role'] %}"
                "{% set ns = namespace(generation_role='assistant') %}"
                "{% for message in messages %}"
                    "{% if loop.last or loop.nextitem['role'] == first_role %}"
                        "{% set ns.generation_role = message['role'] %}"
                    "{% endif %}"
                    "{{ '<|im_start|>' + message['role'] + '\n' }}"
                    "{% if 'content' in message %}"
                        "{# Render all images first #}"
                        "{% for line in message['content'] | selectattr('type', 'equalto', 'image')%}"
                            "{{ '<image>' }}"
                        "{% endfor %}"
                        "{# Render all text next #}"
                        "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
                            "{% if message['role'] != ns.generation_role %}"
                                "{{ '\n' + content['text'] }}"
                            "{% else %}"
                                "{% generation %}"
                                    "{{ '\n' + content['text'] }}"
                                "{% endgeneration %}"
                            "{% endif %}"
                        "{% endfor %}"
                        "{{ '<|im_end|>\n' }}"
                    "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                    "{{ '<|im_start|>' + ns.generation_role + '\n' }}"
                "{% endif %}"
            )
            # fmt: on
        else:
            warn_once(
                f"The model {self.model_name} is not in official llava list {', '.join(name for v in HF_LLAVA.values() for name in v)}. "
                "Please either customize your own prompt template for this model, "
                "or set `model_name` to select a default prompt template."
            )
            return super().default_prompt_template

    def process_input(
        self,
        images: Union[List[Image], List[List[Image]]],
        text: Union[
            List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]
        ],
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Processes text and image inputs for the model.

        Args:
            images (Union[List[Image], List[List[Image]]]):
                A list of images or a list of lists of images. For unbatched input, this should be a single-level list
                of images. For batched input, this should be a nested list where each inner list represents a batch of images.
                Each image should be an instance of the `Image` class.

            text (Union[List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]]):
                A list of texts or a list of lists of texts. For unbatched input, this should be a single-level list
                where each item is either a string or a dictionary. For batched input, this should be a nested list
                (list of lists) where each inner list represents a batch of texts. Dictionaries can follow the
                transformers' conversation format, with keys like "role" and "content".

            prompt_template (str, optional):
                A Jinja template which will be used to convert lists of messages in a chat into a tokenizable string.

            **kwargs:
                Additional keyword arguments passed to the `processor`.

        Returns:
            The output of the `processor` function, which is the processed input ready for the model.
        """
        if isinstance(text[0], dict) or (
            isinstance(text[0], list) and isinstance(text[0][0], dict)
        ):
            text = self.apply_prompt_template(text, prompt_template=prompt_template)  # type: ignore[arg-type]
        
        if isinstance(images[0], list):
            # llava doesn't support images with type List[List[Image]]
            images = list(itertools.chain(*images))

        return self.processor(
            images=images,
            text=text,
            padding=kwargs.pop("padding", True),
            return_tensors=kwargs.pop("return_tensors", "pt"),
            **kwargs,
        )