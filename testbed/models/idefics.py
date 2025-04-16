import torch
import torch_npu

from functools import partial
from typing import Any, Dict, List, Optional, Union
import transformers
from packaging import version
from PIL.Image import Image
from transformers import (
    IdeficsForVisionText2Text,
    AutoProcessor,
)

from testbed.models.model_base import ModelBase


class Idefics(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoProcessor,
        model_class=IdeficsForVisionText2Text,
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
        # see https://arxiv.org/pdf/2306.16527
        # fmt: off
        return (
            "{% if messages[0]['role'].lower() in ['instruction', 'system'] %}"
                "{{ messages[0]['role'].capitalize() + ': ' + messages[0]['content'] + '\n'}}"
                "{% set messages = messages[1:] %}"
            "{% endif %}"
            "{% set first_role = messages[0]['role'] %}"
            "{% set ns = namespace(generation_role='Assistant') %}"
            "{% for message in messages %}"
                "{% set is_end_of_round = loop.last or loop.nextitem['role'] == first_role %}"
                "{% if message['role'] != '' %}"
                    "{{ message['role'].capitalize() }}"
                    "{% if is_end_of_round %}"
                        "{% set ns.generation_role = message['role'] %}"
                    "{% endif %}"
                    "{% if not 'content' in message or message['content'][0]['type'] == 'image' %}"
                        "{{':'}}"
                    "{% else %}"
                        "{{': '}}"
                    "{% endif %}" 
                "{% endif %}"
                "{% if 'content' in message %}"
                    "{% for line in message['content'] %}"
                        "{% if line['type'] == 'text' %}"
                            "{{ line['text'] }}"
                        "{% elif line['type'] == 'image' %}"
                            "{{ '<image>' }}"
                        "{% endif %}"
                        "{% if not loop.last %}"
                            "{{ ' ' }}"
                        "{%+ endif %}"
                    "{% endfor %}"
                    "{% if is_end_of_round %}"
                        "{{ '\n' }}"
                    "{% else %}"
                        "{{ ' ' }}"
                    "{% endif %}"
                "{% endif %}" 
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ ns.generation_role.capitalize() + ':' }}"
            "{% endif %}"
        )
        # fmt: on

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
            text (str, List[str], List[Dict[str, Any]], List[List[Dict[str, Any]]]):
                A single string, a list of strings or dictionaries, or a nested list (batch) of strings/dictionaries.
                For unbatched input (single text), this should be a string or a list of dict, where each item is
                either a string or a doct (following the transformers' conversation format with
                keys like "role" and "content").
                For batched input, this should be a nested list (list of lists) or a list of strings

            images (Union[List[Image], List[List[Image]]]):
                A list of images or a list of lists of images. For unbatched input, this should be a single-level list
                of images. For batched input, this should be a nested list where each inner list represents a batch of images.
                Each image should be an instance of the `Image` class.

            prompt_template (str, optional):
                An optional template string used to format the input texts if they are provided as dictionaries.

            **kwargs:
                Additional keyword arguments passed to the `processor`.

        Returns:
            The output of the `processor` function, which is the processed input ready for the model.
        """
        if isinstance(text, str) or (
            isinstance(text, list) and isinstance(text[0], dict)
        ):
            text = [text]
            images = [images]

        if isinstance(text[0][0], dict):
            text = self.apply_prompt_template(text, prompt_template=prompt_template)

        assert len(text) == len(images)
        inputs = []
        for i, (ctx, image_list) in enumerate(zip(text, images)):
            text_parts = ctx.split("<image>")

            if len(text_parts) - 1 != len(image_list):
                raise ValueError(
                    f"In the {i}-th input, the number of images {len(image_list)} does "
                    f"not match the number of image tokens {len(text_parts) - 1} in the text."
                )
            result = []
            for seg, image in zip(text_parts, image_list):
                if seg != "":
                    result.append(seg)
                result.append(image)
            if text_parts[-1] != "":  # the last question without answer
                result.append(text_parts[-1])
            inputs.append(result)

        if version.parse(transformers.__version__) < version.parse("4.46.0"):
            process = partial(self.processor, prompts=inputs)
        else:
            process = partial(self.processor, text=inputs)

        return process(
            padding=kwargs.pop("padding", True),
            return_tensors=kwargs.pop("return_tensors", "pt"),
            **kwargs,
        )
