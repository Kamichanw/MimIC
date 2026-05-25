from functools import partial
from typing import Any, cast
import transformers
from packaging import version
from PIL.Image import Image
from transformers import (
    IdeficsForVisionText2Text,
    AutoProcessor,
)

from .model_base import ModelBase


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

    def process_input(
        self,
        images: list[Image] | list[list[Image]],
        text: list[str | dict[str, Any]] | list[list[str | dict[str, Any]]],
        prompt_template: str | None = None,
        **kwargs,
    ):
        if isinstance(text, str) or (
            isinstance(text, list) and isinstance(text[0], dict)
        ):
            text = [text]  # type: ignore[list-item]
            images = [images]  # type: ignore[list-item]

        if isinstance(text[0], list) and text[0] and isinstance(text[0][0], dict):
            text = self.apply_prompt_template(text, prompt_template=prompt_template)  # type: ignore[arg-type,index]

        assert len(text) == len(images)
        inputs = []
        for i, (ctx, image_list) in enumerate(zip(text, images)):
            text_parts = ctx.split("<image>")  # type: ignore[union-attr]
            image_list = cast(list[Image], image_list)

            if len(text_parts) - 1 != len(image_list):
                raise ValueError(
                    f"In the {i}-th input, the number of images {len(image_list)} does not match the number of image tokens {len(text_parts) - 1} in the text."
                )
            result = []
            for seg, image in zip(text_parts, image_list):
                if seg != "":
                    result.append(seg)
                result.append(image)
            if text_parts[-1] != "":
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
