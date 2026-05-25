from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

from .model_base import ModelBase

HF_IDEFICS2 = ["idefics2-8b", "idefics2-8b-base", "idefics2-8b-chatty"]


class Idefics2(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoProcessor,
        model_class=AutoModelForVision2Seq,
        processor_args=None,
        model_args=None,
        **common_args,
    ):

        processor_args = (
            processor_args if processor_args else dict(do_image_splitting=False)
        )

        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            support_models=HF_IDEFICS2,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

    @property
    def default_prompt_template(self):
        template = (
            "{% if messages[0]['role'].lower() in ['instruction', 'system'] %}"
                "{{ messages[0]['role'].capitalize() + ': ' + messages[0]['content'] + '<end_of_utterance>\n'}}"
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
                    "{% endfor %}"
                    "{{ '<end_of_utterance>\n' }}"
                "{% endif %}" 
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ ns.generation_role.capitalize() + ':' }}"
            "{% endif %}"
        )

        if self.model_name == "idefics2-8b-base":
            return template.replace("<end_of_utterance>", "")

        return template
