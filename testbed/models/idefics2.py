from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

from testbed.models.model_base import ModelBase

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
        # adopt idefics1 prompt template, see https://arxiv.org/pdf/2306.16527
        # fmt: off
        template = (
            "{% if messages and messages[0]['role'].lower() in ['instruction', 'system'] %}"
                "{{ messages[0]['role'].capitalize() + ':' + messages[0]['content'] + '\n' }}"
                "{% set messages = messages[1:] %}"
            "{% endif %}"
            "{% if messages %}"
                "{% set first_role = messages[0]['role'] %}"
                "{% for message in messages %}"
                    "{% if message['role'] != '' %}"
                        "{{ message['role'].capitalize() }}"
                        "{% if not 'content' in message or (message['content'] and message['content'][0]['type'] == 'image') %}"
                            "{{ ':' }}"
                        "{% else %}"
                            "{{ ': ' }}"
                        "{% endif %}"
                    "{% endif %}"

                    "{% if 'content' in message %}"
                        "{% if message['content'] is string %}"
                            "{{ message['content'] }}\n"
                        "{% else %}"
                            "{% for line in message['content'] %}"
                                "{% if line['type'] == 'text' %}"
                                    "{{ line['text'] }}"
                                "{% elif line['type'] == 'image' %}"
                                    "{{ '<image>' }}"
                                "{% endif %}"
                                "{% if not loop.last %}"
                                    " "
                                "{%+ endif %}"
                            "{% endfor %}"
                            "{% set is_end_of_round = loop.nextitem is not defined or loop.nextitem['role'] == first_role %}"
                            "{% if is_end_of_round %}"
                                "{{ '<end_of_utterance>\n' }}"
                            "{% else %}"
                                " "
                            "{%+ endif %}"
                        "{% endif %}"
                    "{% endif %}"
                "{% endfor %}"
            "{% endif %}"
        )
        # fmt: on

        if self.model_name == "idefics2-8b-base":
            # base model doesn't have <end_of_utterance> token
            return template.replace("<end_of_utterance>", "")

        return template
