from typing import List, Union
import torch
import torch_npu

from testbed.data.common import register_dataset_retriever, register_postprocess


@register_dataset_retriever(__name__.split(".")[-1])
def retriever(item, is_last: bool):
    return (
        [
            {"role": "image", "content": [{"type": "image"}]},
            (
                {"role": "caption"}
                if is_last
                else {
                    "role": "caption",
                    "content": [{"type": "text", "text": item["caption"]}],
                }
            ),
        ],
        item["image"],
    )


@register_postprocess(__name__.split(".")[-1])
def postprocess(text: Union[str, List[str]]) -> Union[str, List[str]]:
    return text
