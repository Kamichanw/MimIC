import os
from typing import Any, cast

from datasets import load_dataset

from src.utils.common import resolve_path
from src.utils.dataset_utils.interface import DatasetBase
from src.utils.icl import register_dataset_retriever


@register_dataset_retriever("mme")
def retriever(item, is_last: bool):
    return (
        [
            {"role": "image", "content": [{"type": "image"}]},
            {
                "role": "question",
                "content": [{"type": "text", "text": item["question"]}],
            },
            (
                {"role": "answer"}
                if is_last
                else {
                    "role": "answer",
                    "content": [{"type": "text", "text": item["answer"]}],
                }
            ),
        ],
        item["image"],
    )


class Dataset(DatasetBase):
    support_datasets = ["mme"]

    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        ds_init_args: dict[str, Any] = dict(
            path="parquet", data_dir=resolve_path(os.environ["MME_DIR"])
        )

        assert (
            data_cfg.num_query_samples
        ), f"num_query_samples must be specified and greater than 0, but got {data_cfg.num_query_samples}"

        dataset = cast(
            Any,
            load_dataset(
                **ds_init_args, trust_remote_code=True, split="test"
            ),
        ).train_test_split(  # type: ignore[arg-type]
            train_size=data_cfg.num_query_samples, seed=data_cfg.seed, shuffle=False
        )
        self._support_set = dataset["train"]

    @property
    def num_role_in_round(self):
        return 3

    def extract_answer(self, item):
        return item["answer"]

    @property
    def instruction(self):
        return 'Provide an answer in "Yes" or "No" to the question. Use the image to answer.'
