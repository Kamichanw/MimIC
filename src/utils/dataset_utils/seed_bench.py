import os
from typing import Any, cast

from datasets import load_dataset

from src.utils.common import ROOT_DIR, resolve_path
from src.utils.dataset_utils.interface import DatasetBase
from src.utils.icl import register_dataset_retriever

HF_DATASETS_DIR = str(ROOT_DIR / "src" / "hf_datasets")


@register_dataset_retriever("seed_bench")
def retriever(item, is_last: bool):
    num_images = len(item["image"]) if isinstance(item["image"], list) else 1
    return (
        [
            {"role": "image", "content": [{"type": "image"}] * num_images},
            {
                "role": "question",
                "content": [{"type": "text", "text": item["question"]}],
            },
            {
                "role": "choices",
                "content": [
                    {
                        "type": "text",
                        "text": f"A. {item['choice_a']} B. {item['choice_b']} C. {item['choice_c']} D. {item['choice_d']}",
                    },
                    {"type": "text", "text": "Answer with the letter."},
                ],
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
    support_datasets = ["seed_bench"]

    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        ds_init_args: dict[str, Any] = dict(
            path=os.path.join(HF_DATASETS_DIR, "seed_bench"),
            data_dir=resolve_path(os.environ["SEED_DIR"]),
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
        return 4

    def extract_answer(self, item):
        return item["answer"]

    @property
    def instruction(self):
        return None
