import os
from typing import Any, cast

from datasets import DatasetDict, load_dataset

from src.utils.common import ROOT_DIR, resolve_path
from src.utils.dataset_utils.interface import DatasetBase
from src.utils.icl import register_dataset_retriever

HF_DATASETS_DIR = str(ROOT_DIR / "src" / "hf_datasets")


@register_dataset_retriever("coco")
@register_dataset_retriever("flickr")
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


class Dataset(DatasetBase):
    support_datasets = ["coco", "flickr"]

    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        if self.name == "coco":
            ds_init_args: dict[str, Any] = dict(
                path=os.path.join(HF_DATASETS_DIR, "coco"),
                data_dir=resolve_path(os.environ["KARPATHY_COCO_CAPTION_DIR"]),
                images_dir=resolve_path(os.environ["COCO_DIR"]),
            )
        else:
            ds_init_args = dict(
                path=os.path.join(HF_DATASETS_DIR, "flickr"),
                data_dir=resolve_path(os.environ["FLICKR30K_DIR"]),
                images_dir=resolve_path(os.environ["FLICKR30K_IMAGES_DIR"]),
                name="flickr30k",
            )

        dataset = cast(
            DatasetDict,
            load_dataset(**ds_init_args, trust_remote_code=True),  # type: ignore[arg-type]
        )
        self._support_set = dataset["train"]

    @property
    def num_role_in_round(self):
        return 2

    def extract_answer(self, item):
        return item["sentences_raw"][0]

    @property
    def instruction(self):
        if self.cfg.is_icl:
            return "provide a short caption of the input image."
        return None
