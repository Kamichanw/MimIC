import os
from typing import Any, cast

from datasets import DatasetDict, load_dataset

from src.utils.common import ROOT_DIR, resolve_path
from src.utils.dataset_utils.interface import DatasetBase
from src.utils.icl import register_dataset_retriever

HF_DATASETS_DIR = str(ROOT_DIR / "src" / "hf_datasets")
@register_dataset_retriever("vqav2")
@register_dataset_retriever("ocr_vqa")
def generic_vqa_retriever(item, is_last: bool):
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


@register_dataset_retriever("ok_vqa")
def ok_vqa_retriever(item, is_last: bool):
    return generic_vqa_retriever(item, is_last)


class Dataset(DatasetBase):
    support_datasets = ["vqav2", "ok_vqa", "ocr_vqa"]

    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        if self.name == "vqav2":
            ds_init_args: dict[str, Any] = dict(
                path=os.path.join(HF_DATASETS_DIR, "vqav2"),
                data_dir=resolve_path(os.environ["VQAV2_DIR"]),
                images_dir=resolve_path(os.environ["COCO_DIR"]),
            )
        elif self.name == "ok_vqa":
            ds_init_args = dict(
                path=os.path.join(HF_DATASETS_DIR, "ok_vqa"),
                data_dir=resolve_path(os.environ["OK_VQA_DIR"]),
                images_dir=resolve_path(os.environ["COCO_DIR"]),
            )
        else:
            ds_init_args = dict(
                path=os.path.join(HF_DATASETS_DIR, "ocr_vqa"),
                data_dir=resolve_path(os.environ["OCR_VQA_DIR"]),
                images_dir=resolve_path(os.environ["OCR_VQA_IMAGES_DIR"]),
            )

        dataset = cast(
            DatasetDict,
            load_dataset(**ds_init_args, trust_remote_code=True),  # type: ignore[arg-type]
        )
        self._support_set = dataset["train"]

    @property
    def num_role_in_round(self):
        return 3

    def extract_answer(self, item):
        if self.name == "ocr_vqa":
            return item["answer"]
        return item["answers"][0]["answer"]

    @property
    def instruction(self):
        return "Provide an answer to the question. Use the image to answer."
