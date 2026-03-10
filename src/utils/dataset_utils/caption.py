import os
from typing import Any, cast

import evaluate
from datasets import DatasetDict, load_dataset

from src.utils.common import ROOT_DIR, get_expand_runname, resolve_path
from src.utils.dataset_utils.interface import DatasetBase
from src.utils.icl import (
    postprocess_generation,
    register_dataset_retriever,
    register_postprocess,
)

HF_DATASETS_DIR = str(ROOT_DIR / "src" / "hf_datasets")
METRICS_DIR = str(ROOT_DIR / "src" / "metrics")


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


@register_postprocess("coco")
@register_postprocess("flickr")
def postprocess(text):
    return text


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
        self._query_set = dataset["validation"]

    @property
    def num_role_in_round(self):
        return 2

    @staticmethod
    def metric_key():
        return "CIDEr"

    def extract_answer(self, item):
        return item["sentences_raw"][0]

    @property
    def instruction(self):
        if self.cfg.is_icl:
            return "provide a short caption of the input image."
        return None

    def eval(self, eval_cfg, model, accelerator=None):
        result = []
        eval_dl = self.validation_dataloader(eval_cfg.batch_size)
        iterations = eval_cfg.iterations or len(eval_dl)
        generation_args = eval_cfg.generation_args
        generation_args["max_new_tokens"] = 20
        for batch in self.iter_eval_batches(
            eval_dl,
            iterations,
            model.model_name,
            get_expand_runname(eval_cfg),
            accelerator,
        ):
            predictions = self.get_prediction(
                model,
                batch,
                max_skip_oom=eval_cfg.max_skip_oom,
                **generation_args,
            )
            if predictions is None:
                continue

            for pred, context in zip(predictions, batch):
                last_item = context[-1]
                prediction = postprocess_generation(
                    self.name,
                    pred,
                    ["\n", "Caption", "Image", "<", "Short"],
                )
                record = {
                    "raw_output": pred,
                    "filename": last_item["filename"],
                    "sentences": last_item["sentences_raw"],
                    "prediction": prediction,
                }
                if self.name == "coco":
                    record.update(cocoid=last_item["cocoid"])
                result.append(record)

        return result

    def compute_metrics(self, records):
        metric = evaluate.load(os.path.join(METRICS_DIR, "CIDEr"))
        for record in records:
            metric.add(prediction=record["prediction"], reference=record["sentences"])
        return metric.compute()
