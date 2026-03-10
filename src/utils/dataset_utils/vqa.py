import os
from typing import Any, cast

import evaluate
from datasets import Dataset as HFDataset, DatasetDict, load_dataset

from src.utils.common import ROOT_DIR, get_expand_runname, resolve_path
from src.utils.dataset_utils.interface import DatasetBase
from src.utils.dataset_utils.ok_vqa_stemmer import stemmer
from src.utils.icl import (
    postprocess_generation,
    register_dataset_retriever,
    register_postprocess,
)

HF_DATASETS_DIR = str(ROOT_DIR / "src" / "hf_datasets")
METRICS_DIR = str(ROOT_DIR / "src" / "metrics")


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


@register_postprocess("vqav2")
@register_postprocess("ocr_vqa")
def passthrough_postprocess(text):
    return text


@register_postprocess("ok_vqa")
def ok_vqa_postprocess(pred):
    return stemmer().stem(pred)


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
        self._query_set = dataset["validation"]

        if self.name == "vqav2":
            internal_eval_set_path = os.path.join(ROOT_DIR, "dataset", "vqav2")
            if os.path.exists(internal_eval_set_path):
                self._query_set = load_dataset(
                    path=os.path.join(HF_DATASETS_DIR, "vqav2"),
                    data_dir=internal_eval_set_path,
                    images_dir=resolve_path(os.environ["COCO_DIR"]),
                    split="validation",
                )  # type: ignore[arg-type]
        elif self.name == "ocr_vqa":
            internal_eval_set_path = os.path.join(ROOT_DIR, "dataset", "ocr_vqa")
            if os.path.exists(internal_eval_set_path):
                self._query_set = cast(HFDataset, load_dataset(
                    path=os.path.join(HF_DATASETS_DIR, "ocr_vqa"),
                    data_dir=internal_eval_set_path,
                    images_dir=resolve_path(os.environ["OCR_VQA_IMAGES_DIR"]),
                    split="validation",
                ))  # type: ignore[arg-type]

    @property
    def num_role_in_round(self):
        return 3

    @staticmethod
    def metric_key():
        return "overall"

    def extract_answer(self, item):
        if self.name == "ocr_vqa":
            return item["answer"]
        return item["answers"][0]["answer"]

    @property
    def instruction(self):
        return "Provide an answer to the question. Use the image to answer."

    @property
    def support_set(self):
        return self._support_set

    @property
    def query_set(self):
        return self._query_set

    def eval(self, eval_cfg, model, accelerator=None):
        if self.name == "ocr_vqa":
            return self.eval_ocr_vqa(eval_cfg, model, accelerator)
        return self.eval_vqa(eval_cfg, model, accelerator)

    def eval_ocr_vqa(self, eval_cfg, lmm, accelerator=None):
        result = []
        eval_dl = self.validation_dataloader(eval_cfg.batch_size)
        iterations = eval_cfg.iterations or len(eval_dl)
        for batch in self.iter_eval_batches(
            eval_dl,
            iterations,
            lmm.model_name,
            get_expand_runname(eval_cfg),
            accelerator,
        ):
            predictions = self.get_prediction(
                lmm,
                batch,
                max_skip_oom=eval_cfg.max_skip_oom,
                **eval_cfg.generation_args,
            )
            if predictions is None:
                continue
            for pred, context in zip(predictions, batch):
                last_qa = context[-1]
                prediction = postprocess_generation(
                    self.name,
                    pred,
                    ["\n", "Question", "Answer", "Image", "Short"],
                )
                result.append(
                    {
                        "question_id": last_qa["question_id"],
                        "raw_output": pred,
                        "question": last_qa["question"],
                        "prediction": prediction,
                        "answer": last_qa["answer"],
                    }
                )

        return result

    def eval_vqa(self, eval_cfg, lmm, accelerator=None):
        result = []
        eval_dl = self.validation_dataloader(eval_cfg.batch_size)
        iterations = eval_cfg.iterations or len(eval_dl)
        for batch in self.iter_eval_batches(
            eval_dl,
            iterations,
            lmm.model_name,
            get_expand_runname(eval_cfg),
            accelerator,
        ):
            predictions = self.get_prediction(
                lmm,
                batch,
                max_skip_oom=eval_cfg.max_skip_oom,
                **eval_cfg.generation_args,
            )
            if predictions is None:
                continue
            for pred, context in zip(predictions, batch):
                last_qa = context[-1]
                prediction = postprocess_generation(
                    self.name,
                    pred,
                    ["\n", "Question", "Answer", "Image", "Short"],
                )
                result.append(
                    {
                        "question_id": last_qa["question_id"],
                        "raw_output": pred,
                        "question": last_qa["question"],
                        "question_type": last_qa["question_type"],
                        "answer_type": last_qa["answer_type"],
                        "prediction": prediction,
                        "answers": last_qa["answers"],
                    }
                )

        return result

    def compute_metrics(self, records):
        if self.name == "ocr_vqa":
            metric = evaluate.load("exact_match")
            for record in records:
                metric.add(
                    prediction=record["prediction"].lower(),
                    reference=record["answer"].lower(),
                )
            return metric.compute()

        metric = evaluate.load(os.path.join(METRICS_DIR, "vqa_accuracy"))
        for record in records:
            metric.add(
                prediction=record["prediction"],
                reference=[item["answer"] for item in record["answers"]],
                question_types=record["question_type"],
                answer_types=record["answer_type"],
            )
        return metric.compute()
