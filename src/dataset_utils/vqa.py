import torch
import torch_npu
from pathlib import Path
import evaluate

import os
from datasets import load_dataset
from tqdm import tqdm

import src.paths as paths
from dataset_utils.interface import DatasetBase
from testbed.data import postprocess_generation
from utils import get_expand_runname


class Dataset(DatasetBase):
    support_datasets = ["vqav2", "ok_vqa", "ocr_vqa"]

    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        if self.name == "vqav2":
            ds_init_args = dict(
                path=os.path.join(paths.testbed_dir, "data", "vqav2"),
                data_dir=paths.vqav2_dir,
                images_dir=paths.coco_dir,
            )
        elif self.name == "ok_vqa":
            ds_init_args = dict(
                path=os.path.join(paths.testbed_dir, "data", "okvqa"),
                data_dir=paths.ok_vqa_dir,
                images_dir=paths.coco_dir,
            )
        elif self.name == "ocr_vqa":
            ds_init_args = dict(
                path=os.path.join(paths.testbed_dir, "data", "ocr_vqa"),
                data_dir=paths.ocr_vqa_dir,
                images_dir=paths.ocr_vqa_images_dir,
            )

        dataset = load_dataset(**ds_init_args, trust_remote_code=True)
        self._support_set = dataset["train"]
        self._query_set = dataset["validation"]

        # special cases for internal evaluation
        if self.name == "vqav2":
            # for vqav2, we only use a fixed set (10,000 samples) from evaluation split
            internal_eval_set_path = os.path.join(
                Path(__file__).parent.parent.parent, "dataset", "vqav2"
            )  # project_path/dataset/vqav2
            if os.path.exists(internal_eval_set_path):
                self._query_set = load_dataset(
                    path=os.path.join(paths.testbed_dir, "data", "vqav2"),
                    data_dir=internal_eval_set_path,
                    images_dir=paths.coco_dir,
                    split="validation",
                )
        elif self.name == "ocr_vqa":
            internal_eval_set_path = os.path.join(
                Path(__file__).parent.parent.parent, "dataset", "ocr_vqa"
            )  # project_path/dataset/ocr_vqa
            if os.path.exists(internal_eval_set_path):
                self._query_set = load_dataset(
                    path=os.path.join(paths.testbed_dir, "data", "ocr_vqa"),
                    data_dir=internal_eval_set_path,
                    images_dir=paths.ocr_vqa_images_dir,
                    split="validation",
                )

    @property
    def num_role_in_round(self):
        # [
        #   { "role" : "image",
        #     "content" :  ... },
        #   { "role" : "question",
        #     "content" : ... },
        #   { "role" : "answer",
        #    "content" : ... }
        # ]
        return 3

    @staticmethod
    def metric_key():
        return "overall"

    def extract_answer(self, item):
        if self.name == "ocr_vqa":
            return item["answer"]
        # we use the first answer as grounding truth
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

    def eval(
        self,
        eval_cfg,
        model,
    ):
        if self.name == "ocr_vqa":
            return self.eval_ocr_vqa(eval_cfg, model)
        return self.eval_vqa(eval_cfg, model)

    def eval_ocr_vqa(self, eval_cfg, lmm):
        result = []
        metric = evaluate.load("exact_match")
        eval_dl = self.validation_dataloader(eval_cfg.batch_size)
        iterations = eval_cfg.iterations or len(eval_dl)
        for _, batch in zip(
            range(iterations),
            tqdm(
                eval_dl,
                total=iterations,
                desc=f"Evaluating {lmm.model_name} with {get_expand_runname(eval_cfg)} ...",
            ),
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
                gt_answer = last_qa["answer"]
                metric.add(prediction=prediction.lower(), reference=gt_answer.lower())
                result.append(
                    {
                        "question_id": last_qa["question_id"],
                        "raw_output": pred,
                        "question": last_qa["question"],
                        "prediction": prediction,
                        "answer": last_qa["answer"],
                    }
                )

        return result, metric.compute()

    def eval_vqa(self, eval_cfg, lmm):
        result = []
        metric = evaluate.load(
            os.path.join(paths.testbed_dir, "evaluate", "metrics", "vqa_accuracy")
        )
        eval_dl = self.validation_dataloader(eval_cfg.batch_size)
        iterations = eval_cfg.iterations or len(eval_dl)
        for _, batch in zip(
            range(iterations),
            tqdm(
                eval_dl,
                total=iterations,
                desc=f"Evaluating {lmm.model_name} with {get_expand_runname(eval_cfg)} ...",
            ),
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
                gt_answer = [item["answer"] for item in last_qa["answers"]]
                metric.add(
                    prediction=prediction,
                    reference=gt_answer,
                    question_types=last_qa["question_type"],
                    answer_types=last_qa["answer_type"],
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

        return result, metric.compute()
