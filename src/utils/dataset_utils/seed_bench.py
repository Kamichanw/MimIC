import os
import random
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


@register_postprocess("seed_bench")
def postprocess(text):
    return text


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
        dataset = cast(DatasetDict, dataset)
        self._support_set = dataset["train"]
        self._query_set = dataset["test"]

    @property
    def num_role_in_round(self):
        return 4

    @staticmethod
    def metric_key():
        return "exact_match"

    def extract_answer(self, item):
        return item["answer"]

    @property
    def instruction(self):
        return None

    def eval(self, eval_cfg, model, accelerator=None):
        result = []
        eval_dl = self.validation_dataloader(eval_cfg.batch_size)
        iterations = eval_cfg.iterations or len(eval_dl)
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
                **eval_cfg.generation_args,
            )
            if predictions is None:
                continue

            for pred, context in zip(predictions, batch):
                last_qa = context[-1]
                prediction = postprocess_generation(
                    self.name, pred, stop_words=["\n", "."]
                )
                prediction = cast(str, prediction)
                if prediction.upper() not in ["A", "B", "C", "D"]:
                    prediction = random.choice(["A", "B", "C", "D"])
                result.append(
                    {
                        "question": last_qa["question"],
                        "question_id": last_qa["question_id"],
                        "raw_output": pred,
                        "choice_a": last_qa["choice_a"],
                        "choice_b": last_qa["choice_b"],
                        "choice_c": last_qa["choice_c"],
                        "choice_d": last_qa["choice_d"],
                        "prediction": prediction,
                        "answer": last_qa["answer"],
                    }
                )

        return result

    def compute_metrics(self, records):
        metric = evaluate.load("exact_match")
        for record in records:
            metric.add(prediction=record["prediction"], reference=record["answer"])
        return metric.compute()
