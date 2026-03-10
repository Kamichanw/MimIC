import os
from typing import Any, cast

import evaluate
from datasets import DatasetDict, load_dataset

from src.utils.common import get_expand_runname, resolve_path
from src.utils.dataset_utils.interface import DatasetBase
from src.utils.icl import (
    postprocess_generation,
    register_dataset_retriever,
    register_postprocess,
)


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


@register_postprocess("mme")
def postprocess(text):
    if text.lower() == "yes":
        return 1
    if text.lower() == "no":
        return 0
    return -1


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
        dataset = cast(DatasetDict, dataset)
        self._support_set = dataset["train"]
        self._query_set = dataset["test"]

    @property
    def num_role_in_round(self):
        return 3

    @staticmethod
    def metric_key():
        return "accuracy"

    def extract_answer(self, item):
        return item["answer"]

    @property
    def instruction(self):
        return 'Provide an answer in "Yes" or "No" to the question. Use the image to answer.'

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
                prediction = postprocess_generation(self.name, pred, stop_words=["\n"])
                result.append(
                    {
                        "prediction": prediction,
                        "answer": last_qa["answer"],
                        "question": last_qa["question"],
                        "raw_output": pred,
                        "question_id": last_qa["question_id"],
                    }
                )

        return result

    def compute_metrics(self, records):
        metric = evaluate.load("accuracy")
        for record in records:
            metric.add(
                prediction=record["prediction"],
                reference=record["answer"].lower() == "yes",
            )
        return metric.compute()
