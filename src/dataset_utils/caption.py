import torch
import torch_npu
import evaluate

import os
from datasets import load_dataset
from tqdm import tqdm

import src.paths as paths
from dataset_utils.interface import DatasetBase
from testbed.data import postprocess_generation
from utils import get_expand_runname


class Dataset(DatasetBase):
    support_datasets = ["coco", "flickr"]

    def __init__(self, data_cfg):
        super().__init__(data_cfg)
        if self.name == "coco":
            ds_init_args = dict(
                path=os.path.join(paths.testbed_dir, "data", "coco"),
                data_dir=paths.karpathy_coco_caption_dir,
                images_dir=paths.coco_dir,
            )
        elif self.name == "flickr":
            ds_init_args = dict(
                path=os.path.join(paths.testbed_dir, "data", "flickr"),
                data_dir=paths.flickr30k_dir,
                images_dir=paths.flickr30k_images_dir,
                name="flickr30k",
            )

        dataset = load_dataset(**ds_init_args, trust_remote_code=True)
        self._support_set = dataset["train"]
        self._query_set = dataset["validation"]

    @property
    def num_role_in_round(self):
        # [
        #   { "role" : "image",
        #     "content" :  ... },
        #   { "role" : "caption",
        #     "content" : ... },
        # ]
        return 2
    
    @staticmethod
    def metric_key():
        return "CIDEr"

    def extract_answer(self, item):
        # we use the first answer as grounding truth
        return item["sentences_raw"][0]

    @property
    def instruction(self):
        if self.cfg.is_icl:
            return "provide a short caption of the input image."
        return None

    def eval(
        self,
        eval_cfg,
        model,
    ):
        result = []
        metric = evaluate.load(
            os.path.join(paths.testbed_dir, "evaluate", "metrics", "CIDEr")
        )
        eval_dl = self.validation_dataloader(eval_cfg.batch_size)
        iterations = eval_cfg.iterations or len(eval_dl)
        generation_args = eval_cfg.generation_args
        generation_args["max_new_tokens"] = 20
        for _, batch in zip(
            range(iterations),
            tqdm(
                eval_dl,
                total=iterations,
                desc=f"Evaluating {model.model_name} with {get_expand_runname(eval_cfg)} ...",
            ),
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
                answer = last_item["sentences_raw"]
                prediction = postprocess_generation(
                    self.name,
                    pred,
                    ["\n", "Caption", "Image", "<", "Short"],
                )
                metric.add(prediction=prediction, reference=answer)
                record = {
                    "raw_output": pred,
                    "filename": last_item["filename"],
                    "sentences": last_item["sentences_raw"],
                    "prediction": prediction,
                }
                if self.name == "coco":
                    record.update(cocoid=last_item["cocoid"])
                result.append(record)

        return result, metric.compute()
