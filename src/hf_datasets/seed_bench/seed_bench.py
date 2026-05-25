import json
import os
from pathlib import Path
from typing import cast

import datasets

from src.hf_datasets.common import split_generators

_SUB_FOLDER_OR_FILE_NAME = {
    "annotations": {
        "test": "SEED-Bench.json",
    },
    "images": {
        "test": "SEED-Bench-image",
    },
}


class SEEDBenchConfig(datasets.BuilderConfig):
    def __init__(self, images_dir=None, verbose=True, **kwargs):
        self.verbose = verbose
        self.images_dir = images_dir
        super().__init__(**kwargs)


class SEEDBenchDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = SEEDBenchConfig

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "answer": datasets.Value("string"),
                    "choice_a": datasets.Value("string"),
                    "choice_b": datasets.Value("string"),
                    "choice_c": datasets.Value("string"),
                    "choice_d": datasets.Value("string"),
                    "data_id": datasets.Value("string"),
                    "data_type": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "question_type_id": datasets.Value("string"),
                    "image": datasets.Image(),
                    "segment": datasets.Value("string", id=None),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        config = cast(SEEDBenchConfig, self.config)
        if config.data_dir is None:
            raise ValueError("Missing arguments for data_dir")

        if config.images_dir is None:
            config.images_dir = os.path.join(
                config.data_dir, _SUB_FOLDER_OR_FILE_NAME["images"]["test"]
            )
        data_dir = config.data_dir
        images_dir = cast(str, config.images_dir)

        def expand_path_fn(file_type, split_name):
            if file_type == "images":
                return Path(images_dir).resolve()
            return (
                Path(data_dir).resolve()
                / _SUB_FOLDER_OR_FILE_NAME[file_type][split_name]
            )

        return split_generators(
            expand_path_fn,
            _SUB_FOLDER_OR_FILE_NAME,
            config.verbose,
        )

    def _generate_examples(self, split, annotations_path, images_path):
        with open(annotations_path, "r") as f:
            data = json.load(f)

        for item in data["questions"]:
            image = str(images_path.resolve() / item["data_id"])
            if os.path.exists(image):
                yield item["question_id"], {
                    "answer": item["answer"],
                    "choice_a": item["choice_a"],
                    "choice_b": item["choice_b"],
                    "choice_c": item["choice_c"],
                    "choice_d": item["choice_d"],
                    "data_id": item["data_id"],
                    "data_type": item["data_type"],
                    "question": item["question"],
                    "question_id": item["question_id"],
                    "question_type_id": item["question_type_id"],
                    "image": image,
                    "segment": item.get("segment", None),
                }
