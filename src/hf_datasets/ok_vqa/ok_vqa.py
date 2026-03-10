import json
from pathlib import Path
from typing import Dict, List, cast
import datasets

from src.hf_datasets.common import split_generators, most_common_from_dict

_CITATION = """\
@article{DBLP:journals/corr/abs-1906-00067,
  author    = {Kenneth Marino and Mohammad Rastegari and Ali Farhadi and Roozbeh Mottaghi},
  title     = {{OK-VQA}},
  year      = {2019},
}
"""

_DESCRIPTION = """\
OK-VQA is a VQA dataset requiring external knowledge.
"""

_HOMEPAGE = "https://okvqa.allenai.org/"
_LICENSE = "CC BY 4.0"

_SUB_FOLDER_OR_FILE_NAME = {
    "questions": {
        "train": "OpenEnded_mscoco_train2014_questions.json",
        "val": "OpenEnded_mscoco_val2014_questions.json",
    },
    "annotations": {
        "train": "mscoco_train2014_annotations.json",
        "val": "mscoco_val2014_annotations.json",
    },
    "images": {
        "train": "train2014",
        "val": "val2014",
    },
}


class OKVQAConfig(datasets.BuilderConfig):
    def __init__(self, images_dir=None, verbose=True, answer_selector=None, **kwargs):
        self.images_dir = images_dir if images_dir is not None else self.data_dir
        self.verbose = verbose
        self.answer_selector = answer_selector if answer_selector is not None else most_common_from_dict
        super().__init__(**kwargs)


class OKVQA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = OKVQAConfig

    def _info(self):
        features = datasets.Features(
            {
                "image": datasets.Image(),
                "question_type": datasets.Value("string"),
                "confidence": datasets.Value("int32"),
                "answers": [
                    {
                        "answer": datasets.Value("string"),
                        "raw_answer": datasets.Value("string"),
                        "answer_confidence": datasets.Value("string"),
                        "answer_id": datasets.Value("int64"),
                    }
                ],
                "answer": datasets.Value("string"),
                "image_id": datasets.Value("int64"),
                "answer_type": datasets.Value("string"),
                "question_id": datasets.Value("int64"),
                "question": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION)

    def _split_generators(self, dl_manager):
        config = cast(OKVQAConfig, self.config)
        if config.data_dir is None or config.images_dir is None:
            raise ValueError("Missing arguments for data_dir and images_dir.")
        data_dir = config.data_dir
        images_dir = config.images_dir

        return split_generators(
            lambda file_type, split_name: Path(
                data_dir if file_type != "images" else images_dir
            )
            / _SUB_FOLDER_OR_FILE_NAME[file_type][split_name],
            _SUB_FOLDER_OR_FILE_NAME,
            config.verbose,
        )

    def _generate_examples(self, split, questions_path, annotations_path, images_path):
        config = cast(OKVQAConfig, self.config)
        dataset = json.load(open(annotations_path, "r"))
        questions = json.load(open(questions_path, "r"))

        qa: Dict[str, List] = {ann["question_id"]: [] for ann in dataset["annotations"]}
        for ann in dataset["annotations"]:
            qa[ann["question_id"]] = ann

        for question in questions["questions"]:
            annotation = qa[question["question_id"]]
            record = question
            record.update(annotation)
            record["image"] = str(
                images_path.resolve() / f"COCO_{images_path.name}_{record['image_id']:0>12}.jpg"
            )
            record["answer"] = config.answer_selector(question["answers"])
            yield question["question_id"], record
