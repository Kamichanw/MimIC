import json
from pathlib import Path
from typing import Dict, List
import datasets
import torch
import torch_npu

from testbed.data.common import split_generators, most_common_from_dict


_CITATION = """\
@InProceedings{VQA,
  author      = {Stanislaw Antol and Aishwarya Agrawal and Jiasen Lu and Margaret Mitchell and Dhruv Batra and C. Lawrence Zitnick and Devi Parikh},
  title       = {VQA: Visual Question Answering},
  booktitle   = {International Conference on Computer Vision (ICCV)},
  year        = {2015},
} 
"""

_DESCRIPTION = """\
VQA is a new dataset containing open-ended questions about images.
These questions require an understanding of vision, language and commonsense knowledge to answer.
"""

_HOMEPAGE = "https://visualqa.org"

_LICENSE = "CC BY 4.0"

_URLS = {
    "questions": {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "test-dev": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
        "test": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
    },
    "annotations": {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    },
    "images": {
        "train": "http://images.cocodataset.org/zips/train2014.zip",
        "val": "http://images.cocodataset.org/zips/val2014.zip",
        "test-dev": "http://images.cocodataset.org/zips/test2015.zip",
        "test": "http://images.cocodataset.org/zips/test2015.zip",
    },
}

_SUB_FOLDER_OR_FILE_NAME = {
    "questions": {
        "train": "v2_OpenEnded_mscoco_train2014_questions.json",
        "val": "v2_OpenEnded_mscoco_val2014_questions.json",
        "test-dev": "v2_OpenEnded_mscoco_test-dev2015_questions.json",
        "test": "v2_OpenEnded_mscoco_test2015_questions.json",
    },
    "annotations": {
        "train": "v2_mscoco_train2014_annotations.json",
        "val": "v2_mscoco_val2014_annotations.json",
    },
    "images": {
        "train": "train2014",
        "val": "val2014",
        "test-dev": "test2015",
        "test": "test2015",
    },
}


class VQAv2Config(datasets.BuilderConfig):

    def __init__(self, images_dir=None, verbose=True, answer_selector=None, **kwargs):
        self.images_dir = images_dir if images_dir is not None else self.data_dir
        self.verbose = verbose
        self.answer_selector = (
            answer_selector if answer_selector is not None else most_common_from_dict
        )
        super().__init__(**kwargs)


class VQAv2(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = VQAv2Config

    def _info(self):
        features = datasets.Features(
            {
                "question_type": datasets.Value("string"),
                "multiple_choice_answer": datasets.Value("string"),
                "answers": [
                    {
                        "answer": datasets.Value("string"),
                        "answer_confidence": datasets.Value("string"),
                        "answer_id": datasets.Value("int64"),
                    }
                ],
                "answer": datasets.Value("string"),
                "image_id": datasets.Value("int64"),
                "answer_type": datasets.Value("string"),
                "question_id": datasets.Value("int64"),
                "question": datasets.Value("string"),
                "image": datasets.Image(),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None or self.config.images_dir is None:
            raise ValueError("Missing arguments for data_dir and images_dir.")

        return split_generators(
            lambda file_type, split_name: Path(
                self.config.data_dir
                if file_type != "images"
                else self.config.images_dir
            ).resolve()
            / _SUB_FOLDER_OR_FILE_NAME[file_type][split_name],
            _SUB_FOLDER_OR_FILE_NAME,
            self.config.verbose,
        )

    def _generate_examples(self, split, questions_path, annotations_path, images_path):
        questions = json.load(open(questions_path, "r"))

        if annotations_path is not None:
            dataset = json.load(open(annotations_path, "r"))

            qa: Dict[str, List] = {ann["question_id"]: [] for ann in dataset["annotations"]}
            for ann in dataset["annotations"]:
                qa[ann["question_id"]] = ann

            for question in questions["questions"]:
                annotation = qa[question["question_id"]]
                record = question
                record.update(annotation)
                record["image"] = str(
                    images_path.resolve()
                    / f"COCO_{images_path.name}_{record['image_id']:0>12}.jpg"
                )
                record["answer"] = self.config.answer_selector(question["answers"])
                yield question["question_id"], record
        else:
            # No annotations for the test split
            for question in questions["questions"]:
                question.update(
                    {
                        "question_type": None,
                        "multiple_choice_answer": None,
                        "answers": None,
                        "answer": None,
                        "answer_type": None,
                    }
                )
                question["image"] = str(
                    images_path.resolve()
                    / f"COCO_{images_path.name}_{question['image_id']:0>12}.jpg"
                )
                yield question["question_id"], question
