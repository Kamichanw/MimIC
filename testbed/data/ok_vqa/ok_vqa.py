import torch
import torch_npu
import json
from pathlib import Path
from typing import Dict, List
import datasets

from testbed.data.common import split_generators, most_common_from_dict

_CITATION = """\
@article{DBLP:journals/corr/abs-1906-00067,
  author    = {Kenneth Marino and
               Mohammad Rastegari and
               Ali Farhadi and
               Roozbeh Mottaghi},
  title     = {{OK-VQA:} {A} Visual Question Answering Benchmark Requiring External
               Knowledge},
  journal   = {CoRR},
  volume    = {abs/1906.00067},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.00067},
  eprinttype = {arXiv},
  eprint    = {1906.00067},
  timestamp = {Thu, 13 Jun 2019 13:36:00 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1906-00067.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


_DESCRIPTION = """\
OK-VQA is a new dataset for visual question answering that requires methods which can draw upon outside knowledge to answer questions.
- 14,055 open-ended questions
- 5 ground truth answers per question
- Manually filtered to ensure all questions require outside knowledge (e.g. from Wikipeida)
- Reduced questions with most common answers to reduce dataset bias
"""


_HOMEPAGE = "https://okvqa.allenai.org/"


_LICENSE = "CC BY 4.0"


_URLS = {
    "annotations": {
        "train": "https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip",
        "val": "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip",
    },
    "questions": {
        "train": "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip",
        "val": "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip",
    },
    "images": {
        "train": "http://images.cocodataset.org/zips/train2014.zip",
        "val": "http://images.cocodataset.org/zips/val2014.zip",
    },
}


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
        self.answer_selector = (
            answer_selector if answer_selector is not None else most_common_from_dict
        )

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
            )
            / _SUB_FOLDER_OR_FILE_NAME[file_type][split_name],
            _SUB_FOLDER_OR_FILE_NAME,
            self.config.verbose,
        )

    def _generate_examples(self, split, questions_path, annotations_path, images_path):
        dataset = json.load(open(annotations_path, "r"))
        questions = json.load(open(questions_path, "r"))

        qa: Dict[str, List] = {ann["question_id"]: [] for ann in dataset["annotations"]}
        for ann in dataset["annotations"]:
            qa[ann["question_id"]] = ann

        for question in questions["questions"]:
            annotation = qa[question["question_id"]]
            # build record
            record = question
            record.update(annotation)
            record["image"] = str(
                images_path.resolve()
                / f"COCO_{images_path.name}_{record['image_id']:0>12}.jpg"
            )
            record["answer"] = self.config.answer_selector(question["answers"])
            yield question["question_id"], record
