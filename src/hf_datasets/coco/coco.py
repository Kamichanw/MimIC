import json
from pathlib import Path
from typing import cast

import datasets
from src.hf_datasets.common import split_generators

_CITATION = """
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{\'{a}}r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
}
"""

_DESCRIPTION = """
MS COCO is a large-scale object detection, segmentation, and captioning dataset.
"""

_HOMEPAGE = "https://cocodataset.org/#home"
_LICENSE = "CC BY 4.0"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "filepath": datasets.Value("string"),
        "sentids": [datasets.Value("int32")],
        "filename": datasets.Value("string"),
        "imgid": datasets.Value("int32"),
        "split": datasets.Value("string"),
        "caption": datasets.Value("string"),
        "sentences_tokens": [[datasets.Value("string")]],
        "sentences_raw": [datasets.Value("string")],
        "sentences_sentid": [datasets.Value("int32")],
        "cocoid": datasets.Value("int32"),
    }
)

_SUB_FOLDER_OR_FILE_NAME = {
    "annotations": {
        "train": "dataset_coco.json",
        "val": "dataset_coco.json",
        "test": "dataset_coco.json",
        "restval": "dataset_coco.json",
    },
    "images": {
        "train": "train2014",
        "validation": "val2014",
        "val": "val2014",
        "test": "val2014",
        "restval": "val2014",
    },
}


class COCOConfig(datasets.BuilderConfig):
    def __init__(self, images_dir=None, caption_selector=None, verbose=True, **kwargs):
        self.images_dir = images_dir if images_dir is not None else self.data_dir
        self.verbose = verbose
        self.caption_selector = (
            caption_selector
            if caption_selector is not None
            else lambda dct: dct[0]["raw"]
        )
        super().__init__(**kwargs)


class COCO(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIG_CLASS = COCOConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        config = cast(COCOConfig, self.config)
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

    def _generate_examples(self, split, annotations_path, images_path):
        config = cast(COCOConfig, self.config)
        with open(annotations_path, "r", encoding="utf-8") as fi:
            annotations = json.load(fi)
            for image_metadata in annotations["images"]:
                if image_metadata["split"] == split:
                    record = {
                        "image": str(images_path.resolve() / image_metadata["filename"]),
                        "filepath": str(images_path.resolve() / image_metadata["filename"]),
                        "sentids": image_metadata["sentids"],
                        "filename": image_metadata["filename"],
                        "imgid": image_metadata["imgid"],
                        "split": image_metadata["split"],
                        "cocoid": image_metadata["cocoid"],
                        "caption": config.caption_selector(image_metadata["sentences"]),
                        "sentences_tokens": [caption["tokens"] for caption in image_metadata["sentences"]],
                        "sentences_raw": [caption["raw"] for caption in image_metadata["sentences"]],
                        "sentences_sentid": [caption["sentid"] for caption in image_metadata["sentences"]],
                    }
                    yield record["imgid"], record
