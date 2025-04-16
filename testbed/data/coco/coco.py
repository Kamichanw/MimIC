import json
import torch
import torch_npu

from pathlib import Path

import datasets
from testbed.data.common import split_generators


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
  url       = {http://arxiv.org/abs/1405.0312},
  eprinttype = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/LinMBHPRDZ14.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """
MS COCO is a large-scale object detection, segmentation, and captioning dataset.
COCO has several features: Object segmentation, Recognition in context, Superpixel stuff segmentation, 330K images (>200K labeled), 1.5 million object instances, 80 object categories, 91 stuff categories, 5 captions per image, 250,000 people with keypoints.
"""

_HOMEPAGE = "https://cocodataset.org/#home"

_LICENSE = "CC BY 4.0"


_IMAGES_URLS = {
    "train": "http://images.cocodataset.org/zips/train2014.zip",
    "validation": "http://images.cocodataset.org/zips/val2014.zip",
}

_KARPATHY_FILES_URL = (
    "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
)


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
            else lambda dct: dct[0]["raw"]  # select the 1st sentence as ground truth
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

    def _generate_examples(self, split, annotations_path, images_path):
        with open(annotations_path, "r", encoding="utf-8") as fi:
            annotations = json.load(fi)

            for image_metadata in annotations["images"]:
                if image_metadata["split"] == split:
                    record = {
                        "image": str(
                            images_path.resolve() / image_metadata["filename"]
                        ),
                        "filepath": str(
                            images_path.resolve() / image_metadata["filename"]
                        ),
                        "sentids": image_metadata["sentids"],
                        "filename": image_metadata["filename"],
                        "imgid": image_metadata["imgid"],
                        "split": image_metadata["split"],
                        "cocoid": image_metadata["cocoid"],
                        "caption": self.config.caption_selector(
                            image_metadata["sentences"]
                        ),
                        "sentences_tokens": [
                            caption["tokens"] for caption in image_metadata["sentences"]
                        ],
                        "sentences_raw": [
                            caption["raw"] for caption in image_metadata["sentences"]
                        ],
                        "sentences_sentid": [
                            caption["sentid"] for caption in image_metadata["sentences"]
                        ],
                    }
                    yield record["imgid"], record
