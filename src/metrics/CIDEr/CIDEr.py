from typing import List
import datasets
import evaluate
import os
import tempfile
import subprocess

from pycocoevalcap.cider.cider import CiderScorer

_DESCRIPTION = """
The CIDEr metric is used to evaluate generated image captions.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (list of str): Predicted captions.
    references (list of str lists): Ground truth captions.
    n (int, defaults to 4): Number of ngrams.
    sigma (float, defaults to 6.0): Gaussian penalty std.

Returns:
    CIDEr (`float`): CIDEr value.
"""

_CITATION = """
@inproceedings{vedantam2015cider,
  title={Cider: Consensus-based image description evaluation},
  author={Vedantam, Ramakrishna and Lawrence Zitnick, C and Parikh, Devi},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4566--4575},
  year={2015}
}
"""

_URLS = {
    "stanford-corenlp": "https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.4.1/stanford-corenlp-3.4.1.jar"
}


def tokenize(tokenizer_path: str, predictions: List[str], references: List[List[str]]):
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    cmd = [
        "java",
        "-cp",
        tokenizer_path,
        "edu.stanford.nlp.process.PTBTokenizer",
        "-preserveLines",
        "-lowerCase",
    ]

    sentences = "\n".join(
        [s.replace("\n", " ") for s in predictions + [ref for refs in references for ref in refs]]
    )

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(sentences.encode())

    cmd.append(f.name)
    p_tokenizer = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    token_lines = p_tokenizer.communicate(input=sentences.rstrip().encode())[0]
    token_lines_str = token_lines.decode()
    lines = [
        " ".join([w for w in line.rstrip().split(" ") if w not in punctuations])
        for line in token_lines_str.split("\n")
    ]

    os.remove(f.name)

    pred_size = len(predictions)
    ref_sizes = [len(ref) for ref in references]

    predictions = lines[:pred_size]
    start = pred_size
    references = []
    for size in ref_sizes:
        references.append(lines[start : start + size])
        start += size

    return predictions, references


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CIDEr(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Sequence(
                        datasets.Value("string", id="sequence"), id="references"
                    ),
                }
            ),
            reference_urls=[
                "https://github.com/salaniz/pycocoevalcap",
                "https://github.com/tylin/coco-caption",
            ],
        )

    def _download_and_prepare(self, dl_manager):
        self.tokenizer_path = dl_manager.download(_URLS["stanford-corenlp"])

    def _compute(self, predictions, references, n=4, sigma=6.0):
        predictions, references = tokenize(
            self.tokenizer_path, predictions, references
        )
        scorer = CiderScorer(n=n, sigma=sigma)
        for pred, refs in zip(predictions, references):
            scorer += (pred, refs)
        score, _ = scorer.compute_score()
        return {"CIDEr": score}
