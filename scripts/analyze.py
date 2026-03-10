import json
import os
import re
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from termcolor import colored

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.utils.common import get_expand_runname
from src.utils.dataset_utils import dataset_mapping

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
if not os.path.isabs(RESULTS_DIR):
    RESULTS_DIR = str((ROOT_DIR / RESULTS_DIR).resolve())


def sort_runname_key(runname: str):
    prefix, query_samples, shot = runname.rsplit("-", 2)
    return (prefix, int(query_samples), int(re.findall(r"\d+", shot)[0]))


@hydra.main(config_path="../configs", config_name="analyze.yaml", version_base=None)
def main(cfg: DictConfig):
    verbose = cfg.verbose
    runname = get_expand_runname(cfg)
    topk = cfg.topk
    metric_key = dataset_mapping[cfg.data.name].metric_key()

    record_base_dir = os.path.join(RESULTS_DIR, "record")
    record_dirs = {
        name: os.path.join(record_base_dir, name)
        for name in os.listdir(record_base_dir)
        if name.startswith(runname)
        and os.path.isdir(os.path.join(record_base_dir, name))
    }
    ckpt_base_dir = os.path.join(RESULTS_DIR, "ckpt")
    ckpt_dirs = {
        name: os.path.join(ckpt_base_dir, name)
        for name in sorted(os.listdir(ckpt_base_dir), key=sort_runname_key)
        if name.startswith(runname) and os.path.isdir(os.path.join(ckpt_base_dir, name))
    }

    summary = {}
    missing_records = {}
    all_runnames = set(record_dirs.keys()) | set(ckpt_dirs.keys())

    for full_runname in all_runnames:
        meta_info = {}
        record_dir = record_dirs.get(full_runname, None)
        ckpt_dir = ckpt_dirs.get(full_runname, None)

        if record_dir and os.path.isdir(record_dir):
            for record_file in os.listdir(record_dir):
                if record_file.endswith(".json"):
                    epoch_ckpt = record_file.removesuffix(".json")
                    try:
                        with open(os.path.join(record_dir, record_file)) as f:
                            content = json.load(f)

                        metric_value = content["eval_result"].get(metric_key, None)
                        if metric_value is None:
                            raise KeyError(
                                f"Metric '{metric_key}' not found in eval_result."
                            )

                        if metric_key == "CIDEr":
                            metric_value *= 100

                        meta_info[epoch_ckpt] = metric_value

                    except (KeyError, json.JSONDecodeError) as e:
                        print(f"Error processing {record_file}: {e}", file=sys.stderr)
        else:
            if verbose:
                print(f"No record directory found for {full_runname}", file=sys.stderr)

        if ckpt_dir:
            if os.path.isdir(ckpt_dir):
                for epoch_ckpt in os.listdir(ckpt_dir):
                    if epoch_ckpt not in meta_info:
                        missing_epoch = int(re.findall(r"\d+", epoch_ckpt)[0])
                        missing_records.setdefault(full_runname, []).append(missing_epoch)
            else:
                print(
                    colored(f"{full_runname:<40}", "light_blue"),
                    "| Cannot find any checkpoints",
                    file=sys.stderr,
                )
                continue

        if meta_info:
            sorted_meta_info = sorted(
                meta_info.items(), key=lambda item: item[1], reverse=True
            )
            topk_meta_info = sorted_meta_info[:topk]
            summary[full_runname] = topk_meta_info

            print(
                colored(f"{full_runname:<40}", "light_blue"),
                f"| Top {min(topk, len(topk_meta_info))} records:",
            )
            for i, (epoch, metric) in enumerate(topk_meta_info):
                metric = round(metric, 2) if metric is not None else None
                print(f"  {i+1}. Epoch: {epoch} (Metric: {metric})")
        else:
            print(
                colored(f"{full_runname:<40}", "light_blue"),
                "| Cannot find any records",
                file=sys.stderr,
            )

    if verbose:
        if missing_records:
            print("Missing records detected:")
            for runname, epochs in missing_records.items():
                print(f"{runname} missing epochs: {sorted(epochs)}")
        elif not summary:
            print("No records or checkpoints are found.")
        else:
            print("All checkpoints are evaluated.")


if __name__ == "__main__":
    main()
