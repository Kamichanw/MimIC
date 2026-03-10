import json
import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import hydra
from accelerate import Accelerator
from accelerate.utils import gather_object
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.utils.common import get_expand_runname, load_from_pretrained
from src.utils.dataset_utils import dataset_mapping, DatasetBase
from src.utils.modeling import build_model
from src.utils.common import resolve_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="../configs", config_name="eval.yaml", version_base=None)
def main(cfg: DictConfig):
    results_dir = resolve_path(os.environ.get("RESULTS_DIR", "results"))
    accelerator = Accelerator()
    runname = get_expand_runname(cfg)
    is_icl = cfg.data.is_icl = cfg.ckpt_path is None
    dataset: DatasetBase = dataset_mapping[cfg.data.name](cfg.data)
    record_dir = os.path.join(results_dir, "record", runname)
    if is_icl:
        record_path = os.path.join(record_dir, f"{cfg.data.num_shot}shot.json")
    else:
        if not os.path.exists(cfg.ckpt_path):
            raise FileNotFoundError(f"Checkpoint path {cfg.ckpt_path} not found.")

        epoch = re.findall(r"\d+", os.path.basename(cfg.ckpt_path))
        if len(epoch) != 1:
            raise ValueError(
                f"Invalid checkpoint path {cfg.ckpt_path}. It should contain a single number in basename for epoch."
            )
        epoch = int(epoch[0])
        record_path = os.path.join(record_dir, f"epoch-{epoch}.json")

    os.makedirs(record_dir, exist_ok=True)

    if cfg.resume and os.path.exists(record_path):
        if accelerator.is_main_process:
            print(f"Found exist record {record_path}, skip...")
        return

    device = accelerator.device
    model = build_model(cfg).to(device, eval(cfg.dtype))

    if not is_icl:
        encoder = hydra.utils.instantiate(cfg.encoder.cls, lmm=model).to(
            device, eval(cfg.dtype)
        )

        print(f"Loading from pretrained: {cfg.ckpt_path}")
        load_from_pretrained(cfg.ckpt_path, model, encoder)
        encoder.eval()
        hooks = encoder.register_shift_hooks()

    model.eval()

    if accelerator.is_main_process:
        Path(record_path).touch()

    try:
        local_records = dataset.eval(cfg, model, accelerator=accelerator)
        gathered_records = gather_object(local_records)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if gathered_records and isinstance(gathered_records[0], list):
                result = [
                    record
                    for worker_records in gathered_records
                    for record in worker_records
                ]
            else:
                result = list(gathered_records)
            eval_result = dataset.compute_metrics(result)
            print(f"Evaluation result for {runname}: {eval_result}")
            config = {"eval_args": OmegaConf.to_container(cfg, resolve=True)}
            if os.path.exists(os.path.join(record_dir, "config.json")):
                with open(os.path.join(record_dir, "config.json")) as f:
                    config["train_args"] = json.load(f)

            with open(record_path, "w") as f:
                json.dump(
                    {
                        **config,
                        "eval_result": eval_result,
                        "records": result,
                    },
                    f,
                    ensure_ascii=False,
                )

    finally:
        if (
            accelerator.is_main_process
            and os.path.exists(record_path)
            and os.path.getsize(record_path) == 0
        ):
            os.remove(record_path)


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    main()
