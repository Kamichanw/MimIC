import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_module import DataModule
from src.shift_model import ShiftModel, Strategy
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from termcolor import colored
import hydra
from omegaconf import DictConfig
from src.utils.common import get_expand_runname
from src.utils.modeling import build_model, convert_to_peft
from src.utils.common import resolve_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):
    results_dir = resolve_path(os.environ.get("RESULTS_DIR", "results"))

    def get_max_epochs():
        num_query_samples = cfg.data.num_query_samples
        model_name = cfg.model_name
        if "idefics-9b" in model_name:
            if num_query_samples < 100:
                return 15
            if num_query_samples <= 500:
                return 10
            return 10
        elif "idefics2-8b" in model_name:
            if num_query_samples < 100:
                return 15
            if num_query_samples <= 500:
                return 10
            return 5
        elif "llava" in model_name:
            if num_query_samples <= 500:
                return 10
            return 5

    def save_when(epoch):
        num_query_samples = cfg.data.num_query_samples
        model_name = cfg.model_name
        if "idefics-9b" in model_name:
            if num_query_samples < 100:
                return epoch >= 10
            if num_query_samples <= 200:
                if cfg.data.name == "coco":
                    return epoch >= 5
                return epoch >= 7
            if num_query_samples <= 500:
                return epoch >= 5
            return epoch >= 5
        elif "idefics2-8b":
            if num_query_samples < 100:
                return epoch >= 10
            if num_query_samples <= 500:
                return epoch >= 5
            return True
        elif "llava" in model_name:
            if num_query_samples <= 1000:
                return epoch >= 5
            return True

    max_epochs = cfg.epochs if cfg.epochs else get_max_epochs()
    max_epochs_value = int(max_epochs or 0)
    runname = get_expand_runname(cfg)
    print(colored(f"Training for {runname} on {cfg.model_name}", "light_blue"))

    if cfg.resume:
        save_dir = os.path.join(results_dir, "ckpt", runname)
        os.makedirs(save_dir, exist_ok=True)
        exist_ckpt_epochs = [
            int(d.split("-")[-1])
            for d in os.listdir(save_dir)
            if os.path.exists(os.path.join(save_dir, d))
        ]
        for i in range(max_epochs_value):
            if save_when(i) and i not in exist_ckpt_epochs:
                break
        else:
            print(f"All checkpoints {runname} matched, skip...")
            return
    pl.seed_everything(cfg.data.seed)
    os.makedirs(results_dir, exist_ok=True)
    wb_logger = WandbLogger(
        save_dir=results_dir,
        name=runname,
        project="VQAInContextVector",
        log_model=False,
    )
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        logger=wb_logger,
        callbacks=[
            LearningRateMonitor(),
            RichProgressBar(),
        ],
        # fast_dev_run=True,
        # devices=1,
        max_epochs=max_epochs,
        devices=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
        use_distributed_sampler=False,
        strategy=cfg.strategy,
        precision=cfg.precision,
        gradient_clip_val=cfg.grad_clip_val,
        log_every_n_steps=2,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        enable_checkpointing=False,
    )

    lmm = build_model(cfg)
    convert_to_peft(cfg, lmm)
    data_module = DataModule(cfg, lmm)
    shift_encoder = hydra.utils.instantiate(cfg.encoder.cls, _partial_=True)(lmm=lmm)

    model = ShiftModel(
        cfg,
        shift_encoder,
        eval(cfg.encoder.model_strategy),
        save_checkpoint_when=save_when,
    )
    trainer.fit(
        model,
        data_module,
    )


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    main()
