import os
import torch
import torch_npu

import paths
from data_module import DataModule
from shift_model import ShiftModel, Strategy
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from termcolor import colored
import hydra
from omegaconf import DictConfig
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(config_path="config", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):
    
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

    max_epochs = 5
    
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
    runname = get_expand_runname(cfg)
    print(colored(f"Training for {runname} on {cfg.model_name}", "blue"))

    if cfg.resume:
        save_dir = os.path.join(paths.result_dir, "ckpt", runname)
        os.makedirs(save_dir, exist_ok=True)
        exist_ckpt_epochs = [
            int(d.split("-")[-1])
            for d in os.listdir(save_dir)
            if os.path.exists(os.path.join(save_dir, d))
        ]
        for i in range(max_epochs):
            if save_when(i) and i not in exist_ckpt_epochs:
                break
        else:
            print(f"All checkpoints {runname} matched, skip...")
            return
    pl.seed_everything(cfg.data.seed)
    os.makedirs(paths.result_dir, exist_ok=True)
    wb_logger = WandbLogger(
        save_dir=paths.result_dir,
        name=runname,
        project="MimIC",
        log_model=False,
    )
    torch.set_float32_matmul_precision("medium")


    trainer = pl.Trainer(
    logger=wb_logger,
    callbacks=[
        LearningRateMonitor(),
        RichProgressBar()
    ],
    max_epochs=max_epochs,
    accelerator="npu",  
    devices=[0],      
    strategy="auto",  
    precision=cfg.precision,
    gradient_clip_val=cfg.grad_clip_val,
    log_every_n_steps=2,
    accumulate_grad_batches=cfg.accumulate_grad_batches,
    enable_checkpointing=True,
)


    lmm = build_model(cfg).to("npu:0")  
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
    main()