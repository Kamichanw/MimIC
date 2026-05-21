import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import set_seed

from src.trainer import MimicTrainer, Strategy
from src.utils.dataset_utils import dataset_mapping
from src.utils.modeling import build_model, convert_to_peft

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.data.seed)

    lmm = build_model(cfg)
    convert_to_peft(cfg, lmm)
    shift_encoder = hydra.utils.instantiate(cfg.encoder.cls, _partial_=True)(lmm=lmm)
    data_module = dataset_mapping[cfg.data.name](cfg.data)
    training_args = hydra.utils.instantiate(cfg.training_args)

    trainer = MimicTrainer(
        cfg=cfg,
        lmm=lmm,
        strategy=eval(cfg.encoder.model_strategy),
        model=shift_encoder,
        args=training_args,
        train_dataset=data_module,
    )
    trainer.train()


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    main()
