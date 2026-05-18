import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.trainer import MimicTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):
    trainer = MimicTrainer.from_config(cfg)
    trainer.train()


if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env")
    main()
