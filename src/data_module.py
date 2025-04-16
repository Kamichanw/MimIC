import torch
import torch_npu
import pytorch_lightning as pl
from PIL import ImageFile
from dataset_utils import dataset_mapping, DatasetBase

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, lmm) -> None:
        super().__init__()
        self.cfg = cfg
        self.lmm = lmm

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.dataset: DatasetBase = dataset_mapping[self.cfg.data.name](
                self.cfg.data
            )

    def train_dataloader(self):
        return self.dataset.train_dataloader(
            self.lmm, 
            self.cfg.batch_size,
            distributed=self.trainer.world_size > 1
        )