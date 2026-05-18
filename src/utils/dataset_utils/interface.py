import logging
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any, cast

import PIL
import PIL.Image
from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler, SequentialSampler

from src.models.model_base import ModelBase
from src.utils.icl import prepare_dataloader, prepare_input

logger = logging.getLogger(__name__)


def convert_images_to_rgb(images):
    if isinstance(images, PIL.Image.Image):
        return images.convert("RGB")
    if isinstance(images, Sequence):
        return [convert_images_to_rgb(img) for img in images]
    raise ValueError(f"Unsupported image type: {type(images)}")


class PostInitMeta(ABCMeta):
    """Post init metaclass to check some conditions after the class is initialized."""

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        assert hasattr(instance, "_support_set"), "Dataset must have _support_set."
        instance._support_set = instance._support_set.shuffle(seed=instance.cfg.seed)

        train_set_size = instance.cfg.num_query_samples
        actual_train_set_size = len(instance.support_set)
        if train_set_size:
            if train_set_size > actual_train_set_size:
                logger.warning(
                    f"cfg.num_query_samples {train_set_size} is larger than the support set size {actual_train_set_size}."
                    f"cfg.num_query_samples is set to the support set size."
                )
                instance.cfg.num_query_samples = actual_train_set_size
        else:
            logger.info(f"Using the full support set of size {actual_train_set_size}.")
            instance.cfg.num_query_samples = actual_train_set_size

        return instance


class DatasetBase(ABC, metaclass=PostInitMeta):
    support_datasets = []

    def __init__(self, data_cfg: DictConfig) -> None:
        self.cfg = data_cfg
        self._support_set: Dataset

        if self.name not in self.support_datasets:
            raise ValueError(
                f"Dataset {self.name} is not supported by {self.__class__.__name__}"
            )

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    @abstractmethod
    def instruction(self) -> str | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_role_in_round(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def extract_answer(self, item: dict) -> str:
        raise NotImplementedError

    @property
    def support_set(self) -> Dataset:
        return self._support_set

    def train_dataloader(
        self, model: ModelBase, batch_size: int, distributed: bool = True
    ) -> Any:
        def collate_fn(batch):
            results = {}
            ret = prepare_input(self.name, batch, instruction=self.instruction)
            if isinstance(ret, tuple) and len(ret) == 2:
                batch_context, batch_images = ret
                results["images"] = convert_images_to_rgb(batch_images)
            else:
                batch_context = ret

            prefix_texts = (
                model.apply_prompt_template(
                    [ctx[: -self.num_role_in_round] for ctx in batch_context]
                )
                if self.cfg.num_shot > 0
                else None
            )

            query_texts = model.apply_prompt_template(
                [ctx[-self.num_role_in_round :] for ctx in batch_context]
            )

            results.update(
                {
                    "prefix_texts": prefix_texts,
                    "query_texts": query_texts,
                    "answers": [self.extract_answer(item[-1]) for item in batch],
                }
            )

            return results

        if self.cfg.num_query_samples > len(self.support_set):
            logger.warning(
                f"cfg.num_query_samples {self.cfg.num_query_samples} is larger than the actual support set size {len(self.support_set)}."
                f"cfg.num_query_samples is set to the actual support set size."
            )
            self.cfg.num_query_samples = len(self.support_set)

        train_set = self.support_set.select(range(self.cfg.num_query_samples))
        example_sampler = RandomSampler(
            train_set,
            replacement=True,
            num_samples=self.cfg.num_shot * self.cfg.num_query_samples,
        )

        if distributed:
            query_sampler = DistributedSampler(train_set, shuffle=False)  # type: ignore[arg-type]
        else:
            query_sampler = SequentialSampler(train_set)

        if self.cfg.num_shot > 0:
            datasets = [train_set, train_set]
            samplers = [
                BatchSampler(
                    example_sampler,
                    batch_size=self.cfg.num_shot,
                    drop_last=True,
                ),
                query_sampler,
            ]
            dl_init_args = dict(
                datasets=datasets,
                samplers=samplers,
                num_per_dataset=[self.cfg.num_shot, 1],
            )
        else:
            dl_init_args = dict(
                datasets=train_set,
                samplers=query_sampler,
                num_shots=0,
            )

        return prepare_dataloader(
            **cast(dict[str, Any], dl_init_args),
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=True,
        )
