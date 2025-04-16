import torch
import torch_npu
import logging
import os
import PIL
import PIL.Image
from torch.utils.data import (
    BatchSampler,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from datasets import Dataset
from typing import Any, Dict, List, Optional, Sequence, Tuple
from omegaconf import DictConfig
from abc import ABC, ABCMeta, abstractmethod
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from testbed.data import prepare_dataloader, prepare_input
from testbed.models.model_base import ModelBase

logger = logging.getLogger(__name__)


def convert_images_to_rgb(images):
    if isinstance(images, PIL.Image.Image):
        return images.convert("RGB")
    elif isinstance(images, Sequence):
        return [convert_images_to_rgb(img) for img in images]
    else:
        raise ValueError(f"Unsupported image type: {type(images)}")


class PostInitMeta(ABCMeta):
    """Post init metaclass to check some conditions after the class is initialized."""

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        assert hasattr(instance, "_support_set") and hasattr(
            instance, "_query_set"
        ), f"Dataset must have _support_set and _query_set attributes."
        instance._support_set = instance._support_set.shuffle(seed=instance.cfg.seed)

        query_set_size = instance.cfg.num_query_samples
        actual_query_set_size = len(instance.query_set)
        if query_set_size:
            if query_set_size > actual_query_set_size:
                logger.warning(
                    f"cfg.num_query_samples {query_set_size} is larger than the actual query set size {actual_query_set_size}."
                    f"cfg.num_query_samples is set to the actual query set size."
                )
                instance.cfg.num_query_samples = actual_query_set_size

            instance._query_set = instance._query_set.shuffle(
                seed=instance.cfg.seed
            ).select(range(int(instance.cfg.num_query_samples)))
        else:
            logger.info(
                f"cfg.num_query_samples is not specified. Using the full query set of size {actual_query_set_size}."
            )
            instance.cfg.num_query_samples = actual_query_set_size

        return instance


class DatasetBase(ABC, metaclass=PostInitMeta):
    """
    Base class for a dataset. It provides a common interface for different datasets.
    The subclasses should:
        1. implement all abstract methods.
        2. create support_datasets attribute. These names will determine what cfg.data.name is allowed to be.
        3. create _support_set and _query_set attributes in the __init__ method.
    Please note that you don't have to select cfg.num_query_samples samples from the query set and
    shuffle it manually. The base class will take care of it.
    """

    # list of dataset names that this class supports
    # defaults to data_cfg.name
    support_datasets = []

    def __init__(self, data_cfg: DictConfig) -> None:
        self.cfg = data_cfg
        self._support_set: Dataset
        self._query_set: Dataset

        if self.name not in self.support_datasets:
            raise ValueError(
                f"Dataset {self.name} is not supported by {self.__class__.__name__}"
            )

    @property
    def name(self) -> str:
        return self.cfg.name

    @staticmethod
    @abstractmethod
    def metric_key() -> str:
        """Returns the key from eval_result dict to analyze"""
        raise NotImplementedError

    @property
    @abstractmethod
    def instruction(self) -> Optional[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_role_in_round(self) -> int:
        """
        Returns the number of roles in a round. For example, in VQA, 3 roles are required
        to form a complete conversation round: image, question, and answer.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_answer(self, item: Dict) -> str:
        """
        Extract the answer from an item in the dataset.
        """
        raise NotImplementedError

    @property
    def support_set(self) -> Dataset:
        """Returns a support set, typically used for training or demonstration selection."""
        return self._support_set

    @property
    def query_set(self) -> Dataset:
        """
        Returns a query set, typically used for query selection. It should only
        contains cfg.num_query_samples samples.
        """
        return self._query_set

    @abstractmethod
    def eval(
        self,
        eval_cfg: DictConfig,
        model: ModelBase,
    ) -> Tuple[List[Dict], Dict]:
        """
        Evaluate a model with the dataset. Returns a list of metadata and a dictionary of metrics.

        Args:
            eval_cfg: Evaluation configuration.
            model: A model to evaluate.

        Returns:
            A tuple of a list of metadata and a dictionary of metrics.
        """
        pass

    def get_prediction(
        self,
        model: ModelBase,
        batch: Any,
        max_skip_oom: Optional[int] = None,
        **generation_args,
    ) -> Optional[List[str]]:
        """
        Get prediction from the model given a batch of data. It is possible to skip the prediction
        if the model runs out of memory, for a maximum number of times specified by `max_skip_oom`.

        Returns:
            A list of strings, each string is a generated
            response from the model. If the model runs out of memory, return None.
        """
        max_skip_oom = max_skip_oom or 0
        ret = prepare_input(self.name, batch, instruction=self.instruction)
        if isinstance(ret, tuple) and len(ret) == 2:
            # it could be a lmm that returns (text, images)
            context, images = ret
            args_to_generate = (convert_images_to_rgb(images), context)
        else:
            # it could be a llm that returns text only
            context = ret
            args_to_generate = (context,)

        try:
            return model.generate(*args_to_generate, **generation_args)
        except Exception as e:
            if "out of memory" not in str(e):
                raise
            if not hasattr(self, "__num_skip_oom"):
                self.__num_skip_oom = 0
            if self.__num_skip_oom >= max_skip_oom:
                raise
            else:
                self.__num_skip_oom += 1

        return None

    def train_dataloader(
        self, model: ModelBase, batch_size: int, distributed: bool = True
    ) -> Any:
        """
        Returns a training dataloader.
        """

        def collate_fn(batch):
            """
            Split batch into full context, in-context examples, query and answer, and process them into model inputs.
            """
            results = {}
            ret = prepare_input(self.name, batch, instruction=self.instruction)
            if isinstance(ret, tuple) and len(ret) == 2:
                # it could be a lmm that returns (text, images)
                batch_context, batch_images = ret
                results["images"] = convert_images_to_rgb(batch_images)
            else:
                # it could be a llm that returns text only
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

        # prepare example sampler
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

        # prepare query sampler
        if distributed:
            query_sampler = DistributedSampler(train_set, shuffle=False)
        else:
            query_sampler = SequentialSampler(train_set)

        if self.cfg.num_shot > 0:
            # we use the same dataset for both query set and support set
            # because we assume we only have cfg.num_query_samples data
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
            datasets = train_set
            samplers = query_sampler
            dl_init_args = dict(
                datasets=datasets,
                samplers=samplers,
                num_shots=0,
            )

        return prepare_dataloader(
            **dl_init_args,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def validation_dataloader(self, batch_size: int) -> Any:
        """
        Returns a validation dataloader.
        """

        if self.cfg.num_shot > 0:
            total_required_examples = self.cfg.num_shot * self.cfg.num_query_samples
            if total_required_examples > len(self.support_set):
                support_set_sampler = RandomSampler(
                    self.support_set,
                    replacement=True,
                    num_samples=total_required_examples,
                )
            else:
                support_set_sampler = RandomSampler(self.support_set)
            dataloader = prepare_dataloader(
                [self.support_set, self.query_set],
                batch_size=batch_size,
                num_per_dataset=[self.cfg.num_shot, 1],
                samplers=[
                    support_set_sampler,
                    SequentialSampler(self.query_set),
                ],
                drop_last=True,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )
        else:
            dataloader = prepare_dataloader(
                self.query_set,
                batch_size=batch_size,
                num_shots=0,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )

        return dataloader
