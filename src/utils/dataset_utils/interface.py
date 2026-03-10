import logging
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, cast

import PIL
import PIL.Image
from accelerate import Accelerator
from datasets import Dataset
from omegaconf import DictConfig
from torch.utils.data import BatchSampler, DistributedSampler, RandomSampler, SequentialSampler
from tqdm import tqdm

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
        raise NotImplementedError

    @property
    @abstractmethod
    def instruction(self) -> Optional[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_role_in_round(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def extract_answer(self, item: Dict) -> str:
        raise NotImplementedError

    @property
    def support_set(self) -> Dataset:
        return self._support_set

    @property
    def query_set(self) -> Dataset:
        return self._query_set

    @abstractmethod
    def eval(
        self,
        eval_cfg: DictConfig,
        model: ModelBase,
        accelerator: Optional[Accelerator] = None,
    ) -> List[Dict]:
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, records: List[Dict]) -> Dict:
        raise NotImplementedError

    def get_prediction(
        self,
        model: ModelBase,
        batch: Any,
        max_skip_oom: Optional[int] = None,
        **generation_args,
    ) -> Optional[List[str]]:
        max_skip_oom = max_skip_oom or 0
        ret = prepare_input(self.name, batch, instruction=self.instruction)
        if isinstance(ret, tuple) and len(ret) == 2:
            context, images = ret
            args_to_generate = (convert_images_to_rgb(images), context)
        else:
            context = ret
            args_to_generate = (context,)

        try:
            return cast(Optional[List[str]], model.generate(*args_to_generate, **generation_args))
        except Exception as e:
            if "out of memory" not in str(e):
                raise
            if not hasattr(self, "__num_skip_oom"):
                self.__num_skip_oom = 0
            if self.__num_skip_oom >= max_skip_oom:
                raise
            self.__num_skip_oom += 1

        return None

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
            **cast(Dict[str, Any], dl_init_args),
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def validation_dataloader(self, batch_size: int) -> Any:
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
                cast(List[Any], [self.support_set, self.query_set]),
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
                cast(Any, self.query_set),
                batch_size=batch_size,
                num_shots=0,
                num_workers=self.cfg.num_workers,
                pin_memory=True,
            )

        return dataloader

    def iter_eval_batches(
        self,
        eval_dl,
        iterations: int,
        model_name: str,
        runname: str,
        accelerator: Optional[Accelerator] = None,
    ):
        world_size = accelerator.num_processes if accelerator else 1
        process_index = accelerator.process_index if accelerator else 0
        local_total = max((iterations - process_index + world_size - 1) // world_size, 0)

        iterator = tqdm(
            total=local_total,
            disable=bool(accelerator and not accelerator.is_local_main_process),
            desc=f"Evaluating {model_name} with {runname} ...",
        )

        try:
            for step, batch in enumerate(eval_dl):
                if step >= iterations:
                    break
                if step % world_size != process_index:
                    continue
                iterator.update(1)
                yield batch
        finally:
            iterator.close()
