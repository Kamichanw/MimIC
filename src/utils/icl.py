import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Union, cast

from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

from .common import (
    DATASET_RETRIEVER_MAPPING,
    POSTPROCESS_MAPPING,
    register_dataset_retriever,
    register_postprocess,
)
from .sampler import ConcatSampler, MultiBatchSampler


IndexBatch = Union[int, List[int]]


def prepare_input(
    dataset_name: Union[str, List[str]],
    batch: List[List[Dict[str, Any]]],
    instruction: Optional[str] = None,
):
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name] * len(batch)
    elif isinstance(dataset_name, list):
        if len(dataset_name) != len(batch):
            raise ValueError(
                "Length of dataset_name list must match the batch size."
            )

    for name in dataset_name:
        if name not in DATASET_RETRIEVER_MAPPING:
            raise ValueError(
                f"The retriever of '{name}' is not registered, use `register_dataset_retriever` first."
            )

    return_annotations = [
        inspect.signature(DATASET_RETRIEVER_MAPPING[name]).return_annotation
        for name in dataset_name
    ]
    if len(set(return_annotations)) > 1:
        raise ValueError("All dataset retrievers must return the same type.")

    batch_context, batch_extra_outputs = [], []

    for name, context in zip(dataset_name, batch):
        messages, extra_outputs = [], []
        if instruction is not None:
            messages.append({"role": "instruction", "content": instruction})

        for item in context:
            retriever = DATASET_RETRIEVER_MAPPING[name]
            prepared_item = retriever(item, item == context[-1])

            if isinstance(prepared_item, tuple):
                msg, *rest = prepared_item
                messages.extend(msg)
                extra_outputs.append(tuple(rest))
            else:
                messages.extend(prepared_item)

        batch_context.append(messages)
        batch_extra_outputs.append(extra_outputs)

    if batch_extra_outputs and batch_extra_outputs[0] and all(
        isinstance(output, tuple) for output in batch_extra_outputs[0]
    ):
        num_extra_outputs = set(
            len(output) for outputs in batch_extra_outputs for output in outputs
        )
        if len(num_extra_outputs) != 1:
            raise RuntimeError(
                "Inconsistent number of extra outputs across different contexts."
            )

        return batch_context, *[
            list(outputs)
            for outputs in zip(
                *[
                    [list(i) for i in zip(*extra_outputs)]
                    for extra_outputs in batch_extra_outputs
                ]
            )
        ]

    return batch_context


def postprocess_generation(
    dataset_name: str,
    predictions: Union[str, List[str]],
    stop_words: Optional[List[str]] = None,
):
    is_batched = True
    if isinstance(predictions, str):
        predictions = [predictions]
        is_batched = False

    def preprocess(pred: str, stop_words: Optional[List[str]]):
        if stop_words is not None:
            pred = re.split(r"|".join(re.escape(word) for word in stop_words), pred, 1)[0]
        return pred.strip()

    if dataset_name not in POSTPROCESS_MAPPING:
        raise ValueError(
            f"The post process method of {dataset_name} is not registered, use `register_postprocess` first."
        )
    result = [
        POSTPROCESS_MAPPING[dataset_name](preprocess(pred, stop_words))
        for pred in predictions
    ]

    return result if is_batched else result[0]


def prepare_dataloader(
    datasets: Union[Dataset, List[Dataset]],
    batch_size: int,
    num_shots: Optional[int] = None,
    num_per_dataset: Optional[Union[int, List[int]]] = None,
    collate_fn: Optional[Callable[[List[List[Any]]], Any]] = None,
    samplers: Optional[Union[Sampler[IndexBatch], List[Sampler[IndexBatch]]]] = None,
    **kwargs,
) -> DataLoader:
    drop_last = kwargs.pop("drop_last", False)
    shuffle = kwargs.pop("shuffle", False)
    sampler = kwargs.pop("sampler", None)
    if sampler is not None:
        if samplers is not None:
            raise ValueError("Cannot specify sampler and samplers at the same time.")
        samplers = [sampler]

    def batchilize_sampler(
        dataset: Dataset,
        sampler: Optional[Sampler[IndexBatch]],
        minibatch_size: int,
    ) -> Sampler[List[int]]:
        if sampler is None:
            sampler = cast(
                Sampler[IndexBatch],
                RandomSampler(dataset) if shuffle else SequentialSampler(dataset),  # type: ignore[arg-type]
            )
        sample_idx = next(iter(sampler))
        if isinstance(sample_idx, int):
            return cast(
                Sampler[List[int]],
                BatchSampler(cast(Sampler[int], sampler), minibatch_size, True),
            )
        if (
            isinstance(sample_idx, list)
            and all(isinstance(idx, int) for idx in sample_idx)
            and len(sample_idx) == minibatch_size
        ):
            return cast(Sampler[List[int]], sampler)
        raise ValueError(
            f"Unable to get correct index from sampler {sampler}, it should yield an int or list of int of length {minibatch_size}."
        )

    def collate_fn_wrapper(batch):
        effective_num_shots = num_shots if num_shots is not None else 0
        batch_list = [
            batch[i * (effective_num_shots + 1) : (i + 1) * (effective_num_shots + 1)]
            for i in range(batch_size)
            if i * (effective_num_shots + 1) < len(batch)
        ]
        if collate_fn:
            return collate_fn(batch_list)
        return batch_list

    def check_consistent(name, obj, default_value) -> List[Any]:
        old_value = obj
        if obj is None:
            obj = default_value
        if isinstance(obj, list):
            if isinstance(datasets, list) and len(obj) != len(datasets):
                raise ValueError(
                    f"{name} should be a list of the same length as datasets, got {old_value}."
                )
            return obj
        return [obj]

    if not isinstance(datasets, list):
        datasets = [datasets]
    if num_shots is None:
        if num_per_dataset is not None:
            num_shots = (
                num_per_dataset - 1
                if isinstance(num_per_dataset, int)
                else sum(num_per_dataset) - 1
            )
        else:
            raise ValueError(
                "num_shots and num_per_dataset can't be None at the same time."
            )

    assert num_shots is not None
    num_per_dataset = check_consistent(
        "num_per_dataset", num_per_dataset, [num_shots + 1]
    )
    samplers = check_consistent("samplers", samplers, [None for _ in datasets])

    if sum(num_per_dataset) != num_shots + 1:
        raise ValueError("The sum of num_per_dataset should be equal to num_shots + 1.")

    prepared_samplers = [
        batchilize_sampler(dataset, sampler, minibatch_size)
        for dataset, sampler, minibatch_size in zip(datasets, samplers, num_per_dataset)
    ]
    concat_dataset: ConcatDataset = ConcatDataset(datasets)
    concat_sampler = ConcatSampler(prepared_samplers, concat_dataset.cumulative_sizes)

    return DataLoader(
        concat_dataset,
        collate_fn=collate_fn_wrapper,
        batch_sampler=MultiBatchSampler(concat_sampler, batch_size, drop_last),
        **kwargs,
    )


__all__ = [
    "prepare_dataloader",
    "prepare_input",
    "postprocess_generation",
    "register_dataset_retriever",
    "register_postprocess",
]
