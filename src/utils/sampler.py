from collections.abc import Iterable, Iterator
from typing import Protocol, cast

from torch.utils.data.sampler import Sampler


class _SizedSampler(Protocol):
    def __len__(self) -> int: ...


class ConcatSampler(Sampler[list[int]]):
    def __init__(
        self,
        samplers: Iterable[Sampler],
        cumulative_dataset_sizes: list[int] | None = None,
    ):
        self.samplers = list(samplers)
        sample_indices = [next(iter(sampler)) for sampler in self.samplers]
        self.batch_size = sum(
            len(idx) if isinstance(idx, list) else 1 for idx in sample_indices
        )
        self.cumulative_indices = (
            [0] + cumulative_dataset_sizes[:-1]
            if cumulative_dataset_sizes is not None
            else [0] * len(self.samplers)
        )

    def __iter__(self) -> Iterator[list[int]]:
        sampler_iters = [iter(sampler) for sampler in self.samplers]
        while True:
            try:
                sample_indices = [next(it) for it in sampler_iters]
                batch: list[int] = []
                for i, mini in enumerate(sample_indices):
                    if isinstance(mini, list):
                        batch.extend(idx + self.cumulative_indices[i] for idx in mini)
                    else:
                        batch.append(mini + self.cumulative_indices[i])

                yield batch
            except StopIteration:
                break

    def __len__(self):
        return min(len(cast(_SizedSampler, sampler)) for sampler in self.samplers)


class MultiBatchSampler(Sampler[list[int]]):
    def __init__(
        self, sampler: Sampler[list[int]], multi_merge_size: int, drop_last: bool
    ) -> None:
        if (
            not isinstance(multi_merge_size, int)
            or isinstance(multi_merge_size, bool)
            or multi_merge_size <= 0
        ):
            raise ValueError(
                "merge_size should be a positive integer value, "
                f"but got merge_size={multi_merge_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        if isinstance(next(iter(sampler)), int):
            raise ValueError("batch_sampler should yield a list of int.")

        self.sampler = sampler
        batch_size = getattr(sampler, "batch_size", 1)
        self.batch_size = multi_merge_size * batch_size
        self.merge_size = multi_merge_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch_list = [next(sampler_iter) for _ in range(self.merge_size)]
                    yield [idx for mini in batch_list for idx in mini]
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for indices in self.sampler:
                for idx in indices:
                    batch[idx_in_batch] = idx
                    idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        sampler_len = len(cast(_SizedSampler, self.sampler))
        if self.drop_last:
            return sampler_len // self.merge_size
        return (sampler_len + self.merge_size - 1) // self.merge_size
