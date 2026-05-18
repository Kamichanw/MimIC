import warnings
from collections.abc import Callable, Iterable
from typing import Any, cast
from pathlib import Path

import datasets

def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)


def split_generators(
    expand_path_fn: Callable[[str, str], Path],
    type_split_file_dict: dict[str, Any],
    verbose: bool = True,
):
    splits = set(
        [
            file_name
            for item in type_split_file_dict.values()
            for file_name in item.keys()
        ]
    )

    data_path = {
        split_name: {
            file_type: expand_path_fn(file_type, split_name)
            for file_type in type_split_file_dict.keys()
            if split_name in type_split_file_dict[file_type]
        }
        for split_name in splits
    }

    missing_files = []

    for split_name, dct in data_path.items():
        for path in dct.values():
            if not isinstance(path, list):
                path = [path]
            for p in path:
                if not p.exists():
                    missing_files.append(str(p))
                    if split_name in splits:
                        splits.remove(split_name)
                    if verbose:
                        warnings.warn(
                            f"{str(p)} is not exists. The {split_name} split will not be loaded."
                            "You can suppress this warning by set verbose to False when loading dataset."
                        )
                    break

    if len(splits) == 0:
        raise FileNotFoundError(
            f"Unable to load any splits because following files are not found: {', '.join(missing_files)}"
        )

    gen_kwargs = {
        split_name: {
            **{"split": split_name},
            **{
                f"{file_type}_path": data_path[split_name][file_type]
                for file_type in type_split_file_dict.keys()
            },
        }
        for split_name in splits
    }

    gen_name = {
        "train": datasets.Split.TRAIN,
        "test": datasets.Split.TEST,
        "val": datasets.Split.VALIDATION,
        "valid": datasets.Split.VALIDATION,
        "validation": datasets.Split.VALIDATION,
    }

    return [
        datasets.SplitGenerator(
            name=cast(str, gen_name[split] if split in gen_name else split),  # type: ignore[arg-type]
            gen_kwargs=gen_kwargs[split],
        )
        for split in cast(Iterable[str], splits)
    ]
