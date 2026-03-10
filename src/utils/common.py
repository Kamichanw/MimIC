import os
from functools import wraps
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, cast
import warnings

import datasets
import torch
from omegaconf import DictConfig
from peft.peft_model import PeftModel

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_RETRIEVER_MAPPING: Dict[str, Callable] = {}
POSTPROCESS_MAPPING: Dict[str, Callable] = {}


def resolve_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return str(path.resolve())


# update scripts/pipeline.py if the runname format changes
def get_expand_runname(cfg: DictConfig):
    """
    Get the expanded runname based on the train, eval or analyze config.
    """
    if hasattr(cfg, "runname"):
        if cfg.data.num_shot == 0:
            return f"{cfg.runname}-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_query_samples}"
        return f"{cfg.runname}-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_query_samples}-{cfg.data.num_shot}shot"

    if hasattr(cfg, "record_dir"):
        return os.path.basename(cfg.record_dir)

    if getattr(cfg, "ckpt_path", None):
        return os.path.basename(os.path.dirname(cfg.ckpt_path))
    return f"icl-{cfg.model_name}-{cfg.data.name}"


def save_pretrained(save_directory, lmm, encoder):
    os.makedirs(save_directory, exist_ok=True)
    encoder_sd = {
        k: v for k, v in encoder.state_dict().items() if not k.startswith("lmm")
    }
    torch.save(encoder_sd, os.path.join(save_directory, "encoder.pth"))
    lmm.model.save_pretrained(save_directory)


def load_from_pretrained(save_directory, lmm, encoder):
    sd = torch.load(os.path.join(save_directory, "encoder.pth"), weights_only=True)
    if sd:
        missing_keys, _unexpected_keys = encoder.load_state_dict(sd, strict=False)
        missing_keys = [k for k in missing_keys if not k.startswith("lmm")]
        if missing_keys:
            raise RuntimeError(f"Missing key(s) in state_dict: {missing_keys}")
    if os.path.exists(os.path.join(save_directory, "adapter_config.json")):
        lmm.model = PeftModel.from_pretrained(encoder.lmm.model, save_directory)


def register_dataset_retriever(dataset_name: str):
    def decorator(retriever: Callable[[Any, bool], Any]):
        if not callable(retriever):
            raise TypeError(f"{retriever.__name__} must be callable.")

        sig = inspect.signature(retriever)
        params = list(sig.parameters.values())
        if len(params) != 2:
            raise TypeError(
                f"{retriever.__name__} must accept two arguments: an item and a boolean flag."
            )

        DATASET_RETRIEVER_MAPPING[dataset_name] = retriever

        @wraps(retriever)
        def wrapper(*args, **kwargs):
            return retriever(*args, **kwargs)

        return wrapper

    return decorator


def register_postprocess(dataset_name: str):
    def decorator(postprocess: Callable[[str], Any]):
        if not callable(postprocess):
            raise TypeError(f"{postprocess.__name__} must be callable.")

        sig = inspect.signature(postprocess)
        params = list(sig.parameters.values())
        if len(params) != 1:
            raise TypeError(
                f"{postprocess.__name__} must accept a single string argument."
            )

        POSTPROCESS_MAPPING[dataset_name] = postprocess

        @wraps(postprocess)
        def wrapper(*args, **kwargs):
            return postprocess(*args, **kwargs)

        return wrapper

    return decorator


def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)


def split_generators(
    expand_path_fn: Callable[[str, str], Path],
    type_split_file_dict: Dict[str, Any],
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
