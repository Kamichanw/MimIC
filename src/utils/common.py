import os
from functools import wraps
import inspect
from pathlib import Path
from collections.abc import Callable
from typing import Any
import torch
from omegaconf import DictConfig
from peft.peft_model import PeftModel

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_RETRIEVER_MAPPING: dict[str, Callable] = {}


def resolve_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return str(path.resolve())


# Keep train/eval record names aligned when this format changes.
def get_expand_runname(cfg: DictConfig):
    """
    Get the expanded runname based on the train or eval config.
    """
    if hasattr(cfg, "runname"):
        if cfg.data.num_shot == 0:
            return f"{cfg.runname}-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_query_samples}"
        return f"{cfg.runname}-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_query_samples}-{cfg.data.num_shot}shot"

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