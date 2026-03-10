from omegaconf import OmegaConf

from .common import (
    ROOT_DIR,
    get_expand_runname,
    load_from_pretrained,
    register_dataset_retriever,
    register_postprocess,
    resolve_path,
    save_pretrained,
)
from .icl import postprocess_generation, prepare_dataloader, prepare_input
from .modeling import ModuleDeviceManager, NullPeftModel, build_model, convert_to_peft
from .sampler import ConcatSampler, MultiBatchSampler

OmegaConf.register_new_resolver("eval", eval, replace=True)


__all__ = [
    "ROOT_DIR",
    "resolve_path",
    "prepare_input",
    "prepare_dataloader",
    "postprocess_generation",
    "register_dataset_retriever",
    "register_postprocess",
    "ConcatSampler",
    "MultiBatchSampler",
    "ModuleDeviceManager",
    "NullPeftModel",
    "build_model",
    "convert_to_peft",
    "get_expand_runname",
    "load_from_pretrained",
    "save_pretrained",
]
