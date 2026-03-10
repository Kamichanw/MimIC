from contextlib import nullcontext
import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from peft import LoraConfig, PrefixTuningConfig, get_peft_model  # type: ignore

from src.models import Idefics, Idefics2
from src.models.llava import LLaVa
from src.utils.common import resolve_path


class NullPeftModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = model
        model.requires_grad_(False)
        self.config = model.config

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def disable_adapter(self):
        return nullcontext()

    def save_pretrained(self, save_directory: str, **kwargs):
        pass


class ModuleDeviceManager:
    def __init__(self, device):
        self.device = device
        self.old_devices = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, old_device in self.old_devices.items():
            module.to(old_device)

    def move_module(self, module):
        self.old_devices[module] = next(module.parameters()).device
        module.to(self.device)


def convert_to_peft(cfg, lmm):
    peft_config_cls = None
    if cfg.model_name in cfg.peft:
        peft_cfg = OmegaConf.to_container(cfg.peft[cfg.model_name], resolve=True)
        if cfg.peft.name == "lora":
            peft_config_cls = LoraConfig
        elif cfg.peft.name == "prefix-tuning":
            peft_config_cls = PrefixTuningConfig

    if peft_config_cls is None:
        model = NullPeftModel(lmm.model)
    else:
        model = get_peft_model(lmm.model, peft_config_cls(**peft_cfg))  # type: ignore[arg-type]
    lmm.model = model


def build_model(cfg):
    if cfg.model_name == "idefics-9b":
        lmm = Idefics(
            resolve_path(os.environ["IDEFICS_9B_PATH"]),
            torch_dtype=eval(cfg.dtype),
        )
    elif cfg.model_name == "idefics2-8b-base":
        processor_args: dict[str, object] = {
            "do_image_splitting": False,
        }
        if "seed" in cfg.data.name or "mme" in cfg.data.name:
            processor_args["largest_edges"] = 448
            processor_args["shortest_edges"] = 378

        lmm = Idefics2(
            resolve_path(os.environ["IDEFICS2_8B_BASE_PATH"]),
            torch_dtype=eval(cfg.dtype),
            processor_args=processor_args,
        )
    elif cfg.model_name == "llava-interleave-7b":
        lmm = LLaVa(
            resolve_path(os.environ["LLAVA_INTERLEAVE_7B_PATH"]),
            torch_dtype=eval(cfg.dtype),
        )
    else:
        raise ValueError(f"Unsupport model {cfg.model_name}")
    return lmm
