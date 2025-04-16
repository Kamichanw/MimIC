from contextlib import nullcontext
import torch
import torch_npu

import os
import torch
import torch.nn as nn
import peft
from peft import PeftModel, LoraConfig, PrefixTuningConfig
from omegaconf import DictConfig, OmegaConf

import paths
from testbed.models.llava import LLaVa
from testbed.models import Idefics, Idefics2

OmegaConf.register_new_resolver("eval", eval, replace=True)


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
        model = peft.get_peft_model(lmm.model, peft_config_cls(**peft_cfg))
    lmm.model = model


def build_model(cfg):
    if cfg.model_name == "idefics-9b":
        lmm = Idefics(
            paths.idefics_9b_path,
            torch_dtype=eval(cfg.dtype),
        )
    elif cfg.model_name == "idefics2-8b-base":
        processor_args = {
            "do_image_splitting": False,
        }
        if "seed" in cfg.data.name or "mme" in cfg.data.name:
            # seed bench cannot even run 1 shot with the default setting
            processor_args["largest_edges"] = 448
            processor_args["shortest_edges"] = 378

        lmm = Idefics2(
            paths.idefics2_8b_base_path,
            torch_dtype=eval(cfg.dtype),
            processor_args=processor_args,
        )
    elif cfg.model_name == "llava-interleave-7b":
        lmm = LLaVa(
            paths.llava_interleave_7b_path,
            torch_dtype=eval(cfg.dtype),
        )
    else:
        raise ValueError(f"Unsupport model {cfg.model_name}")
    return lmm


def save_pretrained(save_directory, lmm, encoder):
    os.makedirs(save_directory, exist_ok=True),
    encoder_sd = {
        k: v for k, v in encoder.state_dict().items() if not k.startswith("lmm")
    }
    torch.save(encoder_sd, os.path.join(save_directory, "encoder.pth"))
    lmm.model.save_pretrained(save_directory)


def load_from_pretrained(save_directory, lmm, encoder):
    sd = torch.load(os.path.join(save_directory, "encoder.pth"), weights_only=True)
    if sd:
        missing_keys, unexpected_keys = encoder.load_state_dict(sd, strict=False)

        # keys started with "lmm" is not related to shift encoder
        missing_keys = [k for k in missing_keys if not k.startswith("lmm")]
        if missing_keys:
            raise RuntimeError(f"Missing key(s) in state_dict: {missing_keys}")
    if os.path.exists(os.path.join(save_directory, "adapter_config.json")):
        lmm.model = PeftModel.from_pretrained(encoder.lmm.model, save_directory)


# update execute_eval in eval.py if runname format is changed
def get_expand_runname(cfg: DictConfig):
    """
    Get the expanded runname based on the train, eval or analyze config.
    """
    if hasattr(cfg, "runname"):
        # training mode
        if cfg.data.num_shot == 0:
            # peft
            return f"{cfg.runname}-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_query_samples}"
        # runname-model-dataset-training_samples-num_shot
        return f"{cfg.runname}-{cfg.model_name}-{cfg.data.name}-{cfg.data.num_query_samples}-{cfg.data.num_shot}shot"

    if hasattr(cfg, "record_dir"):
        # analyze mode
        # record dir format: path/to/{expand-runname}
        return os.path.basename(cfg.record_dir)

    if getattr(cfg, "ckpt_path", None):
        # eval from a specific checkpoint
        # checkpoint dir format: path/to/{expand-runname}/epoch-{epoch}
        return os.path.basename(os.path.dirname(cfg.ckpt_path))
    else:
        # eval from ICL
        # icl-model-dataset
        return f"icl-{cfg.model_name}-{cfg.data.name}"
