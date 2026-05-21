from typing import Any

import hydra
import torch
from accelerate import Accelerator, DistributedType
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.models import MODEL_REGISTRY_V2
from lmms_eval.models.registry_v2 import ModelManifest
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models import Idefics, Idefics2
from src.models.llava import LLaVa
from src.utils.common import load_from_pretrained


class MimicEvalModel(lmms):
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        peft: str | None = None,
        dtype: str = "float16",
        device: str = "cuda",
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided."
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = torch.device(device)

        self.batch_size_per_gpu = int(batch_size)
        self.lmm = self._build_lmm(model_name, pretrained, getattr(torch, dtype))
        self.encoder = None
        if peft:
            ckpt_cfg = OmegaConf.load(f"{peft}/config.json")
            self.encoder = hydra.utils.instantiate(
                ckpt_cfg.encoder.cls,
                _partial_=True,
            )(lmm=self.lmm)
            load_from_pretrained(peft, self.lmm, self.encoder)

        self.lmm.to(self._device)
        self.lmm.eval()
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer(self):
        return self.lmm.processor.tokenizer

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def _build_lmm(self, model_name: str, pretrained: str, torch_dtype: torch.dtype):
        if model_name == "idefics-9b":
            return Idefics(pretrained, torch_dtype=torch_dtype)
        if model_name == "idefics2-8b-base":
            return Idefics2(
                pretrained,
                torch_dtype=torch_dtype,
                processor_args={"do_image_splitting": False},
            )
        if model_name == "llava-interleave-7b":
            return LLaVa(pretrained, torch_dtype=torch_dtype)
        raise ValueError(f"Unsupported model_name: {model_name}")

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(
            string,
            add_special_tokens=add_special_tokens,
        )
        return encoding[-left_truncate_len:] if left_truncate_len else encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _with_image_tokens(self, context: str, visuals: list) -> str:
        if not visuals or "<image>" in context:
            return context
        return " ".join(["<image>"] * len(visuals)) + "\n" + context

    def _remove_hooks(self, hooks) -> None:
        for handles in hooks.values():
            if isinstance(handles, list):
                for handle in handles:
                    handle.remove()
            else:
                handles.remove()

    def _generate(self, visuals, prompts, gen_kwargs):
        hooks = self.encoder.register_shift_hooks() if self.encoder is not None else {}
        try:
            return self.lmm.generate(
                visuals,
                prompts,
                processor_args={"padding": True},
                **gen_kwargs,
            )
        finally:
            self._remove_hooks(hooks)

    def generate_until(self, requests: list[Instance]) -> list[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator(
            [request.args for request in requests],
            _collate,
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(total=len(chunks), disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            visuals = [
                doc_to_visual_fn(self.task_dict[task_name][split_name][idx])
                for doc_to_visual_fn, idx, task_name, split_name in zip(
                    doc_to_visual,
                    doc_id,
                    task,
                    split,
                )
            ]
            prompts = [
                self._with_image_tokens(context, visual)
                for context, visual in zip(contexts, visuals)
            ]
            gen_kwargs = dict(all_gen_kwargs[0])
            until = gen_kwargs.pop("until", None)
            gen_kwargs.pop("image_aspect_ratio", None)
            if gen_kwargs.get("temperature") == 0:
                gen_kwargs.pop("temperature")
                gen_kwargs["do_sample"] = False
            gen_kwargs.setdefault("max_new_tokens", 1024)

            text_outputs = self._generate(visuals, prompts, gen_kwargs)
            if until:
                until = [until] if isinstance(until, str) else until
                text_outputs = [
                    next((text.split(term)[0] for term in until if term in text), text)
                    for text in text_outputs
                ]
            res.extend(text_outputs)
            pbar.update(1)
        pbar.close()
        return re_ords.get_original(res)

    def loglikelihood(self, requests: list[Instance]):
        raise NotImplementedError("Loglikelihood is not implemented for MimIC eval.")

    def generate_until_multi_round(self, requests) -> list[str]:
        raise NotImplementedError("Multi-round generation is not implemented.")


MODEL_REGISTRY_V2.register_manifest(
    ModelManifest(
        model_id="mimic",
        simple_class_path="src.lmms_eval_model.MimicEvalModel",
    ),
    overwrite=True,
)
