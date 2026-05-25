import json
import os
import enum
from functools import reduce
from typing import Any, cast

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import ImageFile
from torch.utils.data import DataLoader
from transformers import Trainer

from src.models.model_base import ModelBase
from src.utils.common import save_pretrained
from src.utils.dataset_utils import DatasetBase

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Strategy(enum.IntFlag):
    LAYER_WISE_MSE = 2
    LAYER_WISE_COS_SIM = 64
    LOGITS_KL_DIV = 4
    LM_LOSS = 8

    def has_layer_wise(self):
        try:
            self.layer_wise_strategy()
            return True
        except ValueError:
            return False

    def validate(self):
        layer_wise_loss = [
            Strategy.LAYER_WISE_MSE,
            Strategy.LAYER_WISE_COS_SIM,
        ]
        if bin(self & reduce(lambda x, y: x | y, layer_wise_loss)).count("1") > 1:
            raise ValueError(
                f"{[e.name for e in layer_wise_loss]} are mutually exclusive."
            )

    def layer_wise_strategy(self):
        if Strategy.LAYER_WISE_MSE in self:
            return "mse_loss"
        if Strategy.LAYER_WISE_COS_SIM in self:
            return "cos_sim"
        raise ValueError("None of layer wise loss strategy is enabled")


class MimicTrainer(Trainer):
    cfg: DictConfig
    train_lmm: ModelBase

    def __init__(
        self,
        *,
        cfg: DictConfig,
        lmm: ModelBase,
        strategy: Strategy,
        **kwargs: Any,
    ) -> None:
        self.cfg = cfg
        self.train_lmm = lmm
        strategy.validate()
        self.strategy = strategy
        self._loss_components: dict[str, float] = {}
        super().__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = cast(DatasetBase, self.train_dataset)
        return self.accelerator.prepare(
            train_dataset.train_dataloader(
                self.train_lmm,
                self.args.per_device_train_batch_size,
                distributed=self.args.world_size > 1,
            )
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        shift_encoder = self.accelerator.unwrap_model(model)
        prefix_texts = inputs["prefix_texts"]
        query_texts = inputs["query_texts"]
        answers = inputs["answers"]
        images = inputs["images"]
        pad_token, pad_token_id, bos_token_id, eos_token = (
            self.train_lmm.processor.tokenizer.pad_token,
            self.train_lmm.processor.tokenizer.pad_token_id,
            self.train_lmm.processor.tokenizer.bos_token_id,
            self.train_lmm.processor.tokenizer.eos_token,
        )
        device = self._device(model)
        dtype = self._dtype(model)
        hooks = shift_encoder.register_record_hooks()
        try:
            outputs = {"loss": 0.0}
            prefix_logits = None
            prefix_hidden_states = None
            prefix_label_mask = None
            query_answer = [
                query + pad_token + answer + eos_token
                for query, answer in zip(query_texts, answers)
            ]
            query_images = [img[-self.cfg.data.num_image_in_query :] for img in images]
            query_inputs = self.train_lmm.process_input(query_images, query_answer).to(
                device=device,
                dtype=dtype,
            )
            query_inputs["attention_mask"] = query_inputs["input_ids"] != pad_token_id

            if self.strategy != Strategy.LM_LOSS:
                # Prefix forward is the teacher path for alignment losses.
                full_text = [
                    ice + pad_token + query + pad_token + answer + eos_token
                    for ice, query, answer in zip(prefix_texts, query_texts, answers)
                ]
                inputs = self.train_lmm.process_input(images, full_text).to(
                    device=device,
                    dtype=dtype,
                )
                inputs["attention_mask"] = inputs["input_ids"] != pad_token_id
                with torch.no_grad(), self.train_lmm.model.disable_adapter():
                    prefix_logits = self.train_lmm.model(**inputs)["logits"]
                prefix_hidden_states = (
                    self.get_hidden_states(
                        shift_encoder,
                        self.generate_label_mask(inputs, 1, device),
                    )
                    if self.strategy.has_layer_wise()
                    else None
                )
                prefix_label_mask = self.generate_label_mask(inputs, 2, device)

            hooks.update(shift_encoder.register_shift_hooks())
            query_outputs = self.train_lmm.model(
                **query_inputs,
                labels=(
                    query_inputs["input_ids"]
                    if Strategy.LM_LOSS in self.strategy
                    else None
                ),
            )
            shift_logits = query_outputs["logits"]
            if Strategy.LM_LOSS in self.strategy:
                ce_loss_weight = (
                    1.0
                    if self.strategy == Strategy.LM_LOSS
                    else self.cfg.ce_loss_weight
                )
                outputs["ce_loss"] = query_outputs["loss"]
                outputs["loss"] += ce_loss_weight * query_outputs["loss"]

            shift_hidden_states = (
                self.get_hidden_states(
                    shift_encoder,
                    query_inputs["attention_mask"]
                    & (query_inputs["input_ids"] != bos_token_id),
                )
                if self.strategy.has_layer_wise()
                else None
            )

            if self.strategy.has_layer_wise():
                layer_loss = self.calculate_layer_wise_loss(
                    shift_hidden_states,
                    prefix_hidden_states,
                )
                outputs.update(layer_loss)
                outputs["loss"] += self.cfg.align_loss_weight * sum(
                    layer_loss.values()
                )

            if Strategy.LOGITS_KL_DIV in self.strategy:
                logits_kl_loss = self.calculate_logits_kl_loss(
                    shift_logits,
                    prefix_logits,
                    self.generate_label_mask(query_inputs, 1, device),
                    prefix_label_mask,
                )
                outputs.update(logits_kl_loss)
                outputs["loss"] += self.cfg.align_loss_weight * sum(
                    logits_kl_loss.values()
                )
        finally:
            self.remove_hooks(hooks)

        self._loss_components = {
            name: value.detach().float().mean().item()
            for name, value in outputs.items()
            if name != "loss" and torch.is_tensor(value)
        }
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float]) -> None:
        super().log({**logs, **self._loss_components})

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args,
            self.model,
        )
        if "params" in optimizer_kwargs:
            optimizer_kwargs.pop("params")
        if "model" in optimizer_kwargs:
            optimizer_kwargs.pop("model")
        self.optimizer = optimizer_cls(
            self._optimizer_groups(),
            **optimizer_kwargs,
        )
        return self.optimizer

    def _optimizer_groups(self) -> list[dict[str, Any]]:
        model = self.accelerator.unwrap_model(self.model)
        trainable_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError("No parameter to optimize.")

        scale_lr = self.cfg.peft.get("scale_lr", None)
        if scale_lr:
            scale_params = [
                item
                for item in trainable_params
                if "log_Z1" in item[0] or "scale" in item[0]
            ]
            scale_names = {name for name, _ in scale_params}
            regular_params = [
                item for item in trainable_params if item[0] not in scale_names
            ]
            return [
                *self._decay_groups(regular_params),
                *self._decay_groups(scale_params, lr=scale_lr),
            ]
        return self._decay_groups(trainable_params)

    def _decay_groups(
        self,
        named_params: list[tuple[str, torch.nn.Parameter]],
        **common_args: Any,
    ) -> list[dict[str, Any]]:
        optimizer_lr = common_args.pop("lr", self.args.learning_rate)
        no_decay = ("bias",)
        decay_params = [
            param
            for name, param in named_params
            if not any(nd in name for nd in no_decay)
        ]
        nodecay_params = [
            param
            for name, param in named_params
            if any(nd in name for nd in no_decay)
        ]
        return [
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
                "lr": optimizer_lr,
                **common_args,
            },
            {
                "params": decay_params,
                "weight_decay": self.args.weight_decay,
                "lr": optimizer_lr,
                **common_args,
            },
        ]

    def save_model(
        self,
        output_dir: str | None = None,
        _internal_call: bool = False,
    ) -> None:
        output_dir = output_dir or self.args.output_dir
        if not self.is_world_process_zero():
            return
        model = self.accelerator.unwrap_model(self.model)
        save_pretrained(output_dir, self.train_lmm, model)
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(self.cfg, resolve=True), f, indent=4)

    def _device(self, model):
        return next(model.parameters()).device

    def _dtype(self, model):
        return next(model.parameters()).dtype

    def generate_label_mask(
        self,
        inputs,
        num_separator: int,
        device,
        keep_bos: bool = False,
    ):
        input_ids = inputs["input_ids"]
        batch_size, _ = input_ids.shape
        pad_mask = input_ids == self.train_lmm.processor.tokenizer.pad_token_id
        non_pad_mask = ~pad_mask
        label_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.train_lmm.processor.tokenizer.padding_side == "left":
            bos_position = non_pad_mask.long().argmax(dim=1)

        for i in range(batch_size):
            seq_pad_positions = pad_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if self.train_lmm.processor.tokenizer.padding_side == "left":
                seq_pad_positions = seq_pad_positions[
                    seq_pad_positions > bos_position[i]
                ]
            if len(seq_pad_positions) < num_separator:
                raise ValueError(
                    f"Sequence {i} has fewer pad tokens than num_separator={num_separator}"
                )
            sep_position = seq_pad_positions[num_separator - 1].item()
            label_mask[i, sep_position + 1 :] = True

        label_mask = label_mask & non_pad_mask
        if keep_bos:
            label_mask[torch.arange(batch_size, device=device), bos_position] = True
        return label_mask

    def remove_hooks(self, hooks) -> None:
        for handles in hooks.values():
            if isinstance(handles, list):
                for handle in handles:
                    handle.remove()
            else:
                handles.remove()

    def get_hidden_states(self, shift_encoder, query_label_mask):
        hidden_states_dict = {}
        for name, attr in vars(shift_encoder).items():
            if "hidden_states" in name:
                hidden_states = torch.stack(attr).transpose(0, 1)
                _, num_layer, _, d_model = hidden_states.shape
                hidden_states_dict[name] = [
                    hs.masked_select(mask[None, :, None]).view(num_layer, -1, d_model)
                    for hs, mask in zip(hidden_states, query_label_mask)
                ]
        if not hidden_states_dict:
            raise RuntimeError("Layer-wise loss requires recorded hidden states.")
        return hidden_states_dict

    def calculate_layer_wise_loss(self, shift_hidden_states, prefix_hidden_states):
        if Strategy.LAYER_WISE_MSE in self.strategy:
            loss_fn = lambda input, target: F.mse_loss(input, target, reduction="mean")
        elif Strategy.LAYER_WISE_COS_SIM in self.strategy:
            loss_fn = lambda input, target: 1 - torch.mean(
                F.cosine_similarity(input, target, dim=-1),
                dim=1,
            )
        else:
            raise ValueError("No layer-wise loss strategy is enabled.")

        layer_loss = {}
        for (shift_name, shift_hs_list), (_, prefix_hs_list) in zip(
            shift_hidden_states.items(),
            prefix_hidden_states.items(),
        ):
            layer_loss[
                shift_name.replace("hidden_states", self.strategy.layer_wise_strategy())
            ] = torch.mean(
                torch.stack(
                    [
                        loss_fn(shift_hs, prefix_hs)
                        for shift_hs, prefix_hs in zip(shift_hs_list, prefix_hs_list)
                    ]
                )
            )
        return layer_loss

    def calculate_logits_kl_loss(
        self,
        shift_logits,
        prefix_logits,
        query_label_inputs,
        prefix_label_mask,
    ):
        return {
            "logits_kl_loss": F.kl_div(
                shift_logits[query_label_inputs].log_softmax(dim=-1),
                prefix_logits[prefix_label_mask].softmax(dim=-1),
                reduction="batchmean",
                log_target=False,
            )
        }
