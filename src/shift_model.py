import torch
import torch_npu
import enum
from functools import reduce
import json
import os
import sys

from omegaconf import OmegaConf

sys.path.insert(0, "..")
import paths
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_cosine_schedule_with_warmup

from utils import save_pretrained, get_expand_runname


class Strategy(enum.IntFlag):
    LAYER_WISE_MSE = 2
    LAYER_WISE_COS_SIM = 64  # equivalent to normalized L2 distance
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
        elif Strategy.LAYER_WISE_COS_SIM in self:
            return "cos_sim"
        else:
            raise ValueError("None of layer wise loss strategy is enabled")


class ShiftModel(pl.LightningModule):
    def __init__(
        self,
        cfg,
        shift_encoder,
        strategy: Strategy,
        save_checkpoint_when=None,  # should be a method f(epoch), save the last ckpt by default
    ) -> None:
        super().__init__()
        self.lmm = shift_encoder.lmm
        self.cfg = cfg
        self.shift_encoder = shift_encoder
        strategy.validate()
        self.strategy = strategy
        self.save_checkpoint_when = (
            save_checkpoint_when
            if save_checkpoint_when is not None
            else lambda epoch: epoch == self.trainer.max_epochs - 1
        )
        self.save_dir = os.path.join(paths.result_dir, "ckpt", get_expand_runname(cfg))

    def generate_label_mask(self, inputs, num_separator, keep_bos=False):
        """
        Generates label mask which masks tokens before num_separator pad_tokens from given inputs.
        """
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        pad_mask = input_ids == self.lmm.processor.tokenizer.pad_token_id
        non_pad_mask = ~pad_mask
        label_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.lmm.processor.tokenizer.padding_side == "left":
            bos_position = non_pad_mask.long().argmax(dim=1)

        for i in range(batch_size):
            seq_pad_positions = pad_mask[i].nonzero(as_tuple=False).squeeze(-1)

            if self.lmm.processor.tokenizer.padding_side == "left":
                seq_pad_positions = seq_pad_positions[
                    seq_pad_positions > bos_position[i]
                ]

            num_pads = len(seq_pad_positions)
            if num_pads < num_separator:
                raise ValueError(
                    f"Sequence {i} has fewer pad tokens ({num_pads}) than num_separator ({num_separator})"
                )

            sep_position = seq_pad_positions[num_separator - 1].item()
            label_mask[i, sep_position + 1 :] = True

        label_mask = label_mask & non_pad_mask
        if keep_bos:
            label_mask[torch.arange(batch_size, device=self.device), bos_position] = (
                True
            )

        return label_mask

    def remove_hooks(self, hooks):
        # remove all hooks
        for name, handles in hooks.items():
            if isinstance(handles, list):
                for handle in handles:
                    handle.remove()
            else:
                handles.remove()

    def get_hidden_states(self, query_label_mask):
        """
        Apply query_label_mask to extract query parts from hidden states (shape: num_layer * [batch_size, seq_len, d_model]),
        and convert to batch_size * [num_layer, query_part_len, d_model].
        """
        hidden_states_dict = {}

        for name, attr in vars(self.shift_encoder).items():
            if "hidden_states" in name:
                # [num_layer, batch_size, seq_len, d_model] -> [batch_size, num_layer, seq_len, d_model]
                hidden_states = torch.stack(attr).transpose(0, 1)
                batch_size, num_layer, seq_len, d_model = hidden_states.shape
                hidden_states_dict[name] = [
                    hs.masked_select(mask[None, :, None]).view(num_layer, -1, d_model)
                    for hs, mask in zip(hidden_states, query_label_mask)
                ]

        if not hidden_states_dict:
            raise RuntimeError(
                "Layer wise loss requires to record hidden states, but no any *_hidden_states in shift encoder."
            )

        return hidden_states_dict

    def calculate_layer_wise_loss(self, shift_hidden_states, prefix_hidden_states):
        if Strategy.LAYER_WISE_MSE in self.strategy:
            loss_fn = lambda input, target: F.mse_loss(
                input,
                target,
                reduction="mean",
            )
        elif Strategy.LAYER_WISE_COS_SIM in self.strategy:
            loss_fn = lambda input, target: 1 - torch.mean(
                F.cosine_similarity(
                    input,
                    target,
                    dim=-1,
                ),
                dim=1,
            )

        layer_loss = dict()
        for (shift_hs_varname, shift_hs_list), (_, prefix_hs_list) in zip(
            shift_hidden_states.items(), prefix_hidden_states.items()
        ):
            # hs_list: batch_size * [num_layer, query_part_len, d_model]
            layer_loss[
                shift_hs_varname.replace(
                    "hidden_states", self.strategy.layer_wise_strategy()
                )
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
        self, shift_logits, prefix_logits, query_label_inputs, prefix_label_mask
    ):
        # extract answer [EOS]
        logits_kl_loss = F.kl_div(
            shift_logits[query_label_inputs].log_softmax(dim=-1),
            prefix_logits[prefix_label_mask].softmax(dim=-1),
            reduction="batchmean",
            log_target=False,
        )
        return {"logits_kl_loss": logits_kl_loss}

    def forward(self, prefix_texts, query_texts, answers, images):
        pad_token, pad_token_id, bos_token_id, eos_token = (
            self.lmm.processor.tokenizer.pad_token,
            self.lmm.processor.tokenizer.pad_token_id,
            self.lmm.processor.tokenizer.bos_token_id,
            self.lmm.processor.tokenizer.eos_token,
        )
        loss_dict = {"loss": 0.0}
        hooks = self.shift_encoder.register_record_hooks()

        # prepare inputs
        query_answer = [
            query + pad_token + answer + eos_token
            for query, answer in zip(query_texts, answers)
        ]
        query_images = [img[-self.cfg.data.num_image_in_query :] for img in images]
        query_inputs = self.lmm.process_input(query_images, query_answer).to(
            device=self.device, dtype=self.dtype
        )
        query_inputs["attention_mask"] = query_inputs["input_ids"] != pad_token_id
        if self.strategy != Strategy.LM_LOSS:
            # if strategy only has lm_loss, full context forward is no need
            full_text = [
                ice + pad_token + query + pad_token + answer + eos_token
                for ice, query, answer in zip(prefix_texts, query_texts, answers)
            ]
            inputs = self.lmm.process_input(images, full_text).to(
                device=self.device, dtype=self.dtype
            )
            inputs["attention_mask"] = inputs["input_ids"] != pad_token_id

            # step 1. [SOS](implicitly added) ICE [PAD] query [PAD] answer [EOS] forward process
            with torch.no_grad(), self.lmm.model.disable_adapter():
                prefix_logits = self.lmm.model(**inputs)["logits"]

            # extract query + [PAD] + answer + [EOS]
            prefix_hidden_states = (
                self.get_hidden_states(self.generate_label_mask(inputs, 1))
                if self.strategy.has_layer_wise()
                else None
            )
            prefix_label_mask = self.generate_label_mask(inputs, 2)

        # step 2. [SOS](implicitly added) + query + [PAD] + answer [EOS] forward process
        hooks.update(self.shift_encoder.register_shift_hooks())
        query_outputs = self.lmm.model(
            **query_inputs,
            labels=(
                query_inputs["input_ids"] if Strategy.LM_LOSS in self.strategy else None
            ),
        )
        shift_logits = query_outputs["logits"]
        if Strategy.LM_LOSS in self.strategy:
            loss_dict["ce_loss"] = query_outputs["loss"]
            ce_loss_weight = (
                1.0 if self.strategy == Strategy.LM_LOSS else self.cfg.ce_loss_weight
            )
            loss_dict["loss"] += ce_loss_weight * query_outputs["loss"]

        # extract query + answer + [EOS]
        shift_hidden_states = (
            self.get_hidden_states(
                query_inputs["attention_mask"]
                & (query_inputs["input_ids"] != bos_token_id)
            )
            if self.strategy.has_layer_wise()
            else None
        )

        self.remove_hooks(hooks)

        # step 3. calculate kl divergency or MSE of each layer
        if self.strategy.has_layer_wise():
            layer_loss = self.calculate_layer_wise_loss(
                shift_hidden_states, prefix_hidden_states
            )
            loss_dict.update(layer_loss)
            loss_dict["loss"] += self.cfg.align_loss_weight * sum(layer_loss.values())

        # step 4. calculate the last logits kl div
        if Strategy.LOGITS_KL_DIV in self.strategy:
            logits_kl_loss = self.calculate_logits_kl_loss(
                shift_logits,
                prefix_logits,
                self.generate_label_mask(query_inputs, 1),
                prefix_label_mask,
            )
            loss_dict.update(logits_kl_loss)
            loss_dict["loss"] += self.cfg.align_loss_weight * sum(
                logits_kl_loss.values()
            )

        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self.forward(**batch)
        self.log_dict(loss_dict, sync_dist=True, prog_bar=True)

        return loss_dict["loss"]

    def on_train_epoch_end(self):
        if self.save_checkpoint_when(self.current_epoch) and self.global_rank == 0:
            save_pretrained(
                os.path.join(self.save_dir, f"epoch-{self.current_epoch}"),
                self.lmm,
                self.shift_encoder,
            )

    def on_train_end(self):
        if self.global_rank == 0:
            with open(os.path.join(self.save_dir, "config.json"), "w") as f:
                json.dump(OmegaConf.to_container(self.cfg, resolve=True), f, indent=4)

    def configure_optimizers(self):
        def filter_decay_params(param_dict, **common_args):
            """filter parameters for optimizer, separate parameters by adding weight_decay or not"""
            non_decay_names = ["bias"]
            non_decay = [
                {
                    "params": [
                        p
                        for n, p in param_dict.items()
                        for name in non_decay_names
                        if name in n
                    ],
                    "weight_decay": 0.0,
                    **common_args,
                }
            ]

            decay = [
                {
                    "params": [
                        p
                        for n, p in param_dict.items()
                        for name in non_decay_names
                        if name not in n
                    ],
                    "weight_decay": self.cfg.weight_decay,
                    **common_args,
                }
            ]

            return [*non_decay, *decay]

        param_dict = {
            n: p for n, p in self.shift_encoder.named_parameters() if p.requires_grad
        }
        if self.cfg.peft.get("scale_lr", None):
            # if scale_lr is provided, separate scale parameters and regular parameters
            # scale parameters will have a different learning rate, which typically is
            # used for LIVE.
            scale_params = {
                n: p for n, p in param_dict.items() if "log_Z1" in n or "scale" in n
            }
            regular_params = {
                n: p for n, p in param_dict.items() if n not in scale_params
            }

            optim_groups = [
                *filter_decay_params(regular_params, lr=self.cfg.lr),
                *filter_decay_params(scale_params, lr=self.cfg.peft.scale_lr),
            ]
        else:
            optim_groups = filter_decay_params(param_dict, lr=self.cfg.lr)

        assert any(
            group["params"] is not None for group in optim_groups if "params" in group
        ), "No parameter to optimize."

        if "deepspeed" in self.cfg.strategy:
            optimizer = DeepSpeedCPUAdam(
                optim_groups,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                optim_groups,
                weight_decay=self.cfg.weight_decay,
            )

        step_batches = self.trainer.estimated_stepping_batches
        warmup_steps = self.cfg.warmup_step
        if isinstance(warmup_steps, float):
            warm_steps = warmup_steps * step_batches
        elif isinstance(warmup_steps, int):
            warm_steps = warmup_steps
        else:
            raise ValueError(
                f"the warm_steps should be int or float, but got {type(warmup_steps)}"
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_steps, num_training_steps=step_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
