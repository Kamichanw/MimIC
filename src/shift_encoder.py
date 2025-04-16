import torch
import torch_npu
import enum
from functools import partial
from typing import List, Callable, Dict, Optional, Tuple, Union
import torch
from torch import nn
import re
import torch.utils


class HookType(enum.Enum):
    TEXT_MODEL_LAYER = enum.auto()
    VISION_MODEL_LAYER = enum.auto()


class ShiftStrategy(enum.IntFlag):
    VECTOR_SHIFT = 1
    RECORD_HIDDEN_STATES = 4
    LEARNABLE_SHIFT_SCALE = 8
    MULTI_HEAD = 16


class BaseHookEncoder(nn.Module):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy(0),
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
    ):
        super().__init__()
        self.attn_strategy = (
            eval(attn_strategy)
            if attn_strategy and eval(attn_strategy)
            else ShiftStrategy(0)
        )
        self.ffn_strategy = (
            eval(ffn_strategy)
            if ffn_strategy and eval(ffn_strategy)
            else ShiftStrategy(0)
        )
        self.lmm = lmm

        if "idefics-9b" in self.lmm.model_name:
            self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                lmm.model.config.hidden_size,
                lmm.model.config.num_hidden_layers,
                lmm.model.config.num_attention_heads,
            )
        elif "idefics2-8b" in self.lmm.model_name:
            self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                lmm.model.config.text_config.hidden_size,
                lmm.model.config.text_config.num_hidden_layers,
                lmm.model.config.text_config.num_attention_heads,
            )
        elif "llava-interleave" in self.lmm.model_name:
            self.lmm_hidden_dim, self.lmm_layers, self.lmm_num_head = (
                lmm.model.config.text_config.hidden_size,
                lmm.model.config.text_config.num_hidden_layers,
                lmm.model.config.text_config.num_attention_heads,
            )
        else:
            raise ValueError(f"{self.lmm.model_name} is not supported")

        def parse_strategy(prefix, strategy):
            if ShiftStrategy.RECORD_HIDDEN_STATES in getattr(
                self, f"{prefix}_strategy"
            ):
                setattr(
                    self,
                    f"{prefix}_hidden_states",
                    [[] for _ in range(self.lmm_layers)],
                )

            if ShiftStrategy.LEARNABLE_SHIFT_SCALE in strategy and (
                ShiftStrategy.VECTOR_SHIFT not in strategy
            ):
                raise ValueError(
                    "ShiftStrategy.LEARNABLE_SHIFT_SCALE should be used with ShiftStrategy.USE_VECTOR_SHIFT"
                )

        parse_strategy("attn", self.attn_strategy)
        parse_strategy("ffn", self.ffn_strategy)

    def register_hooks(
        self,
        register_fn_name: str,
        targets: List[Union[str, HookType]],
        hooks: Dict[str, Callable],
    ):
        return {
            name: getattr(self.lmm, register_fn_name)(target, hook_fn)
            for target, (name, hook_fn) in zip(targets, hooks.items())
            if hook_fn is not None
        }

    @property
    def decoder_mlp_name(self) -> str:
        if "idefics-9b" in self.lmm.model_name:
            return r"model\.layers\.\d+\.mlp$"
        elif "idefics2-8b" in self.lmm.model_name:
            return r"model\.text_model\.layers\.\d+\.mlp$"
        elif "llava-interleave" in self.lmm.model_name:
            return r"language_model\.model\.layers\.\d+\.mlp$"

    @property
    def decoder_self_attn_name(self) -> str:
        if "idefics-9b" in self.lmm.model_name:
            return r"model\.layers\.\d+\.self_attn$"
        elif "idefics2-8b" in self.lmm.model_name:
            return r"model\.text_model\.layers\.\d+\.self_attn$"
        elif "llava-interleave" in self.lmm.model_name:
            return r"language_model\.model\.layers\.\d+\.self_attn$"

    def register_record_hooks(self, **kwargs):
        # NOTE: record hooks should be registered AFTER all hooks
        def record_hook(m, inputs, outputs, module_name, record_varname, **kwargs):
            layer_idx = int(re.findall(r"\d+", module_name)[0])
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            hidden_states, *_ = outputs
            getattr(self, record_varname)[layer_idx] = hidden_states

        return self.register_hooks(
            "register_forward_hook",
            [
                self.decoder_self_attn_name,
                self.decoder_mlp_name,
            ],
            {
                "attn_record_hook": (
                    partial(record_hook, record_varname="attn_hidden_states")
                    if hasattr(self, "attn_hidden_states")
                    else None
                ),
                "ffn_record_hook": (
                    partial(record_hook, record_varname="ffn_hidden_states")
                    if hasattr(self, "ffn_hidden_states")
                    else None
                ),
            },
        )


class AttnFFNShift(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy(0),
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
        shift_scale_init_value=None,
    ):
        """
        Add shift to attention or ffn output. It can also capture hidden states for each layer
        to calculate the layer-wise alignment loss.

        Args:
            lmm: the model to apply shift.
            attn_strategy: the strategy for attention shift.
            ffn_strategy: the strategy for ffn shift.
            shift_scale_init_value: the initial value for the learnable shift scale.
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)

        def parse_strategy(prefix, strategy):
            """
            Create shift modules to ffn output or attention output, based on the strategy.
            """
            if ShiftStrategy.MULTI_HEAD in strategy:
                raise ValueError(
                    f" ShiftStrategy.MULTI_HEAD is not supported, since shift is inserted after {prefix} output"
                )
            if ShiftStrategy.VECTOR_SHIFT in strategy:
                setattr(
                    self,
                    f"{prefix}_shift",
                    torch.nn.Parameter(
                        torch.empty(self.lmm_layers, self.lmm_hidden_dim).normal_(
                            mean=0.0, std=0.01
                        )
                    ),
                )

                if ShiftStrategy.LEARNABLE_SHIFT_SCALE in strategy:
                    setattr(
                        self,
                        f"{prefix}_shift_scale",
                        nn.Parameter(
                            torch.full(
                                [self.lmm_layers],
                                (
                                    shift_scale_init_value
                                    if shift_scale_init_value
                                    else 1.0
                                ),
                            )
                        ),
                    )
                else:
                    self.register_buffer(
                        f"{prefix}_shift_scale", torch.ones(self.lmm_layers)
                    )

        parse_strategy("attn", self.attn_strategy)
        parse_strategy("ffn", self.ffn_strategy)

    def register_shift_hooks(self, **kwargs):
        return self.register_hooks(
            "register_forward_hook",
            [
                self.decoder_self_attn_name,
                self.decoder_mlp_name,
            ],
            {
                "attn_hook": (
                    self._shift_hook("attn") if hasattr(self, "attn_shift") else None
                ),
                "ffn_hook": (
                    self._shift_hook("ffn") if hasattr(self, "ffn_shift") else None
                ),
            },
        )

    def _shift_hook(self, prefix):
        def hook(m, inputs, outputs, module_name, **kwargs):
            layer_idx = int(re.findall(r"\d+", module_name)[0])
            shift = getattr(self, f"{prefix}_shift", None)
            shift_scale = getattr(self, f"{prefix}_shift_scale", None)

            if isinstance(outputs, tuple):
                hidden_states, *rest = outputs
            else:
                hidden_states = outputs

            if shift is not None:
                shift = shift[layer_idx][None, None, :]
                shifted_states = hidden_states + shift_scale[layer_idx] * shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

            if isinstance(outputs, tuple):
                return (hidden_states, *rest)
            else:
                return hidden_states

        return hook


# Copied from transformers.models.idefics.modeling_idefics.IdeficsSelfAttention
def idefics_attn_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states=None,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    module_name=None,
    shift_encoder=None,
):
    # if key_value_states are provided this layer is used as a cross-attention layer
    is_cross_attention = self.is_cross_attention or key_value_states is not None
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    if not is_cross_attention:
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
    else:
        _, kv_len, _ = (
            key_value_states.size()
        )  # Note that, in this case, `kv_len` == `kv_seq_len`
        key_states = (
            self.k_proj(key_value_states)
            .view(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(key_value_states)
            .view(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if isinstance(past_key_value, tuple):
            kv_seq_len += past_key_value[0].shape[-2]
        else:
            kv_seq_len += cache_position[0]

    if not is_cross_attention:
        from transformers.models.idefics.modeling_idefics import apply_rotary_pos_emb

        cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, q_len))
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        if isinstance(past_key_value, tuple):
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            past_key_value = (key_states, value_states) if use_cache else None
        else:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    if self.qk_layer_norms:
        query_states = self.q_layer_norm(query_states)
        key_states = self.k_layer_norm(key_states)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "npu" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = (
        True if self.is_causal and attention_mask is None and q_len > 1 else False
    )

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)

    # ------------------------- The following part is newly added ---------------------
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    attn_output = shift_encoder.do_shift(
        layer_idx, query_states, key_states, attn_output
    )
    # ---------------------------------------------------------------------------------

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    attn_weights = None

    return attn_output, attn_weights, past_key_value


# Copied from transformers.models.mistral.modeling_mistral.MistralSpdaAttention
# The latest version of MistralSpdaAttention is not available in the transformers>=4.46 (not tested)
def idefics2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    module_name=None,
    shift_encoder=None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)

    from transformers.models.mistral.modeling_mistral import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "npu" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()

    # ------------------------- The following part is newly added ---------------------
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    attn_output = shift_encoder.do_shift(
        layer_idx, query_states, key_states, attn_output
    )
    # ---------------------------------------------------------------------------------

    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention
def llava_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    module_name=None,
    shift_encoder=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings

    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        eager_attention_forward,
        ALL_ATTENTION_FUNCTIONS,
    )

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    # ------------------------- The following part is newly added ---------------------
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    attn_output = shift_encoder.do_shift(
        layer_idx, query_states, key_states, attn_output
    )
    # ---------------------------------------------------------------------------------

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


class MultiheadLinear(nn.Module):
    def __init__(self, lmm_num_head, lmm_hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(lmm_num_head, lmm_hidden_dim // lmm_num_head).normal_(0, 0.02)
        )
        self.bias = nn.Parameter(torch.zeros([lmm_num_head]))

    def forward(self, x):
        return torch.einsum("btnd,nd->btn", x, self.weight) + self.bias


class MultiheadProjection(nn.Module):
    def __init__(self, lmm_num_head, lmm_hidden_dim):
        super().__init__()
        head_dim = lmm_hidden_dim // lmm_num_head
        self.weight = nn.Parameter(
            torch.empty(lmm_num_head, head_dim, head_dim).normal_(0, 0.02)
        )
        self.bias = nn.Parameter(torch.zeros([lmm_num_head, head_dim]))

    def forward(self, x):
        return torch.einsum("btnd,ndd->btnd", x, self.weight) + self.bias


class AttnApproxHandle:
    def __init__(self, active=False):
        self.active = active

    def remove(self):
        self.active = False


class AttnApproximator(BaseHookEncoder):
    def __init__(
        self,
        lmm,
        attn_strategy: ShiftStrategy = ShiftStrategy.VECTOR_SHIFT,
        ffn_strategy: ShiftStrategy = ShiftStrategy(0),
    ):
        """
        The implementation of MimIC attention heads. It train learnable shifts and magnitudes for each layer
        to approximate the in-context demonstrations affected terms (Section 3.2).

        Args:
            lmm: the model to apply shift.
            attn_strategy: the strategy for attention shift.
            ffn_strategy: the strategy for ffn shift.
        """
        super().__init__(lmm, attn_strategy, ffn_strategy)
        self.attn_shift_handles = [AttnApproxHandle() for _ in range(self.lmm_layers)]

        if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
            self.log_Z1_lin = nn.ModuleList(
                (
                    MultiheadLinear(self.lmm_num_head, self.lmm_hidden_dim)
                    if ShiftStrategy.MULTI_HEAD in self.attn_strategy
                    else nn.Linear(self.lmm_hidden_dim, 1)
                )
                for _ in range(self.lmm_layers)
            )

        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            self.attn_shift = nn.Parameter(
                torch.randn(
                    [self.lmm_layers]
                    + (
                        [self.lmm_num_head, self.lmm_hidden_dim // self.lmm_num_head]
                        if ShiftStrategy.MULTI_HEAD in self.attn_strategy
                        else [self.lmm_hidden_dim]
                    )
                )
                * 0.001
            )

        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:
            self.ffn_shift = nn.Parameter(
                torch.randn([self.lmm_layers, self.lmm_hidden_dim]) * 0.001
            )

    def register_shift_hooks(self, **kwargs):
        if ShiftStrategy.VECTOR_SHIFT in self.attn_strategy:
            if not hasattr(self, "attn_forward_replaced"):
                if self.lmm.model_name == "idefics-9b":
                    new_attn_foward = idefics_attn_forward
                elif "idefics2-8b" in self.lmm.model_name:
                    new_attn_foward = idefics2_attn_forward
                elif "llava-interleave" in self.lmm.model_name:
                    new_attn_foward = llava_attn_forward
                else:
                    raise ValueError(f"{self.lmm.model_name} is not supported")

                self.lmm.replace_module_method(
                    self.decoder_self_attn_name,
                    "forward",
                    partial(new_attn_foward, shift_encoder=self),
                    strict=False,
                )
                setattr(self, "attn_forward_replaced", True)

            for handle in self.attn_shift_handles:
                handle.active = True

        registered_hooks = {"attn_hook": self.attn_shift_handles}
        if ShiftStrategy.VECTOR_SHIFT in self.ffn_strategy:

            def hook(m, inputs, outputs, module_name, **kwargs):
                layer_idx = int(re.findall(r"\d+", module_name)[0])

                if isinstance(outputs, tuple):
                    hidden_states, *rest = outputs
                else:
                    hidden_states = outputs

                shift = self.ffn_shift[layer_idx][None, None, :]
                shifted_states = hidden_states + shift
                hidden_states = (
                    shifted_states
                    / shifted_states.norm(dim=-1, keepdim=True)
                    * hidden_states.norm(dim=-1, keepdim=True)
                )

                if isinstance(outputs, tuple):
                    return (hidden_states, *rest)
                else:
                    return hidden_states

            registered_hooks.update(
                self.register_hooks(
                    "register_forward_hook", [self.decoder_mlp_name], {"ffn_hook": hook}
                )
            )

        return registered_hooks

    def do_shift(self, layer_idx, query_states, key_states, attn_output):
        head_dim = self.lmm_hidden_dim // self.lmm_num_head
        bsz, nh, t, nd = query_states.shape
        if self.attn_shift_handles[layer_idx].active:
            # [bsz, nh, t, hd] -> [bsz, t, nh, nd]
            query_states_transposed = query_states.transpose(1, 2)

            if ShiftStrategy.MULTI_HEAD not in self.attn_strategy:
                # [bsz, t, nh, nd] -> [bsz, t, nh * nd]
                query_states_transposed = query_states_transposed.reshape(bsz, t, -1)
                attn_output = attn_output.reshape(bsz, t, -1)

            if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
                # Z1 = \sum{ \exp(x_i X^\top) }
                # calculate Z2 = \sum{ \exp(x_i * \hat{x}^\top) }
                log_Z2 = torch.logsumexp(
                    torch.matmul(query_states, key_states.transpose(-2, -1))
                    / (head_dim**0.5),
                    dim=-1,  # [bsz, nh, t, hd] * [bsz, nh, hd, t] -> [bsz, nh, t, t] -> [bsz, nh, t]
                ).transpose(
                    -2, -1
                )  # [bsz, nh, t] -> [bsz, t, nh]

                if ShiftStrategy.MULTI_HEAD not in self.attn_strategy:
                    # [bsz, t, nh] -> [bsz, t, 1]
                    log_Z2 = log_Z2.mean(-1, keepdim=True)

                log_Z1 = self.log_Z1_lin[layer_idx](query_states_transposed)

                # shape: [bsz, t, nh] or [bsz, t, 1]
                mu = torch.exp(log_Z1 - torch.logaddexp(log_Z1, log_Z2))
                if ShiftStrategy.MULTI_HEAD in self.attn_strategy:
                    # shape: [bsz, t, nh] -> [bsz, t, nh, 1]
                    mu = mu.unsqueeze(-1)

            if hasattr(self, "attn_shift"):
                # shape: [1, 1, nh, nd] or [1, 1, hd * nh]
                shift = self.attn_shift[layer_idx][None, None, :]
                if self.training and hasattr(self, "attn_proj"):
                    shift = self.attn_proj[layer_idx](shift)
                if ShiftStrategy.LEARNABLE_SHIFT_SCALE in self.attn_strategy:
                    # shift := SA(q, K_D, V_D) - SA(q, K, V)
                    return attn_output + mu * shift
            else:
                # never fall in here
                shift = torch.zeros_like(attn_output)

            attn_output = attn_output + shift

        # attn_output: [bsz, t, nh, nd]
        return attn_output