from __future__ import annotations

import re
import math
from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .llama import LlamaModel
from .base import ModelBase, TextModel, gguf


@ModelBase.register("LagunaForCausalLM")
class LagunaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LAGUNA

    def set_gguf_parameters(self):
        hparams = self.hparams

        # Handle rope_parameters: can be nested (full_attention / sliding_attention) or flat
        rope_params = hparams.get("rope_parameters", {})
        if "full_attention" in rope_params or "sliding_attention" in rope_params:
            full_rope = rope_params.get("full_attention", {})
            swa_rope = rope_params.get("sliding_attention", {})
            self.rope_parameters["full_attention"] = full_rope
            self.rope_parameters["sliding_attention"] = swa_rope
        else:
            self.rope_parameters["full_attention"] = rope_params

        super().set_gguf_parameters()

        n_layers = hparams["num_hidden_layers"]
        n_head_base = hparams["num_attention_heads"]
        n_kv_base = hparams.get("num_key_value_heads", n_head_base)

        layer_types = hparams.get("layer_types", [])
        partial_rotary_factors = hparams.get("partial_rotary_factors", [])
        attn_other = hparams.get("attention_other_setting", {})

        n_head_swa = attn_other.get("num_attention_heads", n_head_base)
        n_kv_swa = attn_other.get("num_key_value_heads", attn_other.get("num_attention_groups", n_kv_base))

        layer_types = layer_types[:n_layers]
        partial_rotary_factors = partial_rotary_factors[:n_layers]

        # Build per-layer head counts and SWA pattern
        head_arr = []
        kv_arr = []
        swa_pat = []
        for lt in layer_types:
            if lt == "sliding_attention":
                head_arr.append(n_head_swa)
                kv_arr.append(n_kv_swa)
                swa_pat.append(True)
            else:
                head_arr.append(n_head_base)
                kv_arr.append(n_kv_base)
                swa_pat.append(False)

        self.gguf_writer.add_head_count(head_arr)
        self.gguf_writer.add_head_count_kv(kv_arr)

        # SWA window
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(swa_pat)

        # Per-layer value length (head_dim)
        head_dim = hparams.get("head_dim", hparams["hidden_size"] // n_head_base)
        self.gguf_writer.add_value_length(head_dim)

        # MoE params
        n_experts = hparams["moe_num_experts"]
        n_experts_used = hparams["moe_top_k"]
        self.gguf_writer.add_expert_count(n_experts)
        self.gguf_writer.add_expert_used_count(n_experts_used)
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])

        if (shared_dim := hparams.get("share_expert_dim")) is not None:
            self.gguf_writer.add_expert_shared_feed_forward_length(shared_dim)
        if (shared_count := hparams.get("moe_shared_expert", 0)) != 0:
            self.gguf_writer.add_expert_shared_count(shared_count)

        if (routing_scale := hparams.get("moe_routed_scaling_factor")) is not None:
            self.gguf_writer.add_expert_weights_scale(routing_scale)

        # Dense lead layers
        leading_dense = hparams.get("n_layer_dense_lead", 1)
        self.gguf_writer.add_leading_dense_block_count(leading_dense)

        # RMS norm eps
        self.gguf_writer.add_layer_norm_rms_eps(hparams.get("rms_norm_eps", 1e-5))

        # Hard-fail if moe_apply_router_weight_on_input is True
        if hparams.get("moe_apply_router_weight_on_input", False):
            raise ValueError("moe_apply_router_weight_on_input=True is not supported by llama.cpp for Laguna")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Stack split-per-expert tensors
        # HF format: model.layers.{bid}.mlp.experts.{i}.{gate,up,down}_proj.weight
        # GGUF expects: model.layers.{bid}.ffn_{gate,up,down}_exps.weight (stacked)
        if bid is not None:
            m = re.match(r"model\.layers\.\d+\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", name)
            if m is not None:
                # Let the base class handle stacking via modify_tensors
                name = name.replace("mlp.experts.", "mlp.experts.")

        # Map e_score_correction_bias -> exp_probs_b
        if name.endswith(".mlp.experts.e_score_correction_bias"):
            name = name.replace(".mlp.experts.e_score_correction_bias", ".ffn.exp_probs_b.bias")

        # Map shared_expert tensors
        if ".mlp.shared_expert." in name:
            name = name.replace(".mlp.shared_expert.gate_proj.", ".ffn.gate_shexp.")
            name = name.replace(".mlp.shared_expert.up_proj.", ".ffn.up_shexp.")
            name = name.replace(".mlp.shared_expert.down_proj.", ".ffn.down_shexp.")

        # Squeeze expert gate tensors
        if name.endswith((".self_attn.g_proj.weight",)):
            data_torch = data_torch.squeeze().contiguous()

        yield from super().modify_tensors(data_torch, name, bid)
