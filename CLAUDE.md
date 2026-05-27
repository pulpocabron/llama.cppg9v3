IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

---

# Task: Add Laguna (Poolside) Support to llama.cpp

**Goal:** Implement `LLM_ARCH_LAGUNA` targeting `poolside/Laguna-XS.2` — a 33B-A3B MoE model with 40 layers, 256 routed + 1 shared expert, head-wise sigmoid attention gate, per-layer head counts, dual RoPE (YaRN for global, default for SWA), and per-layer-type partial rotary.

**Base commit:** `b4c0549` (2026-05-27)

**Primary template:** `src/models/step35.cpp` — Laguna reuses almost everything from Step3.5. Start there; do not start from scratch.

---

## Implementation Status

### Completed

- **Step 1 — Plumbing** (compile + test-llama-archs): All framework files edited. `test-llama-archs` passes with OK for laguna MoE.
- **Step 3 — Python converter**: `conversion/laguna.py` written. Handles per-layer head counts, MoE config, SWA pattern, YaRN params, split-per-expert tensors.
- **Step 4 — C++ model class**: `src/models/laguna.cpp` fully implemented with `load_arch_hparams`, `load_arch_tensors`, `build_arch_graph` (dense lead branching, attention gate, MoE with shared expert).
- **Step 1 (bonus) — Model saver**: Added laguna to unsupported arch list in `llama-model-saver.cpp` (same as step35) since roundtrip serialization loses SWA pattern keys.

### Remaining

- **Step 2 — YaRN per-layer-type plumbing (Path A)**: The graph builder currently uses the same YaRN params for all layers. Global layers need YaRN (ext_factor, attn_factor, beta_fast, beta_slow) while SWA layers should use defaults (no YaRN). This requires either adding SWA-specific YaRN fields to cparams or selecting params per-layer in the graph builder based on `hparams.is_swa(il)`. Without this, numerical output will be incorrect for global layers at long contexts.
- **Step 5 — Numerical validation**: Compare against HF Transformers. Requires the actual model weights.
- **Step 6 — SWA + long context**: Test prompts > 8K (SWA) and > 32K (YaRN global layers).
- **Step 7 — Quantization**: `llama-quantize` to Q4_K_M. Verify top-1 match against f16.

### Key learnings from implementation

1. **`test-llama-archs` roundtrip test**: The test serializes and reloads models via `llama_model_saver`. If `llama_model_saver_supports_arch()` returns true for your arch, the roundtrip will run. The saver doesn't preserve all custom hparams (e.g., `sliding_window_pattern`), so either add the arch to the unsupported list or make those hparams optional with `false` in `get_key_or_arr`.
2. **`GGML_ABORT` in graph builder**: Using `GGML_ABORT` in any model method kills the entire test process. The `load_arch_hparams` and `load_arch_tensors` must work correctly for the test to pass.
3. **`swa_layers` / `sliding_window_pattern`**: Made optional (`false` parameter) in laguna's `load_arch_hparams` because the model saver doesn't write it back. The test GGUF context provides it on first load but the roundtrip GGUF doesn't.

---

## Architectural Facts

| Property | Value |
|---|---|
| Layers | 40 (10 global + 30 SWA, 3:1 pattern) |
| Attention heads | 48 SWA / 64 global, via `num_attention_heads_per_layer` |
| KV heads | 8 (GQA) |
| SWA window | 512 tokens |
| Q/K norm | RMSNorm |
| Attention gate | `self_attn.g_proj`, head-wise sigmoid, SWA layers only, applied to SDPA output before `o_proj` |
| MoE router | Sigmoid + `e_score_correction_bias` added at selection time only |
| Experts | 256 routed (top-8) + 1 shared |
| Dense layers | Layer 0 only (`n_layer_dense_lead = 1`) |
| Routed scaling | `moe_routed_scaling_factor` → `expert_weights_scale` |
| Global RoPE | YaRN: base 500K, factor 32, original_max 4096, β_fast 64, β_slow 1, partial_rotary 0.5 |
| SWA RoPE | default: base 10K, full rotary |
| Context length | 131,072 |
| Checkpoint layout | split-per-expert, singular `shared_expert`, `e_score_correction_bias` under `experts` not `gate` |

---

## Key Design Decisions

**YaRN per-layer approach:** Use **Path A** — add per-layer-type YaRN siblings to hparams/cparams (`yarn_ext_factor_swa`, `yarn_attn_factor_swa`, `yarn_beta_fast_swa`, `yarn_beta_slow_swa`). Initialize them to non-SWA values so other architectures are unaffected. In the graph builder, select via `hparams.is_swa(il)`.

**Partial rotary:** Keep step35's `n_rot_full / 2` mechanism — it exactly matches Laguna's pattern (global layers half-rotated, SWA layers full-rotated).

**Dense lead:** Layer 0 is dense FFN; layers 1–39 are MoE. Branch on `static_cast<uint32_t>(i) < hparams.n_layer_dense_lead`.

**Attention gate:** `TENSOR_NOT_REQUIRED` on `wqkv_gate`; guard with `if (model.layers[il].wqkv_gate)` in the graph builder.

---

## Files Added (2)

### `src/models/laguna.cpp` ✅
Full implementation based on step35.cpp with:
- `load_arch_hparams`: reads RMS eps, SWA type, MoE params, dense lead, expert shared count, SWA pattern. Maps 40 layers to `LLM_TYPE_33B_A3B`.
- `load_arch_tensors`: per-layer Q/K/V with variable head counts, optional attention gate, dense MLP tensors, MoE routed expert tensors, shared expert tensors.
- `build_arch_graph`: RMS norm, Q/K/V projections with per-head norms, partial rotary RoPE, head-wise sigmoid attention gate, dense/MoE FFN branching, shared expert MLP.

### `conversion/laguna.py` ✅
Subclass `TextModel`; registered as `"LagunaForCausalLM"`. Overrides `set_gguf_parameters` and `modify_tensors`. Handles:
- Per-layer head counts, MoE config, dense-lead count, SWA pattern mask.
- Both nested and flat rope config forms.
- Hard-fail if `moe_apply_router_weight_on_input=True`.

---

## Files Modified (11)

1. **`src/llama-arch.h`** ✅ — Added `LLM_ARCH_LAGUNA` to enum.
2. **`src/llama-arch.cpp`** ✅ — Added name map entry `{ LLM_ARCH_LAGUNA, "laguna" }`.
3. **`src/llama-model.cpp`** ✅ — Factory switch, RoPE-type switch, type name.
4. **`src/llama-model.h`** ✅ — Added `LLM_TYPE_33B_A3B` to `llm_type` enum.
5. **`src/models/models.h`** ✅ — Added `llama_model_laguna` struct.
6. **`gguf-py/gguf/constants.py`** ✅ — Added `LAGUNA = auto()`, name entry, tensor list (copy of STEP35).
7. **`gguf-py/gguf/tensor_mapping.py`** ✅ — Added `e_score_correction_bias` mapping.
8. **`conversion/__init__.py`** ✅ — Added `"LagunaForCausalLM": "laguna"`.
9. **`tests/test-llama-archs.cpp`** ✅ — Added laguna to `moe_mandatory()` and SWA 3:1 pattern.
10. **`src/llama-model-saver.cpp`** ✅ — Added laguna to unsupported arch list.
11. **`src/llama-hparams.{h,cpp}`, `src/llama-cparams.h`, `src/llama-context.cpp`** — Path A YaRN siblings not yet added. Deferred until numerical validation phase.

---

## Reference Output
Greedy decoding on `"The capital of France is"` → `" Paris.\nThe capital of Germany is"` (from sgl-project empirical reproducer).

## Open Item
`moe_router_logit_softcapping`: if non-zero in config.json, plumb through `LLM_KV_EXPERT_GATING_LOGIT_SOFTCAP` to `build_moe_ffn`. If zero, defer to follow-up.
