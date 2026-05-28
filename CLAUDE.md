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
- **Step 2 — YaRN per-layer-type plumbing (Path A)**: Added SWA YaRN sibling fields to `llama_hparams`/`llama_cparams`; initialized in `llama-context.cpp`; per-layer selection in `laguna.cpp` graph builder via `is_swa(il)`.
- **Step 3 — Python converter**: `conversion/laguna.py` written. Handles per-layer head counts, MoE config, SWA pattern, YaRN params, split-per-expert tensors. Also writes correct chat template (resolves HF `{% include %}` indirection) and EOT token ID.
- **Step 4 — C++ model class**: `src/models/laguna.cpp` fully implemented with `load_arch_hparams`, `load_arch_tensors`, `build_arch_graph` (dense lead branching, attention gate, MoE with shared expert).
- **Step 1 (bonus) — Model saver**: Added laguna to unsupported arch list in `llama-model-saver.cpp` (same as step35) since roundtrip serialization loses SWA pattern keys.
- **Step 6 — SWA + long context**: 10K prompt (SWA) clean; 35K needle-in-haystack ("SWORDFISH99") correctly retrieved, confirming YaRN global layers work.

### Remaining

- **Step 5 — Numerical validation**: Compare against HF Transformers. Requires CUDA/ROCm — not feasible on current hardware (AMD RX 7900 XT, no ROCm setup). Defer to a CUDA machine.
- **Step 7 — Quantization top-1 check**: Run f16 GGUF CPU-only (mmap) and compare top-1 tokens against Q4_K_M on GPU for a short prompt. Feasible but slow (f16 = 67 GB, exceeds 60 GB RAM; MoE means actual resident pages are manageable for a few tokens).

### Key learnings from implementation

1. **`test-llama-archs` roundtrip test**: The test serializes and reloads models via `llama_model_saver`. If `llama_model_saver_supports_arch()` returns true for your arch, the roundtrip will run. The saver doesn't preserve all custom hparams (e.g., `sliding_window_pattern`), so either add the arch to the unsupported list or make those hparams optional with `false` in `get_key_or_arr`.
2. **`GGML_ABORT` in graph builder**: Using `GGML_ABORT` in any model method kills the entire test process. The `load_arch_hparams` and `load_arch_tensors` must work correctly for the test to pass.
3. **`swa_layers` / `sliding_window_pattern`**: Made optional (`false` parameter) in laguna's `load_arch_hparams` because the model saver doesn't write it back. The test GGUF context provides it on first load but the roundtrip GGUF doesn't.
4. **Attention gate is softplus, not sigmoid**: HF code uses `F.softplus(g)`, not `torch.sigmoid(g)`. The initial implementation used `ggml_sigmoid`; fixed to `ggml_softplus` (1-op in ggml).
5. **`expert_weights_norm` must be True**: Laguna normalizes routing weights before applying `moe_routed_scaling_factor`. Without this, MoE output magnitudes are wrong.
6. **Chat template HF indirection**: `tokenizer_config.json` stores `{% include 'chat_template.jinja' %}` — SpecialVocab writes this verbatim. Override `set_vocab` to read and write `chat_template.jinja` directly.
7. **EOT token from `eos_token_id` list**: `config.json` has `eos_token_id: [2, 24]`. SpecialVocab only registers index 0 as EOS; token 24 (`</assistant>`) must be written as EOT via `add_eot_token_id` so it gets added to `special_eog_ids`. Without this the model loops past turn boundaries.
8. **ABI mismatch after cparams change**: Adding fields to `llama_cparams` (Step 2) requires a full rebuild of all binaries that link against `libllama-common`. A partial rebuild will crash with unexpected behavior.

---

## Architectural Facts

| Property | Value |
|---|---|
| Layers | 40 (10 global + 30 SWA, 3:1 pattern) |
| Attention heads | 48 SWA / 64 global, via `num_attention_heads_per_layer` |
| KV heads | 8 (GQA) |
| SWA window | 512 tokens |
| Q/K norm | RMSNorm |
| Attention gate | `self_attn.g_proj`, head-wise **softplus**, SWA layers only, applied to SDPA output before `o_proj` |
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
- `build_arch_graph`: RMS norm, Q/K/V projections with per-head norms, partial rotary RoPE, head-wise softplus attention gate (SWA only), dense/MoE FFN branching, shared expert MLP.

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
11. **`src/llama-hparams.h`, `src/llama-cparams.h`, `src/llama-context.cpp`** ✅ — Path A YaRN siblings added: `yarn_ext_factor_swa`, `yarn_attn_factor_swa`, `yarn_beta_fast_swa`, `yarn_beta_slow_swa`.

---

## Reference Output
Greedy decoding on `"The capital of France is"` → `" Paris.\nThe capital of Germany is"` (from sgl-project empirical reproducer).

## Open Items
None. `moe_router_logit_softcapping` is absent from `poolside/Laguna-XS.2` config.json — no plumbing needed.
