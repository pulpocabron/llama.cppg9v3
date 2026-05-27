IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

---

# Task: Add Laguna (Poolside) Support to llama.cpp

**Goal:** Implement `LLM_ARCH_LAGUNA` targeting `poolside/Laguna-XS.2` — a 33B-A3B MoE model with 40 layers, 256 routed + 1 shared expert, head-wise sigmoid attention gate, per-layer head counts, dual RoPE (YaRN for global, default for SWA), and per-layer-type partial rotary.

**Base commit:** `b4c0549` (2026-05-27)

**Primary template:** `src/models/step35.cpp` — Laguna reuses almost everything from Step3.5. Start there; do not start from scratch.

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

## Files to Add (2)

### `src/models/laguna.cpp`
Copy `src/models/step35.cpp` verbatim, then:
1. Replace all `step35`/`STEP35` with `laguna`/`LAGUNA`.
2. In `load_arch_hparams`: keep `n_rot_full / 2` and `swa_layers`; add reads for `LLM_KV_LEADING_DENSE_BLOCK_COUNT`, `LLM_KV_EXPERT_SHARED_COUNT`, `LLM_KV_EXPERT_WEIGHTS_SCALE`; add per-layer-type YaRN KV reads (Path A); add `case 40: type = LLM_TYPE_33B_A3B; break;`.
3. In `load_arch_tensors`: replace step35's FFN block with dense/MoE branch (see plan §"Files to add").
4. In `build_arch_graph`: keep step35's head-wise sigmoid gate block verbatim; pass per-layer-type YaRN params via `hparams.is_swa(il)`; replace FFN block with `build_moe_ffn(…, LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, …)` plus shared-expert via `build_ffn` guarded by `n_expert_shared > 0`.

### `conversion/laguna.py`
Subclass `LlamaModel`; register as `"LagunaForCausalLM"`. Override `set_gguf_parameters` and `modify_tensors`. The converter must:
- Emit per-layer head counts, MoE config (expert count, shared count, intermediate sizes, routing scale), dense-lead count, SWA pattern mask.
- Handle both nested (`rope_parameters.full_attention / sliding_attention`) and flat rope config forms.
- Stack split-per-expert tensors (`mlp.experts.{i}.{gate,up,down}_proj.weight` → stacked).
- Hard-fail if `moe_apply_router_weight_on_input=True`.
- Do **not** call `add_sliding_window` — base class already does it.

---

## Files to Modify (10)

1. **`src/llama-arch.h`** — Add `LLM_ARCH_LAGUNA` to enum (near `LLM_ARCH_AFMOE`).
2. **`src/llama-arch.cpp`** — Add name map entry `{ LLM_ARCH_LAGUNA, "laguna" }` and tensor list block (verbatim copy of STEP35's entry — do **not** add `FFN_EXP_PROBS_B`; STEP35 already has it).
3. **`src/llama-model.cpp`** — Factory switch (`LLM_ARCH_LAGUNA → new llama_model_laguna`); RoPE-type switch (`LLAMA_ROPE_TYPE_NEOX` group); `llm_type_name()` (`LLM_TYPE_33B_A3B → "33B.A3B"`).
4. **`src/llama-model.h`** — Add `LLM_TYPE_33B_A3B` to `llm_type` enum.
5. **`src/models/models.h`** — Add `llama_model_laguna` struct (copy `llama_model_step35`, rename).
6. **`gguf-py/gguf/constants.py`** — Add `LAGUNA = auto()` to `MODEL_ARCH`; add name entry; add tensor list (verbatim copy of STEP35); add Path A YaRN SWA KV keys.
7. **`gguf-py/gguf/tensor_mapping.py`** — Add `"model.layers.{bid}.mlp.experts.e_score_correction"` to `FFN_EXP_PROBS_B` mapping (the `_bias` suffix is stripped by the framework).
8. **`conversion/__init__.py`** — Add `"LagunaForCausalLM": "laguna"` to `TEXT_MODEL_MAP`.
9. **`tests/test-llama-archs.cpp`** — Add `LLM_ARCH_LAGUNA` to `moe_mandatory()` and to the SWA-pattern-setup branch (same location as `STEP35` and `MIMO2`).
10. **`src/llama-hparams.{h,cpp}`, `src/llama-cparams.h`, `src/llama-context.cpp`** — Add Path A YaRN siblings; initialize to non-SWA defaults.

---

## Implementation Order

### Step 1 — Plumbing (compile + test-llama-archs)
Edit files 1–9. `laguna.cpp` build_arch_graph can stub with `GGML_ABORT("laguna not implemented")`.
Run: `./build/bin/test-llama-archs` — must pass.

### Step 2 — YaRN per-layer-type plumbing (Path A)
Edit file 10. Add new KV keys and writer helpers. Verify existing models unaffected.

### Step 3 — Python converter
Write `conversion/laguna.py`. Test with:
```
python convert_hf_to_gguf.py /tmp/laguna-xs2 --outfile /tmp/laguna-xs2.f16.gguf --outtype f16
./build/bin/llama-gguf /tmp/laguna-xs2.f16.gguf
```
Check: arch=laguna, head_count is array[40], dense_lead=1, expert_count=256, expert_shared=1, swa_pattern correct, rope params correct.

### Step 4 — C++ model class
Fill in `load_arch_hparams`, `load_arch_tensors`, `build_arch_graph`.
Test: `./build/bin/llama-cli -m /tmp/laguna-xs2.f16.gguf -p "The capital of France is" -n 5 --no-warmup --temp 0`
Successful load = tensor shapes are right. Expected output: `" Paris.\nThe capital of Germany is"`.

### Step 5 — Numerical validation (do not skip)
Compare against HF Transformers (fp32, temp 0, fixed seed). Require cosine sim > 0.999 on final hidden state, top-1 token match.
Common divergence causes:
1. YaRN params applied to wrong layer type
2. Expert stacking order reversed
3. `e_score_correction_bias` not loading (check tensor_mapping addition)
4. Gate dimension mismatch (`wqkv_gate` shape wrong for that layer's `n_head_l`)
5. `expert_weights_scale` missing

### Step 6 — SWA + long context
Test prompts > 8K (SWA) and > 32K (YaRN global layers). Compare perplexity against HF.

### Step 7 — Quantization
`llama-quantize` to Q4_K_M. Verify top-1 match against f16 on short prompts.

---

## Reference Output
Greedy decoding on `"The capital of France is"` → `" Paris.\nThe capital of Germany is"` (from sgl-project empirical reproducer).

## Open Item
`moe_router_logit_softcapping`: if non-zero in config.json, plumb through `LLM_KV_EXPERT_GATING_LOGIT_SOFTCAP` to `build_moe_ffn`. If zero, defer to follow-up.
