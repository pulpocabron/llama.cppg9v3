IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

---

# Task: Add Laguna (Poolside) Support to llama.cpp

**Goal:** Implement `LLM_ARCH_LAGUNA` covering the Laguna family. Initially targeted `poolside/Laguna-XS.2` (33B-A3B: 40 layers, 256 routed + 1 shared expert, head-wise softplus attention gate, per-layer head counts, dual RoPE (YaRN for global, default for SWA), per-layer-type partial rotary). Later generalized to also support **`poolside/Laguna-M.1`** (~226B: 70 layers, all full attention, per-element attention gate, full rotary) — see "Laguna-M.1 generalization" below.

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
- **Chat template + streaming parser**: `models/templates/poolside-Laguna.jinja` added; name registration and auto-detection via `laguna_glm_thinking_v5` marker in `src/llama-chat.cpp`/`.h`. Autoparser workaround in `common/chat-diff-analyzer.cpp` fixes two streaming bugs (see learnings 9–10).
- **Chat template validation + continuation fix (2026-07-02)**: Full template test pass against live XS.2 (IQ4_XS, llama-server): rendering via `/apply-template`, non-streaming + streaming completions, thinking on/off, tool calls (both modes, parsed to structured `tool_calls`), tool round trips, multi-turn, mid-conversation system msgs, raw mode — all correct, no `</assistant>`/`<think>` leaks. Fixed a `continue_final_message` (assistant prefill) bug for delimiter-style reasoning (learning 14). Added Laguna coverage: peg tests + continuation prompt-placement asserts in `tests/test-chat.cpp`, role-markers entry in `tests/test-chat-auto-parser.cpp`.
- **Laguna-M.1 generalization**: Extended converter + `laguna.cpp` from XS.2-only to the all-full-attention sibling `poolside/Laguna-M.1` (~226B). (1) `conversion/laguna.py` defaults `layer_types` to all-`full_attention` when absent and writes data-driven `rope.dimension_count`/`rope.dimension_count_swa` plus a new `attention.gate_per_head` KV. (2) `laguna.cpp` removed the hardcoded `n_rot_full /= 2`, supports per-element gating, sets `swa_type` from `is_swa_any()`, and branches the attention input path. A 420 GiB f16 GGUF was produced and verified to load + run; XS.2 regression (`test-llama-archs laguna`) still `OK`.

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
9. **Prefilled thinking token breaks autoparser**: Laguna prefills `<think>` as the generation prompt prefix (thinking=on). Auto-detection finds `<think>` as `reasoning.start` from template diffs, but it never appears in the generated stream — the parser never enters reasoning mode. Fix: set `reasoning.start = ""` (delimiter-style) in the workaround. The `analyze_reasoning::build_parser` was extended to return `p.eps()` when `start.empty() && !ctx.inputs.enable_thinking`, preventing the streaming parser from misclassifying all plain content as reasoning when thinking is disabled (`--reasoning off`).
10. **EOT token as regular vocab token**: `</assistant>` (token 24) is a regular vocabulary token, not a special token. `preserved_tokens` only controls special-token decode and doesn't suppress it. Use `additional_stops` instead — this feeds into `task->params.antiprompt` which the `process_token` erase logic checks, stripping the text before it's sent. Added `additional_stops` field to `autoparser` struct; apply in `cli.cpp` as `task.params.antiprompt` additions.
11. **`swa_type` must match `is_swa_any()` (`create_memory` assert)**: `llama_model::create_memory` asserts `swa_type != NONE iff is_swa_any()` — the iswa KV cache is only allocated when there are SWA layers. Laguna-M.1 has zero SWA layers, so `swa_type` must be `NONE` and the graph builder must use `build_attn_inp_kv()` (+ its `build_attn` overload), not `build_attn_inp_kv_iswa()`. `laguna.cpp` sets `swa_type = is_swa_any() ? STANDARD : NONE` and branches the attention input on `has_swa`. (Leaving `swa_type = STANDARD` with zero SWA layers was the first M.1 smoke-test crash.)
12. **Per-element vs per-head attention gate**: the gate exists on **all** layers (not SWA-only). Laguna-M.1 is per-element (g_proj output = `num_heads*head_dim`, element-wise `attn *= softplus(g_proj(x))`); XS.2 is per-head (g_proj output = `num_heads`, broadcast across head_dim). **Config `gating` is a bool (on/off), not a mode string** — the converter must detect the mode from the actual `g_proj` tensor's out_features (per-head ⟺ `out == n_head(layer)`), and writes a `laguna.attention.gate_per_head` bool KV (default `true` = per-head, so old XS.2 GGUFs keep working); `laguna.cpp` declares the gate tensor shape from it (`{n_embd, n_head_l}` vs `{n_embd, n_head_l*n_embd_head_v}`) and branches the application — per-element is a plain `ggml_mul`. *(Reading the mode from `config["gating"] == "per-head"` was the original bug — it wrote per-element for XS.2, which would have made reconverted GGUFs fail to load with a gate-tensor shape mismatch.)*
14. **Continuation with empty `reasoning.start` (assistant prefill)**: `generate_parser` in `chat-auto-parser-generator.cpp` only rebuilt the continuation generation prompt when `reasoning.start` was non-empty. With Laguna's delimiter-style workaround (`start=""`), content continuation appended the content directly after the prefilled `<think>` (inside the think block) and reasoning continuation dropped `reasoning_content` entirely. Fix: `else if (!reasoning.end.empty() && enable_thinking)` branch appends `msg.reasoning_content` (+ `reasoning.end` for CONTENT continuation) directly — the opening tag is already in the generation prompt. Continuation only reproduces the model's native format when the client round-trips reasoning/content verbatim (reasoning ends `\n`, content starts `\n`); hand-stripping those newlines makes the model re-emit a stray `</think>`.
15. **`reasoning_format=none` quirk (known, left as-is)**: with thinking on, content is the raw generated stream — `reasoning…</think>\nanswer` with **no opening `<think>`**, because the tag is prefilled in the prompt and never generated. Raw passthrough is consistent behavior; the peg parser cannot inject text that isn't in the stream.
16. **`layer_types` defaults to all full_attention (Laguna-M); rotary is data-driven**: `configuration_laguna.py` defaults `layer_types` to `["full_attention"]*n_layers` when absent (Laguna-M); Laguna-XS ships an explicit mix. The converter must mirror this or the per-layer `head_count`/`sliding_window_pattern` arrays come out empty. Partial rotary is now data-driven via `rope.dimension_count`/`rope.dimension_count_swa` (written from each rope config's `partial_rotary_factor`: XS.2 0.5 global / 1.0 SWA; M.1 1.0 everywhere), replacing the old hardcoded `n_rot_full /= 2`.

---

## Architectural Facts

| Property | Value |
|---|---|
| Layers | 40 (10 global + 30 SWA, 3:1 pattern) |
| Attention heads | 48 global / 64 SWA, via `num_attention_heads_per_layer` |
| KV heads | 8 (GQA) |
| SWA window | 512 tokens |
| Q/K norm | RMSNorm |
| Attention gate | `self_attn.g_proj`, per-head **softplus**, applied on **all layers** to SDPA output before `o_proj` |
| MoE router | Sigmoid + `e_score_correction_bias` added at selection time only |
| Experts | 256 routed (top-8) + 1 shared |
| Dense layers | Layer 0 only (`n_layer_dense_lead = 1`) |
| Routed scaling | `moe_routed_scaling_factor` (2.5) → `expert_weights_scale` |
| Global RoPE | YaRN: base 500K, factor 64, original_max 4096, β_fast 64, β_slow 1, partial_rotary 0.5 |
| SWA RoPE | default: base 10K, full rotary |
| Context length | 262,144 |
| Checkpoint layout | split-per-expert, singular `shared_expert`, `e_score_correction_bias` under `experts` not `gate` |

> **Laguna-M.1** differs from the XS.2 values above: 70 layers (3 dense + 67 sparse), uniform 64 heads / 8 KV heads (no per-layer head counts), `sliding_window = 0` with **no SWA layers** (`layer_types` all `full_attention`), **per-element** attention gate (vs XS.2's per-head — both apply the gate on every layer), **full rotary on every layer** (partial_rotary 1.0; XS.2 is half-rotary on global layers), top-16 routing, 3 dense-lead layers. YaRN factor (64) and context length (262144) match XS.2. Same tensor layout and MoE structure as XS.2.

---

## Key Design Decisions

**YaRN per-layer approach:** Use **Path A** — add per-layer-type YaRN siblings to hparams/cparams (`yarn_ext_factor_swa`, `yarn_attn_factor_swa`, `yarn_beta_fast_swa`, `yarn_beta_slow_swa`). Initialize them to non-SWA values so other architectures are unaffected. In the graph builder, select via `hparams.is_swa(il)`.

**Partial rotary:** Data-driven via `rope.dimension_count` (global layers) and `rope.dimension_count.swa` (SWA layers), written by the converter from each rope config's `partial_rotary_factor` (XS.2: 0.5 global / 1.0 SWA; M.1: 1.0 everywhere). The earlier hardcoded `n_rot_full /= 2` was removed — it only worked for XS.2.

**Dense lead:** Layer 0 is dense FFN; layers 1–39 are MoE. Branch on `static_cast<uint32_t>(i) < hparams.n_layer_dense_lead`.

**Attention gate:** `TENSOR_NOT_REQUIRED` on `wqkv_gate`; guard with `if (model.layers[il].wqkv_gate)` in the graph builder.

**Attention gate mode (M.1):** Per-element (`gate_per_head=false`) applies a plain element-wise `ggml_mul(attn_out, gate)`; per-head (XS.2) reshapes the gate to `[1, n_head, n_tokens]` and broadcasts. Mode comes from the `laguna.attention.gate_per_head` KV, which the converter sets by inspecting the `g_proj` tensor shape (config `gating` is a bool, not a mode string — see learning 12).

**SWA path selection (M.1):** `swa_type` and the attention input builder are chosen from `is_swa_any()` — iswa path for XS.2 (mixed SWA), plain `build_attn_inp_kv()` for M.1 (no SWA). Required by `create_memory`'s `swa_type ⇔ is_swa_any()` assert.

---

## Files Added (2)

### `src/models/laguna.cpp` ✅
Full implementation based on step35.cpp with:
- `load_arch_hparams`: reads RMS eps, SWA type, MoE params, dense lead, expert shared count, SWA pattern. Maps 40 layers to `LLM_TYPE_33B_A3B`.
- `load_arch_tensors`: per-layer Q/K/V with variable head counts, optional attention gate, dense MLP tensors, MoE routed expert tensors, shared expert tensors.
- `build_arch_graph`: RMS norm, Q/K/V projections with per-head norms, partial rotary RoPE, per-head softplus attention gate (all layers), dense/MoE FFN branching, shared expert MLP.

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
12. **`src/llama-chat.h`, `src/llama-chat.cpp`** ✅ — `LLM_CHAT_TEMPLATE_LAGUNA` enum value, name map entry, auto-detection via `laguna_glm_thinking_v5` marker, C++ fallback formatter.
13. **`models/templates/poolside-Laguna.jinja`** ✅ — Full Jinja template extracted from GGUF; `{%- generation -%}` block added; `enable_thinking` variable controls `<think>`/`</think>` prefix in generation prompt.
14. **`common/chat-auto-parser.h`** ✅ — Added `additional_stops` field to `autoparser` struct.
15. **`common/chat-auto-parser-generator.cpp`** ✅ — Propagate `autoparser.additional_stops` into `data.additional_stops`; check `ctx.inputs.enable_thinking` in `analyze_reasoning::build_parser` for empty-start case.
16. **`common/chat-diff-analyzer.cpp`** ✅ — Laguna workaround: `reasoning.start=""`, `reasoning.end="</think>"`, `mode=TAG_BASED`, `additional_stops=["</assistant>"]`.
17. **`tools/cli/cli.cpp`** ✅ — Apply `chat_params.additional_stops` to `task.params.antiprompt` so the CLI stop-word erase logic strips them from streaming output.

### Laguna-M.1 generalization (additional)

18. **`conversion/laguna.py`** ✅ — Default `layer_types` to all-`full_attention` when absent; write `rope.dimension_count`/`dimension_count_swa` from `partial_rotary_factor`; write `attention.gate_per_head` **detected from the `g_proj` tensor shape** (config `gating` is a bool, not a mode string).
19. **`src/models/laguna.cpp`** ✅ — Removed hardcoded `n_rot_full /= 2`; per-element vs per-head gate (shape + application); `swa_type` from `is_swa_any()` with branched attention input path (`build_attn_inp_kv` vs `_iswa`).
20. **`gguf-py/gguf/constants.py` + `gguf-py/gguf/gguf_writer.py`** ✅ — New `attention.gate_per_head` key (`Keys.Attention.GATE_PER_HEAD`, `KEY_ATTENTION_GATE_PER_HEAD`) + `add_attention_gate_per_head()`.
21. **`src/llama-arch.h` + `src/llama-arch.cpp`** ✅ — `LLM_KV_ATTENTION_GATE_PER_HEAD` enum value + `"attention.gate_per_head"` string mapping.
22. **`src/llama-hparams.h`** ✅ — `bool attn_gate_per_head = true;` field.

---

## Reference Output
Greedy decoding on `"The capital of France is"` → `" Paris.\nThe capital of Germany is"` (from sgl-project empirical reproducer). *(XS.2 only; M.1 reference output not captured — the 226B MoE is I/O-bound on 60 GB RAM, ~50 s/token via mmap.)*

## Open Items
- `moe_router_logit_softcapping` is absent from both Laguna-XS.2 and Laguna-M.1 `config.json` — no plumbing needed.
- **Laguna-M.1 chat template**: ships `laguna_glm_thinking_v4`, but auto-detection in `src/llama-chat.cpp` keys on the `laguna_glm_thinking_v5` marker, so the built-in matcher rejects it (`custom template not supported`). Workaround: pass `--jinja`. Follow-up: add the v4 marker to auto-detection.
- **Step 5 / Step 7** still deferred — numerical validation needs CUDA; the 226B M.1 is also impractical for the quant top-1 check on current hardware.
- **Pre-existing XS.2 GGUFs need reconversion** — they predate the now-required `rope.dimension_count` and would otherwise get full rotary instead of half. **In progress (2026-06-20):** re-cloning latest `poolside/Laguna-XS.2` and reconverting to f16 → Q4_K_M/IQ4_XS; also fixed a converter `gate_per_head` bug (learning 12) that would otherwise have made the new GGUF fail to load.
