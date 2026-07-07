# nanochat tunable surface inventory

Source: https://github.com/karpathy/nanochat @ master (cloned 2026-07-07).
Every knob is a candidate; nothing pre-filtered.

## 1. The --depth dial and auto-derived hyperparameters (scripts/base_train.py)
- `--depth` (default 20): the single "dial". Auto-derives:
  - `model_dim = depth * aspect_ratio` (aspect-ratio default 64), nudged up to a multiple of `head_dim` (128) → `num_heads = model_dim / head_dim`, `n_kv_head = n_head` (GQA off by default).
  - Training horizon: `target_tokens = target_param_data_ratio (12) * scaling_params` where scaling_params = transformer_matrices + lm_head (per dev/LOG.md Jan 27 2026).
  - Batch size via Power-Lines: `B = B_REF(2^19 @ d12) * (D/D_REF)^0.383`, rounded to power of 2.
  - LR scaling: AdamW LRs ∝ 1/√(model_dim/768); all LRs × √(B/B_REF).
  - Weight decay: λ = λ_ref·√(B/B_ref)·(D_ref/D), cosine-decayed to 0 over the run.
  - Muon momentum schedule (0.85→0.97 over 400 steps, →0.90 during warmdown).
  Each coupling is individually breakable (aspect-ratio, head-dim, target-param-data-ratio, total-batch-size, the four LRs, warmup/warmdown/final-lr-frac).

## 2. Tokenizer (scripts/tok_train.py, nanochat/tokenizer.py)
- `--vocab-size` (32768), `--max-chars` (2B), `--doc-cap` (10k chars/doc).
- BPE training corpus itself (which shards, domain mix).
- SPLIT_PATTERN: GPT-4-style regex, deviation: `\p{N}{1,2}` instead of `\p{N}{1,3}` (numbers chunked to ≤2 digits; Karpathy: "2 is the sweet spot for 32K vocab", verified on compression only).
- 9 SPECIAL_TOKENS (<|bos|>, user/assistant/python/output start/end): never seen in pretraining → their wte rows keep their random init (std 0.8) until mid/SFT stages.
- token_bytes table (used by bpb eval; special tokens count 0 bytes).

## 3. Pretraining data & loader (nanochat/dataset.py, dataloader.py)
- Dataset: ClimbMix-400B parquet shards (HF), last shard = val. Shard order is sequential, row groups strided by rank; no document-level shuffle at train time.
- BOS-aligned best-fit packing: every row starts with <|bos|>, best-fit from a 1000-doc buffer, crops to fill; ~35% of tokens discarded by cropping (vs legacy croppy packing, kept as fallback).
- `--max-seq-len` (2048), `--device-batch-size`, tokenizer_threads/batch_size, buffer_size.
- The raw text itself (any content intervention slots in either at parquet level or at `refill_buffer`).

## 4. Precision / runtime (nanochat/common.py, fp8.py)
- NANOCHAT_DTYPE env: bfloat16 / float16 (GradScaler path) / float32; auto: bf16 on SM80+, fp32 on CPU/MPS.
- `--fp8` + `--fp8-recipe` (tensorwise/rowwise) with min-dim filter.
- torch.compile(dynamic=False) applied unconditionally; FA3 vs SDPA fallback; `--window-pattern` "SSSL" (S = quarter-context sliding window, final layer always L).

## 5. Architecture internals (nanochat/gpt.py)
- Rotary only (no learned pos emb), base 100000, cache 10× seq_len.
- QK norm + fixed sharpening q*1.2, k*1.2.
- Untied wte/lm_head; wte init N(0, 0.8); lm_head init N(0, 0.001); wte stored in COMPUTE_DTYPE.
- Value embeddings (ResFormer-style) on alternating layers (last always), gated by `ve_gate` reading **the first 12 channels of x**; gate range (0,3) via 3·sigmoid.
- Smear: previous-token embedding mixed in via gate reading **the first 24 channels**; smear_lambda init 0.
- Backout: subtract 0.2×(mid-layer residual) before final norm (backout_lambda learnable).
- Per-layer scalars: resid_lambdas (init 1.15→1.05 across depth), x0_lambdas (init 0.20→0.05) blending normed input embedding back in each layer.
- MLP: 4×, ReLU²; matrices init uniform (c_fc scaled 0.4×), projections zero-init.
- Logit softcap 15 (tanh); logits fp32; vocab padded to multiple of 64.
- MuonAdamW split (nanochat/optim.py): Muon (momentum 0.95→sched, ns_steps 5, beta2 0.9, cautious weight decay) for block matrices; AdamW for lm_head (lr 0.008, β(0.8,0.96)), wte (0.3, β(0.8,0.995)), value_embeds (0.5×wte lr), resid (0.005), x0 (0.5, β(0.96,0.95)), smear/backout (0.2). Every constant is a knob.

## 6. Evaluation (nanochat/loss_eval.py, core_eval.py, scripts/base_eval.py)
- val_bpb: bits-per-byte, normalized by target-token byte length → tokenizer/vocab-independent; special tokens masked. `--eval-tokens`, `--eval-every`.
- CORE metric (22 tasks, centered vs random baseline), `--core-metric-max-per-task`.
- Sampling probes (7 fixed prompts). All probe sets/decodings are tunable; per-token-class loss decompositions are easy to add (loss_reduction='none').

## 7. Midtraining/SFT (scripts/chat_sft.py, tasks/)
- TaskMixture: SmolTalk + MMLU aux (×3 epochs) + GSM8K (×4 epochs); `--mmlu-epochs`, `--gsm8k-epochs`; custom JSON conversations (customjson pattern); identity infusion via synthetic conversations.
- Inherits max_seq_len/batch/LRs from base checkpoint (each overridable); init_lr_frac 0.8; loss masked to assistant tokens; padded rows.
- Chat special-token rendering (nanochat/engine.py renders conversation schema).

## 8. RL (scripts/chat_rl.py)
- Simplified GRPO ≈ REINFORCE on GSM8K: no KL/trust region, on-policy, token-level DAPO-style normalization, advantage = r − mean(r).
- Knobs: num-samples 16/example, examples-per-step, temperature 1.0, top-k 50, max-new-tokens, reward definition (exact-match), init_lr_frac 0.05, epochs.

## 9. Inference engine (nanochat/engine.py)
- KV cache mgmt, prefill/decode, temperature/top-k, tool-use loop (<|python_start|> sandboxed calculator), batch generation. Decode-time knobs and cache policies.

## Constraints noted for this run (sandbox)
- CPU-only (4 cores, 3GB RAM), 45s max per shell call, background processes killed between calls → training must checkpoint/resume in <45s chunks; torch.compile disabled via TORCH_COMPILE_DISABLE=1.
- huggingface.co blocked by proxy → ClimbMix and HF-hub SFT tasks unavailable; pretraining corpus must come from an allowed host (github.com/PyPI), identical for both arms.
- PyPI torch is the CUDA build (900MB wheel), installed --no-deps for CPU use.
