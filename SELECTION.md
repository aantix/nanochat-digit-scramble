# Phase 4 — Autonomous selection

Novelty filter results (Phase 3, full query logs in NOVELTY.md): survivors were
I1 (digit-scramble fingerprint, conf 4/5), I16 (gate-channel fingerprint, conf 4/5),
I2 (digit split pattern downstream, conf 3/5), I4' (packing at fixed unique-data budget, conf 3/5),
I14 (case augmentation, marginal conf 2/5). All other ideas FAILED with named prior art.

## Scoring (1–5 each; environment: CPU-only sandbox, 4 cores/3GB RAM, 45s process lifetime → chunked checkpoint-resume training)

| Idea | (a) Novelty conf | (b) Surprise/delight | (c) Feasibility here | (d) Clean 1-var design | Total | Notes |
|---|---|---|---|---|---|---|
| **I1 digit-scramble fingerprint** | **4** | **4** | **4** | **5** | **17** | 2 runs; effect is local-statistics (learned early → visible at micro compute); eval identical across arms; quantitative invariant to check (scrambled digit bpb → log2 10 ≈ 3.32) |
| I16 gate-channel fingerprint | 4 | 4 | 2 | 4 | 14 | smear_lambda inits at 0 and gates barely move in a few hundred CPU steps → high risk of an uninformative null-by-undertraining |
| I2 digit split {1,2} vs {1,3} | 3 | 3 | 3 | 4 | 13 | needs 2 tokenizers; a tokenizer swap changes many merges at once; effect likely below micro-scale noise floor |
| I4' packing @ fixed data budget | 3 | 3 | 3 | 4 | 13 | needs legacy-loader port + multi-epoch data-constrained setup; more moving parts |
| I14 case augmentation | 2 | 2 | 4 | 5 | 13 | outcome largely predictable from adjacent literature |

## Winner: I1 — digit-scramble pretraining fingerprint (slug: `digit-scramble`)

Hypothesis (falsifiable): Replacing every digit in the pretraining text with a uniformly
random digit (format preserved; eval text untouched) measurably degrades the model's
prediction of NON-digit tokens that immediately follow numbers, beyond noise, while
far-from-digit text is unaffected — i.e., numeric semantics in web text support the
prediction of surrounding prose (agreement: "199_0s_", "1 item vs 2 item_s_", plausible
dates/scores), and are not merely ignorable noise slots.

Why it won: it causally completes a known correlational result (Razeghi et al. 2022) with a
clean 2-run, single-variable design whose fingerprint metric (bpb decomposed by distance
from digit tokens) is cheap to compute and informative in EITHER direction; it comes with
a built-in sanity invariant (digit-token bpb of the scrambled arm must approach
log2(10) ≈ 3.32 bits/byte); and its signal lives in local statistics that tiny models learn
within the first few hundred steps, matching the compute available.

Runner-up order (fallback if winner blocked mid-implementation): I16 → I4' → I2 → I14.

## Environment-forced deviations (logged)
- No GPU; 45 s max process lifetime → custom chunk-resumable training driver that reuses
  nanochat's model/optimizer/dataloader/eval modules unmodified and replicates
  base_train.py's schedules exactly; torch.compile disabled.
- huggingface.co blocked by sandbox proxy → ClimbMix unavailable. Pretraining corpus:
  AG News (127.6k news snippets, ~29 MB, number-rich), identical for both arms; last
  slice held out as validation. Corpus is a constant, not a variable.
- PyPI torch 2.9.1 (CUDA build, CPU-used) installed --no-deps due to proxy blocks + disk limits.
- Model scaled down honestly: depth 4, aspect-ratio 32 (dim 128), vocab 8192, seq 512 —
  reported as micro-scale; no claims beyond this regime.
