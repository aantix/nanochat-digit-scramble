# Phase 3 — arXiv/Scholar/community novelty filter (full query logs)

Rule applied: an idea FAILS if any published work (arXiv/ACL/NeurIPS, rigorous blogs,
karpathy/nanochat dev/LOG.md + Discussions, KellerJordan/modded-nanogpt records/PRs)
measured the same question. A different scale or codebase is NOT novelty.
Searches executed 2026-07-07 by three parallel research agents (WebSearch + arXiv API
+ direct fetches of modded-nanogpt README/records and nanochat dev/LOG.md).

## I1 digit-scramble fingerprint — SURVIVES (confidence 4/5) ← WINNER
Queries:
1. "digit randomization pretraining data ablation language model numeracy" → NumGPT (arXiv:2109.03137) — constructive number encodings, not destructive scramble.
2. "replacing numbers with random digits training corpus language model effect on loss" → Number Token Loss (arXiv:2411.02083, ICML 2025) — loss modification, no data corruption, no neighbor-token measurement.
3. "Razeghi term frequency numerical reasoning causal intervention pretraining corpus" → Razeghi et al. arXiv:2202.07206 (EMNLP-F 2022) — explicitly correlational; the causal retrain intervention is exactly what's missing.
4. "token-level loss decomposition around number tokens language model perplexity digits" → arXiv:2603.29396 — minimal-pair perplexity, no digit-distance stratification.
5. "corrupting numbers pretraining data ablation 'random number' substitution effect surrounding text prediction LLM" → arXiv:2505.02881 — constructive rewriting, opposite direction.
6. arXiv API all:"digit"+"pretraining"+"randomized"+"language model" → empty.
Closest prior: Razeghi 2022 (correlational). The causal intervention + the near/far bpb decomposition fingerprint are unmeasured.

## I16 gate-channel double-duty fingerprint — SURVIVES (confidence 4/5) — runner-up 1
Queries: "modded-nanogpt ve_gate smear_gate first channels token embedding gate" (mechanism documented, no channel analysis); "per-channel specialization embedding table gate reads fixed channel subset transformer" (no hit); "modded-nanogpt PR 218 tune value embed layout ve_gates channels analysis" (PR tunes layout, not channel structure); "emergent structure specific embedding dimensions used by gating sigmoid 'first N channels' nanogpt" (no hit); arXiv API all:"outlier dimensions"+"token embedding"+gate (empty).
Not selected here: smear_lambda inits at 0 and gates move slowly — at 300 CPU steps high risk of null-by-undertraining.

## I4' best-fit packing at fixed unique-data budget — PARTIAL SURVIVES (3/5) — runner-up 2
Fixed-steps arm measured in nanochat Discussion #481 / dev LOG ("BOS-aligned dataloader… What Worked"); Ding et al. arXiv:2404.10830 killed the generic question; the fixed-unique-data arm (35% discard under data scarcity) is unmeasured (queries: "Fewer Truncations Improve Language Modeling best-fit packing Ding 2024"; "nanochat karpathy BOS-aligned packing dataloader ablation bpb"; "document packing strategy ablation fixed unique data budget vs fixed steps pretraining truncation" → SkyLadder, Seamless Packing arXiv:2505.22018 — none discard data).

## I2 digit split {1,2} vs {1,3} natural-text bpb — SURVIVES (3/5) — runner-up 3
Queries: "'Tokenization counts' Singh Strouse digit grouping tokenization arithmetic" (arXiv:2402.14903 — arithmetic accuracy, not natural-text bpb); "right-to-left digit grouping three-digit tokenization language model perplexity natural text" (arXiv:2604.11582 — reasoning accuracy); "number tokenization scheme comparison language modeling loss bits per byte digits chunks" (arXiv:2605.01188 — not this variable); "nanochat tokenizer digit split regex" (nanochat issue #25 — choice justified by intuition, unmeasured downstream).

## I14 case-randomization augmentation — SURVIVES marginal (2/5) — runner-up 4
Near-killed by: arXiv:1911.05241 (NER casing augmentation), UniCase arXiv:2010.11936, TACL 2025 char-noise continual pretraining, arXiv:2604.16037 (stochastic tokenization robustness — same tradeoff shape). Technically unmeasured for from-scratch LM bpb, but outcome predictable.

## FAILED ideas (killing prior art)
- I3 vocab size @ fixed FLOPs: Tao et al. arXiv:2407.13623 (NeurIPS 2024) — isoFLOPs vocab sweeps incl. head-dominant regime. (4 queries logged.)
- I5 QK sharpening ×1.2: introduced via Karpathy autoresearch with measured d12/d24 validation (X post status/2031135152349524125). (4 queries.)
- I6 backout: nanochat dev/LOG.md (Jan 2026, "Backout | No improvement", later adopted after autoresearch); modded-nanogpt PR #140 = WR record 40; medium record 15. (4 queries.)
- I7 smear: nanochat dev/LOG.md ("Smear gate | Negligible"); modded-nanogpt record 34 (PR #130) + measured removal after bigram embeds. (4 queries.)
- I8 value embeddings at d4: nanochat dev/LOG.md 2026-01-17 (placement/gating ablated); ResFormer arXiv:2410.17897 (depth-dependence); snimu VE blog ablations. d4 rerun = scale change only. (4 queries.)
- I9 LR-rule transfer at d2–d4: Tensor Programs V arXiv:2203.03466; empirical muP-transfer study arXiv:2404.05728; arXiv:2510.19093. (4 queries.)
- I10 special-token re-init before SFT: Hewitt 2021 (mean-embedding init, measured); TRL setup_chat_format practice; Fishing for Magikarp arXiv:2405.05417 (untrained-token characterization); arXiv:2510.21954 (mean vs random convergence). (4 queries.)
- I11 seed-variance noise floor: DataDecide arXiv:2504.11393 (seed noise vs recipe deltas at small scale); Signal & Noise arXiv:2508.13144; Wortsman arXiv:2309.14322. (4 queries.)
- I12 doc-length curriculum: Shortformer (ACL 2021); SLW arXiv:2108.06084; Length-Based CL (2022); Dataset Decomposition arXiv:2405.13226 (doc-aligned packing + length curriculum, ablated). (4 queries.)
- I13 duplication memorization cliff: Carlini arXiv:2202.07646 (log-linear dup→memorization); Secret Sharer (USENIX 2019, planted canaries at N reps); Lee arXiv:2107.06499; TinyMem arXiv:2410.02159 (small-scale). (5 queries.)
- I15 logit softcap: nanochat dev/LOG.md 2026-03-02 (softcap swept 5–30 at d24); modded-nanogpt records 9 & 18; Gemma 2 arXiv:2408.00118. (4 queries.)

Survivors ≥ 3 → proceed to Phase 4 selection (see README).
