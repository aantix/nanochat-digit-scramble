# digit-scramble: does destroying numeric semantics in pretraining damage the prose around numbers?

**Hypothesis (falsifiable).** Replacing every digit in the *pretraining* text with a uniformly
random digit — format preserved, evaluation text untouched — degrades a language model's
prediction of the **non-digit** tokens that follow numbers (agreement and coherence effects:
"*19*-year-old", "1 item / 2 item**s**", plausible dates and scores), and not just the digit
tokens themselves. The alternative is that numbers in web text act as ignorable noise slots
whose surrounding prose is learned independently of their values. Prior work
([Razeghi et al. 2022](https://arxiv.org/abs/2202.07206)) showed *correlations* between
pretraining number frequencies and downstream numeric performance; nobody appears to have run
the causal intervention, nor measured *where* in the token stream the damage lands.

**Result in one line.** The damage halo is real but almost exactly **one token deep**:
digit tokens +1.20 bpb (z = 52.6), the first token after a number +0.066 bpb (z = 10.9,
≈ +4.6 % relative), and by distance 2 the effect is statistically indistinguishable from the
tiny global offset (+0.004 bpb).

---

## Environment (and honest scale disclaimer)

This experiment was executed autonomously inside a constrained sandbox: 4 ARM (aarch64) CPU
cores, 3 GB RAM, no GPU, processes killed after ~45 s (training therefore checkpoints and
resumes in chunks — see `train_chunked.py`, which fast-forwards the deterministic dataloader
so the token stream is *exactly* what an unchunked run would see), and huggingface.co blocked
by the sandbox proxy (so nanochat's default ClimbMix corpus was unavailable — AG News was
used instead, cloned from GitHub). The model is micro-scale (d4, dim 128, ~5.0 M params,
2.05 M training tokens per arm). Claims are made for this regime only; `run.sh` reproduces
everything and scales trivially to real hardware.

## Idea slate & selection (Phases 1–4)

The full tunable-surface inventory of nanochat (upstream commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`) is in [INVENTORY.md](INVENTORY.md); the 16-idea
slate with per-idea hypotheses/variables/metrics is in [IDEAS.md](IDEAS.md); every novelty
query and closest prior hit is logged in [NOVELTY.md](NOVELTY.md). Eleven of sixteen ideas
were killed by prior art (several by nanochat's own dev/LOG.md and modded-nanogpt records —
e.g. backout, smear, softcap, value-embedding ablations are all already measured in-repo).

Scoring of the five survivors (1–5 each: novelty confidence, surprise/delight, feasibility in
this environment, single-variable cleanliness):

| Idea | Novelty | Surprise | Feasibility | Design | Total |
|---|---|---|---|---|---|
| **I1 digit-scramble fingerprint** | **4** | **4** | **4** | **5** | **17** ← winner |
| I16 gate-channel double-duty fingerprint | 4 | 4 | 2 | 4 | 14 |
| I2 digit split `\p{N}{1,2}` vs `{1,3}` downstream | 3 | 3 | 3 | 4 | 13 |
| I4′ best-fit packing at fixed *unique-data* budget | 3 | 3 | 3 | 4 | 13 |
| I14 case-randomization augmentation | 2 | 2 | 4 | 5 | 13 |

Fallback order if the winner had proven infeasible: I16 → I4′ → I2 → I14.

## Novelty check (Phase 3 summary)

Closest prior work, per three parallel search agents (18 ideas-worth of logged queries in
[NOVELTY.md](NOVELTY.md)): Razeghi et al. 2022 (arXiv:2202.07206) — correlational only;
NumGPT (2109.03137) and Number Token Loss (2411.02083) — constructive number-representation
changes, no destructive data intervention; "Rewriting Pre-Training Data" (2505.02881) —
constructive rewriting. No published work was found that (a) causally randomizes digits in
pretraining data while preserving format, or (b) decomposes validation loss by token distance
from digits. This differs from all of the above in both the intervention and the fingerprint
metric.

## What was changed (the single variable)

`prepare_data.py` builds two parallel nanochat base dirs from the same 112,000 shuffled
AG News documents (26.2 MB, 0.87 % digit chars, 44 % of docs contain a digit):

- **baseline**: train shard as-is;
- **scrambled**: every ASCII digit in the train shard replaced by a uniform random digit
  (deterministic per-document rng). String lengths, digit positions, and pre-tokenization
  boundaries are preserved exactly (nanochat's split pattern chunks digits into 1–2 digit
  pre-tokens regardless of their values).

Everything else is bit-identical across arms: one shared BPE tokenizer (vocab 8192, trained
once on the clean train text via nanochat's `RustBPETokenizer`), the same clean 8,000-doc
validation shard, the same seeds (42), schedules, and step count. nanochat source is vendored
unmodified in `nanochat-src/` (model `gpt.py`, `MuonAdamW` optimizer, BOS-aligned best-fit
dataloader, bpb evaluator); the experiment adds standalone scripts only — no patches to
upstream code were needed.

Training config (both arms): depth 4, `aspect_ratio` 32 → dim 128, 2 heads (head_dim 64),
seq 512, window pattern "L" (SDPA/CPU), vocab 8192, fp32, batch 8×512 = 4096 tokens/step,
500 steps = 2.05 M tokens (~1.1 tokens per "scaling" param — far below nanochat's ratio-12
default; micro-budget, both arms identical). Optimizer and LR scaling rules replicated from
`base_train.py`: AdamW LRs ∝ 1/√(d/768), all LRs ∝ √(B/B_ref). Two departures, identical in
both arms and logged: Muon weight decay set to 0 (the T_epoch scaling rule explodes to ~6.0
at this horizon — itself a nice datapoint on where the one-dial abstraction breaks), and a
global LR multiplier calibrated once on the baseline arm (60-step probes, smoothed final
loss: ×1 → 7.25, ×4 → 6.12, **×16 → 5.71**, ×32 → 5.90; chose ×16 for both arms).

## Results

Final train loss (500 steps): baseline 4.265, scrambled 4.357 nats/token (train sets differ
by the intervention, so this gap includes irreducible digit entropy). Validation is the
shared **clean** shard; both arms see byte-identical packed batches, enabling *paired*
per-token statistics (`compare_paired.py`, 225 k target tokens; SE from per-token paired
differences):

| target-token class | n tokens | baseline bpb | scrambled bpb | Δ bpb (paired) | z |
|---|---|---|---|---|---|
| digit token | 5,340 | 2.874 | 4.079 | **+1.205 ± 0.023** | 52.6 |
| 1 after digit | 4,287 | 1.287 | 1.353 | **+0.066 ± 0.006** | 10.9 |
| 2 after digit | 3,285 | 1.417 | 1.422 | +0.005 ± 0.005 | 1.0 |
| 3 after digit | 3,113 | 1.507 | 1.505 | −0.001 ± 0.005 | −0.2 |
| 4 after digit | 3,025 | 1.472 | 1.470 | −0.001 ± 0.005 | −0.3 |
| 5–8 after digit | 11,250 | 1.496 | 1.501 | +0.005 ± 0.003 | 1.7 |
| far (>8) | 191,826 | 1.578 | 1.582 | **+0.004 ± 0.001** | 6.1 |
| overall | 222,126 | 1.577 | 1.593 | +0.016 | — |

Training-step trend (25-batch snapshot evals): the d1 damage is present throughout training
(step 150: +0.086; step 300: +0.048; step 500: +0.066) — it appears early and persists.
Probe color (`results/*_eval.json`): after "The year was 19" the baseline's top continuation
is "-year" (as in *19-year-old*, an AG News signature) with 24 % of mass on digit tokens;
the scrambled model scatters (34 % digit mass, higher digit entropy). Both arms still know
"' points' follows 'scored 2'" — syntax survives, values don't.

Plots: `results/train_loss.png`, `results/bpb_by_class.png`, `results/delta_bpb.png`,
`results/probe_entropy.png`. Raw logs: `runs/*/log.csv`; raw eval JSONs in `results/`.

Compute: 2 × 500 steps × 0.44 s/step ≈ 7.4 min wall on 4 ARM cores (plus ~3 min of probes
and evals). **0 GPU-hours.** Seeds: 42 everywhere (data shuffle, init); per-doc scramble rng
seeded 10000+doc_idx.

## Discussion

**Supported, with a twist.** Numbers are *not* purely ignorable noise slots: destroying their
semantics measurably damages non-digit prose prediction (d1: z ≈ 11). But the surprise is how
*local* the damage is — >90 % of the contextual excess damage sits at distance exactly 1, and
by distance 2 it is lost in the (significant but tiny, +0.23 % relative) global offset. The
intuitive mechanism-based prediction — date/plural/magnitude agreement propagating several
tokens ("in 1945, the war **ended**") — is wrong at this scale; what the model actually loses
is almost entirely bigram-shaped: digit→next-token statistics ("19"→"-year", "2"→" points",
score→score patterns). A second unpredicted number: the scrambled arm's digit bpb lands at
4.08, well *above* the log₂10 = 3.32 uniform-digit floor, because the model must also carry
type/length uncertainty over the 108 digit-bearing vocab entries — the baseline beats that
floor (2.87) by exploiting real-world digit priors (years, small scores).

**Honest confounds and limits.** (1) Micro scale: at d4/2 M tokens the model may simply lack
capacity to *learn* multi-token numeric dependencies; the halo could widen with scale — the
cleanest follow-up is sweeping this fingerprint across the nanochat miniseries (d4→d26).
(2) The small far-field offset (+0.004, z = 6.1) is consistent with unpredictable digits
wasting a bit of capacity/gradient signal globally, but could also reflect subtle
data-schedule differences (scrambled docs tokenize to occasionally different lengths, so
best-fit packing can pick different crops); we did not disentangle these. (3) One seed per
arm — the paired-token z-scores handle *evaluation* noise well, but not *training* seed
noise; the step-150/300/500 stability of d1 suggests the effect is robust, and idea I11's
literature (DataDecide) suggests deltas this size at z ≈ 11 paired are unlikely to be seed
flukes for a localized bucket while the global offset (z = 6) is more fragile.
(4) AG News, not ClimbMix (sandbox constraint) — news text is unusually number-dense, which
helps power but may not transfer.

**Natural follow-up.** Sweep the halo width across model scale; scramble only *some* digit
positions (e.g. keep leading digits) to see which bits of numeric information carry the d1
effect; and run the mirrored intervention (scramble at *evaluation* only) to separate
representation damage from context damage.

## Reproduce

```bash
git clone --depth 1 https://github.com/mhjabreel/CharCnn_Keras /tmp/agnews
./run.sh          # CPU-only, ~15 min on 4 cores; chunk-safe (rerun-able) throughout
```

Repo layout: `prepare_data.py` (the intervention), `train_tokenizer.py`, `train_chunked.py`
(faithful base_train replica, chunk-resumable), `eval_decomposed.py` (digit-distance bpb),
`compare_paired.py` (paired significance), `make_plots.py`, `nanochat-src/` (vendored
unmodified nanochat @ 92d63d4, MIT), `INVENTORY.md`, `IDEAS.md`, `NOVELTY.md`, `SELECTION.md`.
