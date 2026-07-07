# ENGINEER.md — this experiment, translated for software engineers

This document maps the machine-learning decisions in the digit-scramble experiment onto
everyday software-engineering concepts, then walks through every parameter we changed —
before and after — with an engineering hypothesis for *why* the numbers came out the way
they did. If you have never trained a model but you have shipped code, this is written for you.

## Part 1 — the ML↔SWE dictionary

| ML concept (as used here) | Closest engineering concept | Why the analogy holds |
|---|---|---|
| Pretraining corpus | The production input log you replay in tests | The model's entire "spec" is inferred from this data; there is no other requirements doc. Corrupt the log and you corrupt the learned behavior. |
| Tokenizer (BPE) | A lexer that turns bytes into a fixed symbol table (enum) | It chunks raw text into ~8k reusable symbols. "19" is one token the way `>=` is one lexer token, not two chars. |
| Vocabulary (8,192 tokens) | An enum / interned string table | Fixed at build time. 108 of the 8,192 entries "contain a digit" — a known subset of the enum. |
| A model parameter (weight) | A tuning constant discovered by search, not written by a human | Nobody sets the 5M numbers by hand; gradient descent fits them, like a profiler-guided auto-tuner picking constants. |
| Loss (cross-entropy) | The error metric your optimizer minimizes | Lower = the model's predicted next-token distribution matches reality better. It is the objective function of the whole build. |
| bits-per-byte (bpb) | A normalized benchmark (throughput per core, not raw throughput) | Raw loss depends on tokenizer choices; bpb divides by byte length so two different builds are comparable — like reporting ns/op instead of wall-clock. |
| Gradient descent | A feedback controller minimizing error | Each step nudges the constants downhill. The learning rate is the controller gain. |
| Learning rate (LR) | Step size / controller gain / backoff aggressiveness | Too small: converges too slowly to finish in budget. Too large: overshoots and oscillates (diverges). Classic control-loop tuning. |
| Weight decay | Regularization = pressure against overfitting | Engineering-wise: a penalty that stops the system from hard-coding to the exact training inputs, like a linter that flags magic numbers memorized from one test case. |
| Overfitting | Hard-coding to the test fixture instead of solving the general case | The model "passes" on seen data but generalizes poorly. |
| Depth / width / the "one dial" | A build preset that derives dozens of config values | nanochat's `--depth` is a single knob that auto-derives width, heads, LR, token budget — like choosing `--profile=prod` and having 30 settings cascade. |
| Validation set | Your held-out integration test suite | Never trained on; the honest measurement of generalization. Here it is *clean* (real digits) for both arms — a fixed test harness. |
| Evaluation probes | Assertion-level unit tests on specific inputs | "After 'The year was 19', what does it predict?" is a targeted unit test of one behavior. |

## Part 2 — the experiment as a controlled A/B test

In engineering terms this experiment is a **single-variable A/B test with a shared test
harness**. Two builds of the same system are compiled from two copies of the input log that
differ in exactly one field-level transformation; both are then scored against the *same*
untouched test suite. Everything else — the lexer, the test set, the RNG seeds, the optimizer
schedule, the step count — is pinned, the way you would pin every dependency version before
attributing a latency change to your one code diff.

**The one variable:** in the "scrambled" build's input log, every digit character is replaced
by a uniformly random digit. Crucially, this is a *semantics-only* corruption that *preserves
the schema*: string lengths, digit positions, and lexer boundaries are untouched — "1969"
becomes "4207", never "19@#". In engineering terms, we kept the type and the field width and
only randomized the value. That is what makes the test clean: any measured difference must come
from the model no longer being able to rely on *what the numbers mean*, not from the text
looking structurally different.

## Part 3 — every adjusted parameter, before → after, with an engineering hypothesis

### 3a. The independent variable (the diff under test)

| Parameter | Before (baseline) | After (scrambled) | Engineering hypothesis for the effect |
|---|---|---|---|
| Digit values in the **training** log | real digits | uniform-random digits (format preserved) | We severed the correlation between a field's value and its context. Prediction that depended on *reading* the value now has no signal; prediction that only depended on the field's *shape* is unaffected. This is the whole experiment: measure the blast radius of corrupting one field's semantics. |

**Result and its engineering reading.** The damage was sharply *local*, which is the
interesting part:

| Test bucket (target token) | baseline bpb | scrambled bpb | Δ | Engineering interpretation |
|---|---|---|---|---|
| the digit token itself | 2.874 | 4.079 | **+1.205** (z=52.6) | The field is now genuinely unpredictable. Note it lands *above* the log₂10 = 3.32 "uniform digit" floor: the model also lost track of *how many* digits and *which vocab entry*, so it pays entropy on type+length too, not just value. Like a checksum going from "predictable" to "worse than random" because you also lost the length header. |
| 1 token after a digit | 1.287 | 1.353 | **+0.066** (z=10.9) | Real, significant coupling — but only one hop. The token immediately downstream of a number *did* depend on the number's value ("19"→"-year", score→score). |
| 2+ tokens after a digit | ~1.47–1.50 | ~1.47–1.50 | ~0 (z≈1) | **No measurable coupling.** The dependency did not propagate. |
| far from any digit (>8) | 1.578 | 1.582 | +0.004 (z=6.1) | A tiny global tax, discussed below. |

**Hypothesis, in engineering terms.** We expected the corruption to have a wide blast
radius — dates driving verb tense, quantities driving plurals, several tokens downstream
("in 1945, the war **ended**"). Instead the coupling was **one function call deep**. The model,
at this scale, had mostly learned *bigram-shaped* dependencies: local `value → next-token`
lookups, not long-range reasoning. In code-review language: the number field was tightly
coupled to its immediate successor and essentially *decoupled* from everything two hops out.
Corrupting it broke exactly the one edge that depended on it and nothing else — a system with
much better locality/encapsulation than we predicted.

**The small far-field tax (+0.004).** Statistically real (z=6.1) but ~250× smaller than the
digit effect. Two plausible causes, exactly like diagnosing a small global regression:
(1) *capacity contention* — the model spends a slice of its finite parameters/compute trying,
and failing, to predict now-random digits, starving everything else slightly (noisy input
consuming a shared resource); (2) *a scheduling confound* — scrambled documents occasionally
tokenize to different lengths, so the best-fit packer can cut batches at different offsets,
a subtle change in the order records are fed. We did not disentangle these — an honest "known
unreproduced variable," flagged rather than hidden.

### 3b. Parameters we changed to make the test *run at all* (held identical across both arms)

These are not the experiment; they are the test rig. Each is pinned identically for both
builds, so none can explain the A/B difference. They matter because choosing them wrong would
have made the result meaningless — the ML equivalent of a flaky test harness.

| Parameter | nanochat default | Our value | Engineering rationale |
|---|---|---|---|
| `--depth` (the "one dial") | 20 | **4** | Downscale to fit a CPU with no GPU and a 45s-per-process ceiling. Like running the integration suite against a 1/50th-size fixture so it finishes in CI. Honest scale caveat, not a result. |
| aspect ratio / width | 64 → dim 1280 | 32 → **dim 128** | Same downscale; keeps the model ~5M params so a step is <0.5s on 4 cores. |
| vocab size | ~65k | **8,192** | Smaller symbol table for a smaller corpus; fewer enum entries to fit. |
| sequence length | 2048 | **512** | Shorter context = cheaper step. The digit-coupling signal is local, so we don't need long context to see it — we sized the fixture to the hypothesis. |
| window pattern | SSSL (sliding) | **L (full)** | The optimized sliding-window attention kernel needs a GPU; on CPU we fall back to the simple full-attention path. Correctness over speed, both arms identical. |
| training horizon | data:param ratio 12 | **500 steps (2.05M tokens)** | Fixed, identical step budget for both arms — a fixed benchmark duration. Below the "compute-optimal" ratio, but we only need enough training for the local statistics to be learned, which happens early. |

### 3c. Two parameters we deliberately overrode from nanochat's auto-derivation

These are the interesting engineering calls — places where the framework's "one dial"
auto-configuration, tuned for big GPU runs, produced a wrong value at our tiny scale, so we
overrode it. Both overrides are applied identically to both arms.

**Learning rate multiplier: implicit 1.0 → 16.0 (calibrated, not guessed).**
nanochat derives the LR from a rule tuned at depth-12 on GPUs. At depth-4 that rule
under-shoots badly. Rather than pick a number, we did what you'd do for any perf constant: a
quick sweep on the *baseline arm only* (so the choice can't bias the comparison), reading the
60-step training loss:

| LR multiplier | loss @ step 60 | Engineering reading |
|---|---|---|
| ×1 (framework default) | 7.25 | Controller gain too low — converging, but far too slowly to finish in our step budget. |
| ×4 | 6.12 | Better. |
| **×16 (chosen)** | **5.71** | Best. Fastest stable descent. |
| ×32 | 5.90 | Past the sweet spot — gain so high the loop starts to overshoot and lose ground. |

The ×1→×16→×32 curve is a textbook control-loop tuning result: raise the gain until just
before it destabilizes, then stop. We locked ×16 for *both* arms. **Engineering lesson: a
framework's cascade of auto-derived defaults is a preset tuned for one operating point; verify
it empirically when you run far outside that point instead of trusting the cascade.**

**Muon weight decay: auto-scaled ~6.0 → 0.0.**
nanochat scales weight decay by a rule (`λ·√(B/B_ref)·(D_ref/D)`) that assumes a large token
budget `D`. At our tiny `D`, the `D_ref/D` term blows the effective decay up to ~6.0 — an
absurd regularization pressure that would crush the weights toward zero faster than the model
can learn anything. So we set it to 0. In engineering terms: an auto-tuning formula divided by
a quantity that is near-zero in our regime and produced a nonsense constant; we clamped it to a
safe default rather than ship the overflow. This is itself a small datapoint on *where the
"one-dial" abstraction leaks* — the same class of finding the experiment set out to look for.

## Part 4 — the one-paragraph takeaway for engineers

We shipped two builds of a 5M-parameter next-token predictor that differ by exactly one
field-level data transformation (randomize digit *values*, keep their *shape*), and scored both
against one pinned test suite. Corrupting the semantics of numbers made numbers themselves
unpredictable (expected) and made the *single* token immediately after a number ~4.6% harder to
predict (a real, tight coupling) — but had **no measurable effect two tokens away or beyond**.
The system had far better locality than we hypothesized: at this scale the model encodes numbers
as local `value→next` lookups, not as inputs to long-range reasoning, so the blast radius of the
corruption was one hop. Along the way, two of the framework's auto-derived "one-dial"
hyperparameters (learning rate, weight decay) were wrong by large factors at our small operating
point and had to be overridden — a reminder that cascading config presets are calibrated for a
specific scale and leak when you run outside it.
