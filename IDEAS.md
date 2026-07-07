# Phase 2 — Experiment idea slate (16 ideas)

Format per idea: Hypothesis (falsifiable) / Variable (single) / Metric / Predicted surprise.

**I1. Digit-scramble pretraining fingerprint.**
H: Replacing every digit in the *pretraining* text with a uniformly random digit (format preserved, eval data untouched) raises val bpb not only on digit bytes but measurably on *non-digit* tokens in a window after numbers (agreement/coherence effects), while leaving far-from-digit text unchanged.
V: digit randomization in training text only. M: val_bpb decomposed into digit tokens / non-digit tokens within k tokens after a digit / rest. S: either "numbers are ignorable noise slots" (near-digit prose unharmed — surprising) or a quantified halo of damage around numbers (mechanism: plural/date/magnitude agreement).

**I2. Digit split pattern {1,2} vs {1,3} downstream.**
H: The SPLIT_PATTERN digit-chunk choice (nanochat deviates from GPT-4: \p{N}{1,2}) changes trained-model bpb on numeric text beyond what compression stats predict.
V: split pattern digit clause. M: val_bpb on numeric vs non-numeric bytes at fixed FLOPs. S: compression-equivalent tokenizers might diverge after training.

**I3. Vocab size at fixed FLOPs at micro-depth.**
H: At d4, where lm_head dominates FLOPs/token, a 4× smaller vocab wins val_bpb at matched FLOPs despite worse compression.
V: vocab size. M: val_bpb (bytes-normalized → cross-tokenizer fair). S: the one-dial abstraction silently makes tiny models vocab-compute-bound.

**I4. Best-fit packing vs legacy cropping.**
H: BOS-aligned best-fit packing (35% tokens discarded) beats the legacy loader at fixed steps but loses at fixed *data* budget.
V: dataloader packing. M: val_bpb at both budgets. S: the 35% discard could flip the verdict under data scarcity.

**I5. QK sharpening (q,k ×1.2) ablation at micro scale.**
H: Removing the fixed 1.2 sharpening hurts d4 less than d20 (attention sharpness matters more with depth).
V: sharpening constant. M: val_bpb. S: a hardcoded magic number with depth-dependent value.

**I6. Backout ablation.** H: subtracting 0.2×mid-residual before the head is neutral at micro scale. V: backout on/off. M: val_bpb. S: feature only pays off at scale.

**I7. Smear ablation.** H: previous-token smear gate is worth more at tiny dims (bigram info is scarcer). V: smear on/off. M: val_bpb. S: inverse scaling of a micro-feature.

**I8. Value-embedding (ResFormer) ablation at d4.** H: VE layers dominate micro-scale gains (embeddings are most of the model). V: VE on/off. M: val_bpb per FLOP. S: at d4 VE ≈ a second embedding table—may carry the model.

**I9. Break the LR·1/√dmodel coupling at d2–d4.**
H: The AdamW LR transfer rule tuned at d12 is off by >2× at the bottom of the miniseries.
V: LR multiplier sweep. M: val_bpb. S: the one-dial promise breaks exactly where hobbyists use it.

**I10. Special-token embedding initialization (cross-stage).**
H: The 9 chat special tokens never receive gradient in pretraining (they never occur in text), so they enter SFT as frozen random N(0,0.8) vectors; re-initializing them to the mean embedding before SFT speeds early SFT convergence.
V: special-token init at SFT start. M: SFT loss curve/val. S: a silent pretrain→SFT coupling baked into every nanochat run.

**I11. Seed-variance noise floor of the micro-pipeline.**
H: At d4-scale, run-to-run val_bpb variance exceeds typical ablation effect sizes reported at this scale.
V: seed only (n=4). M: val_bpb spread. S: many micro-ablations may be noise.

**I12. Document-length curriculum.** H: feeding short docs first (vs shuffled) improves early convergence but not final bpb. V: doc order. M: bpb curve. S: packing interacts with curriculum.

**I13. Duplicate-document memorization threshold at micro scale.** H: a doc repeated ≥N times becomes verbatim-extractable even in a 1M-token run; N is small (~10). V: duplication count of one planted doc. M: extraction rate / doc loss. S: memorization cliff visible at toy scale.

**I14. Case-randomization robustness.** H: randomly lowercasing 10% of training docs costs ~0 clean bpb but halves bpb degradation on all-lowercase eval text. V: augmentation on/off. M: bpb clean vs lowercased eval. S: free robustness.

**I15. Logit softcap 15 at micro scale.** H: softcap is neutral-to-harmful at d4 (tiny logits never reach ±15). V: softcap on/off. M: val_bpb + logit norm stats. S: dead knob detection.

**I16. Gate-channel double-duty fingerprint.**
H: Because ve_gate reads only embedding channels 0–12 and smear_gate channels 0–24, trained wte develops statistically distinct behavior in those channels (variance/importance), and rewiring gates to read *random fixed* channels changes nothing else.
V: which channels gates read. M: per-channel wte variance profile + val_bpb. S: a quirky implementation detail imprints structure on the embedding table.

## Feasibility notes (this environment)
CPU-only micro runs (d4-ish, seq 256–512, 300–600 steps, ~1–2M tokens), chunked training with checkpoint/resume every ≤45s. Ideas needing HF-hub data (I10 partially: SFT data can be synthesized locally) or many runs (I9: 5+, I11: 4) are costlier; 2-run designs (I1, I2, I4–I8, I14–I16) preferred.
