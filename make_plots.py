"""
Plots for the digit-scramble experiment.
Reads runs/*/log.csv and results/*_eval.json, writes PNGs into results/.
Usage: python make_plots.py --runs-dir runs --results-dir results
"""
import os
import csv
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--runs-dir", type=str, default="runs")
parser.add_argument("--results-dir", type=str, default="results")
args = parser.parse_args()

ARMS = ["baseline", "scrambled"]
COLORS = {"baseline": "#1f77b4", "scrambled": "#d62728"}

# 1) training loss curves
fig, ax = plt.subplots(figsize=(7, 4.5))
for arm in ARMS:
    path = os.path.join(args.runs_dir, arm, "log.csv")
    steps, losses = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"])); losses.append(float(row["loss"]))
    # light smoothing (EMA) for readability
    ema, sm = None, []
    for l in losses:
        ema = l if ema is None else 0.9 * ema + 0.1 * l
        sm.append(ema)
    ax.plot(steps, sm, label=f"{arm} (EMA)", color=COLORS[arm])
    ax.plot(steps, losses, alpha=0.15, color=COLORS[arm])
ax.set_xlabel("step"); ax.set_ylabel("train loss (nats/token)")
ax.set_title("Training loss: baseline vs digit-scrambled pretraining data")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(args.results_dir, "train_loss.png"), dpi=150)

# 2) bpb by token class
evals = {}
for arm in ARMS:
    with open(os.path.join(args.results_dir, f"{arm}_eval.json")) as f:
        evals[arm] = json.load(f)

classes = ["digit", "d1", "d2", "d3", "d4", "d5_8", "far", "near_1_8", "overall"]
labels = ["digit\ntokens", "dist 1", "dist 2", "dist 3", "dist 4", "dist 5-8", "far\n(>8)", "near\n(1-8)", "overall"]
x = range(len(classes))
w = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
for i, arm in enumerate(ARMS):
    vals = [evals[arm]["bpb"][c] for c in classes]
    ax.bar([xi + (i - 0.5) * w for xi in x], vals, width=w, label=arm, color=COLORS[arm])
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("validation bits per byte (clean val set)")
ax.set_title("val bpb by target-token distance from nearest preceding digit token")
ax.axhline(3.3219, ls="--", c="gray", lw=1)
ax.text(0.02, 3.35, "log2(10) = 3.32 (uniform digits)", fontsize=8, color="gray")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(args.results_dir, "bpb_by_class.png"), dpi=150)

# 3) delta bpb (scrambled - baseline) by distance
fig, ax = plt.subplots(figsize=(7, 4.5))
dcls = ["digit", "d1", "d2", "d3", "d4", "d5_8", "far"]
deltas = [evals["scrambled"]["bpb"][c] - evals["baseline"]["bpb"][c] for c in dcls]
ax.bar(range(len(dcls)), deltas, color=["#d62728"] + ["#ff9896"] * 5 + ["#7f7f7f"])
ax.set_xticks(range(len(dcls)))
ax.set_xticklabels(["digit", "1", "2", "3", "4", "5-8", "far"])
ax.set_xlabel("target distance from nearest preceding digit token")
ax.set_ylabel("Δ bpb (scrambled − baseline)")
ax.set_title("Damage fingerprint of digit-scrambled pretraining")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(args.results_dir, "delta_bpb.png"), dpi=150)

# 4) probe digit entropies
fig, ax = plt.subplots(figsize=(8, 4.5))
probes = list(evals["baseline"]["probes"].keys())
for i, arm in enumerate(ARMS):
    ents = [evals[arm]["probes"][p]["digit_entropy_bits"] for p in probes]
    ax.bar([j + (i - 0.5) * w for j in range(len(probes))], ents, width=w, label=arm, color=COLORS[arm])
maxent = evals["baseline"]["probes"][probes[0]]["max_digit_entropy_bits"]
ax.axhline(maxent, ls="--", c="gray", lw=1)
ax.text(0.02, maxent + 0.05, "uniform over digit tokens", fontsize=8, color="gray")
ax.set_xticks(range(len(probes)))
ax.set_xticklabels([p if len(p) < 22 else p[:20] + "…" for p in probes], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("entropy of next-token digit distribution (bits)")
ax.set_title("Digit-continuation entropy on numeric probe prompts")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(args.results_dir, "probe_entropy.png"), dpi=150)

print("wrote plots to", args.results_dir)
