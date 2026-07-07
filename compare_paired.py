"""
Paired per-token comparison of the two arms on the IDENTICAL validation stream.

Because both arms share the tokenizer and the clean val shard, the packed val
batches are byte-identical, so we can compute per-target-token paired loss
differences d_t = loss_scrambled(t) - loss_baseline(t), bucket them by distance
from the nearest preceding digit token, and report the bucket-level bpb delta
with a paired standard error (SE = sqrt(sum var(d)) / (ln2 * sum bytes)).

Usage:
  python compare_paired.py --base-dir /tmp/ds/baseline \
      --ckpt-a runs/baseline/ckpt.pt --ckpt-b runs/scrambled/ckpt.pt \
      --out results/paired_delta.json
"""
import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import json
import math
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanochat-src"))

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", type=str, required=True, help="any arm's base dir (val shard + tokenizer are shared)")
parser.add_argument("--ckpt-a", type=str, required=True, help="baseline checkpoint")
parser.add_argument("--ckpt-b", type=str, required=True, help="scrambled checkpoint")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--eval-batches", type=int, default=55)
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--max-seq-len", type=int, default=512)
args = parser.parse_args()

os.environ["NANOCHAT_BASE_DIR"] = args.base_dir

import torch
torch.set_num_threads(4)
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device="cpu")
vocab_size = tokenizer.get_vocab_size()

DIGSET = set(b"0123456789")
is_digit_token = torch.zeros(vocab_size, dtype=torch.bool)
for tid in range(vocab_size):
    if token_bytes[tid] > 0 and any(b in DIGSET for b in tokenizer.decode_single_token_bytes(tid)):
        is_digit_token[tid] = True

def load(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    n_embd = state["transformer.wte.weight"].shape[1]
    n_layer = len(set(k.split(".")[2] for k in state if k.startswith("transformer.h.")))
    config = GPTConfig(sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=n_layer,
                       n_head=n_embd // 64, n_kv_head=n_embd // 64, n_embd=n_embd, window_pattern="L")
    model = GPT(config)
    model.init_weights()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

model_a, model_b = load(args.ckpt_a), load(args.ckpt_b)

val_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device="cpu")

BUCKETS = ["digit", "d1", "d2", "d3", "d4", "d5_8", "far"]
sum_d = {b: 0.0 for b in BUCKETS}   # sum of paired nat diffs
sum_d2 = {b: 0.0 for b in BUCKETS}  # sum of squared diffs
cnt = {b: 0 for b in BUCKETS}       # token counts
byts = {b: 0 for b in BUCKETS}      # byte counts

@torch.no_grad()
def run():
    for it in range(args.eval_batches):
        x, y = next(val_loader)
        la = model_a(x, y, loss_reduction="none").view(x.shape[0], -1)
        lb = model_b(x, y, loss_reduction="none").view(x.shape[0], -1)
        ydig = is_digit_token[y]
        ybytes = token_bytes[y]
        B, T = y.shape
        for b in range(B):
            last_dig = -10**9
            for t in range(T):
                nb = int(ybytes[b, t])
                if nb > 0:
                    d = float(lb[b, t]) - float(la[b, t])
                    if bool(ydig[b, t]):
                        bucket = "digit"
                    else:
                        dist = t - last_dig
                        bucket = ("d1" if dist == 1 else "d2" if dist == 2 else "d3" if dist == 3 else
                                  "d4" if dist == 4 else "d5_8" if dist <= 8 else "far")
                    sum_d[bucket] += d
                    sum_d2[bucket] += d * d
                    cnt[bucket] += 1
                    byts[bucket] += nb
                if bool(ydig[b, t]):
                    last_dig = t
        if (it + 1) % 10 == 0:
            print(f"batch {it+1}/{args.eval_batches}")

run()

out = {}
LN2 = math.log(2)
for b in BUCKETS:
    n = cnt[b]
    mean = sum_d[b] / n
    var = sum_d2[b] / n - mean * mean
    se_sum = math.sqrt(var * n)  # SE of the sum of iid-ish diffs
    delta_bpb = sum_d[b] / (LN2 * byts[b])
    se_bpb = se_sum / (LN2 * byts[b])
    out[b] = {"tokens": n, "bytes": byts[b], "delta_bpb": delta_bpb, "se_bpb": se_bpb,
              "z": delta_bpb / se_bpb if se_bpb > 0 else None}
with open(args.out, "w") as f:
    json.dump(out, f, indent=2)
for b in BUCKETS:
    o = out[b]
    print(f"{b:6s} n={o['tokens']:7d} Δbpb={o['delta_bpb']:+.4f} ± {o['se_bpb']:.4f} (z={o['z']:.1f})")
