"""
Digit-decomposed bits-per-byte evaluation on the (clean, shared) validation set.

Extends nanochat's evaluate_bpb (nanochat/loss_eval.py) with a token-class
decomposition. Every non-special target token is assigned to one class by its
distance from the most recent digit-bearing token in its left context:
  dist 0            -> the target itself contains an ASCII digit ("digit")
  dist 1,2,3,4,5-8  -> non-digit target, nearest digit token N positions back
  dist >8 / none    -> "far" (no digit within 8 tokens of left context)
bpb per class = sum(nats) / (ln2 * sum(target token bytes)), exactly as in
evaluate_bpb; special tokens (byte length 0) are excluded and never count as
digit context. Also dumps next-token digit distributions for numeric probe
prompts (does the scrambled-arm model predict uniform digits?).

Usage:
  python eval_decomposed.py --base-dir /tmp/ds/baseline --ckpt runs/baseline/ckpt.pt \
      --out results/baseline_eval.json --eval-batches 60
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
parser.add_argument("--base-dir", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--eval-batches", type=int, default=55)  # 55 * 8 * 512 ≈ 225k target tokens (< one val epoch)
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--max-seq-len", type=int, default=512)
parser.add_argument("--near-max", type=int, default=8)
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

# vocab-level masks
DIGSET = set(b"0123456789")
is_digit_token = torch.zeros(vocab_size, dtype=torch.bool)
for tid in range(vocab_size):
    if token_bytes[tid] > 0 and any(b in DIGSET for b in tokenizer.decode_single_token_bytes(tid)):
        is_digit_token[tid] = True
print(f"digit-bearing vocab entries: {is_digit_token.sum().item()}/{vocab_size}")

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
state = ckpt["model"]
n_embd = state["transformer.wte.weight"].shape[1]
n_layer = len(set(k.split(".")[2] for k in state if k.startswith("transformer.h.")))
head_dim = 64
config = GPTConfig(sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=n_layer,
                   n_head=n_embd // head_dim, n_kv_head=n_embd // head_dim, n_embd=n_embd,
                   window_pattern="L")
model = GPT(config)
model.init_weights()
model.load_state_dict(state, strict=True)
model.eval()
print(f"loaded {args.ckpt} (step {ckpt.get('step')})")

val_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device="cpu")

BUCKETS = ["digit", "d1", "d2", "d3", "d4", "d5_8", "far"]
nats = {b: 0.0 for b in BUCKETS}
byts = {b: 0 for b in BUCKETS}

@torch.no_grad()
def run():
    for it in range(args.eval_batches):
        x, y = next(val_loader)
        loss2d = model(x, y, loss_reduction="none").view(x.shape[0], -1)  # (B, T)
        ydig = is_digit_token[y]  # (B, T) target contains digit
        ybytes = token_bytes[y]   # (B, T)
        B, T = y.shape
        # distance from most recent digit-bearing *target* token strictly before t
        # (equivalently: digit token in the input context x[1..t], since y[t-1] == x[t])
        for b in range(B):
            last_dig = -10**9
            for t in range(T):
                nb = int(ybytes[b, t])
                if nb > 0:  # special tokens excluded from metric AND from context distance
                    l = float(loss2d[b, t])
                    if bool(ydig[b, t]):
                        bucket = "digit"
                    else:
                        d = t - last_dig
                        bucket = ("d1" if d == 1 else "d2" if d == 2 else "d3" if d == 3 else
                                  "d4" if d == 4 else "d5_8" if d <= args.near_max else "far")
                    nats[bucket] += l
                    byts[bucket] += nb
                if bool(ydig[b, t]):
                    last_dig = t
        if (it + 1) % 10 == 0:
            print(f"eval batch {it+1}/{args.eval_batches}")

run()

result = {"ckpt": args.ckpt, "step": ckpt.get("step"), "eval_batches": args.eval_batches, "bpb": {}, "bytes": {}}
tot_n, tot_b = 0.0, 0
for b in BUCKETS:
    result["bpb"][b] = nats[b] / (math.log(2) * byts[b]) if byts[b] > 0 else None
    result["bytes"][b] = byts[b]
    tot_n += nats[b]; tot_b += byts[b]
result["bpb"]["overall"] = tot_n / (math.log(2) * tot_b)
near_n = sum(nats[k] for k in ["d1", "d2", "d3", "d4", "d5_8"])
near_b = sum(byts[k] for k in ["d1", "d2", "d3", "d4", "d5_8"])
result["bpb"]["near_1_8"] = near_n / (math.log(2) * near_b)
result["bytes"]["near_1_8"] = near_b

# ---- digit-continuation probes ----
probes = ["The year was 19", "He scored 2", "The company reported profits of $1",
          "on Sunday, September 1", "at the age of 6", "The Dow rose 1"]
probe_out = {}
digit_ids = [i for i in range(vocab_size) if is_digit_token[i]]
with torch.no_grad():
    for p in probes:
        ids = tokenizer.encode(p, prepend="<|bos|>")
        logits = model(torch.tensor([ids]))[0, -1]
        probs = torch.softmax(logits, dim=-1)
        dp = probs[digit_ids]
        dp = dp / dp.sum()
        ent = float(-(dp * (dp + 1e-12).log()).sum() / math.log(2))
        top = torch.topk(probs, 5)
        probe_out[p] = {
            "digit_mass": float(probs[digit_ids].sum()),
            "digit_entropy_bits": ent,
            "max_digit_entropy_bits": math.log2(len(digit_ids)),
            "top5": [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)],
        }
result["probes"] = probe_out

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result["bpb"].items()}, indent=2))
print(f"wrote {args.out}")
