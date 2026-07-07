"""
Chunk-resumable CPU pretraining driver for the digit-scramble experiment.

Faithfully replicates scripts/base_train.py (same model, optimizer split, LR /
Muon-momentum schedules, BOS-aligned best-fit dataloader) while reusing the
nanochat modules unmodified. Two departures, forced by this sandbox and applied
IDENTICALLY to both arms (see README):
  1. Chunked execution: the sandbox kills processes after ~45 s, so we save a
     checkpoint every --save-every steps and on resume we fast-forward the
     dataloader from scratch by consuming `step` batches. Packing is
     deterministic, so the token stream is EXACTLY the one an unchunked run
     would see — chunking cannot influence the result.
  2. Muon weight decay is set to 0: base_train's T_epoch-scaled rule
     (wd·sqrt(B/B_ref)·(D_ref/D)) explodes to ~6.0 at this micro horizon.
  torch.compile is disabled via TORCH_COMPILE_DISABLE=1 (CPU inductor + 45s
  process lifetime don't mix); optim.py's @torch.compile decorators become no-ops.

Usage (repeat until exit code 0; exit code 3 = progress made, more remains):
  python train_chunked.py --base-dir /tmp/ds/baseline --out-dir runs/baseline \
      --num-iterations 300 --max-seconds 30
"""
import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import csv
import math
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanochat-src"))

parser = argparse.ArgumentParser(description="Chunked CPU pretraining (digit-scramble experiment)")
parser.add_argument("--base-dir", type=str, required=True, help="NANOCHAT_BASE_DIR for this arm (data+tokenizer)")
parser.add_argument("--out-dir", type=str, required=True, help="where checkpoints/logs go")
# model (defaults = the micro config used in the experiment)
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--aspect-ratio", type=int, default=32)
parser.add_argument("--head-dim", type=int, default=64)
parser.add_argument("--max-seq-len", type=int, default=512)
parser.add_argument("--window-pattern", type=str, default="L")  # SDPA/CPU: full context (base_train's CPU guidance)
# training
parser.add_argument("--num-iterations", type=int, default=500)
parser.add_argument("--device-batch-size", type=int, default=8)  # 8*512 = 4096 = total batch (no grad accum)
parser.add_argument("--embedding-lr", type=float, default=0.3)
parser.add_argument("--unembedding-lr", type=float, default=0.008)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--scalar-lr", type=float, default=0.5)
parser.add_argument("--lr-mult", type=float, default=1.0, help="global LR multiplier (calibrated once, shared by both arms)")
parser.add_argument("--warmup-steps", type=int, default=40)
parser.add_argument("--warmdown-ratio", type=float, default=0.65)
parser.add_argument("--final-lr-frac", type=float, default=0.05)
parser.add_argument("--seed", type=int, default=42)
# chunking
parser.add_argument("--max-seconds", type=float, default=30.0)
parser.add_argument("--save-every", type=int, default=10, help="checkpoint every N steps")
parser.add_argument("--snapshot-at", type=str, default="150,300", help="save model-only snapshots at these steps")
args = parser.parse_args()

os.environ["NANOCHAT_BASE_DIR"] = args.base_dir  # must be set before nanochat imports
t_start = time.time()

import torch
torch.set_num_threads(4)

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

os.makedirs(args.out_dir, exist_ok=True)
ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
log_path = os.path.join(args.out_dir, "log.csv")
snapshot_steps = set(int(s) for s in args.snapshot_at.split(",") if s)

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

# ---- model (same construction math as base_train.build_model_meta) ----
base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
config = GPTConfig(
    sequence_len=args.max_seq_len, vocab_size=vocab_size,
    n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
    window_pattern=args.window_pattern,
)
torch.manual_seed(args.seed)
model = GPT(config)  # small enough to skip the meta-device dance
model.init_weights()

# ---- optimizer (same scaling rules as base_train, minus the exploding wd rule) ----
total_batch_size = args.device_batch_size * args.max_seq_len
B_REF = 2**19
batch_lr_scale = (total_batch_size / B_REF) ** 0.5
m = args.lr_mult
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale * m,
    embedding_lr=args.embedding_lr * batch_lr_scale * m,
    scalar_lr=args.scalar_lr * batch_lr_scale * m,
    matrix_lr=args.matrix_lr * batch_lr_scale * m,
    weight_decay=0.0,
)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

# ---- schedules (verbatim math from base_train.py) ----
num_iterations = args.num_iterations

def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# ---- resume ----
step = 0
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]
    del ckpt
    print(f"resumed at step {step}")

if step >= num_iterations:
    print("training complete")
    sys.exit(0)

# ---- dataloader: build fresh and fast-forward `step` batches (exact replay) ----
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device="cpu")
t_ff = time.time()
for _ in range(step):
    next(train_loader)
x, y, dl_state = next(train_loader)
print(f"fast-forwarded {step} batches in {time.time()-t_ff:.1f}s")

def save_ckpt():
    tmp = ckpt_path + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "config": vars(args)}, tmp)
    os.replace(tmp, ckpt_path)

# ---- train ----
if not os.path.exists(log_path):
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "lrm", "dt"])
log_f = open(log_path, "a", newline="")
log_w = csv.writer(log_f)

model.train()
while step < num_iterations:
    t0 = time.time()
    loss = model(x, y)
    loss.backward()
    lrm = get_lr_multiplier(step)
    mom = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = mom
    optimizer.step()
    model.zero_grad(set_to_none=True)
    loss_f = loss.item()
    x, y, dl_state = next(train_loader)
    step += 1
    dt = time.time() - t0
    log_w.writerow([step, f"{loss_f:.6f}", f"{lrm:.4f}", f"{dt:.3f}"])
    print(f"step {step:04d}/{num_iterations} | loss {loss_f:.4f} | lrm {lrm:.2f} | {dt*1000:.0f}ms")
    if step in snapshot_steps:
        torch.save({"model": model.state_dict(), "step": step},
                   os.path.join(args.out_dir, f"snapshot_{step:06d}.pt"))
    if step % args.save_every == 0 or step == num_iterations:
        save_ckpt()
        log_f.flush()
    if time.time() - t_start > args.max_seconds and step < num_iterations:
        save_ckpt()
        log_f.flush()
        print(f"chunk done at step {step} ({time.time()-t_start:.0f}s elapsed)")
        sys.exit(3)

save_ckpt()
log_f.close()
print(f"training complete at step {step}")
sys.exit(0)
