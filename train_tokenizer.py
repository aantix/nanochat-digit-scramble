"""
Train the BPE tokenizer for the digit-scramble experiment.

Mirrors scripts/tok_train.py (rustbpe train -> tiktoken encoding -> token_bytes.pt),
but trains on OUR corpus's clean training documents and installs the SAME tokenizer
into both arms' base dirs — the tokenizer is a held-fixed constant, so tokenization
boundaries are identical across arms and val bpb is directly comparable.

Usage: python train_tokenizer.py --out-root /tmp/ds --vocab-size 8192
"""
import os
import sys
import shutil
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanochat-src"))

import torch
import pyarrow.parquet as pq
from nanochat.tokenizer import RustBPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--out-root", type=str, required=True)
parser.add_argument("--vocab-size", type=int, default=8192)
parser.add_argument("--doc-cap", type=int, default=10_000)  # same as tok_train.py
args = parser.parse_args()

train_shard = os.path.join(args.out_root, "baseline", "base_data_climbmix", "shard_00000.parquet")

def text_iterator():
    pf = pq.ParquetFile(train_shard)
    for rg_idx in range(pf.num_row_groups):
        for doc in pf.read_row_group(rg_idx).column("text").to_pylist():
            yield doc[:args.doc_cap]

tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), args.vocab_size)

# token_bytes table, exactly as in scripts/tok_train.py
vocab_size = tokenizer.get_vocab_size()
special_ids = set(tokenizer.encode_special(s) for s in tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    if token_id in special_ids:
        token_bytes.append(0)
    else:
        token_bytes.append(len(tokenizer.decode_single_token_bytes(token_id)))
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")

for arm in ["baseline", "scrambled"]:
    tok_dir = os.path.join(args.out_root, arm, "tokenizer")
    tokenizer.save(tok_dir)
    with open(os.path.join(tok_dir, "token_bytes.pt"), "wb") as f:
        torch.save(token_bytes, f)
    print(f"installed tokenizer -> {tok_dir}")

# quick sanity: digits are chunked <=2 per pre-token by SPLIT_PATTERN
enc = tokenizer.encode("In 1969, Apollo 11 landed. Scores: 21-14, price $1,234.56")
print("sanity:", tokenizer.decode(enc))
digit_tokens = sum(1 for t in enc if any(ch.isdigit() for ch in tokenizer.decode([t])))
print(f"digit-bearing tokens in sanity string: {digit_tokens}")
