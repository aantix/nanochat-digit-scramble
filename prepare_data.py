"""
Prepare the pretraining data for the digit-scramble experiment.

Builds two parallel nanochat base dirs that are byte-identical EXCEPT for the
single experimental variable: in the "scrambled" arm, every ASCII digit in the
TRAINING documents is replaced by a uniformly random digit (deterministic rng).
The validation shard is clean and identical in both arms.

Corpus: AG News (127.6k news snippets, title + description), number-rich text.
(ClimbMix, nanochat's default corpus, is unreachable from this sandbox; the
corpus is a held-fixed constant of the experiment, not a variable.)

Usage: python prepare_data.py --agnews-dir /tmp/agnews/data/ag_news_csv --out-root /tmp/ds
Writes: {out-root}/{baseline,scrambled}/base_data_climbmix/shard_0000{0,1}.parquet
"""
import os
import csv
import html
import random
import argparse

import pyarrow as pa
import pyarrow.parquet as pq

parser = argparse.ArgumentParser()
parser.add_argument("--agnews-dir", type=str, required=True)
parser.add_argument("--out-root", type=str, required=True)
parser.add_argument("--val-docs", type=int, default=8000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

DIGITS = "0123456789"

def clean(text):
    # AG News artifacts: '\' stands for a newline; HTML entities; mangled '&#39;'-style
    # entities that lost their '&' ("...s #39; ..." etc.)
    text = text.replace("\\", " ")
    for src, dst in [(" #39;", "'"), (" #36;", "$"), (" quot;", '"'), (" amp;", "&"),
                     ("#39;", "'"), ("quot;", '"')]:
        text = text.replace(src, dst)
    text = html.unescape(text)
    return " ".join(text.split())

def scramble_digits(text, rng):
    # THE intervention: every digit char -> uniform random digit. Format (digit
    # positions, string lengths, pre-tokenization boundaries) is preserved exactly;
    # numeric semantics are destroyed.
    return "".join(rng.choice(DIGITS) if c.isdigit() else c for c in text)

def read_docs(path):
    docs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            _, title, desc = row[0], row[1], row[2]
            docs.append(clean(title) + ". " + clean(desc))
    return docs

docs = read_docs(os.path.join(args.agnews_dir, "train.csv"))
print(f"read {len(docs)} docs")
rng = random.Random(args.seed)
rng.shuffle(docs)
val_docs = docs[-args.val_docs:]
train_docs = docs[:-args.val_docs]
print(f"train: {len(train_docs)} docs, val: {len(val_docs)} docs")

# deterministic scramble, independent rng per document
def scrambled_copy(doc_list):
    out = []
    for i, d in enumerate(doc_list):
        r = random.Random(10_000 + i)
        out.append(scramble_digits(d, r))
    return out

def write_shard(path, doc_list):
    table = pa.table({"text": doc_list})
    pq.write_table(table, path, row_group_size=2000)
    print(f"wrote {path} ({len(doc_list)} docs)")

for arm, tdocs in [("baseline", train_docs), ("scrambled", scrambled_copy(train_docs))]:
    data_dir = os.path.join(args.out_root, arm, "base_data_climbmix")
    os.makedirs(data_dir, exist_ok=True)
    write_shard(os.path.join(data_dir, "shard_00000.parquet"), tdocs)   # train
    write_shard(os.path.join(data_dir, "shard_00001.parquet"), val_docs) # val (clean, identical)

# stats
n_digits = sum(c.isdigit() for d in train_docs for c in d)
n_chars = sum(len(d) for d in train_docs)
print(f"digit chars in train: {n_digits:,} / {n_chars:,} ({100*n_digits/n_chars:.2f}%)")
docs_with_digit = sum(any(c.isdigit() for c in d) for d in train_docs)
print(f"docs containing a digit: {docs_with_digit:,} ({100*docs_with_digit/len(train_docs):.1f}%)")
