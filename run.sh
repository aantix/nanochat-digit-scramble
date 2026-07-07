#!/usr/bin/env bash
# Full reproduction of the digit-scramble experiment.
# Requirements: python3 with torch (CPU is enough), pyarrow, rustbpe, tiktoken, matplotlib.
# The AG News CSVs: git clone --depth 1 https://github.com/mhjabreel/CharCnn_Keras /tmp/agnews
#
# The ONLY difference between the two arms is the digit-scramble applied to the
# training shard in prepare_data.py. Same tokenizer, same clean val shard, same
# seeds, same schedules, same step count.
set -e
cd "$(dirname "$0")"

AGNEWS=${AGNEWS:-/tmp/agnews/data/ag_news_csv}
DSROOT=${DSROOT:-/tmp/ds}
ITERS=${ITERS:-500}
LR_MULT=${LR_MULT:-16.0}  # calibrated once on the baseline arm (see README: LR calibration)

# 1) data: two arms, identical except digits in the train shard
python3 prepare_data.py --agnews-dir "$AGNEWS" --out-root "$DSROOT"

# 2) one shared tokenizer (trained on clean train text), installed into both arms
python3 train_tokenizer.py --out-root "$DSROOT" --vocab-size 8192

# 3) train both arms (chunk-resumable: rerun until exit code 0; exit 3 = keep going)
for ARM in baseline scrambled; do
  while true; do
    set +e
    python3 train_chunked.py --base-dir "$DSROOT/$ARM" --out-dir "runs/$ARM" \
        --num-iterations "$ITERS" --lr-mult "$LR_MULT" --max-seconds 30
    CODE=$?
    set -e
    if [ "$CODE" -eq 0 ]; then break; fi
    if [ "$CODE" -ne 3 ]; then exit "$CODE"; fi
  done
done

# 4) digit-decomposed evaluation on the shared clean val set (+ snapshots for curves)
mkdir -p results
for ARM in baseline scrambled; do
  python3 eval_decomposed.py --base-dir "$DSROOT/$ARM" --ckpt "runs/$ARM/ckpt.pt" \
      --out "results/${ARM}_eval.json"
  for SNAP in runs/$ARM/snapshot_*.pt; do
    [ -e "$SNAP" ] || continue
    STEP=$(basename "$SNAP" .pt | cut -d_ -f2)
    python3 eval_decomposed.py --base-dir "$DSROOT/$ARM" --ckpt "$SNAP" \
        --out "results/${ARM}_eval_step${STEP}.json" --eval-batches 25
  done
done

# 5) paired per-token significance test (both models on the identical val stream)
python3 compare_paired.py --base-dir "$DSROOT/baseline" \
    --ckpt-a runs/baseline/ckpt.pt --ckpt-b runs/scrambled/ckpt.pt \
    --out results/paired_delta.json

# 6) plots
python3 make_plots.py --runs-dir runs --results-dir results
echo "done. see results/"
