# Publishing (blocked in sandbox — one command left for you)

This automated run had no GitHub credentials (no `gh`, no token; sandbox proxy also blocks
credential flows), so Phase 6 could not push. The repo is fully committed locally
(`git log` → "Initial experiment: digit-scramble pretraining fingerprint on nanochat").

From this directory on a machine with `gh` authenticated as **aantix**:

```bash
gh repo create aantix/nanochat-digit-scramble --public --source . --push \
  --description "Scrambling every digit in pretraining data damages prediction of the surrounding non-digit prose — but the damage halo is only one token deep."
```

(Checkpoints in `runs/*.pt` are gitignored; everything else — code, logs, eval JSONs,
plots, phase docs — is committed.)
