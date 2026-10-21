#!/bin/bash
# Contamination audit: every benchmark target text against every training-mix family (README "Data and contamination").
# Run from the directory holding the training mixes (mix400_*, mix32_*, long_v3, mix_v*; not distributed).
# Env: REPO (default: this script's parent dir), PY (default ~/venv/bin/python), OUT (default ~/exp/audit),
#      BTZSC_TARGETS / S1_TARGETS (from `btzsc/adapter.py targets` and `s1arena/adapter.py targets`).
set -x
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; C=$REPO/common/contam.py; PY=${PY:-$HOME/venv/bin/python}
O=${OUT:-$HOME/exp/audit}; mkdir -p $O
export PYTHONHASHSEED=0
for T in btzsc s1arena; do
  TG=${BTZSC_TARGETS:-$HOME/btzsc/targets.jsonl}; [ $T = s1arena ] && TG=${S1_TARGETS:-$HOME/s1arena/targets.jsonl}
  [ $T = s1arena ] && $PY $C --targets $TG --train $(ls mix*.gz long_v3.jsonl.gz | grep -v minimix) --out $O/${T}_audit_allmixes.json --workers 15 > $O/${T}_allmixes.log 2>&1
  $PY $C --targets $TG --train mix400_s1.train.jsonl.gz mix400_s2.train.jsonl.gz mix400_s3.train.jsonl.gz mix400_s1.dev.jsonl.gz mix400_s2.dev.jsonl.gz mix400_s3.dev.jsonl.gz --out $O/${T}_audit_v2family.json --workers 15 > $O/${T}_v2family.log 2>&1
  $PY $C --targets $TG --train mix32_s1.train.jsonl.gz mix32_s2.train.jsonl.gz mix32_s1.dev.jsonl.gz long_v3.jsonl.gz --out $O/${T}_audit_32mv1.json --workers 15 > $O/${T}_32mv1.log 2>&1
  $PY $C --targets $TG --train mix_v1.train.jsonl.gz mix_v2c.train.jsonl.gz mix_v3.train.jsonl.gz mix_v1.dev.jsonl.gz mix_v2c.dev.jsonl.gz mix_v3.dev.jsonl.gz --out $O/${T}_audit_150mv1.json --workers 15 > $O/${T}_150mv1.log 2>&1
done
echo ALLDONE
