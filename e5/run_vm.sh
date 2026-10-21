#!/usr/bin/env bash
# E5 calibration + public-suite error analysis driver (CPU; 16 vCPU used). Order as run for the paper.
# Wall time: extract ~1 min, kit rescore ~1 min/run, calib.py ~2 min (B=1000 x 2 weightings), errors.py ~2 min (B=500).
# Env:
#   RUNS  one folder per system with the kit's public-suite output (results.jsonl.gz + index.json), named
#         c17m c32v1 c32v2 c68m c150v1 c150v2 c400m c1b prumo (the released runs are in the HF dataset
#         tardellirs/decision-index-results; the two v1 runs are the earlier releases of crivo-32m / crivo-150m)
#   KIT   Decision Index kit checkout with suite-0.3 built (default ~/decision-index)
#   WORK  output directory (default ~/e5)
#   PY    python with the kit installed (default ~/venv/bin/python)
set -euo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R=${RUNS:?set RUNS}; KIT=${KIT:-$HOME/decision-index}; W=${WORK:-$HOME/e5}; PY=${PY:-$HOME/venv/bin/python}
mkdir -p $W/rescore
TOK=$($PY -c "from huggingface_hub import snapshot_download as s; print(s('tardellirs/crivo-17m', allow_patterns=['tokenizer*', '*.json']))")   # Ettin tokenizer (same for all sizes)
cd $EXP/e5
$PY extract.py --suite $KIT/suite-0.3 --tok $TOK --out $W/tab \
  c17m=$R/c17m c32v1=$R/c32v1 c32v2=$R/c32v2 c68m=$R/c68m c150v1=$R/c150v1 c150v2=$R/c150v2 \
  c400m=$R/c400m c1b=$R/c1b prumo=$R/prumo 2>&1 | tee $W/extract.log
# pipeline check: the official kit re-scores every run to its published public index
for m in c17m c32v1 c32v2 c68m c150v1 c150v2 c400m c1b prumo; do
  $PY -m decision_index score --edition 0.3 --suite-dir $KIT/suite-0.3 \
    --results $R/$m/results.jsonl.gz --out $W/rescore/$m > $W/rescore/$m.log 2>&1
done
$PY calib.py --tab $W/tab --board $EXP/board/di-index-20261020.json \
  --board032 $EXP/board/di-index-v0.3.2-20261020.json --out $W/calib.json
cd $EXP/errors
$PY errors.py --tab $W/tab --runs $R --out $W/errors.json
$PY sample_errors.py --tab $W/tab --suite $KIT/suite-0.3 --per 2 > $W/sample68m.txt
