#!/bin/bash
# Score every finished run: BTZSC (macro-F1, official metric), S1-Arena (its own metrics.score), E4 flips.
E=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; R=$HOME/results; PY=${PY:-$HOME/venv/bin/python}
for f in $R/btzsc/*.jsonl; do s=${f%.jsonl}.score.json; [ -s $s ] || $PY $E/btzsc/adapter.py score --requests $HOME/req/btzsc_full.jsonl --results $f --out $s > /dev/null; done
for f in $R/s1/*.jsonl; do s=${f%.jsonl}.score.json; q=$HOME/req/s1_all.jsonl; [[ $f == *rot4* ]] && q=$HOME/req/s1_60.jsonl; [ -s $s ] || $PY $E/s1arena/adapter.py score --data $HOME/s1arena/data --code $HOME/s1arena/code --requests $q --results $f --out $s > /dev/null; done
for d in $R/e4/*/; do $PY $E/e4/perturb.py flips --rows $HOME/e4/req/orig.jsonl --results $d > /dev/null; done
echo scored
