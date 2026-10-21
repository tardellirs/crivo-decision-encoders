#!/bin/bash
# E6: latency/memory, one request at a time, GPU not shared. 1,000 fixed requests (seeded shuffle of the E4 sample).
E=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; R=$HOME/results/e6; PY=${PY:-$HOME/venv/bin/python}; mkdir -p $R
export HF_HUB_OFFLINE=1 PYTHONPATH=${KIT:-$HOME/decision-index}:$E/a1a2
[ -s $HOME/req/lat1000.jsonl ] || $PY -c "
import json,random; L=open('$HOME/e4/req/orig.jsonl').readlines(); random.Random(7).shuffle(L); open('$HOME/req/lat1000.jsonl','w').writelines(L[:1000])"
N=${LAT_N:-1000}
for m in ${E6_CRIVOS:-17m 32m 68m 150m 400m 1b}; do
  [ -s $R/crivo-$m.gpu.json ] || $PY $E/e6/latency.py --engine crivo --opt path=tardellirs/crivo-$m --requests $HOME/req/lat1000.jsonl --limit $N --out $R/crivo-$m.gpu.json
done
[ -n "${E6_SKIP_BASELINES:-}" ] && exit 0
[ -s $R/reranker-32m.gpu.json ] || $PY $E/e6/latency.py --engine reranker --opt model=cross-encoder/ettin-reranker-32m-v1 --requests $HOME/req/lat1000.jsonl --limit $N --out $R/reranker-32m.gpu.json
[ -s $R/nli-deberta.gpu.json ] || $PY $E/e6/latency.py --engine nli --opt model=MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --opt fp16=1 --requests $HOME/req/lat1000.jsonl --limit $N --out $R/nli-deberta.gpu.json
[ -s $R/letter-qwen06.gpu.json ] || $PY $E/e6/latency.py --engine letter --opt model=Qwen/Qwen3-0.6B --requests $HOME/req/lat1000.jsonl --limit $N --out $R/letter-qwen06.gpu.json
# A1 cost check: the 32M-v2 weights read per option (one pass per option) vs jointly, same requests
[ -s $R/crivo-32m-peropt.gpu.json ] || $PY $E/e6/latency.py --engine peropt:PerOptionEngine --opt path=$(ls -d $HOME/.cache/huggingface/hub/models--tardellirs--crivo-32m/snapshots/c14221dd*) --requests $HOME/req/lat1000.jsonl --limit $N --out $R/crivo-32m-peropt.gpu.json
