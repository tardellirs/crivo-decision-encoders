#!/bin/bash
# Retry requests that failed with CUDA OOM while 3 lanes shared the GPU (run_requests.py retries status=error ids;
# scorers key by id, last line wins). Smaller batches so this fits next to lane1.
export HF_HUB_OFFLINE=1 PYTHONPATH=${KIT:-$HOME/decision-index}
PY=${PY:-$HOME/venv/bin/python}; E=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; R=$HOME/results
st(){ echo "$(date -u +%FT%TZ) $*" >> $R/STATUS; }
st "retry start"
for v in orig perm1 perm2 perm3 rev yn; do
  $PY $E/common/run_requests.py --engine crivo --opt path=tardellirs/crivo-1b --requests $HOME/e4/req/$v.jsonl --out $R/e4/crivo-1b/$v.jsonl --threads 4 --batch 32 >> $R/logs/retry.log 2>&1
done
$PY $E/common/run_requests.py --engine letter --opt model=Qwen/Qwen3-0.6B --opt rotations=4 --requests $HOME/req/s1_60.jsonl --out $R/s1/letter-qwen3-0.6b-rot4.jsonl --threads 4 --batch 4 >> $R/logs/retry.log 2>&1
$PY $E/common/run_requests.py --engine reranker --opt model=cross-encoder/ettin-reranker-32m-v1 --requests $HOME/req/s1_all.jsonl --out $R/s1/reranker-32m-raw.jsonl --threads 4 --batch 0 >> $R/logs/retry.log 2>&1
st "retry done"
