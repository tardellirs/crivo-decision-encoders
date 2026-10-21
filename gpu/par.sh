#!/bin/bash
# Run two lane1 jobs in parallel with lane1 (GPU was at ~38%); their .done markers were created first so lane1 skips them.
export HF_HUB_OFFLINE=1 PYTHONPATH=${KIT:-$HOME/decision-index}
PY=${PY:-$HOME/venv/bin/python}; E=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; R=$HOME/results
st(){ echo "$(date -u +%FT%TZ) $*" >> $R/STATUS; }
st "par start s1/nli-bart-large-mnli"
$PY $E/common/run_requests.py --engine nli --opt model=facebook/bart-large-mnli --opt fp16=1 --opt batch_size=64 --opt max_length=1024 --requests $HOME/req/s1_all.jsonl --out $R/s1/nli-bart-large-mnli.jsonl --threads 4 --batch 0 >> $R/logs/s1_nli-bart-large-mnli.log 2>&1 || st "FAIL par s1/nli-bart-large-mnli"
st "par done s1/nli-bart-large-mnli"
mkdir -p $R/e4/reranker-32m-raw
for v in orig perm1 perm2 perm3 rev yn; do
  $PY $E/common/run_requests.py --engine reranker --opt model=cross-encoder/ettin-reranker-32m-v1 --requests $HOME/e4/req/$v.jsonl --out $R/e4/reranker-32m-raw/$v.jsonl --threads 4 --batch 0 >> $R/logs/e4_reranker-32m-raw.log 2>&1 || st "FAIL par e4/reranker-32m-raw $v"
done
st "par done e4/reranker-32m-raw"
touch $R/PAR_DONE
