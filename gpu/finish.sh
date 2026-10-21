#!/bin/bash
# After run_lanes.sh prints ALL DONE: (1) redo E6 entries that were measured while another job shared the GPU
# (crivo 17m-400m) or failed with OOM (crivo-1b, reranker), GPU alone; (2) finish e4/reranker-32m-raw (resumable);
# (3) rescore. Marker: ~/results/FINISH_DONE.
export HF_HUB_OFFLINE=1 PYTHONPATH=${KIT:-$HOME/decision-index}
PY=${PY:-$HOME/venv/bin/python}; E=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; R=$HOME/results
st(){ echo "$(date -u +%FT%TZ) $*" >> $R/STATUS; }
until grep -q "ALL DONE" $R/STATUS; do sleep 30; done
st "finish start"
mkdir -p $R/e6/contaminated; mv $R/e6/crivo-17m.gpu.json $R/e6/crivo-32m.gpu.json $R/e6/crivo-68m.gpu.json $R/e6/crivo-150m.gpu.json $R/e6/crivo-400m.gpu.json $R/e6/nli-deberta.gpu.json $R/e6/contaminated/ 2>/dev/null
bash $E/gpu/e6.sh >> $R/logs/e6_redo.log 2>&1; st "finish E6 redo done"
for v in orig perm1 perm2 perm3 rev yn; do
  $PY $E/common/run_requests.py --engine reranker --opt model=cross-encoder/ettin-reranker-32m-v1 --requests $HOME/e4/req/$v.jsonl --out $R/e4/reranker-32m-raw/$v.jsonl --threads 4 --batch 0 >> $R/logs/e4_reranker-32m-raw.log 2>&1 || st "FAIL finish e4/reranker $v"
done
st "finish e4/reranker done"
bash $E/gpu/score_all.sh >> $R/logs/score.log 2>&1
st "finish ALL DONE"; touch $R/FINISH_DONE
