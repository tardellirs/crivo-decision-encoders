#!/bin/bash
# Run jobs_infer.tsv on one GPU: one background loop per lane (3 lanes share the GPU; small encoders leave it
# mostly idle otherwise), each job resumable (run_requests.py skips finished ids). Then E6 latency ALONE on the GPU
# (no other lane running), then scoring. Status: ~/results/STATUS (one line per event), ~/results/logs/*.log.
# jobs file: tab-separated; an empty options field must be written as '-' (bash read collapses empty tab fields).
# Usage: bash run_lanes.sh [jobs file]      (HF_HUB_OFFLINE=1 after setup_gpu.sh prefetched the models)
JOBS=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/jobs_infer.tsv}
E=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; R=$HOME/results; Q=$HOME/req; [ -s $Q/s1_60.jsonl ] || cp $HOME/evalsets/s1_60.jsonl $Q/ 2>/dev/null; PY=${PY:-$HOME/venv/bin/python}
mkdir -p $R/logs; export HF_HUB_OFFLINE=1 PYTHONPATH=${KIT:-$HOME/decision-index}
st(){ echo "$(date -u +%FT%TZ) $*" >> $R/STATUS; }
run_job(){  # name engine opts reqkey threads batch
  local name=$1 eng=$2 opts=$3 req=$4 th=$5 bt=${6:-0} o=()
  for kv in $opts; do [ "$kv" = - ] || o+=(--opt "$kv"); done
  if [ "$req" = E4 ]; then
    for v in orig perm1 perm2 perm3 rev yn; do
      mkdir -p $R/$name
      $PY $E/common/run_requests.py --engine $eng "${o[@]}" --requests $HOME/e4/req/$v.jsonl --out $R/$name/$v.jsonl \
        --threads $th --batch $bt >> $R/logs/${name//\//_}.log 2>&1 || st "FAIL $name $v"
    done
  else
    mkdir -p $(dirname $R/$name)
    $PY $E/common/run_requests.py --engine $eng "${o[@]}" --requests $Q/$req.jsonl --out $R/$name.jsonl --threads $th --batch $bt \
      >> $R/logs/${name//\//_}.log 2>&1 || st "FAIL $name"
  fi
}
lane(){
  grep -v '^#' $JOBS | awk -F'\t' -v L=$1 '$1==L' | while IFS=$'\t' read -r l name eng opts req th bt; do
    [ -f $R/$name.done ] && continue
    st "start lane$1 $name"; t=$(date +%s); run_job "$name" "$eng" "$opts" "$req" "$th" "$bt"
    mkdir -p $(dirname $R/$name); touch $R/$name.done; st "done lane$1 $name $(( $(date +%s) - t ))s"
  done
}
st "lanes start $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for L in $(grep -v '^#' $JOBS | cut -f1 | sort -u); do lane $L & done
wait
st "lanes finished; E6 latency (GPU alone)"
bash $E/gpu/e6.sh >> $R/logs/e6.log 2>&1; st "E6 done"
bash $E/gpu/score_all.sh >> $R/logs/score.log 2>&1; st "ALL DONE"
