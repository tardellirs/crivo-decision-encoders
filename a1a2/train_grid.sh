#!/bin/bash
# A1/A2 grid on one GPU: 5 arms x 3 seeds at 32M (grid.tsv), each trained by train_ablate.py on the same fixed
# 150k-row subset of the v2 stage-1 data (teacher logits: Crivo-400M), then evaluated on pub15, btzsc500, s1_60 and
# the six E4 files. Two training lanes + one evaluation lane share the GPU. Resumable (skips finished outputs).
# Needs: ~/exp/a1a2/{sub150k.jsonl.gz,sub150k.t400.npz,gold_only}, ~/evalsets/*.jsonl, ~/e4/req/*.jsonl, setup_gpu.sh.
E=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; A=$E/a1a2; D=$HOME/exp/a1a2; O=$HOME/results/a1a2; PY=${PY:-$HOME/venv/bin/python}
EV=${EVALDIR:-$HOME/evalsets}; E4=${E4DIR:-$HOME/e4/req}; SEEDS=${SEEDS:-0 1 2}; O=${OUT:-$O}
mkdir -p $O/ck $O/ev $O/logs; export HF_HUB_OFFLINE=1 PYTHONPATH=${KIT:-$HOME/decision-index}:$A
st(){ echo "$(date -u +%FT%TZ) $*" >> $O/STATUS; }
G=$(cat $D/gold_only)
CODE=${CODE:-$(ls -d $HOME/.cache/huggingface/hub/models--tardellirs--crivo-32m/snapshots/c14221dd*)}   # released crivo.py/model.py
COMMON="--train $D/sub150k.jsonl.gz --epochs 1 --lr 1e-4 --head_lr 5e-4 --warmup 0.05 --max_len 2048 --max_tokens 24576 --max_items 128 --workers 6 --skip_oom ${EXTRA_TRAIN:-}"
train_one(){ # arm reader base extra seed
  local arm=$1 rd=$2 base=$3 extra=$4 s=$5 out=$O/ck/${1}_s$5 kd=()
  [ -f $out/model.safetensors ] && return
  [[ $extra == KD* ]] && kd=(--teacher $D/sub150k.t400.npz --kd_gold_only "$G") && extra=${extra#KD}
  st "train start $arm s$s"; t=$(date +%s)
  $PY $A/train_ablate.py --reader $rd --base $base $COMMON "${kd[@]}" $extra --seed $s --out $out > $O/logs/${arm}_s$s.train.log 2>&1 \
    && st "train done $arm s$s $(( $(date +%s) - t ))s" || st "FAIL train $arm s$s"
}
eval_one(){ # arm seed
  local arm=$1 s=$2 ck=$O/ck/${1}_s$2 ev=$O/ev/${1}_s$2 eng=crivo o=(--opt path=$O/ck/${1}_s$2 --opt code=$CODE)
  [ "$arm" = P ] && eng=peropt:PerOptionEngine && o=(--opt path=$ck)
  mkdir -p $ev
  for set in pub15 btzsc500 s1_60; do
    $PY $E/common/run_requests.py --engine $eng "${o[@]}" --requests $EV/$set.jsonl --out $ev/$set.jsonl >> $O/logs/${arm}_s$s.eval.log 2>&1
  done
  mkdir -p $ev/e4
  for v in orig perm1 perm2 perm3 rev yn; do
    $PY $E/common/run_requests.py --engine $eng "${o[@]}" --requests $E4/$v.jsonl --out $ev/e4/$v.jsonl >> $O/logs/${arm}_s$s.eval.log 2>&1
  done
  $PY $A/score_pub.py --requests $EV/pub15.jsonl --results $ev/pub15.jsonl --out $ev/pub15.score.json > /dev/null
  $PY $E/btzsc/adapter.py score --requests $EV/btzsc500.jsonl --results $ev/btzsc500.jsonl --out $ev/btzsc500.score.json > /dev/null
  $PY $E/s1arena/adapter.py score --data $HOME/s1arena/data --code $HOME/s1arena/code --requests $EV/s1_60.jsonl --results $ev/s1_60.jsonl --out $ev/s1_60.score.json > /dev/null
  $PY $E/e4/perturb.py flips --rows $E4/orig.jsonl --results $ev/e4 > /dev/null
  if [ -s $ev/pub15.score.json ] && [ -s $ev/btzsc500.score.json ] && [ -s $ev/s1_60.score.json ] && [ -s $ev/e4/flips.json ]; then
    touch $ev/DONE; st "eval done $arm s$s"; else touch $ev/FAILED; st "FAIL eval scoring $arm s$s (see logs; delete FAILED to retry)"; fi
}
ARMS=$(grep -v '^#' $A/grid.tsv)
lane_train(){ # lane 0: the per-option arm (P, ~4x the batches of a joint run); lane 1: the 12 joint runs (balanced)
  echo "$ARMS" | while IFS=$'\t' read -r arm rd base extra; do
    if { [ "$arm" = P ] && [ $1 = 0 ]; } || { [ "$arm" != P ] && [ $1 = 1 ]; }; then
      for s in $SEEDS; do train_one "$arm" "$rd" "$base" "$extra" $s; done
    fi
  done
  touch $O/lane$1.finished
}
lane_eval(){  # evaluates each checkpoint as soon as it exists; stops when both training lanes ended and nothing is left
  while :; do
    for arm in $(echo "$ARMS" | cut -f1); do for s in $SEEDS; do
      [ -f $O/ev/${arm}_s$s/DONE ] || [ -f $O/ev/${arm}_s$s/FAILED ] && continue
      [ -f $O/ck/${arm}_s$s/model.safetensors ] && eval_one $arm $s
    done; done
    if [ -f $O/lane0.finished ] && [ -f $O/lane1.finished ]; then
      pending=0
      for arm in $(echo "$ARMS" | cut -f1); do for s in $SEEDS; do
        [ -f $O/ck/${arm}_s$s/model.safetensors ] && [ ! -f $O/ev/${arm}_s$s/DONE ] && [ ! -f $O/ev/${arm}_s$s/FAILED ] && pending=1
      done; done
      [ $pending = 0 ] && break
    fi
    sleep 60
  done
}
rm -f $O/lane0.finished $O/lane1.finished
st "grid start $(nvidia-smi --query-gpu=name --format=csv,noheader)"
lane_train 0 & lane_train 1 & lane_eval & wait
EVALDIR=$EV $PY $A/summarize.py $O > $O/summary.txt 2>&1; st "GRID DONE"
