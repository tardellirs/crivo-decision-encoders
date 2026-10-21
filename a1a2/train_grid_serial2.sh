#!/bin/bash
# Serial version of train_grid.sh for a 40 GB A100 (2026-10-21). The parallel version failed there: the per-option
# run's allocator took the whole GPU, so the other trainings and the eval lane died with CUDA OOM at load, and J s0
# silently skipped 64/2112 batches (--skip_oom). Here:
#   - one training at a time; joint arms first (capped at CUDA_MEMFRAC 0.72 so the eval lane can share the GPU),
#     then the three per-option runs uncapped and alone (the eval lane waits while they train);
#   - every run logs oom_skips; a run with skips is marked "SKIPS" in STATUS (it must not be reported as is);
#   - joint-arm evals use --batch 64 (batched == unbatched, common/check_batch.py).
# Same arms, data, flags, eval sets and scoring as train_grid.sh. Resumable.
# v2 (2026-10-21 09:25 UTC): evals (~11.5 min each) were the bottleneck and would have waited ~2 h while the
# per-option runs trained alone. Now two eval lanes run all the time (a lock dir per checkpoint), and the per-option
# runs are capped at 0.75 instead of running alone. Any run that drops batches still shows "SKIPS" in STATUS and is
# rerun alone at the end before reporting.
E=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; A=$E/a1a2; D=$HOME/exp/a1a2; O=$HOME/results/a1a2; PY=${PY:-$HOME/venv/bin/python}
EV=${EVALDIR:-$HOME/evalsets}; E4=${E4DIR:-$HOME/e4/req}; SEEDS=${SEEDS:-0 1 2}
mkdir -p $O/ck $O/ev $O/logs; export HF_HUB_OFFLINE=1 PYTHONPATH=${KIT:-$HOME/decision-index}:$A PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st(){ echo "$(date -u +%FT%TZ) $*" >> $O/STATUS; }
G=$(cat $D/gold_only)
CODE=${CODE:-$(ls -d $HOME/.cache/huggingface/hub/models--tardellirs--crivo-32m/snapshots/c14221dd*)}
COMMON="--train $D/sub150k.jsonl.gz --epochs 1 --lr 1e-4 --head_lr 5e-4 --warmup 0.05 --max_len 2048 --max_tokens 24576 --max_items 128 --workers 6 --skip_oom"
train_one(){ # arm reader base extra seed
  local arm=$1 rd=$2 base=$3 extra=$4 s=$5 out=$O/ck/${1}_s$5 kd=() frac=0.72
  [ -f $out/model.safetensors ] && return
  [ "$rd" = peropt ] && frac=0.75
  [[ $extra == KD* ]] && kd=(--teacher $D/sub150k.t400.npz --kd_gold_only "$G") && extra=${extra#KD}
  echo "$arm $s" > $O/TRAINING
  st "train start $arm s$s"; t=$(date +%s)
  CUDA_MEMFRAC=$frac $PY $A/train_ablate.py --reader $rd --base $base $COMMON "${kd[@]}" $extra --seed $s --out $out > $O/logs/${arm}_s$s.train.log 2>&1 \
    && st "train done $arm s$s $(( $(date +%s) - t ))s $(grep -o '"oom_skips": [0-9]*' $O/logs/${arm}_s$s.train.log | tail -1)" || st "FAIL train $arm s$s"
  grep -q '"saved".*"oom_skips": [1-9]' $O/logs/${arm}_s$s.train.log && st "SKIPS $arm s$s (batches dropped by OOM)"
  rm -f $O/TRAINING $O/TRAINING_ALONE
}
eval_one(){ # arm seed
  local arm=$1 s=$2 ck=$O/ck/${1}_s$2 ev=$O/ev/${1}_s$2 eng=crivo o=(--opt path=$O/ck/${1}_s$2 --opt code=$CODE --batch 64)
  [ "$arm" = P ] && eng=peropt:PerOptionEngine && o=(--opt path=$ck)
  mkdir -p $ev; touch $O/EVALUATING; st "eval start $arm s$s"
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
  rm -f $O/EVALUATING
  nerr=$(cat $ev/*.jsonl $ev/e4/*.jsonl | grep -c '"status": "error"')
  if [ -s $ev/pub15.score.json ] && [ -s $ev/btzsc500.score.json ] && [ -s $ev/s1_60.score.json ] && [ -s $ev/e4/flips.json ]; then
    touch $ev/DONE; st "eval done $arm s$s (error lines: $nerr)"; else touch $ev/FAILED; st "FAIL eval scoring $arm s$s (see logs; delete FAILED to retry)"; fi
}
ARMS=$(grep -v '^#' $A/grid.tsv)
lane_train(){ # joint arms first, then the per-option arm
  for pass in joint peropt; do
    echo "$ARMS" | while IFS=$'\t' read -r arm rd base extra; do
      [ "$rd" = $pass ] || continue
      for s in $SEEDS; do train_one "$arm" "$rd" "$base" "$extra" $s; done
    done
  done
  touch $O/lane.finished
}
lane_eval(){  # evaluates each finished checkpoint; waits while a per-option run trains alone
  while :; do
    for arm in $(echo "$ARMS" | cut -f1); do for s in $SEEDS; do
      [ -f $O/ev/${arm}_s$s/DONE ] || [ -f $O/ev/${arm}_s$s/FAILED ] && continue
      [ -f $O/ck/${arm}_s$s/model.safetensors ] && [ "$(cat $O/TRAINING 2>/dev/null)" != "$arm $s" ] && mkdir $O/ev/${arm}_s$s.lock 2>/dev/null && { eval_one $arm $s; rmdir $O/ev/${arm}_s$s.lock; }
    done; done
    if [ -f $O/lane.finished ]; then
      pending=0
      for arm in $(echo "$ARMS" | cut -f1); do for s in $SEEDS; do
        [ -f $O/ck/${arm}_s$s/model.safetensors ] && [ ! -f $O/ev/${arm}_s$s/DONE ] && [ ! -f $O/ev/${arm}_s$s/FAILED ] && pending=1
      done; done
      [ $pending = 0 ] && break
    fi
    sleep 60
  done
}
rm -f $O/lane.finished $O/TRAINING $O/TRAINING_ALONE $O/EVALUATING; rmdir $O/ev/*.lock 2>/dev/null
st "serial grid v2 start $(nvidia-smi --query-gpu=name --format=csv,noheader)"
lane_train & lane_eval & sleep 30; lane_eval & wait
EVALDIR=$EV $PY $A/summarize.py $O > $O/summary.txt 2>&1; st "GRID DONE"
