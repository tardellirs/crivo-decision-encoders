#!/bin/bash
# A1 confound check (added 2026-10-21 12:40 UTC). Same token budget per step gives the per-option arm P ~3.25x the
# optimizer steps of the joint arm J (6,858 vs 2,112 on the same 150k rows: P's rows are n options long). P s0 beat J
# on every eval set, so this arm, Jsteps, is J with --max_tokens 6784 (6,800-6,860 steps, ~22 questions per step,
# like P). Starts when the grid's training lane has finished (P s2), so it never shares the GPU with a P training.
# Joint runs capped at 0.6 (an eval runs alongside); oom_skips logged; evals as in train_grid_serial3.sh.
E=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; A=$E/a1a2; D=$HOME/exp/a1a2; O=$HOME/results/a1a2; PY=${PY:-$HOME/venv/bin/python}
EV=$HOME/evalsets; E4=$HOME/e4/req
export HF_HUB_OFFLINE=1 PYTHONPATH=${KIT:-$HOME/decision-index}:$A PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st(){ echo "$(date -u +%FT%TZ) $*" >> $O/STATUS; }
G=$(cat $D/gold_only)
CODE=${CODE:-$(ls -d $HOME/.cache/huggingface/hub/models--tardellirs--crivo-32m/snapshots/c14221dd*)}
COMMON="--train $D/sub150k.jsonl.gz --epochs 1 --lr 1e-4 --head_lr 5e-4 --warmup 0.05 --max_len 2048 --max_tokens 6784 --max_items 128 --workers 6 --skip_oom"
eval_one(){ # seed
  local s=$1 ev=$O/ev/Jsteps_s$1 o=(--opt path=$O/ck/Jsteps_s$1 --opt code=$CODE --batch 64)
  mkdir -p $ev/e4; st "eval start Jsteps s$s"
  for set in pub15 btzsc500 s1_60; do
    $PY $E/common/run_requests.py --engine crivo "${o[@]}" --requests $EV/$set.jsonl --out $ev/$set.jsonl >> $O/logs/Jsteps_s$s.eval.log 2>&1
  done
  for v in orig perm1 perm2 perm3 rev yn; do
    $PY $E/common/run_requests.py --engine crivo "${o[@]}" --requests $E4/$v.jsonl --out $ev/e4/$v.jsonl >> $O/logs/Jsteps_s$s.eval.log 2>&1
  done
  $PY $A/score_pub.py --requests $EV/pub15.jsonl --results $ev/pub15.jsonl --out $ev/pub15.score.json > /dev/null
  $PY $E/btzsc/adapter.py score --requests $EV/btzsc500.jsonl --results $ev/btzsc500.jsonl --out $ev/btzsc500.score.json > /dev/null
  $PY $E/s1arena/adapter.py score --data $HOME/s1arena/data --code $HOME/s1arena/code --requests $EV/s1_60.jsonl --results $ev/s1_60.jsonl --out $ev/s1_60.score.json > /dev/null
  $PY $E/e4/perturb.py flips --rows $E4/orig.jsonl --results $ev/e4 > /dev/null
  touch $ev/DONE; st "eval done Jsteps s$s"
}
while [ ! -f $O/lane.finished ]; do sleep 60; done
pids=()
for s in 0 1 2; do
  out=$O/ck/Jsteps_s$s
  if [ ! -f $out/model.safetensors ]; then
    st "train start Jsteps s$s"; t=$(date +%s)
    CUDA_MEMFRAC=0.6 $PY $A/train_ablate.py --reader joint --base cross-encoder/ettin-reranker-32m-v1 $COMMON --teacher $D/sub150k.t400.npz --kd_gold_only "$G" --seed $s --out $out > $O/logs/Jsteps_s$s.train.log 2>&1 \
      && st "train done Jsteps s$s $(( $(date +%s) - t ))s $(grep -o '"oom_skips": [0-9]*' $O/logs/Jsteps_s$s.train.log | tail -1)" || st "FAIL train Jsteps s$s"
    grep -q '"saved".*"oom_skips": [1-9]' $O/logs/Jsteps_s$s.train.log && st "SKIPS Jsteps s$s"
  fi
  [ -f $O/ev/Jsteps_s$s/DONE ] || { eval_one $s & pids+=($!); }
done
wait "${pids[@]}"
while ! grep -q "P EVALS DONE" $O/STATUS; do sleep 60; done
EVALDIR=$EV $PY $A/summarize.py $O > $O/summary.txt 2>&1; st "JSTEPS DONE"
