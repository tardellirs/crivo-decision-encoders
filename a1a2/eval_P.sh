#!/bin/bash
# Evaluate the per-option checkpoints with the batched PerOptionEngine.batch_call (--batch 32), as each P checkpoint
# appears (the grid eval lanes were stopped at 11:43 UTC: PerOptionEngine.__call__ ran ~0.6 req/s). batch_call was
# checked against __call__ on 300 pub15 requests: 17,464/17,464 answers equal, max |dp| 0.009 (bf16 batch shapes).
# Same sets and scorers as train_grid_serial*.sh. Then summarize.py and "P EVALS DONE" in STATUS.
E=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; A=$E/a1a2; O=$HOME/results/a1a2; PY=${PY:-$HOME/venv/bin/python}; EV=$HOME/evalsets; E4=$HOME/e4/req
export HF_HUB_OFFLINE=1 PYTHONPATH=${KIT:-$HOME/decision-index}:$A PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st(){ echo "$(date -u +%FT%TZ) $*" >> $O/STATUS; }
while [ ! -f $O/lane.finished ]; do sleep 60; done   # 13:21 UTC: P evals never share the GPU with a P training (P s2 dropped 6 batches beside the P s1 eval)
for s in 0 1 2; do
  ck=$O/ck/P_s$s ev=$O/ev/P_s$s; o=(--opt path=$ck --batch 32)
  [ -f $ev/DONE ] && continue
  while [ ! -f $ck/model.safetensors ] || [ "$(cat $O/TRAINING 2>/dev/null)" = "P $s" ]; do sleep 60; done
  mkdir -p $ev/e4; st "eval start P s$s (batched)"
  for set in pub15 btzsc500 s1_60; do
    $PY $E/common/run_requests.py --engine peropt:PerOptionEngine "${o[@]}" --requests $EV/$set.jsonl --out $ev/$set.jsonl >> $O/logs/P_s$s.eval.log 2>&1
  done
  for v in orig perm1 perm2 perm3 rev yn; do
    $PY $E/common/run_requests.py --engine peropt:PerOptionEngine "${o[@]}" --requests $E4/$v.jsonl --out $ev/e4/$v.jsonl >> $O/logs/P_s$s.eval.log 2>&1
  done
  $PY $A/score_pub.py --requests $EV/pub15.jsonl --results $ev/pub15.jsonl --out $ev/pub15.score.json > /dev/null
  $PY $E/btzsc/adapter.py score --requests $EV/btzsc500.jsonl --results $ev/btzsc500.jsonl --out $ev/btzsc500.score.json > /dev/null
  $PY $E/s1arena/adapter.py score --data $HOME/s1arena/data --code $HOME/s1arena/code --requests $EV/s1_60.jsonl --results $ev/s1_60.jsonl --out $ev/s1_60.score.json > /dev/null
  $PY $E/e4/perturb.py flips --rows $E4/orig.jsonl --results $ev/e4 > /dev/null
  touch $ev/DONE; st "eval done P s$s (batched)"
done
EVALDIR=$EV $PY $A/summarize.py $O > $O/summary.txt 2>&1; st "P EVALS DONE"
