#!/bin/bash
# GPU-4 (pre-registered in docs/EXPERIMENT_LOG.md §18.y): completes the initialisation 2x2 of the reader ablation on one A100 40 GB. Resumable (every
# finished output is skipped; re-run this script after a crash). STATUS: ~/results/gpu4/STATUS. Phases:
#   0  inputs md5 + prefetch check (raw Ettin builds offline from its .bin; crivo-32m c14221dd snapshot present)
#   1  Praw s1, then Praw s2: per-option reader from jhu-clsp/ettin-encoder-32m with GPU-3 Praw s0's flags
#      (--grad_ckpt, uncapped, 1 epoch), each ALONE on the GPU (the script refuses to start one beside any job)
#   2  JrawComp s0, s1, s2: joint reader from the raw encoder with Jcomp's flags (KD from the Crivo-400M logits, Brier
#      0.5, 24,576 tokens/step, --epochs 4 --max_steps 6858/6860/6859 = P's step counts), CUDA_MEMFRAC 0.6, serial.
#      ONE background eval lane runs beside them, one eval at a time: Praw s1, Praw s2, JrawComp s0, JrawComp s1
#      (each JrawComp eval waits for its checkpoint). The JrawComp s2 eval runs after the last training (no
#      training on the GPU then; it may overlap the lane's tail).
#   3  any JrawComp seed whose capped run dropped a batch (oom_skips > 0) is retrained alone, uncapped (old one kept
#      *.skipped), then evaluated alone
#   4  error sweep of every eval (GPU free, as GPU-3), scoring, compare.py (the 2x2)
# Evals: pub15, btzsc500, s1_60 only (no E4, no latency). Joint = crivo engine, code= crivo-32m c14221dd, --batch 64;
# per-option = PerOptionEngine.batch_call, --batch 32 (as GPU-3 / eval_P.sh).
set -uo pipefail
MAIN=$$
BOX=${BOX:-$HOME}; E=${E:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; A=$E/a1a2; G4=$E/gpu4; TR=${TR:-$E/train}
D=${D:-$HOME/exp/a1a2}; O=${O:-$HOME/results/gpu4}; PY=${PY:-$HOME/venv/bin/python}; EV=${EV:-$HOME/evalsets}
S1D=${S1D:-$HOME/s1arena/data}; S1C=${S1C:-$HOME/s1arena/code}; GPU2=${GPU2:-$HOME/exp/gpu2_ref}; GPU3=${GPU3:-$HOME/exp/gpu3_ref}
JMEM=${JMEM:-0.6}; SEEDS=${SEEDS:-0 1 2}; PSEEDS=${PSEEDS:-1 2}; JSTEPS=${JSTEPS:-"0:6858 1:6860 2:6859"}   # P_s0..2 "total" in GPU-2 logs
EXTRA_TRAIN=${EXTRA_TRAIN:-}; SERIAL_EVAL=${SERIAL_EVAL:-0}   # SERIAL_EVAL=1: dry runs on a small CPU machine only (one process at a time)
RAW=jhu-clsp/ettin-encoder-32m
REV32=c14221dd5cb4953fa83dc40ccf3b90b37126668e   # crivo-32m main = v2 weights 93970f0 + 0.4 card (the grid's code=)
mkdir -p $O/ck $O/ev $O/logs
export HF_HUB_OFFLINE=1 PYTHONPATH=${KIT:-$HOME/decision-index}:$A PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
ST=$O/STATUS
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
die(){ st "FAILED $*"; exit 1; }
snap(){ ls -d $HOME/.cache/huggingface/hub/models--${1//\//--}/snapshots/${2:-*} 2>/dev/null | head -1; }
CODE=${CODE:-$(snap tardellirs/crivo-32m ${REV32:0:8}*)}
[ -f $CODE/crivo.py ] && [ -f $CODE/model.safetensors ] || die "crivo-32m snapshot $REV32 missing (CODE=$CODE)"
G=$(cat $D/gold_only)
COMMON="--train $D/sub150k.jsonl.gz --lr 1e-4 --head_lr 5e-4 --warmup 0.05 --max_len 2048 --max_tokens 24576 --max_items 128 --workers 6 --skip_oom --teacher $D/sub150k.t400.npz"
JLAST=$(echo $SEEDS | awk '{print $NF}')

# ---------------- 0. inputs and prefetch
if [ ! -f $O/PREFETCH_OK ]; then
  $PY - "$BOX" ${MD5LIST:-$G4/inputs.md5.box} > $O/logs/md5.log 2>&1 <<'EOF' || die "input md5 mismatch (logs/md5.log)"
import hashlib, sys, os
bad = 0
for line in open(sys.argv[2]):
    h, f = line.split(None, 1); f = os.path.join(sys.argv[1], f.strip())
    got = hashlib.md5(open(f, "rb").read()).hexdigest() if os.path.exists(f) else "missing"
    print(got == h and "ok" or "BAD", f, got); bad += got != h
sys.exit(1 if bad else 0)
EOF
  $PY - > $O/logs/prefetch.log 2>&1 <<EOF || die "prefetch check (logs/prefetch.log)"
import sys; sys.path.insert(0, "$TR")
from model import build
m, t = build("$RAW"); print("ok", "$RAW", sum(p.numel() for p in m.parameters()))
EOF
  touch $O/PREFETCH_OK; st "inputs md5 ok; prefetch ok: $(grep -m1 '^ok' $O/logs/prefetch.log); code=$(basename $CODE); GPU $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
fi

steps_of(){ for x in $JSTEPS; do [ "${x%%:*}" = "$1" ] && echo ${x#*:}; done; }
busy(){ pgrep -f 'run_requests.py|train_ablate.py' > /dev/null; }
train(){ # train <arm> <seed> <memfrac|-> args...   (writes <ck>/TRAINED when finished and checked for OOM skips)
  local arm=$1 s=$2 mf=$3; shift 3; local out=$O/ck/${arm}_s$s log=$O/logs/${arm}_s$s.train.log
  [ -f $out/model.safetensors ] && { touch $out/TRAINED; return 0; }
  echo "$arm $s" > $O/TRAINING; st "train start $arm s$s"; local t=$(date +%s)
  if [ "$mf" = - ]; then $PY $A/train_ablate.py "$@" --seed $s --out $out $EXTRA_TRAIN > $log 2>&1
  else CUDA_MEMFRAC=$mf $PY $A/train_ablate.py "$@" --seed $s --out $out $EXTRA_TRAIN > $log 2>&1; fi
  local rc=$?; rm -f $O/TRAINING
  [ $rc = 0 ] && [ -f $out/model.safetensors ] || { st "FAIL train $arm s$s (exit $rc)"; return 1; }
  st "train done $arm s$s $(( $(date +%s) - t ))s $(grep -o '"saved".*' $log | sed 's/.*"steps"/"steps"/' | cut -c1-160)"
  grep -q '"saved".*"oom_skips": [1-9]' $log && { st "SKIPS $arm s$s (batches dropped by OOM)"; touch $out/SKIPS; }
  touch $out/TRAINED; return 0
}
run_set(){ # run_set <eng> <ev dir> <set> <requests> opts...
  local eng=$1 ev=$2 name=$3 req=$4; shift 4
  $PY $E/common/run_requests.py --engine $eng "$@" --requests $req --out $ev/$name.jsonl >> $ev/eval.log 2>&1
}
eval_ck(){ # eval_ck <arm> <seed>: joint (crivo engine + code=, --batch 64) or per-option (batch_call, --batch 32)
  local arm=$1 s=$2 ck=$O/ck/${1}_s$2 ev=$O/ev/${1}_s$2 eng=crivo o
  { [ -f $ev/DONE ] || [ -f $ev/RAN ]; } && return 0
  if [ "$arm" = Praw ]; then eng=peropt:PerOptionEngine; o=(--opt path=$ck --batch 32)
  else o=(--opt path=$ck --opt code=$CODE --batch 64); fi
  mkdir -p $ev; st "eval start $arm s$s"
  for set in pub15 btzsc500 s1_60; do run_set $eng $ev $set $EV/$set.jsonl "${o[@]}"; done
  touch $ev/RAN; st "eval ran $arm s$s"
}
score_ck(){ # score_ck <arm> <seed>
  local ev=$O/ev/${1}_s$2
  $PY $A/score_pub.py --requests $EV/pub15.jsonl --results $ev/pub15.jsonl --out $ev/pub15.score.json > /dev/null 2>>$ev/eval.log
  $PY $E/btzsc/adapter.py score --requests $EV/btzsc500.jsonl --results $ev/btzsc500.jsonl --out $ev/btzsc500.score.json > /dev/null 2>>$ev/eval.log
  $PY $E/s1arena/adapter.py score --data $S1D --code $S1C --requests $EV/s1_60.jsonl --results $ev/s1_60.jsonl --out $ev/s1_60.score.json > /dev/null 2>>$ev/eval.log
  [ -s $ev/pub15.score.json ] && [ -s $ev/btzsc500.score.json ] && [ -s $ev/s1_60.score.json ]
}
errs(){ $PY $G4/eff_errors.py "$@"; }   # effective errors (ids with no ok/unsupported line)
sweep_ck(){ # sweep_ck <arm> <seed>: re-run effective errors with the GPU free of training (as GPU-3), then score
  local arm=$1 s=$2 ck=$O/ck/${1}_s$2 ev=$O/ev/${1}_s$2 eng=crivo o
  [ -f $ev/DONE ] && return 0
  if [ "$arm" = Praw ]; then eng=peropt:PerOptionEngine; o=(--opt path=$ck --opt batch_size=8)
  else o=(--opt path=$ck --opt code=$CODE --opt batch_size=8); fi
  local n set
  for set in pub15 btzsc500 s1_60; do
    n=$(errs $ev/$set.jsonl $EV/$set.jsonl)
    if [ "$n" != 0 ]; then st "sweep $arm s$s $set: $n effective errors -> batch 1"; run_set $eng $ev $set $EV/$set.jsonl "${o[@]}"; fi
    n=$(errs $ev/$set.jsonl $EV/$set.jsonl); [ "$n" = 0 ] || { st "FAIL $arm s$s $set: $n effective errors after sweep"; return 1; }
  done
  score_ck $arm $s || { st "FAIL scoring $arm s$s"; return 1; }
  touch $ev/DONE; st "eval done $arm s$s (0 effective errors)"
}
waitck(){ # waitck <arm> <seed>: wait for train()'s TRAINED marker; give up if the main script has died
  while [ ! -f $O/ck/${1}_s$2/TRAINED ]; do kill -0 $MAIN 2>/dev/null || return 1; sleep 15; done
}
lane(){ # one eval at a time beside the JrawComp trainings: the Praw evals, then every JrawComp seed but the last
  local s
  for s in $PSEEDS; do eval_ck Praw $s; done
  for s in $SEEDS; do
    [ "$s" = "$JLAST" ] && break
    waitck JrawComp $s || return 1
    [ -f $O/ck/JrawComp_s$s/SKIPS ] || eval_ck JrawComp $s   # a seed with skips is retrained (phase 3) and evaluated then
  done
}

# ---------------- 1. Praw s1, s2: each alone (as GPU-3 Praw s0 / the grid's P: --grad_ckpt, uncapped)
for s in $PSEEDS; do
  [ -f $O/ck/Praw_s$s/model.safetensors ] && continue
  busy && die "another training or eval is running before Praw s$s (it must train alone)"
  train Praw $s - --reader peropt --base $RAW $COMMON --kd_gold_only "$G" --epochs 1 --grad_ckpt || die "train Praw s$s"
done
# ---------------- 2. JrawComp s0-s2 (joint, raw init, Jcomp flags), the eval lane alongside
if [ ! -f $O/JOINT_DONE ]; then
  LP=""
  if [ "$SERIAL_EVAL" != 1 ]; then busy && die "a job is running before the JrawComp block"; ( lane ) & LP=$!; fi
  for s in $SEEDS; do
    n=$(steps_of $s); [ -n "$n" ] || die "no step count for seed $s"
    train JrawComp $s $JMEM --reader joint --base $RAW $COMMON --kd_gold_only "$G" --epochs 4 --max_steps $n || die "train JrawComp s$s"
  done
  pids=()
  if [ "$SERIAL_EVAL" = 1 ]; then lane || die "eval lane"; [ -f $O/ck/JrawComp_s$JLAST/SKIPS ] || eval_ck JrawComp $JLAST
  else [ -f $O/ck/JrawComp_s$JLAST/SKIPS ] || { ( eval_ck JrawComp $JLAST ) & pids+=($!); }; fi
  [ -n "$LP" ] && { wait $LP || die "eval lane"; }
  for p in ${pids[@]+"${pids[@]}"}; do wait $p; done
  # ---------------- 3. any seed that dropped a batch: retrain alone, uncapped, then evaluate it
  for s in $SEEDS; do
    if [ -f $O/ck/JrawComp_s$s/SKIPS ]; then
      busy && die "a job is running before the JrawComp s$s retrain (it must run alone)"
      st "retrain JrawComp s$s alone (uncapped): the capped run dropped batches"
      rm -rf $O/ck/JrawComp_s$s.skipped $O/ev/JrawComp_s$s.skipped; mv $O/ck/JrawComp_s$s $O/ck/JrawComp_s$s.skipped
      [ -d $O/ev/JrawComp_s$s ] && mv $O/ev/JrawComp_s$s $O/ev/JrawComp_s$s.skipped
      mv $O/logs/JrawComp_s$s.train.log $O/logs/JrawComp_s$s.skipped.train.log
      train JrawComp $s - --reader joint --base $RAW $COMMON --kd_gold_only "$G" --epochs 4 --max_steps $(steps_of $s) || die "retrain JrawComp s$s"
      [ -f $O/ck/JrawComp_s$s/SKIPS ] && { st "SKIPS persist for JrawComp s$s alone: report them"; mv $O/ck/JrawComp_s$s/SKIPS $O/ck/JrawComp_s$s/SKIPS_ALONE; }
      eval_ck JrawComp $s
    fi
  done
  touch $O/JOINT_DONE
fi
# ---------------- 4. sweeps (GPU free), scores, the 2x2
for s in $PSEEDS; do eval_ck Praw $s; done; for s in $SEEDS; do eval_ck JrawComp $s; done   # resume safety (no-op normally)
for s in $PSEEDS; do sweep_ck Praw $s || die "Praw s$s eval"; done
for s in $SEEDS; do sweep_ck JrawComp $s || die "JrawComp s$s eval"; done
$PY $G4/compare.py --gpu2 $GPU2 --gpu3 $GPU3 --gpu4 $O --out $O/compare.json > $O/compare.txt 2>&1 \
  && st "compare: $(grep -A3 'Praw - JrawComp' $O/compare.txt | head -4 | tr '\n' ' ' | cut -c1-300)" || st "compare.py failed (run it locally after fetch)"
st "ALL DONE"
