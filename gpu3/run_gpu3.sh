#!/bin/bash
# GPU-3 (compute-matched controls; pre-registered in docs/EXPERIMENT_LOG.md §18.1): one A100 40 GB, strictly serial trainings. Resumable (every finished output is
# skipped; re-run this script after a crash). STATUS: ~/results/gpu3/STATUS. Phases:
#   0  inputs md5 + prefetch check (every model loads offline, incl. the raw Ettin encoder weights)
#   1  Jcomp s0, s1, s2: joint arm J (grid.tsv flags) for P's optimizer-step count per seed (--epochs 4 --max_steps
#      6858/6860/6859 -> LR warmup 5% + linear decay over exactly those steps), CUDA_MEMFRAC 0.6. Evals of Jcomp
#      checkpoints run beside a Jcomp training only (s0 eval beside s1, s1 eval beside s2).
#   2  any Jcomp seed whose run dropped a batch (oom_skips > 0) is retrained alone, uncapped (old one kept *.skipped)
#   3  Praw s0: per-option reader from jhu-clsp/ettin-encoder-32m, --grad_ckpt, uncapped, ALONE (no eval running)
#   4  evals with no training on the GPU: Praw s0 (PerOptionEngine.batch_call, --batch 32, as eval_P.sh) beside the
#      Jcomp s2 eval; then an error sweep of every eval (batch 1, GPU free), rescoring, compare.py
#   5  E6 GPU latency, GPU alone: released Crivo-32M v2 (joint) vs per-option 32M (GPU-2 P_s0), lat1000, 2 passes
#   6  E6 CPU latency, GPU idle and CUDA hidden, 16 threads, the same first CPU_N lat1000 requests for every system
set -uo pipefail
BOX=${BOX:-$HOME}; E=${E:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}; A=$E/a1a2; G3=$E/gpu3; D=${D:-$HOME/exp/a1a2}; O=${O:-$HOME/results/gpu3}
PY=${PY:-$HOME/venv/bin/python}; EV=${EV:-$HOME/evalsets}; E4=${E4:-$HOME/e4/req}; LAT=${LAT:-$HOME/req/lat1000.jsonl}
S1D=${S1D:-$HOME/s1arena/data}; S1C=${S1C:-$HOME/s1arena/code}; PS0=${PS0:-$HOME/exp/P_s0}; GPU2=${GPU2:-$HOME/exp/gpu2_ref}
JMEM=${JMEM:-0.6}; SEEDS=${SEEDS:-0 1 2}; JSTEPS=${JSTEPS:-"0:6858 1:6860 2:6859"}   # P_s0..2 "total" in GPU-2 logs
EXTRA_TRAIN=${EXTRA_TRAIN:-}; LAT_N=${LAT_N:-1000}; CPU_N=${CPU_N:-80}; CPU_THREADS=${CPU_THREADS:-16}
SERIAL_EVAL=${SERIAL_EVAL:-0}; SKIP_E6CPU=${SKIP_E6CPU:-0}   # dry runs on a small CPU machine only: one python process at a time, no CPU block
CPU_CRIVOS=${CPU_CRIVOS:-"17m 32m 68m 150m 400m 1b"}; CPU_DEB=${CPU_DEB:-1}; DEB=MoritzLaurer/deberta-v3-large-zeroshot-v2.0
REV32=c14221dd5cb4953fa83dc40ccf3b90b37126668e   # crivo-32m main = v2 weights 93970f0 + 0.4 card (the grid's code=)
mkdir -p $O/ck $O/ev $O/logs $O/e6gpu $O/e6cpu
export HF_HUB_OFFLINE=1 PYTHONPATH=${KIT:-$HOME/decision-index}:$A PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
ST=$O/STATUS
st(){ echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
die(){ st "FAILED $*"; exit 1; }
snap(){ ls -d $HOME/.cache/huggingface/hub/models--${1//\//--}/snapshots/${2:-*} 2>/dev/null | head -1; }
CODE=${CODE:-$(snap tardellirs/crivo-32m ${REV32:0:8}*)}
[ -f $CODE/crivo.py ] && [ -f $CODE/model.safetensors ] || die "crivo-32m snapshot $REV32 missing (CODE=$CODE)"
G=$(cat $D/gold_only)
COMMON="--train $D/sub150k.jsonl.gz --lr 1e-4 --head_lr 5e-4 --warmup 0.05 --max_len 2048 --max_tokens 24576 --max_items 128 --workers 6 --skip_oom --teacher $D/sub150k.t400.npz"

# ---------------- 0. inputs and prefetch
if [ ! -f $O/PREFETCH_OK ]; then
  $PY - "$BOX" ${MD5LIST:-$G3/inputs.md5.box} > $O/logs/md5.log 2>&1 <<'EOF' || die "input md5 mismatch (logs/md5.log)"
import hashlib, sys, os
bad = 0
for line in open(sys.argv[2]):
    h, f = line.split(None, 1); f = os.path.join(sys.argv[1], f.strip())
    got = hashlib.md5(open(f, "rb").read()).hexdigest() if os.path.exists(f) else "missing"
    print(got == h and "ok" or "BAD", f, got); bad += got != h
sys.exit(1 if bad else 0)
EOF
  $PY - > $O/logs/prefetch.log 2>&1 <<EOF || die "prefetch check (logs/prefetch.log)"
import sys, os, glob; sys.path.insert(0, "$E/train")
from model import build
for b in ("cross-encoder/ettin-reranker-32m-v1", "jhu-clsp/ettin-encoder-32m"):
    m, t = build(b); print("ok", b, sum(p.numel() for p in m.parameters()))
for r in "${PREFETCH_MODELS:-tardellirs/crivo-17m tardellirs/crivo-32m tardellirs/crivo-68m tardellirs/crivo-150m tardellirs/crivo-400m tardellirs/crivo-1b $DEB}".split():
    w = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--%s/snapshots/*/*.safetensors" % r.replace("/", "--")))
    assert w, "no weights for " + r; print("ok", r, len(w))
EOF
  touch $O/PREFETCH_OK; st "inputs md5 ok; prefetch ok: $(grep -c '^ok' $O/logs/prefetch.log) models load offline; GPU $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
fi

steps_of(){ for x in $JSTEPS; do [ "${x%%:*}" = "$1" ] && echo ${x#*:}; done; }
train(){ # train <arm> <seed> <memfrac|-> args...
  local arm=$1 s=$2 mf=$3; shift 3; local out=$O/ck/${arm}_s$s log=$O/logs/${arm}_s$s.train.log
  [ -f $out/model.safetensors ] && return 0
  echo "$arm $s" > $O/TRAINING; st "train start $arm s$s"; local t=$(date +%s)
  if [ "$mf" = - ]; then $PY $A/train_ablate.py "$@" --seed $s --out $out $EXTRA_TRAIN > $log 2>&1
  else CUDA_MEMFRAC=$mf $PY $A/train_ablate.py "$@" --seed $s --out $out $EXTRA_TRAIN > $log 2>&1; fi
  local rc=$?; rm -f $O/TRAINING
  [ $rc = 0 ] && [ -f $out/model.safetensors ] || { st "FAIL train $arm s$s (exit $rc)"; return 1; }
  st "train done $arm s$s $(( $(date +%s) - t ))s $(grep -o '"saved".*' $log | sed 's/.*"steps"/"steps"/' | cut -c1-160)"
  grep -q '"saved".*"oom_skips": [1-9]' $log && { st "SKIPS $arm s$s (batches dropped by OOM)"; touch $out/SKIPS; }
  return 0
}
run_set(){ # run_set <eng> <ev dir> <set> <requests> opts...
  local eng=$1 ev=$2 name=$3 req=$4; shift 4
  $PY $E/common/run_requests.py --engine $eng "$@" --requests $req --out $ev/$name.jsonl >> $ev/eval.log 2>&1
}
eval_ck(){ # eval_ck <arm> <seed> : grid eval (crivo engine + code=, --batch 64) or per-option (batch_call, --batch 32)
  local arm=$1 s=$2 ck=$O/ck/${1}_s$2 ev=$O/ev/${1}_s$2 eng=crivo o
  { [ -f $ev/DONE ] || [ -f $ev/RAN ]; } && return 0
  if [ "$arm" = Praw ]; then eng=peropt:PerOptionEngine; o=(--opt path=$ck --batch 32)
  else o=(--opt path=$ck --opt code=$CODE --batch 64); fi
  mkdir -p $ev/e4; st "eval start $arm s$s"
  for set in pub15 btzsc500 s1_60; do run_set $eng $ev $set $EV/$set.jsonl "${o[@]}"; done
  for v in orig perm1 perm2 perm3 rev yn; do run_set $eng $ev e4/$v $E4/$v.jsonl "${o[@]}"; done
  touch $ev/RAN; st "eval ran $arm s$s"
}
score_ck(){ # score_ck <arm> <seed>
  local ev=$O/ev/${1}_s$2
  $PY $A/score_pub.py --requests $EV/pub15.jsonl --results $ev/pub15.jsonl --out $ev/pub15.score.json > /dev/null 2>>$ev/eval.log
  $PY $E/btzsc/adapter.py score --requests $EV/btzsc500.jsonl --results $ev/btzsc500.jsonl --out $ev/btzsc500.score.json > /dev/null 2>>$ev/eval.log
  $PY $E/s1arena/adapter.py score --data $S1D --code $S1C --requests $EV/s1_60.jsonl --results $ev/s1_60.jsonl --out $ev/s1_60.score.json > /dev/null 2>>$ev/eval.log
  $PY $E/e4/perturb.py flips --rows $E4/orig.jsonl --results $ev/e4 > /dev/null 2>>$ev/eval.log
  [ -s $ev/pub15.score.json ] && [ -s $ev/btzsc500.score.json ] && [ -s $ev/s1_60.score.json ] && [ -s $ev/e4/flips.json ]
}
errs(){ $PY $G3/eff_errors.py "$@"; }   # effective errors (last line per id = error, or id missing)
sweep_ck(){ # sweep_ck <arm> <seed>: re-run effective errors at batch 1 with the GPU free of training, then score
  local arm=$1 s=$2 ck=$O/ck/${1}_s$2 ev=$O/ev/${1}_s$2 eng=crivo o
  [ -f $ev/DONE ] && return 0
  if [ "$arm" = Praw ]; then eng=peropt:PerOptionEngine; o=(--opt path=$ck --opt batch_size=8)
  else o=(--opt path=$ck --opt code=$CODE --opt batch_size=8); fi
  local n pair
  for pair in pub15:$EV/pub15.jsonl btzsc500:$EV/btzsc500.jsonl s1_60:$EV/s1_60.jsonl e4/orig:$E4/orig.jsonl e4/perm1:$E4/perm1.jsonl \
              e4/perm2:$E4/perm2.jsonl e4/perm3:$E4/perm3.jsonl e4/rev:$E4/rev.jsonl e4/yn:$E4/yn.jsonl; do
    n=$(errs $ev/${pair%%:*}.jsonl ${pair#*:})
    if [ "$n" != 0 ]; then st "sweep $arm s$s ${pair%%:*}: $n effective errors -> batch 1"; run_set $eng $ev ${pair%%:*} ${pair#*:} "${o[@]}"; fi
    n=$(errs $ev/${pair%%:*}.jsonl ${pair#*:}); [ "$n" = 0 ] || { st "FAIL $arm s$s ${pair%%:*}: $n effective errors after sweep"; return 1; }
  done
  score_ck $arm $s || { st "FAIL scoring $arm s$s"; return 1; }
  touch $ev/DONE; st "eval done $arm s$s (0 effective errors)"
}

# ---------------- 1. Jcomp (joint, grid J flags, P's step count), evals of the previous seed alongside
if [ ! -f $O/JOINT_DONE ]; then
  EVP=""
  for s in $SEEDS; do
    n=$(steps_of $s); [ -n "$n" ] || die "no step count for seed $s"
    train Jcomp $s $JMEM --reader joint --base cross-encoder/ettin-reranker-32m-v1 $COMMON --kd_gold_only "$G" --epochs 4 --max_steps $n || die "train Jcomp s$s"
    if [ "$s" != "$(echo $SEEDS | awk '{print $NF}')" ]; then   # eval this seed beside the next Jcomp training
      [ -n "$EVP" ] && wait $EVP
      if [ "$SERIAL_EVAL" = 1 ]; then eval_ck Jcomp $s; else [ -f $O/ev/Jcomp_s$s/RAN ] || { ( eval_ck Jcomp $s ) & EVP=$!; }; fi
    fi
  done
  [ -n "$EVP" ] && wait $EVP          # Praw must train alone
  # 2. any seed that dropped a batch: retrain alone, uncapped
  for s in $SEEDS; do
    if [ -f $O/ck/Jcomp_s$s/SKIPS ]; then
      st "retrain Jcomp s$s alone (uncapped): the capped run dropped batches"
      rm -rf $O/ck/Jcomp_s$s.skipped $O/ev/Jcomp_s$s.skipped; mv $O/ck/Jcomp_s$s $O/ck/Jcomp_s$s.skipped
      [ -d $O/ev/Jcomp_s$s ] && mv $O/ev/Jcomp_s$s $O/ev/Jcomp_s$s.skipped
      mv $O/logs/Jcomp_s$s.train.log $O/logs/Jcomp_s$s.skipped.train.log
      train Jcomp $s - --reader joint --base cross-encoder/ettin-reranker-32m-v1 $COMMON --kd_gold_only "$G" --epochs 4 --max_steps $(steps_of $s) || die "retrain Jcomp s$s"
      [ -f $O/ck/Jcomp_s$s/SKIPS ] && st "SKIPS persist for Jcomp s$s alone: report them"
    fi
  done
  touch $O/JOINT_DONE
fi
# ---------------- 3. Praw alone (as the grid's P: train_grid_serial3.sh adds --grad_ckpt for per-option; uncapped)
pgrep -f run_requests.py > /dev/null && die "an eval is still running before Praw (it must train alone)"
train Praw 0 - --reader peropt --base jhu-clsp/ettin-encoder-32m $COMMON --kd_gold_only "$G" --epochs 1 --grad_ckpt || die "train Praw s0"
# ---------------- 4. evals with no training: Praw ∥ the remaining Jcomp seeds; then sweeps (GPU free), scores
pids=()
if [ "$SERIAL_EVAL" = 1 ]; then eval_ck Praw 0; for s in $SEEDS; do eval_ck Jcomp $s; done
else
  [ -f $O/ev/Praw_s0/RAN ] || { ( eval_ck Praw 0 ) & pids+=($!); }
  for s in $SEEDS; do [ -f $O/ev/Jcomp_s$s/RAN ] || { ( eval_ck Jcomp $s ) & pids+=($!); }; done
fi
for p in ${pids[@]+"${pids[@]}"}; do wait $p; done
for s in $SEEDS; do sweep_ck Jcomp $s || die "Jcomp s$s eval"; done
sweep_ck Praw 0 || die "Praw eval"
$PY $G3/compare.py --gpu2 $GPU2 --gpu3 $O --evaldir $EV --out $O/compare.json > $O/compare.txt 2>&1 && st "compare: $(head -3 $O/compare.txt | tr '\n' ' ' | cut -c1-300)" || st "compare.py failed (run it locally after fetch)"
st "TRAIN+EVAL DONE"

# ---------------- 5. E6 GPU, GPU alone
lat(){ # lat <dir> <name> <engine> <opts...>  (THREADS, NREQ, WARM from the environment)
  local d=$1 n=$2 eng=$3; shift 3; local o=(); for kv in "$@"; do o+=(--opt "$kv"); done
  [ -s $d/$n.json ] && return 0
  $PY $E/e6/latency.py --engine $eng "${o[@]}" --requests $LAT --limit $NREQ --warmup $WARM --threads ${THREADS:-0} \
    --out $d/$n.json --per_request $d/$n.per.jsonl >> $O/logs/e6.log 2>&1 || st "FAIL e6 $n"
}
if [ ! -f $O/e6gpu/summary.json ]; then
  pgrep -f 'run_requests.py|train_ablate.py' > /dev/null && die "E6 GPU: the GPU is not idle"
  st "E6 GPU start (GPU alone) $(nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader)"
  for pass in a b; do
    order="v2 po pob"; [ $pass = b ] && order="pob po v2"
    for x in $order; do case $x in
      v2)  NREQ=$LAT_N WARM=20 lat $O/e6gpu crivo-32m-v2.$pass crivo path=$CODE ;;
      po)  NREQ=$LAT_N WARM=20 lat $O/e6gpu P_s0-per-question.$pass peropt:PerOptionEngine path=$PS0 ;;
      pob) NREQ=$LAT_N WARM=20 lat $O/e6gpu P_s0-request-batched.$pass peropt:PerOptionEngine path=$PS0 request_batched=1 ;;
    esac; done
  done
  $PY $G3/e6_common.py --dir $O/e6gpu --out $O/e6gpu/summary.json > $O/e6gpu/summary.txt 2>&1
  st "E6 GPU done ($(head -1 $O/e6gpu/summary.txt))"
fi
# ---------------- 6. E6 CPU, GPU idle, CUDA hidden, nothing else running
if [ "$SKIP_E6CPU" = 1 ]; then st "E6 CPU skipped (SKIP_E6CPU=1: GPU-box only)"
elif [ ! -f $O/e6cpu/summary.json ]; then
  pgrep -f 'run_requests.py|train_ablate.py|e6/latency.py' > /dev/null && die "E6 CPU: another job is running"
  { echo "nproc $(nproc 2>/dev/null || sysctl -n hw.ncpu)"; lscpu 2>/dev/null || sysctl -n machdep.cpu.brand_string; echo; free -g; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv; } > $O/e6cpu/cpu.txt 2>&1
  st "E6 CPU start: $CPU_THREADS threads, first $CPU_N requests of lat1000 for every system; $(grep -m1 'Model name' $O/e6cpu/cpu.txt | sed 's/  */ /g')"
  export CUDA_VISIBLE_DEVICES=""
  for m in $CPU_CRIVOS; do
    v=CRIVO_SRC_$m; src=${!v:-}   # override: dry runs only
    [ -n "$src" ] || { if [ $m = 32m ]; then src=$CODE; else src=$(snap tardellirs/crivo-$m); fi; }   # HF main snapshots
    THREADS=$CPU_THREADS NREQ=$CPU_N WARM=10 lat $O/e6cpu crivo-$m crivo path=$src
  done
  THREADS=$CPU_THREADS NREQ=$CPU_N WARM=10 lat $O/e6cpu crivo-32m-per-option peropt:PerOptionEngine path=$PS0 request_batched=1
  THREADS=$CPU_THREADS NREQ=$CPU_N WARM=10 lat $O/e6cpu reranker-32m reranker model=cross-encoder/ettin-reranker-32m-v1
  [ "$CPU_DEB" = 1 ] && THREADS=$CPU_THREADS NREQ=$CPU_N WARM=10 lat $O/e6cpu nli-deberta-v3-large-zs2 nli model=$DEB
  unset CUDA_VISIBLE_DEVICES
  $PY $G3/e6_common.py --dir $O/e6cpu --out $O/e6cpu/summary.json > $O/e6cpu/summary.txt 2>&1
  st "E6 CPU done ($(head -1 $O/e6cpu/summary.txt))"
fi
st "ALL DONE"
