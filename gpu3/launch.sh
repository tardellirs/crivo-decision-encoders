#!/bin/bash
# GPU-3 launch (compute-matched joint control Jcomp, per-option-from-raw-encoder Praw, E6 GPU and CPU latency) on a
# box you started yourself. This script never creates or terminates an instance. Keep a monitor running from the
# first rsync so the GPU is never idle:
#   bash gpu3/launch.sh <user@host>            push, setup, start tmux gpu3
#   bash gpu3/launch.sh <user@host> status     STATUS tail, GPU use, last train line
#   bash gpu3/launch.sh <user@host> fetch      -> results_gpu3/ (+ md5 check against the box)
#
# Environment (local side):
#   KIT        local Decision Index kit checkout (https://github.com/apolinario/decision-index); required.
#   INPUTS     directory with the GPU-3 inputs (default gpu3/inputs/, not distributed; rebuild it as described in
#              README "GPU-3"): sub150k.jsonl.gz, sub150k.t400.npz, gold_only, evalsets/{pub15,btzsc500,s1_60}.jsonl,
#              e4/{orig,perm1,perm2,perm3,rev,yn}.jsonl, lat1000.jsonl. md5s are in gpu3/inputs.md5.box.
#   S1_DATA    local S1-Arena HF snapshot (C-Tianyu/S1-Arena); S1_CODE  local clone of TianyuCodings/S1-Arena.
#   P_S0       the GPU-2 per-option checkpoint P_s0 (E6 latency).
#   GPU2_EV    results_gpu2/ev (score files of the GPU-2 grid, for compare.py on the box; default: this repo's copy).
# Remote layout: code in ~/repo, kit in ~/decision-index, results in ~/results/gpu3.
set -euo pipefail
H=$1; WHAT=${2:-start}
X=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); I=${INPUTS:-$X/gpu3/inputs}
SSH="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15"
case $WHAT in
status)
  $SSH $H 'tail -12 ~/results/gpu3/STATUS; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; f=$(ls -t ~/results/gpu3/logs/*.train.log 2>/dev/null | head -1); [ -n "$f" ] && { echo $f; grep "^{\"step\"" $f | tail -1; }'; exit 0;;
fetch)
  mkdir -p $X/results_gpu3
  rsync -a -e "$SSH" $H:results/gpu3/ $X/results_gpu3/
  $SSH $H 'cd ~/results/gpu3 && find . -type f -print0 | xargs -0 md5sum' > $X/results_gpu3/box.md5
  python3 - $X/results_gpu3 <<'PY'
import hashlib, os, sys
bad = n = 0
for line in open(os.path.join(sys.argv[1], "box.md5")):
    h, f = line.split(None, 1); f = os.path.join(sys.argv[1], f.strip()); n += 1
    if not os.path.exists(f) or hashlib.md5(open(f, "rb").read()).hexdigest() != h:
        bad += 1; print("BAD", f)
print(f"md5: {n - bad}/{n} files ok" + ("" if not bad else "  -> fetch again before terminating"))
PY
  exit 0;;
esac
KIT=${KIT:?set KIT}; S1_DATA=${S1_DATA:?set S1_DATA}; S1_CODE=${S1_CODE:?set S1_CODE}; P_S0=${P_S0:?set P_S0}
GPU2_EV=${GPU2_EV:-$X/results_gpu2/ev}
$SSH $H 'mkdir -p ~/repo ~/decision-index ~/exp/a1a2 ~/exp/gpu2_ref ~/evalsets ~/e4/req ~/req ~/s1arena ~/results/gpu3'
rsync -az -e "$SSH" --exclude work --exclude '__pycache__' --exclude '*.log' $KIT/ $H:decision-index/
rsync -az -e "$SSH" --exclude '__pycache__' --exclude '.git' --exclude 'audit_raw' --exclude 'results_*' --exclude 'gpu3/inputs' $X/ $H:repo/
rsync -az -e "$SSH" $I/sub150k.jsonl.gz $I/sub150k.t400.npz $I/gold_only $H:exp/a1a2/
rsync -az -e "$SSH" $I/evalsets/ $H:evalsets/
rsync -az -e "$SSH" $I/e4/ $H:e4/req/
rsync -az -e "$SSH" $I/lat1000.jsonl $H:req/
rsync -az -e "$SSH" $S1_DATA/ $H:s1arena/data/
rsync -az -e "$SSH" $S1_CODE/ $H:s1arena/code/
rsync -az -e "$SSH" $P_S0/ $H:exp/P_s0/
# GPU-2 score files (no checkpoints) so compare.py can run on the box
rsync -az -e "$SSH" --include '*/' --include '*.score.json' --include 'flips.json' --include 'DONE' --exclude '*' $GPU2_EV $H:exp/gpu2_ref/
$SSH $H "bash ~/repo/gpu3/setup_gpu3.sh > ~/setup_gpu3.log 2>&1; tail -14 ~/setup_gpu3.log"
$SSH $H 'tmux new -d -s gpu3 "bash ~/repo/gpu3/run_gpu3.sh > ~/results/gpu3/run.log 2>&1"'
$SSH $H 'sleep 90; tail -5 ~/results/gpu3/STATUS; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
