#!/bin/bash
# GPU-4 launch (initialisation 2x2 of the reader ablation: JrawComp s0-s2, Praw s1-s2) on a box you started yourself.
# This script never creates or terminates an instance. Keep a monitor running from the first rsync:
#   bash gpu4/launch.sh <user@host>            push, setup, start tmux gpu4
#   bash gpu4/launch.sh <user@host> status     STATUS tail, GPU use, last train line
#   bash gpu4/launch.sh <user@host> fetch      -> results_gpu4/ (+ md5 check against the box)
# Environment (local side):
#   KIT      local Decision Index kit checkout (https://github.com/apolinario/decision-index); required.
#   INPUTS   GPU-3 inputs directory (default gpu3/inputs/, not distributed; md5s in gpu4/inputs.md5.box).
#   S1_DATA  local S1-Arena HF snapshot; S1_CODE  local clone of TianyuCodings/S1-Arena.
#   GPU3_CK  optional: GPU-3 checkpoint dir (only its */config.json files are pushed, for training-token counts).
# Remote layout: code in ~/repo, kit in ~/decision-index, results in ~/results/gpu4.
set -euo pipefail
H=$1; WHAT=${2:-start}
X=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); I=${INPUTS:-$X/gpu3/inputs}   # GPU-4 reuses GPU-3's inputs
SSH="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15"
case $WHAT in
status)
  $SSH $H 'tail -12 ~/results/gpu4/STATUS; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; f=$(ls -t ~/results/gpu4/logs/*.train.log 2>/dev/null | head -1); [ -n "$f" ] && { echo $f; grep "^{\"step\"" $f | tail -1; }'; exit 0;;
fetch)
  mkdir -p $X/results_gpu4
  rsync -a -e "$SSH" $H:results/gpu4/ $X/results_gpu4/
  $SSH $H 'cd ~/results/gpu4 && find . -type f -print0 | xargs -0 md5sum' > $X/results_gpu4/box.md5
  python3 - $X/results_gpu4 <<'PY'
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
KIT=${KIT:?set KIT}; S1_DATA=${S1_DATA:?set S1_DATA}; S1_CODE=${S1_CODE:?set S1_CODE}
$SSH $H 'mkdir -p ~/repo ~/decision-index ~/exp/a1a2 ~/exp/gpu2_ref ~/exp/gpu3_ref ~/evalsets ~/s1arena ~/results/gpu4'
rsync -az -e "$SSH" --exclude work --exclude '__pycache__' --exclude '*.log' $KIT/ $H:decision-index/
rsync -az -e "$SSH" --exclude '__pycache__' --exclude 'audit_raw' --exclude 'results_*' --exclude 'stats' --exclude 'e5' --exclude 'gpu3/inputs' --exclude '.git' --exclude 'docs' $X/ $H:repo/   # code only
rsync -az -e "$SSH" $I/sub150k.jsonl.gz $I/sub150k.t400.npz $I/gold_only $H:exp/a1a2/
rsync -az -e "$SSH" $I/evalsets/ $H:evalsets/
rsync -az -e "$SSH" $S1_DATA/ $H:s1arena/data/
rsync -az -e "$SSH" $S1_CODE/ $H:s1arena/code/
# GPU-2 (P) and GPU-3 (Jcomp, Praw s0) score files (no checkpoints, no outputs) so compare.py can run on the box
rsync -az -e "$SSH" --include '*/' --include '*.score.json' --include 'flips.json' --include 'DONE' --exclude '*' $X/results_gpu2/ev $H:exp/gpu2_ref/
rsync -az -e "$SSH" --include '*/' --include '*.score.json' --include 'flips.json' --include 'DONE' --include 'config.json' --exclude '*' $X/results_gpu3/ev ${GPU3_CK:-} $H:exp/gpu3_ref/
$SSH $H "bash ~/repo/gpu4/setup_gpu4.sh > ~/setup_gpu4.log 2>&1; tail -6 ~/setup_gpu4.log"
$SSH $H 'tmux new -d -s gpu4 "bash ~/repo/gpu4/run_gpu4.sh > ~/results/gpu4/run.log 2>&1"'
$SSH $H 'sleep 90; tail -5 ~/results/gpu4/STATUS; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
