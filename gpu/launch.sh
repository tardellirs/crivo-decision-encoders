#!/bin/bash
# Launch a GPU session (GPU-1 inference and/or GPU-2 A1/A2 grid) on a box you started yourself.
# This script never creates or terminates an instance and uploads nothing except to the box you name.
#
#   bash gpu/launch.sh <user@host> infer|grid|both
#
#   infer: E1 (BTZSC), E2 (S1-Arena), E4 (robustness), then E6 (latency, GPU alone), then scoring -> ~/results/
#   grid : A1/A2 grid at 32M (arms x seeds of a1a2/grid.tsv, plus evaluations)                     -> ~/results/a1a2/
#
# Environment (local side):
#   KIT        local checkout of the Decision Index kit with the frozen public suite built at $KIT/suite-0.3
#              (https://github.com/apolinario/decision-index); required.
#   DATA_DIR   grid only: directory holding mix400_s1.train.jsonl.gz, mix400_s1.t400.npz and
#              mix400_s1.t400.gold_only (training data and Crivo-400M teacher logits; not distributed).
#   HF_TOKEN   optional; only needed to download the gated google/gemma-3-1b-it baseline. Passed to the box for the
#              download step only.
# Remote layout: code in ~/repo, kit in ~/decision-index, venv in ~/venv, results in ~/results.
# Monitor ~/results/STATUS (and ~/results/a1a2/STATUS) and fetch with: rsync -a <host>:results/ results_gpu/
set -euo pipefail
H=$1; WHAT=${2:-both}
REPO_LOCAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
KIT=${KIT:?set KIT to your local decision-index checkout}
SSH="ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15"
$SSH $H 'mkdir -p ~/repo ~/decision-index ~/exp/a1a2 ~/e4 ~/evalsets'
rsync -az -e "$SSH" --exclude work --exclude '__pycache__' --exclude '*.log' $KIT/ $H:decision-index/
rsync -az -e "$SSH" --exclude '__pycache__' --exclude '.git' --exclude 'results_*' --exclude 'audit_raw' $REPO_LOCAL/ $H:repo/
if [ "$WHAT" != infer ]; then   # grid inputs: stage-1 data + Crivo-400M teacher logits + gold-only list
  D=${DATA_DIR:?set DATA_DIR for the grid}
  rsync -az -e "$SSH" $D/mix400_s1.train.jsonl.gz $D/mix400_s1.t400.npz $D/mix400_s1.t400.gold_only $H:exp/a1a2/
fi
$SSH $H "HF_TOKEN=${HF_TOKEN:-} bash ~/repo/gpu/setup_gpu.sh > ~/setup.log 2>&1; tail -3 ~/setup.log"
$SSH $H 'cd ~/repo && PY=~/venv/bin/python && K=~/decision-index &&
  { [ -s ~/e4/req/orig.jsonl ] || { $PY e4/perturb.py sample --suite $K/suite-0.3 --out ~/e4/e4_orig.jsonl && $PY e4/perturb.py make --rows ~/e4/e4_orig.jsonl --outdir ~/e4/req; }; } &&
  { [ -s ~/evalsets/s1_60.jsonl ] || $PY a1a2/build_eval.py --suite $K/suite-0.3 --btzsc ~/btzsc/data --s1 ~/s1arena/data --outdir ~/evalsets; }'
if [ "$WHAT" != infer ]; then
  $SSH $H 'cd ~/exp/a1a2 && cp mix400_s1.t400.gold_only gold_only && [ -s sub150k.jsonl.gz ] || ~/venv/bin/python ~/repo/a1a2/make_subset.py --train mix400_s1.train.jsonl.gz --teacher mix400_s1.t400.npz --n 150000 --seed 2026 --out sub150k.jsonl.gz --out_teacher sub150k.t400.npz'
fi
# GPU-2 used a1a2/train_grid_serial3.sh (serial trainings, see its header), then a1a2/jsteps.sh and a1a2/eval_P.sh.
GRID=${GRID_SCRIPT:-train_grid_serial3.sh}
case $WHAT in
  infer) $SSH $H 'tmux new -d -s infer "bash ~/repo/gpu/run_lanes.sh > ~/results_lanes.log 2>&1"' ;;
  grid)  $SSH $H "tmux new -d -s grid \"bash ~/repo/a1a2/$GRID > ~/results_grid.log 2>&1\"" ;;
  both)  $SSH $H "tmux new -d -s infer \"bash ~/repo/gpu/run_lanes.sh > ~/results_lanes.log 2>&1\"; tmux new -d -s grid \"bash ~/repo/a1a2/$GRID > ~/results_grid.log 2>&1\"" ;;
esac
$SSH $H 'sleep 30; tail -3 ~/results/STATUS ~/results/a1a2/STATUS 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
