#!/bin/bash
# CPU machine setup for the CPU experiments (contamination audit, Prumo runs, E5, CPU latency):
# venv with CPU torch + transformers 5.19.0 + the Decision Index kit. Ubuntu 24.04 (Python 3.12); uses sudo if needed.
# Env: KIT (default ~/decision-index), VENV (default ~/venv).
set -e
export DEBIAN_FRONTEND=noninteractive
sudo -n true 2>/dev/null && SUDO=sudo || SUDO=
$SUDO apt-get -qq update >/dev/null 2>&1 || true
$SUDO apt-get -qq install -y python3-venv python3-dev build-essential rsync tmux pigz jq >/dev/null 2>&1 || true
V=${VENV:-$HOME/venv}; KIT=${KIT:-$HOME/decision-index}
[ -x $V/bin/python ] || python3 -m venv $V
P=$V/bin/pip
$P -q install -U pip setuptools wheel
$P -q install torch --index-url https://download.pytorch.org/whl/cpu
$P -q install "transformers==5.19.0" safetensors accelerate datasets pyarrow pandas numpy scipy scikit-learn \
  huggingface_hub sentencepiece protobuf tiktoken chess==1.11.2 mido httpx datasketch pyyaml
[ -d $KIT ] && $P -q install -e "$KIT" --no-deps || echo "kit not found at $KIT (needed for Prumo and the public suite)"
$V/bin/python -c "import torch, transformers; print('ok', torch.__version__, transformers.__version__, torch.get_num_threads())"
