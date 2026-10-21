#!/bin/bash
# GPU-4 box setup (Ubuntu with NVIDIA driver >= 570). gpu3/setup_gpu3.sh with the minimal prefetch: only the raw
# encoder jhu-clsp/ettin-encoder-32m (weights + tokenizer; it ships only pytorch_model.bin, hence "*.bin") and the
# released crivo-32m snapshot c14221dd (crivo.py/model.py used as code= by the joint eval). No reranker, DeBERTa or
# larger Crivos. Inputs, eval sets and S1-Arena data/code are pushed by gpu4/launch.sh and md5-checked by run_gpu4.sh.
# Idempotent.
set -euo pipefail
cd ~
sudo -n true 2>/dev/null && SUDO=sudo || SUDO=
$SUDO apt-get -qq update >/dev/null 2>&1 || true; $SUDO apt-get -qq install -y python3-venv tmux rsync >/dev/null 2>&1 || true
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
if [ -x ~/venv/bin/python ] && ! ~/venv/bin/python -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then rm -rf ~/venv; fi
[ -x ~/venv/bin/python ] || uv venv --seed --python 3.12 ~/venv
P=~/venv/bin/pip
$P -q install -U pip setuptools wheel
$P -q install torch --index-url https://download.pytorch.org/whl/cu128
$P -q install "transformers==5.19.0" safetensors accelerate datasets pyarrow pandas numpy scipy scikit-learn \
  huggingface_hub sentencepiece protobuf tiktoken pyyaml pydantic httpx
$P -q install -e ${KIT:-$HOME/decision-index} --no-deps
export HF_HUB_DISABLE_XET=1
~/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download as s
W = ["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "tokenizer*"]
for r, rev, pat in [("tardellirs/crivo-32m", "c14221dd5cb4953fa83dc40ccf3b90b37126668e", W),
                    ("jhu-clsp/ettin-encoder-32m", None, W + ["*.bin"])]:          # weights are pytorch_model.bin only
    print("ok", r, rev, s(r, revision=rev, allow_patterns=pat), flush=True)
PY
~/venv/bin/python -c "import torch, transformers; print('ok', torch.__version__, torch.cuda.get_device_name(0), transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
