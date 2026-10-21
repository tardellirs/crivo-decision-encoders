#!/bin/bash
# GPU-3 box setup (Ubuntu with NVIDIA driver >= 570). gpu/setup_gpu.sh without the benchmark downloads and request
# builders (GPU-3 gets its eval sets, E4 files, lat1000, sub150k and S1-Arena data/code pushed by gpu3/launch.sh, md5
# checked by run_gpu3.sh) and with a full prefetch: jhu-clsp/ettin-encoder-32m ships only pytorch_model.bin, which
# setup_gpu.sh's patterns (*.safetensors) skipped -> GPU-2 try 1 Jraw failed. Idempotent.
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
for r, rev, pat in [("tardellirs/crivo-17m", None, W), ("tardellirs/crivo-32m", "c14221dd5cb4953fa83dc40ccf3b90b37126668e", W),
                    ("tardellirs/crivo-68m", None, W), ("tardellirs/crivo-150m", None, W), ("tardellirs/crivo-400m", None, W),
                    ("tardellirs/crivo-1b", None, W), ("cross-encoder/ettin-reranker-32m-v1", None, W),
                    ("jhu-clsp/ettin-encoder-32m", None, W + ["*.bin"]),          # weights are pytorch_model.bin only
                    ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0", None, W + ["*.bin"])]:
    print("ok", r, rev, s(r, revision=rev, allow_patterns=pat), flush=True)
PY
~/venv/bin/python -c "import torch, transformers; print('ok', torch.__version__, torch.cuda.get_device_name(0), transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
