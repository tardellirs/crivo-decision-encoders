#!/bin/bash
# GPU box setup (any Ubuntu image with NVIDIA drivers >= 570; tested on an A100 40 GB cloud image).
# Expects ~/repo (this repository) and ~/decision-index (the kit) pushed by gpu/launch.sh. Idempotent.
# Env: REPO (default: this script's parent dir), KIT (default ~/decision-index), HF_TOKEN (gated Gemma only).
set -euo pipefail
cd ~
sudo -n true 2>/dev/null && SUDO=sudo || SUDO=
$SUDO apt-get -qq update >/dev/null 2>&1 || true; $SUDO apt-get -qq install -y python3-venv tmux rsync >/dev/null 2>&1 || true
# Python 3.12 venv via uv (some cloud images ship Python 3.10; transformers 5.19 needs >= 3.11)
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
if [ -x ~/venv/bin/python ] && ! ~/venv/bin/python -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then rm -rf ~/venv; fi
[ -x ~/venv/bin/python ] || uv venv --seed --python 3.12 ~/venv
P=~/venv/bin/pip
$P -q install -U pip setuptools wheel
$P -q install torch --index-url https://download.pytorch.org/whl/cu128   # driver 570 = CUDA 12.8; the PyPI wheel is cu130
$P -q install "transformers==5.19.0" safetensors accelerate datasets pyarrow pandas numpy scipy scikit-learn \
  huggingface_hub sentencepiece protobuf tiktoken chess==1.11.2 mido httpx pyyaml pydantic
KIT=${KIT:-$HOME/decision-index}
$P -q install -e $KIT --no-deps
export HF_HUB_DISABLE_XET=1
# benchmarks: BTZSC test parquet (no aggregate configs), S1-Arena HF snapshot + code (for its metrics)
~/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download as s
s("btzsc/btzsc", repo_type="dataset", local_dir=__import__("os").path.expanduser("~/btzsc/data"),
  allow_patterns=["*/test-*.parquet", "README.md"], ignore_patterns=["all/*", "topic/*", "emotion/*", "sentiment/*", "intent/*"])
s("C-Tianyu/S1-Arena", repo_type="dataset", local_dir=__import__("os").path.expanduser("~/s1arena/data"))
PY
[ -d ~/s1arena/code ] || git clone -q --depth 1 https://github.com/TianyuCodings/S1-Arena.git ~/s1arena/code
# request files (same builders as on the CPU VMs)
cd ${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
mkdir -p ~/req
[ -s ~/req/btzsc_full.jsonl ] || ~/venv/bin/python btzsc/adapter.py requests --root ~/btzsc/data --out ~/req/btzsc_full.jsonl 2>/dev/null
[ -s ~/req/s1_all.jsonl ] || ~/venv/bin/python s1arena/adapter.py requests --data ~/s1arena/data --out ~/req/s1_all.jsonl
# models (public HF; gemma-3-1b-it is gated: HF_TOKEN must be set in the environment for that one)
~/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download as s
for r, rev in [("tardellirs/crivo-17m", None), ("tardellirs/crivo-32m", None), ("tardellirs/crivo-32m", "bbf3f1924523f530b3a9e621f2f20f314ea462aa"),
               ("tardellirs/crivo-68m", None), ("tardellirs/crivo-150m", None), ("tardellirs/crivo-150m", "7a6c7453c06be576e1b48aeb70429318402791d7"),
               ("tardellirs/crivo-400m", None), ("tardellirs/crivo-1b", None), ("tardellirs/prumo", None),
               ("cross-encoder/ettin-reranker-17m-v1", None), ("cross-encoder/ettin-reranker-32m-v1", None),
               ("jhu-clsp/ettin-encoder-32m", None),
               ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0", None), ("MoritzLaurer/ModernBERT-large-zeroshot-v2.0", None),
               ("facebook/bart-large-mnli", None), ("Qwen/Qwen3-0.6B", None), ("google/gemma-3-1b-it", None)]:
    try:
        print("ok", r, rev, s(r, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "tokenizer*", "prumo_rules/**", "tools/**"]), flush=True)
    except Exception as e:
        print("FAIL", r, rev, repr(e)[:150], flush=True)
    if r == "tardellirs/prumo":   # a real folder: Prumo's params.py resolves its path through the cache symlinks
        print("ok prumo", s(r, local_dir=__import__("os").path.expanduser("~/models/prumo")), flush=True)
PY
~/venv/bin/python -c "import torch, transformers; print('ok', torch.__version__, torch.cuda.get_device_name(0), transformers.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
