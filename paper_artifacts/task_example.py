#!/usr/bin/env python3
"""Run a released Crivo once, on CPU, on the invented request drawn in fig_task (task illustration).

    python3 paper_artifacts/task_example.py 17m     # CPU, python with torch + transformers 5.19.0; ~5 s, < 1 GB
    python3 paper_artifacts/task_example.py 150m    # ~10 s, ~1.5 GB

Adds the model's entry to paper_artifacts/task_example.json (request, answers, weights hash, CPU wall time); the figure draws the
entry named FIG_MODEL in make_figs.py. make_figs.py only READS that file, so the
normal build stays light and needs no torch. The request is invented (no customer data, no brand names); the
probabilities in the figure are this model's real output on it, not a benchmark result.
"""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODELS = {"17m": ("Crivo-17M", "tardellirs/crivo-17m"), "150m": ("Crivo-150M", "tardellirs/crivo-150m")}   # released packages
NAME, REPO_ID = MODELS[sys.argv[1] if len(sys.argv) > 1 else "17m"]
from huggingface_hub import snapshot_download  # noqa: E402
W = Path(snapshot_download(REPO_ID))
sys.path.insert(0, str(W))
from crivo import Crivo  # noqa: E402

STATE = {"channel": "chat", "plan": "basic",
         "message": "I was charged twice for this month's subscription and now the app says my account is locked. "
                    "Please refund the extra charge, I need this fixed today."}
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "charges, invoices and refunds", "technical": "app errors and bugs",
                          "security": "locked accounts and suspicious logins", "sales": "plans and upgrades"}},
    "refund": {"type": "noul", "instructions": "The customer asks for money back."},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                "criteria": {"1": "low", "2": "normal", "3": "high", "4": "critical"}},
}

torch.set_num_threads(4)
m = Crivo(W, device="cpu")
m.answer(STATE, QUESTIONS)                       # warm-up
t0 = time.perf_counter()
ans = m.answer(STATE, QUESTIONS)
ms = 1000 * (time.perf_counter() - t0)
sha = hashlib.sha256(open(W / "model.safetensors", "rb").read()).hexdigest()
J = HERE / "task_example.json"
out = json.load(open(J)) if J.exists() else {}
if out.get("state") not in (None, STATE) or out.get("questions") not in (None, QUESTIONS):
    out = {}                                     # request changed: start over
out.update({"state": STATE, "questions": QUESTIONS, "note": "invented request; answers are each model's real output"})
out.setdefault("runs", {})[NAME] = {"weights": REPO_ID, "sha256": sha, "device": "cpu",
                                    "threads": 4, "machine": platform.machine(), "wall_ms_one_call": round(ms, 1),
                                    "answers": ans}
json.dump(out, open(J, "w"), indent=1)
print(json.dumps(ans, indent=1), f"{ms:.1f} ms")
