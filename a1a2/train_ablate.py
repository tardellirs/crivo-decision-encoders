"""Training for the A1/A2 grid: train/train.py's recipe (same rows, loss CE + w*Brier with optional KD, AdamW,
warmup + linear decay, option-order shuffling, length-bucketed batches) with one switch for the reader:

    --reader joint    the released Crivo reader (all options in one sequence)                    [= train/train.py]
    --reader peropt   each option in its own sequence, softmax across the n sequences (A1)

A2 switches are train.py's own flags: --brier 0 (no Brier), no --teacher (no distillation), --base
jhu-clsp/ettin-encoder-32m (raw Ettin encoder instead of the reranker).
Batching: the token budget counts every sequence, so a per-option batch holds the same number of tokens as a joint
one (fewer questions per step); --max_tokens is the same for both readers, and both run the same number of epochs
over the same questions, so per-option training costs ~n x the compute of joint (reported, part of A1's finding).

    python train_ablate.py --reader peropt --base cross-encoder/ettin-reranker-32m-v1 --train sub.jsonl.gz \
        --teacher sub.t400.npz --kd_gold_only "$(cat gold_only)" --out ck/p_s0 --seed 0 ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]   # repository root; train/ = vendored training code
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from model import QUESTION_TYPES, Encoder, build, collate, option_texts  # noqa: E402
from peropt import forward_peropt, peropt_items  # noqa: E402


class PerOptRows(T.Rows):
    def __getitem__(self, i):
        r = self.rows[i]
        texts, _ = option_texts(r["type"], r["criteria"], r.get("instructions"))
        target = list(r["target"])
        order = list(range(len(texts)))
        if self.shuffle and r["type"] == "choice":
            random.shuffle(order)
        texts = [texts[k] for k in order]; target = [target[k] for k in order]
        longest_opt = max(len(self.enc.ids(t)) + 1 for t in texts)
        instr = len(self.enc.ids(str(r.get("instructions") or "")))
        budget = self.max_len - longest_opt - instr - 3   # same state budget rule as joint, per sequence
        if budget < 16:
            return None
        items = peropt_items(self.enc, r.get("instructions"), r.get("state"), texts, QUESTION_TYPES[r["type"]], state_budget=budget)
        t, a = None, 1.0
        if self.teacher is not None and "_t" in r and r.get("kd", True) and not str(r.get("src", "")).startswith(self.gold_only):
            t = self.teacher.get(r["_t"])
            t, a = ([t[k] for k in order], self.alpha) if t is not None and len(t) == len(order) else (None, 1.0)
        return items, target, r.get("src", "?"), t, a


class PerOptCollate:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if not batch:
            return None
        flat = [it for x in batch for it in x[0]]
        out = {"flat": collate(flat, self.pad_id), "counts": [len(x[0]) for x in batch]}
        n = max(out["counts"])
        tgt = torch.zeros((len(batch), n)); tlog = torch.zeros((len(batch), n)); alpha = torch.ones(len(batch))
        mask = torch.zeros((len(batch), n), dtype=torch.bool)
        for j, x in enumerate(batch):
            tgt[j, :len(x[1])] = torch.tensor(x[1], dtype=torch.float); mask[j, :len(x[1])] = True
            if x[3] is not None:
                tlog[j, :len(x[3])] = torch.tensor(x[3], dtype=torch.float); alpha[j] = x[4]
        out.update(target=tgt / tgt.sum(-1, keepdim=True).clamp(min=1e-9), teacher=tlog, alpha=alpha, opt_mask=mask,
                   src=[x[2] for x in batch])
        return out


def batches(rows, a, seed, peropt):
    if not peropt:
        return T.length_batches(rows, a.max_tokens, a.max_items, seed)
    # per-option: a row's estimated length is n_options x (state + instructions + one option)
    est_rows = []
    for r in rows:
        n = 2 if r["type"] == "noul" else len(r["criteria"])
        est_rows.append({"state": [r.get("state")] * n, "criteria": r["criteria"], "instructions": [r.get("instructions")] * n})
    return T.length_batches(est_rows, a.max_tokens, a.max_items, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", choices=["joint", "peropt"], default="joint")
    ap.add_argument("--base", default="cross-encoder/ettin-reranker-32m-v1")
    ap.add_argument("--train", required=True); ap.add_argument("--dev"); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--head_lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=2048); ap.add_argument("--max_tokens", type=int, default=24576)
    ap.add_argument("--max_items", type=int, default=128); ap.add_argument("--brier", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--teacher"); ap.add_argument("--kd_alpha", type=float, default=0.5)
    ap.add_argument("--kd_tau_multi", type=float, default=2.0); ap.add_argument("--kd_tau_bin", type=float, default=1.0)
    ap.add_argument("--kd_gold_only", default="t1/,extra/home,bekko/bpomp")
    ap.add_argument("--skip_oom", action="store_true"); ap.add_argument("--grad_ckpt", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # cap this process so an eval lane can share the GPU; never for per-option runs (capped at 0.75 they dropped batches
    # to OOM even with --grad_ckpt; an eval that OOMs is rerun, a dropped training batch is not)
    if dev == "cuda" and os.environ.get("CUDA_MEMFRAC") and a.reader != "peropt":
        torch.cuda.set_per_process_memory_fraction(float(os.environ["CUDA_MEMFRAC"]))
    oom_skips = 0
    peropt = a.reader == "peropt"
    model, tok = build(a.base)
    if a.grad_ckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(dev).train()
    enc = Encoder(tok)
    rows = T.read_rows(a.train)
    teacher = None
    if a.teacher:
        teacher = T.Teacher(a.teacher); assert len(teacher) == len(rows)
        for i, r in enumerate(rows):
            r["_t"] = i
    if a.limit:
        rows = rows[:a.limit]   # fixed prefix (the subset file is already shuffled): same rows for every seed
    gold = tuple(x for x in a.kd_gold_only.split(",") if x)
    RowsCls, Coll = (PerOptRows, PerOptCollate) if peropt else (T.Rows, T.Collate)
    ds = RowsCls(rows, enc, a.max_len, shuffle_options=True, teacher=teacher, alpha=a.kd_alpha, gold_only=gold)
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    decay = [p for n, p in model.encoder.named_parameters() if p.ndim >= 2 and "norm" not in n]
    nodecay = [p for n, p in model.encoder.named_parameters() if not (p.ndim >= 2 and "norm" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "lr": a.lr, "weight_decay": a.wd}, {"params": nodecay, "lr": a.lr, "weight_decay": 0.0},
                             {"params": head, "lr": a.head_lr, "weight_decay": 0.0}], betas=(0.9, 0.98), eps=1e-6, fused=dev == "cuda")
    base_lrs = [g["lr"] for g in opt.param_groups]
    per_epoch = len(batches(rows, a, a.seed, peropt))
    total = int(per_epoch * a.epochs)
    if a.max_steps:
        total = min(total, a.max_steps)
    warm = max(1, int(total * a.warmup))
    print(json.dumps({"reader": a.reader, "rows": len(rows), "steps": total, "per_epoch": per_epoch, "base": a.base,
                      "brier": a.brier, "kd": bool(a.teacher), "seed": a.seed}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    log = open(str(a.out) + ".log.jsonl", "a")
    step, t0, run, toks = 0, time.time(), defaultdict(float), 0
    tok_total, q_total, t_start = 0, 0, time.time()   # whole-run counters (GPU-3: compute matching is reported)
    for ep in range(math.ceil(a.epochs)):
        loader = DataLoader(ds, batch_sampler=batches(rows, a, a.seed + ep, peropt), collate_fn=Coll(tok.pad_token_id),
                            num_workers=a.workers, prefetch_factor=4 if a.workers else None)
        for batch in loader:
            if step >= total:
                break
            if batch is None:
                continue
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    if peropt:
                        fb = {k: v.to(dev) for k, v in batch["flat"].items()}
                        logits = forward_peropt(model, fb, batch["counts"])
                        b = {"opt_mask": batch["opt_mask"].to(dev), "target": batch["target"].to(dev),
                             "teacher": batch["teacher"].to(dev), "alpha": batch["alpha"].to(dev)}
                        ntok = int(fb["attention_mask"].sum())
                    else:
                        b = {k: v.to(dev) for k, v in batch.items() if k != "src"}
                        logits = model(b["input_ids"], b["attention_mask"], b["opt_positions"], b["opt_mask"], b["qtype"])
                        ntok = int(b["attention_mask"].sum())
                loss, _, _ = T.loss_fn(logits.float(), b, a.brier, a.kd_tau_multi, a.kd_tau_bin)
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                if not a.skip_oom:
                    raise
                oom_skips += 1
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); continue
            frac = step / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * frac
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            run["loss"] += loss.item(); run["n"] += 1; run["q"] += len(batch["src"]); toks += ntok
            tok_total += ntok; q_total += len(batch["src"])
            if step % 50 == 0 or step == total:
                el = time.time() - t0
                rec = {"step": step, "total": total, "loss": run["loss"] / run["n"], "gn": float(gn), "q_per_s": run["q"] / el,
                       "tok_per_s": toks / el, "oom_skips": oom_skips, "eta_min": (total - step) * el / max(1, run["n"]) / 60}
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                run, toks, t0 = defaultdict(float), 0, time.time()
    tot = {"tokens_total": tok_total, "questions_total": q_total, "train_seconds": round(time.time() - t_start, 1), "total_planned": total}
    T.save(model, tok, a.out, {"base": a.base, "step": step, "reader": a.reader, "oom_skips": oom_skips, **tot, "args": vars(a)})
    print(json.dumps({"saved": a.out, "steps": step, "oom_skips": oom_skips, **tot}), flush=True)


if __name__ == "__main__":
    main()
