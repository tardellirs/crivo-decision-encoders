"""Train the typed-decision encoder on unified rows.

Row format (jsonl or jsonl.gz), one question per row, criteria exactly as in the Decision Index wire format:
    {"type": "choice"|"noul"|"score", "state": ..., "instructions": ..., "criteria": ...,
     "target": [p_1, ..., p_n],   # aligned with option_texts(type, criteria); noul = [p_false, p_true]
     "src": "dataset/config"}

    python train.py --base cross-encoder/ettin-reranker-150m-v1 --train mix.jsonl.gz --dev dev.jsonl.gz --out ckpt
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from model import QUESTION_TYPES, Encoder, build, collate, option_texts, save


def save_ckpt(model, tok, out, extra):
    return save(model, tok, out, extra)


def read_rows(path: str) -> list[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


class Teacher:
    """Teacher logits per training row (original option order), written by teach.py."""

    def __init__(self, path: str):
        import numpy as np
        z = np.load(path)
        self.logits, self.offsets = z["logits"], z["offsets"]

    def __len__(self):
        return len(self.offsets) - 1

    def get(self, i):
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return None if a == b else self.logits[a:b].astype("float32").tolist()


class Rows(Dataset):
    def __init__(self, rows, enc: Encoder, max_len: int, shuffle_options: bool, teacher: "Teacher | None" = None,
                 alpha: float = 1.0, gold_only: tuple = ()):
        self.rows, self.enc, self.max_len, self.shuffle = rows, enc, max_len, shuffle_options
        self.teacher, self.alpha, self.gold_only = teacher, alpha, gold_only

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        texts, _ = option_texts(r["type"], r["criteria"], r.get("instructions"))
        target = list(r["target"])
        assert len(texts) == len(target), (r.get("src"), len(texts), len(target))
        order = list(range(len(texts)))
        if self.shuffle and r["type"] == "choice":
            random.shuffle(order)
        texts = [texts[k] for k in order]
        target = [target[k] for k in order]
        opt_ids = sum(len(self.enc.ids(t)) + 1 for t in texts)
        instr = len(self.enc.ids(str(r.get("instructions") or "")))
        budget = self.max_len - opt_ids - instr - 3
        if budget < 16:
            return None
        ids, pos = self.enc.encode(r.get("instructions"), r.get("state"), texts, state_budget=budget)
        if len(ids) > self.max_len:
            return None
        t, a = None, 1.0
        if self.teacher is not None and "_t" in r and r.get("kd", True) and not str(r.get("src", "")).startswith(self.gold_only):
            t = self.teacher.get(r["_t"])
            if t is not None and len(t) == len(order):
                t, a = [t[k] for k in order], self.alpha
            else:
                t = None
        return ids, pos, QUESTION_TYPES[r["type"]], target, r.get("src", "?"), t, a


def length_batches(rows, max_tokens: int, max_items: int, seed: int):
    """Group rows of similar (estimated) length so each batch holds ~max_tokens tokens."""
    rng = random.Random(seed)
    est = [min(len(json.dumps(r.get("state"))) + len(json.dumps(r["criteria"])) + len(str(r.get("instructions"))), 40000) // 4 + 8
           for r in rows]
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    batches = []
    chunk = 20000
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s:s + chunk], key=lambda i: est[i])
        cur, longest = [], 0
        for i in part:
            l = max(longest, est[i])
            if cur and (l * (len(cur) + 1) > max_tokens or len(cur) >= max_items):
                batches.append(cur)
                cur, l = [], est[i]
            cur.append(i)
            longest = l
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


class Collate:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, items):
        items = [x for x in items if x is not None]
        if not items:
            return None
        batch = collate([(x[0], x[1], x[2]) for x in items], self.pad_id)
        n = batch["opt_mask"].size(1)
        tgt = torch.zeros((len(items), n))
        tlog = torch.zeros((len(items), n))
        alpha = torch.ones(len(items))
        for j, x in enumerate(items):
            t = x[3]
            tgt[j, :len(t)] = torch.tensor(t, dtype=torch.float)
            if len(x) > 5 and x[5] is not None:
                tlog[j, :len(x[5])] = torch.tensor(x[5], dtype=torch.float)
                alpha[j] = x[6]
        batch["target"] = tgt / tgt.sum(-1, keepdim=True).clamp(min=1e-9)
        batch["teacher"], batch["alpha"] = tlog, alpha
        batch["src"] = [x[4] for x in items]
        return batch


def loss_fn(logits, batch, brier_w: float, tau_multi: float = 2.0, tau_bin: float = 1.0):
    mask, tgt = batch["opt_mask"], batch["target"]
    logp = logits.log_softmax(-1)
    ce = -(tgt * torch.where(mask, logp, torch.zeros_like(logp))).sum(-1)
    brier = ((logp.exp() - tgt) ** 2 * mask).sum(-1)
    gold = ce + brier_w * brier
    alpha = batch.get("alpha")
    if alpha is None or bool((alpha >= 1).all()):
        return gold.mean(), ce, brier
    k = mask.sum(-1)
    tau = torch.where(k > 2, torch.full_like(alpha, tau_multi), torch.full_like(alpha, tau_bin))[:, None]
    neg = torch.full_like(logits, -1e4)
    t_logp = torch.where(mask, batch["teacher"] / tau, neg).log_softmax(-1)
    s_logp = torch.where(mask, logits / tau, neg).log_softmax(-1)
    kl = (t_logp.exp() * (t_logp - s_logp) * mask).sum(-1) * tau.squeeze(-1) ** 2
    loss = alpha * gold + (1 - alpha) * kl
    return loss.mean(), ce, brier


@torch.no_grad()
def evaluate(model, loader, device, brier_w):
    model.eval()
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    for batch in loader:
        if batch is None:
            continue
        b = {k: v.to(device) for k, v in batch.items() if k != "src"}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits = model(b["input_ids"], b["attention_mask"], b["opt_positions"], b["opt_mask"], b["qtype"])
        _, ce, _ = loss_fn(logits, b, brier_w)
        hit = (logits.argmax(-1) == b["target"].argmax(-1)).float()
        for s, c, h in zip(batch["src"], ce.tolist(), hit.tolist()):
            for key in (s, "ALL"):
                a = agg[key]
                a[0] += 1; a[1] += c; a[2] += h
    model.train()
    return {k: {"n": n, "ce": c / n, "acc": h / n} for k, (n, c, h) in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cross-encoder/ettin-reranker-150m-v1")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--max_items", type=int, default=128)
    ap.add_argument("--brier", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--init", help="our checkpoint dir to start from (weights only)")
    ap.add_argument("--teacher", help="teacher logits .npz from teach.py, aligned with --train")
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="weight of the gold loss when a teacher is present")
    ap.add_argument("--kd_tau_multi", type=float, default=2.0)
    ap.add_argument("--kd_tau_bin", type=float, default=1.0)
    ap.add_argument("--kd_gold_only", default="t1/,extra/home,bekko/bpomp", help="src prefixes trained on gold only")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights (0 = off); saved as <out>-ema")
    ap.add_argument("--grad_accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing in the encoder")
    ap.add_argument("--skip_oom", action="store_true", help="on CUDA OOM drop the batch (and the partial accumulation) and go on")
    ap.add_argument("--save_at", type=float, nargs="*", default=[],
                    help="fractions of the run at which to save <out>-p<NN> (e.g. 0.3 for an early gate)")
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = build(a.base, attn=a.attn)
    if a.init:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(Path(a.init) / "model.safetensors")), strict=True)
        print(f"init from {a.init}", flush=True)
    if a.grad_ckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device).train()
    enc = Encoder(tok)

    rows = read_rows(a.train)
    teacher = None
    if a.teacher:
        teacher = Teacher(a.teacher)
        assert len(teacher) == len(rows), (len(teacher), len(rows))
        for i, r in enumerate(rows):
            r["_t"] = i
        print(f"teacher logits for {len(teacher):,} rows, alpha {a.kd_alpha}", flush=True)
    if a.limit:
        random.Random(a.seed).shuffle(rows); rows = rows[:a.limit]
    print(f"train rows {len(rows):,}", flush=True)
    ds = Rows(rows, enc, a.max_len, shuffle_options=True, teacher=teacher, alpha=a.kd_alpha,
              gold_only=tuple(x for x in a.kd_gold_only.split(",") if x))
    dev_loader = None
    if a.dev:
        dev = read_rows(a.dev)
        dds = Rows(dev, enc, 8192, shuffle_options=False)
        dev_loader = DataLoader(dds, batch_sampler=length_batches(dev, a.max_tokens, a.max_items, 1),
                                collate_fn=Collate(tok.pad_token_id), num_workers=min(2, a.workers))

    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    body_decay = [p for n, p in model.encoder.named_parameters() if p.ndim >= 2 and "norm" not in n]
    body_nodecay = [p for n, p in model.encoder.named_parameters() if not (p.ndim >= 2 and "norm" not in n)]
    groups = [{"params": body_decay, "lr": a.lr, "weight_decay": a.wd},
              {"params": body_nodecay, "lr": a.lr, "weight_decay": 0.0},
              {"params": head, "lr": a.head_lr, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6, fused=device == "cuda")
    base_lrs = [g["lr"] for g in opt.param_groups]

    n_epochs = math.ceil(a.epochs)
    per_epoch = len(length_batches(rows, a.max_tokens, a.max_items, a.seed))
    total = int(per_epoch * a.epochs) // a.grad_accum  # optimizer steps
    warm = max(1, int(total * a.warmup))
    print(f"steps {total:,} ({per_epoch:,} micro-batches/epoch, accum {a.grad_accum}), warmup {warm}", flush=True)

    ema = {n: p.detach().clone().float() for n, p in model.named_parameters()} if a.ema > 0 else None
    step, micro, t0, run = 0, 0, time.time(), defaultdict(float)
    log = open(Path(a.out).with_suffix(".log.jsonl"), "a") if a.out else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(n_epochs):
        loader = DataLoader(ds, batch_sampler=length_batches(rows, a.max_tokens, a.max_items, a.seed + ep),
                            collate_fn=Collate(tok.pad_token_id), num_workers=a.workers,
                            persistent_workers=False, prefetch_factor=4 if a.workers else None)
        for batch in loader:
            if step >= total:
                break
            if batch is None:
                continue
            try:
                b = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "src"}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = model(b["input_ids"], b["attention_mask"], b["opt_positions"], b["opt_mask"], b["qtype"])
                loss, ce, br = loss_fn(logits, b, a.brier, a.kd_tau_multi, a.kd_tau_bin)
                (loss / a.grad_accum).backward()
            except torch.cuda.OutOfMemoryError:
                if not a.skip_oom:
                    raise
                b = logits = loss = ce = br = None
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                micro -= micro % a.grad_accum   # the partial accumulation was dropped with the gradients
                print(json.dumps({"step": step, "oom_skipped": batch["input_ids"].numel()}), flush=True)
                continue
            run["loss"] += loss.item(); run["n"] += 1; run["q"] += len(batch["src"])
            micro += 1
            if micro % a.grad_accum:
                continue
            frac = step / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * frac
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if ema is not None:
                with torch.no_grad():
                    for n_, p_ in model.named_parameters():
                        ema[n_].mul_(a.ema).add_(p_.detach().float(), alpha=1 - a.ema)
            if step % 100 == 0:
                el = time.time() - t0
                rec = {"step": step, "total": total, "loss": run["loss"] / run["n"], "gn": float(gn),
                       "q_per_s": run["q"] / el, "eta_min": (total - step) * el / 100 / 60,
                       "lr": opt.param_groups[0]["lr"]}
                print(json.dumps(rec), flush=True)
                if log: log.write(json.dumps(rec) + "\n"); log.flush()
                run = defaultdict(float); t0 = time.time()
            if dev_loader and step % a.eval_every == 0:
                res = evaluate(model, dev_loader, device, a.brier)
                rec = {"step": step, "dev": res}
                print(json.dumps({"step": step, "dev_ALL": res["ALL"]}), flush=True)
                if log: log.write(json.dumps(rec) + "\n"); log.flush()
            for fr in a.save_at:
                if step == max(1, int(total * fr)):
                    save_ckpt(model, tok, f"{a.out}-p{int(round(fr * 100))}", {"base": a.base, "step": step, "frac": fr})
                    print(json.dumps({"step": step, "saved": f"{a.out}-p{int(round(fr * 100))}"}), flush=True)
            if a.save_every and step % a.save_every == 0:
                save_ckpt(model, tok, f"{a.out}-step{step}", {"base": a.base, "step": step})
    save_ckpt(model, tok, a.out, {"base": a.base, "step": step, "args": vars(a)})
    if ema is not None:
        live = {n_: p_.detach().clone() for n_, p_ in model.named_parameters()}
        with torch.no_grad():
            for n_, p_ in model.named_parameters():
                p_.copy_(ema[n_].to(p_.dtype))
        save_ckpt(model, tok, f"{a.out}-ema", {"base": a.base, "step": step, "ema": a.ema, "args": vars(a)})
        with torch.no_grad():
            for n_, p_ in model.named_parameters():
                p_.copy_(live[n_])
    if dev_loader:
        res = evaluate(model, dev_loader, device, a.brier)
        (Path(a.out) / "dev.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({"final_dev_ALL": res["ALL"]}), flush=True)


if __name__ == "__main__":
    main()
