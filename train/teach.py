"""Teacher logits for distillation: run a trained checkpoint over a training file with exactly the student's
rendering and truncation (original option order), and save them aligned with the file's rows.

    python teach.py --ckpt ../runs/soup23 --train ../data/mix_v1.train.jsonl.gz --max_len 2048 --out mix_v1.teacher.npz

Output .npz: logits (float16, concatenated per row in option_texts order), offsets (int64, n_rows+1; a row whose
question did not fit has an empty slice), n_opts (int32 per row).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import collate, load
from train import Rows, length_batches, read_rows
from model import Encoder


class Indexed(Dataset):
    def __init__(self, rows: Rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = self.rows[i]
        return None if x is None else (i, x[0], x[1], x[2])


class C:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, items):
        items = [x for x in items if x is not None]
        if not items:
            return None
        b = collate([(ids, pos, q) for _, ids, pos, q in items], self.pad_id)
        b["idx"] = torch.tensor([i for i, *_ in items])
        b["n"] = torch.tensor([len(pos) for _, _, pos, _ in items])
        return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--max_tokens", type=int, default=65536)
    ap.add_argument("--max_items", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load(a.ckpt, device=device)
    rows = read_rows(a.train)
    if a.limit:
        rows = rows[:a.limit]
    ds = Rows(rows, Encoder(tok), a.max_len, shuffle_options=False)
    loader = DataLoader(Indexed(ds), batch_sampler=length_batches(rows, a.max_tokens, a.max_items, 0),
                        collate_fn=C(tok.pad_token_id), num_workers=a.workers,
                        prefetch_factor=4 if a.workers else None)
    out = [None] * len(rows)
    t0, done = time.time(), 0
    with torch.no_grad():
        for b in loader:
            if b is None:
                continue
            x = {k: v.to(device) for k, v in b.items() if k not in ("idx", "n")}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                lg = model(x["input_ids"], x["attention_mask"], x["opt_positions"], x["opt_mask"], x["qtype"])
            lg = lg.float().cpu().numpy()
            for j, (i, n) in enumerate(zip(b["idx"].tolist(), b["n"].tolist())):
                out[i] = lg[j, :n].astype(np.float16)
            done += len(b["idx"])
            if done % 50000 < len(b["idx"]):
                print(f"{done:,}/{len(rows):,} rows, {done / (time.time() - t0):.0f} rows/s", flush=True)
    n_opts = np.array([0 if o is None else len(o) for o in out], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(n_opts)]).astype(np.int64)
    logits = np.concatenate([o for o in out if o is not None]) if any(o is not None for o in out) else np.zeros(0, np.float16)
    np.savez(a.out, logits=logits, offsets=offsets, n_opts=n_opts)
    print(f"saved {a.out}: {int((n_opts > 0).sum()):,}/{len(rows):,} rows with logits, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
