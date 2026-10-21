"""Fixed training subset for the A1/A2 grid (same rows for every arm and seed).

    python make_subset.py --train mix400_s1.train.jsonl.gz --teacher mix400_s1.t400.npz --n 150000 --seed 2026 \
        --out sub150k.jsonl.gz --out_teacher sub150k.t400.npz

Uniform sample without replacement of stage-1 rows of the v2 recipe (mix400_s1, the 32M v2 stage-1 data, firewalled
against the 0.3 suite), shuffled; teacher logits (Crivo-400M, the 32M v2 teacher) re-indexed to the subset.
"""
import argparse
import gzip
import json
import random

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--teacher", required=True)
    ap.add_argument("--n", type=int, default=150000); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True); ap.add_argument("--out_teacher", required=True)
    a = ap.parse_args()
    with gzip.open(a.train, "rt") as f:
        lines = [l for l in f if l.strip()]
    z = np.load(a.teacher); logits, offsets = z["logits"], z["offsets"]
    assert len(offsets) - 1 == len(lines), (len(offsets), len(lines))
    idx = random.Random(a.seed).sample(range(len(lines)), a.n)
    out_l, out_o = [], [0]
    with gzip.open(a.out, "wt") as f:
        for i in idx:
            f.write(lines[i] if lines[i].endswith("\n") else lines[i] + "\n")
            seg = logits[offsets[i]:offsets[i + 1]]
            out_l.append(seg); out_o.append(out_o[-1] + len(seg))
    np.savez(a.out_teacher, logits=np.concatenate(out_l) if out_l else logits[:0], offsets=np.array(out_o, dtype=offsets.dtype))
    srcs = {}
    for i in idx:
        s = json.loads(lines[i]).get("src", "?").split("/")[0]
        srcs[s] = srcs.get(s, 0) + 1
    print(json.dumps({"rows": a.n, "of": len(lines), "seed": a.seed, "src_prefix": srcs}))


if __name__ == "__main__":
    main()
