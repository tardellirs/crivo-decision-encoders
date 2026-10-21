"""Contamination audit: do external test items appear in our training rows?

    PYTHONHASHSEED=0 python contam.py --targets targets.jsonl --train a.jsonl.gz b.jsonl.gz ... --out audit.json [--workers 16]

targets.jsonl: one {"bench", "ds", "id", "text"} per external test item.
Training rows: our unified rows ({"state", "instructions", "criteria", "src", ...}); every string leaf of the row is
checked (state, instructions, criteria keys and descriptions).

Normalization: NFKC, lowercase, every run of non-word characters (Unicode-aware) -> one space.
Checks per target item:
  exact     the normalized target equals a normalized training leaf, or one line of it, or the leaf/line after a
            leading "<word(s)>:" prefix is stripped.
  contain   (targets with >= 8 words) share of the target's distinct word 8-grams that occur in ONE training leaf;
            reported as the max over leaves. >= 0.8 = near-duplicate, >= 0.5 = partial overlap.
Targets with < 5 words are checked by `exact` only and counted apart (`short_exact`): a match of a short generic text
("thank you", "delete alarm") is weak evidence. `any` = exact on a >= 5-word target, or contain >= 0.8.
Output: per dataset counts (exact, contain>=0.8, contain>=0.5), the training sources (src) and files of the hits,
and up to 5 examples per dataset.
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import re
import unicodedata
from collections import Counter, defaultdict

N = 8
_nonalnum = re.compile(r"[\W_]+")
_prefix = re.compile(r"^\w[\w ]{0,29}: ?")

T_TEXT, T_DS = [], []
GRAMS: dict[int, object] = {}
NGR: list[int] = []
EXACT: dict[str, list[int]] = {}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return _nonalnum.sub(" ", s).strip()


def grams(words: list[str]) -> set[int]:
    return {hash(" ".join(words[i:i + N])) for i in range(len(words) - N + 1)}


def leaves(o, out):
    if isinstance(o, str):
        out.append(o)
    elif isinstance(o, dict):
        for k, v in o.items():
            out.append(str(k)); leaves(v, out)
    elif isinstance(o, list):
        for v in o:
            leaves(v, out)


def build(targets):
    for i, t in enumerate(targets):
        n = norm(t["text"])
        T_TEXT.append(n); T_DS.append(t["ds"])
        EXACT.setdefault(n, []).append(i)
        w = n.split()
        g = grams(w) if len(w) >= N else set()
        NGR.append(len(g))
        for h in g:
            cur = GRAMS.get(h)
            if cur is None:
                GRAMS[h] = i
            elif isinstance(cur, int):
                if cur != i:
                    GRAMS[h] = [cur, i]
            elif cur[-1] != i:
                cur.append(i)


def scan(chunk):
    fname, lines = chunk
    hits = []   # (target, kind, value, src)
    for line in lines:
        try:
            r = json.loads(line)
        except Exception:
            continue
        src = str(r.get("src", "?"))
        ls = []
        leaves(r.get("state"), ls); leaves(r.get("instructions"), ls); leaves(r.get("criteria"), ls)
        seen_exact = set()
        for leaf in ls:
            if not leaf or len(leaf) < 8:
                continue
            n = norm(leaf)
            cands = {n}
            for ln in leaf.split("\n"):
                m = norm(ln)
                if m:
                    cands.add(m)
            for c in list(cands):
                s = _prefix.sub("", c)
                if s:
                    cands.add(s)
            for c in cands:
                for ti in EXACT.get(c, ()):
                    if ti not in seen_exact:
                        seen_exact.add(ti); hits.append((ti, "exact", 1.0, src, fname))
            w = n.split()
            if len(w) < N:
                continue
            cnt = Counter()
            for h in grams(w):
                cur = GRAMS.get(h)
                if cur is None:
                    continue
                if isinstance(cur, int):
                    cnt[cur] += 1
                else:
                    for ti in cur:
                        cnt[ti] += 1
            for ti, c in cnt.items():
                if NGR[ti] and c / NGR[ti] >= 0.5:
                    hits.append((ti, "contain", c / NGR[ti], src, fname))
    return hits


def read_chunks(files, size=2000):
    for f in files:
        op = gzip.open if f.endswith(".gz") else open
        buf = []
        with op(f, "rt") as fh:
            for line in fh:
                buf.append(line)
                if len(buf) >= size:
                    yield f, buf; buf = []
        if buf:
            yield f, buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--hits_out", help="jsonl of every overlapping target (ds, id, kind, src), for decontaminated scoring")
    a = ap.parse_args()
    targets = [json.loads(l) for l in open(a.targets)]
    build(targets)
    print(f"targets {len(targets):,}, grams {len(GRAMS):,}", flush=True)
    best = {}            # target -> (max contain, src, file)
    exact = {}           # target -> (src, file)
    rows = Counter()
    srcs_hit = defaultdict(Counter)
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, hits in enumerate(pool.imap_unordered(scan, read_chunks(a.train), chunksize=1)):
            for ti, kind, v, src, f in hits:
                if kind == "exact":
                    exact.setdefault(ti, (src, f))
                    srcs_hit[T_DS[ti]][f"exact:{src}"] += 1
                elif v > best.get(ti, (0,))[0]:
                    best[ti] = (v, src, f)
            if k % 200 == 0:
                print(f"chunks {k:,} exact {len(exact):,} contain {len(best):,}", flush=True)
    per = defaultdict(lambda: {"n": 0, "short": 0, "short_exact": 0, "exact": 0, "c80": 0, "c50": 0, "any": 0, "files": Counter(),
                               "srcs": Counter(), "examples": []})
    hit_rows = []
    for i, t in enumerate(targets):
        d = per[t["ds"]]
        d["n"] += 1
        short = len(T_TEXT[i].split()) < 5
        d["short"] += short
        e, b = exact.get(i), best.get(i)
        if e is not None and short:
            d["short_exact"] += 1
            e = None
        c = b[0] if b else 0.0
        d["exact"] += e is not None
        d["c80"] += c >= 0.8
        d["c50"] += c >= 0.5
        hit = e is not None or c >= 0.8
        d["any"] += hit
        if hit:
            hit_rows.append({"ds": t["ds"], "id": t["id"], "kind": "exact" if e else f"contain {c:.2f}",
                             "src": (e[0] if e else b[1])})
        if hit:
            s, f = (e if e else (b[1], b[2]))
            d["srcs"][s] += 1; d["files"][f.split("/")[-1]] += 1
            if len(d["examples"]) < 5:
                d["examples"].append({"id": t["id"], "text": t["text"][:200], "kind": "exact" if e else f"contain {c:.2f}",
                                      "src": s, "file": f.split("/")[-1]})
    out = {ds: {**{k: v for k, v in d.items() if k not in ("srcs", "files")}, "srcs": dict(d["srcs"].most_common(10)),
                "files": dict(d["files"])} for ds, d in per.items()}
    if a.hits_out:
        with open(a.hits_out, "w") as fh:
            for h in hit_rows:
                fh.write(json.dumps(h, ensure_ascii=False) + "\n")
    json.dump({"train_files": a.train, "n_targets": len(targets), "per_dataset": out}, open(a.out, "w"), indent=1)
    for ds, d in sorted(out.items()):
        print(f"{ds:28s} n={d['n']:6d} exact={d['exact']:5d} c80={d['c80']:5d} c50={d['c50']:5d} any={d['any']:5d} "
              f"{list(d['srcs'])[:3]}", flush=True)


if __name__ == "__main__":
    main()
