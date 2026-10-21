"""A1 ablation: the same Crivo architecture, but each option scored in its own pass.

Joint reader (released Crivo):  [CLS] instr [SEP] state [SEP] [OPT] o_1 [OPT] o_2 ... [OPT] o_n   -> n logits, 1 pass
Per-option reader (this file):  [CLS] instr [SEP] state [SEP] [OPT] o_i                           -> 1 logit, n passes
Both read the hidden state at [OPT] + question-type embedding -> the same MLP -> softmax over the question's
options. Same tokenizer, special tokens, rendering (model.option_texts / render), loss and data; only the
attention between options (and the cost: n sequences instead of 1) differs.

    PerOptionEngine(path=<ckpt>)   runs a checkpoint trained with train_ablate.py --reader peropt (or, for a check,
                                   a joint checkpoint read per option).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]   # repository root; train/ = vendored training code
sys.path.insert(0, str(ROOT / "train"))
from model import NEG, QUESTION_TYPES, collate, load, option_texts, render  # noqa: E402


def peropt_items(enc, instructions, state, texts, qtype: int, state_budget: int | None = None):
    """One (ids, [opt position], qtype) item per option."""
    t = enc.t
    s_ids = enc.ids(render(state))
    if state_budget is not None and len(s_ids) > state_budget:
        s_ids = s_ids[:max(state_budget, 0)]
    prefix = [t.cls_token_id, *enc.ids(render(instructions)), t.sep_token_id, *s_ids, t.sep_token_id]
    return [(prefix + [enc.opt_id, *enc.ids(o)], [len(prefix)], qtype) for o in texts]


def regroup(flat_logits: torch.Tensor, counts: list[int]) -> torch.Tensor:
    """[sum(counts)] option logits -> [B, max(counts)] padded with NEG."""
    b, n = len(counts), max(counts)
    out = torch.full((b, n), NEG, dtype=flat_logits.dtype, device=flat_logits.device)
    i = 0
    for j, c in enumerate(counts):
        out[j, :c] = flat_logits[i:i + c]
        i += c
    return out


def forward_peropt(model, flat_batch: dict, counts: list[int], head_fp32: bool = False) -> torch.Tensor:
    z = model(flat_batch["input_ids"], flat_batch["attention_mask"], flat_batch["opt_positions"], flat_batch["opt_mask"],
              flat_batch["qtype"], head_fp32=head_fp32)[:, 0]
    return regroup(z, counts)


class PerOptionEngine:
    def __init__(self, path: str, device: str | None = None, batch_size: int = 64, max_length: int = 32768,
                 token_budget: int = 65536, request_batched=False, **_):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
        from engines import Unsupported, answer_from_probs  # noqa: E402
        self.Unsupported, self.answer_from_probs = Unsupported, answer_from_probs
        from model import Encoder
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tok, _ = load(path, device=self.dev)
        self.enc = Encoder(self.tok, max_length=int(max_length))
        self.bs, self.budget = int(batch_size), int(token_budget)
        self.name = f"peropt:{path}"
        # request_batched=1 (added 2026-10-21, GPU-3 E6): a single request is answered through batch_call, i.e. all
        # options of all its questions in shared length-sorted batches (as the joint engine batches all questions of
        # a request). Default (0): one batch per question, the GPU-1 E6 / grid behaviour. Answers are the same.
        self.request_batched = str(request_batched).lower() in ("1", "true", "yes")
        if self.request_batched:
            self.name += " [request-batched]"

    @torch.no_grad()
    def __call__(self, state, questions):
        if self.request_batched:
            r = self.batch_call([(state, questions)])[0]
            if isinstance(r, Exception):
                raise r
            return r
        out = {}
        for k, q in questions.items():
            texts, labels = option_texts(q["type"], q.get("criteria"), q.get("instructions"))
            items = peropt_items(self.enc, q.get("instructions"), state, texts, QUESTION_TYPES[q["type"]])
            if max(len(i[0]) for i in items) > self.enc.max_length:
                raise self.Unsupported("question above max_length")
            z = []
            cur = []
            for it in items + [None]:
                if it is None or (cur and (len(cur) >= self.bs or len(it[0]) * (len(cur) + 1) > self.budget)):
                    if cur:
                        b = {kk: v.to(self.dev) for kk, v in collate(cur, self.tok.pad_token_id).items()}
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.startswith("cuda")):
                            z += self.model(b["input_ids"], b["attention_mask"], b["opt_positions"], b["opt_mask"],
                                            b["qtype"], head_fp32=True)[:, 0].float().tolist()
                    cur = []
                if it is not None:
                    cur.append(it)
            p = torch.tensor(z, dtype=torch.float64).softmax(-1).tolist()
            keys = labels if q["type"] == "choice" else (["false", "true"] if q["type"] == "noul" else [str(i) for i in range(len(p))])
            out[k] = self.answer_from_probs(q, keys, p)
        return out

    @torch.no_grad()
    def batch_call(self, reqs):
        """Many requests at once (run_requests.py --batch N): every option of every question is one row; rows are
        length-sorted and packed under the same batch_size / token_budget as __call__, then regrouped per question.
        Added 2026-10-21: __call__ alone ran at ~0.6 requests/s on pub15 (one small batch per question)."""
        rows, meta, bad = [], [], {}
        for ri, (state, qs) in enumerate(reqs):
            for k, q in qs.items():
                texts, labels = option_texts(q["type"], q.get("criteria"), q.get("instructions"))
                items = peropt_items(self.enc, q.get("instructions"), state, texts, QUESTION_TYPES[q["type"]])
                if max(len(i[0]) for i in items) > self.enc.max_length:
                    bad[ri] = self.Unsupported("question above max_length"); break
                meta.append((ri, k, q, labels, len(rows), len(items)))
                rows += items
        keep = {m[0] for m in meta} - set(bad)
        live = [i for m in meta if m[0] in keep for i in range(m[4], m[4] + m[5])]
        live.sort(key=lambda i: len(rows[i][0]))
        z = [0.0] * len(rows)
        cur = []
        for i in live + [None]:
            if i is None or (cur and (len(cur) >= self.bs or len(rows[i][0]) * (len(cur) + 1) > self.budget)):
                if cur:
                    b = {kk: v.to(self.dev) for kk, v in collate([rows[c] for c in cur], self.tok.pad_token_id).items()}
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.startswith("cuda")):
                        zz = self.model(b["input_ids"], b["attention_mask"], b["opt_positions"], b["opt_mask"],
                                        b["qtype"], head_fp32=True)[:, 0].float().tolist()
                    for c, v in zip(cur, zz):
                        z[c] = v
                cur = []
            if i is not None:
                cur.append(i)
        out = [dict() for _ in reqs]
        for ri, k, q, labels, s, n in meta:
            if ri in bad:
                continue
            p = torch.tensor(z[s:s + n], dtype=torch.float64).softmax(-1).tolist()
            keys = labels if q["type"] == "choice" else (["false", "true"] if q["type"] == "noul" else [str(i) for i in range(len(p))])
            out[ri][k] = self.answer_from_probs(q, keys, p)
        return [bad.get(ri, out[ri]) for ri in range(len(reqs))]

    def sync(self):
        if self.dev.startswith("cuda"):
            torch.cuda.synchronize()
