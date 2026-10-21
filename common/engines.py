"""Every system of the paper behind one interface: engine(state, questions) -> {question_key: answer}.

Requests are Decision Index wire format: `state` (str or JSON) and `questions` {key: {"type": "choice"|"noul"|"score",
"instructions", "criteria"}}. Answers: choice {"type", "choice", "probabilities"}, noul {"type", "noul": P(true)},
score {"type", "score": expected level, "probabilities": {"0": p, ...}}. One fixed rendering per system for every
benchmark; nothing is tuned per task. Engines raise `Unsupported` for inputs they cannot read (counted as wrong).

    crivo      released Crivo package (tardellirs/crivo-*): crivo.Crivo.answer, the model's own code
    prumo      released Prumo engine (rule-based, CPU)
    nli        zero-shot NLI cross-encoder (DeBERTa-v3 / ModernBERT zeroshot-v2.0, bart-large-mnli): one pass per option
    letter     causal LM, option-letter logit readout (Qwen3-0.6B, Gemma-3-1B-it), fixed prompt, optional rotations
    reranker   untrained-for-decisions Ettin reranker (cross-encoder/ettin-reranker-*): one pass per option
    majority   constant predictor from a label-prior file (built from gold labels: an oracle constant baseline)
    uniform    uniform distribution (chance), argmax = first option
"""
from __future__ import annotations

import json
import math
import string
import sys
from pathlib import Path


class Unsupported(ValueError):
    pass


def render(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def options_of(q: dict) -> tuple[list[str], list[str]]:
    """(option ids, option texts) for any question type; ids are what the answer is keyed by."""
    t = q["type"]
    if t == "choice":
        keys = list(q["criteria"])
        texts = []
        for k in keys:
            d = render(q["criteria"][k])
            texts.append(f"{k}: {d}" if d and d != k else str(k))
        return keys, texts
    if t == "noul":
        c = q.get("criteria") or {}
        f, tr = render(c.get("false")), render(c.get("true"))
        return ["false", "true"], [f"false: {f}" if f else "false", f"true: {tr}" if tr else "true"]
    if t == "score":
        levels = list(q["criteria"].values()) if isinstance(q["criteria"], dict) else list(q["criteria"])
        return [str(i) for i in range(len(levels))], [f"{i}: {render(x)}" for i, x in enumerate(levels)]
    raise Unsupported(f"question type {t}")


def answer_from_probs(q: dict, keys: list[str], p: list[float]) -> dict:
    s = sum(p)
    p = [x / s for x in p] if s > 0 else [1 / len(p)] * len(p)
    if q["type"] == "noul":
        return {"type": "noul", "noul": float(p[keys.index("true")])}
    if q["type"] == "score":
        return {"type": "score", "score": float(sum(i * x for i, x in enumerate(p))),
                "probabilities": {k: float(x) for k, x in zip(keys, p)}}
    best = max(range(len(p)), key=p.__getitem__)
    return {"type": "choice", "choice": keys[best], "probabilities": {k: float(x) for k, x in zip(keys, p)}}


def softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def _device(device):
    import torch
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


PATTERNS = ["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "tokenizer*"]   # what setup_gpu.sh prefetches


class CrivoEngine:
    """The released package: `crivo.Crivo` from the model folder (or HF repo), unchanged."""

    def __init__(self, path: str, device: str | None = None, revision: str | None = None, batch_size: int = 32,
                 code: str | None = None, **_):
        """code: a released Crivo folder whose crivo.py/model.py read `path` (a train_ablate.py checkpoint)."""
        p = Path(path)
        if not p.is_dir():
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(path, revision=revision, allow_patterns=PATTERNS))
        sys.path.insert(0, str(Path(code) if code else p))
        for m in ("crivo", "model"):
            sys.modules.pop(m, None)
        import crivo  # noqa: E402
        self.c = crivo
        self.m = crivo.Crivo(p, device=_device(device), batch_size=int(batch_size))
        self.name = f"crivo:{path}" + (f"@{revision}" if revision else "")
        # v1 revisions (e.g. Crivo-150M @7a6c745) predate the fp32 head: call their forward exactly as their own
        # engine does (no head_fp32 argument), so the batched path stays equivalent to the released answer()
        import inspect
        self._fwd_kw = {"head_fp32": True} if "head_fp32" in inspect.signature(self.m.model.forward).parameters else {}

    def __call__(self, state, questions):
        try:
            return self.m.answer(state, questions)
        except ValueError as e:
            raise Unsupported(str(e)) from e

    def batch_call(self, reqs):
        """Many requests in one call: the questions of all requests are length-sorted and batched exactly as
        Crivo.answer batches the questions of one request (same released functions, same fp32 head). Returns one
        answers dict or Unsupported per request. Equivalence with answer() is checked in check_batch.py."""
        import torch
        c, m = self.c, self.m
        items, meta, bad = [], [], {}
        for ri, (state, qs) in enumerate(reqs):
            for key, q in qs.items():
                if ri in bad:
                    break
                texts, labels = c.option_texts(q["type"], q.get("criteria"), q.get("instructions"))
                ids, pos = m.enc.encode(q.get("instructions"), state, texts)
                if len(ids) > m.enc.max_length:
                    bad[ri] = Unsupported(f"question {key!r} needs {len(ids)} tokens, above max_length={m.enc.max_length}")
                    break
                items.append((ids, pos, c.QUESTION_TYPES[q["type"]])); meta.append((ri, key, q["type"], labels))
        keep = [i for i in range(len(items)) if meta[i][0] not in bad]
        order = sorted(keep, key=lambda i: len(items[i][0]))
        batches, cur = [], []
        for i in order:
            if cur and (len(cur) >= m.batch_size or len(items[i][0]) * (len(cur) + 1) > m.token_budget):
                batches.append(cur); cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        out = [dict() for _ in reqs]
        with torch.no_grad():
            for batch in batches:
                b = {k: v.to(m.device) for k, v in c.collate([items[i] for i in batch], m.tok.pad_token_id).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=m.device.startswith("cuda")):
                    logits = m.model(b["input_ids"], b["attention_mask"], b["opt_positions"], b["opt_mask"], b["qtype"], **self._fwd_kw)
                for j, i in enumerate(batch):
                    ri, key, typ, labels = meta[i]
                    p = logits[j, :len(items[i][1])].double().softmax(-1).cpu().tolist()
                    if typ == "noul":
                        out[ri][key] = {"type": "noul", "noul": p[1]}
                    elif typ == "choice":
                        best = max(range(len(p)), key=p.__getitem__)
                        out[ri][key] = {"type": "choice", "choice": labels[best], "probabilities": dict(zip(labels, p))}
                    else:
                        out[ri][key] = {"type": "score", "score": sum(k * x for k, x in enumerate(p)),
                                        "probabilities": {str(k): x for k, x in enumerate(p)}}
        return [bad.get(ri, out[ri]) for ri in range(len(reqs))]

    def sync(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class PrumoEngine:
    """path: a real folder (not the HF cache: prumo_rules/params.py resolves its own path through the cache's
    symlinks); a repo id is downloaded to ~/models/<name> with local_dir."""

    def __init__(self, path: str, **opts):
        if not Path(path).is_dir():
            from huggingface_hub import snapshot_download
            path = snapshot_download(path, local_dir=str(Path.home() / "models" / path.split("/")[-1]))
        sys.path.insert(0, str(Path(path)))
        import importlib
        mod = importlib.import_module(opts.get("module", "decision_index_engine"))
        cls = getattr(mod, opts.get("cls", "DecisionEngine"))
        self.e = cls()
        self.name = "prumo"

    def __call__(self, state, questions):
        qs = {}
        for k, q in questions.items():
            if q["type"] == "score":   # Prumo answers choice/noul only: score levels as a choice over "0".."n-1"
                keys, _ = options_of(q)
                qs[k] = {"type": "choice", "instructions": q.get("instructions"),
                         "criteria": {kk: render(v) for kk, v in zip(keys, q["criteria"] if isinstance(q["criteria"], list) else q["criteria"].values())}}
            else:
                qs[k] = q
        resp, _ = self.e(state, qs)
        out = {}
        for k, q in questions.items():
            a = resp["answers"][k]
            if q["type"] == "score":
                keys, _ = options_of(q)
                out[k] = answer_from_probs(q, keys, [a["probabilities"][kk] for kk in keys])
            else:
                out[k] = a
        return out


class NLIEngine:
    """Zero-shot NLI: premise = rendered state; hypothesis = fixed template per option; score = entailment logit,
    softmax over the question's options (noul: options true/false, so P(true) = softmax of the two hypotheses).
    The premise is truncated to the model's context (only_first), as the BTZSC harness does."""

    TEMPLATE = 'Question: {ins} Answer: {opt}'

    def __init__(self, model: str, device: str | None = None, batch_size: int = 32, max_length: int = 512, fp16: str = "0", **_):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.dev = _device(device)
        self.tok = AutoTokenizer.from_pretrained(model)
        dt = torch.bfloat16 if (fp16 == "1" and self.dev == "cuda") else torch.float32   # bf16 (DeBERTa-v3 overflows in fp16)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, dtype=dt).to(self.dev).eval()
        l2i = {str(k).lower(): int(v) for k, v in (self.model.config.label2id or {}).items()}
        self.ent = next((l2i[k] for k in ("entailment", "label_2", "true", "yes") if k in l2i), max(l2i.values()) if l2i else 0)
        self.bs, self.max_length = int(batch_size), int(max_length)
        self.name = f"nli:{model}"

    def _hyp(self, q, text):
        return self.TEMPLATE.format(ins=render(q.get("instructions")), opt=text)

    def __call__(self, state, questions):
        torch = self.torch
        prem = render(state) or "(empty)"
        out = {}
        for k, q in questions.items():
            keys, texts = options_of(q)
            hyps = [self._hyp(q, t) for t in texts]
            scores = []
            for i in range(0, len(hyps), self.bs):
                try:
                    enc = self.tok([prem] * len(hyps[i:i + self.bs]), hyps[i:i + self.bs], truncation="only_first",
                                   max_length=self.max_length, padding=True, return_tensors="pt").to(self.dev)
                except Exception as e:   # an option alone exceeds max_length: unanswerable without cutting it
                    if "Truncation error" in str(e):
                        raise Unsupported(f"option of question {k!r} longer than max_length={self.max_length}") from e
                    raise
                with torch.no_grad():
                    lg = self.model(**enc).logits.float()
                scores += lg[:, self.ent].tolist()
            out[k] = answer_from_probs(q, keys, softmax(scores))
        return out

    def sync(self):
        if self.dev == "cuda":
            self.torch.cuda.synchronize()


class RerankerEngine:
    """Untrained-for-decisions Ettin reranker (sentence-transformers CrossEncoder, one relevance logit):
    query = instructions + blank line + rendered state; document = option text; softmax over the options."""

    def __init__(self, model: str = "cross-encoder/ettin-reranker-32m-v1", device: str | None = None, batch_size: int = 32,
                 max_length: int = 8192, **_):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.dev = _device(device)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.dev).eval()
        self.bs, self.max_length = int(batch_size), int(max_length)
        self.name = f"reranker:{model}"

    def __call__(self, state, questions):
        torch = self.torch
        out = {}
        for k, q in questions.items():
            keys, texts = options_of(q)
            query = (render(q.get("instructions")) + "\n\n" + render(state)).strip()
            scores = []
            for i in range(0, len(texts), self.bs):
                try:
                    enc = self.tok([query] * len(texts[i:i + self.bs]), texts[i:i + self.bs], truncation="only_first",
                                   max_length=self.max_length, padding=True, return_tensors="pt").to(self.dev)
                except Exception as e:   # an option alone exceeds max_length: unanswerable without cutting it
                    if "Truncation error" in str(e):
                        raise Unsupported(f"option of question {k!r} longer than max_length={self.max_length}") from e
                    raise
                with torch.no_grad():
                    lg = self.model(**enc).logits.float()
                scores += lg[:, 0].tolist()
            out[k] = answer_from_probs(q, keys, softmax(scores))
        return out

    def sync(self):
        if self.dev == "cuda":
            self.torch.cuda.synchronize()


SYMBOLS = (list(string.ascii_uppercase) + list(string.ascii_lowercase) + list("αβγδεζηθικλμνξοπρστυφχψω")
           + list("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"))


class LetterEngine:
    """Causal LM read at one position: the next-token probabilities of the option letters after a fixed prompt
    (the BTZSC LLM prompt, generalized with the question's instructions), renormalized over the options. No chat
    template (as BTZSC). `rotations=k` averages k cyclic rotations of the option order (AnyJev-style), mapped back.
    Questions with more options than single-token symbols are Unsupported. Context limit: inputs longer than
    `max_length` tokens are Unsupported (never truncated)."""

    PROMPT = ("You are a decision engine.\n"
              "You will be given an input, a question and several mutually exclusive options.\n"
              "Each option is prefixed by a single letter (e.g. A, b, γ, ...).\n"
              "Your task is to choose the single best option.\n\n"
              "IMPORTANT:\n- Answer with EXACTLY ONE LETTER used to prefix the options.\n"
              "- Do NOT output any words, punctuation, or explanation.\n\n"
              "INPUT:\n{state}\n\nQUESTION:\n{ins}\n\nOPTIONS:\n{options}\n\nAnswer: The correct option is letter ")

    def __init__(self, model: str, device: str | None = None, rotations: int = 1, max_length: int = 32768,
                 dtype: str | None = None, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = _device(device)
        dt = getattr(torch, dtype) if dtype else (torch.bfloat16 if self.dev == "cuda" else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dt).to(self.dev).eval()
        self.rot, self.max_length = int(rotations), int(max_length)
        self.ids = []
        for s in SYMBOLS:
            t = self.tok(s, add_special_tokens=False).input_ids
            if len(t) != 1:
                break
            self.ids.append(t[0])
        self.name = f"letter:{model}" + (f"@rot{self.rot}" if self.rot > 1 else "")

    def _probs(self, state, q, texts):
        torch = self.torch
        n = len(texts)
        opts = "\n".join(f"{SYMBOLS[i]}) {t}" for i, t in enumerate(texts))
        prompt = self.PROMPT.format(state=render(state) or "(empty)", ins=render(q.get("instructions")), options=opts)
        enc = self.tok(prompt, return_tensors="pt")
        if enc.input_ids.size(1) > self.max_length:
            raise Unsupported(f"prompt of {enc.input_ids.size(1)} tokens > {self.max_length}")
        with torch.no_grad():
            lg = self.model(**enc.to(self.dev), logits_to_keep=1).logits[0, -1].float()
        p = lg.softmax(-1)[self.ids[:n]].double()
        return (p / p.sum()).tolist()

    def _prompt(self, state, q, texts):
        opts = "\n".join(f"{SYMBOLS[i]}) {t}" for i, t in enumerate(texts))
        return self.PROMPT.format(state=render(state) or "(empty)", ins=render(q.get("instructions")), options=opts)

    def batch_call(self, reqs, token_budget: int = 32768):
        """Many requests at once: every (question, rotation) prompt is one row; rows are length-sorted, left-padded,
        with explicit position ids (so a padded row computes what it computes alone), and read at the last token."""
        torch = self.torch
        jobs, out, bad = [], [dict() for _ in reqs], {}
        for ri, (state, qs) in enumerate(reqs):
            for k, q in qs.items():
                keys, texts = options_of(q)
                if q["type"] == "noul":
                    keys, texts = ["true", "false"], ["true", "false"]
                n = len(texts)
                if n > len(self.ids):
                    bad[ri] = Unsupported(f"{n} options > {len(self.ids)} single-token letters"); break
                for s in range(min(self.rot, n)):
                    order = [(i + s) % n for i in range(n)]
                    ids = self.tok(self._prompt(state, q, [texts[o] for o in order])).input_ids
                    if len(ids) > self.max_length:
                        bad[ri] = Unsupported(f"prompt of {len(ids)} tokens > {self.max_length}"); break
                    jobs.append((ri, k, q, keys, order, ids))
        jobs = [j for j in jobs if j[0] not in bad]
        acc = {}
        jobs.sort(key=lambda j: len(j[5]))
        i = 0
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.tok.eos_token_id or 0)
        while i < len(jobs):
            L = len(jobs[i][5]); j = i
            while j < len(jobs) and len(jobs[j][5]) * (j - i + 1) <= token_budget and j - i < 64:
                L = len(jobs[j][5]); j += 1
            j = max(j, i + 1); chunk = jobs[i:j]; L = max(len(c[5]) for c in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long); mask = torch.zeros((len(chunk), L), dtype=torch.long)
            for r, c in enumerate(chunk):
                ids[r, L - len(c[5]):] = torch.tensor(c[5]); mask[r, L - len(c[5]):] = 1
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                lg = self.model(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev), position_ids=pos.to(self.dev),
                                logits_to_keep=1).logits[:, -1].float()
            for r, c in enumerate(chunk):
                ri, k, q, keys, order, _ = c
                n = len(order)
                p = lg[r].softmax(-1)[self.ids[:n]].double(); p = (p / p.sum()).tolist()
                a = acc.setdefault((ri, k), [0.0] * n)
                r_ = min(self.rot, n)
                for slot, o in enumerate(order):
                    a[o] += p[slot] / r_
            i = j
        for ri, (state, qs) in enumerate(reqs):
            if ri in bad:
                continue
            for k, q in qs.items():
                keys, _ = options_of(q)
                a = acc[(ri, k)]
                out[ri][k] = {"type": "noul", "noul": float(a[0] / sum(a))} if q["type"] == "noul" else answer_from_probs(q, keys, a)
        return [bad.get(ri, out[ri]) for ri in range(len(reqs))]

    def __call__(self, state, questions):
        out = {}
        for k, q in questions.items():
            keys, texts = options_of(q)
            if q["type"] == "noul":
                keys, texts = ["true", "false"], ["true", "false"]
            n = len(texts)
            if n > len(self.ids):
                raise Unsupported(f"{n} options > {len(self.ids)} single-token letters")
            acc = [0.0] * n
            r = min(self.rot, n)
            for s in range(r):
                order = [(i + s) % n for i in range(n)]
                p = self._probs(state, q, [texts[o] for o in order])
                for slot, o in enumerate(order):
                    acc[o] += p[slot] / r
            if q["type"] == "noul":
                out[k] = {"type": "noul", "noul": float(acc[0] / sum(acc))}
            else:
                out[k] = answer_from_probs(q, keys, acc)
        return out

    def sync(self):
        if self.dev == "cuda":
            self.torch.cuda.synchronize()


class MajorityEngine:
    """Constant predictor: for each question, the label that is most frequent in `prior` (JSON {family: {label: n}}
    keyed by the request `family` and question key), else the first option. The prior is built from gold labels of
    the evaluated split, so this is an oracle constant baseline (an upper bound for any constant answer)."""

    def __init__(self, prior: str, **_):
        self.prior = json.load(open(prior))
        self.family = None
        self.name = "majority"

    def __call__(self, state, questions):
        out = {}
        for k, q in questions.items():
            keys, _ = options_of(q)
            pr = self.prior.get(f"{self.family}|{k}", {})
            best = max(keys, key=lambda x: pr.get(x, 0)) if pr else keys[0]
            p = [1.0 if x == best else 0.0 for x in keys]
            out[k] = answer_from_probs(q, keys, p)
        return out


class UniformEngine:
    name = "uniform"

    def __init__(self, **_):
        pass

    def __call__(self, state, questions):
        out = {}
        for k, q in questions.items():
            keys, _ = options_of(q)
            out[k] = answer_from_probs(q, keys, [1.0] * len(keys))
        return out


ENGINES = {"crivo": CrivoEngine, "prumo": PrumoEngine, "nli": NLIEngine, "reranker": RerankerEngine,
           "letter": LetterEngine, "majority": MajorityEngine, "uniform": UniformEngine}


def make(spec: str, opts: dict):
    """spec = registry name (above) or module:Class (e.g. peropt:PerOptionEngine)."""
    if spec in ENGINES:
        return ENGINES[spec](**opts)
    mod, cls = spec.split(":")
    import importlib
    return getattr(importlib.import_module(mod), cls)(**opts)
