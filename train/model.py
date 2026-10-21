"""Typed-decision encoder: one sequence per question, one score per option read at its [OPT] marker.

    [CLS] instructions [SEP] state [SEP] [OPT] option_1 [OPT] option_2 ... [OPT] option_n

Softmax over the options of the question. The same `render`/`encode` code is used in training and in the
Decision Index engine, so the model sees exactly the text it was trained on.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
OPT_TOKEN = "[OPT]"
NEG = -1e4
NOUL_FALSE = "No."
NOUL_TRUE = "Yes."


def render(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def option_texts(qtype: str, criteria: Any, instructions: Any = None) -> tuple[list[str], list[str] | None]:
    """Return (option texts, keys). Keys are None for noul (false, true) and score (level index)."""
    if qtype == "choice":
        keys = list(criteria)
        texts = []
        for k in keys:
            d = render(criteria[k])
            texts.append(f"{k}: {d}" if d and d != k else str(k))
        return texts, keys
    if qtype == "noul":
        c = criteria if isinstance(criteria, dict) else {}
        f, t = render(c.get("false")), render(c.get("true"))
        return [f"false: {f}" if f else NOUL_FALSE, f"true: {t}" if t else NOUL_TRUE], None
    if qtype == "score":
        levels = list(criteria.values()) if isinstance(criteria, dict) else list(criteria)
        return [f"{i}: {render(x)}" for i, x in enumerate(levels)], None
    raise ValueError(f"unknown question type {qtype!r}")


class Encoder:
    """Tokenization shared by training and inference."""

    def __init__(self, tokenizer, max_length: int = 8192):
        self.t = tokenizer
        self.opt_id = tokenizer.convert_tokens_to_ids(OPT_TOKEN)
        self.max_length = max_length

    def ids(self, text: str) -> list[int]:
        return self.t(text, add_special_tokens=False)["input_ids"] if text else []

    def encode(self, instructions: Any, state: Any, options: list[str], state_budget: int | None = None):
        t = self.t
        state_ids = self.ids(render(state))
        if state_budget is not None and len(state_ids) > state_budget:   # training only
            state_ids = state_ids[:max(state_budget, 0)]
        ids = [t.cls_token_id, *self.ids(render(instructions)), t.sep_token_id, *state_ids, t.sep_token_id]
        pos = []
        for o in options:
            pos.append(len(ids))
            ids += [self.opt_id, *self.ids(o)]
        return ids, pos


def collate(items: list[tuple[list[int], list[int], int]], pad_id: int, pad_multiple: int = 16) -> dict[str, torch.Tensor]:
    b = len(items)
    length = max(len(i) for i, _, _ in items)
    length = -(-length // pad_multiple) * pad_multiple
    n = max(len(p) for _, p, _ in items)
    ids = torch.full((b, length), pad_id, dtype=torch.long)
    mask = torch.zeros((b, length), dtype=torch.long)
    pos = torch.zeros((b, n), dtype=torch.long)
    omask = torch.zeros((b, n), dtype=torch.bool)
    for j, (i, p, _) in enumerate(items):
        ids[j, :len(i)] = torch.tensor(i)
        mask[j, :len(i)] = 1
        pos[j, :len(p)] = torch.tensor(p)
        omask[j, :len(p)] = True
    qtype = torch.tensor([q for _, _, q in items], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": mask, "opt_positions": pos, "opt_mask": omask, "qtype": qtype}


class DecisionModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int):
        super().__init__()
        self.encoder = encoder
        self.type_embedding = nn.Embedding(len(QUESTION_TYPES), hidden)
        nn.init.zeros_(self.type_embedding.weight)
        self.scorer = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, input_ids, attention_mask, opt_positions, opt_mask, qtype, head_fp32: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = opt_positions.unsqueeze(-1).expand(-1, -1, h.size(-1))
        if head_fp32:   # gather first, then type embedding + scorer outside autocast (fewer ties in bf16)
            with torch.autocast(device_type=h.device.type, enabled=False):
                g = h.gather(1, idx).float() + self.type_embedding(qtype).float()[:, None, :]
                logits = self.scorer(g).squeeze(-1).float()
        else:
            h = h + self.type_embedding(qtype)[:, None, :]
            logits = self.scorer(h.gather(1, idx)).squeeze(-1).float()
        return torch.where(opt_mask, logits, torch.full_like(logits, NEG))


def build(base: str, attn: str = "sdpa", dtype=torch.float32):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if OPT_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [OPT_TOKEN]})
    enc = AutoModel.from_pretrained(base, attn_implementation=attn, torch_dtype=dtype)
    if hasattr(enc, "model") and not hasattr(enc, "embeddings"):   # *ForMaskedLM -> backbone
        enc = enc.model
    emb = enc.get_input_embeddings()
    if emb.num_embeddings < len(tok):
        old = emb.weight.data
        enc.resize_token_embeddings(len(tok), mean_resizing=False)
        new = enc.get_input_embeddings().weight.data
        new[old.size(0):] = old[tok.sep_token_id]          # [OPT] starts as a copy of [SEP]
    model = DecisionModel(enc, enc.config.hidden_size)
    return model, tok


def save(model: DecisionModel, tok, out: str | Path, extra: dict | None = None):
    from safetensors.torch import save_file
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(torch.float32).contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"))
    cfg = model.encoder.config.to_dict()
    cfg["decision"] = {"opt_token": OPT_TOKEN, "opt_token_id": tok.convert_tokens_to_ids(OPT_TOKEN),
                       "question_types": QUESTION_TYPES, **(extra or {})}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    tok.save_pretrained(str(out))


def load(path: str | Path, device: str = "cpu", attn: str = "sdpa"):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    extra = cfg.pop("decision")
    mc = ModernBertConfig(**{k: v for k, v in cfg.items() if k not in ("architectures",)})
    mc._attn_implementation = attn
    model = DecisionModel(ModernBertModel(mc), mc.hidden_size)
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    tok = AutoTokenizer.from_pretrained(str(path))
    return model.to(device).eval(), tok, extra
