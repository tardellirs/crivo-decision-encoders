"""Task 1: stock / untuned models and well-known baselines already on the Decision Index 0.4 board.

    python stock_baselines.py <di-index-20261020.json> <di-methodology-20261020.json> <out.md> <out.json>

Group A: meta.kind == "inference technique" (the board's rule: no own weights; a stock vendor checkpoint served
through the entrant's own prompt/decoding/scoring code). These are the closest the board has to "untuned stock
models"; notes say which ones tuned prompts/temperatures on public-index rows.
Group B: well-known reference points that are trained but not by us (Jev hosted, Bekko v0, Laya), shown for context.
Every number is the board's ([board]): public index, private same-skills (ss), private new-domains (nd), Full 0.4 =
0.6*eq.S + 0.4*eq.O, rank group.
"""
import json
import sys

d04, meth, out_md, out_json = sys.argv[1:5]
a = json.load(open(d04))
REF = {"jev-1.13.0", "bekko-v0-17m", "bekko-v0-68m", "bekko-v0-400m", "laya-1e28ac20", "jeff-uncapped"}
rows = []
for m in a["models"]:
    meta, v = m["meta"], m["v03"]
    grp = "A" if meta.get("kind") == "inference technique" else ("B" if m["engine"] in REF else None)
    if not grp:
        continue
    notes = " ".join(n for n in (m.get("notes") or []) if "No training" in n or "chosen on" in n or "fitted" in n
                     or "Inference technique" in n or "Rule-based" in n or "prompt" in n.lower())
    rows.append({"group": grp, "engine": m["engine"], "name": m["name"], "base": meta.get("base_model"),
                 "params": meta.get("served_params"), "technique": meta.get("technique"), "pub": v.get("pub"),
                 "ss": v.get("same_skills"), "nd": v.get("new_domains"), "full": v.get("v"), "eqS": (v.get("eq") or {}).get("S"),
                 "eqO": (v.get("eq") or {}).get("O"), "rank": v.get("rank_tie"), "notes": notes[:220]})
rows.sort(key=lambda r: (r["group"], r["params"] or 0))
json.dump(rows, open(out_json, "w"), indent=1)
fmt = lambda p: "—" if not p else (f"{p/1e9:.2f}B" if p >= 1e9 else f"{p/1e6:.0f}M")
L = ["| Grp | Entry | Base checkpoint | Params | Public | Same-skills | New-domains | Full 0.4 | Rank grp | Board note (tuning) |",
     "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
for r in rows:
    L.append(f"| {r['group']} | {r['name']} | {r['base'] or '—'} | {fmt(r['params'])} | {r['pub']} | {r['ss']} | {r['nd']} | {r['full']} | {r['rank']} | {r['notes'] or '—'} |")
open(out_md, "w").write("\n".join(L) + "\n")
print("\n".join(L))
