"""Effective errors of a run_requests output: request ids of the request file that have no line with status ok or
unsupported (run_requests.py's own notion of done: those ids are skipped on resume, error ids are retried). Prints
the count (0 = complete).    python eff_errors.py <results.jsonl> <requests.jsonl>"""
import json
import sys

done = set()
try:
    for l in open(sys.argv[1]):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("status") in ("ok", "unsupported"):
            done.add(r["id"])
except FileNotFoundError:
    pass
print(sum(1 for l in open(sys.argv[2]) if json.loads(l)["id"] not in done))
