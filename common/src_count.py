"""Count training rows per `src` in each mix file (to map external datasets to training sources).

    python src_count.py out.json file1.jsonl.gz file2 ...
"""
import gzip, json, sys, re
from collections import Counter
from multiprocessing import Pool
_src = re.compile(r'"src":\s*"([^"]*)"')

def count(f):
    c = Counter()
    with gzip.open(f, "rt") as fh:
        for line in fh:
            m = _src.search(line)
            c[m.group(1) if m else "?"] += 1
    return f, c

if __name__ == "__main__":
    out, files = sys.argv[1], sys.argv[2:]
    with Pool(len(files)) as p:
        res = dict((f.split("/")[-1], dict(c)) for f, c in p.map(count, files))
    json.dump(res, open(out, "w"), indent=0)
    for f, c in res.items():
        print(f, sum(c.values()), len(c))
