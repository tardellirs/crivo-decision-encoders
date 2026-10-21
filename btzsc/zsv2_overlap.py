import sys, os, re, json, hashlib, collections, unicodedata
import pyarrow.parquet as pq
train_pq, btz_dir, out = sys.argv[1:4]
def norm(s):
    s = unicodedata.normalize('NFKC', s or '').lower()
    return re.sub(r'\W+', ' ', s).strip()
def h(s): return hashlib.blake2b(s.encode(), digest_size=8).digest()
DS = sorted(f[:-8] for f in os.listdir(btz_dir) if f.endswith('.parquet'))
# train side
all_exact = {}   # hash -> task
task_texts = collections.defaultdict(set)
task_hyps = collections.defaultdict(set)
pf = pq.ParquetFile(train_pq)
for b in pf.iter_batches(columns=['task_name','text','hypothesis'], batch_size=50000):
    d = b.to_pydict()
    for t, x, y in zip(d['task_name'], d['text'], d['hypothesis']):
        n = norm(x); k = h(n)
        all_exact.setdefault(k, t)
        if t in DS:
            task_texts[t].add(n)
            task_hyps[t].add(y.strip())
def grams(n, k=8):
    w = n.split()
    return {h(' '.join(w[i:i+k])) for i in range(len(w)-k+1)}
res = {}
for ds in DS:
    tg = set()
    for n in task_texts.get(ds, ()): tg |= grams(n)
    t = pq.read_table(os.path.join(btz_dir, ds+'.parquet'), columns=['text','hypothesis']).to_pydict()
    texts = list(dict.fromkeys(t['text']))  # unique, order-preserving
    hyps = set(x.strip() for x in t['hypothesis'])
    ex = ex_same = near = ex5 = 0
    for x in texts:
        n = norm(x); k = h(n)
        if k in all_exact:
            if len(n.split()) >= 5: ex5 += 1
            ex += 1
            if all_exact[k] == ds: ex_same += 1
        g = grams(n)
        if g and len(g & tg) / len(g) >= 0.8: near += 1
    res[ds] = dict(test_unique_texts=len(texts), laurer_train_unique_texts=len(task_texts.get(ds, ())),
                   exact_any=ex, exact_any_ge5w=ex5, exact_same_task=ex_same, near8_same_task=near,
                   hyp_btzsc=len(hyps), hyp_in_laurer_train=len(hyps & task_hyps.get(ds, set())))
    print(ds, res[ds], flush=True)
json.dump(res, open(out, 'w'), indent=1)
