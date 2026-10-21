"""Cross-check: run a Crivo checkpoint through the official BTZSC harness (BTZSCBenchmark.evaluate with a BaseModel
adapter that sends exactly our adapter's request: state = text, fixed INSTRUCTIONS, criteria {label_name: hypothesis})
and print the harness's per-dataset macro-F1, to compare with adapter.py score on the same items.

    python harness_check.py <crivo dir> <btzsc data dir> <max_samples> <out.json>
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from adapter import DATASETS, INSTRUCTIONS, load  # noqa: E402
from engines import CrivoEngine  # noqa: E402
from btzsc import BTZSCBenchmark  # noqa: E402
from btzsc.models.base import BaseModel  # noqa: E402

path, root, ms, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
NAMES = {}
for d in DATASETS:
    ds = load(d, root, 2)
    for h, n in zip(ds["labels"], ds["label_names"]):
        NAMES[(d, h)] = n


class CrivoBTZSC(BaseModel):
    model_type = "crivo"
    def __init__(self):
        self.e = CrivoEngine(path, batch_size=32)
    def predict_scores(self, texts, labels, batch_size=32):
        ds = next(d for d in DATASETS if (d, labels[0]) in NAMES and all((d, h) in NAMES for h in labels))
        names = [NAMES[(ds, h)] for h in labels]
        rows = []
        for t in texts:
            a = self.e(t, {"label": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(zip(names, labels))}})["label"]
            rows.append([a["probabilities"][n] for n in names])
        return np.array(rows)
    def predict(self, texts, labels, batch_size=32):
        return self.predict_scores(texts, labels).argmax(1)


res = BTZSCBenchmark().evaluate(model=CrivoBTZSC(), max_samples=ms)
json.dump(res.to_dict(), open(out, "w"), indent=1)
print(res.per_dataset()[["dataset", "macro_f1"]].to_string()); print("overall", res.task_summary["overall"]["macro_f1"])
