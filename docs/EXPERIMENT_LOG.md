# Paper experiments, days 1–3 (IEEE Access plan): full hand-off

> **Note (code release).** This is the experiment hand-off log, kept during the work on 2026-10-20 and 2026-10-21,
> with its original section numbering. It includes the decision rules written before the GPU runs (§17 for the A1/A2
> grid, §18.1 for GPU-3, §18.y.1 for GPU-4). For the release, host names, IP addresses, user names, instance ids and
> local absolute paths were replaced by placeholders such as `<gpu1-host>` or `<cpu-vm1>`, and one excluded
> third-party evaluation panel is not named. Nothing else was edited. Paths are relative to the original working
> directory `pareto/paper/exp/`, which corresponds to the root of this repository. Files the log mentions that are not
> in this repository (training data, teacher logits, checkpoints, per-item raw outputs, request files, benchmark data)
> were not released; see the README.

## Status

> **Status — 2026-10-20 18:40 UTC (15:40 BRT).**
>
> **Done:**
> - Day 1 is complete: stock baselines, BTZSC audit, typed-suite choice with its audit, and the board statistics.
> - Days 2–3 preparation is complete: adapters, baseline wrappers, the per-option engine, the A1/A2 grid, and GPU
>   scripts.
> - All of it was verified on CPU.
>
> **Running at hand-off:**
> - **GPU-1 (inference: E1, E2, E4, E6)** was launched by the main session on Lambda A100 `<gpu1-host>`
>   (gpu_1x_a100_sxm4) with `gpu/launch.sh … infer`. Results go to `~/results/` on that box.
> - **GPU-2 (A1/A2 grid) is ready and waits for the GPU-1 results.**
> - CPU, on the VMs: **Prumo** on BTZSC, S1-Arena and E4 (cpu-vm2) has **finished** (§10.1). **E6 CPU latency**
>   (cpu-vm1, tmux `e6cpu`) is still running (§10.2).
>
> **Added 2026-10-20 ~19:15 UTC — E5 calibration and public-suite error analysis are done** (CPU, cpu-vm2;
> §13–§15). Headline: every Crivo is over-confident in every confidence bin (benchmark-weighted ECE-15 0.076 for
> 17M up to 0.158 for 1B; Prumo 0.054), accuracy, Brier and AURC improve with size while ECE does not; our public-row
> numbers track the board's calibration fields (Spearman 0.98 over the nine systems). cpu-vm2 has no running
> jobs and everything it produced is on the Mac (`e5/`, `errors/`), so it can be deleted.
>
> **Stop-point verdict:**
> - BTZSC has **4 of 22 unseen datasets** for the v2 family. That is below the threshold of about 6, so BTZSC becomes a
>   secondary validation.
> - The typed suite **System1-Arena is usable**: 145 of 209 non-agentic tasks are unseen.
> - The two do not both fail, so we continue.
>
> **Rules kept throughout:**
> - No training or tuning on any test item.
> - One fixed rendering per system for every dataset.
> - Nothing published or uploaded, no HF repo touched, no rented GPU (the main session launched GPU-1).

Labels used for every number:
- **[board]**: maintainer-run Decision Index numbers (`release/di-index-20261020.json`, 0.4; and
  `release/di-index-v0.3.2-20261020.json`).
- **[published]**: from a benchmark's own results files.
- **[derived]**: computed by our scripts from those files.
- **[ours-cpu]**: CPU smoke tests on tiny samples. These are pipeline checks only and **never paper numbers**.

Paths below are relative to `pareto/paper/exp/` unless they start with `~` (a VM or GPU home) or `pareto/`.

---

## 0. Machines and where everything lives

| Where | What |
|---|---|
| Mac `pareto/paper/exp/` | All code and docs: `README.md`, `common/`, `btzsc/`, `s1arena/`, `e4/`, `e6/`, `a1a2/`, `gpu/`, `stats/`, `scripts/`. All small results: the tables and JSON named below. `audit_raw/` holds the raw audits, source counts and per-item hit lists. |
| VPS `<vps-host>:<workdir>/pareto/` | The canonical copy of `data/` (training mixes) and `runs/teacher_logits/mix400_s1.t400.npz` plus `runs/mix400_s1.t400.gold_only`, which are the grid inputs. `launch.sh` reads them from `$PARETO` (default this path; on the Mac `data/` and `runs/` also exist). The VPS key was added to both VMs' `authorized_keys`, and the stale known_hosts entry for <cpu-vm2> on the VPS was replaced; the fingerprint was checked. |
| VM1 `<cpu-vm1>` (cpu-vm1) | `~/pareto/data/` (all mixes, synced from the VPS). `~/btzsc/{data,results,harness}`. `~/s1arena/{data,code}`. `~/exp/audit/` (audits, hit lists, `src_counts.json`). `~/exp/a1a2/{sub150k.jsonl.gz, sub150k.t400.npz, gold_only, smoke/}`. `~/evalsets/{pub15,btzsc500,s1_60}.jsonl`. `~/e4/req/`. `~/exp/harness_check_17m_s50.json`. `/root/dry/` (grid dry run). **`~/exp/e6cpu/` (E6 CPU latency, tmux `e6cpu`).** |
| VM2 `<cpu-vm2>` (cpu-vm2) | `~/btzsc/data`, `~/s1arena/{data,code,req_all.jsonl,req_s5.jsonl,tasks.json}`, `~/req/{btzsc_full,s1_all}.jsonl`, `~/e4/req/` (md5 of `orig.jsonl` = 4643cfec… on both VMs). Smoke outputs `~/exp/smoke*`, `~/exp/smallsmoke_*`. `/root/dry2/` (inference-lanes dry run). **`~/exp/prumo/` (Prumo runs, tmux `prumo`).** HF cache with every model, including gated Gemma. `~/e5/` (E5 and error analysis, §13–§15: runs, per-question tables, outputs; all copied to the Mac). |
| GPU-1 `<gpu1-host>` | `~/results/` (STATUS, logs, `btzsc/*.jsonl`, `s1/*.jsonl`, `e4/<system>/<variant>.jsonl`, `e6/*.json`, `*.score.json`). **Bring back with** `rsync -a <gpu1-host>:results/ pareto/paper/exp/results_gpu/` **before terminating.** |

VM software (`scripts/setup_vm.sh`):
- Python 3.12 venv;
- torch 2.14.1+cpu;
- transformers 5.19.0, the version the Crivo engine uses;
- datasets, pyarrow, sklearn, scipy;
- python-chess 1.11.2;
- the kit installed with `pip -e --no-deps`.

The BTZSC harness is installed on VM1 with `pip -e --no-deps`.

---

## 1. Stock baselines already on the board (task 1)

Script: `stats/stock_baselines.py`. Outputs: `stats/stock_baselines.{md,json}`.

**Finding.** The board has **no untuned standard classifier**. Absent:
- NLI zero-shot DeBERTa, ModernBERT or BART;
- plain GLiClass or GLiNER;
- a small Qwen or Gemma through the kit's stock engine;
- raw Ettin or ModernBERT.

The "untuned stock models" that the 0.4 mapping uses are not listed in `models[]`, and the methodology file does not
name them (`release/di-methodology-20261020.json`, `changelog[0]`).

The nearest entries have `meta.kind == "inference technique"`. The board's rule for that kind: no own weights; a stock
vendor checkpoint served through the entrant's own code. Their official private numbers [board]:

| Entry | Base | Params | Public | Same-skills | New-domains | Full 0.4 | Board note |
|---|---|---:|---:|---:|---:|---:|---|
| jeff | knowledgator/gliformer-large-v1 | 576M | 8.27 | 6.86 | 1.53 | 4.81 | stock GLiFormer + own code |
| Qwen-2.5-1B-RLCD | Qwen2.5-1.5B(-Instruct) | 1.54B | 4.02 | 6.21 | 5.78 | 6.48 | stock checkpoint, RLCD decoding harness |
| mini-jev / openvons | Qwen3-4B-Instruct-2507 | 4.02B | 21.25 / 28.66 | 23.36 / 32.13 | 20.83 / 22.48 | 24.22 / 30.34 | prompting |
| Ryotide / Jobe / SemIf | Qwen3.5-4B(-Base) | 4.0–4.7B | 37.28 / 32.93 / 26.5 | 36.67 / 34.34 / 28.04 | 21.17 / 22.92 / 23.43 | 32.46 / 31.9 / 28.33 | — |
| decisio 0.4.0 | gemma-4-12B-it | 12.0B | 49.72 | 48.44 | 38.26 | 47.93 | prompt partly chosen on index rows |
| AnyJev L0 | gemma-4-31B-it | 31.3B | 59.79 | 59.13 | 58.44 | 64.24 | label readout + rotation + running prior |
| reflex | Qwen3.8-27B-FP8 | 27.8B | 57.34 | 57.39 | 49.71 | 58.94 | label-logprob readout |
| Prumo / Zev | rule-based | — | 20.14 / 1.55 | 15.78 / 1.79 | 5.60 / 1.89 | 12.21 / 1.91 | — |

All 25 entries are in `stats/stock_baselines.md`.

**Trained references** [board]:

| Entry | Full | Same-skills | New-domains | Public |
|---|---:|---:|---:|---:|
| Bekko v0 17M | 5.72 | 5.89 | 4.61 | 7.44 |
| Bekko v0 68M | 9.99 | 12.62 | 4.98 | 13.01 |
| Bekko v0 400M | 16.00 | 21.55 | 6.20 | 22.45 |
| Laya 421M | 7.21 | 9.08 | 3.69 | 6.19 |
| Jev 1.13 | 61.26 | 58.04 | 53.67 | 57.96 |

**Consequences:**
- The A3 baselines must be run by us. They will have external and public numbers, never private ones.
- The only below-2B stock entries with private numbers, jeff (Full 4.81) and Qwen-2.5-1.5B (6.48), are far below
  Crivo-17M (11.11 [board]).

---

## 2. BTZSC audit (task 2)

### 2.1 Benchmark and loader

**Sources** (all read-only):
- data: HF `btzsc/btzsc`, test split only, in NLI-pair format (`text`, `hypothesis`, `labels` ∈ {0,1}, `label_text`);
- results: HF `btzsc/btzsc-results` @70bcc8b (MIT), one JSON per model;
- harness: GitHub `IliasAarab/btzsc` @4eac6a9;
- paper: arXiv 2603.11991 / OpenReview IxMryAz2p3.

**Metric:** macro-F1 per dataset (sklearn `f1_score(average="macro", zero_division=0)`). "Overall" is the plain mean
over the 22 datasets. The harness also reports means by task family (sentiment, emotion, intent, topic).

**Loader** (`btzsc/adapter.py::load`) replicates `btzsc.data.load_btzsc_dataset` exactly:
- the class count is the period of the binary label column;
- yahootopics: the first 20 rows plus all rows whose `label_text != "Business & Finance"`;
- the gold index is the first entailment row, or **0 when a group has none**. That affects 200 banking77 and 169 massive
  items, whose gold class has no hypothesis: banking77 has 72 hypotheses for 77 intents and massive has 59. We keep
  this harness quirk so that numbers compare with the published ones;
- `max_samples` sampling uses seed 0 and keeps the first two groups.

**Verified on VM1:** identical texts, label lists and gold indices to the official loader for all 22 datasets, both in
full and at `max_samples=50`: 0 mismatches (`btzsc/loader_check.log`).

**Label names:** each hypothesis maps to the `label_text` of its entailment rows, for example "This example news text
is about sports" → `Sports`.

### 2.2 Datasets and contamination classes

#### Method

`common/contam.py`, run on VM1 with 15 processes over 3.7M training rows, about 4 minutes per pass.
- **Training side:** every string leaf of every row (state, instructions, criteria keys and descriptions).
  Normalization: NFKC, lowercase, every run of non-word characters becomes one space (Unicode-aware, so non-Latin
  texts are not emptied; an early ASCII-only version did empty them and was fixed).
- **Exact match:** a normalized test text equals a training leaf, a line of a leaf, or a leaf or line with a leading
  "word(s):" prefix removed. It counts only for test texts of **at least 5 words**. Shorter texts are counted apart
  (`short_exact`), because "thank you" or "delete alarm" matches are weak evidence.
- **Near-duplicate:** at least 80% of the test text's distinct word 8-grams occur in a single training leaf. The
  50–80% band is reported as `c50`.

#### Families

Separate passes, one per family:

| Family | Models | Training files |
|---|---|---|
| v2 | 17M, 32M v2, 68M, 150M v2, 400M, 1B | `mix400_s{1,2,3}.{train,dev}` |
| 32M v1 | 32M v1 | `mix32_s1.{train,dev}`, `mix32_s2.train`, `long_v3` |
| 150M v1 | 150M v1 | `mix_v1`, `mix_v2c`, `mix_v3` (train and dev) |

The row counts match the cards: mix_v1 1,011,412 / v2c 311,287 / v3 56,690 for 150M v1, and mix32_s1 1,143,435 for
32M v1.

#### Source-level exposure

- `common/src_count.py` writes `btzsc/src_counts_all_mixes.json` (rows per `src` per mix file).
- A dataset is "in training" if a `src` names the dataset itself or its source corpus. The mapping is in
  `btzsc/classify.py::SRC`. Two examples:
  - `wikitoxic_*` maps to `t3/jigsaw_toxicity`: Jigsaw toxic comments are the same Wikipedia talk comments;
  - `yelpreviews` maps to `yelp_review_full`, the same Yelp corpus.
- Related but different corpora are not counted, only listed: `civil_comments`, `go_emotions`, bitext banking,
  `Pol_NLI`-like.
- Note on the 150M v1 mix: it has **no tasksource (`t3/*`) block**, which explains most of its "unseen" column.

#### Classes

| Class | Rule |
|---|---|
| **seen-test-overlap** | overlap on at least 0.5% of the test items |
| **seen-train-split** | the source is in training, with overlap below 0.5% |
| **unseen** | neither |

#### Files

- `btzsc/classify.py` writes `btzsc/contamination.{md,json}`.
- Raw audits: `audit_raw/btzsc_audit_{v2family,32mv1,150mv1,allmixes}.json`.
- Per-item hit list, for a decontaminated re-score: `audit_raw/btzsc_hits_allmixes.jsonl`. It has 428 items; fields
  `ds`, `id` (the item's index in the BTZSC group order; the request id is `btzsc/<ds>/<id>`), `kind`, `src`.

#### Table

| Dataset | Task | Test items | Classes | v2 family | Same-source train rows (v2) | Overlapping items (v2) | Overlap sources | 32M v1 | 150M v1 |
|---|---|---:|---:|---|---:|---:|---|---|---|
| amazonpolarity | sentiment | 10,000 | 2 | seen-train-split | 297 (t3/amazon_polarity) | 0 | — | seen-train-split | unseen |
| imdb | sentiment | 10,000 | 2 | seen-train-split | 541 (t3/imdb, CAD-imdb) | 1 | t3/imdb | seen-train-split | unseen |
| appreviews | sentiment | 4,000 | 2 | **unseen** | 0 | 0 | — | seen-train-split | unseen |
| yelpreviews | sentiment | 10,000 | 2 | seen-train-split | 297 (yelp_review_full) | 3 | t3/yelp_review_full | seen-train-split | unseen |
| rottentomatoes | sentiment | 1,066 | 2 | seen-train-split | 525 (rotten_tomatoes, sst2) | 3 (0.3%) | t3/glue/sst2 | seen-train-split | unseen |
| financialphrasebank | sentiment | 690 | 3 | seen-test-overlap | 514 | **77 (11.2%)** | t3/financial_phrasebank, t3/auditor_review | seen-test-overlap | unseen |
| emotiondair | emotion | 2,000 | 6 | seen-train-split | 297 (t3/emotion) | 0 | — | seen-train-split | unseen |
| empathetic | emotion | 2,542 | 32 | **unseen** | 0 | 0 | — | unseen | unseen |
| banking77 | intent | 3,080 | 72 | seen-test-overlap | 18,744 | 33 (1.1%) | bekko/banking77, src400/exact | seen-test-overlap | seen-test-overlap |
| biasframes_intent | intent | 3,648 | 2 | seen-test-overlap | 294 (SBIC) | 43 (1.2%) | t3/SBIC_Disagreement | seen-test-overlap | unseen* |
| massive | intent | 2,974 | 59 | seen-test-overlap | 2,892 | 21 (0.7%) | bekko/massive | seen-test-overlap | seen-test-overlap |
| agnews | topic | 7,600 | 4 | seen-train-split | 5,824 | 22 (0.3%) | bekko/agnews, laya__ag_news | seen-test-overlap | seen-test-overlap |
| yahootopics | topic | 45,002 | 10 | seen-train-split | 293 | 0 | — | seen-train-split | unseen |
| trueteacher | topic | 8,955 | 2 | **unseen** | 0 | 0 | — | unseen* | unseen |
| manifesto | topic | 17,018 | 56 | seen-train-split | 282 (t3/Pol_NLI) | 2 | t3/Pol_NLI | seen-train-split | unseen* |
| capsotu | topic | 3,355 | 21 | **unseen** | 0 | 0 | — | unseen | unseen |
| biasframes_offensive | topic | 3,838 | 2 | seen-test-overlap | 294 | 45 (1.2%) | t3/SBIC_Disagreement | seen-test-overlap | unseen* |
| biasframes_sex | topic | 4,404 | 2 | seen-test-overlap | 294 | 51 (1.2%) | t3/SBIC_Disagreement | seen-test-overlap | unseen* |
| wikitoxic_toxicaggregated | topic | 10,000 | 2 | seen-train-split | 297 (jigsaw) | 14 (0.1%) | t3/jigsaw_toxicity | seen-train-split | unseen* |
| wikitoxic_obscene | topic | 8,691 | 2 | seen-train-split | 297 | 14 (0.2%) | t3/jigsaw_toxicity | seen-train-split | unseen* |
| wikitoxic_threat | topic | 5,211 | 2 | seen-train-split | 297 | 7 (0.1%) | t3/jigsaw_toxicity | seen-train-split | unseen* |
| wikitoxic_insult | topic | 8,427 | 2 | seen-train-split | 297 | 14 (0.2%) | t3/jigsaw_toxicity | seen-train-split | unseen* |

\* 1–5 stray overlaps (generic comments) below 0.5%.

**Counts:**

| Family | Unseen | Seen-train-split | Seen-test-overlap |
|---|---:|---:|---:|
| v2 | **4** | 12 | 6 |
| 32M v1 | 3 | 12 | 7 |
| 150M v1 | **19** | 0 | 3 |

The 4 v2-unseen datasets have **18,852 items**.

**Decision.** Below the threshold of about 6, so per plan_review §3 E1 **BTZSC is secondary validation**. Report:
1. the 4 v2-unseen datasets;
2. all 22 with the class column;
3. the 6 seen-test-overlap datasets also re-scored without their hit items.

Crivo-150M v1 is the "clean" BTZSC point: 19 of 22 unseen.

### 2.3 Published BTZSC baselines

Script: `btzsc/published.py`. Outputs: `btzsc/published.json` (every model × dataset) and `btzsc/published_table.md`.
The metric is macro-F1 [published]; "Unseen-4" = mean over appreviews, empathetic, trueteacher and capsotu.

| Model | Type | Params | Overall (22) | Unseen-4 |
|---|---|---:|---:|---:|
| Qwen3-Reranker-8B | reranker | 8B | 0.722 | 0.608 |
| Mistral-Nemo-Instruct-2407 | LLM | 12.2B | 0.670 | — (in json) |
| Qwen3-4B | LLM | 4B | 0.649 | 0.531 |
| gte-large-en-v1.5 | embedding | 434M | 0.617 | 0.551 |
| Qwen3-Reranker-0.6B | reranker | 600M | 0.606 | 0.540 |
| e5-base-v2 | embedding | 110M | 0.597 | 0.566 |
| deberta-v3-large-nli-triplet | NLI | 434M | 0.596 | 0.501 |
| deberta-v3-large-nli | NLI | 434M | 0.591 | 0.481 |
| gte-modernbert-base | embedding | 149M | 0.586 | 0.568 |
| gte-reranker-modernbert-base | reranker | 149M | 0.578 | 0.577 |
| modernbert-large-nli | NLI | 395M | 0.552 | 0.408 |
| deberta-v3-base-nli | NLI | 184M | 0.550 | 0.424 |
| modernbert-base-nli | NLI | 149M | 0.534 | 0.435 |
| bge-reranker-large | reranker | 560M | 0.535 | 0.540 |
| bart-large-mnli | NLI | 407M | 0.508 | 0.539 |
| bge-reranker-base | reranker | 278M | 0.471 | 0.488 |
| ms-marco-MiniLM-L6-v2 | reranker | 22M | 0.422 | 0.395 |
| all-MiniLM-L6-v2 | embedding | 22M | 0.366 | 0.360 |
| gemma-3-1b-it | LLM | 1B | 0.359 | 0.347 |
| gemma-3-270m-it | LLM | 270M | 0.279 | 0.243 |
| raw bert-large / ModernBERT-large / DeBERTa-v3-large | base encoder | 335–434M | 0.299 / 0.273 / 0.270 | 0.180 / 0.205 / 0.179 |

**Not published,** so GPU-1 runs these itself:
- No Ettin model: GPU-1 runs `cross-encoder/ettin-reranker-32m-v1` as-is.
- No MoritzLaurer `*-zeroshot-v2.0`, since BTZSC's NLI models are the authors' own NLI fine-tunes: GPU-1 runs both.
- No LLM below 4B except Gemma: GPU-1 runs Qwen3-0.6B, and Gemma-3-1B-it again as a check against 0.359.

---

## 3. Typed-decision suite for E2 (task 3)

A research subagent read the four papers and their data and code links (2026-10-20). It downloaded nothing large; its
notes are in the session scratchpad and summarized here. JEVal's and TypedBench's numbers are as reported in their
papers, unverified by us.

| Suite | Open test data / licence | Size | Format vs DI wire format | Published small / System-1 rows | Verdict |
|---|---|---|---|---|---|
| **System1-Arena** 2610.21244 | HF `C-Tianyu/S1-Arena`, not gated; generated requests CC BY 4.0, sources keep their own licences; code `TianyuCodings/S1-Arena` Apache-2.0; 14 sources only via `s1arena build --local` | 268 sources, 105 leaves, 16 domains; **non-agentic 13 domains, 43,226 requests, 78,362 questions** (our HF snapshot: 41,183 requests, 209 tasks; the 2,043 local-only requests are missing) | Same as DI (`boolean` maps to `noul`); `score` with 2–10 levels; up to 255 options; 138 closed loops (skipped) | [published] `leaderboard/results.json` (2026-10-14), accuracy as mean over tasks / Brier: Decision-2.0-Kai-0.6B 50.3% / 0.537; **Laya 421M 44.5% / 0.661**; **Qwen3-0.6B 39.2% / 0.937**; CLM-8B 41.4%; Jev 1.13 72.6% / 0.349; Gemma-4-31B 73.9%; Claude Opus 5.5 81.1%. Per-domain values for all 30 models are in the same file | **Chosen**: open data and code, the same format, a sub-1B tier, a few GPU-hours |
| JEVal 2610.03935 | HF `carlosxiang/JEVal`; **no licence on the data**; code `amazingljy1206/SII-Decision` | 11,257 items (1,800 Chinese) | Native typed | Kev-0.8B 49.6, Laya 26.3, Intern-0.8B 47.7, Jev 78.4 (accuracy) | Backup only |
| TypedBench 2610.11392 | **No data or code link anywhere** | 7 generators | — | Laya 0.34–0.38, Kev-0.8B 0.28, bart-mnli | Not usable |
| 2610.00346 harness | GitHub `pozapas/system-one-models-survey` (MIT; curated data CC BY 4.0); D1 = HF `LocalLLaMA/typed-decisions` (Apache-2.0) | About 6k decisions (D1: 2,000; CLINC 800; Banking77 616; GoEmotions 492; D3) | Typed | D1 accuracy: fine-tuned DeBERTa-v3-large 0.793; DeBERTa NLI zero-shot 0.411; Kev-0.8B 0.451; Laya 0.366. Yes/no swap flips Jev 50.5% | Optional cheap second panel; the only one with fine-tuned and NLI rows |

### 3.1 S1-Arena adapter, scoring and audit

**Adapter** (`s1arena/adapter.py`):
- Requests are taken unchanged from the parquet `request` column.
- The only mapping is `boolean` (criteria null) → `noul` (criteria {}), answered as P(true). It maps back as
  {"true": p, "false": 1 − p}.
- `choice` and `score` pass through. Score criteria are the ordered level list, with ids "0".."n-1".
- Request id = S1-Arena item id.

**Scoring** calls the benchmark's own `s1arena.metrics.score` per task, from S1-Arena code at its current
`--depth 1` clone:
- accuracy (argmax == gold), Brier and ECE;
- unanswered = wrong (uniform for Brier);
- domain = mean over its tasks;
- overall = mean over all tasks of the 13 non-agentic domains, the leaderboard definition (checked: the published
  overall is not the mean of domain means).

**Audit:**
- Target = the longest string leaf of each item's state (`adapter.py targets`).
- This is conservative: a shared premise counts as overlap even with a new hypothesis, which is why ANLI R3 shows 40%.
- Source exposure: the task name or its HF dataset id is matched against v2-family `src` names (`s1arena/classify.py`,
  with an alias list for squad_v2, nycc_captions to newyorker, isarcasmeval, scruples, nq_open and others). It is
  approximate; every non-unseen task is listed with its matched sources in `s1arena/contamination.md`.
- Files: `s1arena/contamination.{md,json}` and `s1arena/tasks.json`. Raw: `audit_raw/s1arena_audit_*.json`. Hits:
  `audit_raw/s1arena_hits_allmixes.jsonl` (933 items).

| Domain | Tasks | Unseen | Seen-train-split | Seen-test-overlap | Items in unseen tasks / all |
|---|---:|---:|---:|---:|---:|
| basic_cls | 21 | 6 | 6 | 9 | 998 / 4,083 |
| know | 30 | 15 | 9 | 6 | 3,000 / 6,000 |
| social | 13 | 6 | 3 | 4 | 833 / 1,874 |
| pro | 25 | 16 | 2 | 7 | 2,832 / 4,333 |
| ret | 11 | 9 | 1 | 1 | 1,749 / 2,165 |
| struct | 13 | 13 | 0 | 0 | 2,500 / 2,500 |
| code | 17 | 17 | 0 | 0 | 3,886 / 3,886 |
| fore | 5 | 5 | 0 | 0 | 920 / 920 |
| dialog | 10 | 7 | 2 | 1 | 1,714 / 2,464 |
| batch | 6 | 5 | 0 | 1 | 1,250 / 1,500 |
| judge | 24 | 19 | 1 | 4 | 3,831 / 4,998 |
| safety | 16 | 11 | 0 | 5 | 2,165 / 2,998 |
| route | 18 | 16 | 1 | 1 | 3,045 / 3,462 |
| **all** | **209** | **145** | **25** | **39** | **28,723 / 41,183** |

**Heaviest overlaps (v2 family):**

| Task | Overlap | Training source |
|---|---:|---|
| know/multihop/strategyqa | 53% | bekko/strategyqa |
| pro/legal/cuad | 41% | our CUAD sister source |
| basic_cls/nli/anli_r3 | 40% | premises shared with the ANLI train split |
| sms_spam | 32% | Bekko |
| scifact | 23% | tasksource |
| attributionbench | 19% | RAGBench ExpertQA |
| anthropic_persuasion | 18% | |
| equate | 14% | |
| toxicchat_jailbreak | 12% | |
| financial_phrasebank | 11% | |
| winogrande | 10% | |

**Caveats:**
- S1-Arena publishes per-domain and overall scores, not per task.
- The unseen/seen split is therefore available for our runs and for the baselines we run.
- Published rows (Kai, Laya, Qwen3-0.6B) can be compared only on all tasks, and on the HF snapshot, which lacks 2,043
  requests: say so in the paper. GPU-1's Qwen3-0.6B row is an approximate check against the published 39.2%. It is
  only approximate: the published row uses S1-Arena's `lm` adapter with label-token logprobs and its own prompt, which
  is a different readout.

---

## 4. Board statistics (task 4, E7)

Script: `stats/board_stats.py`. Output: `stats/board_stats.json`. Inputs: `release/di-index-20261020.json` (0.4, 208
models) and `release/di-index-v0.3.2-20261020.json`. CIs are percentile bootstraps over models, B = 10,000, seed 0.
All numbers are [derived] from [board] data.

**Public index vs private same-skills, all 208 models:**
- Pearson r = **0.981 [0.974, 0.986]**;
- Spearman ρ = **0.978 [0.969, 0.983]**.

**Other correlations with the public index:**

| Pair | Pearson r | Spearman ρ |
|---|---:|---:|
| public vs new-domains | 0.864 [0.828, 0.896] | 0.886 [0.851, 0.911] |
| public vs Full 0.4 | 0.946 [0.931, 0.959] | 0.947 [0.927, 0.959] |
| same-skills vs new-domains | 0.916 [0.894, 0.935] | 0.944 [0.923, 0.957] |

**Inside size buckets** (`meta.served_params`; 4 models without a size are left out):

| Bucket | n | r pub~same-skills [CI] | ρ [CI] | r pub~Full | ρ pub~Full |
|---|---:|---|---|---:|---:|
| < 100M | 6 | 0.972 [0.879, 0.999] | 0.943 [0.515, 1.000] | 0.968 | 0.943 |
| 100–500M | 20 | 0.952 [0.852, 0.993] | 0.941 [0.754, 1.000] | 0.928 | 0.905 |
| 0.5–2B | 29 | 0.974 [0.957, 0.986] | 0.969 [0.909, 0.987] | 0.932 | 0.935 |
| 2–10B | 78 | 0.975 [0.963, 0.984] | 0.977 [0.952, 0.987] | 0.918 | 0.880 |
| > 10B | 71 | 0.935 [0.884, 0.963] | 0.894 [0.807, 0.945] | 0.881 | 0.828 |

**Partial correlation controlling for log10(params)** (n = 204):
- pub~same-skills: Pearson **0.965 [0.951, 0.974]**; rank-partial **0.954 [0.929, 0.965]**;
- pub~Full: 0.890 [0.857, 0.917], rank-partial 0.891 [0.848, 0.916];
- for reference, r(pub, log params) = 0.763 and r(same-skills, log params) = 0.837.
- **Reading:** the 0.98 is not an artefact of pooling 17M with 30B models. The correlation stays at 0.94–0.97 within
  buckets and after controlling for size. It is weakest above 10B, where models crowd near the top.

**Crivo v1 vs v2, per benchmark:**
- Inputs: 34 benchmarks with `private_eq`; `models[e].benchmarks[k].{private_eq,random}`.
- Skill = clip((private_eq − random)/(1 − random), 0, 1).
- Paired across benchmarks; per-item data is not public.

| Pair | Mean private skill | Diff [bootstrap CI] | Down / up / same | Paired t p | Wilcoxon p | Sign-test p | Raw private_eq diff (Wilcoxon p) |
|---|---|---|---|---:|---:|---:|---|
| 150M v1 (`crivo-150m`, 0.3.2) → v2 (`crivo-150m-v2`, 0.4) | 0.242 → 0.221 | −0.021 [−0.075, +0.038] | 15 / 14 / 5 | 0.47 | 0.36 | 1.00 | −0.015 (0.43) |
| 32M v1 → v2 | 0.171 → 0.166 | −0.005 [−0.027, +0.018] | 15 / 10 / 9 | 0.67 | 0.24 | 0.42 | −0.003 (0.25) |

- Per-benchmark rows, sorted by change, are in `stats/board_stats.json` → `crivo_v1_vs_v2.*.rows`.
- **Reading:** per benchmark, the v1→v2 change is not significant. Large moves cancel out: NLI4CT, ACOS and CLadder
  went down; BPoMP, API-Bank and MMLU-Pro went up.
- The board's same-skill aggregate fell 27.74 → 24.93 because of the index weighting, which includes the gold ★
  weights.
- This supports compressing the negative result into a "model selection" paragraph, as plan_review §2 says.
- The same-skill tests are identical across the two editions (`facts.md` §4), so the comparison is valid.

---

## 5. Adapters (task 5)

### 5.1 BTZSC to Decision Index `choice` (`btzsc/adapter.py`)

One fixed rendering for all 22 datasets, chosen **before any run** and never tuned:
- `state` = the text, as given by BTZSC;
- `instructions` = `"Which label best describes the text?"`;
- `criteria` = {label name: BTZSC hypothesis}, in BTZSC's label order;
- Crivo renders each option as `key: description`.

**Why the hypotheses are the descriptions.** They are BTZSC's own label verbalizations, which every published BTZSC
model receives. Names alone are often opaque: `not_toxicaggregated`, `Refund_not_showing_up`.

**Secondary check:** `--render names` (names only) is declared as secondary.

**Scoring** (`adapter.py score`):
- macro-F1 per dataset with sklearn, as in the harness;
- overall is the mean over datasets; task means follow BTZSC's groups;
- unanswered = wrong: an impossible class −1. This is stricter than BTZSC, which has no unanswered case.

**Equivalence check** (`btzsc/harness_check.py`, result `btzsc/harness_check_17m_s50.json`):
- Crivo-17M was run at `max_samples=50` through the official `BTZSCBenchmark.evaluate`, with a `BaseModel` that sends
  exactly our request.
- Overall macro-F1 = **0.5100688**, and all 22 per-dataset values equal `adapter.py score` on our run of the same items
  (0.510068826796949) [ours-cpu].
- The protocol and the metric are identical.

**Request sets:**

| File | Requests | Contents |
|---|---:|---|
| `~/req/btzsc_full.jsonl` | 172,501 | all items |
| `~/evalsets/btzsc500.jsonl` | 11,000 | harness max_samples=500 |
| `~/btzsc/req_s50.jsonl` | 1,100 | max_samples=50 |

### 5.2 S1-Arena

See §3.1.

**Smoke test** [ours-cpu]: Crivo-17M on 5 items per task (1,045 requests, 209 tasks, 1,795 questions) answered 100%;
S1-Arena metrics give 0.406. This is a pipeline check, not a result.

**Score questions** are answered by the released model API: `crivo.Crivo.answer` supports choice, noul and score, and
the v2 mix contains `score` rows, so the type embedding is trained.

### 5.3 E4 robustness set (`e4/perturb.py`)

**Sample:**
- Public 0.3 suite: rebuilt GSM8K, ForecastBench dropped, the 442 exclusions dropped, and the non-index benchmarks
  (MMLU, ARC, SimpleBench, RouterBench, SGD) left out.
- Requests whose questions are all `choice` with 2–255 options and at most 32 questions.
- Per benchmark, up to 100 requests and 150 questions (seed 0).
- Result: **2,833 requests, 3,841 questions, 34 benchmarks**. RAGTruth and PhishNChips (noul) and ACOS (more than 32
  questions) are not in it.

**Variants** (only the order of `criteria` changes):

| Variant | Change |
|---|---|
| `perm1–3` | seeded random permutations (sha256 of id, question and k), never the identity |
| `rev` | reversed order |
| `yn` | binary yes/no-type questions (keys or descriptions in yes/no/true/false) swapped; 258 requests, mostly CLadder, BFCL, iSarcasmEval and BBH |

**Metrics** (`perturb.py flips`):
- flip rate = the share of questions whose chosen key differs from the original order's;
- accuracy per variant;
- the share consistent across all three permutations.

**Note:** "yes/no swap" here is an order swap. The 2610.00346 swap may be defined differently, so cite theirs as
"their harness".

The sample and variants are deterministic: `~/e4/req/orig.jsonl` has the same md5 on both VMs.

---

## 6. Baseline wrappers (task 6)

All wrappers live in `common/engines.py`. They take one request format and run through `common/run_requests.py`:
- resumable;
- every answer validated (keys, types, probability sums);
- unsupported inputs counted as wrong;
- `--batch N` for engines with `batch_call`.

| Engine (`--engine`) | Models | Rendering and reading (fixed for every task) | dtype |
|---|---|---|---|
| `crivo` | `tardellirs/crivo-{17m,32m,68m,150m,400m,1b}` (main = v2); v1 = 32m@`bbf3f1924523f530b3a9e621f2f20f314ea462aa`, 150m@`7a6c7453c06be576e1b48aeb70429318402791d7` | The released `crivo.Crivo.answer` (model folder's own `crivo.py`/`model.py`, max_length 32,768, nothing truncated). `batch_call` packs many requests' questions with the same released functions; **checked equal to `answer()`**: same decision 100%, max \|Δp\| 5.6e-7 (`common/check_batch.py`, 35 requests) [ours-cpu]. `code=<released folder>` lets it read `train_ablate.py` checkpoints | bf16 autocast on CUDA, fp32 head, fp32 on CPU |
| `nli` | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`, `MoritzLaurer/ModernBERT-large-zeroshot-v2.0`, `facebook/bart-large-mnli` | Premise = rendered state (JSON compact, sorted keys); hypothesis = `Question: {instructions} Answer: {option text}`; score = entailment logit (label2id `entailment`), softmax over the question's options; noul = options true/false. The premise is truncated `only_first` at 512 tokens (BART 1,024), as the BTZSC harness truncates: declared | bf16 on CUDA (`fp16=1`; DeBERTa-v3 overflows in fp16), batch 64 pairs |
| `letter` | `Qwen/Qwen3-0.6B`, `google/gemma-3-1b-it` (gated; downloaded with the user's token) | The BTZSC LLM prompt generalized with the question: "You are a decision engine… INPUT / QUESTION / OPTIONS (A) …) / Answer: The correct option is letter "; **no chat template** (as BTZSC); next-token probabilities of the option letters (A–Z, a–z, Greek; only single-token symbols) renormalized; noul = A) true / B) false; `rotations=k` = mean over k cyclic rotations (AnyJev-style, mapped back). Too many options or more than 32,768 tokens = Unsupported. `batch_call` uses left padding with explicit position ids | bf16 on CUDA |
| `reranker` | `cross-encoder/ettin-reranker-32m-v1` as-is: the "before" point of Crivo-32M | Query = instructions + blank line + rendered state; document = option text; one relevance logit; softmax over options; truncation `only_first` at 8,192 | fp32 |
| `prumo` | `tardellirs/prumo` (downloaded to a real folder: its `params.py` resolves its own path, which breaks in the HF cache's symlinks) | Released engine; `score` questions asked as a choice over the level ids | CPU |
| `majority` | — | Constant answer from a gold-label prior `{family\|question: {label: n}}`. An oracle constant baseline, best computed at scoring time from `expected`; not in the GPU jobs | — |
| `uniform` | — | Chance: uniform distribution, argmax = first option | — |
| `peropt:PerOptionEngine` (`a1a2/peropt.py`) | per-option checkpoints, or any Crivo read per option | See §7 | bf16 autocast |

**Smoke tests** [ours-cpu], on `smoke_small` (35 requests: 1 per BTZSC dataset plus 1 per S1-Arena domain), every
answer valid:

| Wrapper | Result |
|---|---|
| ModernBERT-zs | 35/35, 86 s |
| bart-mnli | 35/35, 140 s |
| reranker-32m | 35/35, 58 s |
| Qwen3-0.6B | 35/35, 246 s |
| Qwen3-0.6B rot4 | 35/35 |
| Prumo | 35/35, 1 s |
| uniform | 35/35 |
| DeBERTa-zs | 137/137 on the larger `smoke_mix` |
| Gemma-3-1B | 35/35, batched (`--batch 8`, `HF_HUB_OFFLINE=1`) |

**Two fixes found by the smoke tests:**
- Gemma's tokenizer tries to fetch an optional file that is gated online. Fixed by `HF_HUB_OFFLINE=1`, which the GPU
  lanes set.
- Prumo's path issue, described in the table.

---

## 7. Per-option engine and the A1/A2 grid (task 7)

### 7.1 A1 per-option reader (`a1a2/peropt.py`)

The joint reader is the released Crivo:

    [CLS] instr [SEP] state [SEP] [OPT] o1 [OPT] o2 … [OPT] on    → n logits, 1 pass

The per-option reader:

    [CLS] instr [SEP] state [SEP] [OPT] oi                        → 1 logit, n passes; softmax across the n

Everything else is shared: tokenizer, `[OPT]` token, `render`/`option_texts`, type embedding + MLP head (fp32), loss
and data. The only differences are attention between options and the cost: n sequences instead of 1.

### 7.2 Training (`a1a2/train_ablate.py`)

- It imports `train/train.py`'s rows, loss (CE + w·Brier, KD with α 0.5, τ 2 for more than 2 options, τ 1 otherwise),
  Teacher, `length_batches` and `save`, and adds `--reader joint|peropt`.
- Per-option rows build n sequences with the same state budget rule.
- Batches are capped by the **same token budget**. The per-option row estimate is n × (state + instructions + option),
  so a per-option run has about 4× the batches: 69 against 18 per epoch on the same 400 rows. That cost is part of A1's
  finding (1 pass against n).
- The same epochs run over the same questions.

### 7.3 Data

`a1a2/make_subset.py`:
- **150,000 rows** (plan_review: about 150k questions), sampled uniformly without replacement (seed 2026) from
  `mix400_s1.train` (the firewalled v2 stage-1 data) and shuffled.
- Teacher logits come from Crivo-400M (`runs/teacher_logits/mix400_s1.t400.npz`, the 32M v2 teacher), re-indexed.
- Gold-only sources: `runs/mix400_s1.t400.gold_only`.
- **The same rows serve every arm and seed.**
- Composition: bekko 41,560; src400 24,029; gen400 22,504; t2 21,944; t3 14,701; proc 10,892; pub 9,766; t1 4,604.
- Built at `~/exp/a1a2/sub150k.*` on VM1. `launch.sh grid` rebuilds it on the GPU box from the VPS copy; it is
  deterministic.

### 7.4 Grid

`a1a2/grid.tsv`: 5 arms × seeds {0, 1, 2} = 15 runs at 32M, 1 epoch.

**Shared settings:**
- lr 1e-4, head 5e-4: the 32M v2 stage-1 values (`facts.md` §1.2);
- AdamW (β 0.9/0.98, wd 0.01), warmup 5%, linear decay;
- max_len 2,048; 24,576 tokens per step; at most 128 items;
- option order shuffled; bf16; `--skip_oom`.

| Arm | Reader | Init | Loss | Tests |
|---|---|---|---|---|
| **J** (reference) | joint | `cross-encoder/ettin-reranker-32m-v1` | CE + 0.5·Brier + KD (400M) | — |
| **P** | per-option | reranker | same | A1: joint vs per option (accuracy, calibration, order robustness, cost) |
| **JnoBrier** | joint | reranker | CE + KD (`--brier 0`) | A2: Brier term |
| **JnoKD** | joint | reranker | CE + 0.5·Brier, no teacher | A2: distillation |
| **Jraw** | joint | `jhu-clsp/ettin-encoder-32m` (raw encoder) | CE + 0.5·Brier + KD | A2: reranker vs raw Ettin init |

**Evaluation sets** (`a1a2/build_eval.py`; none is trained on; all fixed):

| Set | Requests | Contents |
|---|---:|---|
| `pub15` | 13,358 | 15% of each of the 37 public 0.3 index benchmarks, at most 600 each, seed 0; choice and noul |
| `btzsc500` | 11,000 | the BTZSC harness sample |
| `s1_60` | 12,540 | 60 items per S1-Arena task |
| E4 files | — | the six files of §5.3 |

**Metrics:**
- `a1a2/score_pub.py`:
  - pub15 question accuracy, chance-corrected per benchmark (chance = mean 1/n), unweighted mean;
  - declared as **not** the official index;
  - pooled ECE (10 equal-mass bins) and Brier on choice questions.
- BTZSC macro-F1: all 22, and the unseen-4 mean.
- S1-Arena accuracy and Brier: all tasks and unseen tasks.
- E4 flip rates.

**Summary** (`a1a2/summarize.py` → `summary.json`):
- per arm, mean ± sd over seeds;
- each arm minus J, with a Welch 95% CI over seeds;
- a paired item-level bootstrap on pub15 (each question's correctness averaged over seeds, B = 2,000).

### 7.5 Verification

All on CPU [ours-cpu]:
- All 5 arms trained 6 steps on the real subset and saved. Every checkpoint answered 35/35 smoke requests:
  - joint arms through `crivo` with `code=` the released 32M folder;
  - P through `PerOptionEngine`.
- **Full dry run** of `a1a2/train_grid.sh` on VM1 (`HOME=/root/dry`, `SEEDS=0`, `EXTRA_TRAIN="--limit 200 --max_steps 3
  --max_tokens 8192 --workers 0"`, tiny eval sets): train → eval → score → `summarize.py` all ran.
- Two bugs found and fixed:
  - the S1-Arena code clone was missing on VM1;
  - `summarize.py` needed to tolerate missing datasets.
- The eval lane now marks `FAILED` instead of looping when a score file is missing.

### 7.6 Budget

- A100 throughput from past logs: 32M v2 stage 1 at a median **304 q/s**, on a GPU shared with the 150M v2 training
  (`runs/c32v2s1.log.jsonl`); 17M at 631 q/s.
- Joint run: 150k rows ≈ 8 min. Per-option run ≈ 33 min.
- Training: 12 × 8 + 3 × 33 ≈ 3.3 GPU-h. Evaluation ≈ 1.2 GPU-h.
- **About 4.5 GPU-h of work, about 3–3.5 h wall** with 2 training lanes and 1 eval lane on one A100.
- Up to about 250k rows would still fit in 8–10 A100-hours.

---

## 8. GPU job plan (task 8)

### Scripts

| Script | Role |
|---|---|
| `gpu/launch.sh <user@host> infer\|grid\|both` | Push code (and, for grid, the stage-1 data and teacher logits), run `setup_gpu.sh`, build E4 and the eval sets, start tmux `infer` and/or `grid` |
| `gpu/setup_gpu.sh` | venv, PyPI CUDA torch, transformers 5.19.0, kit; BTZSC and S1-Arena data plus S1-Arena code; request files; prefetch of every model (Prumo to `~/models/prumo`) |
| `gpu/jobs_infer.tsv` | One job per line: lane, name, engine, options (`-` if empty), request set, threads, batch |
| `gpu/run_lanes.sh` | 3 lanes on one GPU, `HF_HUB_OFFLINE=1`, resumable. Then `e6.sh` **alone on the GPU**, then `score_all.sh` |
| `gpu/e6.sh` | Latency |
| `gpu/score_all.sh` | Scoring |
| `a1a2/train_grid.sh` | 2 training lanes + 1 eval lane |

**Dry run:** a tiny `run_lanes.sh` on VM2 (`/root/dry2`) covered crivo, reranker, uniform, letter, and letter-rot4
partially. Then `e6.sh` (`LAT_N=10`) and `score_all.sh` ran: BTZSC, S1-Arena and E4 flips were all scored. Two bugs were
found and fixed before launch:
- empty-field parsing;
- offline partial snapshots.

**Status files:** `~/results/STATUS` and `~/results/a1a2/STATUS`. Monitor every 10 minutes.

### GPU-1: inference (launched 2026-10-20 on Lambda A100 SXM4)

| Experiment | Requests | Systems |
|---|---|---|
| E1 BTZSC | 172,501 | 6 Crivos + 32M v1 + 150M v1 (Crivo batch 256), Ettin-32M raw, DeBERTa-zs-v2, ModernBERT-zs-v2, Qwen3-0.6B, Gemma-3-1B (letter batch 16) |
| E2 S1-Arena | 41,183 | 6 Crivos + 150M v1, raw Ettin, DeBERTa-zs, BART-MNLI, Qwen3-0.6B (rot4 only on `s1_60`), Gemma-1B, uniform |
| E4 | 2,833 × 6 variants | 6 Crivos + reranker, DeBERTa-zs, Qwen3-0.6B, Gemma-1B |
| E6 GPU | 1,000 fixed requests, one at a time, GPU unshared | 6 Crivos, reranker, DeBERTa-zs, Qwen3-0.6B, and 32M joint vs per-option (same weights; cost only) |

Expected about 4–5 GPU-h of work, **about 3–4 h wall** plus about 20 min setup.

### Cost estimate

| Session | Verda spot A100 (US$0.75/h) | Lambda A100 40 GB (US$1.99/h) |
|---|---:|---:|
| GPU-1 inference | US$2.5–3.5 | US$7–9 |
| GPU-2 grid | US$2.5–3 (preemption risk) | **US$6–8 (recommended)** |
| **Total** | **US$9–17**, depending on the provider mix | |

### Optional jobs (not run, not in the totals)

- A4: DeBERTa-v3-base vs Ettin-150M, 2 seeds, about 2 GPU-h.
- The official S1-Arena `lm:Qwen/Qwen3-0.6B` vLLM adapter, to reproduce the published 39.2% exactly: about 0.5 h,
  needs vLLM.
- Public index of Qwen3-0.6B through the kit's `transformers` engine, the board's stock protocol: about 1.5 h.
- The 2610.00346 harness D1 panel: minutes.

---

## 9. Known issues and caveats (read before writing)

1. **BTZSC:**
   - Only 4 datasets are unseen for v2. The "seen-train-split" class covers anything from about 300 tasksource rows
     (most datasets) to 18.7k rows (banking77).
   - The banking77 and massive gold-0 quirk is the harness's, and we keep it for comparability.
   - Our unanswered = wrong rule is stricter than BTZSC's.
2. **S1-Arena:**
   - The HF snapshot lacks 2,043 local-only requests.
   - Its leaderboard has no per-task numbers.
   - Our source matching is name-based and approximate.
   - The item audit uses the longest state leaf, which is conservative (premise-level).
3. **Truncation:**
   - Crivo never truncates.
   - The NLI wrappers truncate the premise to 512 (BART 1,024), as BTZSC's harness does.
   - The letter LLMs refuse prompts above 32,768 tokens.
   - The reranker truncates at 8,192.
   - All declared.
4. **The letter readout** has no chat template (the BTZSC choice). Qwen3-0.6B may score lower than with S1-Arena's own
   `lm` adapter.
5. **Crivo on GPU** runs bf16 autocast with an fp32 head (released behaviour). CPU smoke runs were fp32.
6. **pub15 metric:** for ablations only, unweighted chance-corrected question accuracy. It is not the official index.
7. **E4 "yes/no swap"** is an option-order swap on binary yes/no-type questions only: 258 requests, mostly CLadder,
   BFCL, iSarcasmEval and BBH. RAGTruth and PhishNChips (noul) have no order.
8. **Board stats:** per-benchmark tests use benchmarks as units, since private per-item data is not public. Rank groups
   are the board's.
9. **GPU-1 first launch failed** on Lambda Stack. The cause was two things:
   - the image has Python 3.10, and transformers 5.19 needs 3.11 or later;
   - `python-chess==1.11.2` does not exist: the package is `chess==1.11.2`.

   The main session patched `gpu/setup_gpu.sh`:
   - it installs `uv` and builds `~/venv` with Python 3.12 (`uv venv --seed --python 3.12`), recreating an old venv
     below 3.11;
   - it installs `chess==1.11.2`.

   It then relaunched. The VMs (Ubuntu 24.04, Python 3.12) were not affected. Their `scripts/setup_vm.sh` also lists
   `python-chess`; the 1.11.2 found on VM2 came from the pip resolution.
10. **No content from one excluded third-party evaluation panel** is used anywhere; its files were never read.
11. **Credentials:**
    - The Gemma download used the user's HF token, passed in the environment for the download only. On VM2 it went
      through a 0600 temp file that was deleted.
    - The VPS public key was added to both VMs.

---

## 10. CPU runs at hand-off: Prumo and E6 CPU latency

### 10.1 Prumo (finished)

Run on VM2 (`~/exp/prumo/run.sh`) with 12 shards. Copies on the Mac: `results_cpu/prumo/{btzsc.score.json,
s1.score.json,flips.json,e4.flips.txt}`. Raw runs: `~/exp/prumo/{btzsc,s1}.jsonl` and `e4/` on VM2.

| Benchmark | Result |
|---|---|
| BTZSC, 172,501 requests | all ok; **overall macro-F1 0.209**; unseen-4 0.201 |
| S1-Arena, 41,183 requests (72,817 questions) | all ok; **accuracy 0.340**, Brier 0.605 (mean over 209 tasks); unseen tasks (145) 0.345 |
| E4 | flip rate **0.000** on perm1–3, rev and yn (894 yes/no questions); accuracy 0.423 on every variant |

**BTZSC per dataset:**
- Mostly near a constant answer: 0.333 on the binary sentiment sets and trueteacher.
- 0.000 on banking77 and 0.001 on massive, 0.004 on manifesto and 0.005 on capsotu.
- With the hypothesis descriptions, the label rules do not separate the options. This is Prumo's behaviour under the
  fixed rendering and is reported as is (a floor).

**E4:** Prumo is order-invariant by construction.

**All Prumo numbers are [ours-cpu] full runs**, so they are real results, not smoke tests: Prumo is deterministic, and
CPU is its released runtime.

### 10.2 E6 CPU latency (running at hand-off)

Run on VM1, tmux `e6cpu`, script `~/exp/e6cpu2.sh`; nothing else runs on VM1.

| Setting | Requests |
|---|---|
| 16 threads | first 300 of `~/req/lat1000.jsonl` (seed-7 shuffle of the E4 sample) |
| 1 thread | first 100 (the 400M/1B take seconds per request at 1 thread) |
| Prumo, 1 thread | 1,000 |
| DeBERTa-zs, 16 threads | 100 |
| 32M per-option, 16 threads | 300 |

- Every run is one request at a time with 10 warm-up requests.
- Output: `~/exp/e6cpu/*.json`; `~/exp/e6cpu/DONE` when finished.
- Fetch with `rsync -a <cpu-vm1>:exp/e6cpu/ pareto/paper/exp/results_cpu/e6cpu/`.
- Finished so far (fetched 2026-10-20 ~20:30 UTC; the rest (1B at 1 thread, Prumo, DeBERTa-zs, 32M per-option) still running) [ours-cpu]:

| Run (model.threads) | Requests timed | Median ms | p95 ms | Mean ms | Requests/s | Peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|
| crivo-150m.t1 | 100 | 700.9 | 40106.8 | 5327.0 | 0.19 | 1843 |
| crivo-150m.t16 | 300 | 91.6 | 4957.4 | 720.5 | 1.39 | 3527 |
| crivo-17m.t1 | 100 | 34.8 | 2798.7 | 437.1 | 2.29 | 1318 |
| crivo-17m.t16 | 300 | 9.7 | 479.6 | 72.7 | 13.76 | 2101 |
| crivo-1b.t16 | 300 | 596.6 | 22495.5 | 3699.3 | 0.27 | 9645 |
| crivo-32m.t1 | 100 | 93.7 | 7002.3 | 973.7 | 1.03 | 1440 |
| crivo-32m.t16 | 300 | 19.0 | 1069.0 | 146.5 | 6.83 | 2399 |
| crivo-400m.t1 | 100 | 2070.0 | 83979.0 | 12363.6 | 0.08 | 3502 |
| crivo-400m.t16 | 300 | 253.5 | 10569.1 | 1661.5 | 0.60 | 5489 |
| crivo-68m.t1 | 100 | 290.7 | 21293.4 | 2660.0 | 0.38 | 1522 |
| crivo-68m.t16 | 300 | 49.0 | 2853.2 | 386.4 | 2.59 | 2737 |
| reranker-32m.t16 | 300 | 40.2 | 18198.0 | 2822.7 | 0.35 | 11141 |

- The heavy tail comes from long ToolRet, BRIGHT and BFCL requests with up to 32 questions.
- Read medians against the board's GPU latency.
- The VM CPU is 16 vCPU AMD Turin, a shared cloud VM: say so in the paper.

## 11. What remains

1. **Collect GPU-1 results** from `<gpu1-host>:~/results/` (rsync), then terminate the instance. Then:
   - fill the E1 table: Crivos vs published BTZSC, unseen-4 and all-22 with classes, plus a decontaminated re-score
     with `audit_raw/btzsc_hits_allmixes.jsonl`;
   - fill the E2 table: S1-Arena overall, by domain, unseen-task subset, vs Kai, Laya and Qwen3-0.6B [published];
   - fill E4 flip rates and E6 latency.
   - **Second stop point:** does any Crivo of 150M or less tie or beat a known baseline of equal or larger size with a
     CI? The candidates are DeBERTa-v3-large NLI, Qwen3-0.6B, Laya and Kai-0.6B. If none does, the plan says TMLR.
2. **GPU-2 grid,** once the user approves: `bash gpu/launch.sh <host> grid`. The PARETO path must contain `data/` and
   `runs/` (the VPS, or the Mac copy).
3. **Bootstrap CIs** for E1 and E2 (paired, per item, Crivo vs each baseline, on identical items). The run files hold
   per-item answers. **Done 2026-10-21 (§16):** `stats/paired_bootstrap.py` (`extract` + `s1`/`btzsc --items`) ran the
   per-item paired bootstrap on the request files copied from GPU-2; the task/dataset CIs are kept alongside.
4. ~~**E5 calibration** from our public-suite runs~~ **done (§13)**, with the public-suite error analysis (§14). E1/E2
   calibration (ECE-10 equal-mass, Brier) on the GPU-1 runs is **done** too (§16.5).
5. **Fold §1–§4 numbers into the paper:** the tables in `stats/`, `btzsc/` and `s1arena/` are ready to copy.

---

## 12. Reproduction commands

```bash
# VMs
bash scripts/setup_vm.sh                                   # on each VM (CPU torch, transformers 5.19.0, kit -e)
# BTZSC data / results / harness (VM1)
python -c "from huggingface_hub import snapshot_download as s; s('btzsc/btzsc-results',repo_type='dataset',local_dir='results'); s('btzsc/btzsc',repo_type='dataset',local_dir='data')"
git clone https://github.com/IliasAarab/btzsc harness && pip install -e harness --no-deps
python btzsc/published.py ~/btzsc/results ~/btzsc/harness btzsc/published.json
python btzsc/adapter.py targets  --root ~/btzsc/data --out ~/btzsc/targets.jsonl
python btzsc/adapter.py requests --root ~/btzsc/data --out ~/req/btzsc_full.jsonl          # [--max_samples 50] [--render names]
python btzsc/adapter.py score --requests REQ --results RES --out S.json
python btzsc/harness_check.py <crivo dir> ~/btzsc/data 50 out.json                     # official-harness equivalence
# S1-Arena (VM2)
python -c "from huggingface_hub import snapshot_download as s; s('C-Tianyu/S1-Arena',repo_type='dataset',local_dir='data')"
git clone --depth 1 https://github.com/TianyuCodings/S1-Arena.git code
python s1arena/adapter.py requests --data ~/s1arena/data --out ~/req/s1_all.jsonl       # [--per_task 5]
python s1arena/adapter.py targets  --data ~/s1arena/data --out ~/s1arena/targets.jsonl
python s1arena/adapter.py score --data ~/s1arena/data --code ~/s1arena/code --requests REQ --results RES --out S.json
# contamination (VM1, from ~/pareto/data; ~4 min per pass with 15 workers)
PYTHONHASHSEED=0 python common/contam.py --targets T.jsonl --train mix400_s*.jsonl.gz --out A.json --hits_out H.jsonl --workers 15
bash ~/exp/audit/run_groups.sh                              # = audit_raw/run_groups.sh: the 3 families x 2 benchmarks
python common/src_count.py src_counts.json $(ls mix*.gz long_v3.jsonl.gz | grep -v minimix)
python btzsc/classify.py audit_raw audit_raw/src_counts.json btzsc/contamination.json btzsc/contamination.md
python s1arena/classify.py audit_raw audit_raw/src_counts.json s1arena/tasks.json s1arena/contamination.json s1arena/contamination.md
# board statistics (Mac or VM; seconds)
python stats/board_stats.py ../../release/di-index-20261020.json ../../release/di-index-v0.3.2-20261020.json stats/board_stats.json
python stats/stock_baselines.py ../../release/di-index-20261020.json ../../release/di-methodology-20261020.json stats/stock_baselines.md stats/stock_baselines.json
# any system on any request file (PYTHONPATH must contain the kit only for prumo; a1a2 for peropt)
python common/run_requests.py --engine crivo --opt path=tardellirs/crivo-17m --requests REQ --out RES --batch 256
python common/run_requests.py --engine nli --opt model=MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --opt fp16=1 --requests REQ --out RES
python common/run_requests.py --engine letter --opt model=Qwen/Qwen3-0.6B [--opt rotations=4] --requests REQ --out RES --batch 16
python common/check_batch.py --engine crivo --opt path=... --requests REQ --limit 200      # batched == unbatched
# E4 / E6
python e4/perturb.py sample --suite ~/pareto/kit/suite-0.3 --out ~/e4/e4_orig.jsonl && python e4/perturb.py make --rows ~/e4/e4_orig.jsonl --outdir ~/e4/req
python e4/perturb.py flips --rows ~/e4/req/orig.jsonl --results <dir with orig.jsonl perm1.jsonl … yn.jsonl>
python e6/latency.py --engine crivo --opt path=tardellirs/crivo-17m --requests ~/req/lat1000.jsonl --threads 1 --out lat.json
# A1/A2
python a1a2/make_subset.py --train mix400_s1.train.jsonl.gz --teacher mix400_s1.t400.npz --n 150000 --seed 2026 --out sub150k.jsonl.gz --out_teacher sub150k.t400.npz
python a1a2/build_eval.py --suite ~/pareto/kit/suite-0.3 --btzsc ~/btzsc/data --s1 ~/s1arena/data --outdir ~/evalsets
bash a1a2/train_grid.sh                                     # env: SEEDS, EXTRA_TRAIN, EVALDIR, E4DIR, OUT for dry runs
# GPU
bash gpu/launch.sh <user@host> infer|grid|both
```

---

## 13. E5 calibration (done 2026-10-20, CPU)

**Label for every number in §13–§14: [ours-public].** These are our own computations on the public 0.3 items, from the
released official-kit runs. They are not board numbers and not smoke tests: every run is the complete public suite.
Board numbers are marked [board]. Nothing was fitted: no temperature, no recalibration, no threshold. Temperature
scaling would need a dev split, and fitting it on the public suite would be tuning on test data, so we did not do it.

### 13.1 Runs used (provenance)

| Model | Run on the Mac (`pareto/…`) | Kit / engine | Public index (published) | Kit re-score on VM2 |
|---|---|---|---:|---:|
| Crivo-17M | `release/decision-index-results/runs/crivo-17m` (compact; = `runs/crivo17m-suite`) | kit `9eb2dbe`, `decision_index_engine` | 22.15 | 22.15 |
| Crivo-32M v1 | `results32/crivo32-suite` (engine2, `runs/c32W`, tta 1, fp32 head) | 0.3 | 25.06 | 25.06 |
| Crivo-32M v2 | `release/decision-index-results/runs/crivo-32m-v2` | 0.3 | 27.34 | 27.34 |
| Crivo-68M | `release/decision-index-results/runs/crivo-68m` | 0.3 | 32.82 | 32.82 |
| Crivo-150M v1 | `results/soup23-suite` (engine, `runs/soup23`, max_length 32,768) | 0.3 | 36.39 | 36.39 |
| Crivo-150M v2 | `release/decision-index-results/runs/crivo-150m-v2` | 0.3 | 34.38 | 34.38 |
| Crivo-400M | `release/decision-index-results/runs/crivo-400m` | 0.3 | 39.79 | 39.79 |
| Crivo-1B | `release/decision-index-results/runs/crivo-1b` | 0.3 | 47.77 | 47.77 |
| Prumo | `release/decision-index-results/runs/prumo` (= `runs/zero/prumo-suite`) | 0.3 | 20.14 | 20.14 |

- Each run has 140,620 rows (140,178 scoreable + 442 excluded) and every request is `ok`.
- The two v1 runs are not in the local `release/decision-index-results` copy. They were identified by their index
  values, which match the cards (`release/crivo-32m/README.md`, `release/crivo-150m/README.md`), and by their engine
  paths.
- **Pipeline check:** `python -m decision_index score --edition 0.3` re-scores all nine runs on VM2 to exactly their
  published public index. The outputs are in `e5/out/rescore/<model>/`.

### 13.2 Item set and definitions (`e5/extract.py`, `e5/calib.py`)

**Per-question table.** `extract.py` reads the frozen suite through the kit's own `Suite(...).rows(apply_exclusions=True)`,
so exclusions, the ACOS 400-review subset, the ToolRet/BRIGHT/Home-appliance cuts and the rebuilt GSM8K all apply as
they do in `score`. Per field:
- **correct** follows the kit's rule: `a["choice"]` is in the accepted set. The accepted set is `scoring.accepted`
  for ChessBench and Habermas (ties accepted), and `[gold]` otherwise. For noul, `(noul ≥ 0.5) == gold`.
- **Added benchmarks** use the kit's `scored_keys`: unscored fields are dropped, and PhishNChips counts only
  `verdict`.
- iSarcasmEval track B is skipped, because the kit does not score it.
- Probabilities are renormalised to sum to 1. A noul answer p becomes [1 − p, p].
- No field was unanswered or invalid in any run.

**Item set "MAIN"** = every scored field of the **35 index benchmarks that have a per-field right/wrong answer**: the
37 index benchmarks minus ToolRet and BRIGHT. Those two are nDCG over candidates, and unjudged candidates are not
verified negatives. RouterBench and the other not-in-index benchmarks are excluded too.
- Size: **218,870 fields in 107,782 requests**.
- "noul" in MAIN is RAGTruth only (2,700 fields). Every other binary decision in the suite is a two-option `choice`
  (yes/no and the like), so §13.4 also splits choice by k.

**Weighting.** The headline is "bench" weighting: each benchmark weighs 1/35, and fields weigh equally inside it. This
is the board's convention ("each [benchmark] weighted equally"). "Pooled" weighting gives every field equal weight
and is dominated by ACOS (91k fields), SATA-Bench, MMLU-Pro and HellaSwag.

**Metrics** (top-label):
- **acc**: accuracy.
- **conf**: mean of max p.
- **ECE-15**: 15 equal-width bins.
- **aECE-15**: adaptive ECE, 15 equal-mass bins (equal weight, stable sort).
- **Brier**: multi-class, Σ_options (p − y)², with the accepted set merged into one class.
- **NLL**: −log p(accepted), clipped at 1e-12.
- **over-confidence**: conf − acc.
- **over95**: the share of all fields answered wrong with conf ≥ 0.95, the board's field. Our check: this definition
  reproduces the board's values to within 0.011.
- **share≥.95 / acc@.95**: the share of fields at conf ≥ 0.95 and the accuracy among them.
- **AURC**: area under the selective risk–coverage curve, with tied confidences entering together. Lower is better.

**CIs:** stratified cluster bootstrap. Requests are resampled with replacement inside each benchmark, and the fields of
a request move together. B = 1,000, seed 0. The draws are the same for every model, so differences between models
are paired.

### 13.3 Main table (MAIN, bench-weighted) [ours-public]

| Model | Acc | Conf | Conf − acc | ECE-15 [95% CI] | aECE-15 [95% CI] | Brier [95% CI] | NLL | over95 | share≥.95 / acc@.95 | AURC |
|---|---:|---:|---:|---|---|---|---:|---:|---|---:|
| Crivo-17M | 0.514 | 0.590 | +0.076 | 0.076 [0.072, 0.079] | 0.076 [0.072, 0.079] | 0.574 [0.570, 0.577] | 1.311 | 0.021 | 0.139 / 0.85 | 0.290 |
| Crivo-32M v1 | 0.513 | 0.615 | +0.102 | 0.102 [0.099, 0.106] | 0.102 [0.099, 0.106] | 0.573 [0.570, 0.576] | 1.282 | 0.014 | 0.164 / 0.91 | 0.263 |
| Crivo-32M v2 | 0.534 | 0.659 | +0.125 | 0.125 [0.121, 0.128] | 0.125 [0.121, 0.128] | 0.574 [0.570, 0.578] | 1.356 | 0.025 | 0.203 / 0.88 | 0.260 |
| Crivo-68M | 0.557 | 0.672 | +0.115 | 0.115 [0.112, 0.119] | 0.115 [0.112, 0.119] | 0.551 [0.547, 0.555] | 1.313 | 0.032 | 0.248 / 0.87 | 0.236 |
| Crivo-150M v1 | 0.571 | 0.725 | +0.154 | 0.154 [0.150, 0.158] | 0.154 [0.150, 0.158] | 0.559 [0.554, 0.564] | 1.290 | 0.041 | 0.312 / 0.87 | 0.211 |
| Crivo-150M v2 | 0.578 | 0.701 | +0.123 | 0.123 [0.119, 0.127] | 0.123 [0.119, 0.127] | 0.544 [0.541, 0.549] | 1.302 | 0.032 | 0.260 / 0.88 | 0.226 |
| Crivo-400M | 0.593 | 0.726 | +0.132 | 0.132 [0.129, 0.136] | 0.132 [0.129, 0.136] | 0.526 [0.521, 0.530] | 1.280 | 0.040 | 0.336 / 0.88 | 0.196 |
| Crivo-1B | 0.643 | 0.801 | +0.158 | 0.158 [0.154, 0.161] | 0.158 [0.154, 0.161] | 0.499 [0.495, 0.504] | 1.338 | 0.047 | 0.440 / 0.89 | 0.160 |
| Prumo | 0.458 | 0.501 | +0.043 | 0.054 [0.051, 0.058] | 0.057 [0.054, 0.061] | 0.600 [0.597, 0.603] | 1.440 | 0.021 | 0.093 / 0.77 | 0.398 |

**Pooled weighting** (every field equal):

| Model | Acc | Conf | Conf − acc | ECE-15 [CI] | aECE-15 [CI] | Brier [CI] | NLL | over95 | AURC |
|---|---:|---:|---:|---|---|---|---:|---:|---:|
| Crivo-17M | 0.691 | 0.669 | −0.021 | 0.062 [0.061, 0.064] | 0.066 [0.065, 0.068] | 0.389 [0.387, 0.391] | 0.836 | 0.008 | 0.120 |
| Crivo-32M v1 | 0.688 | 0.751 | +0.062 | 0.062 [0.061, 0.064] | 0.062 [0.061, 0.064] | 0.380 [0.377, 0.383] | 0.791 | 0.017 | 0.102 |
| Crivo-32M v2 | 0.690 | 0.759 | +0.070 | 0.070 [0.068, 0.072] | 0.070 [0.068, 0.072] | 0.391 [0.389, 0.394] | 0.834 | 0.019 | 0.109 |
| Crivo-68M | 0.702 | 0.771 | +0.069 | 0.069 [0.067, 0.071] | 0.069 [0.067, 0.071] | 0.370 [0.367, 0.372] | 0.804 | 0.015 | 0.094 |
| Crivo-150M v1 | 0.727 | 0.820 | +0.093 | 0.093 [0.092, 0.095] | 0.093 [0.092, 0.095] | 0.353 [0.350, 0.356] | 0.757 | 0.024 | 0.074 |
| Crivo-150M v2 | 0.729 | 0.782 | +0.053 | 0.053 [0.052, 0.055] | 0.054 [0.053, 0.056] | 0.348 [0.346, 0.350] | 0.775 | 0.013 | 0.094 |
| Crivo-400M | 0.742 | 0.821 | +0.079 | 0.079 [0.078, 0.081] | 0.079 [0.078, 0.081] | 0.337 [0.334, 0.339] | 0.746 | 0.024 | 0.072 |
| Crivo-1B | 0.787 | 0.872 | +0.085 | 0.085 [0.083, 0.086] | 0.085 [0.083, 0.086] | 0.292 [0.290, 0.295] | 0.709 | 0.027 | 0.051 |
| Prumo | 0.463 | 0.490 | +0.027 | 0.036 [0.035, 0.038] | 0.041 [0.039, 0.043] | 0.565 [0.564, 0.567] | 1.077 | 0.010 | 0.453 |

**Reading:**
- **The Crivo models are over-confident in every bin, so ECE equals the over-confidence gap.** With bench weighting,
  accuracy is below confidence in all 15 bins for every Crivo (14 of 14 occupied bins for the 1B). The signed sum
  then equals the absolute sum, so ECE-15 = aECE-15 = conf − acc exactly. The binning choice therefore does not matter
  here. Prumo is the exception: 11 of its 15 bins are over-confident.
- **Size improves accuracy, Brier and AURC, but not ECE.**
  - From 17M to 1B: Brier falls from 0.574 to 0.499, AURC from 0.290 to 0.160, and acc@0.95 stays at 0.85–0.91. The
    probabilities rank a model's own errors better as it grows (AURC), which is the useful property for abstention
    or routing.
  - ECE grows from 0.076 to 0.158, and NLL stays flat at 1.28–1.36.
  - Every adjacent step in the v2 family changes ECE significantly (§13.5). The direction is mixed below 400M and
    rises at 400M→1B.
- **Do not say "calibrated"** (plan_review §3, E5). A defensible phrasing: *"probabilities are informative (AURC
  0.16–0.29, Brier 0.50–0.57 against 0.673 for a uniform guess) but systematically over-confident, by 8–16 points of confidence on
  average; the smallest model is the least over-confident."*
- Pooled weighting gives lower ECE (0.053–0.093), because the many-field benchmarks are easy. Pooled, the 17M is
  slightly *under*-confident (−0.021). Reference Brier for a uniform guess: 0.673 bench-weighted, 0.598 pooled
  (mean chance 0.326). The paper should report the bench-weighted figures, which follow the board's
  convention.

### 13.4 By question type and by area [ours-public]

**By type** (bench-weighted within each subset; acc / conf / ECE-15 / aECE-15 / Brier / NLL):

| Model | choice, k = 2 (18 benchmarks, 149,019 fields) | choice, k > 2 (22 benchmarks, 67,151 fields) | noul = RAGTruth (2,700 fields) |
|---|---|---|---|
| Crivo-17M | 0.632 / 0.667 / **0.036** / 0.036 / 0.426 / 0.613 | 0.405 / 0.517 / 0.112 / 0.112 / 0.698 / 1.801 | 0.766 / 0.700 / 0.081 / 0.081 / 0.361 / 0.546 |
| Crivo-32M v1 | 0.629 / 0.696 / 0.067 / 0.067 / 0.431 / 0.619 | 0.419 / 0.542 / 0.123 / 0.123 / 0.676 / 1.724 | 0.796 / 0.783 / 0.029 / 0.029 / 0.308 / 0.482 |
| Crivo-32M v2 | 0.648 / 0.736 / 0.090 / 0.091 / 0.443 / 0.648 | 0.442 / 0.593 / 0.151 / 0.151 / 0.678 / 1.843 | 0.783 / 0.745 / 0.047 / 0.052 / 0.323 / 0.500 |
| Crivo-68M | 0.660 / 0.746 / 0.087 / 0.088 / 0.432 / 0.646 | 0.475 / 0.603 / 0.128 / 0.128 / 0.637 / 1.751 | 0.790 / 0.755 / 0.068 / 0.066 / 0.323 / 0.508 |
| Crivo-150M v1 | 0.661 / 0.781 / 0.122 / 0.122 / 0.449 / 0.689 | 0.496 / 0.668 / 0.172 / 0.172 / 0.646 / 1.705 | 0.813 / 0.896 / 0.082 / 0.083 / 0.285 / 0.462 |
| Crivo-150M v2 | 0.672 / 0.766 / 0.093 / 0.097 / 0.426 / 0.690 | 0.495 / 0.644 / 0.149 / 0.149 / 0.640 / 1.713 | 0.782 / 0.749 / 0.071 / 0.068 / 0.337 / 0.537 |
| Crivo-400M | 0.684 / 0.805 / 0.122 / 0.122 / 0.428 / 0.668 | 0.514 / 0.655 / 0.141 / 0.141 / 0.605 / 1.695 | 0.751 / 0.824 / 0.078 / 0.081 / 0.365 / 0.594 |
| Crivo-1B | 0.729 / 0.860 / 0.133 / 0.131 / 0.394 / 0.682 | 0.556 / 0.737 / 0.182 / 0.182 / 0.603 / 1.821 | 0.835 / 0.891 / 0.057 / 0.056 / 0.253 / 0.431 |
| Prumo | 0.591 / 0.597 / 0.014 / 0.035 / 0.424 / 0.611 | 0.390 / 0.455 / 0.080 / 0.092 / 0.694 / 1.955 | 0.659 / 0.700 / 0.041 / 0.043 / 0.453 / 0.646 |

- Multi-option questions carry most of the miscalibration: ECE 0.11–0.18, against 0.04–0.13 on binary choice.
- The noul type is a single benchmark (RAGTruth) and cannot support claims about "binary predicates" in general.
  The 32M v1 is best there (0.029).

**By area** (bench-weighted). Each cell is ours: ECE-15 (acc/conf), followed by the [board] area ECE. The board's area
fields come from its 1-in-6 sample, 10 bins:

| Model | Knowledge | Language | Retrieval | Tools | Arts |
|---|---|---|---|---|---|
| Crivo-17M | 0.100 (0.31/0.41) / 0.130 | 0.035 (0.65/0.64) / 0.053 | 0.085 (0.69/0.77) / 0.123 | 0.213 (0.68/0.89) / 0.191 | 0.059 (0.44/0.50) / 0.056 |
| Crivo-32M v1 | 0.134 (0.29/0.43) / 0.118 | 0.041 (0.66/0.70) / 0.056 | 0.083 (0.72/0.80) / 0.111 | 0.196 (0.62/0.82) / 0.177 | 0.104 (0.45/0.55) / 0.068 |
| Crivo-32M v2 | 0.174 (0.31/0.49) / 0.194 | 0.047 (0.67/0.71) / 0.053 | 0.122 (0.71/0.84) / 0.148 | 0.150 (0.75/0.90) / 0.140 | 0.136 (0.45/0.58) / 0.141 |
| Crivo-68M | 0.139 (0.33/0.46) / 0.120 | 0.046 (0.70/0.75) / 0.036 | 0.141 (0.74/0.88) / 0.133 | 0.133 (0.79/0.92) / 0.138 | 0.142 (0.46/0.61) / 0.145 |
| Crivo-150M v1 | 0.191 (0.36/0.55) / 0.175 | 0.077 (0.75/0.83) / 0.072 | 0.138 (0.77/0.91) / 0.174 | 0.189 (0.74/0.92) / 0.189 | 0.190 (0.41/0.60) / 0.161 |
| Crivo-150M v2 | 0.155 (0.36/0.51) / 0.115 | 0.038 (0.73/0.77) / 0.044 | 0.158 (0.74/0.90) / 0.208 | 0.116 (0.80/0.92) / 0.109 | 0.168 (0.47/0.64) / 0.183 |
| Crivo-400M | 0.172 (0.36/0.53) / 0.141 | 0.060 (0.75/0.81) / 0.057 | 0.155 (0.75/0.90) / 0.157 | 0.117 (0.83/0.95) / 0.105 | 0.161 (0.49/0.65) / 0.183 |
| Crivo-1B | 0.195 (0.42/0.61) / 0.165 | 0.072 (0.82/0.89) / 0.072 | 0.147 (0.78/0.93) / 0.157 | 0.077 (0.89/0.97) / 0.090 | 0.281 (0.52/0.80) / 0.310 |
| Prumo | 0.031 (0.42/0.43) / 0.031 | 0.035 (0.38/0.42) / 0.018 | 0.090 (0.53/0.57) / 0.105 | 0.124 (0.64/0.75) / 0.154 | 0.075 (0.45/0.52) / 0.093 |

- **Language is well calibrated at every size** (0.035–0.077).
- **Tools are the worst area for the small models.** The 17M is 0.89 confident at 0.68 accuracy. Tools improve
  steadily with size, down to 0.077 for the 1B.
- **Knowledge, arts and retrieval get worse with size.** The 1B's arts ECE is 0.28: it is 0.80 confident on taste
  tasks where its accuracy is 0.52.

### 13.5 Paired differences (bench-weighted; difference and 95% paired-bootstrap CI)

| Pair (A − B) | ΔECE-15 | ΔBrier | Δacc |
|---|---|---|---|
| 32M v1 − 32M v2 | −0.022 [−0.027, −0.018] | −0.001 [−0.005, +0.003] | −0.021 [−0.025, −0.017] |
| 150M v1 − 150M v2 | **+0.031 [+0.027, +0.035]** | +0.015 [+0.010, +0.019] | −0.007 [−0.011, −0.003] |
| 17M − 32M v2 | −0.049 [−0.054, −0.044] | −0.000 [−0.004, +0.004] | −0.020 [−0.025, −0.016] |
| 32M v2 − 68M | +0.009 [+0.005, +0.014] | +0.023 [+0.019, +0.028] | −0.023 [−0.027, −0.018] |
| 68M − 150M v2 | −0.008 [−0.012, −0.003] | +0.006 [+0.003, +0.010] | −0.021 [−0.025, −0.017] |
| 150M v2 − 400M | −0.009 [−0.014, −0.005] | +0.019 [+0.015, +0.022] | −0.016 [−0.020, −0.012] |
| 400M − 1B | −0.025 [−0.030, −0.021] | +0.026 [+0.023, +0.030] | −0.050 [−0.054, −0.046] |
| 17M − Prumo | +0.021 [+0.016, +0.026] | −0.026 [−0.030, −0.022] | +0.056 [+0.051, +0.061] |

- **v1 → v2 does not move ECE in one direction.** At 150M, v2 is better calibrated than v1: ECE −0.031, with Brier
  also better. At 32M, v2 is worse: ECE +0.022, with Brier unchanged.
- The pooled-weighting differences, which agree in sign, are in `e5/out/calib.json` → `paired.pooled`.

### 13.6 Comparison with the board's calibration fields [board]

**Source.** `release/di-index-20261020.json` → `models[e].calibration` for 0.4. The v1 models come from
`release/di-index-v0.3.2-20261020.json`. The board's note (`suite.calibration_note`) says:
- "32 benchmarks with a per-field right or wrong answer (ToolRet, BRIGHT and RouterBench excluded), each weighted
  equally";
- "a 1-in-6 sample of requests";
- "confidence is the probability placed on the chosen option";
- "rows an entrant trained on count as wrong here too".

It uses 10 bins (`rel`). Every Crivo, Prumo and most neighbours have n = 34,411 fields from 31 benchmarks, so the
sample is fixed. **The note does not say whether the sampled rows are public or private.** Its wording ("rows an
entrant trained on"), the benchmark names and the close agreement below all fit a sample of the public suite. Call it
"the board's calibration (maintainer-computed, 1-in-6 request sample)", and do not call it a private-test number
unless the maintainers confirm it.

**Board vs our replica.** Each cell is board / ours. Ours is MAIN, bench-weighted, all fields, 10 bins:

| Model | Edition | Acc | Conf | ECE (10 bins) | Brier | over95 | Board n (benchmarks) |
|---|---|---|---|---|---|---|---|
| Crivo-17M | 0.4 | 0.505 / 0.514 | 0.586 / 0.590 | 0.080 / 0.076 | 0.577 / 0.574 | 0.020 / 0.021 | 34,411 (31) |
| Crivo-32M v1 | 0.3.2 | 0.521 / 0.513 | 0.608 / 0.615 | 0.087 / 0.102 | 0.570 / 0.573 | 0.011 / 0.014 | 34,411 (31) |
| Crivo-32M v2 | 0.4 | 0.527 / 0.534 | 0.660 / 0.659 | 0.133 / 0.125 | 0.578 / 0.574 | 0.028 / 0.025 | 34,411 (31) |
| Crivo-68M | 0.4 | 0.565 / 0.557 | 0.668 / 0.672 | 0.103 / 0.115 | 0.545 / 0.551 | 0.021 / 0.032 | 34,411 (31) |
| Crivo-150M v1 | 0.3.2 | 0.579 / 0.571 | 0.723 / 0.725 | 0.143 / 0.154 | 0.550 / 0.559 | 0.035 / 0.041 | 34,411 (31) |
| Crivo-150M v2 | 0.4 | 0.587 / 0.578 | 0.702 / 0.701 | 0.115 / 0.123 | 0.536 / 0.544 | 0.024 / 0.032 | 34,411 (31) |
| Crivo-400M | 0.4 | 0.612 / 0.593 | 0.733 / 0.726 | 0.121 / 0.132 | 0.503 / 0.526 | 0.032 / 0.040 | 34,411 (31) |
| Crivo-1B | 0.4 | 0.662 / 0.643 | 0.813 / 0.801 | 0.151 / 0.158 | 0.479 / 0.499 | 0.043 / 0.047 | 34,411 (31) |
| Prumo | 0.4 | 0.439 / 0.458 | 0.468 / 0.501 | 0.047 / 0.052 | 0.621 / 0.600 | 0.020 / 0.021 | 34,411 (31) |

- Agreement: |Δacc| ≤ 0.019, |Δconf| ≤ 0.033, |ΔECE| ≤ 0.015 and |ΔBrier| ≤ 0.023.
- Across the nine systems, board ECE vs our ECE: Spearman ρ = 0.98, Pearson r = 0.97.
- The remaining gap is expected. The board uses a 31-of-35 benchmark set we cannot identify exactly and a 1-in-6 row
  sample. Adding the five non-index per-field benchmarks (MMLU, ARC-E/C, SimpleBench, SGD) moves our numbers away from
  the board's (`board_replica["index35+notindex"]`).
- **Use the board numbers as the official reference and ours as the full-public-set view with CIs.** The two give the
  same picture.

**Neighbours on the board** [board] (0.4; all 45 entries of ≤ 1.1B with calibration are in
`e5/out/calib.json` → `board.neighbours`):

| Entry | Params | Public | Full 0.4 | Acc | Conf | ECE | Brier | over95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bekko System One v0 17M | 17M | 7.44 | 5.72 | 0.336 | 0.496 | 0.161 | 0.746 | 0.005 |
| Bekko System One v0 68M | 68M | 13.01 | 9.99 | 0.413 | 0.526 | 0.113 | 0.685 | 0.011 |
| Mola-68M | 68M | 27.27 | 14.96 | 0.545 | 0.660 | 0.119 | 0.559 | 0.011 |
| tasksource Decider nano v4 | 149M | 30.90 | 15.63 | 0.565 | 0.632 | 0.069 | 0.532 | 0.015 |
| Mola-150M | 149M | 31.34 | 16.15 | 0.576 | 0.657 | 0.080 | 0.522 | 0.010 |
| Dinah-1 | 150M | 34.32 | 16.08 | 0.552 | 0.770 | 0.218 | 0.632 | 0.065 |
| DM-JEPA 1.1 | 149M | 19.19 | 3.34 | 0.466 | 0.911 | 0.445 | 0.936 | 0.280 |
| DecisionTune 1.0 | 395M | 28.85 | 16.88 | 0.551 | 0.698 | 0.147 | 0.603 | 0.051 |
| Amber 0.4B | 396M | 49.74 | 25.41 | 0.651 | 0.725 | 0.075 | 0.433 | 0.017 |
| Bekko System One v0 400M | 400M | 22.45 | 16.00 | 0.489 | 0.600 | 0.111 | 0.631 | 0.019 |
| Laya | 421M | 6.19 | 7.21 | 0.377 | 0.517 | 0.140 | 0.721 | 0.012 |
| Sieve 0.8B | 590M | 32.26 | 21.23 | 0.579 | 0.666 | 0.088 | 0.541 | 0.027 |
| Decision 2.0 Kai 0.6B | 600M | 16.68 | 14.89 | 0.468 | 0.481 | 0.038 | 0.615 | 0.000 |
| jiwo 0.8B v2 | 752M | 29.58 | 24.48 | 0.558 | 0.626 | 0.068 | 0.538 | 0.002 |
| vLLM-SR d3-lite (0.8B) | 853M | 36.93 | 25.89 | 0.614 | 0.705 | 0.094 | 0.508 | 0.022 |
| OneJev1.6-0.8B | 853M | 48.76 | 29.66 | 0.704 | 0.738 | 0.035 | 0.383 | 0.008 |
| Kev 0.8B r15 | 870M | 16.78 | 15.03 | 0.505 | 0.519 | 0.040 | 0.604 | 0.001 |
| StartLux-Decision v1.1 0.8B | 873M | 53.79 | 29.02 | 0.725 | 0.727 | 0.015 | 0.362 | 0.005 |
| Jev 1.13 | — | 57.96 | 61.26 | 0.738 | 0.815 | 0.078 | 0.358 | 0.021 |
| Zev (rule-based) | — | 1.55 | 1.91 | 0.297 | 0.434 | 0.138 | 0.776 | 0.017 |

**Reading:**
- The Crivo ECEs (board 0.080–0.151) sit in the middle of the small-model field. Better-calibrated neighbours of
  similar or higher accuracy exist: Amber 0.075, Mola-150M 0.080, tasksource nano 0.069, OneJev1.6 0.035 and StartLux
  0.015. Dinah-1 (0.218), DecisionTune (0.147) and DM-JEPA (0.445) are worse.
- On Brier the Crivos are competitive for their size: 17M 0.577 against Mola-68M 0.559; 400M 0.503 against Amber
  0.433; 1B 0.479.
- Prumo's board ECE is 0.047; the board's other rule-based entry, Zev, has 0.138.

### 13.7 Prumo: claimed vs unclaimed fields [ours-public]

When no capability claims a question, Prumo returns the uniform distribution (`zero/prumo/README.md`:74). We mark a
field "claimed" when its output is non-uniform.

| Subset | Fields | Acc | Conf | ECE-15 | Brier | NLL |
|---|---:|---:|---:|---:|---:|---:|
| Claimed (pooled) | 47,743 (21.8% of MAIN; 38.7% bench-weighted) | 0.552 | 0.628 | 0.120 | 0.628 | 1.914 |
| Claimed (bench-weighted, 17 benchmarks) | 47,743 | 0.525 | 0.654 | 0.153 | 0.656 | 2.252 |
| Unclaimed (pooled) | 171,127 | 0.439 | 0.452 | 0.013 | 0.548 | 0.843 |

**Prumo's low overall ECE is mostly an artefact of uniform answers.** Where it actually decides, it is as
over-confident as a mid-size Crivo (ECE 0.12–0.15) and has a high NLL. Say this when citing its board ECE of 0.047.

### 13.8 Figure

`e5/out/reliability.pdf` (PNG preview `e5/out/reliability.png`):
- 3×3 panels, one per system, generated by `e5/plot_reliability.py` from `e5/out/calib.json` alone.
- Bars show our bench-weighted 15-bin reliability on the public suite; bins holding less than 0.2% of the weight are
  hidden.
- Orange markers show the board's 10-bin `rel` points; the grey diagonal is perfect calibration.
- Each panel is annotated with our ECE-15 [CI], the board ECE, acc and conf.
- Colours were checked with the dataviz palette validator (blue #2a78d6 and orange #eb6834, CVD ΔE 24.7).

### 13.9 Caveats (E5)

1. The 35 benchmarks have different native metrics (macro-F1, F1, nDCG, whole-case accuracy). Calibration here is
   always **per-field top-label correctness**, which is not the index metric. Per-benchmark field accuracy is in
   `e5/out/calib.json` → `by_bench`; the official per-benchmark skill is in §14.
2. Multi-field requests (ACOS, SATA-Bench, BFCL, PhishNChips, Home appliances) have correlated fields. The CIs
   resample whole requests, so this correlation is in the intervals. The point estimates still weigh a benchmark by its
   fields only inside the benchmark.
3. Seen-in-training sources: the v2 family trains on train splits of many suite sources; no test items (`facts.md`
   §firewall). Calibration on those benchmarks is in-distribution. There is no unseen/seen split of the public suite
   at the benchmark level, and it is not attempted here.
4. The v1 runs use their own engine versions (engine2 for 32M v1, engine for 150M v1). The calibration is that of the
   released runs, not of a re-run.
5. The CIs cover item sampling only, not training seeds. A1/A2 seeds will give seed variance for the 32M later.
6. All numbers are [ours-public], computed once with no tuning. The board's calibration is the official reference.

---

## 14. Public-suite error analysis (done 2026-10-20, CPU)

Script: `errors/errors.py` → `errors/results.json`. It reads the same per-question tables as §13: MAIN, 218,870
fields, 35 benchmarks.
- **norm** = chance-normalised field accuracy, the mean of (correct − c)/(1 − c) with c = |accepted|/k. 0 is chance
  and 1 is perfect.
- **Official skill** = each run's own kit `index.json` (`benchmarks[n].skill`), the index metric.
- Unless stated otherwise, benchmarks weigh equally.
- No two systems' outputs are ever combined into a score.

### 14.1 Where small Crivos fail and large ones do not (official per-benchmark skill) [ours-public]

The table is sorted by 1B − 17M. "Only 1B right / only 17M right" are field shares. Input tokens = the whole Crivo
encoder sequence of a question: instructions + rendered state + option texts, Ettin tokenizer.

| Benchmark | Area | Fields | k (median) | Input tokens (median) | 17M | 32M v1 | 32M v2 | 68M | 150M v1 | 150M v2 | 400M | 1B | Prumo | 1B − 17M | Only 1B right / only 17M right (fields) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Home appliance simulator | tools | 1,590 | 2 | 838 | 0.03 | 0.27 | 0.24 | 0.75 | 0.80 | 0.64 | 0.72 | 0.88 | 0.00 | +0.84 | 0.14 / 0.01 |
| HellaSwag | language | 10,042 | 4 | 232 | 0.07 | 0.11 | 0.11 | 0.19 | 0.49 | 0.39 | 0.64 | 0.79 | 0.00 | +0.72 | 0.57 / 0.03 |
| CLadder | knowledge | 5,000 | 2 | 128 | 0.00 | 0.08 | 0.03 | 0.08 | 0.61 | 0.32 | 0.47 | 0.65 | 0.00 | +0.65 | 0.41 / 0.09 |
| WinoGrande | knowledge | 1,267 | 2 | 42 | 0.04 | 0.03 | 0.01 | 0.06 | 0.22 | 0.09 | 0.26 | 0.51 | 0.01 | +0.47 | 0.35 / 0.12 |
| BFCL | tools | 4,768 | 2 | 890 | 0.48 | 0.54 | 0.69 | 0.76 | 0.79 | 0.68 | 0.83 | 0.91 | 0.67 | +0.43 | 0.14 / 0.01 |
| iSarcasmEval | language | 3,200 | 2 | 42 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | 0.42 | 0.04 | +0.42 | 0.08 / 0.07 |
| MuSR | knowledge | 752 | 3 | 1,017 | 0.01 | 0.00 | 0.04 | 0.20 | 0.27 | 0.21 | 0.28 | 0.41 | 0.00 | +0.40 | 0.38 / 0.13 |
| HoVer | retrieval | 4,000 | 2 | 390 | 0.35 | 0.58 | 0.57 | 0.63 | 0.69 | 0.60 | 0.64 | 0.73 | 0.39 | +0.38 | 0.25 / 0.06 |
| FinEntity | language | 2,129 | 3 | 124 | 0.52 | 0.52 | 0.65 | 0.78 | 0.77 | 0.79 | 0.81 | 0.88 | 0.01 | +0.36 | 0.26 / 0.03 |
| When2Call | tools | 3,652 | 4 | 700 | 0.38 | 0.47 | 0.50 | 0.54 | 0.62 | 0.53 | 0.61 | 0.74 | 0.70 | +0.36 | 0.33 / 0.06 |
| NLI4CT | language | 5,500 | 2 | 409 | 0.02 | 0.00 | 0.05 | 0.13 | 0.15 | 0.17 | 0.26 | 0.36 | 0.00 | +0.34 | 0.26 / 0.11 |
| ANLI | language | 3,200 | 3 | 123 | 0.00 | 0.08 | 0.02 | 0.04 | 0.19 | 0.08 | 0.16 | 0.33 | 0.00 | +0.33 | 0.35 / 0.13 |
| New Yorker | arts | 528 | 5 | 167 | 0.27 | 0.32 | 0.33 | 0.39 | 0.44 | 0.43 | 0.53 | 0.60 | 0.00 | +0.33 | 0.33 / 0.07 |
| API-Bank | tools | 508 | 53 | 6,216 | 0.49 | 0.08 | 0.56 | 0.60 | 0.29 | 0.70 | 0.69 | 0.80 | 0.45 | +0.31 | 0.32 / 0.02 |
| Amazon ESCI | retrieval | 5,000 | 4 | 345 | 0.07 | 0.21 | 0.20 | 0.26 | 0.30 | 0.21 | 0.26 | 0.37 | 0.06 | +0.30 | 0.17 / 0.09 |
| BPoMP | arts | 5,000 | 2 | 146 | 0.62 | 0.67 | 0.68 | 0.73 | 0.00 | 0.73 | 0.83 | 0.89 | 0.00 | +0.28 | 0.17 / 0.03 |
| VAST | language | 3,006 | 3 | 147 | 0.40 | 0.51 | 0.47 | 0.54 | 0.61 | 0.59 | 0.63 | 0.67 | 0.00 | +0.27 | 0.26 / 0.08 |
| ACOS | language | 91,092 | 2 | 120 | 0.08 | 0.25 | 0.15 | 0.20 | 0.39 | 0.20 | 0.30 | 0.33 | 0.01 | +0.25 | 0.02 / 0.02 |
| RAGTruth | language | 2,700 | 2 | 789 | 0.28 | 0.30 | 0.30 | 0.26 | 0.44 | 0.24 | 0.30 | 0.50 | 0.24 | +0.22 | 0.13 / 0.06 |
| ContractNLI | language | 2,091 | 3 | 1,983 | 0.59 | 0.59 | 0.58 | 0.71 | 0.75 | 0.71 | 0.74 | 0.77 | 0.00 | +0.18 | 0.15 / 0.04 |
| BBH | knowledge | 5,507 | 4 | 148 | 0.02 | 0.01 | 0.08 | 0.06 | 0.12 | 0.08 | 0.13 | 0.18 | 0.47 | +0.16 | 0.26 / 0.15 |
| SATA-Bench | knowledge | 15,517 | 2 | 510 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.01 | 0.15 | 0.02 | +0.15 | 0.34 / 0.13 |
| CLINC150+OOS | retrieval | 5,500 | 151 | 1,103 | 0.80 | 0.86 | 0.85 | 0.88 | 0.92 | 0.89 | 0.89 | 0.93 | 0.45 | +0.13 | 0.17 / 0.02 |
| BANKING77 | retrieval | 3,080 | 77 | 887 | 0.79 | 0.87 | 0.86 | 0.86 | 0.91 | 0.86 | 0.86 | 0.91 | 0.43 | +0.12 | 0.13 / 0.02 |
| MMLU-Pro | knowledge | 12,032 | 10 | 155 | 0.00 | 0.01 | 0.01 | 0.03 | 0.03 | 0.04 | 0.07 | 0.11 | 0.00 | +0.11 | 0.17 / 0.07 |
| Humicroedit | arts | 2,628 | 2 | 52 | 0.02 | 0.00 | 0.00 | 0.00 | 0.04 | 0.01 | 0.03 | 0.09 | 0.02 | +0.06 | 0.18 / 0.15 |
| Habermas Machine | arts | 1,676 | 4 | 855 | 0.08 | 0.12 | 0.09 | 0.12 | 0.10 | 0.12 | 0.10 | 0.14 | 0.25 | +0.06 | 0.21 / 0.17 |
| GSM8K | knowledge | 2,638 | 7 | 148 | 0.01 | 0.00 | 0.01 | 0.01 | 0.02 | 0.01 | 0.02 | 0.03 | 0.11 | +0.03 | 0.14 / 0.12 |
| GPQA Diamond | knowledge | 196 | 4 | 174 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.01 | 0.00 | 0.01 | 0.00 | +0.01 | 0.22 / 0.18 |
| ChessBench | knowledge | 5,000 | 32 | 830 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | +0.00 | 0.04 / 0.04 |
| HLE | knowledge | 501 | 5 | 207 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 | 0.11 / 0.11 |
| POP909-CL | arts | 2,000 | 129 | 4,488 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.52 | -0.00 | 0.01 / 0.02 |
| cfcolor | arts | 5,000 | 2 | 396 | 0.03 | 0.00 | 0.04 | 0.01 | 0.01 | 0.00 | 0.02 | 0.01 | 0.02 | -0.01 | 0.25 / 0.25 |
| CRUXEval | knowledge | 570 | 3 | 136 | 0.05 | 0.08 | 0.09 | 0.12 | 0.07 | 0.14 | 0.03 | 0.00 | 1.00 | -0.05 | 0.16 / 0.22 |
| PhishNChips | retrieval | 2,000 | 2 | 245 | 0.16 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | 0.05 | 0.04 | 0.00 | -0.12 | 0.04 / 0.11 |
| ToolRet (nDCG) | tools | — | — | — | 0.40 | 0.49 | 0.46 | 0.54 | 0.56 | 0.57 | 0.59 | 0.58 | 0.39 | +0.19 | — |
| BRIGHT (nDCG) | retrieval | — | — | — | 0.13 | 0.18 | 0.16 | 0.21 | 0.19 | 0.22 | 0.27 | 0.28 | 0.20 | +0.15 | — |

**Reading:**
- **Largest gains with size: commonsense, causal and narrative reasoning and multi-step tool use.**
  - Home-appliance simulator: +0.84. The 17M is at 0.03, and the step comes at 68M (0.75).
  - HellaSwag +0.72, CLadder +0.65, WinoGrande +0.47, MuSR +0.40 and ANLI +0.33. All are at most 0.20 up to 68M
    and climb at 150M–1B.
  - BFCL +0.43, iSarcasmEval +0.42 (zero up to 150M), NLI4CT +0.34.
- **Flat at chance for the whole family (17M–1B): computation and expert knowledge.**
  - Skill ≤ 0.03 on ChessBench, HLE, GPQA Diamond, GSM8K, POP909-CL and cfcolor. Skill ≤ 0.14 on CRUXEval and
    Humicroedit, and ≤ 0.11 on MMLU-Pro.
  - Where Prumo has an exact solver it is far ahead: CRUXEval 1.00, POP909 0.52, BBH 0.47, ChessBench 0.33, GSM8K
    0.11.
- **Count of near-chance benchmarks** (official skill < 0.05, of 35):

  | 17M | 32M v1 | 32M v2 | 68M | 150M v1 | 150M v2 | 400M | 1B | Prumo |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 17 | 15 | 15 | 12 | 12 | 11 | 9 | 8 | 21 |

  The 1B's eight are CRUXEval, ChessBench, GPQA, GSM8K, HLE, POP909, PhishNChips and cfcolor.
- **Already high at 17M:** CLINC150 (0.80 → 0.93) and BANKING77 (0.79 → 0.91); then BPoMP (0.62), ContractNLI
  (0.59) and FinEntity (0.52).
- **Oddities to keep in mind:**
  - 150M v1 scores 0.00 on BPoMP; v2 scores 0.73.
  - 32M v1 scores 0.08 on API-Bank against 0.56 for v2.
  - PhishNChips: 17M 0.16, the others ≤ 0.05. Larger models call the phishing emails "legitimate"; see §14.5.

### 14.2 Question type and label count [ours-public]

Each cell is field accuracy / norm (bench-weighted within each row).

| Bucket | Fields | Benchmarks | 17M | 32M v1 | 32M v2 | 68M | 150M v1 | 150M v2 | 400M | 1B | Prumo |
|---|---:|---:|---|---|---|---|---|---|---|---|---|
| noul (RAGTruth) | 2,700 | 1 | 0.77 / 0.53 | 0.80 / 0.59 | 0.78 / 0.57 | 0.79 / 0.58 | 0.81 / 0.63 | 0.78 / 0.56 | 0.75 / 0.50 | 0.83 / 0.67 | 0.66 / 0.32 |
| choice, k = 2 | 149,019 | 18 | 0.63 / 0.25 | 0.63 / 0.25 | 0.65 / 0.29 | 0.66 / 0.31 | 0.66 / 0.31 | 0.67 / 0.33 | 0.68 / 0.36 | 0.73 / 0.45 | 0.59 / 0.17 |
| choice, k > 2 | 67,151 | 22 | 0.40 / 0.25 | 0.42 / 0.27 | 0.44 / 0.30 | 0.48 / 0.34 | 0.50 / 0.38 | 0.49 / 0.37 | 0.51 / 0.40 | 0.56 / 0.45 | 0.39 / 0.22 |
| k = 2 (all) | 151,719 | 19 | 0.64 / 0.27 | 0.64 / 0.27 | 0.66 / 0.30 | 0.67 / 0.32 | 0.67 / 0.33 | 0.68 / 0.35 | 0.69 / 0.36 | 0.74 / 0.46 | 0.59 / 0.18 |
| k = 3–4 | 34,978 | 16 | 0.43 / 0.19 | 0.47 / 0.25 | 0.48 / 0.27 | 0.51 / 0.31 | 0.55 / 0.37 | 0.53 / 0.33 | 0.55 / 0.36 | 0.59 / 0.42 | 0.44 / 0.20 |
| k = 5–9 | 4,835 | 9 | 0.28 / 0.12 | 0.32 / 0.17 | 0.30 / 0.14 | 0.34 / 0.19 | 0.35 / 0.20 | 0.34 / 0.19 | 0.36 / 0.22 | 0.37 / 0.22 | 0.38 / 0.25 |
| k = 10–31 | 13,727 | 6 | 0.07 / −0.02 | 0.14 / 0.06 | 0.07 / −0.02 | 0.11 / 0.02 | 0.08 / −0.01 | 0.08 / −0.01 | 0.08 / −0.01 | 0.11 / 0.03 | 0.30 / 0.23 |
| k ≥ 32 | 13,611 | 5 | 0.42 / 0.41 | 0.36 / 0.35 | 0.45 / 0.44 | 0.47 / 0.46 | 0.43 / 0.42 | 0.50 / 0.49 | 0.49 / 0.49 | 0.53 / 0.53 | 0.45 / 0.44 |

- **The pooled k buckets are confounded with task.**
  - k = 10–31 is MMLU-Pro, GSM8K-10, ChessBench and similar: computation, at chance.
  - k ≥ 32 is CLINC150, BANKING77, API-Bank, POP909 and ChessBench: intent classification, which is easy.
- After chance correction, binary and multi-option questions are equally hard on average (norm 0.25 vs 0.25 at 17M;
  0.45 vs 0.45 at 1B).

**Within-benchmark label-count effect** (norm, low-k half → high-k half; split at each benchmark's median k):

| Benchmark | k | Fields | 17M | 68M | 150M v2 | 1B | Prumo |
|---|---|---|---|---|---|---|---|
| Home appliance simulator | 2 → 5 | 1,266 / 324 | 0.82 → **0.50** | 0.97 → 0.94 | 0.95 → 0.89 | 0.99 → 0.95 | 0.02 → −0.01 |
| GSM8K (same problems, 4- vs 10-choice tracks) | 4 → 10 | 1,319 / 1,319 | 0.02 → 0.00 | −0.01 → 0.01 | 0.01 → 0.01 | 0.04 → 0.02 | 0.14 → 0.08 |
| MuSR | 3 → 4 | 616 / 136 | 0.01 → 0.00 | 0.21 → 0.15 | 0.22 → 0.14 | 0.40 → 0.35 | −0.03 → 0.02 |
| BBH | 2 → 6 | 3,371 / 2,136 | 0.05 → −0.00 | 0.05 → 0.07 | 0.11 → 0.06 | 0.14 → 0.21 | 0.46 → 0.49 |
| CRUXEval | 2 → 5 | 395 / 175 | 0.01 → 0.10 | 0.12 → 0.13 | 0.15 → 0.13 | −0.05 → −0.05 | 0.99 → 1.00 |
| ChessBench | 24 → 38 | 2,658 / 2,342 | −0.01 → −0.01 | −0.00 → −0.01 | 0.00 → −0.00 | −0.01 → −0.00 | 0.33 → 0.34 |
| HLE | 5 → 8 | 278 / 223 | −0.06 → 0.02 | −0.02 → −0.00 | −0.00 → 0.01 | −0.02 → −0.01 | −0.04 → −0.02 |
| Habermas Machine | 4 → 5 | 1,497 / 179 | 0.09 → 0.09 | 0.13 → 0.17 | 0.13 → 0.11 | 0.15 → 0.14 | 0.26 → 0.22 |

- More options cost chance-normalised accuracy mainly where the model has real skill, and mostly for the smallest
  model. Home appliances: 17M 0.82 → 0.50, against the 1B's 0.99 → 0.95. MuSR shows the same pattern at 68M–1B.
- GSM8K, where the same problems come with 4 and 10 options, shows only that raw accuracy falls with chance (0.26 →
  0.10 at 17M); the model is at chance in both tracks.
- Elsewhere the half-splits differ in item type as well as k, so read them as descriptive only.

### 14.3 Input length [ours-public]

Fixed length buckets (input tokens, bench-weighted; acc / norm):

| Input tokens | Fields | Benchmarks | 17M | 32M v1 | 32M v2 | 68M | 150M v1 | 150M v2 | 400M | 1B | Prumo |
|---|---:|---:|---|---|---|---|---|---|---|---|---|
| < 128 | 96,748 | 21 | 0.50 / 0.22 | 0.54 / 0.27 | 0.56 / 0.30 | 0.57 / 0.32 | 0.59 / 0.35 | 0.58 / 0.36 | 0.63 / 0.44 | 0.67 / 0.50 | 0.48 / 0.17 |
| 128–511 | 79,727 | 27 | 0.52 / 0.27 | 0.52 / 0.27 | 0.54 / 0.31 | 0.56 / 0.33 | 0.57 / 0.35 | 0.58 / 0.36 | 0.59 / 0.38 | 0.64 / 0.47 | 0.48 / 0.19 |
| 512–2,047 | 37,741 | 20 | 0.46 / 0.22 | 0.55 / 0.32 | 0.56 / 0.37 | 0.58 / 0.37 | 0.52 / 0.30 | 0.51 / 0.28 | 0.53 / 0.32 | 0.57 / 0.37 | 0.52 / 0.30 |
| 2,048–8,191 | 4,652 | 11 | 0.60 / 0.46 | 0.45 / 0.27 | 0.49 / 0.30 | 0.50 / 0.30 | 0.51 / 0.34 | 0.62 / 0.46 | 0.53 / 0.34 | 0.61 / 0.48 | 0.46 / 0.23 |

- Only 2 MAIN fields exceed 8,192 tokens (HLE, POP909). The long tail of the suite is in ToolRet, which is not in
  MAIN.
- State-only token buckets are in `results.json` → `by_state_len`. Many benchmarks put the item text in
  `instructions` and leave `state` empty (HellaSwag, CLadder, WinoGrande, ANLI, CLINC, BANKING), so state length alone
  is a poor proxy.
- The buckets mix benchmarks, so the controlled estimate is the within-benchmark split below.

**Within-benchmark median split.** For each of 16 benchmarks with enough spread (q75 ≥ 1.5·q25 and at least 50
fields per half), the difference is norm(long half) − norm(short half). The table reports the benchmark-equal mean,
with a 95% cluster-bootstrap CI (B = 500):

| | 17M | 32M v1 | 32M v2 | 68M | 150M v1 | 150M v2 | 400M | 1B | Prumo |
|---|---|---|---|---|---|---|---|---|---|
| Δnorm (long − short) | −0.022 [−0.040, −0.005] | +0.012 [−0.005, +0.027] | −0.009 [−0.025, +0.006] | −0.020 [−0.037, −0.005] | −0.043 [−0.057, −0.027] | −0.024 [−0.039, −0.008] | −0.038 [−0.052, −0.021] | −0.017 [−0.034, −0.000] | +0.029 [+0.013, +0.047] |

- The 16 benchmarks: BFCL, ContractNLI, GPQA, HellaSwag, MuSR, SATA-Bench, Amazon ESCI, iSarcasmEval, VAST, NLI4CT,
  HLE, MMLU-Pro, BBH, RAGTruth, HoVer and When2Call. Per-benchmark values are in `results.json` → `len_within`.
- **Longer inputs cost little:** about 2–4 points of chance-normalised accuracy within a benchmark. There is no
  cliff, and Crivo reads up to 32,768 tokens untruncated.
- **The size gap does not widen with length:** (1B − 17M) long − short = +0.005 [−0.017, +0.028].
- Largest drops: RAGTruth (17M −0.19, 1B −0.14; longer source passages) and When2Call for the 17M (−0.10). MuSR at
  the 1B drops −0.31: longer stories with more distractors.
- iSarcasmEval's split mixes tracks (A-En, A-Ar, C-En, C-Ar), so its −0.32 is a track effect, not a length effect.

### 14.4 Automatic error-category proxies [ours-public]

- **Confident errors.** Wrong answers with conf ≥ 0.9, as a share of all errors (bench-weighted):

  | 17M | 32M v1 | 32M v2 | 68M | 150M v1 | 150M v2 | 400M | 1B | Prumo |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 5.8% | 5.3% | 8.8% | 10.0% | 16.1% | 10.7% | 14.6% | **25.8%** | 4.9% |

  - As a share of all fields, these are 2.8% (17M) and 9.2% (1B).
  - This is the error-side view of §13: larger models make fewer errors, but more of the remaining ones are confident.
- **Label collapse on binary benchmarks.** The modal predicted option's share minus that option's gold share, averaged
  over the 18 binary benchmarks with at least 100 fields:

  | 17M | 32M v1 | 32M v2 | 68M | 150M v1 | 150M v2 | 400M | 1B | Prumo |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 0.090 | 0.126 | 0.119 | 0.130 | 0.098 | 0.119 | 0.098 | 0.075 | 0.322 |

  Benchmarks where a model puts at least 90% of answers on one option while gold has less than 80%:
  - 68M: SATA-Bench, Humicroedit, PhishNChips.
  - 1B: Humicroedit, PhishNChips.
  - 150M v1 / 150M v2 / 400M: PhishNChips.
  - Prumo: nine benchmarks; it falls back to a default when no capability claims the question.
- **Position bias** on choice questions with k ≥ 3 (67,151 fields):
  - Gold is first in 20.3% and last in 19.4% (uniform first-rate 19.7%).
  - Crivos pick the first option in 20–24% and the last in 14–18%.
  - So there is a mild preference for the first option and a mild avoidance of the last. Prumo picks first in 26.7%.
  - E4 (option permutations) measures this causally.

### 14.5 Hand-coded sample of Crivo-68M errors (A5)

`errors/sample_errors.py` drew 2 random errors per MAIN benchmark (seed 0): 70 errors. I read them, truncated, on
VM2 and coded each into one category. The coding is in `errors/sample68m_coded.tsv`, with run ids and keys only and
no benchmark text. The sample is stratified by benchmark, so categories partly mirror benchmark identity; read the
counts as a profile, not as population shares.

| Category | n / 70 | Median 68M conf | 1B right | 17M also wrong | Prumo right | Benchmarks (sampled) |
|---|---:|---:|---:|---:|---:|---|
| A. Computation not carried out (arithmetic, code tracing, chess, chord inference, causal calculus, logic puzzles) | 15 | 0.32 | 4 | 13 | 10 | GSM8K, CRUXEval, BBH, ChessBench, CLadder, MMLU-Pro, POP909, HLE (1) |
| B. Expert knowledge missing | 3 | 0.30 | 1 | 2 | 1 | GPQA, HLE (1) |
| C. Near-label confusion (adjacent intents, neutral vs polar, not-mentioned vs entailed) | 14 | 0.73 | 5 | 14 | 4 | BANKING77, CLINC150, ESCI, VAST, FinEntity, ANLI, ContractNLI |
| D. Evidence verification misses a detail in the evidence | 6 | 0.61 | 2 | 5 | 1 | HoVer, NLI4CT, RAGTruth |
| E. Tool/action selection: needed tool rejected or wrong API chosen | 8 | **0.94** | 7 | 8 | 4 | BFCL, API-Bank, When2Call, Home appliances |
| F. Taste and pragmatics (humour, sarcasm, captions, palettes, consensus, originality) | 12 | 0.69 | 10 | 8 | 6 | Humicroedit, iSarcasmEval, New Yorker, cfcolor, Habermas, BPoMP |
| G. Commonsense and narrative inference | 6 | 0.54 | 3 | 4 | 4 | WinoGrande, HellaSwag, MuSR |
| H. Multi-label over-inclusion (says "present" for an absent label) | 4 | 0.77 | 3 | 1 | 2 | ACOS, SATA-Bench |
| I. Trusting surface cues (friendly phishing email judged legitimate) | 2 | 0.97 | 0 | 1 | 2 | PhishNChips |

**Short paraphrased examples.** These are anonymised and paraphrased, with no benchmark text copied.
- **E, confident tool error.**
  - A user asks to book two smoking rooms in a hotel in a large European city. A hotel-search tool is among the
    candidates and should be invoked. The 68M says "do not invoke" at p = 0.96; the 17M and 1B also miss it.
  - In an API dialogue whose next step requires obtaining a user token, the 68M picks a password-reset API among 53
    options at p ≈ 1.00. The 1B is right.
- **C, near-label confusion.**
  - A banking customer asks whether now is a good time to exchange money. Gold is the exchange-rate intent; the
    model picks "exchange via the app" (p = 0.35).
  - A personal-fitness question that is out of scope for an assistant's intent list is mapped to "smart home"
    (p = 0.41).
  - An NDA hypothesis that the contract explicitly entails is labelled "not mentioned" (p = 0.99).
- **A, computation not carried out (low confidence).**
  - A word problem that must be solved backwards (sold a fraction, then a few more, then half the rest) gets a
    distractor answer at p = 0.29.
  - A short Python function that counts occurrences of a character gets an off-by-one output (p = 0.43).
  - Prumo's exact solvers answer 10 of these 15 correctly.

**Reading:** the 68M's errors fall into two groups.
- **Confident errors** come from tool selection and phishing (median conf 0.94–0.97) and from near-label confusion
  (0.73). These are where calibration hurts. The 1B fixes most tool and taste errors.
- **Low-confidence errors** come from computation and expert knowledge (median 0.30–0.32). No Crivo size fixes them
  (1B right on 5 of 18), and the rule engine's exact solvers do.

### 14.6 Error overlap across sizes [ours-public]

Pooled over MAIN fields:

| Pair (small \| large) | Small errors | Large errors | Large's errors also made by small | Small's errors fixed by large | Same predicted option |
|---|---:|---:|---:|---:|---:|
| 17M \| 1B | 0.309 | 0.213 | 0.73 | 0.50 | 0.70 |
| 17M \| 150M v2 | 0.309 | 0.271 | 0.70 | 0.39 | 0.71 |
| 68M \| 1B | 0.298 | 0.213 | 0.75 | 0.46 | 0.74 |
| 150M v2 \| 1B | 0.271 | 0.213 | 0.77 | 0.39 | 0.78 |
| 32M v1 \| 32M v2 | 0.312 | 0.310 | 0.79 | 0.21 | 0.79 |
| 150M v1 \| 150M v2 | 0.273 | 0.271 | 0.75 | 0.26 | 0.80 |

- Errors are largely nested: about three quarters of a larger model's errors are also made by the smaller one.
- The 1B fixes half of the 17M's errors.
- v1 and v2 of the same size disagree on 20–21% of fields with nearly identical error rates. The recipe change
  reshuffles errors more than it removes them, which matches §4's paired per-benchmark result.

### 14.7 Prumo complementarity (descriptive only; no combined score)

On the 47,743 MAIN fields Prumo claims (21.8%, in 17 benchmarks), Prumo's accuracy is 0.552 (norm 0.446).

| Crivo | Acc on Prumo-claimed | Acc on unclaimed | Both right | Prumo right, Crivo wrong | Crivo right, Prumo wrong | Both wrong | P(Crivo wrong \| Prumo right) | Crivo error rate (all MAIN) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 17M | 0.486 | 0.748 | 0.354 | 0.197 | 0.132 | 0.317 | 0.358 | 0.309 |
| 32M v1 | 0.515 | 0.737 | 0.360 | 0.191 | 0.154 | 0.294 | 0.347 | 0.312 |
| 32M v2 | 0.528 | 0.735 | 0.375 | 0.177 | 0.153 | 0.295 | 0.320 | 0.310 |
| 68M | 0.540 | 0.748 | 0.376 | 0.176 | 0.164 | 0.284 | 0.319 | 0.298 |
| 150M v1 | 0.560 | 0.774 | 0.387 | 0.165 | 0.173 | 0.276 | 0.299 | 0.273 |
| 150M v2 | 0.542 | 0.781 | 0.379 | 0.173 | 0.163 | 0.285 | 0.313 | 0.271 |
| 400M | 0.555 | 0.794 | 0.385 | 0.167 | 0.171 | 0.278 | 0.303 | 0.258 |
| 1B | 0.590 | 0.842 | 0.399 | 0.152 | 0.191 | 0.257 | 0.276 | 0.213 |

(The four middle columns are shares of the claimed fields.)

**Per benchmark** (claimed fields; share of fields where Prumo is right and the Crivo wrong):

| Benchmark | Claimed (share) | Prumo acc | 17M acc | 1B acc | Prumo-only vs 17M | Prumo-only vs 1B |
|---|---|---:|---:|---:|---:|---:|
| BBH | 2,570 (47%) | 0.993 | 0.332 | 0.417 | 0.664 | 0.580 |
| CRUXEval | 569 (100%) | 1.000 | 0.402 | 0.336 | 0.598 | 0.664 |
| POP909-CL | 2,000 (100%) | 0.530 | 0.018 | 0.015 | 0.516 | 0.521 |
| ChessBench | 4,961 (99%) | 0.386 | 0.072 | 0.068 | 0.341 | 0.343 |
| When2Call | 3,652 (100%) | 0.775 | 0.536 | 0.806 | 0.344 | 0.114 |
| Habermas Machine | 1,676 (100%) | 0.480 | 0.364 | 0.406 | 0.260 | 0.248 |
| GSM8K | 1,854 (70%) | 0.286 | 0.173 | 0.196 | 0.231 | 0.210 |
| HoVer | 4,000 (100%) | 0.697 | 0.674 | 0.867 | 0.155 | 0.051 |
| API-Bank | 502 (99%) | 0.468 | 0.496 | 0.807 | 0.159 | 0.016 |
| BFCL | 4,768 (100%) | 0.878 | 0.838 | 0.971 | 0.127 | 0.017 |
| RAGTruth | 2,700 (100%) | 0.659 | 0.766 | 0.835 | 0.099 | 0.071 |
| Amazon ESCI | 4,328 (87%) | 0.620 | 0.624 | 0.713 | 0.053 | 0.095 |
| MMLU-Pro | 5,569 (46%) | 0.092 | 0.088 | 0.169 | 0.066 | 0.070 |
| CLINC150+OOS | 5,500 (100%) | 0.462 | 0.765 | 0.907 | 0.053 | 0.021 |
| BANKING77 | 3,048 (99%) | 0.442 | 0.798 | 0.913 | 0.019 | 0.010 |

**Reading:**
- **Yes, Prumo-claimed questions are disproportionately ones the Crivos get wrong.** Crivo accuracy on claimed fields
  is 0.49–0.59, against 0.74–0.84 on unclaimed fields, and P(Crivo wrong | Prumo right) exceeds the Crivo's overall
  error rate at every size.
- The complementarity sits in Prumo's **exact solvers** (BBH, CRUXEval, POP909, ChessBench, GSM8K). On 21–66% of
  the claimed fields there, Prumo is right and even the 1B is wrong.
- On lexical and heuristic claims (BFCL, API-Bank, HoVer, CLINC, BANKING), the overlap shrinks with Crivo size. Against
  the 1B, Prumo-only-right is ≤ 0.051.
- This fits the board transfer pattern: exact solvers do not transfer to private items, lexical rules do (`facts.md`
  §Prumo). These are descriptive counts. Per the rules, no combination is scored here, and none goes into the paper
  as a number.

### 14.8 Caveats (error analysis)

1. Official skill uses each benchmark's own metric; field accuracy and norm are per-field. They can disagree. For
   example, GSM8K field accuracy is 0.17–0.20 while official skill is 0.01–0.03, because GSM8K averages two tracks
   with different chance.
2. Per-type, per-k and per-length buckets are observational and confounded with benchmark identity. Only the
   within-benchmark analyses (GSM8K tracks, median splits) control for it, and only partly: the halves of a benchmark
   can differ in item type.
3. The hand coding is one reader (the agent) on 70 items stratified by benchmark. It was not double-coded.
4. "Claimed" for Prumo = non-uniform output. A claim can still return a near-uniform distribution, so a few such
   fields may count as unclaimed.
5. Licences: no benchmark text is stored in the repo. The examples above are paraphrases. `sample_errors.py`
   prints truncated text to the VM only, for reading.

---

## 15. E5 / error-analysis files and commands

**Files on the Mac** (all under `pareto/paper/exp/`):

| Path | What |
|---|---|
| `e5/extract.py` | Per-question tables from the suite and the runs; writes `questions.parquet` and `preds_<model>.npz` |
| `e5/calib.py` | All E5 metrics, bootstrap, board comparison → `e5/out/calib.json` |
| `e5/plot_reliability.py` | Figure → `e5/out/reliability.pdf` (+ `.png`) |
| `e5/run_vm.sh` | The exact VM driver, in the order it ran |
| `e5/tab/` (59 MB) | `questions.parquet` (238,961 fields in 127,873 requests, MAIN = `in_index`) and the 9 `preds_*.npz`, copied back from VM2 |
| `e5/out/calib.json` | Every E5 number: `overall`, `by_qtype`, `by_area`, `by_bench`, `reliability`, `ci`, `paired`, `prumo_claimed`, `board_replica`, `board` |
| `e5/out/rescore/<model>/` | Official kit re-scores (index check) |
| `errors/errors.py`, `errors/results.json` | §14 analysis and every number in it |
| `errors/sample_errors.py`, `errors/sample68m_coded.tsv` | A5 sample (ids + categories) |

**Commands:**

```bash
# Mac → VM2 (rsync only): the nine runs and the release JSONs
R=release/decision-index-results/runs
for m in 17m:crivo-17m 32v2:crivo-32m-v2 68m:crivo-68m 150v2:crivo-150m-v2 400m:crivo-400m 1b:crivo-1b; do
  rsync -a $R/${m#*:}/ <cpu-vm2>:e5/runs/c${m%%:*}/; done
rsync -a $R/prumo/ <cpu-vm2>:e5/runs/prumo/
rsync -a --exclude '*.log' results/soup23-suite/ <cpu-vm2>:e5/runs/c150v1/
rsync -a results32/crivo32-suite/ <cpu-vm2>:e5/runs/c32v1/
rsync -a paper/exp/e5 paper/exp/errors <cpu-vm2>:pareto/paper/exp/
# VM2 (suite at ~/pareto/kit/suite-0.3, same md5 as the Mac; kit installed -e in /root/venv)
bash ~/pareto/paper/exp/e5/run_vm.sh
# VM2 → Mac
rsync -a <cpu-vm2>:e5/calib.json paper/exp/e5/out/
rsync -a <cpu-vm2>:e5/errors.json paper/exp/errors/results.json
rsync -a <cpu-vm2>:e5/tab/ paper/exp/e5/tab/
# Figure (Mac, light)
cd paper/exp/e5 && uv run --python 3.12 --with matplotlib --with numpy python plot_reliability.py out/calib.json out/reliability.pdf
```

**Re-running without VM2.** Re-running needs only the nine runs, `kit/suite-0.3` and the crivo-17m `tokenizer.json`.
- `extract.py` takes about 1 minute on 16 cores.
- `calib.py` and `errors.py` take about 2 minutes each, and need only `e5/tab/` and the runs' `index.json`.
- Do not run them on the 8 GB Mac while other jobs are running.
- All steps are deterministic (bootstrap seed 0).

### Known issue fixed during GPU-1 (2026-10-20, main session)
- `btzsc/crivo-150m-v1` first ran with every request in error: the v1 revision's `DecisionModel.forward()` (Crivo-150M
  @7a6c745) has no `head_fp32` argument (v1 predates the fp32 head). `common/engines.py` now passes `head_fp32=True`
  only when the loaded model's forward accepts it (inspected per model), so v1 runs exactly as its own engine does.
  Crivo-32M v1 (@bbf3f19) was unaffected (0 errors). The bad output was kept as
  `~/results/btzsc/crivo-150m-v1.errors.jsonl.bak` on the GPU box, and the job was rerun in tmux `fix150`
  (log `~/results/logs/btzsc_crivo-150m-v1.rerun.log`). `s1/crivo-150m-v1` had not run yet and uses the fixed code.
- Lambda Stack image: Python 3.10 and a CUDA 12.8 driver. `setup_gpu.sh` now builds a Python 3.12 venv with uv and
  installs torch from the cu128 index (the PyPI wheel is cu130 and fails with "driver too old").
- NLI and reranker baselines: when one option alone exceeds the model's `max_length`, the tokenizer raised
  "Truncation error" and crashed the CPU latency run for DeBERTa-zs (VM1, `~/exp/e6cpu/log.txt`). `common/engines.py`
  now raises `Unsupported` for such a question: it counts as unanswered/wrong (the kit rule) instead of crashing. BTZSC
  runs had 0 such cases. The DeBERTa CPU latency was rerun with the fix (`~/exp/e6cpu/nli-deberta.t16.json`), and the
  GPU box got the same file before its E4/E6 baseline jobs started.
- GPU memory: with 3 lanes sharing the A100, `s1/letter-qwen3-0.6b-rot4` (long prompts × 4 rotations) hit CUDA OOM
  on some requests. They are recorded as `status: error`, and `run_requests.py` retries errors on rerun. The scorers
  key results by id, so the last line wins. Plan: after ALL DONE, rerun every job whose output has errors, alone on
  the GPU, then delete those jobs' `.score.json` and rerun `score_all.sh`. `run_lanes.sh` is not edited while it
  runs, since bash reads scripts incrementally.
- Retry run (2026-10-21 ~04:10 UTC): when lane0 finished, `~/retry.sh` (tmux `retry`, log `~/results/logs/retry.log`)
  re-ran the OOM failures next to lane1, with smaller batches:
  - `e4/crivo-1b` at batch 32 (orig 2,560 and perm1 1,792 errors; the other variants were clean);
  - `s1/letter-qwen3-0.6b-rot4` at batch 4 (208 errors);
  - `s1/reranker-32m-raw` (2 errors).

  STATUS gets "retry start/done" lines.
- Parallel lane1 (2026-10-21 04:35 UTC): with only lane1 left, the GPU sat at ~38%. `s1/nli-bart-large-mnli` and
  `e4/reranker-32m-raw` were moved to `~/par.sh` (tmux `par`; STATUS lines "par …"; `~/results/PAR_DONE` when done).
  Their `.done` markers were created first, so lane1 skips them. If run_lanes.sh starts E6 before PAR_DONE exists,
  the E6 outputs are deleted and `gpu/e6.sh` is rerun alone afterwards (latency must be measured with the GPU idle).
- E6 contention (2026-10-21 ~05:50 UTC): run_lanes.sh started E6 while `par` (e4/reranker) was still running.
  Pausing par with SIGSTOP kept its 32 GB of GPU memory, and the E6 runs for crivo-1b and the reranker failed with OOM.
  par was killed. `~/finish.sh` (tmux `finish`) waits for ALL DONE and then:
  1. moves the contaminated E6 files (crivo 17m–400m) to `~/results/e6/contaminated/`;
  2. reruns `gpu/e6.sh` with the GPU alone (it remeasures every missing file; DeBERTa/Qwen/per-option measured after
     the kill are kept);
  3. finishes `e4/reranker-32m-raw` (resumable);
  4. rescores and writes `~/results/FINISH_DONE`.

  Only E6 numbers from the redo (log `logs/e6_redo.log`) may be reported.

---

## 16. GPU-1 statistics (E7)

First pass 2026-10-21 ~06:00 (score-file CIs, E4, E6). **Item-level E1/E2 pass 2026-10-21 (later the same day)**, once
the request files with gold were copied from the GPU-2 box (§17). Everything ran on the Mac: CPU, one Python process
at a time, every JSONL streamed; peak RSS ≤ 0.53 GB; no model, no GPU. Nothing uploaded. Labels: all numbers are
[derived] from our GPU-1 runs (`results_gpu/`) and the VM1 CPU latency runs (`results_cpu/e6cpu/`).

### 16.1 Status

- **E1/E2 per-item paired bootstrap: done.** Inputs:
  - `stats/gpu1/req/s1_all.jsonl` (md5 `3159ca24f95da9a39d35c6588a1a587a`) and `stats/gpu1/req/btzsc_full.jsonl`
    (md5 `9bbbabf3c0758a0757efafbd5f07858a`). Both are equal to the md5s recorded in §17. They were built
    deterministically by `gpu/setup_gpu.sh` on GPU-2, the same way as on GPU-1.
  - `stats/gpu1/ext/s1arena_data/` (HF S1-Arena snapshot), `stats/gpu1/ext/s1arena_code/` (the scorer) and
    `stats/gpu1/ext/btzsc_data/`.
- **The official scorers were reproduced exactly.**
  - `stats/rescore_check.py` re-scored Crivo-17M through the official code paths. For S1 that is
    `s1arena.metrics.score` per task on `schema.Item` objects, as `s1arena/adapter.py score` does. For BTZSC it is
    sklearn macro-F1 via `btzsc/adapter.py::macro_f1`.
  - Result: all 209 tasks × {n, accuracy, accuracy_se, Brier, ECE, unanswered rate} and all 22 datasets ×
    {macro-F1, accuracy} are **bit-identical** to `results_gpu/*/crivo-17m.score.json` (max |diff| 0.0).
  - The extraction step then re-checked **every** system against its `.score.json`:
    - S1 (13 systems): max |diff| 1.1e-16 (Brier float order), all other fields 0;
    - BTZSC (13 systems): 0.
- **E1/E2 calibration on our own definitions: done** (ECE-10 equal-mass, Brier; §16.3). The result files store
  probabilities for every answered question.
- **Gold subtlety.** For S1 the gold comes from the parquet `gold` column, not from the request files' `expected`.
  - 167 of the 72,817 questions have only a soft label (`expected` = null). The scorer's hard label for them is the
    argmax of the distribution.
  - 18,387 questions have a distribution, which S1's Brier uses as its target.
  - Where `expected` is set, it equals the parquet label (0 mismatches).
- **Bugs in the earlier untested item-mode code, fixed before running:**
  - `noul` options were ordered ["false", "true"]; the scorer uses ("true", "false"). With ties at p = 0.5 (e.g. the
    uniform system) the argmax then picked the other label.
  - soft-only golds were read as `expected` = null.
  - every request question was scored, not only those with gold.

  The new code calls the scorer's own `metrics._answer` and `schema.Gold`, so these cannot recur.
- E4 (complete at item level) and E6 (summaries only, no CIs) are unchanged from the first pass.

### 16.2 Files

| Path | What |
|---|---|
| `stats/paired_bootstrap.py` | All E7 GPU-1 statistics; subcommands `integrity`, `extract`, `s1`, `btzsc`, `e4`, `e6`, `md`, `headline` (docstring = method) |
| `stats/rescore_check.py` | Official-path re-score of one system (S1 per domain, BTZSC streamed) vs its `.score.json` |
| `stats/e4_rows.py` | Streamed rebuild of the E4 sample (same selection as `e4/perturb.py sample`) |
| `stats/gpu1/req/{s1_all,btzsc_full}.jsonl` | E2/E1 request files with gold (167 MB / 427 MB; from GPU-2) |
| `stats/gpu1/req/e4_orig.jsonl` | The rebuilt E4 sample (2,833 requests; md5 `4643cfec…`) |
| `stats/gpu1/ext/` | S1-Arena data + scorer code, BTZSC data (from GPU-2; read-only inputs) |
| `stats/gpu1/item/s1/`, `stats/gpu1/item/btzsc/` | Extracted per-question arrays: `common.npz` (question→request, request→task/dataset, group names, ids, option counts; BTZSC `ref`) and `<system>.npz` (`correct`, `conf`, `brier`, `answered`; BTZSC also `pred`); `check.json` = max diffs vs the score files (34 MB / 100 MB) |
| `stats/gpu1/item/rescore_{s1,btzsc}_crivo-17m.json` | Official re-score check results (all diffs 0) |
| `stats/gpu1/integrity.json` | Per result file: lines, unique ids, repeated lines, non-ok lines, final status |
| `stats/gpu1/s1.json` | E2. `score_mode.{all,unseen,seen}.{accuracy,accuracy_pooled,brier,ece}` (task CIs). `item_mode.{all,unseen,seen}.{macro,pooled,ece10,brier}.{own,pairs}` (item CIs), plus `item_mode.check` |
| `stats/gpu1/btzsc.json` | E1. `score_mode.{all22,unseen4,seen18}.{macro_f1,accuracy}` (dataset CIs for all22/seen18; per-dataset diffs). `item_mode.{all22,unseen4,seen18}.{macro,pooled,ece10,brier}` (`pooled` = mean per-dataset accuracy here) |
| `stats/gpu1/e4.json`, `stats/gpu1/e6.json` | E4 and E6 (unchanged) |
| `stats/gpu1/gpu1_stats.md` | All summary tables (generated from the JSONs) |
| `results_gpu/HEADLINE.md` | Regenerated by `headline`: S1 task-mean all/unseen with item CIs, the pooled unseen column labelled separately, BTZSC all/unseen-4/seen-18, the zs-v2 caveat |

**Pair fields (JSON).** Pairs cover each Crivo vs each baseline plus every Crivo pair, with diff = A − B and A the
smaller/earlier Crivo. The md tables flip consecutive and 1B rows to larger − smaller. Fields: `diff`, `ci95`,
`p_boot_two_sided`, `excludes_0`; score-mode accuracy also has task win counts and a Wilcoxon p over tasks. Baselines
present:
- S1: DeBERTa-v3-large zs-v2, BART-MNLI, Qwen3-0.6B letter, Gemma-3-1B-it letter, raw Ettin-32M reranker, uniform;
- BTZSC: DeBERTa zs-v2, ModernBERT-large zs-v2, Qwen3-0.6B, Gemma-3-1B, raw Ettin. BART and uniform were not run on
  BTZSC; Crivo-32M v1 was run on BTZSC only.

### 16.3 Methods

Common: B = 2,000, seed 0, percentile 95% CIs. Inside one call the same draws serve all systems and subsets, so
differences are paired. Last line per id wins (as in the scorers); errors, `unsupported` and missing answers count as
wrong, with uniform probabilities for Brier/ECE.

**Item CI (the main CI, E1/E2).**
- Units are requests, resampled with replacement inside each task (S1) or dataset (BTZSC); the questions of a request
  move together.
- Per draw the official statistic is recomputed:
  - S1: per-task accuracy, then the mean over the subset's tasks;
  - BTZSC: per-dataset macro-F1 with sklearn's class set (classes present in the resample's y_true or y_pred, unanswered
    = class −1, zero_division 0), then the mean over the subset's datasets.
- This CI is conditional on the tasks/datasets: it is item-sampling noise for *this* benchmark.

**Task / dataset CI (score mode, kept).**
- S1: tasks resampled inside the 13 domains. BTZSC: datasets resampled (all-22 and seen-18; unseen-4 has too few
  units, so per-dataset differences are listed).
- This is uncertainty over which tasks the benchmark contains, and it is much wider. Report both: item CI for "on
  S1-Arena", task CI for "on tasks like these".
- The old normal-approximation item CI (`item_normal_ci95`) is superseded. Its widths turned out almost equal to the
  bootstrap's (e.g. 17M − DeBERTa all: [+0.005, +0.017] vs [+0.006, +0.018]).

**Subsets.**
- S1: all 209 / unseen 145 / seen 64 tasks (`s1arena/contamination.json`, v2 family).
- BTZSC: all-22 / unseen-4 (appreviews, empathetic, trueteacher, capsotu) / seen-18.
- `pooled` (S1) = question-weighted accuracy over the subset. It is not the leaderboard statistic. Its item CI comes
  from the same draws.

**Calibration (E1/E2).**
- ECE-10 = `e5/calib.py::ece_equal_mass` with 10 equal-mass bins. Weights put each task/dataset of the subset at 1/T
  and its questions equal inside it. Confidence = probability of the predicted option (S1: the scorer's argmax after
  its normalization; BTZSC: the chosen label). Correct = top-1 against the hard gold.
- Brier = multi-class Brier with the same weights. S1 uses the scorer's target (soft distribution where given), so the
  S1 Brier point value equals the score files' task-mean Brier.
- `S1-ECE` = S1-Arena's own per-task ECE (pooled within a task, tied confidences grouped), task mean, task CI only.
- e5's ECE does not group tied confidences. That matters only for near-constant systems (uniform, raw Ettin).

**Exclusions.** Qwen3-0.6B rot4 is excluded from S1 (it ran on `s1_60` only).

**E4, E6 and integrity:** as in the first pass. Key points:
- E4 flips use the `e4/perturb.py` logic; own-set flip rates equal the GPU box's `flips.json` exactly; paired
  differences are on the common set; CIs come from a stratified cluster bootstrap by benchmark.
- E6 rows are valid only if they appear in `logs/e6_redo.log`. Qwen3-0.6B and the 32M per-option row are first-pass
  only.
- Integrity: every file has every id. Repeats come only from retries; final statuses are `ok` or `unsupported`.
- S1 unanswered questions: Crivos 72 each, DeBERTa 170, BART 89, Qwen 642, Gemma 659, Ettin and uniform 0. BTZSC: 0
  for all systems.

### 16.4 Commands

```bash
cd pareto/paper/exp            # run each line alone (one python at a time); add -I when reading ext/ data
python3 -I stats/rescore_check.py s1 --data stats/gpu1/ext/s1arena_data --code stats/gpu1/ext/s1arena_code \
    --results results_gpu/s1/crivo-17m.jsonl --score results_gpu/s1/crivo-17m.score.json --out stats/gpu1/item/rescore_s1_crivo-17m.json      # 6 s, 0.53 GB
python3 -I stats/rescore_check.py btzsc --requests stats/gpu1/req/btzsc_full.jsonl \
    --results results_gpu/btzsc/crivo-17m.jsonl --score results_gpu/btzsc/crivo-17m.score.json --out stats/gpu1/item/rescore_btzsc_crivo-17m.json  # 4 s
python3 -I stats/paired_bootstrap.py extract --bench s1 --requests stats/gpu1/req/s1_all.jsonl --data stats/gpu1/ext/s1arena_data \
    --code stats/gpu1/ext/s1arena_code --res results_gpu/s1 --out stats/gpu1/item/s1                                    # 10 s
python3 -I stats/paired_bootstrap.py extract --bench btzsc --requests stats/gpu1/req/btzsc_full.jsonl --res results_gpu/btzsc --out stats/gpu1/item/btzsc   # 34 s, 0.25 GB
python3 -I stats/paired_bootstrap.py s1    --res results_gpu/s1 --contam s1arena/contamination.json --items stats/gpu1/item/s1 --out stats/gpu1/s1.json          # 2.5 min, 0.23 GB
python3 -I stats/paired_bootstrap.py btzsc --res results_gpu/btzsc --contam btzsc/contamination.json --items stats/gpu1/item/btzsc --out stats/gpu1/btzsc.json  # 7 min, 0.30 GB
python3 -I stats/paired_bootstrap.py md       --dir stats/gpu1
python3 -I stats/paired_bootstrap.py headline --dir stats/gpu1 --out results_gpu/HEADLINE.md
# unchanged from the first pass:
python3 stats/e4_rows.py --suite ../../kit/suite-0.3 --out stats/gpu1/req/e4_orig.jsonl     # md5 4643cfec…
python3 stats/paired_bootstrap.py integrity --res results_gpu --out stats/gpu1/integrity.json
python3 stats/paired_bootstrap.py e4 --rows stats/gpu1/req/e4_orig.jsonl --res results_gpu/e4 --out stats/gpu1/e4.json
python3 stats/paired_bootstrap.py e6 --gpu results_gpu/e6 --cpu results_cpu/e6cpu --redo_log results_gpu/logs/e6_redo.log --out stats/gpu1/e6.json
```
Without `--items`, `s1`/`btzsc` write score mode only, with `item_mode: null`. `extract` aborts if any recomputed
per-task or per-dataset value differs from the `.score.json` by more than 1e-9.

### 16.5 Results (full tables in `stats/gpu1/gpu1_stats.md`)

**E2 S1-Arena, official task-mean accuracy, Crivo − DeBERTa-v3-large zs-v2.** Each cell gives Δ, the per-item paired
bootstrap CI and the task CI:

| Crivo | all 209 tasks | unseen 145 tasks |
|---|---|---|
| 17M | +0.011, item [+0.006, +0.018], task [−0.013, +0.037] | +0.000, item [−0.007, +0.008], task [−0.021, +0.021] |
| 32M v2 | +0.031, item [+0.026, +0.037], task [+0.007, +0.057] | +0.015, item [+0.008, +0.022], task [−0.005, +0.035] |
| 68M | +0.049, item [+0.043, +0.055], task [+0.024, +0.074] | +0.027, item [+0.020, +0.033], task [+0.004, +0.049] |
| 150M v2 | +0.059, item [+0.053, +0.065], task [+0.031, +0.086] | +0.028, item [+0.021, +0.035], task [+0.001, +0.055] |
| 150M v1 | +0.066, item [+0.061, +0.072], task [+0.039, +0.093] | +0.034, item [+0.027, +0.041], task [+0.007, +0.060] |
| 400M | +0.093, item [+0.088, +0.099], task [+0.066, +0.120] | +0.060, item [+0.053, +0.067], task [+0.032, +0.086] |
| 1B | +0.123, item [+0.117, +0.128], task [+0.095, +0.151] | +0.082, item [+0.075, +0.089], task [+0.055, +0.109] |

- **Second stop point (§11.1): yes, confirmed at item level.**
  - On all tasks every Crivo ≤ 150M beats DeBERTa-v3-large zs-v2 (435M) with the item CI excluding 0. 17M is the
    exception at the task level: its task CI includes 0.
  - On unseen tasks, 32M v2, 68M, 150M v2 and 150M v1 beat it with the item CI excluding 0; 68M, 150M v2 and 150M v1
    also have the task CI excluding 0. 17M ties (+0.000).
- **Crivo ≤ 150M vs the other baselines.** Every Crivo beats BART-MNLI, Qwen3-0.6B letter, Gemma-3-1B-it letter, raw
  Ettin and uniform with the item CI excluding 0, on all and on unseen tasks. The task CI excludes 0 for all of these
  pairs too, except 17M vs Qwen3-0.6B on unseen tasks (+0.012, item [+0.005, +0.019], task [−0.011, +0.035]).
  Besides that pair, the only Crivo-vs-baseline task CIs that include 0 are 17M vs DeBERTa (all and unseen) and 32M v2
  vs DeBERTa (unseen).
- **Crivo-1B vs everything.** It beats every baseline and every smaller Crivo with both CIs excluding 0, on all and on
  unseen tasks.
  - Baselines, all / unseen: DeBERTa +0.123 / +0.082; BART +0.167 / +0.121; Qwen3-0.6B +0.141 / +0.094 (item
    [+0.086, +0.102]); Gemma +0.196 / +0.151; Ettin +0.155 / +0.109; uniform +0.212 / +0.162.
  - Smaller Crivos, all: 1B − 400M +0.030, item [+0.024, +0.034], task [+0.014, +0.046]; 150M v2 +0.064; 150M v1
    +0.057; 68M +0.074; 32M +0.091; 17M +0.111.
  - Smaller Crivos, unseen: 1B − 400M +0.023, item [+0.016, +0.028], task [+0.002, +0.042]; 150M v2 +0.054; 150M v1
    +0.048; 68M +0.056; 32M +0.067; 17M +0.082.
- **Consecutive sizes** (larger − smaller; item CI and task CI):
  - all tasks: 32M − 17M +0.020 (both CIs exclude 0); 68M − 32M +0.018 (both); 150M v2 − 68M +0.009 (item
    [+0.005, +0.015] only); 400M − 150M v2 +0.035 (both); 1B − 400M +0.030 (both); 150M v2 − 150M v1 −0.008 (item
    [−0.013, −0.002] only).
  - unseen tasks: 32M − 17M +0.015 (item only); 68M − 32M +0.012 (item only); 150M v2 − 68M +0.002 (neither);
    400M − 150M v2 +0.032 (both); 1B − 400M +0.023 (both); 150M v2 − v1 −0.006 (neither).
- **Question-pooled view** (labelled separately in HEADLINE now). On unseen tasks, Crivo − DeBERTa with the item CI:
  - worse: 17M −0.044 [−0.052, −0.037]; 68M −0.015 [−0.022, −0.008];
  - level: 32M −0.002; 150M v2 −0.006 [−0.013, +0.000]; 150M v1 +0.000;
  - better: 400M +0.031 and 1B +0.043 [+0.035, +0.050]. The task CI for 1B was [−0.009, +0.089].

  The task mean is the paper's statistic; the pooled view must not be presented as "S1 unseen".

**E2 calibration** (ECE-10 equal-mass and Brier, each task weighing equally; item CIs):
- **ECE-10, all tasks:** Crivo 17M 0.069, 32M 0.125, 68M 0.133, 150M v2 0.174, 150M v1 0.185, 400M 0.164, 1B 0.191.
  Baselines: DeBERTa 0.090, BART 0.133, Qwen 0.229, Gemma 0.417, Ettin 0.046, uniform 0.038.
  - ECE does not fall with size.
  - Only 17M beats DeBERTa (−0.021 [−0.027, −0.015]). Every Crivo beats Qwen and Gemma.
- **Brier, all tasks:** 17M 0.603, 32M 0.610, 68M 0.595, 150M v2 0.617, 150M v1 0.634, 400M 0.593, 1B 0.579. Baselines:
  DeBERTa 0.632, BART 0.665, Qwen 0.712, Gemma 0.936, Ettin 0.612, uniform 0.616.
  - Every Crivo except 150M v1 (+0.002, n.s.) has a lower Brier than DeBERTa (1B −0.053 [−0.058, −0.047]). With the
    task CI only 17M, 68M, 400M and 1B were separable.
- **Unseen tasks (Brier):**
  - vs DeBERTa: 17M −0.018, 32M −0.005 and 68M −0.014 are better (CIs exclude 0); 400M and 1B tie; 150M v2 +0.019 and
    150M v1 +0.041 are worse.
  - **vs uniform: every Crivo has a Brier no better than uniform's 0.619** (17M +0.002, n.s.; 1B +0.016
    [+0.010, +0.023]). That is over-confidence on new tasks, consistent with E5.

**E1 BTZSC** (macro-F1, item CIs; see the zs-v2 caveat below):
- **DeBERTa-zs-v2 beats every Crivo** on all-22, unseen-4 and seen-18 with the item CI and the all-22 dataset CI
  excluding 0. For 1B: −0.049 [−0.053, −0.046] on all-22, −0.058 [−0.067, −0.049] on unseen-4, −0.047 [−0.051, −0.044]
  on seen-18 (seen-18 dataset CI [−0.097, +0.005]). For 150M v2: −0.129, −0.180 and −0.118.
- **ModernBERT-zs-v2 beats every Crivo at item level**, 1B included: −0.025 [−0.028, −0.021] on all-22, −0.031 on
  unseen-4. The dataset CI for 1B includes 0 ([−0.067, +0.023]).
- **Every Crivo, 17M included, beats Qwen3-0.6B, Gemma-3-1B and raw Ettin** on all-22, unseen-4 and seen-18 at item
  level. The tightest case is 17M − Qwen on unseen-4: +0.017 [+0.007, +0.027].
- **Consecutive sizes, all-22:**
  - 32M − 17M +0.090, 68M − 32M +0.065, 400M − 150M v2 +0.028, 1B − 400M +0.051 (both CIs exclude 0);
  - 150M v2 − 68M +0.004 [+0.001, +0.007] (item only);
  - v2 − v1: 32M +0.089; 150M +0.006 [+0.002, +0.009] (item only).
- **BTZSC calibration (all-22):**
  - ECE-10: Crivo 0.022 (32M v2) to 0.086 (150M v2), 1B 0.049. Baselines: DeBERTa 0.099, ModernBERT 0.100, Ettin
    0.127, Qwen 0.257, Gemma 0.363. **Every Crivo has a lower ECE than every baseline** (item CIs exclude 0).
  - Brier: 1B 0.298 vs DeBERTa 0.281 (DeBERTa better, +0.018*) and ModernBERT 0.306 (1B better, −0.008*). Every Crivo
    beats Qwen, Gemma and Ettin.
- **zs-v2 caveat** (`btzsc/zsv2_overlap.md`): both zs-v2 models were fine-tuned on training splits of all 22 BTZSC
  datasets with BTZSC's verbalizers. They are an **upper reference, not a zero-shot baseline**, and on unseen-4 the
  exposure is asymmetric in their favour.

**E4 / E6:** unchanged; see the first-pass summary in `gpu1_stats.md`.
- E4 flip rate (perm mean) falls with size, from 17M 0.231 to 1B 0.118. DeBERTa-zs is at 0.019; the LLMs and raw Ettin
  are at 0.67–0.76.
- E4-orig Brier improves with size, and every Crivo beats every E4 baseline in Brier. DeBERTa and Ettin have the lower
  ECE.
- E6 GPU median latency: Crivo-17M 8.2 ms to 1B 24.3 ms; DeBERTa-zs 36.1 ms.

### 16.6 Caveats

1. **Two kinds of CI.** The item CI is conditional on the benchmark's tasks/datasets. The task/dataset CI asks about
   other tasks and is wider: about 4× on S1, more than 10× on BTZSC. Say which one a claim uses. Claims vs DeBERTa that
   survive both: on all tasks, every Crivo ≥ 32M; on unseen tasks, 68M, 150M v2, 150M v1, 400M and 1B.
2. **"S1 unseen" means the task mean** (official). The question-pooled numbers (old HEADLINE column) are labelled
   "pooled", and the conclusions differ for Crivo ≤ 150M vs DeBERTa on unseen tasks.
3. **Unseen sets are the v2-family audits.** For the v1 models they are approximate (their own audits differ, §2.2).
4. **BTZSC zs-v2 rows are supervised few-shot** (caveat above). The like-for-like BTZSC baselines are Qwen, Gemma and
   Ettin here, plus the published BTZSC rows (§2.3).
5. **ECE.** e5's ECE-10 does not group tied confidences, unlike S1's own ECE, so uniform and raw Ettin get very low ECE
   from near-constant confidences. A low ECE alone does not make a system useful. BTZSC confidence = p(chosen label).
   BTZSC's gold-0 quirk (banking77/massive) also enters Brier.
6. E4 baselines have unsupported requests, and E4 accuracy has no accepted sets (unchanged). E6 has no CIs, and its
   unsupported requests are untimed.
7. Qwen3-0.6B rot4 is excluded from S1 (different items). The S1 snapshot lacks 2,043 local-only requests (§3).

## 17. GPU-2: A1/A2 grid (2026-10-21, Lambda A100 SXM4 40 GB, id <redacted>, `<gpu2-host>`)

- **Try 1, discarded.** `a1a2/train_grid.sh` (2 training lanes + eval lane) does not fit 40 GB. The per-option run's
  caching allocator took the whole GPU, so 12 joint trainings and the J s0 eval died with CUDA OOM at load. Jraw also
  failed because `jhu-clsp/ettin-encoder-32m` had only its config prefetched; it was downloaded afterwards. J s0
  finished but silently dropped 64 of 2,112 batches under `--skip_oom` (2,048 steps saved), so it is invalid. Try 1 is
  archived on the box as `~/results/a1a2_try1/` and must not be reported.
- **Fixes.**
  - `train_ablate.py` now counts `oom_skips`, logs it every 50 steps and stores it in the checkpoint metadata. It also
    honours `CUDA_MEMFRAC` (`torch.cuda.set_per_process_memory_fraction`).
  - New `a1a2/train_grid_serial.sh` runs one training at a time:
    - joint arms first, capped at 0.72, with the eval lane alongside (`--batch 64`);
    - then the 3 per-option runs uncapped and alone (the eval lane waits; `TRAINING_ALONE`/`EVALUATING` flags);
    - `SKIPS` lines appear in STATUS for any run that dropped batches;
    - eval lines report the count of error lines.
  - Arms, data, flags, eval sets and scoring are unchanged.
- **Started** 09:03 UTC (box clock). A joint run alone takes about 3.5 min (about 720 q/s); a per-option run alone about
  40 min.
- **Results are synced** to `results_gpu2/` (without `ck/` until the end).
- **E7 inputs:** the request files `~/req/s1_all.jsonl` and `~/req/btzsc_full.jsonl`, the BTZSC/S1-Arena data and the
  S1-Arena code were copied from this box to `stats/gpu1/req/` and `stats/gpu1/ext/` for the item-level bootstrap.
  - md5 s1_all `3159ca24f95da9a39d35c6588a1a587a`
  - md5 btzsc_full `9bbbabf3c0758a0757efafbd5f07858a`
- **v2 (09:24 UTC).**
  - **Why:** evals take about 11.5 min per checkpoint, against about 5 min per joint training, and serial v1 would have
    left 7 evals waiting about 2 h behind the per-option runs.
  - **What changed:** `a1a2/train_grid_serial2.sh` runs two eval lanes, with a lock dir `ev/<arm>_s<seed>.lock` per
    checkpoint. The per-option runs are capped at 0.75 instead of running alone.
  - **Rule:** any `SKIPS` line means that run is rerun alone before reporting.
  - **How it restarted:** v1 was stopped while JnoBrier s1 trained (its partial log was deleted; it restarted from
    scratch). Evals resume by id (`run_requests.py`).
  - **Checks:** J s0–s2 and JnoBrier s0 trained with `oom_skips: 0`. J s0 eval: 0 error lines.
- **v3 (10:24 UTC).**
  - **Why:** under the 0.75 cap, P s0 dropped 187 of its first 800 batches to OOM; per-option batches need more than
    30 GB.
  - **What changed:** `a1a2/train_grid_serial3.sh` adds `--grad_ckpt` to the per-option runs only. The gradients are
    the same and activation memory is much lower; P s0 shows 0 skips at 42 q/s, about 76 min per seed.
  - **Unchanged:** all 12 joint runs, already trained with 0 skips.
  - **Bad attempt:** that P s0 attempt is archived on the box in `~/results/a1a2_badP/`.
  - **Cost note:** gradient checkpointing changes only the cost, not the result. A1's cost comparison must therefore
    use E6 latency (inference) and the joint-vs-per-option training throughput measured without checkpointing, from the
    try-1 logs (P alone at about 65 q/s), not the v3 P training times.
- **Eval OOM while P trains (10:45 UTC).**
  - **What failed:** the Jraw s2 `s1_60` eval (batch 64) shared the GPU with P s0 and ended with 192 effective OOM
    errors.
  - **Retries:** `~/fix_jraw2.sh` (on the box) reran them at batch 8, then at batch 1; 5 very long requests still need
    a free GPU.
  - **Rule:** after GRID DONE, sweep every `ev/*` for effective errors (last line per id), rerun them with the GPU
    free, rescore and rerun `summarize.py` before reporting.
  - **Other evals:** all other joint-arm evals had 0 effective errors at 10:54 UTC. Their "error lines" in STATUS were
    superseded lines.
- **P s0 dropped 1 batch to OOM (step about 4,700; 1 of 6,858).** It trained alone at the 0.75 cap with `--grad_ckpt`.
  - **Fix:** from 11:12 UTC, `train_ablate.py` ignores `CUDA_MEMFRAC` for per-option runs, so P s1 and P s2 run
    uncapped. An eval that OOMs beside them is swept and rerun at the end; a dropped training batch cannot be.
  - **P s0 is kept and disclosed** (0.015% of its batches) unless its final count rises. Rerunning it costs about 80 min.
- **Per-option eval batched (11:43 UTC).**
  - **Why:** `PerOptionEngine.__call__` (one small batch per question) ran at about 0.6 requests/s on pub15, which is
    more than 20 h per P checkpoint.
  - **What changed:** `a1a2/peropt.py` gained `batch_call`, which packs all options of all questions of up to N
    requests, length-sorted, under the same batch_size and token_budget. It is about 20× faster with P s1 training
    alongside.
  - **Check** on 300 pub15 requests with P_s0: 17,464/17,464 answers equal to `__call__`; max |Δp| 0.009 (bf16 batch
    shapes).
  - **Restart:** the grid's two eval lanes were stopped by PID (the training lane runs on). The 455 unbatched P s0
    lines are archived in `~/results/a1a2_badP/P_s0_unbatched/`, so P s0 is evaluated in one mode.
  - **New script:** `a1a2/eval_P.sh` (a copy of `~/eval_P.sh`, tmux `evalP`) evaluates P s0–s2 with `--batch 32` as
    each checkpoint appears, then runs `summarize.py` and writes `P EVALS DONE`.
  - **Note:** the grid's own `GRID DONE` line comes earlier and its `summary.txt` lacks P, so it must be ignored.
- **A1 confound found and fixed (12:36 UTC).**
  - **The confound:** with the same token budget per step, P gets about 3.25× the optimizer steps of J on the same
    150k rows (6,858 against 2,112), because a per-option row is n options long, so P sees about 22 questions per step
    against about 71.
  - **P s0 against J s0–s2, before any fix:**
    - pub15 skill 0.248 against 0.172–0.186 for every joint arm;
    - S1 s1_60 accuracy 0.445 against 0.403–0.408;
    - BTZSC500 macro-F1 0.579 against 0.35–0.49;
    - E4 flips 0, against about 0.26–0.29 for J's permutations.

    The P flips are 0 by construction, since each option is read alone; only the yes/no swap can flip.
  - **Reading:** the joint arms at 2,112 steps are likely under-trained, and J's BTZSC varies a lot across seeds.
  - **Fix:** a new arm, **Jsteps** (`grid.tsv`): J with `--max_tokens 6784`, which gives about 6,845 steps (step count
    measured with `length_batches`: 6,656 → 6,978; 7,168 → 6,442).
    - `a1a2/jsteps.sh` (tmux `jsteps`) trains it after the grid's training lane ends, so it never shares the GPU with
      a P training. It caps at 0.6, evaluates, waits for `P EVALS DONE`, reruns `summarize.py` and writes
      `JSTEPS DONE`.
    - **The fair A1 comparison is P against Jsteps;** J against Jsteps is the step effect.
    - `summarize.py` compares each arm with J; P minus Jsteps has to be computed separately.
- **P s2 dropped 6 batches by step 1,850 (13:20 UTC).** It was uncapped but shared the GPU with the batched P s1 eval.
  - `evalP` was stopped and restarted with `eval_P.sh` waiting for `lane.finished`, so P evals never share the GPU with
    a P training. P s2 runs alone from 13:21; P s1's eval resumes by id afterwards.
  - **Final P skip counts** (all disclosed): s0 2, s1 0, s2 ≥ 6 of about 6,860 batches.
- **GPU-2 finished (15:03 UTC).** `P EVALS DONE` at 14:55 and `JSTEPS DONE` at 14:56.
  - **Final error sweep, GPU free (`logs/fix_final.sh`):** Jraw_s2 s1_60 (5) and P_s0 s1_60 (224, from the P s0 eval
    run beside P s1) were rerun at batch 1, rescored, and `summarize.py` was rerun. All 18 eval dirs (6 arms × 3 seeds)
    now have **0 effective errors**.
  - **Backup:** `results_gpu2/` (3.1 GB, including `ck/`) holds 388 files, all with md5 equal to the box's (one stale
    local `TRAINING` flag removed). The discarded try 1 and the bad P attempt are in `results_gpu2_discarded/`.
  - **Shutdown:** instance terminated; `lambda budget` reads US$0/h.
  - **Cost:** GPU-2 ran from about 08:45 to 15:04 UTC, about 6.3 h × US$1.99 ≈ US$12.6.
- **Results (mean ± sd over 3 seeds; `results_gpu2/summary.txt`, `summary.json`, `P_vs_Jsteps.txt`):**

| Arm | pub15 skill | pub15 ECE | S1 acc | S1 Brier | BTZSC500 F1 | flip perm | flip yn |
|---|---|---|---|---|---|---|---|
| J | 0.180±0.002 | 0.123±0.017 | 0.406±0.003 | 0.612±0.002 | 0.425±0.069 | 0.277±0.015 | 0.121±0.040 |
| JnoBrier | 0.182±0.005 | 0.135±0.021 | 0.425±0.011 | 0.603±0.003 | 0.483±0.049 | 0.265±0.034 | 0.126±0.079 |
| JnoKD | 0.177±0.001 | 0.127±0.009 | 0.405±0.004 | 0.608±0.004 | 0.380±0.031 | 0.276±0.002 | 0.072±0.044 |
| Jraw | 0.175±0.004 | 0.149±0.019 | 0.413±0.001 | 0.608±0.003 | 0.350±0.010 | 0.223±0.007 | 0.036±0.021 |
| Jsteps | 0.187±0.004 | 0.102±0.036 | 0.417±0.014 | 0.602±0.002 | 0.485±0.029 | 0.262±0.030 | 0.084±0.017 |
| P | 0.245±0.003 | 0.103±0.019 | 0.455±0.001 | 0.584±0.003 | 0.602±0.022 | 0 | 0 |

- **P − Jsteps (step-matched, Welch 95% CI over 3 + 3 seeds):**
  - pub15 skill +0.058 [+0.050, +0.066];
  - S1 acc +0.037 [+0.002, +0.073];
  - S1 Brier −0.019 [−0.024, −0.013];
  - BTZSC F1 +0.116 [+0.055, +0.177];
  - ECE ±0 (CI wide);
  - permutation flips −0.26, which is 0 by construction for P.

  The pub15 question accuracy paired over seeds is only +0.012 for P − J. The skill gain is concentrated in benchmarks
  with few questions per request, while big multi-question benchmarks such as ACOS barely move: check per benchmark
  before writing.
- **Reading, with caveats:**
  - **The per-option reader is clearly better at this scale and regime** (32M, 150k rows, 1 epoch), at n× inference
    cost. Possibly because the reranker init is trained on (query, document) pairs, the per-option format. Untested:
    P from the raw encoder, and P at the full recipe's scale.
  - **Brier and KD:** no clear effect on pub15. Without Brier, S1 is +0.019 and BTZSC +0.058 (both noisy), and ECE is
    slightly worse. Without KD, BTZSC is −0.045.
  - **Reranker init vs raw Ettin:** reranker init helps BTZSC (+0.075) and calibration (ECE 0.123 vs 0.149); pub15 and
    S1 are about equal.
  - **All J* arms are in a short regime;** J's BTZSC sd is 0.069.
  - **P OOM skips:** s0 2, s1 0, s2 6 of about 6,860 batches.

### 17.1 A1 per-benchmark analysis (2026-10-21, Mac CPU) [ours-public]

- **Files.** `a1a2/analyze_a1.py` (docstring = method; one process, ~1 min, ~0.45 GB). Outputs in
  `results_gpu2/analysis/`: `a1_analysis.json` (all numbers) and `a1_analysis.md` (tables).
  - **Run:** `cd pareto/paper/exp && python3 -I a1a2/analyze_a1.py`
- **Inputs and checks.**
  - pub15 requests were not local. They are rebuilt in memory from `kit/suite-0.3` with `build_eval.py`'s rule (seed 0).
    The ids equal every result file: 13,358 requests, 52,837 judged questions. Accuracy per benchmark equals every
    `pub15.score.json` exactly.
  - BTZSC500 and s1_60 request rows come from `stats/gpu1/req/{btzsc_full,s1_all}.jsonl`, matched by id (11,000/11,000
    and 12,540/12,540). The recomputed BTZSC macro-F1 equals the score files exactly.
  - E4 orders are regenerated with `e4/perturb.py`'s rule. 0 missing answers.
  - Token counts use the released 32M `tokenizer.json` (no model).
- **Method.**
  - Seed means of per-run metrics.
  - Two CIs: Welch over seeds (3 + 3), and a paired cluster bootstrap with B = 2,000. The bootstrap resamples requests
    within each benchmark, recomputes the per-seed metric on each draw and then takes the seed mean; the overall mean
    is stratified by benchmark.
  - Breakdowns are benchmark-weighted, unclipped skill over benchmarks with ≥ 20 questions in the cell.
  - Position: within-question contrasts on E4, where each question is compared with itself across its permutations.

**Findings (P − Jsteps unless stated).**

1. **Skill vs question accuracy.** The gap between +0.058 skill and +0.012 question accuracy comes from weighting, not
   from the chance correction.
   - The +0.012 is pooled over questions, and ACOS holds 66% of them: 58 binary questions per request, every arm at
     0.994, Δ +0.000.
   - Bench-weighted accuracy moves +0.055 [+0.047, +0.065], about the same as skill. Pooled accuracy without ACOS moves
     +0.038.
2. **The skill gain is broad but top-heavy.**
   - The two largest gains: BPoMP +0.60 (acc 0.549 → 0.851) and VAST +0.39 (0.345 → 0.608).
   - Then FinEntity +0.18, CLINC150 +0.15, ContractNLI +0.11, BANKING77 +0.11, When2Call +0.10, BFCL +0.09 and
     HoVer +0.08.
   - Without the top 2 benchmarks the mean gain is +0.033; without the top 4, +0.025. The median benchmark gain is
     +0.019.
   - 24 benchmarks go up and 8 go down. The bootstrap CI is above 0 for 12 and below 0 for 2.
   - 12 benchmarks are at the floor for every arm (skill < 0.03): ANLI, BRIGHT, CLadder, ChessBench, GPQA, GSM8K, HLE,
     Humicroedit, MMLU-Pro, POP909, ToolRet and WinoGrande.
   - **Clipping hides some gains.** BRIGHT and ToolRet are below chance for every arm, so they score 0 skill even
     though P's accuracy is +0.32 and +0.18 higher. Unclipped mean skill rises +0.091.
3. **No growth with the number of options** (hypothesis a).
   - The benchmark-level Spearman ρ between Δskill and mean options is −0.02.
   - By option bin (bench-weighted Δskill): 2 options +0.135, 3–4 +0.084, 5–9 −0.014, 10–29 +0.018, 30+ +0.066.
   - Question type: choice +0.092, noul +0.118 (2 benchmarks only).
   - Length: no trend. Δ is +0.09 to +0.12 below 2,048 tokens and +0.025 (n.s., 4 benchmarks) at 2,048–8,191.
   - Questions per request: 1 → +0.118; 5–16 → −0.031 (n.s.; SATA-Bench).
4. **Option position is real but small** (hypothesis b). The E4 within-question contrasts are exactly 0 for P, as they
   must be.
   - The joint reader is 1.3–1.5 points less accurate when the gold option is first or last than when it is in the
     middle (n ≥ 3): Jsteps −0.015 [−0.025, −0.004], J −0.013.
   - On binary questions it is 2.6–3.5 points more accurate when the gold is the first option: Jsteps +0.026 [+0.007,
     +0.046].
   - The joint reader under-picks the first and last options. Jsteps picks them 0.171 / 0.144 of the time on pub15
     choice questions with ≥ 3 options, while gold is there 0.194 / 0.183 of the time.
   - Position explains a few points at most of the roughly 5.5-point gap.
5. **Where joint beats per-option** (hypothesis c).
   - **SATA-Bench:** Δskill −0.098 [−0.127, −0.064] against Jsteps, −0.174 against J. Accuracy is 0.603 for J, 0.560
     for Jsteps and 0.513 for P (2,347 binary questions, 9.5 per request).
   - **POP909-CL:** −0.009, at the floor.
   - **RAGTruth (noul):** −0.013, CI touching 0.
   - On the E4 yes/no subset, J reaches 0.673 and P 0.647.
6. **S1-Arena** (task mean): all tasks +0.037 Welch [+0.002, +0.073], task bootstrap [+0.022, +0.053]; P is better on
   135 of 209 tasks and worse on 70.
   - P is better in basic_cls (+0.089), judge (+0.077), ret (+0.075) and pro (+0.069); all four CIs exclude 0.
   - Small and n.s.: route −0.030, fore −0.021 and dialog −0.001.
7. **BTZSC500.**
   - P is ahead on 21 of 22 datasets. The exception is empathetic (32 labels): −0.039, item CI [−0.067, −0.008].
   - Largest gains: wikitoxic_insult +0.29, imdb +0.22, biasframes_offensive +0.20, banking77 +0.19 and massive +0.18.
   - The per-dataset Welch CIs are wide because the Jsteps seeds vary a lot.
8. **Cost.**
   - **Forward sequences per request**, joint → per-option: pub15 4.6 → 24.0 (5.2×), s1_60 1.7 → 10.8 (6.4×),
     BTZSC500 1.0 → 13.1 (13.1×).
   - **Encoded tokens:** 6.6×, 5.4× and 2.8× (median request about 2×). The per-option reader re-encodes the state once
     per option.
   - **E6 latency, 32M**, same released weights, one request at a time:
     - GPU: per-option / joint = mean 4.6×, p95 10.7×, median 1.02×; throughput 15.8 vs 71.9 req/s.
     - CPU (16 threads): mean 19.2×, p95 16.7×, median 1.95×; 0.36 vs 6.83 req/s.
   - **Training:** about 65–80 q/s for P (try 1, no gradient checkpointing) against about 670–720 q/s for J alone, so
     about 9–11× per question.

**Caveats.**
- **Regime.** 32M only, 150k rows, 1 epoch. P skipped 2/0/6 of about 6,860 batches to OOM. Neither reader is tested at
  the full recipe's scale.
- **Metric.** pub15 skill is not the official index; it is clipped at 0 and benchmark-unweighted. The item bootstrap is
  conditional on the benchmarks; with 37 benchmarks and a top-heavy gain, a benchmark-level CI would be wider.
- **E6 GPU per-option row.** It is first-pass, measured right after `par` was killed, and is not in
  `logs/e6_redo.log`; the CPU rows are valid.
  - E6 per-option used the released joint weights read per option through `PerOptionEngine.__call__` (cost only).
  - The redo-valid `reranker-32m` row, also per option, is slower still: mean 148.7 ms.
- **Training q/s** are indicative only: the GPU was shared with the eval lanes, and v3 P used `--grad_ckpt`.

### 17.2 Release of P checkpoints (2026-10-21, Mac; author approved publishing)

- **Repo:** [`tardellirs/crivo-32m-per-option-ablation`](https://huggingface.co/tardellirs/crivo-32m-per-option-ablation)
  - public model repo, CC BY-NC 4.0, base model `cross-encoder/ettin-reranker-32m-v1`;
  - not added to any collection, not submitted to the board, no other repo touched.
- **Commit:** `b3733ea2ef7a5ea88d7edc8dac311fd6b7f6e5f8`, a single commit made with `create_repo` + `upload_folder`
  ("Add per-option 32M ablation checkpoints (seeds 0-2), engine and model card").
- **Local staging folder:** `pareto/release/crivo-32m-per-option-ablation/`. The `seed*/` files are hardlinks to
  `results_gpu2/ck/P_s{0,1,2}/`, unchanged.
- **Files** (bytes):

  | Path | Size |
  |---|---:|
  | `.gitattributes` | 1,519 |
  | `README.md` (model card, English) | 18,123 |
  | `peropt_engine.py` | 15,085 |
  | `seed{0,1,2}/config.json` | 6,718 each |
  | `seed{0,1,2}/model.safetensors` | 128,141,676 each |
  | `seed{0,1,2}/tokenizer.json` | 3,583,508 each |
  | `seed{0,1,2}/tokenizer_config.json` | 533 each |

- **Checkpoints and md5 of `model.safetensors`:**

  | Folder | Source | md5 | step | oom_skips |
  |---|---|---|---:|---:|
  | `seed0/` | `P_s0` | `54326c3e4b9a4a459338f5239ed97edd` | 6,856 | 2 |
  | `seed1/` | `P_s1` | `64882bcfdd53eabbb21e058b979e2f99` | 6,860 | 0 |
  | `seed2/` | `P_s2` | `22d593920c6ef6319571edea526ccb81` | 6,853 | 6 |

  The Hub LFS sha256 of each file equals the local sha256.
- **Verification.**
  - `model_info` lists the files above, with `private: False`.
  - All three `seed*/config.json` downloaded back at the commit are byte-identical to the local files.
- **Engine.** `peropt_engine.py` is self-contained: it vendors `render`, `option_texts`, `collate`, `DecisionModel` and
  `load` from `train/model.py`, plus `answer_from_probs` and `Unsupported` from `common/engines.py`. The
  `PerOptionEngine.__call__` and `batch_call` logic is unchanged from `a1a2/peropt.py`.
  - The CLI is `python peropt_engine.py [--seed N] [--requests in.jsonl --out out.jsonl --batch 32]`. With no
    arguments it answers a demo request.
  - If `seed<N>/` is not next to the script, it downloads it from the Hub.
- **Local test** (Mac CPU, fp32, transformers 5.19.0, torch 2.13.0, seed 0 only, from a fresh copy holding only
  `peropt_engine.py` and `seed0/`; peak RSS about 0.57 GB).
  - **Setup:** transformers 5.19.0 was installed into a scratch `--target` directory. This Python build returns an
    empty `platform.mac_ver()`, which breaks `truststore` on import, so a local wrapper patched `mac_ver`. The engine
    itself is unchanged.
  - **Requests:** 5 requests (6 questions): `btzsc/yelpreviews/2088`, `btzsc/emotiondair/492`,
    `know.multihop.strategyqa:9ee9394f42ae` (noul), `basic_cls.language_style.pavlick_formality:2737c3f3af07`
    (score) and `pro.science.iclr2025_reviews:67051c92b036` (score + noul). The rows came from
    `stats/gpu1/req/{btzsc_full,s1_all}.jsonl`.
  - **Against `results_gpu2/ev/P_s0/{btzsc500,s1_60}.jsonl`** (GPU bf16, `batch_call --batch 32`): 6/6 decisions
    equal. The largest probability difference was 0.016 (strategyqa P(true) 0.536 vs 0.521); the others were at most
    0.006.
  - **CPU `__call__` against CPU `batch_call`:** 6/6 equal, max |Δp| 0.
- **Card content.**
  - Results come from §17 and §18.x: the per-arm table (J, JnoBrier, JnoKD, Jraw, Jsteps, Jcomp, P, Praw), P − Jsteps
    and P − Jcomp with Welch CIs, per-seed P numbers, the OOM skips 2/0/6, the costs (5–13× passes; GPU median ≈ joint,
    mean about 4.4×, p95 about 10×), usage, limitations and the citation (Tardelli Stekel, paper in preparation).
  - The card has no affiliation, no numbers from unrelated experiments and no session links.

---

## 18. GPU-3: reduced A1 controls and E6 latency (prepared 2026-10-21; not launched)

**Scope.** The author approved about US$6–7, about 3 A100-hours. That replaces the full-recipe Crivo-32M-PO run
(§19, deferred). GPU-3 has four items:
1. **Jcomp:** a compute-matched joint control for A1, 3 seeds.
2. **Praw:** the per-option reader from the raw Ettin encoder, 1 seed.
3. **E6 GPU latency** with the GPU alone.
4. **E6 CPU latency** on the same box with the GPU idle.

Everything lives in `gpu3/`. Nothing has been launched.

### 18.1 Pre-registration (written before any GPU-3 run)

- **Arms.**
  - **Jcomp** is the grid's J arm with identical flags (`grid.tsv` J): joint reader, `cross-encoder/ettin-reranker-32m-v1`
    init, KD from the Crivo-400M logits with α 0.5, Brier 0.5, 24,576 tokens per step, max 128 items, lr 1e-4 / head
    5e-4, warmup 5%, the same sub150k rows. Training: `train_ablate.py --epochs 4 --max_steps N_s`.
  - N_s is P's optimizer-step count for the same seed in GPU-2: **6,858 / 6,860 / 6,859** for seeds 0 / 1 / 2.
  - In `train_ablate.py` the schedule length is `total = min(int(per_epoch × epochs), max_steps)`, and warmup
    (5% of total) and linear decay run over `total`. So the LR schedule spans exactly N_s steps (about 3.25 passes over
    sub150k, since J has about 2,112 batches per epoch). No new option was needed.
  - Jcomp is matched to P in optimizer steps, LR schedule and per-step token budget. It still sees about 3.25× P's
    questions. Encoded training tokens are logged for both (`tokens_total` in each checkpoint's config) and reported.
- **Evaluation.** Exactly as in the grid:
  - sets pub15, btzsc500, s1_60 and the six E4 files;
  - `crivo` engine with `code=` the crivo-32m snapshot c14221dd, `--batch 64`;
  - the grid's scorers.
- **Primary comparison, the decision rule.** **P (GPU-2, seeds 0–2) − Jcomp (seeds 0–2)**, Welch 95% CI over seeds
  (3 + 3), on four metrics:
  - pub15 skill (`score_pub.py` mean chance-corrected skill);
  - S1-Arena s1_60 accuracy;
  - BTZSC500 macro-F1 (all 22 datasets);
  - E4 flip rate (mean over the three permutations; the yes/no swap is reported beside it).
- **How each metric is read.** For pub15, S1 and BTZSC:
  - CI entirely above 0: the per-option advantage **survives** the compute-matched control;
  - CI containing 0: **not distinguishable**;
  - CI entirely below 0: **reversed**.

  For flips, lower is better, so a CI below 0 means P is more robust. That is true by construction for permutations.
- **Secondary, descriptive, not decision rules.**
  - Jcomp − J and Jcomp − Jsteps (Welch).
  - Praw s0 against P s0, and against P's 3-seed mean ± sd. One seed gives no CI over seeds.
  - Praw s0 against the Jraw mean.
  - pub15 question accuracy P − Jcomp, paired over questions (seed-averaged, bootstrap B = 2,000).
  - Training tokens.
- **Script.** `gpu3/compare.py`. Dry-checked on the GPU-2 files with Jsteps in Jcomp's place, it reproduces
  `results_gpu2/P_vs_Jsteps.txt` to 4 decimals.
- **Praw.** Per-option reader, `jhu-clsp/ettin-encoder-32m` init, otherwise P's flags: KD, Brier 0.5, 24,576
  tokens/step, 1 epoch, `--grad_ckpt`. It runs uncapped, alone on the GPU, seed 0. It is evaluated with
  `PerOptionEngine.batch_call --batch 32` (as `eval_P.sh`), never while a training runs.
- **No silent OOM drops.**
  - Every run logs `oom_skips`. Any run with skips gets a `SKIPS` line in STATUS.
  - A Jcomp seed that drops a batch at the 0.6 cap is automatically retrained **alone and uncapped**. The capped run is
    kept as `*.skipped` and not reported.
  - Praw keeps P's `--skip_oom` flag. Its count is reported, like P's 2/0/6. Rerunning it is not in the budget.
  - Every eval is swept with the GPU free of training: effective errors are rerun at batch 1, and 0 is required before
    `DONE`.

### 18.2 Inputs (built and verified on the Mac, pushed and md5-checked on the box, `gpu3/inputs.md5.box`)

| Input | How | Check |
|---|---|---|
| `gpu3/inputs/sub150k.{jsonl.gz,t400.npz}`, `gold_only` | `make_subset.py` (seed 2026) from `data/mix400_s1.train.jsonl.gz` and `runs/teacher_logits/mix400_s1.t400.npz`; gold list `runs/mix400_s1.t400.gold_only` | composition equals §7.3 exactly (bekko 41,560 … t1 4,604) |
| `gpu3/inputs/evalsets/{pub15,btzsc500,s1_60}.jsonl` | `build_eval.py` from `kit/suite-0.3`, `stats/gpu1/ext/{btzsc_data,s1arena_data}` | request ids equal GPU-2 `J_s0` outputs (13,358 / 11,000 / 12,540); `score_pub.py` on GPU-2 J_s0 output reproduces its `pub15.score.json` exactly |
| `gpu3/inputs/e4/*.jsonl` | `e4/perturb.py make` from `stats/gpu1/req/e4_orig.jsonl` | orig md5 4643cfec… (= GPU-1/2); ids of all six files equal GPU-2's |
| `gpu3/inputs/lat1000.jsonl` | `e6.sh` rule: seed-7 shuffle of E4 `orig`, first 1,000 | 1,000 requests, 1,386 questions, 19.556 options per request: equal to GPU-1's E6 JSON; the first 300 have 445 questions, equal to the VM CPU run |
| `results_gpu2/ck/P_s0` | GPU-2 checkpoint (E6, both blocks) | md5 in the manifest |

**Raw encoder.** `jhu-clsp/ettin-encoder-32m` ships only `pytorch_model.bin`. `gpu/setup_gpu.sh`'s allow-patterns
(`*.safetensors` …) skipped it, which is why GPU-2 try 1 Jraw failed. `gpu3/setup_gpu3.sh` adds `*.bin` for it.
`run_gpu3.sh` phase 0 then builds both 32M bases offline and checks weight files for every other model. Checked on the
Mac: with `.bin` the raw encoder builds (32,033,665 parameters); without it, only the config arrives.

### 18.3 Order (`gpu3/run_gpu3.sh`, one tmux `gpu3`, STATUS `~/results/gpu3/STATUS`, resumable)

0. Input md5 and the offline load / prefetch check.
1. Training, strictly serial:
   - Jcomp s0, then s1, then s2, each capped at 0.6 (as `jsteps.sh`);
   - the s0 eval runs beside the s1 training, and the s1 eval beside the s2 training. These are joint-arm evals beside
     joint-arm trainings only.
2. A Jcomp seed with skips is retrained alone, uncapped.
3. **Praw s0, alone.** The script refuses to start it while any eval runs.
4. With no training on the GPU:
   - the Praw eval runs beside the Jcomp s2 eval;
   - then the error sweeps, scoring and `compare.py`.
5. **E6 GPU, GPU alone** (the script checks that no training or eval runs).
   - Systems: released Crivo-32M v2 (joint, snapshot c14221dd), and per-option 32M = GPU-2 `P_s0` read in two ways:
     - `PerOptionEngine` per question (GPU-1's mode);
     - the new `request_batched=1` (all options of all questions of the request in shared batches, like the joint
       engine).
   - The fixed `lat1000` set, 20 warm-up requests, two passes in alternating order.
   - Median / mean / p95 per run, plus per-request files (`e6/latency.py --per_request`).
6. **E6 CPU, GPU idle and CUDA hidden, nothing else running.**
   - `torch.set_num_threads(16)`. The CPU model, `lscpu` and `nproc` go to `e6cpu/cpu.txt` and to each JSON
     (`cpu_model`).
   - **The same first CPU_N = 80 requests of lat1000 for every system:** Crivo 17M / 32M / 68M / 150M / 400M / 1B, 32M
     per-option (P_s0, request-batched), reranker-32m raw, DeBERTa-v3-large-zeroshot-v2.0. 10 warm-up requests each.
   - `gpu3/e6_common.py` reports, per system:
     - skipped (unsupported) counts;
     - median / p95 / mean on every request it answered;
     - the same on the subset **every** system answered.

### 18.4 Time and cost (Lambda A100 40 GB SXM4, US$1.99/h; basis = GPU-2 STATUS times)

| Block | Basis | Wall |
|---|---|---:|
| Push + setup + prefetch check | GPU-2: about 18 min to the first training | 0:20 |
| Jcomp s0, alone | J alone 232 s per 2,112 steps → × 3.25 | 0:13 |
| Jcomp s1 ∥ eval s0; Jcomp s2 ∥ eval s1 | J beside an eval 301–333 s → × 3.25; joint eval about 11.5 min | 0:35 |
| Praw s0, alone | P with grad_ckpt: 64.5 / 70 min (GPU-2, with evals beside); alone about 60–65 | 1:05 |
| Praw eval ∥ Jcomp s2 eval, sweeps, scoring | P eval alone 33 min (P_s2) | 0:37 |
| E6 GPU (3 systems × 2 passes × 1,000 requests) | joint about 10 ms/request, per option about 30–65 ms | 0:06 |
| E6 CPU (9 systems × 80 requests, 16 threads) | VM run per-request means (DeBERTa 8.6 s/request) | 0:32 |
| Fetch (about 0.8 GB) + md5 | | 0:05 |
| **Total** | | **≈ 3.5 h (3.1–4.0)** |
| **Cost** | | **≈ US$7.0 (6.2–8.0)** |

`CPU_N=100` adds about 8 min (+US$0.3).

### 18.5 Files changed or added

- **`a1a2/train_ablate.py`.** Whole-run counters only: `tokens_total`, `questions_total`, `train_seconds` and
  `total_planned` go to the final `saved` line and the checkpoint metadata. Training behaviour is unchanged.
- **`a1a2/peropt.py`.** `PerOptionEngine(request_batched=0|1)`. With 1, `__call__` answers one request through
  `batch_call`. The default is unchanged, so the grid and `eval_P.sh` are unaffected.
- **`e6/latency.py`** (changed for §19, kept): `--per_request` and a `cpu_model` field. Backward compatible.
- **New in `gpu3/`:**
  - `run_gpu3.sh` (`SERIAL_EVAL` and `SKIP_E6CPU` are for dry runs only; never set them on the box), `setup_gpu3.sh`, `launch.sh`;
  - `compare.py`, `eff_errors.py`, `e6_common.py`;
  - `inputs/` and `inputs.md5.box`.

### 18.6 Dry run (Mac CPU, 2026-10-21) and what it does not cover

**Setup.** A fake box tree in the scratchpad: real code, 300 sub150k rows with sliced teacher logits, 8 requests per
eval set, 6 per E4 file, 4 lat1000 requests. Knobs: `JSTEPS="0:8 1:9 2:7"`, and later
`EXTRA_TRAIN="--workers 0 --limit 40 --max_steps 3 --max_tokens 4096 --max_len 1024"`. The dry-only knobs
`SERIAL_EVAL=1` (one Python process at a time) and `SKIP_E6CPU=1` keep the Mac light. **Peak RSS 1.67 GB.**

**Passed:**
- input md5 and the offline prefetch check;
- Jcomp s0–s2 at the 0.6 cap; s1 crossed an epoch boundary, so `--epochs 4 --max_steps N` works; `oom_skips` 0;
  `tokens_total` logged;
- Jcomp evals beside the next Jcomp training (first pass);
- Praw s0 with `--grad_ckpt`, raw encoder from `.bin`;
- Praw and Jcomp evals (crivo `code=` / `PerOptionEngine.batch_call`), the scorers and `compare.py`;
- E6 "GPU" block on CPU: 3 systems × 2 alternating passes, plus `e6_common.py` summary;
- **recovery paths:**
  - a simulated `SKIPS` on Jcomp s1: kept as `*.skipped`, retrained alone uncapped, re-evaluated;
  - an injected eval error: swept at batch 1, 0 effective errors, rescored.

**Bugs found and fixed:**
- the raw-encoder `.bin` was not fetched by the old patterns (the try-1 failure);
- a `$HOME` path in the prefetch check;
- `eval_ck` re-entered evals that had already run (a harmless resume);
- `eff_errors.py` now uses `run_requests`' rule: an id is done once any line is ok or unsupported.

The first dry pass also ran an eval beside a training on the Mac (about 2.9 GB RSS, heavy swap). It was stopped, and
the rest ran serially.

**GPU-box only** (not run on the Mac by design):
- the E6 CPU block (Crivo 68M–1B, DeBERTa-v3-large);
- real throughput and memory: Jcomp at the 0.6 cap with an eval beside it, and Praw alone;
- bf16;
- `setup_gpu3.sh` itself.

**Launch** (after you start the instance yourself; the script never creates or terminates one):
`caffeinate -dims bash pareto/paper/exp/gpu3/launch.sh <user>@<gpu-host>`. Then:
- monitor with `… status` every 10 min;
- on `ALL DONE`, run `… fetch` (it writes `paper/exp/results_gpu3/` and checks md5), terminate, and check
  `lambda budget`.

---

## 19. DEFERRED: Crivo-32M-PO full recipe (prepared, not run)

Prepared 2026-10-21 on the Mac. **Deferred the same day:** the approved budget is about US$6–7, and this run needs
about US$20. GPU-3 (§18) replaced it. Nothing was launched; no GPU was rented; nothing was uploaded. The cost and padding
findings below still hold.

The full plan is `po32/PLAN.md`; read it first. Everything lives in `po32/`.

**What it is.**
- The released Crivo-32M v2 recipe (`remote/run_small.sh`, size 32m, tag c32v2) with only the reader changed to per
  option.
- Everything else is the same:
  - data `mix400_s{1,2,3}`;
  - Crivo-400M logits for stages 1–2, with the v2 gold-only lists; stage 3 gold only;
  - the loss, LRs, warmups and EMA 0.999;
  - candidates s1 / s1-ema → s2, s3, S23 → WiSE 10/20;
  - the selection: `score400.py` + `select400.py --no_gates --tie_ss 0.5` on the same panels nd, ss, one external panel and
    pub_panel. None of them is the board.

**Key design: the same optimizer steps as v2.**
- `train.length_batches` on the rows reproduces the released step counts exactly: **16,318 / 3,134 / 3,287**,
  recomputed on the Mac. So each step has the same questions.
- Inside a step, the per-option sequences are packed into exact, length-sorted micro-batches of at most 65,536
  padded tokens.
- Questions larger than that use a **two-pass exact gradient**: no-grad logits → dL/dz → recompute chunks with grad.
  Ettin's dropout is 0.
- CPU check (`po32/check_po.py`): one batch, micro-batches and two-pass give the same loss and gradients (max |Δg|
  about 2e-7).
- KD alignment: per-option and joint rows match on order, target and teacher logits for 300/300 real rows.
- **No batch is ever dropped.** An OOM redoes the step with the micro budget halved. At the floor, the trainer saves
  its state and exits with code 2. Training is resumable (`--resume`).

**Findings from the CPU preparation (no model needed).**
- **Per-option token cost vs joint** (`po32/cost.json`): S1 3.76×, S2 4.19×, **S3 22.6×**.
  - 91% of S3's per-option tokens are the 5,000 "which API next" dialogues: about 44 options × a 6.7k-token catalog,
    about 300k tokens per question.
  - S3 alone costs as much as both S1 epochs: 1.42 B against 1.38 B per-option tokens.
- **GPU-2's per-option batching padded 5.2 tokens per real token**, because it padded to the longest sequence of a
  flat batch (`po32/padding.json`). The new rule pads 1.32 (S1), 1.18 (S2) and 1.007 (S3).
  - **Correction to §17.1 "Training: about 9–11× per question":** that ratio is inflated by padding. In tokens, the
    per-option reader costs about 3.8–4.2× the joint reader on stages 1–2 and about 23× on long multi-option inputs.
  - The paper should quote the token ratios, or this run's measured q/s, not the GPU-2 P training times.

**Stage 3 choice.** Full S3 at 8,192 tokens with two-pass chunks, for fidelity: the selection must see the same
candidates as v2. Fallback, a main-session decision only: if S3's ETA after its first 300 steps exceeds 5 h on A100,
select among s1 → s2 only and disclose it.

**Evaluation in the same session** (`po32/run_eval.sh`):
- package (release-style folder, kit `decision_index_engine:DecisionEngine`) + G4b;
- the **full public suite 0.3 through the kit's own runner**, 3 shards via `po32/suite_shard.py`, then merge, then
  the kit `pipeline` (resumes and scores);
- E1 BTZSC full, E2 S1-Arena full, E4 (6 variants), with sweep and scorers;
- **E6 GPU alone:** released 32M v2 joint vs Crivo-32M-PO vs GPU-2 P_s0 (+ the v2 weights read per option), 1,000
  requests, two alternating passes. This replaces the first-pass GPU-1 per-option row.
- **E6 CPU (16 threads, nothing else running):** every system (Crivo 17M–1B, 32M-PO, DeBERTa-zs-v2, reranker-32m) on
  the **same first 300 lat1000 requests**. Per-request records go through `e6/latency.py --per_request` (new,
  backward compatible), with the `cpu_model` field. Stats are reported on all answered requests and on the subset
  every system answered, with skipped counts, CPU model and thread count (`po32/e6_common.py`).

**Estimate** (Lambda; the basis is GPU-2's measured P/J throughput, corrected for padding; PLAN §5):

| | A100 40 GB SXM4 (US$1.99/h) | H100 80 GB SXM5 (US$4.29/h) |
|---|---|---|
| Training (S1 + S2 + S3) | about 4.4 h (3.3–6.5) | about 2.2 h (1.5–3.5) |
| Panels after S3 | about 0.85 h | about 0.5 h |
| Suite + E1/E2/E4 | about 1.6 h | about 0.9 h |
| E6 GPU + E6 CPU | 0.2 + 2.1 h | 0.15 + 2.1 h |
| Setup + fetch | about 0.55 h | about 0.55 h |
| **Total** | **≈ 9.9 h (8–14), ≈ US$20 (16–28)** | **≈ 6.6 h (5–9), ≈ US$28 (21–39)** |

**Recommended: A100 40 GB.** It is cheaper, and the 2.1 h CPU block costs 2.2× more on the H100 instance. With exact
micro-budgets, 40 GB is enough.

**Dry run (Mac CPU, tiny limits, all passed).**
- `run_train.sh`, serial and parallel. The parallel run covered the speculative S2→S3 chain both kept and cancelled
  (no orphan process, redone from s1-ema).
- `train_po.py`: OOM retry, fatal exit, resume.
- `run_eval.sh`, serial and parallel: package, G4b, suite shards → merge → score; kit pipeline resume; E1/E2/E4 and
  scorers; E6 GPU and E6 CPU blocks with summaries.
- Not testable on CPU:
  - GPU throughput and memory (watch `tok_per_s`, `pad_tok_per_s`, `mem_gb` and the ETA in the first 300 steps of each
    stage, above all S3);
  - bf16 G4b;
  - concurrent-eval memory;
  - DeBERTa, 68M–1B CPU latency;
  - `setup_po.sh`;
  - full suite timing.

**Launch, when approved:** `caffeinate -dims bash paper/exp/po32/launch.sh <user>@<gpu-host>`, after you start the instance
yourself. Then `… status` every 10 min and `… fetch` before terminating. See PLAN §8.

### 18.x GPU-3 results (run 2026-10-21 17:42–20:41 UTC, Lambda A100 SXM4 40 GB, id <redacted>, about 3.0 h ≈ US$6)

- **Run status.** ALL DONE at 20:38:56. `launch.sh … fetch` brought `results_gpu3/` back (697 MB, including `ck/`):
  md5 132/132 files ok. Then the instance was terminated (`lambda budget` US$0/h) and the cron removed.
  - All 4 evals (Jcomp s0–s2, Praw s0) had **0 effective errors**.
  - Every training ran its full planned steps with **0 OOM skips**. Training tokens: Jcomp about 136.7M each;
    Praw 161.3M.
- **Primary result (pre-registered):** P − Jcomp, Welch 95% CI over seeds (`results_gpu3/compare.txt`). The per-option
  reader still wins at compute parity:

  | Metric | P − Jcomp | Welch 95% CI |
  |---|---|---|
  | pub15 skill | +0.028 | [+0.015, +0.042] |
  | S1 acc | +0.031 | [+0.010, +0.052] |
  | BTZSC500 F1 | +0.070 | [+0.019, +0.121] |
  | flips perm | −0.253 | — |
  | flips yn | −0.134 | — |

  About half of the step-matched gap (+0.058 on pub15) was compute.
  - **Caveat:** Jcomp matches P in steps and per-step token budget. P's real training tokens are not logged for
    GPU-2, but Praw, with the same per-option batching, used 161M against Jcomp's 137M (about 1.18×).
- **Arm means:**

  | Arm | pub15 skill | S1 acc | BTZSC F1 |
  |---|---|---|---|
  | J | 0.180 | 0.406 | 0.425 |
  | Jsteps | 0.187 | 0.417 | 0.485 |
  | Jcomp | 0.217 | 0.424 | 0.532 |
  | P | 0.245 | 0.455 | 0.602 |
  | Praw (1 seed) | 0.204 | 0.419 | 0.494 |
  | Jraw | 0.175 | 0.413 | 0.350 |

  The reranker init matters a lot for the per-option reader (P − Praw: −0.041 on pub15, one seed). Per-option is still
  better than joint at the same raw init (Praw 0.204 vs Jraw 0.175, step counts differ: Jraw is 2,112 steps).
- **E6 GPU, GPU alone** (1,000 requests, 2 alternating passes; `e6gpu/summary.txt`). This replaces GPU-1's first-pass
  per-option row.

  | 32M reader | median | mean | p95 |
  |---|---|---|---|
  | joint v2 | 10.2–10.5 ms | 13.6–13.9 ms | 32–33 ms |
  | per-option | 10.6 ms | 58–63 ms | 314–352 ms |

  Equal at the median; the per-option reader costs about 4.4× at the mean and about 10× at p95.
- **E6 CPU, AMD EPYC 7J13, 16 threads, GPU idle** (same 80 requests for every system, 0 skipped; `e6cpu/summary.txt`).
  Medians and means in ms:

  | System | median | mean |
  |---|---|---|
  | 17M | 13.5 | 80 |
  | 32M | 21.9 | 139 |
  | 32M per-option | 41.0 | 3,728 |
  | 68M | 53.2 | 384 |
  | 150M | 97.3 | 684 |
  | 400M | 253 | 1,434 |
  | 1B | 579 | 3,122 |
  | reranker-32m (per-option) | 38.5 | 3,750 |
  | DeBERTa-v3-large-zs-v2 | 728 | 12,103 |

  These 80 requests (the first 80 of lat1000) include very long ones, so the means and p95s are dominated by a few
  long requests. Report medians, plus means with this caveat.

### 18.y GPU-4: initialisation 2×2 of the reader ablation (prepared 2026-10-21; not launched)

**Scope.** GPU-3 left the raw-init row of the 2×2 incomplete: Praw has 1 seed, and the only joint raw arm (Jraw,
GPU-2) is not compute-matched (2,112 steps). GPU-4 adds exactly:
1. **JrawComp s0, s1, s2:** joint reader from the raw encoder `jhu-clsp/ettin-encoder-32m`, otherwise Jcomp's flags.
2. **Praw s1, s2:** per-option reader from the raw encoder, GPU-3 Praw s0's flags.

Evals: pub15, btzsc500 and s1_60 only. No E4 (per-option flips are 0 by construction, so the 2×2 does not use flips),
no latency, no other model. Everything lives in `gpu4/`; inputs are GPU-3's (`gpu3/inputs/`, same md5). Nothing has
been launched.

#### 18.y.1 Pre-registration (written before any GPU-4 run)

- **Cells** (rows = init, columns = reader). Every cell has 3 seeds.

  | init | joint, compute-matched | per option |
  |---|---|---|
  | reranker `cross-encoder/ettin-reranker-32m-v1` | **Jcomp** (GPU-3 s0–s2) | **P** (GPU-2 s0–s2) |
  | raw `jhu-clsp/ettin-encoder-32m` | **JrawComp** (GPU-4 s0–s2) | **Praw** (GPU-3 s0 + GPU-4 s1, s2) |

- **Arms.**
  - **JrawComp** = Jcomp with `--base jhu-clsp/ettin-encoder-32m`: `--reader joint`, KD from the Crivo-400M logits
    (α 0.5, gold-only sources as `gold_only`), Brier 0.5, 24,576 tokens per step, max 128 items, max_len 2048, lr 1e-4 /
    head 5e-4, warmup 5%, `--skip_oom`, the same sub150k rows, `--epochs 4 --max_steps N_s` with N_s = P's step count
    for the seed, **6,858 / 6,860 / 6,859** (LR warmup and decay over exactly N_s steps, §18.1), `CUDA_MEMFRAC` 0.6.
    JrawComp is to Jraw what Jcomp is to J.
  - **Praw s1, s2** = Praw s0 (§18.1): `--reader peropt --base jhu-clsp/ettin-encoder-32m`, the same common flags,
    `--epochs 1 --grad_ckpt`, uncapped, alone on the GPU.
- **Evaluation**, exactly as §18.1 without E4: joint = `crivo` engine with `code=` the crivo-32m snapshot c14221dd,
  `--batch 64`; per-option = `PerOptionEngine.batch_call --batch 32`; the grid's scorers (`score_pub.py`, `btzsc/adapter.py`,
  `s1arena/adapter.py`); the error sweep with the GPU free of training, 0 effective errors required before `DONE`.
- **Metrics** (each read separately, as §18.1): pub15 skill (headline), S1-Arena s1_60 accuracy, BTZSC500 macro-F1 (22
  datasets).
- **Contrasts**, each with a Welch 95% CI over seeds (Welch–Satterthwaite df; for two cells this is the ordinary Welch
  test, 3 vs 3):
  - **C1 (primary): Praw − JrawComp**, per option minus joint at raw init.
    - CI entirely above 0: the per-option advantage **holds at raw init** too; it is not created by the reranker init.
    - CI containing 0: **not distinguishable** at raw init.
    - CI entirely below 0: **reversed** at raw init.
  - **C2: P − Praw** (reranker minus raw, per-option reader) and **C3: Jcomp − JrawComp** (reranker minus raw, joint
    reader). Above 0 = the reranker init helps that reader; contains 0 = not distinguishable; below 0 = it hurts.
  - **C4: interaction (P − Praw) − (Jcomp − JrawComp)** = (P − Jcomp) − (Praw − JrawComp), the four cells' variances
    added (Welch–Satterthwaite df).
    - Above 0: the reranker init helps the per-option reader more than the joint one, so part of the per-option
      advantage at reranker init comes from the init.
    - Below 0: the reranker init helps the joint reader more; the per-option advantage is larger at raw init.
    - Containing 0: the interaction is **not resolved** at 3 seeds per cell. It is reported that way, never as "no
      interaction".
  - P − Jcomp (GPU-3's primary) is printed again for reference; its numbers do not change.
- **Multiplicity.** 3 metrics × 4 contrasts, 95% CIs unadjusted, as in §18.1. The text leads with pub15 skill; a claim
  that rests on one metric alone is labelled so.
- **Secondary, descriptive, not decision rules:** the same contrasts on pub15 ECE / Brier, S1 Brier, S1 unseen
  accuracy and BTZSC unseen F1; JrawComp − Jraw (the compute effect at raw init, Welch); training tokens per run.
- **Pooling.** Praw pools GPU-3 s0 with GPU-4 s1, s2, and the four cells come from three boxes (GPU-2/3/4). Code,
  inputs (md5), flags and scorers are identical; a box effect cannot be separated and is named as a caveat.
- **No silent OOM drops** (as §18.1). Every run logs `oom_skips`. A JrawComp seed that drops a batch at the 0.6 cap is
  retrained alone, uncapped (`*.skipped` kept, not reported). Praw keeps `--skip_oom`; its counts are reported.
- **Fallback, decided now.** If the budget stops the run before Praw s2 finishes, C1/C2/C4 are reported with Praw at
  n = 2 and labelled so; nothing else changes.
- **Script.** `gpu4/compare.py --gpu2 results_gpu2 --gpu3 results_gpu3 --gpu4 results_gpu4`. Checked on the Mac: its
  P − Jcomp line reproduces `results_gpu3/compare.txt` to 4 decimals (+0.0283 [+0.0146, +0.0420] etc.), so the
  general Welch code equals scipy's for two cells.

#### 18.y.2 Order (`gpu4/run_gpu4.sh`, one tmux `gpu4`, STATUS `~/results/gpu4/STATUS`, resumable)

0. Input md5 (`gpu4/inputs.md5.box`: sub150k, teacher logits, gold_only, the three eval sets) and the offline build of
   the raw encoder (from its `.bin`); the crivo-32m c14221dd snapshot must hold `crivo.py` and weights.
1. **Praw s1, then Praw s2, each alone.** The script refuses to start one while any training or eval runs.
2. **JrawComp s0 → s1 → s2**, serial, each at the 0.6 cap. One background **eval lane** runs beside them, one eval at a
   time: Praw s1, Praw s2, JrawComp s0, JrawComp s1 (each JrawComp eval waits for its checkpoint's `TRAINED` marker).
   So a per-option eval only ever shares the GPU with a capped joint training. The JrawComp s2 eval starts when the
   last training ends (no training then).
3. A JrawComp seed with skips is retrained alone, uncapped, after the lane finishes, then evaluated alone.
4. Error sweeps (GPU free), scoring, `compare.py` → `compare.txt` / `compare.json`.

Why this order: the Praw trainings must run alone, so they cannot overlap anything. Running them first lets both Praw
evals hide behind the JrawComp trainings, which is the only overlap the rules allow. Joint first would leave the two
Praw evals for the end with nothing to hide behind (about 15 min more).

#### 18.y.3 Time and cost (Lambda A100 40 GB SXM4, US$1.99/h; basis = GPU-3 STATUS and file times)

| Block | Basis (GPU-3 measured) | Wall |
|---|---|---:|
| Push (code about 1.5 MB + about 235 MB inputs and S1 data) + setup + 2-model prefetch | GPU-3: 7 min to the first training with 9 models | 0:10 |
| Praw s1, Praw s2, alone | Praw s0 alone 3,399 s | 1:53 |
| JrawComp s0–s2 at 0.6, each beside the lane | Jcomp beside an eval 911 s (alone 723 s) | 0:46 |
| Lane tail: JrawComp s2 eval alone | lane: Praw eval 13.5 min ×2, joint eval 8.2 min ×2 beside training, all done by about 0:43 of the block; joint eval alone 6.4 min | 0:07 |
| Sweeps, scores, compare | GPU-3: 43 s | 0:01 |
| Fetch (5 checkpoints × 126 MB + outputs) + md5, then terminate | | 0:05 |
| **Total** | | **≈ 3.0 h (2.8–3.3)** |
| **Cost** | | **≈ US$6.0 (5.6–6.6)** |

The ~2.5 h target is not reachable under the rules: the two Praw trainings alone are 1:53, and the joint block cannot
be shorter than about 0:45 with its evals. The only lever is a seed: `PSEEDS=1` (Praw n = 2, the fallback above) saves
about 57 min (≈ 2.1 h, ≈ US$4.2), at the cost of the weakest C1/C2/C4 CIs. Not recommended.

#### 18.y.4 Files added (nothing outside `gpu4/` changed)

- `gpu4/run_gpu4.sh`: the order above. `SERIAL_EVAL=1` (one process at a time) and `SEEDS` / `PSEEDS` are for dry runs
  only; never set them on the box.
- `gpu4/setup_gpu4.sh`: `gpu3/setup_gpu3.sh` with the prefetch cut to `jhu-clsp/ettin-encoder-32m` (`*.bin` included)
  and `tardellirs/crivo-32m@c14221dd`.
- `gpu4/launch.sh`: pushes code (without `stats/`, `e5/`, results), GPU-3's sub150k / teacher / gold_only / eval sets,
  S1-Arena data and code, and the GPU-2 / GPU-3 score files (for `compare.py` on the box); then setup and tmux `gpu4`.
  No E4 files, no lat1000, no P_s0 checkpoint.
- `gpu4/compare.py` (the 2×2 above), `gpu4/eff_errors.py` (copy of GPU-3's), `gpu4/inputs.md5.box` (GPU-3's list minus
  E4, lat1000 and P_s0).

#### 18.y.5 Dry runs (Mac CPU, 2026-10-21)

**1. Control flow (stub Python, no torch).** `$PY` was replaced by a stub that sleeps and writes markers, with 3
requests per eval set. Checked:
- order: Praw s1 → Praw s2 alone (no eval started); then the lane ran Praw s1, Praw s2, JrawComp s0, JrawComp s1 one
  at a time beside JrawComp s0–s2; the JrawComp s2 eval started after the last training;
- every JrawComp training ran with `CUDA_MEMFRAC=0.6`; both Praw trainings ran uncapped;
- **recovery:** a simulated `oom_skips 1` on JrawComp s1 → kept as `*.skipped`, retrained alone uncapped after the lane,
  then evaluated; an injected eval error → swept at batch 1, 0 effective errors, rescored;
- **resume:** a run killed by a `FAILED` line resumed from the same STATUS and finished `ALL DONE`, skipping every
  finished training and eval.

**2. Real code, tiny (Mac CPU, serial).** 50 sub150k rows with sliced teacher logits, 4 requests per eval set,
`SEEDS=0 PSEEDS=1 SERIAL_EVAL=1`, `--limit 40 --max_steps 3 --max_tokens 4096 --max_len 1024 --workers 0`.
**Peak RSS 1.26 GB.** Passed:
- md5 list and the offline build of `jhu-clsp/ettin-encoder-32m` from `.bin` (32,033,665 parameters);
- Praw s1 (`--grad_ckpt`) and JrawComp s0 (`--epochs 4 --max_steps`), `oom_skips` 0, `tokens_total` logged;
- the Praw eval through `PerOptionEngine.batch_call`, the JrawComp eval through `crivo` + `code=`, all three scorers,
  the sweep, and `compare.py` reading GPU-2 / GPU-3 / GPU-4 together (Praw pooled GPU-3 s0 with the dry s1);
- **flags:** the saved `args` of the dry Praw s1 equal GPU-3 `Praw_s0`'s exactly, and the dry JrawComp s0's equal
  GPU-3 `Jcomp_s0`'s except `base` (dry-run overrides `limit/max_steps/max_tokens/max_len/workers` aside).

**GPU-box only** (not run on the Mac): real throughput and memory (JrawComp at the 0.6 cap with a per-option eval
beside it; Praw alone), bf16, `setup_gpu4.sh` and `launch.sh` themselves. JrawComp's memory at the 0.6 cap with a per-option eval beside it is
unmeasured; if it drops batches, the SKIPS path above retrains it alone.

**Launch** (after you start the instance yourself; the script never creates or terminates one):
`caffeinate -dims bash pareto/paper/exp/gpu4/launch.sh <user>@<gpu-host>`. Then:
- monitor with `… status` every 10 min;
- on `ALL DONE`, run `… fetch` (it writes `paper/exp/results_gpu4/` and checks md5), then on the Mac
  `python3 paper/exp/gpu4/compare.py --gpu2 paper/exp/results_gpu2 --gpu3 paper/exp/results_gpu3 --gpu4 paper/exp/results_gpu4 --out paper/exp/results_gpu4/compare.json`,
  terminate, and check `lambda budget`.
