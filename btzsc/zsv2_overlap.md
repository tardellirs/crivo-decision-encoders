# BTZSC × MoritzLaurer zeroshot-v2.0: training-data overlap

Read-only audit, 2026-10-21. Question: did `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` and
`MoritzLaurer/ModernBERT-large-zeroshot-v2.0` (BTZSC macro-F1 0.800 / 0.776 on our GPU-1 runs) train on BTZSC datasets?

**Short answer: yes, all 22.** Both models are non-"-c" models. Their training mix includes a train split of
**every one of the 22 BTZSC datasets**, at up to 500 texts per class. It also contains the **same hypothesis strings**
that BTZSC uses as label verbalizers. On 2 datasets, actual BTZSC test items are in training: yelpreviews (4.5%) and
agnews (1.7%). trueteacher shares the source article with a training item for 6.1% of its test items. So no
"BTZSC restricted to zs-v2-unseen datasets" subset exists. The comparisons that remain are in §5.

## 1. Sources (pinned)

| Id | Source |
|---|---|
| [C1] | DeBERTa card, `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` @cf44676, README "Training data" L28–39 and "Metrics" L98–100: https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0/blob/cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2/README.md |
| [C2] | ModernBERT card, `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` @a51e07b, README L18–19: "fine-tuned on the same dataset mix as the `zeroshot-v2.0` models". Its results table (L32–35) reports F1 on all 22 BTZSC datasets plus 6 more. There is **no** `ModernBERT-large-zeroshot-v2.0-c` (HF API answers 401). |
| [C3] | Laurer dataset list, `v1_human_data/datasets_overview.csv` @7f82e4a, the commit the card links to: https://github.com/MoritzLaurer/zeroshot-classifier/blob/7f82e4ab88d7aa82a4776f161b368cc9fa778001/v1_human_data/datasets_overview.csv |
| [N1] | v2 training notebook, `v2_synthetic_data/synth_train_eval.ipynb` @e680edc. Cell 3: `ONLY_COMMERCIALLY_FRIENDLY_DATA = False`. Cell 9: synthetic subsets `mixtral_written_texts_for_tasks_v4` and `mixtral_refinedweb_nli`, 50k each. Cell 11: non-commercial branch `load_dataset("MoritzLaurer/dataset_train_nli")`; the commercial branch is `mnli_fever_clean`. Cell 16: `--dataset_name_heldout none` keeps every task. |
| [N2] | `v1_human_data/3_data_formatting_universal_nli.ipynb`, cell 15: `n_max_sample_per_class = 500`, `n_max_sample_per_task = 5_000`. Cell 21: test files are downsampled to 5,000 per class. |
| [N3] | `v1_human_data/1_data_harmonization_huggingface.ipynb`. Cell 38 (yelp test): `df_yelpreviews_test = df_yelpreviews_train[...]`, so the "test" file is built from the **train** frame (a bug). Cell 76 (agnews): `df_agnews_train = pd.concat([train, test])`. Cells 85–89 (trueteacher): 5 T5 summaries per CNN/DM article, then a random 80/20 split with `train_test_split` at row level. |
| [D1] | `MoritzLaurer/dataset_train_nli` @1e00964: 1,018,733 rows, the v1.1 human-data training set. Counted locally (table below). |
| [P1] | Laurer et al., arXiv 2312.17543, p.4: "maximum 500 texts per class and maximum 5000 texts per dataset"; "80-20 train-test split if no test-set exists". p.6: "deberta-v3-zeroshot-v1.1-all-33 has seen up to 500 examples for each class in each dataset". |
| [B1] | BTZSC paper, arXiv 2603.11991 (ICLR 2026). p.4 §3: "BTZSC reuses a subset of the dataset pool compiled by Laurer et al. (2023)". p.4, footnote 2: "We mitigate this by checking publicly documented … supervised training data and avoiding models that explicitly list our datasets as supervised training targets". p.15 App. A: "we adopt their task definitions and label verbalizers". p.16 A.2: "The exact templates and class names are inherited from the release of Laurer et al. (2023)". p.19, Table 4: FT data of every baseline. The NLI cross-encoders are the author's own models, trained on MNLI/ANLI/WANLI/FEVERNLI/LingNLI only (p.7). |
| [B2] | BTZSC dataset card `btzsc/btzsc` @fef2a2a, and harness README (`IliasAarab/btzsc`), "Benchmark Protocol": "No BTZSC-label training or tuning on evaluation datasets". |

**What each name trains on** ([C1] L29–39, [N1]):

| Model | Synthetic Mixtral data | MNLI, FEVER-NLI | ANLI, WANLI, LingNLI | 28 Laurer classification datasets (train splits, ≤500 per class) |
|---|---|---|---|---|
| `*-zeroshot-v2.0-c` (commercial) | yes | yes | no | **no** |
| `deberta-v3-large-zeroshot-v2.0` (**what we ran**) | yes | yes | yes | **yes**: all `used_in_v1.1==TRUE` rows of [C3] |
| `ModernBERT-large-zeroshot-v2.0` (**what we ran**) | same mix as v2.0 [C2] | | | **yes** (no -c variant exists) |

The DeBERTa card's own metrics table ([C1] L65–100) has two numbers per non-c model. The first comes from a run
"without any data from these 28 datasets". The bracketed number comes from "the final run including up to 500
training data points per class from each of the 28 datasets". The **released** non-c checkpoint is that final run:
the notebook uses `heldout none` [N1], and the ModernBERT card reports trained-level scores [C2].

## 2. The 22 BTZSC datasets × zs-v2 training

Class key:
- **(a)** the dataset itself (a train split) is in non-c zs-v2 training;
- **(b)** a sibling task is in training;
- **(c)** not found.

For the non-c models, every row is (a). The "-c (b/c)" column applies only to the commercial models, which we did
not run. It is based on the 543 task descriptions of `mixtral_written_texts_for_tasks_v4`, searched with a keyword
regex.

Columns:
- **[C3] row:** the CSV line where `used_in_v1.1==TRUE`.
- **zs-v2 train texts:** unique texts for that `task_name` in [D1] (rows are 2× that: one entailment pair and one not-entailment pair).
- **Test items in zs-v2 train:** BTZSC unique test texts whose normalized text exactly equals a [D1] text from any task.
  Normalization is NFKC, lowercase, and non-word runs collapsed to one space. The trueteacher entry instead counts items
  with ≥80% of their word 8-grams in a single training text of the same task.
- **Hyp. in train:** BTZSC hypotheses that appear verbatim among the [D1] hypotheses of the same task.

| # | BTZSC dataset | Source in Laurer pool ([C3] row) | Non-c class | zs-v2 train texts [D1] | Test items in zs-v2 train | Hyp. in train | -c class (synthetic sibling) |
|---:|---|---|---|---:|---|---|---|
| 1 | amazonpolarity | amazon_polarity (L12) | (a) | 1,000 | 0 / 10,000 | 2/2 | (b) "Classify customer reviews as positive, negative, or neutral" |
| 2 | imdb | IMDB (L13) | (a) | 1,000 | 6 / 9,956 (0.06%) | 2/2 | (b) same review-sentiment tasks |
| 3 | appreviews | app_reviews (L14) | (a) | 1,000 | 0 / 4,000 | 2/2 | (b) "Classify the sentiment expressed in a text about a software product, such as user reviews" |
| 4 | yelpreviews | yelp_reviews (L15) | (a) | 949 | **446 / 10,000 (4.46%)**, see §3 | 2/2 | (b) review sentiment |
| 5 | rottentomatoes | rotten_tomatoes (L11) | (a) | 1,000 | 0 / 1,066 | 2/2 | (b) review sentiment ("art critique … positive, neutral, or negative") |
| 6 | financialphrasebank | financial_phrasebank (L9) | (a) | 1,262 | 0 / 690 | 3/3 | (b) "Classify financial news articles based on their sentiment: positive, neutral, or negative" |
| 7 | emotiondair | emotion6_twitter = dair-ai/emotion (L25) | (a) | 2,518 | 2 / 2,000 (0.1%) | 6/6 | (b) "Identify the tone of social media posts as happy, sad, angry, or surprised" |
| 8 | empathetic | empathetic_dialogues (L28) | (a) | 2,113 | 0 / 2,542 | 32/32 | (b, weak) 4–6-way emotion tasks only |
| 9 | banking77 | banking77 (L24) | (a) | 4,881 | 18 / 3,080 (0.58%) | 72/72 | (b) "Identify the intent of customer support queries…" |
| 10 | biasframes_intent | social_bias_frames (L5) | (a) | 1,000 | 22 / 3,648 (0.60%; 6 from the same task, the rest from sibling biasframes tasks) | 2/2 | (c) no toxicity or offense task found |
| 11 | massive | massive (L23) | (a) | 4,959 | 15 / 2,970 (0.51%) | 59/59 | (b) "Classify the intent of a user's query in a … chatbot conversation" |
| 12 | agnews | ag_news (L29) | (a) | 2,000 | **126 / 7,600 (1.66%)**, see §3 | 4/4 | (b) "Categorizing articles based on their topic: politics, business, or culture" |
| 13 | yahootopics | yahoo_answers (L30) | (a) | 5,000 | 0 / 50,000 | 10/10 | (b, weak) "Classify online discussions based on the topic" |
| 14 | trueteacher | trueteacher (L21) | (a) | 1,000 | 0 exact. **547 / 8,955 (6.1%) share the full article** with a training item (different summary), see §3 | 2/2 | (b) MNLI/FEVER-NLI (entailment and fact verification) |
| 15 | manifesto | manifesto (L36; v1.1 only) | (a) | 4,910 | 188 / 16,678 (1.13%) | 56/56 | (b, weak) generic politics/economics topic tasks |
| 16 | capsotu | cap_sotu (L37; v1.1 only) | (a) | 2,324 | 0 / 3,355 | 21/21 | (b, weak) as manifesto |
| 17 | biasframes_offensive | social_bias_frames (L5) | (a) | 1,000 | 22 / 3,838 (0.57%; 4 same task) | 2/2 | (c) |
| 18 | biasframes_sex | social_bias_frames (L5) | (a) | 1,000 | 26 / 4,404 (0.59%; 5 same task) | 2/2 | (c) |
| 19 | wikitoxic_toxicaggregated | wiki_toxic (L17) | (a) | 1,000 | 13 / 10,000 (0.13%) | 2/2 | (c) |
| 20 | wikitoxic_obscene | wiki_toxic (L17) | (a) | 1,000 | 11 / 8,691 (0.13%) | 2/2 | (c) |
| 21 | wikitoxic_threat | wiki_toxic (L17) | (a) | 920 | 8 / 5,211 (0.15%) | 2/2 | (c) |
| 22 | wikitoxic_insult | wiki_toxic (L17) | (a) | 1,000 | 5 / 8,427 (0.06%) | 2/2 | (c) |

**Totals, non-c:**
- 22 of 22 datasets are class (a);
- 22 of 22 have all BTZSC hypotheses in training verbatim (289 of 289 hypotheses);
- 15 of 22 have at least one test item in training: 14 by exact text, plus trueteacher by shared article.

The 7 datasets with no item-level hit are amazonpolarity, appreviews, rottentomatoes, financialphrasebank,
empathetic, yahootopics and capsotu. They are still train-split-exposed.

**Training also has 6 Laurer datasets that are not in BTZSC:** emocontext, wikitoxic_identityhate, hateoffensive,
hatexplain, spam and wellformedquery.

**Totals, -c:**
- (b): 15;
- (c): 7 (biasframes ×3 and wikitoxic ×4).

The -c count is a coarse keyword reading of the synthetic task list. `mixtral_refinedweb_nli` was not inspected.

## 3. Item-level exposure: how it arises

- **yelpreviews.** All 10,000 BTZSC "test" reviews are in HF `yelp_review_full` **train**: we checked 10,000 of 10,000
  against the train split and found only 3 in its test split. This follows from the Laurer bug in [N3] cell 38: his
  "test" file is a sample of his own 20k train pool. BTZSC took that file, capped at 5k per class [N2]. zs-v2 took
  its ≤500 per class from the same pool. As a result, 446 of the 949 yelp texts in zs-v2 training are BTZSC test
  items. With a binary task at macro-F1 0.988, this hardly moves the score. It is still direct test exposure.
- **agnews.** BTZSC test equals HF test (7,599 of 7,599 unique texts found). Laurer pooled HF train and HF test before
  sampling his training texts ([N3] cell 76), so 126 BTZSC test articles are in zs-v2 training.
- **trueteacher.** Laurer split rows with a random 80/20 `train_test_split`, and each CNN/DM article has up to 5
  model summaries ([N3] cells 85–89). 547 BTZSC test items therefore share their full article with a training item,
  with a different summary.
- **The rest** (≤1.1%) are natural duplicates in the source corpora: banking77, massive and manifesto repeat
  sentences, and the SBIC posts recur across biasframes tasks.

Script: `btzsc/zsv2_overlap.py`. Raw counts: `btzsc/zsv2_overlap.json`. The script ran on a laptop CPU in 27 s; the downloaded
data was not kept.

## 4. What BTZSC itself says about baselines

- BTZSC ([B1] footnote 2) defines zero-shot as "no fine-tuning or labeled examples from BTZSC tasks". It says the
  authors avoided "models that explicitly list our datasets as supervised training targets". Table 4 lists FT data
  for every baseline. None of them trains on BTZSC datasets: the NLI models use MNLI/ANLI/WANLI/FEVERNLI/LingNLI,
  and the rerankers and embedders use generic pair data.
- **No MoritzLaurer zeroshot model appears in BTZSC.** This is consistent with that rule, since the zs-v2 cards list
  the datasets as training data. The paper does not name them as excluded.
- BTZSC's test splits and verbalizers come straight from Laurer's release ([B1] pp.4, 15–16). The yelp bug in §3 is
  therefore inherited by the benchmark.
- BTZSC's protocol line ([B2]) is "No BTZSC-label training or tuning on evaluation datasets". By that rule, both
  zs-v2 non-c models are **not zero-shot on BTZSC**, so they cannot be compared like-for-like with the published
  BTZSC baselines (~0.59 for the NLI models).

Our measured scores agree. On the 22 datasets, our DeBERTa-zs-v2 BTZSC run averages **0.800**. The card's
held-out-run numbers for the same datasets average **0.704**, and its trained-run numbers average **0.842** ([C1]
L65–95). Our run is closer to the trained run on 13 of 22 datasets. Examples (held-out → trained, then ours):

| Dataset | Card held-out run | Card trained run | Our BTZSC run |
|---|---:|---:|---:|
| trueteacher | 0.538 | 0.846 | 0.854 |
| financialphrasebank | 0.582 | 0.913 | 0.908 |
| biasframes_intent | 0.687 | 0.891 | 0.898 |
| biasframes_offensive | 0.656 | 0.883 | 0.893 |

ModernBERT-zs-v2's own card reports trained-level F1 on these datasets [C2], for example trueteacher 0.821.

## 5. Restricted comparisons (our GPU-1 scores, `results_gpu/btzsc/*.score.json`)

**A "zs-v2-unseen" BTZSC subset is empty (0 of 22).** The subsets below are the useful replacements. All values are
macro-F1, the mean over datasets.

Subsets:
- **CrivoV2-unseen-4**: appreviews, empathetic, trueteacher, capsotu. These are unseen by the Crivo v2 family but
  **seen** (train split) by zs-v2.
- **150Mv1-unseen-19**: all except banking77, massive and agnews.
- **zs-v2 no-item-hit-7**: amazonpolarity, appreviews, rottentomatoes, financialphrasebank, empathetic, yahootopics,
  capsotu. These are still train-split-exposed for zs-v2.

| Model | All-22 | CrivoV2-unseen-4 | CrivoV2-seen-18 | 150Mv1-unseen-19 | zs-v2 no-item-hit-7 |
|---|---:|---:|---:|---:|---:|
| Crivo-17M | 0.513 | 0.469 | 0.523 | 0.501 | 0.519 |
| Crivo-32M (v2) | 0.603 | 0.514 | 0.622 | 0.586 | 0.617 |
| Crivo-32M v1 | 0.514 | 0.484 | 0.520 | 0.508 | 0.572 |
| Crivo-68M | 0.667 | 0.553 | 0.693 | 0.656 | 0.690 |
| Crivo-150M (v2) | 0.671 | 0.564 | 0.695 | 0.660 | 0.721 |
| Crivo-150M v1 | 0.665 | 0.570 | 0.687 | 0.650 | 0.677 |
| Crivo-400M | 0.699 | 0.636 | 0.713 | 0.691 | 0.718 |
| Crivo-1B | 0.751 | 0.686 | 0.765 | 0.738 | 0.760 |
| DeBERTa-v3-large-zs-v2.0 | 0.800 | 0.744 | 0.813 | 0.813 | 0.799 |
| ModernBERT-large-zs-v2.0 | 0.776 | 0.717 | 0.789 | 0.789 | 0.785 |
| Qwen3-0.6B (letter) | 0.343 | 0.451 | 0.319 | 0.362 | 0.420 |
| Gemma-3-1B-it (letter) | 0.370 | 0.265 | 0.394 | 0.402 | 0.373 |
| Ettin reranker-32M (raw) | 0.351 | 0.358 | 0.349 | 0.363 | 0.348 |

How to read this:
- The only exposure-symmetric framing is "**both** model families saw these datasets' train splits": CrivoV2-seen-18,
  where Crivo-1B scores 0.765 and DeBERTa-zs-v2 0.813.
- On CrivoV2-unseen-4 the exposure is **asymmetric** in zs-v2's favour: Crivo-1B 0.686 vs DeBERTa-zs-v2 0.744, a
  gap of 0.058 even though Crivo never saw these datasets and zs-v2 did.
- Per dataset, Crivo-1B beats both zs-v2 models on banking77 (0.856 vs 0.607 / 0.562) and massive (0.762 vs 0.650 /
  0.625). It also beats DeBERTa-zs-v2 on biasframes_sex (0.765 vs 0.735). These datasets are in zs-v2 training too.

## 6. Suggested caveat for the paper (one sentence)

> The MoritzLaurer `zeroshot-v2.0` checkpoints we compare against (non-commercial variants) were fine-tuned on
> training splits of all 22 BTZSC datasets (up to 500 examples per class) with BTZSC's exact label verbalizers
> (Laurer et al., 2023; model cards), so their BTZSC scores are supervised few-shot rather than zero-shot and are
> reported as an upper reference, not as a like-for-like baseline.
