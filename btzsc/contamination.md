| Dataset | Test items | v2 family (17M-1B): class | rows same source | overlap items | 32M v1 | 150M v1 | Overlap sources (v2) |
|---|---:|---|---:|---:|---|---|---|
| amazonpolarity | 10,000 | seen-train-split | 297 | 0 (0.0%) | seen-train-split (0) | unseen (0) |  |
| imdb | 10,000 | seen-train-split | 541 | 1 (0.0%) | seen-train-split (1) | unseen (0) | t3/imdb |
| appreviews | 4,000 | unseen | 0 | 0 (0.0%) | seen-train-split (5) | unseen (0) |  |
| yelpreviews | 10,000 | seen-train-split | 297 | 3 (0.0%) | seen-train-split (3) | unseen (0) | t3/yelp_review_full/yelp_review_full |
| rottentomatoes | 1,066 | seen-train-split | 525 | 3 (0.3%) | seen-train-split (3) | unseen (0) | t3/glue/sst2 |
| financialphrasebank | 690 | seen-test-overlap | 514 | 77 (11.2%) | seen-test-overlap (77) | unseen (0) | t3/financial_phrasebank/sentences_allagree, t3/auditor_review |
| emotiondair | 2,000 | seen-train-split | 297 | 0 (0.0%) | seen-train-split (0) | unseen (0) |  |
| empathetic | 2,542 | unseen | 0 | 0 (0.0%) | unseen (0) | unseen (0) |  |
| banking77 | 3,080 | seen-test-overlap | 18,744 | 33 (1.1%) | seen-test-overlap (52) | seen-test-overlap (52) | bekko/banking77, src400/exact/banking77, pub/banking77 |
| biasframes_intent | 3,648 | seen-test-overlap | 294 | 43 (1.2%) | seen-test-overlap (45) | unseen (stray overlap) (1) | t3/SBIC_Disagreement, t3/hate_speech18, bekko/measuring_hate |
| biasframes_offensive | 3,838 | seen-test-overlap | 294 | 45 (1.2%) | seen-test-overlap (48) | unseen (stray overlap) (2) | t3/SBIC_Disagreement, t3/hate_speech18, bekko/measuring_hate |
| biasframes_sex | 4,404 | seen-test-overlap | 294 | 51 (1.2%) | seen-test-overlap (53) | unseen (stray overlap) (2) | t3/SBIC_Disagreement, t3/hate_speech18, bekko/measuring_hate |
| massive | 2,974 | seen-test-overlap | 2,892 | 21 (0.7%) | seen-test-overlap (34) | seen-test-overlap (34) | bekko/massive, src400/exact/clinc150, t2/opaque/bekko/massive |
| agnews | 7,600 | seen-train-split | 5,824 | 22 (0.3%) | seen-test-overlap (45) | seen-test-overlap (45) | bekko/synthetic_relevance__agnews, bekko/agnews, bekko/laya__ag_news |
| yahootopics | 45,002 | seen-train-split | 293 | 0 (0.0%) | seen-train-split (0) | unseen (0) |  |
| trueteacher | 8,955 | unseen | 0 | 0 (0.0%) | unseen (stray overlap) (2) | unseen (0) |  |
| manifesto | 17,018 | seen-train-split | 282 | 2 (0.0%) | seen-train-split (5) | unseen (stray overlap) (3) | t3/Pol_NLI |
| capsotu | 3,355 | unseen | 0 | 0 (0.0%) | unseen (0) | unseen (0) |  |
| wikitoxic_toxicaggregated | 10,000 | seen-train-split | 297 | 14 (0.1%) | seen-train-split (16) | unseen (stray overlap) (2) | t3/jigsaw_toxicity, t1/gutenberg-poetry |
| wikitoxic_obscene | 8,691 | seen-train-split | 297 | 14 (0.2%) | seen-train-split (15) | unseen (stray overlap) (4) | t3/jigsaw_toxicity, bekko/synthetic_relevance__trivia_qa, bekko/taskmaster |
| wikitoxic_threat | 5,211 | seen-train-split | 297 | 7 (0.1%) | seen-train-split (7) | unseen (stray overlap) (1) | t3/jigsaw_toxicity, t2/nota_present/bekko/taskmaster |
| wikitoxic_insult | 8,427 | seen-train-split | 297 | 14 (0.2%) | seen-train-split (16) | unseen (stray overlap) (3) | t3/jigsaw_toxicity, t2/nota_present/bekko/taskmaster |

Counts: v2: seen-test-overlap 6, seen-train-split 12, unseen 4; 32Mv1: seen-test-overlap 7, seen-train-split 12, unseen 3; 150Mv1: seen-test-overlap 3, unseen 19
