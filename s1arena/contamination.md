| Domain | Tasks | unseen | seen-train-split | seen-test-overlap | items | items in unseen tasks |
|---|---:|---:|---:|---:|---:|---:|
| basic_cls | 21 | 6 | 6 | 9 | 4,083 | 998 |
| batch | 6 | 5 | 0 | 1 | 1,500 | 1,250 |
| code | 17 | 17 | 0 | 0 | 3,886 | 3,886 |
| dialog | 10 | 7 | 2 | 1 | 2,464 | 1,714 |
| fore | 5 | 5 | 0 | 0 | 920 | 920 |
| judge | 24 | 19 | 1 | 4 | 4,998 | 3,831 |
| know | 30 | 15 | 9 | 6 | 6,000 | 3,000 |
| pro | 25 | 16 | 2 | 7 | 4,333 | 2,832 |
| ret | 11 | 9 | 1 | 1 | 2,165 | 1,749 |
| route | 18 | 16 | 1 | 1 | 3,462 | 3,045 |
| safety | 16 | 11 | 0 | 5 | 2,998 | 2,165 |
| social | 13 | 6 | 3 | 4 | 1,874 | 833 |
| struct | 13 | 13 | 0 | 0 | 2,500 | 2,500 |
| **all** | 209 | 145 | 25 | 39 | 41,183 | 28,723 |

| Task | Items | Class | Overlap | Training sources matched (rows) |
|---|---:|---|---:|---|
| basic_cls/entity_relation/fewnerd | 167 | seen-test-overlap | 1 (0.6%) | bekko/few_nerd (1578), t2/opaque/bekko/few_nerd (945) |
| basic_cls/entity_relation/fewrel | 167 | seen-test-overlap | 2 (1.2%) | t3/few_rel/default (297) |
| basic_cls/intent/clinc150 | 250 | seen-test-overlap | 3 (1.2%) | src400/sister/clinc150/bitext_retail_ecommerce (8850), pub/clinc150 (4490), src400/exact/clinc150 (2436) |
| basic_cls/intent/massive_en | 250 | seen-train-split | 1 (0.4%) | bekko/massive (1614), t2/opaque/bekko/massive (491), t2/nota_present/bekko/massive (292) |
| basic_cls/language_style/pavlick_formality | 167 | seen-test-overlap | 1 (0.6%) | - |
| basic_cls/multilabel/goemotions | 250 | seen-train-split | 0 (0.0%) | bekko/go_emotions (1627) |
| basic_cls/nli/anli_r3 | 250 | seen-test-overlap | 101 (40.4%) | pub/anli (5609), src400/exact/anli (1932), t2/opaque/pub/anli (433) |
| basic_cls/nli/chaosnli | 250 | seen-test-overlap | 2 (0.8%) | - |
| basic_cls/sentiment/sst5 | 167 | seen-test-overlap | 3 (1.8%) | t3/rotten_tomatoes (297) |
| basic_cls/spam_phishing/sms_spam | 167 | seen-test-overlap | 54 (32.3%) | bekko/sms_spam (1589) |
| basic_cls/spam_phishing/spamassassin_nazario | 166 | seen-test-overlap | 13 (7.8%) | - |
| basic_cls/stance_sarcasm_humor/isarcasmeval | 167 | seen-train-split | 0 (0.0%) | pub/isarcasm (3007), src400/exact/isarcasm (2438), t2/opaque/pub/isarcasm (977) |
| basic_cls/topic/dbpedia_l2 | 167 | seen-train-split | 0 (0.0%) | bekko/dbpedia (1550), t2/opaque/bekko/dbpedia (419), t2/keyonly/bekko/dbpedia (322) |
| basic_cls/toxicity/civil_comments | 250 | seen-train-split | 0 (0.0%) | bekko/civil_comments (1549) |
| basic_cls/toxicity/dynahate | 250 | seen-train-split | 0 (0.0%) | t3/dynahate (297) |
| batch/semantic_grouping/airline_complaints | 250 | seen-test-overlap | 2 (0.8%) | - |
| dialog/clarification_need/ambigqa | 250 | seen-test-overlap | 2 (0.8%) | t3/AmbigNQ-clarifying-question (297) |
| dialog/clarification_need/sharc | 250 | seen-train-split | 0 (0.0%) | t3/sharc (297) |
| dialog/end_of_turn/switchboard | 250 | seen-train-split | 0 (0.0%) | t3/pragmeval/switchboard (133) |
| judge/instruction_following/ifeval_ppe | 250 | seen-test-overlap | 2 (0.8%) | - |
| judge/pointwise_rating/helpsteer2 | 250 | seen-test-overlap | 2 (0.8%) | bekko/helpsteer2 (1559) |
| judge/rag_faithfulness/begin | 250 | seen-test-overlap | 2 (0.8%) | - |
| judge/rag_faithfulness/ragtruth | 250 | seen-train-split | 0 (0.0%) | pub/ragtruth (5269), src400/sister/ragtruth/ragbench_expertqa (2340), src400/exact/ragtruth (2249) |
| judge/summary_consistency/faithbench | 167 | seen-test-overlap | 1 (0.6%) | - |
| know/answerability/squad_v2 | 250 | seen-train-split | 0 (0.0%) | bekko/squad2 (1629), bekko/synthetic_relevance__bge_m3__squad (1525), bekko/bge-hn-squad (1401) |
| know/causal/copa | 250 | seen-train-split | 0 (0.0%) | bekko/copa (423), t2/distract/bekko/copa (244), t2/opaque/bekko/copa (244) |
| know/causal/e_care | 250 | seen-train-split | 0 (0.0%) | t3/e-CARE (290) |
| know/citation_support/attributionbench | 250 | seen-test-overlap | 47 (18.8%) | - |
| know/citation_support/wice | 250 | seen-train-split | 0 (0.0%) | t3/wice (296) |
| know/commonsense/commonsense_qa | 167 | seen-train-split | 0 (0.0%) | bekko/commonsense_qa (1633), bekko/commonsense_qa2 (1584), t2/distract/bekko/commonsense_qa (374) |
| know/commonsense/piqa | 167 | seen-train-split | 0 (0.0%) | bekko/piqa (1610), t2/distract/bekko/piqa (545), t2/opaque/bekko/piqa (465) |
| know/commonsense/winogrande | 166 | seen-test-overlap | 17 (10.2%) | pub/winogrande (4463), src400/exact/winogrande (2438), bekko/winogrande (1571) |
| know/fact_verification/scifact | 167 | seen-test-overlap | 39 (23.4%) | t3/scifact_entailment (297) |
| know/fact_verification/vitaminc | 166 | seen-test-overlap | 2 (1.2%) | t3/vitaminc (297) |
| know/factual_judgment/nq_open | 250 | seen-train-split | 0 (0.0%) | bekko/synthetic_relevance__bge_m3__nq (1564), bekko/synthetic_relevance__natural_questions (1563), bekko/bge-hn-nq (1433) |
| know/logic_puzzles/folio | 167 | seen-train-split | 0 (0.0%) | t3/folio (297) |
| know/multihop/hotpotqa_comparison | 167 | seen-train-split | 0 (0.0%) | bekko/synthetic_relevance__bge_m3__hotpotqa (1637), bekko/bge-hn-hotpotqa (1409) |
| know/multihop/strategyqa | 166 | seen-test-overlap | 88 (53.0%) | bekko/strategyqa (1321) |
| know/numeric_comparison/equate | 167 | seen-test-overlap | 23 (13.8%) | t3/equate (175) |
| pro/document_fields/cord | 167 | seen-train-split | 0 (0.0%) | proc/record_aggregation (3316), t2/distract/proc/record_aggregation (605), t2/nota_removed/proc/record_aggregation (412) |
| pro/ecommerce/esci | 167 | seen-test-overlap | 1 (0.6%) | pub/esci (6005), src400/exact/esci (1431), bekko/esci (1354) |
| pro/finance/financial_phrasebank | 167 | seen-test-overlap | 18 (10.8%) | t3/auditor_review (294), t3/financial_phrasebank/sentences_allagree (220) |
| pro/legal/contractnli | 167 | seen-train-split | 0 (0.0%) | src400/sister/contractnli/cuad (6408), pub/contractnli (4605), src400/exact/contractnli (1752) |
| pro/legal/cuad | 167 | seen-test-overlap | 68 (40.7%) | src400/sister/contractnli/cuad (6408) |
| pro/legal/legalbench | 166 | seen-test-overlap | 1 (0.6%) | - |
| pro/medical/pubmedqa | 166 | seen-test-overlap | 1 (0.6%) | bekko/synthetic_relevance__bge_m3__pubmed_qa_labeled (1624), bekko/bge-hn-pubmed_qa_labeled (1380) |
| pro/support_tickets/banking77 | 167 | seen-test-overlap | 1 (0.6%) | src400/sister/banking77/bitext_retail_banking (8850), pub/banking77 (3794), src400/exact/banking77 (2436) |
| pro/support_tickets/cfpb_complaints | 167 | seen-test-overlap | 4 (2.4%) | - |
| ret/context_sufficiency/ms_marco_qna | 166 | seen-train-split | 0 (0.0%) | bekko/synthetic_relevance__bge_m3__msmarco (1579), bekko/bge-hn-msmarco (1394) |
| ret/retrieval_need/freshqa | 250 | seen-test-overlap | 3 (1.2%) | - |
| route/context_pruning/qasper | 250 | seen-train-split | 0 (0.0%) | bekko/qasper (418) |
| route/tool_selection/bfcl_live | 167 | seen-test-overlap | 2 (1.2%) | extra/bfcl-toolace (10547), extra/bfcl-glaive (5292), src400/sister/bfcl/hermes_fc_v1 (3150) |
| safety/harmfulness/aegis2 | 167 | seen-test-overlap | 7 (4.2%) | bekko/aegis2 (1581) |
| safety/harmfulness/beavertails | 167 | seen-test-overlap | 3 (1.8%) | t3/BeaverTails (281) |
| safety/injection/toxicchat_jailbreak | 166 | seen-test-overlap | 20 (12.0%) | t3/toxic-chat/toxicchat0124/jailbreaking (297), t3/toxic-chat/toxicchat0124/toxicity (296) |
| safety/refusal/xstest_prompts | 167 | seen-test-overlap | 4 (2.4%) | - |
| safety/refusal/xstest_responses | 166 | seen-test-overlap | 5 (3.0%) | - |
| social/moral_judgment/ethics_commonsense | 167 | seen-train-split | 0 (0.0%) | bekko/ethics (1593), t2/distract/bekko/ethics (588), t2/opaque/bekko/ethics (420) |
| social/moral_judgment/scruples_anecdotes | 167 | seen-train-split | 0 (0.0%) | t3/scruples/verdict_votes (297) |
| social/moral_judgment/social_chem_101 | 166 | seen-test-overlap | 6 (3.6%) | - |
| social/perspective_labeling/measuring_hate_speech | 166 | seen-test-overlap | 7 (4.2%) | - |
| social/persuasion_humor/anthropic_persuasion | 125 | seen-test-overlap | 23 (18.4%) | t3/persuasion (297) |
| social/persuasion_humor/expun | 125 | seen-test-overlap | 3 (2.4%) | - |
| social/persuasion_humor/nycc_captions | 125 | seen-train-split | 0 (0.0%) | pub/newyorker (4975), src400/exact/newyorker (1329), t2/distract/pub/newyorker (219) |
