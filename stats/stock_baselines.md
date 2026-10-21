| Grp | Entry | Base checkpoint | Params | Public | Same-skills | New-domains | Full 0.4 | Rank grp | Board note (tuning) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| A | Prumo | — | — | 20.14 | 15.78 | 5.6 | 12.21 | 46 | Rule-based engine, no trained weights. |
| A | Zev | — | — | 1.55 | 1.79 | 1.89 | 1.91 | 54 | Rule-based engine, no weights; its three submitted variants give identical rows. |
| A | jeff | knowledgator/gliformer-large-v1 | 576M | 8.27 | 6.86 | 1.53 | 4.81 | 52 | — |
| A | Qwen-2.5-1B-RLCD | Qwen/Qwen2.5-1.5B | 1.54B | 4.02 | 6.21 | 5.78 | 6.48 | 51 | — |
| A | Ryotide Qwen v0.3.1 | Qwen/Qwen3.5-4B | 4.00B | 37.28 | 36.67 | 21.17 | 32.46 | 33 | — |
| A | openvons | Qwen/Qwen3-4B-Instruct-2507 | 4.02B | 28.66 | 32.13 | 22.48 | 30.34 | 34 | — |
| A | mini-jev | Qwen/Qwen3-4B-Instruct-2507 | 4.02B | 21.25 | 23.36 | 20.83 | 24.22 | 38 | — |
| A | Jobe | Qwen/Qwen3.5-4B-Base | 4.66B | 32.93 | 34.34 | 22.92 | 31.9 | 33 | — |
| A | SemIf | Qwen/Qwen3.5-4B-Base | 4.66B | 26.5 | 28.04 | 23.43 | 28.33 | 36 | — |
| A | decisio 0.4.0 (Gemma 4 12B) | google/gemma-4-12B-it | 11.96B | 49.72 | 48.44 | 38.26 | 47.93 | 19 | Inference technique on frozen weights; prompt partly chosen on index rows (author-declared). |
| A | JoshuaSP diffusiongemma (open-jev) | google/diffusiongemma-26B-A4B-it | 25.82B | 50.14 | 50.02 | 38.66 | 49.08 | 18 | — |
| A | djev | google/diffusiongemma-26B-A4B-it | 25.82B | 40.89 | 41.63 | 28.78 | 39.18 | 27 | — |
| A | razorback16 openjev diffusiongemma (NVFP4, vLLM) | google/diffusiongemma-26B-A4B-it | 25.82B | 37.75 | 39.74 | 30.42 | 38.83 | 27 | — |
| A | mmastrac diffusiongemma (vLLM PR 57250) | google/diffusiongemma-26B-A4B-it | 25.82B | 32.69 | 33.44 | 31.36 | 35.47 | 30 | — |
| A | simple-jev · Qwen3.8-27B (featherless) | Qwen/Qwen3.8-27B | 27.78B | 56.05 | 55.46 | 47.1 | 56.49 | 13 | — |
| A | Decider chat · Qwen3.6-27B | Qwen/Qwen3.6-27B | 27.78B | 51.57 | 52.15 | 48.87 | 55.35 | 14 | — |
| A | Jevfire | Qwen/Qwen3.8-27B | 27.78B | 49.8 | 46.86 | 32.63 | 44.22 | 23 | — |
| A | reflex Qwen3.8-27B-FP8 (pmscr) | Qwen/Qwen3.8-27B-FP8 | 27.78B | 57.34 | 57.39 | 49.71 | 58.94 | 11 | No training: prompt scaffold and label-logprob readout on the stock Qwen3.8-27B-FP8 checkpoint. |
| A | AnyJev L0 · Gemma-4-31B | google/gemma-4-31B-it | 31.27B | 59.79 | 59.13 | 58.44 | 64.24 | 6 | No training: prompt, restricted label readout, rotation averaging and a label-free running prior on stock gemma-4-31B-it. |
| A | decisio 0.6.0 (Gemma 4 31B) | google/gemma-4-31B-it | 31.27B | 58.09 | 58.5 | 57.24 | 63.27 | 7 | No training: prompt, summed letter-logit readout and fitted temperatures on frozen gemma-4-31B-it. Temperatures fitted on a suite containing 650 Decision Index public-pool questions; base choice and rendering rules made  |
| A | MORJ 1.0.1 (Gemma 4 31B) | google/gemma-4-31B-it | 31.27B | 60.24 | 59.76 | 58.17 | 64.49 | 6 | No training: prompt, label-logit readout with order averaging and an option-count temperature curve on stock gemma-4-31B-it. Method and serving defaults chosen on public index runs; the temperature fit set held 1 CLINC15 |
| A | jev-brain · Gemma-4-31B | google/gemma-4-31B-it | 31.27B | 61.69 | 61.56 | 57.56 | 65.29 | 5 | Inference technique on frozen Gemma 4 31B weights. |
| A | deck31b (Gemma 4 31B) | google/gemma-4-31B-it | 32.68B | 58.19 | 58.6 | 57.66 | 63.54 | 7 | — |
| A | Decider chat · Gemma-4-31B | google/gemma-4-31B | 32.68B | 57.79 | 56.88 | 56.65 | 62 | 8 | — |
| A | decisio 0.4.0 (Qwen3.6-35B-A3B) | Qwen/Qwen3.6-35B-A3B | 35.95B | 48.43 | 49.15 | 43.37 | 50.85 | 17 | Inference technique on frozen weights; prompt partly chosen on index rows (author-declared). |
| B | Jev 1.13 | — | — | 57.96 | 58.04 | 53.67 | 61.26 | 9 | — |
| B | Bekko System One v0 17M | jhu-clsp/ettin-encoder-17m | 17M | 7.44 | 5.89 | 4.61 | 5.72 | 51 | — |
| B | Bekko System One v0 68M | jhu-clsp/ettin-encoder-68m | 68M | 13.01 | 12.62 | 4.98 | 9.99 | 48 | — |
| B | Bekko System One v0 400M | jhu-clsp/ettin-encoder-400m | 400M | 22.45 | 21.55 | 6.2 | 16 | 43 | — |
| B | Laya | — | 421M | 6.19 | 9.08 | 3.69 | 7.21 | 50 | — |
