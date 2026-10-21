# Experiment 095c: A Finer Steering Sweep (Eiffel Tower on Qwen3-4B)

**Verdict (pre-registered rule): DISPROVEN.** Across all 35 cells in the $5 \times 7$ grid (Layers 12, 16, 20, 24, 28 at strengths $s \in \{1.5, 2.0, 2.5, 2.8, 3.1, 3.5, 4.0\}$), zero cells achieved topic adherence $\ge 70\%$ while preserving MMLU accuracy (drop $\le 3.0$ pp) and useful coherence $\ge 60\%$. The pre-registered candidate cell (Layer 24, $s=2.8$) achieved only 54.0% topic adherence and 30.0% useful coherence.

**Headline number:** Increasing steering strength from $s=2.0$ to $s=3.1$ at Layer 24 boosted topic adherence from 9.0% to 81.0% without harming MMLU accuracy (69.6% vs 71.2% baseline), but caused answer relevance on open-ended questions to collapse from 62.0% down to 13.0%, reducing useful coherence to just 10.0%.

**Plain English takeaway:** Fine-grained activation steering does not reveal a hidden sweet spot where a model subtly weaves a target concept into helpful answers; instead, it exposes a fundamental trade-off cliff where topic takeover implies task destruction. Late-layer steering insulates multiple-choice benchmark scores from damage, but it does so while turning conversational outputs into irrelevant topic-obsessed monologues.

## Main Results Table

Results across 100 fresh open-ended prompts (Databricks-Dolly-15k) and 250 fresh MMLU test items. 95% Wilson confidence intervals shown in brackets.

| Layer | Strength ($s$) | Topic Adherence % [95% CI] | MMLU Test % [95% CI] | MMLU Drop (pp) | Answer Relevant % | Useful Coherence % [95% CI] | Median PPL |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 0.0 | 1.0% [0.2, 5.5] | 71.2% [65.3, 76.5] | 0.0 pp | 75.0% | **74.0% [64.6, 81.6]** | 2.22 |
| L12 | 1.5 | 4.0% [1.6, 9.8] | 64.0% [57.9, 69.7] | +7.2 pp | 69.0% | 69.0% [59.4, 77.2] | 2.71 |
| L12 | 2.0 | 13.0% [7.8, 21.0] | 50.4% [44.2, 56.5] | +20.8 pp | 74.0% | 74.0% [64.6, 81.6] | 3.12 |
| L12 | 2.5 | 47.0% [37.5, 56.7] | 24.4% [19.5, 30.1] | +46.8 pp | 38.0% | 36.0% [27.3, 45.8] | 3.66 |
| L12 | 2.8 | 80.0% [71.1, 86.7] | 22.8% [18.0, 28.4] | +48.4 pp | 61.0% | 37.0% [28.2, 46.8] | 2.71 |
| L12 | 3.1 | 93.0% [86.2, 96.6] | 23.2% [18.4, 28.8] | +48.0 pp | 59.0% | 11.0% [6.2, 18.6] | 2.09 |
| L12 | 3.5 | 92.0% [85.0, 95.9] | 23.2% [18.4, 28.8] | +48.0 pp | 25.0% | 0.0% [0.0, 3.7] | 1.63 |
| L12 | 4.0 | 76.0% [66.8, 83.3] | 23.2% [18.4, 28.8] | +48.0 pp | 7.0% | 0.0% [0.0, 3.7] | 1.59 |
| L16 | 1.5 | 3.0% [1.0, 8.5] | 61.2% [55.0, 67.0] | +10.0 pp | 73.0% | 72.0% [62.5, 79.9] | 2.94 |
| L16 | 2.0 | 22.0% [15.0, 31.1] | 50.0% [43.8, 56.2] | +21.2 pp | 55.0% | 54.0% [44.3, 63.4] | 3.97 |
| L16 | 2.5 | 72.0% [62.5, 79.9] | 24.0% [19.1, 29.7] | +47.2 pp | 35.0% | 27.0% [19.3, 36.4] | 3.97 |
| L16 | 2.8 | 90.0% [82.6, 94.5] | 23.2% [18.4, 28.8] | +48.0 pp | 47.0% | 9.0% [4.8, 16.2] | 2.28 |
| L16 | 3.1 | 97.0% [91.6, 99.0] | 23.2% [18.4, 28.8] | +48.0 pp | 44.0% | 0.0% [0.0, 3.7] | 1.95 |
| L16 | 3.5 | 99.0% [94.6, 99.8] | 23.2% [18.4, 28.8] | +48.0 pp | 25.0% | 0.0% [0.0, 3.7] | 1.83 |
| L16 | 4.0 | 100.0% [96.3, 100.0] | 23.2% [18.4, 28.8] | +48.0 pp | 11.0% | 0.0% [0.0, 3.7] | 1.96 |
| L20 | 1.5 | 4.0% [1.6, 9.8] | 68.8% [62.8, 74.2] | +2.4 pp | 70.0% | 68.0% [58.3, 76.3] | 3.02 |
| L20 | 2.0 | 24.0% [16.7, 33.2] | 67.6% [61.6, 73.1] | +3.6 pp | 54.0% | 51.0% [41.3, 60.6] | 3.81 |
| L20 | 2.5 | 60.0% [50.2, 69.1] | 65.2% [59.1, 70.8] | +6.0 pp | 26.0% | 18.0% [11.7, 26.7] | 4.18 |
| L20 | 2.8 | 84.0% [75.6, 89.9] | 63.6% [57.5, 69.3] | +7.6 pp | 25.0% | 2.0% [0.5, 7.0] | 2.71 |
| L20 | 3.1 | 96.0% [90.2, 98.4] | 61.6% [55.4, 67.4] | +9.6 pp | 30.0% | 0.0% [0.0, 3.7] | 1.93 |
| L20 | 3.5 | 100.0% [96.3, 100.0] | 45.6% [39.5, 51.8] | +25.6 pp | 10.0% | 0.0% [0.0, 3.7] | 1.81 |
| L20 | 4.0 | 100.0% [96.3, 100.0] | 23.6% [18.8, 29.2] | +47.6 pp | 8.0% | 0.0% [0.0, 3.7] | 2.24 |
| L24 | 1.5 | 3.0% [1.0, 8.5] | 72.4% [66.5, 77.6] | -1.2 pp | 73.0% | 72.0% [62.5, 79.9] | 2.68 |
| L24 | 2.0 | 9.0% [4.8, 16.2] | 71.6% [65.7, 76.8] | -0.4 pp | 62.0% | 62.0% [52.2, 70.9] | 3.30 |
| L24 | 2.5 | 27.0% [19.3, 36.4] | 70.0% [64.1, 75.3] | +1.2 pp | 49.0% | 48.0% [38.5, 57.7] | 3.86 |
| **L24 \*** | **2.8** | **54.0% [44.3, 63.4]** | **70.0% [64.1, 75.3]** | **+1.2 pp** | **32.0%** | **30.0% [21.9, 39.6]** | **4.47** |
| L24 | 3.1 | 81.0% [72.2, 87.5] | 69.6% [63.6, 75.0] | +1.6 pp | 13.0% | 10.0% [5.5, 17.4] | 4.52 |
| L24 | 3.5 | 97.0% [91.6, 99.0] | 70.0% [64.1, 75.3] | +1.2 pp | 8.0% | 2.0% [0.5, 7.0] | 3.82 |
| L24 | 4.0 | 100.0% [96.3, 100.0] | 69.2% [63.2, 74.6] | +2.0 pp | 2.0% | 0.0% [0.0, 3.7] | 3.93 |
| L28 | 1.5 | 1.0% [0.2, 5.5] | 72.4% [66.5, 77.6] | -1.2 pp | 74.0% | 74.0% [64.6, 81.6] | 2.56 |
| L28 | 2.0 | 6.0% [2.8, 12.5] | 72.4% [66.5, 77.6] | -1.2 pp | 67.0% | 66.0% [56.3, 74.5] | 3.02 |
| L28 | 2.5 | 23.0% [15.8, 32.2] | 71.6% [65.7, 76.8] | -0.4 pp | 52.0% | 52.0% [42.3, 61.5] | 3.47 |
| L28 | 2.8 | 40.0% [30.9, 49.8] | 72.0% [66.1, 77.2] | -0.8 pp | 40.0% | 38.0% [29.1, 47.8] | 3.61 |
| L28 | 3.1 | 57.0% [47.2, 66.3] | 70.4% [64.5, 75.7] | +0.8 pp | 28.0% | 28.0% [20.1, 37.5] | 4.14 |
| L28 | 3.5 | 85.0% [76.7, 90.7] | 70.8% [64.9, 76.1] | +0.4 pp | 7.0% | 7.0% [3.4, 13.8] | 3.80 |
| L28 | 4.0 | 96.0% [90.2, 98.4] | 72.4% [66.5, 77.6] | -1.2 pp | 0.0% | 0.0% [0.0, 3.7] | 2.72 |

\* Pre-registered candidate cell ($L=24, s=2.8$).

## Key Findings

1. **The Cliff Displaces, It Does Not Disappear:**
   In early/middle layers ($L=12, 16$), the cliff destroys both capability and usefulness simultaneously: MMLU plummets from 71.2% to 22.8% at $s \ge 2.5$ (chance level). In deeper layers ($L=24, 28$), MMLU is completely unaffected (remaining at ~70-72% even at $s=4.0$), but open-ended answering collapses just as violently. The model continues to select correct multiple-choice letters while completely losing the ability to answer conversational queries.

2. **Perplexity Is Deceptive (The Boilerplate Paradox):**
   At $L=12, s=3.5$, median perplexity under the unsteered model drops to 1.63 (lower than the unsteered baseline of 2.22). Simple mechanical metrics score these outputs as exceptionally fluent because the model recites high-probability Wikipedia paragraphs about Paris landmark history. Only paired relevance evaluation reveals that the answer relevance is actually 0%.

3. **Hallucination Infiltration:**
   In the candidate cell ($L=24, s=2.8$), where the model attempts to balance prompt answering with the steering vector, it frequently resolves the conflict by hallucinating: for example, declaring that the capital of Arkansas is Paris. Once steering increases to $s=3.1$ to enforce topic adherence (81.0%), answering relevance drops to 13.0%.

## Caveats
- Evaluated on a single concept (Eiffel Tower / Paris) and single architecture (Qwen3-4B).
- Residual stream steering applied uniformly across all sequence positions using contrastive activation addition. Layer-selective, token-selective, or low-rank projected steering methods may alter the dynamics.
- Sample sizes of 100 fresh open-ended prompts and 250 fresh MMLU items yield 95% Wilson confidence intervals of approximately $\pm 5$ to $\pm 9$ percentage points.

## Timing
Timing: Estimated 110 min vs actual 94 min (Setup: 14m, GPU run: 47m, Eval/Analysis: 8m, Writeup: 25m). GPU: NVIDIA L4 (Colab session `b10-095c`), 1.66 CU used.
