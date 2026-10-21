# 095 — Steering vectors: "Golden-Gate my model" (Eiffel Tower edition)

**Verdict (pre-registered rule): INCONCLUSIVE — leaning "no sweet spot".** None of the 30 pre-registered
(layer, strength) cells cleared both bars (topic-rate CI lower bound > 80% *and* coherence-loss CI upper bound
< 20%). The exploratory fine grid (12 more cells, not used for the verdict) didn't clear them either. The rule
can't say DISPROVEN because one cell, layer 24 at s=3, doesn't fail either bar with confidence: its topic-rate
CI reaches 82%.

**Headline number:** At strength 3 on layer 12, Qwen3-4B talked about the Eiffel Tower in **100% of 50 answers
to unrelated questions**, and 68% of those answers still read as fluent text. But **0% of them actually answered
the question**, and MMLU fell from 68% to 25%, which is chance.

**Plain English:** You really can "Golden-Gate" a 4B model with one vector computed from 200 prompts. The
dial runs from "ignores the vector" to "talks only about the Eiffel Tower" to "gibberish," and it covers all
three within a strength range of about 2 to 4. There's no setting where the model drops Eiffel references into
answers that are still good. By the time it reliably mentions the tower, it has stopped doing the task.

## Setup (short; full spec in README.md)
- Qwen3-4B (bf16, chat template, thinking off). The steering vector is the mean residual activation on
  100 Eiffel prompts minus the mean on 100 neutral prompts, excluding the template's first tokens. We add
  s·v at every position after decoder layer 12, 18 or 24 (of 36). The vector norm is about 0.22× the
  residual norm.
- 50 fixed neutral eval prompts, greedy decoding, 128 new tokens. Strengths 0–16, plus an exploratory
  fine grid at 2.25/2.5/2.75/3.5.
- Topic: regex for eiffel|paris|gustave|… An LLM judge agreed closely (see `results/metrics.csv`).
  Coherence: perplexity under the unsteered model ≤ 95th percentile of baseline outputs, and repeated-trigram
  fraction ≤ 0.20. MMLU: 100 fixed items with the vector active. 95% bootstrap CIs over prompts.
- **Post-hoc metric** (added after seeing that "coherent" outputs ignored the question, so it isn't part of
  the decision rule): the unsteered Qwen3-4B judges "Does the response actually address the user's question?"

## Main results (selected cells; full table in `figures/table.png` / `results/metrics.csv`)

| Layer | s | Topic % [95% CI] | Coherent % | Coherence loss % [CI] | Answers Q %† | MMLU % |
|---|---|---|---|---|---|---|
| – | 0 | 0 [0, 0] | 94 | 0 | 100 | 68 |
| 12 | 2 | 6 [0, 14] | 98 | −4 [−12, 0] | 100 | 58 |
| 12 | 2.5* | 66 [54, 78] | 88 | +6 [−7, +18] | 62 | 26 |
| 12 | 3 | **100 [100, 100]** | 68 | +28 [+13, +42] | **0** | **25** |
| 12 | 4 | 92 [84, 98] | 10 | +89 [+80, +98] | 2 | 25 |
| 18 | 2 | 34 [22, 46] | 84 | +11 [+2, +20] | 74 | 69 |
| 18 | 2.5* | 88 [78, 96] | 30 | +68 [+54, +81] | 22 | 66 |
| 18 | 3 | 100 [100, 100] | 0 | +100 | 6 | 55 |
| 24 | 2.75* | 42 [28, 56] | 82 | +13 [+4, +23] | 80 | 66 |
| 24 | 3 | 70 [58, 82] | 70 | +26 [+13, +38] | 62 | 66 |
| 24 | 3.5* | 96 [90, 100] | 26 | +72 [+59, +85] | 14 | 69 |
| 24 | 8–16 | 0 | 0 | +100 | 0 | 49–53 |

\* exploratory fine grid (not used for verdict). † post-hoc judge.

## What we saw
1. **It's a cliff, not a slope.** At every layer the topic rate goes from about 0% to about 100% within roughly
   one unit of strength (s≈2→3). The useful window is narrow.
2. **Early-layer steering hits capability hardest.** Layer 12 at s≥2.5 takes MMLU to chance (25%) even while
   the text still reads fluently. Layer 24 keeps MMLU at 66–70% up to s=4, because steering late in the network
   mostly changes *what the model says*, not how it reasons over the multiple-choice question.
3. **"Coherent" isn't the same as "useful."** At L12 s=3 the outputs are fluent Wikipedia-style Eiffel
   paragraphs ("…is a famous landmark in Paris, France. The Eiffel Tower is a symbol of the city…"). The
   perplexity metric scores them as *more* fluent than baseline (median PPL 2.1). They only fail the
   coherence bar because they repeat themselves. They answer none of the questions.
4. **The best compromise** is layer 24 at s=2.75–3. There, 42–70% of answers mention the tower, 62–80% still
   answer the question, coherence loss is 13–26%, and MMLU is unchanged. The prettiest examples come from
   L18 s=2.5, e.g. "3. **Use of the Eiffel Tower Clock** – for timekeeping in Paris" as one of three sleep tips.
5. At very high strengths (s≥8–12) the regex topic rate falls back to 0. That's because the output collapses
   into "Tower ( ( ( E (" or "!!!!" and no longer contains a full keyword. It isn't a return to normal.

## Caveats
- One model, one concept (the Eiffel Tower), one vector recipe (unnormalised mean difference, added at all
  positions). Other recipes, such as steering only generated tokens, normalising, or using several layers at
  lower strength, may widen the window.
- n=50 prompts and 100 MMLU items, so CIs are about ±10–15 points. The 0%/100% cells have degenerate bootstrap CIs.
- The coherence metric (perplexity threshold plus trigram repetition) is a proxy. It rewards on-topic
  boilerplate, which is why we added the post-hoc "answers the question" judge. That judge is the same
  model family and is not pre-registered.
- The regex counts "Paris" as on-topic. The strict "eiffel"-only rate is a bit lower (see table).
- A pilot vector that averaged over all tokens (including the attention-sink token) was dominated by a
  length artefact. We fixed this before computing any metrics (documented in README).

## Figures
- `figures/topic_coherence_vs_strength.png`: topic rate, coherence, both, and answers-question vs strength, per layer.
- `figures/steer_vs_coherence.png`: the steer-rate vs coherence trade-off curve with the hypothesised sweet-spot box (empty).
- `figures/mmlu_vs_strength.png`: capability cost.
- `figures/table.png`: full results table. `figures/examples_L12.png`, `figures/examples_L24.png`: 5 outputs per strength.
- `figures/colab_run.txt`: nvidia-smi and timing log lines from the Colab runs.

## Timing
Estimated 105 min vs actual ~112 min (incl. ~30 min idle after a sandbox reboot). GPU: NVIDIA L4 (Colab,
2 VMs because the first was lost mid-run). About 2.3 CU used (estimate: ~1.5 L4-hours at 1.54 CU/hr).
