# 095 — Steering vectors: "Golden-Gate my model" (Eiffel Tower edition)

## Question
Can a single contrastive activation-addition (CAA) vector make Qwen3-4B talk about the
Eiffel Tower in response to *unrelated* prompts — and is there a strength where it does so
reliably **without** wrecking the model's coherence?

## Hypothesis
There is a "sweet-spot" steering strength at which the model mentions the topic in **>80%** of
answers to neutral prompts while losing **<20%** of its coherence (relative to unsteered).

## Setup (fixed before running)
- **Model:** `Qwen/Qwen3-4B` (36 decoder layers, d=2560), bf16, chat template with `enable_thinking=False`.
- **Steering vector:** for layer L, v_L = mean residual-stream activation (output of decoder layer L,
  averaged over prompt tokens **excluding the first 3 template tokens** `<|im_start|>user\n`) on **100 Eiffel-Tower prompts** minus the same on **100 neutral prompts**
  (`scripts/prompts.py`). The vector is used **unnormalised**; "strength" s means we add `s · v_L`
  to the residual stream at every position (prompt and generated tokens).
  *Pilot correction (before any metrics were computed):* the first version averaged over all tokens; the
  `<|im_start|>` attention-sink token has massive activations (|h|~1000s) and the two prompt sets differ in
  length, so the "vector" was mostly a length artefact (|v| = 281, ~0.3x the mean residual norm; s=0.5
  already produced gibberish). Excluding those tokens gives |v| = 11.1 / 13.2 / 25.9 at layers 12 / 18 / 24
  (≈0.22x the mean residual norm of ~47 / 59 / 120). Pilot outputs are kept on the VM only as `pilot_bad/`.
- **Layers:** 12, 18 (primary, "mid"), 24.
- **Strength grid:** s ∈ {0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16}.
- **Eval prompts:** 50 fixed open-ended neutral prompts (no travel/France/architecture), disjoint from
  the 100 neutral prompts used to build the vector. Greedy decoding, 128 new tokens.
- **Topic rate (primary):** fraction of outputs matching regex
  `eiffel|paris|parisian|gustave|champ[- ]de[- ]mars|tour eiffel|trocad[eé]ro` (case-insensitive).
  Secondary: strict `eiffel` only; and an LLM judge (unsteered Qwen3-4B, P(Yes) > 0.5 to
  "Does this text mention or talk about the Eiffel Tower or Paris?").
- **Coherence (primary):** an output is *coherent* iff
  (a) its perplexity as plain text under the **unsteered** model ≤ T, where T = 95th percentile of
  the strength-0 outputs' perplexities (computed per layer run — identical since s=0 is the same), **and**
  (b) its repeated-trigram fraction ≤ 0.20 (share of word 3-grams that are duplicates of an earlier 3-gram).
  Coherence rate C_s = share of coherent outputs; **coherence loss** = 1 − C_s / C_0.
  Secondary: median perplexity, repetition, LLM-judge fluency ("Is this fluent, grammatical, sensible
  English? Ignore the topic.").
- **Capability:** MMLU (`cais/mmlu`, `all`/test), 100 fixed items (seed 0), zero-shot, letter-logit argmax,
  with the steering vector active.
- **CIs:** 95% bootstrap over the 50 prompts (10,000 resamples); for coherence loss, prompts are
  resampled jointly for s and s=0.

## Decision rule (fixed)
Over all (layer, strength>0) cells:
- **PROVEN** if at least one cell has topic-rate 95% CI lower bound > 0.80 **and** coherence-loss 95% CI
  upper bound < 0.20.
- **DISPROVEN** if every cell has topic-rate CI upper bound < 0.80 **or** coherence-loss CI lower bound > 0.20
  (i.e. every cell confidently fails at least one criterion).
- **INCONCLUSIVE** otherwise.
(33 cells are searched, so a PROVEN is "exists a sweet spot", not "every layer has one".)

**Exploratory add-on (decided after seeing generations, before any metric/verdict was computed):** the
generations showed a sharp cliff between s=2 and s=4, so a finer grid s ∈ {2.25, 2.5, 2.75, 3.5} is also run
for all three layers. These 12 cells are reported separately and are **not** used for the verdict above.

## Layout
- `scripts/` — `prompts.py` (all prompt sets), `steer_run.py` (Colab GPU job, resumable),
  `analyze.py` (metrics, CIs, verdict), `make_figures.py`.
- `results/` — raw generations/metrics (JSONL/CSV), `summary.json`, `timing.json`.
- `figures/` — charts, `table.png`, `examples_*.png`, `colab_run.txt`.

## Timing
Estimated 105 min vs actual ~112 min (incl. ~30 min idle after a sandbox reboot). GPU: NVIDIA L4 (Colab session b6-095, 2 VMs). ~2.3 CU used (estimate). Details: `results/timing.json`.

## Result
INCONCLUSIVE (no cell passes; see writeup.md). Post-hoc addition: `scripts/score_relevance.py` ("does the answer address the question" judge), not part of the decision rule.
