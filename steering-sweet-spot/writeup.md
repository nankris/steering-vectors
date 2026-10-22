# Is there a steering-vector sweet spot?

**Verdict: DISPROVEN for the registered joint prediction.** This is a verdict on this vector recipe, grid, and depth-based transfer rule. Stage A registration: `07e9245`; numerical predictions locked in `d4d6ab3` before confirmation generation.

**Headline number: Doubling archaic-style steering on Qwen3-4B reduced judge-defined useful coherence by 79.0 percentage points (95% paired bootstrap CI 73.0-84.5).**

**Plain-English takeaway:** A stronger style vector made archaic wording common while readable, relevant answer openings collapsed. This setup did not demonstrate a reliable sweet spot across the three concepts and two Qwen models.

## Main results

All rates describe the first at most 80 generated tokens. Most answers are truncated; partial relevant answers can qualify.

| Model | Concept | L / s* | Concept % [95% CI] | Useful % [95% CI] | Loss pp [95% CI] | Median PPL | H1 | H2 cliff pp [95% CI]; status |
|---|---|---:|---:|---:|---:|---:|---|---|
| Qwen3-4B | archaic | 24/4 | 13.5 [9.4, 18.9] | 79.5 [73.4, 84.5] | 16.0 [10.5, 22.0] | 3.56 | UNRESOLVED | 79.0 [73.0, 84.5]; PASS |
| Qwen3-4B | ocean | 30/6 | 5.0 [2.7, 9.0] | 87.5 [82.2, 91.4] | 8.0 [3.5, 12.5] | 2.27 | FAIL | 81.5 [76.0, 86.5]; PASS |
| Qwen3-4B | optimistic | 24/4 | 59.5 [52.6, 66.1] | 88.0 [82.8, 91.8] | 7.5 [3.0, 12.5] | 2.69 | PASS | 73.0 [66.5, 79.5]; PASS |
| Qwen3-1.7B | archaic | 19/4 | 2.5 [1.1, 5.7] | 77.5 [71.2, 82.7] | 13.0 [6.5, 19.5] | 2.69 | UNRESOLVED | 76.0 [69.5, 82.0]; PASS |
| Qwen3-1.7B | ocean | 23/6 | 3.0 [1.4, 6.4] | 63.5 [56.6, 69.9] | 27.0 [20.0, 34.0] | 4.34 | FAIL | 59.5 [52.0, 66.5]; PASS |
| Qwen3-1.7B | optimistic | 19/4 | 52.0 [45.1, 58.8] | 86.5 [81.1, 90.6] | 4.0 [-0.5, 9.0] | 2.55 | UNRESOLVED | 62.0 [55.5, 68.5]; PASS |

H1 outcomes: one PASS, two FAIL (ocean on both models), three UNRESOLVED. H2 outcomes: all six PASS. The minimum observed cliff was 59.5 pp [52.0, 66.5] on 1.7B ocean. Thus the predicted cliff transferred, while the combined sweet-spot prediction failed.

Each selected row uses 200 fresh Dolly prompts. Useful coherence means readable AND relevant, including partial relevant answers; it does not assess factual correctness. Loss is baseline useful coherence minus steered useful coherence, in absolute percentage points. H2 cliff is useful coherence at s* minus at 2s*. Wilson 95% intervals for rates; 10,000 paired prompt bootstraps for differences. Intervals describe prompt-sampling uncertainty conditional on the fixed vector and labels, not systematic judge error. No multiplicity correction.

Unsteered baselines: Qwen3-1.7B: useful 90.5 [85.6, 93.8]%, median PPL 1.19; Qwen3-4B: useful 95.5 [91.7, 97.6]%, median PPL 1.14.

The concept floors for ocean and archaic style are **zero**, so those H1 concept tests are vacuous. Optimism's floor is 48.03449984%. Coherence-loss ceilings: ocean 0 pp; others 15.625 pp. Ocean's zero ceiling came from a degenerate 16-item paired pilot, making it particularly brittle. H2 requires a cliff of at least 20 pp. The registered rule and failed claims were retained unchanged.

## What was run

- Qwen/Qwen3-4B (36 layers) pilot: 16 construction instructions, 16 different evaluation prompts from tatsu-lab/alpaca. Three concepts x 3 layers x 4 strengths plus shared baseline = 592 generations. Layers 12,24,30; strengths 2,4,6,8.
- Confirmation: 200 databricks/databricks-dolly-15k prompts, reused across concepts and models for pairing, absent from construction and pilot by exact text hash. Two models x (baseline + 3 concepts x 2 strengths) x 200 = 2,800 generations. These are 200 unique held-out prompts, not 1,200 independent prompts.
- Qwen/Qwen3-1.7B (28 layers) replication: layers 19 for archaic/optimistic and 23 for ocean. Fresh vectors computed from the same 16 construction instructions, relative norm matched within model. No model-specific retuning.
- Vector recipe: mean final prompt-token activation with an explicit concept instruction minus without it, scaled to 0.1 x mean neutral residual norm; add strength x vector after the layer at every prompt and decoding position. BF16, greedy, thinking off, maximum 80 new tokens. This differs from Turner's exact ActAdd implementation and Anthropic's feature intervention.
- Separate checkpoint judge: unsteered Qwen/Qwen3-8B. Three forced Yes/No token decisions, using relative Yes/No probability. Separate readability, relevance, and concept criteria. Generation/decision token counts saved per output. Unsteered-generator PPL scores exact generated tokens, excluding EOS/padding. PPL is diagnostic and does not gate H1/H2.

Generation token ledger (pilot + confirmation): 80,788 prompt tokens and 261,343 generated tokens. Judge: 1,752,615 input tokens and 10,976 scored label-token decisions, including auxiliary baseline scoring. Label decisions use a forward pass, not sampled free-text generation. The recovery QA additionally rescored 36 existing outputs; its separate token counts are in `recovery_qa_tokens.json`. Additional vector-extraction tokens are in `vector_token_counts.json`; likelihood-scoring passes are separate from these generation totals.

## Judge check

**Validation limitation:** the 50 hand-checked labels are AI operator assessments, with no independent human annotation. The mechanical pre-registered verdict is conditional on this weaker validation and is not human-validated evidence.

Fifty pilot responses were inspected blind to judge labels by **the AI coding agent that ran the experiment**, with zero independent human annotators. Concept agreement: 43/50 (86%); useful-coherence agreement: 47/50 (94%). The registered 80% point-estimate gate passes. Concept positives are a weakness: the judge recovered only 7/12 operator positives (58.3% recall), with 36/38 specificity. See `results/manual_validation.json`, `validation_protocol.json`, `validation_confusion.json`, and `verdict.json` for full labels, disagreements and agreement CIs. The subgroup breakdown is more concerning: archaic agreement 18/19, ocean 16/16 with zero operator-positive ocean examples, and optimism only 9/15 (60%). See `validation_by_concept.json`. The pooled gate masked a weak sentiment check and untested positive ocean detection. This is an operator spot check, not human validation or a calibrated gold standard.

## Auxiliary baseline concept comparison

Declared before inspecting confirmatory outputs, after confirmation generation began. Uses the same rubric, with extra baseline-only labels for optimism and archaic style. These comparisons do not change the registered verdict:

- Qwen3-4B, archaic: baseline 0.0%, steered 13.5%, paired uplift 13.5 [9.0, 18.5] pp.
- Qwen3-4B, ocean: baseline 0.5%, steered 5.0%, paired uplift 4.5 [2.0, 7.5] pp.
- Qwen3-4B, optimistic: baseline 21.5%, steered 59.5%, paired uplift 38.0 [31.0, 45.0] pp.
- Qwen3-1.7B, archaic: baseline 0.0%, steered 2.5%, paired uplift 2.5 [0.5, 5.0] pp.
- Qwen3-1.7B, ocean: baseline 1.0%, steered 3.0%, paired uplift 2.0 [-0.5, 5.0] pp.
- Qwen3-1.7B, optimistic: baseline 19.5%, steered 52.0%, paired uplift 32.5 [26.0, 39.0] pp.

## Caveats

- One direction per concept/model, 16 construction items, one deterministic decoding setting, and a finite layer/strength grid. Other vectors or recipes remain untested. Depth-scaled layers need not perform equivalent computations.
- Concept versus no-instruction contrasts also change instruction wording and length. The direction is not a verified pure concept feature. This is not a direct replication or refutation of either source.
- Same-family model judge and AI operator checks can share errors. Conditional CIs do not include those errors. Label uncertainty may dominate a small observed effect.
- The 80-token cap truncates many answers (rates in raw metrics), and the rubric accepts partial answers. Readability and relevance are weaker than verified task correctness. No capability benchmark or latency comparison was run.
- Pilot selection and small pilot-derived margins are noisy. The zero concept floors make the joint test weaker as evidence of steering; the zero ocean quality ceiling makes it harsher as a quality test.
- Shared prompts across concepts/models create dependence. Rates measure absolute concept presence; baseline uplifts are auxiliary. PPL values should be interpreted within a model.

## Artifacts and reproduction

The original VM disappeared after a locally saved partial judging snapshot. A second L4 allocation resumed 1,180 missing judge records without regenerating any answers, then ran the auxiliary baseline scorer. A diagnostic random sample of 36 earlier responses reproduced all 108 saved binary decisions (36/36 on each of concept, readability, and relevance); see `recovery_qa.json`. All final results were copied locally before `colab stop -s sweetspot`.

`figures/pilot_heatmap.png`: pilot concept rate with useful-coherence annotations. `figures/frontier.png`: pilot and fresh-data trade-offs, Wilson intervals. `figures/coherence_cliff.png`: paired cliff CIs against the registered 20 pp line. `figures/table.png`: main results table. `figures/diagram.png` and editable `.excalidraw`: pipeline. `figures/colab_run.txt`: nvidia-smi, model identities, batch timings and completion lines.

Raw JSONL, CSV summaries, exact prompts, model revisions, vectors, environment and accounting are in `results/`. `scripts/run_gpu.py` is resumable by output key; launch scripts run detached on Colab. CPU reproduction after downloading results: `python scripts/analyze.py confirm`, `python scripts/auxiliary.py`, `python scripts/plot.py`, `python scripts/audit.py`, `python scripts/report.py`. Pilot selection is already frozen; do not rerun it to choose a different setting.

Sources: [Turner et al.](https://arxiv.org/abs/2308.10248), [Golden Gate Claude demonstration](https://www.anthropic.com/research/golden-gate-claude). No Claude, DeepSeek, Grok, Gemini, or paid API was used for inference.

Timing: estimated 185 min vs actual 269.6 min elapsed, including 233.9 min without new persisted results; NVIDIA L4 (two allocations); 1.427 CU estimated at the observed 1.54 CU/hour. Allocation intervals come from session-specific CLI history. The first termination is marked 'pruned', so its interval is a conservative lifecycle estimate, not exact billed GPU time. See results/timing.json for stage durations, overlap and accounting.
