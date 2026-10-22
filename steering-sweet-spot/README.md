# Is there a steering-vector sweet spot?

## Question and evidence
Can activation addition steer topic (ocean), sentiment (optimistic), and style (archaic English), while preserving useful answers, on Qwen3-4B and Qwen3-1.7B?
Source: Turner et al., [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248), originally Activation Addition. Its abstract reports control while preserving off-target performance. [Golden Gate Claude](https://www.anthropic.com/research/golden-gate-claude) illustrates strong concept activation, using a different feature intervention. This experiment does not use Claude. The [original run](../original-run/) found fluent topic fixation without answering the question. Its result motivates measuring relevance as well as fluency. No numerical success prediction is made before the pilot.

## Registered stage A: pilot protocol (before any GPU work)
- Generators: Qwen/Qwen3-4B, then independent replication Qwen/Qwen3-1.7B. BF16, greedy, thinking disabled, maximum 80 generated tokens. All inference on one NVIDIA L4, no paid APIs.
- Public HF data only: tatsu-lab/alpaca for pilot; databricks/databricks-dolly-15k for confirmation. Filter for standalone English instructions, no supplied context, short prompts. Exact item IDs and hashes saved. Pilot 16 items, confirmation 200 fresh items per concept per model, paired across strengths. Vector construction uses 16 separate Alpaca items.
- Contrasts: identical instruction with versus without an ocean, optimistic, or archaic-English instruction. Mean final prompt-token residual difference. Each vector rescaled to 0.1 times mean neutral residual norm at that layer. Add after the layer at every input/generated position. This is an ActAdd-inspired variant, not an exact replication.
- Pilot grid: layers 12,24,30 (1-indexed, of 36), strengths 2,4,6,8 plus shared zero baseline. 3 concepts x 12 cells x 16 = 576 steered outputs. No pilot items in confirmation. Maximum strength 8 corresponds to a vector norm of 0.8 times reference residual norm.
- Separate judge: unsteered Qwen/Qwen3-8B, forced choice on three binary questions: intended concept evident, readable/coherent, answers original request. Preserve responses, label-token probabilities and model token counts. Useful coherence = readable AND relevant. Judge is a separate checkpoint, but same family. Validate on 50 manually checked pilot outputs before selection; save human labels, disagreements and accuracy. If either concept or useful-coherence accuracy <80%, report failed validation and make overall verdict INCONCLUSIVE (80% minimum prevents trusting a plainly unreliable grader). Manual labels are the experiment operator's assessment, not independent human annotation.
- Perplexity: unsteered generator's mean negative log likelihood of generated tokens only, with EOS/padding excluded. Also log repetition and truncation. Report PPL alongside judge; do not confuse high likelihood with task success.
- Pilot selection: maximize concept rate among cells with useful-coherence loss <=12.5 percentage points against paired baseline; tie-break by higher coherence, then smaller strength, then later layer. If none qualify select maximum concept-rate minus positive coherence-loss. 12.5 pp is two of sixteen pilot answers and an intentionally loose screening filter, not the confirmation margin.
- H1 numerical floors/ceilings will be locked from pilot before confirmation, in a second README commit. Concept floor X = selected pilot rate minus one Wilson 95% half-width, clipped at zero. Coherence-loss ceiling Y = selected pilot paired loss plus one paired-bootstrap 95% half-width, clipped to [0,1]. Report weak or vacuous pilot-derived predictions as such, never manufacture an ambitious threshold.
- H2: test selected strength s* against 2s* at selected layer. Predict at least 20 percentage points more useful-coherence loss at 2s*. This practical margin means one extra failed answer in five; the cliff seen in the original run motivates its direction. If selected s*=8, test 16 as the overshoot. Transfer layer = round(layer / 36 * second-model depth), clipped to valid range; same relative norm, new vectors built from the same construction items.
- Confirmation size: 200 prompts per concept per model, baseline plus s* plus 2s*. This gives roughly +/-7 pp worst-case single-rate 95% uncertainty and can resolve a large 20 pp cliff, not a tiny improvement. Seed 9502 for sampling and 9503 for 10,000 paired prompt bootstraps. Wilson intervals for rates, paired bootstrap for differences; no degenerate 0/100 bootstrap rate intervals. Three concepts x two models are distinct claims, not six independent replications.

## Fixed decision rule
For each model/concept H1 passes if the 95% concept-rate lower bound >= X AND paired useful-coherence-loss upper bound <= Y. H1 fails if concept upper bound < X OR loss lower bound > Y; otherwise unresolved. H2 passes if the lower bound of coherence(s*) minus coherence(2s*) >=20 pp, fails if upper bound <20 pp; otherwise unresolved. Overall PROVEN only if all six H1 and six H2 claims pass and validation passes. DISPROVEN if any claim fails with its interval entirely on the wrong side (and validation passes). Otherwise INCONCLUSIVE. These are pointwise 95% intervals, with no familywise guarantee; the conjunction is deliberately strict. No changing this rule after results.

## Operations, reproducibility, timing
Budget hard cap 6 CU. One Colab session only. I checked the compute-unit floor before provisioning and usage during the run, ran the job detached on the VM, appended JSONL after every batch, downloaded results frequently, and stopped on completion or failure. Results were kept in local git only.
Timing: estimated 185 minutes (setup25/GPU115/eval20/writeup25); final elapsed time, observed GPU and CU are reported below. See results/timing.json. CPU environment aarch64, 2 CPUs, about 7 GB RAM.

## Registered stage B: numerical predictions locked after pilot, before confirmation

Pilot generation and judging completed at 2026-10-18 23:37:03 UTC. Blind operator labels were committed in 6c3876f before consulting judge labels. Agreement: concept 43/50 (86%), readability 45/50 (90%), relevance 48/50 (96%), useful coherence 47/50 (94%). These pass the registered point-estimate gate. **The checker is an AI coding agent (the one that ran the experiment), not an independent human; there were zero human annotators.** The 50-item check is limited evidence, especially with few positives.

| Concept | 4B layer | s* / doubled | Pilot concept / useful | H1 floor X | H1 loss ceiling Y | H2 minimum cliff |
|---|---:|---:|---:|---:|---:|---:|
| archaic | 24 | 4 / 8 | 6.25% / 93.75% | 0.00000000% | 15.625 pp | 20 pp |
| ocean | 30 | 6 / 12 | 6.25% / 100.00% | 0.00000000% | 0.000 pp | 20 pp |
| optimistic | 24 | 4 / 8 | 68.75% / 93.75% | 48.03449984% | 15.625 pp | 20 pp |

Thresholds in selection.json are authoritative at full precision. Both models use these same concept floors and loss ceilings. The second model uses round(layer/36 * its depth), with half-up rounding, and its own newly computed vectors.

**The pilot did not support nonzero lower-floor predictions for ocean or archaic style. Their H1 concept floors are zero and therefore vacuous; passing those H1 tests would not demonstrate useful steering.** The ocean zero-loss ceiling is brittle: all 16 selected answers matched baseline coherence, making the pilot paired-bootstrap interval degenerate. We retain it rather than repair the hypothesis after looking. Optimism predicts at least 48.03449984% concept activation with no more than 15.625 pp useful-coherence loss. The pilot alone does not establish an improvement over naturally upbeat baseline answers. H2 remains a strong cross-concept and cross-model prediction, motivated by the pilot's large declines at higher strengths and the cliff in #095. The registered joint verdict is about this specific recipe and these predictions, not all possible steering methods.

Auxiliary measurement declared while confirmation generation is running, before inspecting its outputs or judge labels: also score unsteered baseline answers for optimism and archaic style, using the identical concept rubric. The original shared baseline was scored only for ocean. This enables descriptive within-prompt concept uplifts and prevents mistaking natural optimism for steering. These added baseline scores do not change the absolute-rate H1, H2, selection, or verdict.

## Final observed results

Verdict: **DISPROVEN** for the registered joint prediction. Headline: **Doubling archaic-style steering on Qwen3-4B reduced judge-defined useful coherence by 79.0 percentage points (95% paired bootstrap CI 73.0-84.5).**

| Model | Concept | L / s* | Concept % [95% CI] | Useful % [95% CI] | Loss pp [95% CI] | Median PPL | H1 | H2 cliff pp [95% CI]; status |
|---|---|---:|---:|---:|---:|---:|---|---|
| Qwen3-4B | archaic | 24/4 | 13.5 [9.4, 18.9] | 79.5 [73.4, 84.5] | 16.0 [10.5, 22.0] | 3.56 | UNRESOLVED | 79.0 [73.0, 84.5]; PASS |
| Qwen3-4B | ocean | 30/6 | 5.0 [2.7, 9.0] | 87.5 [82.2, 91.4] | 8.0 [3.5, 12.5] | 2.27 | FAIL | 81.5 [76.0, 86.5]; PASS |
| Qwen3-4B | optimistic | 24/4 | 59.5 [52.6, 66.1] | 88.0 [82.8, 91.8] | 7.5 [3.0, 12.5] | 2.69 | PASS | 73.0 [66.5, 79.5]; PASS |
| Qwen3-1.7B | archaic | 19/4 | 2.5 [1.1, 5.7] | 77.5 [71.2, 82.7] | 13.0 [6.5, 19.5] | 2.69 | UNRESOLVED | 76.0 [69.5, 82.0]; PASS |
| Qwen3-1.7B | ocean | 23/6 | 3.0 [1.4, 6.4] | 63.5 [56.6, 69.9] | 27.0 [20.0, 34.0] | 4.34 | FAIL | 59.5 [52.0, 66.5]; PASS |
| Qwen3-1.7B | optimistic | 19/4 | 52.0 [45.1, 58.8] | 86.5 [81.1, 90.6] | 4.0 [-0.5, 9.0] | 2.55 | UNRESOLVED | 62.0 [55.5, 68.5]; PASS |

H1: one pass, two failures, three unresolved; H2: all six pass. Validation uses 50 AI operator checks, with zero human annotators. The verdict is conditional on those judge labels.

See writeup.md for interpretation, zero-floor caveats, auxiliary baselines, judge limitations and all artifacts. Final reports preserve the predictions above.

Timing: estimated 185 min vs actual 269.6 min elapsed, including 233.9 min without new persisted results; NVIDIA L4 (two allocations); 1.427 CU estimated at the observed 1.54 CU/hour. Allocation intervals come from session-specific CLI history. The first termination is marked 'pruned', so its interval is a conservative lifecycle estimate, not exact billed GPU time. See results/timing.json for stage durations, overlap and accounting.
