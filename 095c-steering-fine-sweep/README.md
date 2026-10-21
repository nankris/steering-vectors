# Experiment 095c: A Finer Steering Sweep (Eiffel Tower on Qwen3-4B)

## Research Question
Does a finer-grained grid of activation steering across layers and strengths reveal a true "sweet spot" where a model reliably steers toward a concept (the Eiffel Tower) while preserving task capability (MMLU) and answer relevance, or is the trade-off fundamentally a cliff where topic takeover implies task destruction?

## Background & Motivation
In **Experiment 095**, contrastive activation addition (CAA) on Qwen3-4B using an Eiffel Tower vector achieved 100% topic adherence at layer 12 with strength $s=3$, but completely destroyed task answering (0% of outputs answered the user's prompt) and collapsed MMLU accuracy from 68% down to 25% (chance). At layer 24 with $s=2.75\text{--}3.0$, an exploratory slice hinted at a possible compromise (42–70% topic adherence, 62–80% question answering, MMLU preserved at 66%).
In **Experiment 095b**, doubling steering strength cut useful coherence by 79.0 percentage points, confirming an abrupt coherence cliff across multiple concepts and models.
However, neither experiment mapped the boundary with fine resolution across both depth (layers) and magnitude (strengths) simultaneously with paired capability and usefulness benchmarks.

## Experimental Setup
- **Model**: `Qwen/Qwen3-4B` in bfloat16, default chat template with thinking disabled.
- **Steering Vector**: CAA mean difference between 100 Eiffel Tower prompts and 100 neutral prompts, excluding the initial prompt tokens (`SKIP=3` attention-sink tokens).
- **Intervention**: Hook added after transformer decoder layer $L$ adding $s \cdot v_L$ to the residual stream across all token positions.
- **Grid Design**:
  - **Layers** (5 layers): $L \in \{12, 16, 20, 24, 28\}$ of 36 decoder layers.
  - **Strengths** (7 non-zero strengths + baseline $s=0$): $s \in \{1.5, 2.0, 2.5, 2.8, 3.1, 3.5, 4.0\}$.
  - Total cells: $1 \text{ (baseline)} + 5 \times 7 = 36 \text{ cells}$.
- **Evaluation Datasets**:
  - **Capability**: 250 fresh MMLU items from `cais/mmlu` (test split), disjoint from 095's 100 items (deterministic seed 95).
  - **Pilot Slice**: 20 open-ended prompts across coarse cells ($L \in \{12, 20, 28\}$, $s \in \{1.5, 2.5, 3.5\}$) to observe the transition points.
  - **Confirmation Prompts**: 100 fresh open-ended prompts from Databricks-Dolly-15k (`databricks/databricks-dolly-15k`), disjoint from all construction and pilot prompts.
- **Metrics**:
  1. **Topic Adherence (%)**: Output contains Eiffel/Paris keywords (regex `(?i)(eiffel|paris|gustave\s+eiffel|champ\s+de\s+mars|trocad[eé]ro)`) AND unsteered Qwen3-4B topic judge score $> 0.5$.
  2. **Capability (MMLU %)**: Multiple choice accuracy on 250 fresh MMLU test items.
  3. **Mechanical Fluency**: Perplexity (PPL) under unsteered Qwen3-4B $\le 95\text{th}$ percentile of baseline outputs, and repeated trigram fraction $\le 0.20$.
  4. **Usefulness / Answer Relevance (%)**: Unsteered Qwen3-4B judge answering "Does the response actually address the user's question?" ($P(\text{Yes}) > 0.5$).
  5. **Useful Coherence (%)**: Fluent AND Relevant (addresses prompt without degenerating).

## Two-Stage Hypothesis Protocol
1. **Stage 1 (Pilot)**: Run a coarse slice of layers and strengths to establish baseline norms, verify metric sensitivity, and empirically estimate the best candidate cell $(L^*, s^*)$ and threshold trade-offs.
2. **Pre-registration**: Before launching confirmatory evaluation on the 100 fresh Dolly prompts and 250 fresh MMLU items, pre-register the exact predicted cell $(L^*, s^*)$, target topic adherence %, MMLU drop margin, and useful coherence %.
3. **Stage 2 (Confirmatory)**: Evaluate the complete $5 \times 7$ grid on the fresh held-out prompts and MMLU items.

## Pilot Findings & Pre-Registration (Locked Before Confirmatory Run)
The coarse pilot slice evaluated 10 cells ($L \in \{12, 20, 28\}$, $s \in \{0, 1.5, 2.5, 3.5\}$) on 20 pilot prompts and 50 pilot MMLU items:
- **Baseline ($s=0$)**: Topic: 0.0%, MMLU: 72.0%, Relevant: 80.0%, Useful: 80.0%, Median PPL: 2.12.
- **Layer 12**:
  - $s=1.5$: Topic 0.0%, MMLU 62.0% (10 pp drop), Useful 85.0%.
  - $s=2.5$: Topic 85.0%, MMLU 30.0% (42 pp drop!), Useful 30.0% (relevance drops to 30%).
  - $s=3.5$: Topic 95.0%, MMLU 30.0%, Useful 0.0% (incoherence / trigram repetition).
- **Layer 20**:
  - $s=1.5$: Topic 0.0%, MMLU 72.0%, Useful 80.0%.
  - $s=2.5$: Topic 75.0%, MMLU 70.0% (only 2 pp drop), Useful 30.0% (relevance fell to 45%).
  - $s=3.5$: Topic 95.0%, MMLU 54.0% (18 pp drop), Useful 0.0%.
- **Layer 28**:
  - $s=1.5$: Topic 0.0%, MMLU 74.0%, Useful 90.0%.
  - $s=2.5$: Topic 10.0%, MMLU 72.0%, Useful 80.0%.
  - $s=3.5$: Topic 60.0%, MMLU 72.0%, Useful 20.0% (relevance fell to 20%).

### Pre-Registered Best Candidate Cell & Thresholds
Guided by the pilot results and 095's exploratory layer 24 observations:
- **Pre-Registered Best Cell**: **Layer 24 at Strength $s=2.8$** ($L^*=24, s^*=2.8$).
- **Pre-Registered Quantitative Predictions**:
  1. **Topic Adherence**: $\ge 70.0\%$ (lower bound of 95% Wilson CI $\ge 60.0\%$).
  2. **MMLU Drop**: $\le 3.0\text{ pp}$ relative to unsteered baseline (upper bound of 95% paired bootstrap CI $\le 5.0\text{ pp}$).
  3. **Useful Coherence**: $\ge 60.0\%$ (lower bound of 95% Wilson CI $\ge 50.0\%$).

## Decision Rule (Pre-registered)
- **PROVEN**: The pre-registered best cell $(L^*, s^*)$ or ANY cell in the $5 \times 7$ grid achieves on the confirmatory test:
  1. Topic adherence $\ge 70\%$ (lower bound of 95% Wilson CI $\ge 60\%$),
  2. MMLU accuracy drop $\le 3.0$ percentage points relative to baseline (upper bound of 95% paired bootstrap CI $\le 5.0$ pp),
  3. Useful coherence $\ge 60\%$ (lower bound of 95% Wilson CI $\ge 50\%$).
- **DISPROVEN**: Across the entire $5 \times 7$ fine grid, no cell meets the criteria (topic adherence $\ge 70\%$ with MMLU drop $\le 3.0$ pp and useful coherence $\ge 60\%$), demonstrating that topic steering and task preservation are mutually incompatible across all tested layers and strengths.
- **INCONCLUSIVE**: The candidate cell or other frontier cells have confidence intervals overlapping the thresholds, or data collection is incomplete within the Colab budget.

## Compute & Resource Accounting
- Hardware: NVIDIA L4 GPU on Colab (`b10-095c`).
- Budget Cap: 8.0 Colab Compute Units (CU).
- Actual CU Used: 1.66 CU (Balance: 102.51 -> 100.85 CU).
- Total Tokens Evaluated: 4,129,269 tokens across vector extraction, generation, MMLU, and judging.

## Timing
- Timing: Estimated 110 min vs actual 94 min (Setup: 14m, GPU run: 47m, Eval/Analysis: 8m, Writeup: 25m). GPU: NVIDIA L4 (Colab session `b10-095c`), 1.66 CU used.

## Final Results & Verdict
- **Verdict: DISPROVEN**.
- Across the complete $5 \times 7$ grid (35 steered cells), **zero cells** satisfied the pre-registered sweet spot criteria (Topic Adherence $\ge 70\%$ with MMLU drop $\le 3.0$ pp and Useful Coherence $\ge 60\%$).
- **Pre-Registered Candidate Cell ($L=24, s=2.8$)**:
  - Topic Adherence: 54.0% [95% Wilson CI: 44.3%, 63.4%] (Target $\ge 70.0\%$ — **FAILED**)
  - MMLU Test Accuracy: 70.0% [95% Wilson CI: 64.1%, 75.3%], drop of 1.2 pp from 71.2% baseline (Target $\le 3.0$ pp — **PASSED**)
  - Useful Coherence: 30.0% [95% Wilson CI: 21.9%, 39.6%] (Target $\ge 60.0\%$ — **FAILED**)
  - Answer Relevance: 32.0% (collapsed from baseline 75.0%)
- **Grid-Wide Finding**:
  - In middle layers ($L=12, 16$), steering at $s \ge 2.5$ causes catastrophic capability collapse, taking MMLU to random chance (~23-24%) and useful coherence to 0%.
  - In late layers ($L=24, 28$), MMLU reasoning is preserved (~70-72%), but open-ended answering collapses: increasing strength to $s=3.1$ achieves 81.0% topic adherence but crashes answer relevance to 13.0% and useful coherence to 10.0%. At $s=3.5$, topic adherence is 97.0% but useful coherence is 2.0%.
  - The trade-off is an unavoidable cliff: topic takeover implies task destruction.

## Screenshots
The experiment captures four real program execution screenshots in `figures/screenshots/`:
1. `figures/screenshots/01_first_broken_result.png`: Output log showing early-layer collapse at Layer 12 where MMLU drops to chance (22.8%–24.4%) and Useful Coherence drops to 0.0%.
2. `figures/screenshots/02_evidence_of_flaw.png`: Raw generation and judge outputs showing fluent Eiffel Tower boilerplate that achieves low perplexity (PPL=1.18–1.59) but completely fails answer relevance.
3. `figures/screenshots/03_fixed_run_candidate.png`: Confirmatory log output for Layer 24 across fine strengths on fresh Dolly prompts.
4. `figures/screenshots/04_final_verdict_stats.png`: Final stats and verdict printout confirming the sweet spot hypothesis is DISPROVEN across all 35 grid cells.

## Decisions
1. **Layer Selection**: Selected layers 12, 16, 20, 24, 28 to span from middle layers (where 095 showed early collapse) to upper layers (where 095 showed delayed collapse), with regular 4-layer intervals.
2. **Strength Granularity**: Spanning 1.5 to 4.0 in steps of 0.3 to 0.5 around the critical 2.0 to 3.5 zone where 095 observed the cliff.
3. **MMLU Sample Size**: 250 fresh items (well exceeding the $\ge 200$ requirement) to keep the standard error below 2.8 percentage points.
4. **Autonomous Execution**: Followed the autonomous execution mandate without stopping for interactive prompts, executing the complete planned sweep on Colab and terminating the session immediately upon completion.

