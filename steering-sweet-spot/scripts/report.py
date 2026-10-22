"""Render a concise research record from saved statistics (no inference)."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
v=json.load(open(R/'verdict.json'));c=pd.read_csv(R/'confirmation_metrics.csv');claims=pd.read_csv(R/'claims.csv');u=pd.read_csv(R/'concept_uplifts.csv');h=json.load(open(R/'headline.json'));t=json.load(open(R/'timing.json'))
def ci(x,lo,hi):return f'{100*x:.1f} [{100*lo:.1f}, {100*hi:.1f}]'
lines=['| Model | Concept | L / s* | Concept % [95% CI] | Useful % [95% CI] | Loss pp [95% CI] | Median PPL | H1 | H2 cliff pp [95% CI]; status |','|---|---|---:|---:|---:|---:|---:|---|---|']
for _,r in claims.iterrows():
 m=c[(c.model==r.model)&(c.concept==r.concept)&(c.strength==r.strength)].iloc[0]
 lines.append(f'| {r.model.split("/")[-1]} | {r.concept} | {r.layer}/{r.strength} | {ci(m.concept_rate,m.concept_lo,m.concept_hi)} | {ci(m.coherence,m.coherence_lo,m.coherence_hi)} | {ci(m.loss,m.loss_lo,m.loss_hi)} | {m.median_ppl:.2f} | {r.H1} | {ci(r.cliff,r.cliff_lo,r.cliff_hi)}; {r.H2} |')
table='\n'.join(lines)
audit=json.load(open(R/'audit.json'));aux_tokens=json.load(open(R/'auxiliary_tokens.json'))
totals={k:sum(d[k] for d in audit['token_totals'].values()) for k in ['input_tokens','output_tokens','judge_input_tokens','judge_output_tokens']}
totals['judge_input_tokens']+=aux_tokens['auxiliary_judge_input_tokens'];totals['judge_output_tokens']+=aux_tokens['auxiliary_judge_decision_tokens']
base='; '.join(f'{r.model.split("/")[-1]}: useful {ci(r.coherence,r.coherence_lo,r.coherence_hi)}%, median PPL {r.median_ppl:.2f}' for _,r in c[c.concept=='baseline'].iterrows())
up=[]
for _,r in claims.iterrows():
 a=u[(u.model==r.model)&(u.concept==r.concept)&(u.strength==r.strength)].iloc[0]
 up.append(f'- {r.model.split("/")[-1]}, {r.concept}: baseline {100*a.baseline_rate:.1f}%, steered {100*a.steered_rate:.1f}%, paired uplift {ci(a.uplift,a.uplift_lo,a.uplift_hi)} pp.')
time=f"Timing: estimated {t['estimated_minutes']['total']} min vs actual {t['actual_minutes']:.1f} min elapsed, including {t['interruption_minutes']:.1f} min without new persisted results; NVIDIA L4 (two allocations); {t['cu_estimated']:.3f} CU estimated at the observed 1.54 CU/hour. Allocation intervals come from session-specific CLI history. The first termination is marked 'pruned', so its interval is a conservative lifecycle estimate, not exact billed GPU time. See results/timing.json for stage durations, overlap and accounting."

text=f'''# Is there a steering-vector sweet spot?

**Verdict: {v['verdict']} for the registered joint prediction.** This is a verdict on this vector recipe, grid, and depth-based transfer rule. Stage A registration: `07e9245`; numerical predictions locked in `d4d6ab3` before confirmation generation.

**Headline number: {h['text']}**

**Plain-English takeaway:** {h['takeaway']}

## Main results

All rates describe the first at most 80 generated tokens. Most answers are truncated; partial relevant answers can qualify.

{table}

H1 outcomes: one PASS, two FAIL (ocean on both models), three UNRESOLVED. H2 outcomes: all six PASS. The minimum observed cliff was 59.5 pp [52.0, 66.5] on 1.7B ocean. Thus the predicted cliff transferred, while the combined sweet-spot prediction failed.

Each selected row uses 200 fresh Dolly prompts. Useful coherence means readable AND relevant, including partial relevant answers; it does not assess factual correctness. Loss is baseline useful coherence minus steered useful coherence, in absolute percentage points. H2 cliff is useful coherence at s* minus at 2s*. Wilson 95% intervals for rates; 10,000 paired prompt bootstraps for differences. Intervals describe prompt-sampling uncertainty conditional on the fixed vector and labels, not systematic judge error. No multiplicity correction.

Unsteered baselines: {base}.

The concept floors for ocean and archaic style are **zero**, so those H1 concept tests are vacuous. Optimism's floor is 48.03449984%. Coherence-loss ceilings: ocean 0 pp; others 15.625 pp. Ocean's zero ceiling came from a degenerate 16-item paired pilot, making it particularly brittle. H2 requires a cliff of at least 20 pp. The registered rule and failed claims were retained unchanged.

## What was run

- Qwen/Qwen3-4B (36 layers) pilot: 16 construction instructions, 16 different evaluation prompts from tatsu-lab/alpaca. Three concepts x 3 layers x 4 strengths plus shared baseline = 592 generations. Layers 12,24,30; strengths 2,4,6,8.
- Confirmation: 200 databricks/databricks-dolly-15k prompts, reused across concepts and models for pairing, absent from construction and pilot by exact text hash. Two models x (baseline + 3 concepts x 2 strengths) x 200 = 2,800 generations. These are 200 unique held-out prompts, not 1,200 independent prompts.
- Qwen/Qwen3-1.7B (28 layers) replication: layers 19 for archaic/optimistic and 23 for ocean. Fresh vectors computed from the same 16 construction instructions, relative norm matched within model. No model-specific retuning.
- Vector recipe: mean final prompt-token activation with an explicit concept instruction minus without it, scaled to 0.1 x mean neutral residual norm; add strength x vector after the layer at every prompt and decoding position. BF16, greedy, thinking off, maximum 80 new tokens. This differs from Turner's exact ActAdd implementation and Anthropic's feature intervention.
- Separate checkpoint judge: unsteered Qwen/Qwen3-8B. Three forced Yes/No token decisions, using relative Yes/No probability. Separate readability, relevance, and concept criteria. Generation/decision token counts saved per output. Unsteered-generator PPL scores exact generated tokens, excluding EOS/padding. PPL is diagnostic and does not gate H1/H2.

Generation token ledger (pilot + confirmation): {totals['input_tokens']:,} prompt tokens and {totals['output_tokens']:,} generated tokens. Judge: {totals['judge_input_tokens']:,} input tokens and {totals['judge_output_tokens']:,} scored label-token decisions, including auxiliary baseline scoring. Label decisions use a forward pass, not sampled free-text generation. The recovery QA additionally rescored 36 existing outputs; its separate token counts are in `recovery_qa_tokens.json`. Additional vector-extraction tokens are in `vector_token_counts.json`; likelihood-scoring passes are separate from these generation totals.

## Judge check

**Validation limitation:** the 50 hand-checked labels are AI operator assessments, with no independent human annotation. The mechanical pre-registered verdict is conditional on this weaker validation and is not human-validated evidence.

Fifty pilot responses were inspected blind to judge labels by **the AI coding agent that ran the experiment**, with zero independent human annotators. Concept agreement: 43/50 (86%); useful-coherence agreement: 47/50 (94%). The registered 80% point-estimate gate passes. Concept positives are a weakness: the judge recovered only 7/12 operator positives (58.3% recall), with 36/38 specificity. See `results/manual_validation.json`, `validation_protocol.json`, `validation_confusion.json`, and `verdict.json` for full labels, disagreements and agreement CIs. The subgroup breakdown is more concerning: archaic agreement 18/19, ocean 16/16 with zero operator-positive ocean examples, and optimism only 9/15 (60%). See `validation_by_concept.json`. The pooled gate masked a weak sentiment check and untested positive ocean detection. This is an operator spot check, not human validation or a calibrated gold standard.

## Auxiliary baseline concept comparison

Declared before inspecting confirmatory outputs, after confirmation generation began. Uses the same rubric, with extra baseline-only labels for optimism and archaic style. These comparisons do not change the registered verdict:

{chr(10).join(up)}

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

{time}
'''
(ROOT/'writeup.md').write_text(text)
readme=(ROOT/'README.md').read_text().split('\n## Final observed results')[0]
readme=readme.replace('actual pending. GPU L4 planned; CU pending.', 'final elapsed time, observed GPU and CU are reported below.')
readme+='\n## Final observed results\n\n'+f"Verdict: **{v['verdict']}** for the registered joint prediction. Headline: **{h['text']}**\n\n"+table+'\n\nH1: one pass, two failures, three unresolved; H2: all six pass. Validation uses 50 AI operator checks, with zero human annotators. The verdict is conditional on those judge labels.\n\nSee writeup.md for interpretation, zero-floor caveats, auxiliary baselines, judge limitations and all artifacts. Final reports preserve the predictions above.\n\n'+time+'\n'
(ROOT/'README.md').write_text(readme)
print(time)
