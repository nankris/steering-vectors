# 20 (095c): Absolute useful threshold 60% vs baseline 74%

- **Experiment:** 095c (`095c-steering-fine-sweep/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `095c-steering-fine-sweep.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Absolute useful threshold 60% vs baseline 74% (unsteered model only 75% "relevant" per judge) leaves a 14 pp loss budget; relevance judge is the generator's own model and visibly miscalibrated (false negatives on baseline, false positive on Paris-as-capital-of-Arkansas).

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
