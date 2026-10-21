# 18 (095c): PROVEN was structurally unreachable

- **Experiment:** 095c (`095c-steering-fine-sweep/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `095c-steering-fine-sweep.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**PROVEN was structurally unreachable.** With the unpaired SE at n=250, p~0.71, half-width ~7.9 pp; MMLU-drop UB <= 5 pp requires the steered MMLU to be ~3 pp ABOVE baseline. No cell achieved that (best was -1.2 pp). The verdict space was effectively {DISPROVEN, INCONCLUSIVE} by construction. (Moot here since topic/useful also fail everywhere, but the rule was not a fair test.)

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
