# 14 (095b): 80-token cap truncates ~95%+ of answers

- **Experiment:** 095b (`095b-steering-sweet-spot/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `095b-steering-sweet-spot.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
80-token cap truncates ~95%+ of answers; "useful coherence" judges answer openings, not correctness. No capability benchmark (unlike 095).

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
