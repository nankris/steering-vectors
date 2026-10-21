# 12 (095b): Ocean steering basically didn't work

- **Experiment:** 095b (`095b-steering-sweet-spot/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `095b-steering-sweet-spot.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Ocean steering basically didn't work (5% / 3% concept rate, uplift CI touching 0 on 1.7B) — selection picked a cell that barely steers because the pilot rule prioritised low coherence loss.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
