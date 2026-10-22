# 01: Joint hypothesis structurally hard to pass

- **Experiment:** original run (`original-run/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `original-run.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**Joint hypothesis structurally hard to pass**: with n=50, topic CI LB > 0.80 needs observed >= ~0.92; coh-loss UB < 0.20 needs observed loss ~<= 0.05 given bootstrap width ~+/-0.13. Against a cliff spanning ~1 unit of s on a coarse grid (2 -> 3), PROVEN required landing exactly in a window the grid doesn't sample. Conversely DISPROVEN was also hard (a single borderline cell blocks it), so INCONCLUSIVE was the likely outcome a priori.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the blog post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
