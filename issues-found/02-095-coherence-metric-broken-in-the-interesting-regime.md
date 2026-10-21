# 02 (095): Coherence metric broken in the interesting regime

- **Experiment:** 095 (`095-steering-vectors/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `095-steering-vectors.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**Coherence metric broken in the interesting regime**: PPL-under-unsteered rewards on-topic boilerplate (L12 s3 median PPL 2.14, ratio 1.005 vs baseline). Cells fail only via trigram repetition. Authors acknowledge and add a post-hoc relevance judge — that judge is the same Qwen3-4B, not pre-registered.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
