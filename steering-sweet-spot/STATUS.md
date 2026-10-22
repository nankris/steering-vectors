# Status: completed

The Colab session was stopped on 2026-10-19 03:54:39 UTC. No further GPU work is needed.

All 592 pilot and 2,800 confirmation outputs and labels are saved locally, plus 800 auxiliary baseline labels and 36 diagnostic rejudgments. Registration commits 07e9245 and d4d6ab3 are preserved. Verdict: DISPROVEN for the registered joint prediction; all six H2 cliffs passed. See writeup.md and blog.md. Validation is AI-agent spot checks, with no independent human annotation.

Estimated L4 lifecycle cost: 1.427 CU, with the first-allocation pruning caveat in results/timing.json. Both allocations appear in results/recovery_cli.jsonl. The pilot settings were not reselected after the pilot.
