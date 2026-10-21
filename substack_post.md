TITLE: I made Qwen3-4B obsessed with the Eiffel Tower, and it stopped answering questions
SUBTITLE: A steering vector from 200 prompts works, but there is no setting where the model mentions the tower and still helps you.

Anthropic's "Golden Gate Claude" showed that you can nudge a model toward one concept by adding a vector to its activations. I wanted to know if there is a sweet spot on a small open model: a strength where the model slips in the Eiffel Tower while still giving a good answer. My predeclared bar was a topic rate above 80% with less than 20% coherence loss.

Verdict: INCONCLUSIVE, leaning toward "no sweet spot". None of the 30 predeclared layer and strength cells cleared both bars. One cell, layer 24 at strength 3, did not fail either bar with confidence, because its topic-rate interval reaches 82%. So the rule cannot say DISPROVEN.

[IMAGE table.png]

Setup

Model: Qwen3-4B in bf16, chat template, thinking off. The vector is the mean residual activation on 100 Eiffel Tower prompts minus the mean on 100 neutral prompts. I add the vector times a strength s at every position after decoder layer 12, 18 or 24 (of 36). I tested strengths from 0 to 16, plus an exploratory fine grid at 2.25, 2.5, 2.75 and 3.5 that was not used for the verdict.

I scored 50 fixed neutral prompts with greedy decoding and 128 new tokens. Topic rate is a regex for eiffel, paris, gustave and similar words. Coherence is perplexity under the unsteered model plus a repeated-trigram check. Capability is 100 fixed MMLU items with the vector active. Confidence intervals are 95% bootstrap over prompts. GPU: one NVIDIA L4 on Colab, about 112 minutes, about 2.3 CU.

The result is a cliff, not a slope

At every layer, the topic rate goes from near 0% to near 100% within about one unit of strength.

On layer 12, strength 2 gave a 6% topic rate (CI 0 to 14). Strength 2.5 gave 66% (CI 54 to 78). Strength 3 gave 100% on all 50 answers. That sounds like a win until you look at the other columns. At layer 12, strength 3, a post-hoc judge found that 0% of the answers actually addressed the question, and MMLU fell from 68% to 25%, which is chance. The text still read as fluent Wikipedia-style Eiffel paragraphs. They just were not answers.

[IMAGE topic_coherence_vs_strength.png]

Where you steer matters

Early-layer steering hurts capability the most. On layer 12, MMLU hits chance once strength reaches 2.5. On layer 24, MMLU stays at 66 to 70% up to strength 4, because steering late mostly changes what the model says, not how it reasons about a multiple-choice question.

The best compromise was layer 24 at strength 2.75 to 3. There, 42 to 70% of answers mention the tower, 62 to 80% still answer the question, coherence loss is 13 to 26%, and MMLU stays at 66%. That is a real trade-off, and it still misses my 80% topic bar.

[IMAGE mmlu_vs_strength.png]

At very high strengths (8 to 16) the regex topic rate falls back to 0%. That is not a return to normal. The output collapses into repeated symbols and no longer contains a full keyword.

Verdict

INCONCLUSIVE under the predeclared rule. Plainly: you can Golden-Gate a 4B model with one vector, but by the time it reliably mentions the tower, it has stopped doing the task.

Caveats

One model, one concept, one vector recipe (an unnormalized mean difference added at all positions). Other recipes, such as steering only generated tokens, normalizing, or spreading a smaller push across several layers, might widen the window. With 50 prompts and 100 MMLU items, the intervals are about plus or minus 10 to 15 points, and the 0% and 100% cells have degenerate intervals. The coherence metric is a proxy that rewards on-topic boilerplate, which is why I added the post-hoc "answers the question" judge. That judge is the same model family and was not predeclared. The regex counts "Paris" as on-topic. A pilot that averaged over all tokens, including the attention-sink token, was dominated by a length artifact, and I fixed that before computing any metrics.

What to try next

Steer only the generated tokens, try multi-layer steering at lower strength, and try a concept that is less tied to a few keywords than the Eiffel Tower.

All code and raw outputs are in the experiment folder. If you want more measured tests like this, subscribe to Tech in General.
