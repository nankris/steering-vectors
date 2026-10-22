"""Metrics, bootstrap CIs and verdict for the original steering run. Reads results/raw/, writes results/."""
import csv, json, math, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
RAW = os.path.join(RES, "raw")
LAYERS = [12, 18, 24]
STRENGTHS = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16]
FINE = [2.25, 2.5, 2.75, 3.5]  # exploratory, not used for verdict
TOPIC_RE = re.compile(r"eiffel|paris|parisian|gustave|champ[- ]de[- ]mars|tour eiffel|trocad[eé]ro", re.I)
STRICT_RE = re.compile(r"eiffel", re.I)
REP_MAX = 0.20
B = 10000
rng = np.random.default_rng(0)


def rep3(text):
    w = re.findall(r"\w+", text.lower())
    tri = list(zip(w, w[1:], w[2:]))
    return 0.0 if not tri else 1 - len(set(tri)) / len(tri)


def load(L, s):
    key = (0, 0) if s == 0 else (L, s)
    sc = json.load(open(f"{RAW}/score_L{key[0]}_s{key[1]}.json"))
    mm = json.load(open(f"{RAW}/mmlu_L{key[0]}_s{key[1]}.json"))
    return sc, mm


def ci(boot):
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


base, base_mm = load(None, 0)
T = float(np.percentile([r["ppl"] for r in base], 95))


def per_prompt(rows):
    topic = np.array([bool(TOPIC_RE.search(r["output"])) for r in rows], float)
    strict = np.array([bool(STRICT_RE.search(r["output"])) for r in rows], float)
    jt = np.array([r["judge_topic"] > 0.5 for r in rows], float)
    rep = np.array([rep3(r["output"]) for r in rows])
    ppl = np.array([r["ppl"] for r in rows])
    coh = ((ppl <= T) & (rep <= REP_MAX)).astype(float)
    jf = np.array([r["judge_fluent"] > 0.5 for r in rows], float)
    return dict(topic=topic, strict=strict, judge_topic=jt, rep=rep, ppl=ppl, coh=coh, judge_fluent=jf)


bp = per_prompt(base)
base_acc = np.array([r["correct"] for r in base_mm], float)
idx_p = rng.integers(0, 50, (B, 50))
idx_m = rng.integers(0, 100, (B, 100))

rows_out = []
for L in LAYERS:
    for s in sorted(STRENGTHS + FINE):
        if s in FINE and not os.path.exists(f"{RAW}/score_L{L}_s{s}.json"):
            continue
        sc, mm = load(L, s)
        p = per_prompt(sc)
        acc = np.array([r["correct"] for r in mm], float)
        rk = (0, 0) if s == 0 else (L, s)
        relev = np.array(json.load(open(f"{RAW}/relev_L{rk[0]}_s{rk[1]}.json"))) > 0.5  # post-hoc, see score_relevance.py
        topic_b = p["topic"][idx_p].mean(1)
        c0_b = bp["coh"][idx_p].mean(1)
        cs_b = p["coh"][idx_p].mean(1)
        loss_b = 1 - cs_b / np.where(c0_b == 0, np.nan, c0_b)
        C0, Cs = bp["coh"].mean(), p["coh"].mean()
        d_acc_b = (acc - base_acc)[idx_m].mean(1)
        fin = np.isfinite(p["ppl"])
        row = dict(
            layer=L, strength=s, exploratory=s in FINE,
            topic_rate=p["topic"].mean(), topic_lo=ci(topic_b)[0], topic_hi=ci(topic_b)[1],
            strict_eiffel_rate=p["strict"].mean(), judge_topic_rate=p["judge_topic"].mean(),
            coherent_rate=Cs, coherence_loss=1 - Cs / C0,
            coh_loss_lo=ci(loss_b)[0], coh_loss_hi=ci(loss_b)[1],
            median_ppl=float(np.median(p["ppl"])), ppl_ratio=float(np.median(p["ppl"]) / np.median(bp["ppl"])),
            mean_rep3=p["rep"].mean(), judge_fluent_rate=p["judge_fluent"].mean(),
            both_rate=(p["topic"] * p["coh"]).mean(),
            answers_question_rate=relev.mean(),  # post-hoc exploratory
            topic_coherent_and_answers_rate=(p["topic"] * p["coh"] * relev).mean(),
            mmlu_acc=acc.mean(), mmlu_delta=acc.mean() - base_acc.mean(),
            mmlu_delta_lo=ci(d_acc_b)[0], mmlu_delta_hi=ci(d_acc_b)[1],
            n_inf_ppl=int((~fin).sum()),
        )
        row["pass"] = bool(s > 0 and row["topic_lo"] > 0.80 and row["coh_loss_hi"] < 0.20)
        row["fail_confident"] = bool(row["topic_hi"] < 0.80 or row["coh_loss_lo"] > 0.20)
        rows_out.append(row)

with open(f"{RES}/metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader()
    for r in rows_out:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

cells = [r for r in rows_out if r["strength"] > 0 and not r["exploratory"]]
if any(r["pass"] for r in cells):
    verdict = "PROVEN"
elif all(r["fail_confident"] for r in cells):
    verdict = "DISPROVEN"
else:
    verdict = "INCONCLUSIVE"

# best cell: highest topic rate among cells with coherence loss point estimate < 20%
ok = [r for r in cells if r["coherence_loss"] < 0.20]
best = max(ok, key=lambda r: (r["topic_rate"], -r["coherence_loss"])) if ok else None
best_both = max(cells, key=lambda r: r["both_rate"])
summary = dict(verdict=verdict, ppl_threshold_T=T, baseline_coherent_rate=bp["coh"].mean(),
               baseline_topic_rate=bp["topic"].mean(), baseline_mmlu=base_acc.mean(),
               passing_cells=[(r["layer"], r["strength"]) for r in cells if r["pass"]],
               exploratory_passing_cells=[(r["layer"], r["strength"]) for r in rows_out if r["exploratory"] and r["pass"]],
               best_exploratory_cell=max([r for r in rows_out if r["exploratory"]] or [None],
                                         key=lambda r: r["both_rate"] if r else 0),
               best_cell_under_20pct_loss=best, best_topic_and_coherent_cell=best_both)
json.dump(summary, open(f"{RES}/summary.json", "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
for r in rows_out:
    print(f"L{r['layer']:>2} s={r['strength']:>4}: topic {r['topic_rate']:.2f} [{r['topic_lo']:.2f},{r['topic_hi']:.2f}] "
          f"strict {r['strict_eiffel_rate']:.2f} judge {r['judge_topic_rate']:.2f} | coh {r['coherent_rate']:.2f} "
          f"loss {r['coherence_loss']:+.2f} [{r['coh_loss_lo']:+.2f},{r['coh_loss_hi']:+.2f}] ppl {r['median_ppl']:.1f} "
          f"rep {r['mean_rep3']:.2f} fluentJ {r['judge_fluent_rate']:.2f} answersQ {r['answers_question_rate']:.2f} | mmlu {r['mmlu_acc']:.2f} {'PASS' if r['pass'] else ''}")
