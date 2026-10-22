"""Figures for the original steering run. Reads results/metrics.csv + results/raw, writes figures/."""
import csv, json, os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {12: "#2a6fdb", 18: "#e0702a", 24: "#2f9e5b"}
rows = list(csv.DictReader(open(f"{RES}/metrics.csv")))
for r in rows:
    for k, v in r.items():
        if k not in ("pass", "fail_confident", "exploratory"):
            r[k] = float(v)
    r["layer"] = int(r["layer"]); r["exploratory"] = r["exploratory"] == "True"
summ = json.load(open(f"{RES}/summary.json"))
C0 = summ["baseline_coherent_rate"]
by = {L: sorted([r for r in rows if r["layer"] == L], key=lambda r: r["strength"]) for L in COL}

# 1. topic & coherence vs strength
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, L in zip(axs, COL):
    rr = by[L]; x = [r["strength"] for r in rr]
    ax.fill_between(x, [r["topic_lo"] for r in rr], [r["topic_hi"] for r in rr], color=COL[L], alpha=.15)
    ax.plot(x, [r["topic_rate"] for r in rr], "o-", color=COL[L], label="Mentions Eiffel/Paris (regex)")
    ax.plot(x, [r["coherent_rate"] for r in rr], "s--", color="#555", label="Coherent (PPL + repetition)")
    ax.plot(x, [r["both_rate"] for r in rr], "^:", color="#b0228c", label="On-topic AND coherent")
    ax.plot(x, [r["answers_question_rate"] for r in rr], "x-", color="#999", lw=1, label="Still answers the question (judge, post-hoc)")
    ax.axhline(0.8, color="#999", lw=.8, ls=":")
    ax.set_xscale("symlog", linthresh=1); ax.set_xticks([0, .5, 1, 2, 4, 8, 16]); ax.set_xticklabels(["0", ".5", "1", "2", "4", "8", "16"])
    ax.set_title(f"Layer {L} of 36"); ax.set_xlabel("Steering strength s (× mean-difference vector)")
axs[0].set_ylabel("Share of 50 neutral prompts"); axs[0].legend(fontsize=9, loc="center left")
fig.suptitle("Qwen3-4B Eiffel-Tower steering: topic rate rises as coherence collapses", fontsize=15, y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/topic_coherence_vs_strength.png", bbox_inches="tight"); plt.close(fig)

# 2. trade-off curve
fig, ax = plt.subplots(figsize=(8, 6.2))
ax.add_patch(plt.Rectangle((0.8 * C0, 0.8), 1.05 - 0.8 * C0, 0.25, color="#ffd54a", alpha=.25, lw=0))
ax.text(0.8 * C0 + .01, 1.02, "Hypothesised sweet spot\n(>80% on-topic, <20% coherence loss)", fontsize=9, va="top")
for L, rr in by.items():
    ax.plot([r["coherent_rate"] for r in rr], [r["topic_rate"] for r in rr], "o-", color=COL[L], label=f"Layer {L}", alpha=.9)
    for r in rr:
        if r["strength"] in (0, 1, 2, 3, 4, 8) or r["exploratory"]:
            ax.annotate(f"{r['strength']:g}", (r["coherent_rate"], r["topic_rate"]), fontsize=8,
                        xytext=(4, 3), textcoords="offset points", color=COL[L])
ax.set_xlim(-.03, 1.05); ax.set_ylim(-.03, 1.05)
ax.set_xlabel("Coherence rate (share of outputs coherent)"); ax.set_ylabel("Topic rate (share mentioning Eiffel/Paris)")
ax.set_title("Steer rate vs coherence (labels = strength s)"); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIG}/steer_vs_coherence.png"); plt.close(fig)

# 3. MMLU
fig, ax = plt.subplots(figsize=(8, 4.8))
for L, rr in by.items():
    x = [r["strength"] for r in rr]
    ax.plot(x, [r["mmlu_acc"] for r in rr], "o-", color=COL[L], label=f"Layer {L}")
ax.axhline(.25, color="#999", ls=":", lw=1); ax.text(10, .265, "chance (25%)", fontsize=9, color="#777")
ax.set_xscale("symlog", linthresh=1); ax.set_xticks([0, .5, 1, 2, 4, 8, 16]); ax.set_xticklabels(["0", ".5", "1", "2", "4", "8", "16"])
ax.set_ylim(0, .8); ax.set_xlabel("Steering strength s"); ax.set_ylabel("MMLU accuracy (100 items)")
ax.set_title("Capability cost of steering: MMLU accuracy vs strength"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/mmlu_vs_strength.png"); plt.close(fig)

# 4. table
main = [r for r in rows if not r["exploratory"]]
hdr = ["Layer", "s", "Topic % [95% CI]", "Strict 'Eiffel' %", "Coherent %", "Coh. loss % [CI]", "Topic+coh. %", "Answers Q %†", "MMLU %"]
cell = [[f"{r['layer'] if r['strength'] else 'all'}", f"{r['strength']:g}{'*' if r['exploratory'] else ''}",
         f"{100*r['topic_rate']:.0f} [{100*r['topic_lo']:.0f}, {100*r['topic_hi']:.0f}]", f"{100*r['strict_eiffel_rate']:.0f}",
         f"{100*r['coherent_rate']:.0f}", f"{100*r['coherence_loss']:+.0f} [{100*r['coh_loss_lo']:+.0f}, {100*r['coh_loss_hi']:+.0f}]",
         f"{100*r['both_rate']:.0f}", f"{100*r['answers_question_rate']:.0f}", f"{100*r['mmlu_acc']:.0f}"] for r in rows if r["strength"] > 0 or r["layer"] == 12]
fig, ax = plt.subplots(figsize=(13, 0.32 * len(cell) + 1.2)); ax.axis("off")
t = ax.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.35)
for (i, j), c in t.get_celld().items():
    if i == 0: c.set_facecolor("#333"); c.get_text().set_color("white"); c.get_text().set_weight("bold")
    elif cell[i - 1][1].endswith("*"): c.set_facecolor("#f3f3f3")
ax.set_title("Qwen3-4B Eiffel steering: all cells (n=50 prompts; MMLU n=100; * = exploratory fine grid; † = post-hoc judge)", fontsize=12, pad=8)
fig.tight_layout(); fig.savefig(f"{FIG}/table.png", bbox_inches="tight"); plt.close(fig)

# 5. examples: 5 per strength, layers 12 and 24
RAW = os.path.join(RES, "raw")
S = [0, 1, 2, 2.5, 2.75, 3, 4, 8, 16]
ex_idx = [0, 7, 14, 21, 28]
for Lb in (12, 24):
  fig, ax= plt.subplots(figsize=(16, 2.3 * len(S))); ax.axis("off")
  y = 1.0
  for s in S:
      g = json.load(open(f"{RAW}/gen_L{0 if s == 0 else Lb}_s{s}.json"))
      ax.text(0, y, f"s = {s}", fontsize=13, weight="bold", va="top", transform=ax.transAxes); y -= 0.018
      for i in ex_idx:
          o = " ".join(g[i]["output"].split())[:170]
          line = f"Q: {g[i]['prompt'][:55]}  →  {o}"
          ax.text(0.01, y, textwrap.shorten(line, 230), fontsize=8.6, va="top", family="DejaVu Sans", transform=ax.transAxes); y -= 0.0215
      y -= 0.01
  ax.set_title(f"Five example outputs per strength (layer {Lb}, greedy, first 170 chars)", fontsize=14)
  fig.savefig(f"{FIG}/examples_L{Lb}.png", bbox_inches="tight"); plt.close(fig)
print("figures written")
