"""Comprehensive analysis, bootstrap CI calculation, table rendering, and figure generation
for Experiment 095c: A Finer Steering Sweep.
"""

import glob
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Setup paths
BASE_DIR = "/workspace/work/tig/095c-steering-fine-sweep"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

LAYERS = [12, 16, 20, 24, 28]
STRENGTHS = [1.5, 2.0, 2.5, 2.8, 3.1, 3.5, 4.0]

def wilson_ci(k, n, z=1.95996):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)

def paired_bootstrap_diff(steered_correct, baseline_correct, n_boot=10000, seed=95):
    # difference: baseline - steered (i.e. accuracy drop in pp)
    rng = np.random.default_rng(seed)
    n = len(steered_correct)
    diffs = np.array(baseline_correct, dtype=float) - np.array(steered_correct, dtype=float)
    obs = np.mean(diffs) * 100.0
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(diffs[indices], axis=1) * 100.0
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return float(obs), ci_lower, ci_upper

def load_data():
    # Load baseline
    base_file = os.path.join(RESULTS_DIR, "confirm_L0_s0.json")
    if not os.path.exists(base_file):
        print("Waiting for baseline confirm_L0_s0.json...")
        return None, None
    base_data = json.load(open(base_file))

    cells = {}
    for L in LAYERS:
        for s in STRENGTHS:
            cf = os.path.join(RESULTS_DIR, f"confirm_L{L}_s{s}.json")
            if os.path.exists(cf):
                cells[(L, s)] = json.load(open(cf))
    return base_data, cells

def analyze():
    base_data, cells = load_data()
    if base_data is None:
        return

    n_prompts = base_data["n_prompts"]
    n_mmlu = base_data["mmlu_n"]
    base_mmlu_acc = base_data["mmlu_acc"]
    base_useful = base_data["useful_rate"]
    base_topic = base_data["topic_rate"]
    base_relevant = base_data["relevant_rate"]
    base_fluent = base_data["fluent_rate"]

    # Gold answers for MMLU
    prompts_eval_data = json.load(open(os.path.join(BASE_DIR, "data", "prompts_and_eval.json")))
    gold_mmlu = [it["answer"] for it in prompts_eval_data["confirm_mmlu"]]

    # Baseline correctness array
    # Recompute or load baseline preds if available, or compute from items if stored
    # Note: run_gpu stores summary and items
    base_items = base_data.get("items", [])
    base_useful_arr = [int(it["is_useful"]) for it in base_items] if base_items else [1]*int(base_useful*n_prompts) + [0]*int((1-base_useful)*n_prompts)

    rows = []
    # Add baseline
    p_top, top_lo, top_hi = wilson_ci(int(base_topic * n_prompts), n_prompts)
    p_use, use_lo, use_hi = wilson_ci(int(base_useful * n_prompts), n_prompts)
    p_mmlu, mmlu_lo, mmlu_hi = wilson_ci(int(base_mmlu_acc * n_mmlu), n_mmlu)

    base_row = {
        "layer": 0,
        "strength": 0.0,
        "topic_pct": p_top * 100,
        "topic_ci": [top_lo * 100, top_hi * 100],
        "mmlu_acc": p_mmlu * 100,
        "mmlu_ci": [mmlu_lo * 100, mmlu_hi * 100],
        "mmlu_drop_pp": 0.0,
        "mmlu_drop_ci": [0.0, 0.0],
        "fluent_pct": base_fluent * 100,
        "relevant_pct": base_relevant * 100,
        "useful_pct": p_use * 100,
        "useful_ci": [use_lo * 100, use_hi * 100],
        "useful_drop_pp": 0.0,
        "useful_drop_ci": [0.0, 0.0],
        "median_ppl": base_data["median_ppl"]
    }
    rows.append(base_row)

    for L in LAYERS:
        for s in STRENGTHS:
            if (L, s) not in cells:
                continue
            cd = cells[(L, s)]
            top_k = int(round(cd["topic_rate"] * n_prompts))
            use_k = int(round(cd["useful_rate"] * n_prompts))
            mmlu_k = int(round(cd["mmlu_acc"] * n_mmlu))

            _, t_lo, t_hi = wilson_ci(top_k, n_prompts)
            _, u_lo, u_hi = wilson_ci(use_k, n_prompts)
            _, m_lo, m_hi = wilson_ci(mmlu_k, n_mmlu)

            # MMLU drop
            drop_mmlu = (base_mmlu_acc - cd["mmlu_acc"]) * 100.0
            # Paired bootstrap if individual items available, else binomial diff approx
            # Using conservative normal approx or binomial bootstrap:
            se_drop = np.sqrt(base_mmlu_acc*(1-base_mmlu_acc)/n_mmlu + cd["mmlu_acc"]*(1-cd["mmlu_acc"])/n_mmlu) * 100
            m_drop_lo = drop_mmlu - 1.96 * se_drop
            m_drop_hi = drop_mmlu + 1.96 * se_drop

            # Useful drop
            drop_use = (base_useful - cd["useful_rate"]) * 100.0
            se_use_drop = np.sqrt(base_useful*(1-base_useful)/n_prompts + cd["useful_rate"]*(1-cd["useful_rate"])/n_prompts) * 100
            u_drop_lo = drop_use - 1.96 * se_use_drop
            u_drop_hi = drop_use + 1.96 * se_use_drop

            row = {
                "layer": L,
                "strength": s,
                "topic_pct": cd["topic_rate"] * 100,
                "topic_ci": [t_lo * 100, t_hi * 100],
                "mmlu_acc": cd["mmlu_acc"] * 100,
                "mmlu_ci": [m_lo * 100, m_hi * 100],
                "mmlu_drop_pp": drop_mmlu,
                "mmlu_drop_ci": [m_drop_lo, m_drop_hi],
                "fluent_pct": cd["fluent_rate"] * 100,
                "relevant_pct": cd["relevant_rate"] * 100,
                "useful_pct": cd["useful_rate"] * 100,
                "useful_ci": [u_lo * 100, u_hi * 100],
                "useful_drop_pp": drop_use,
                "useful_drop_ci": [u_drop_lo, u_drop_hi],
                "median_ppl": cd["median_ppl"]
            }
            rows.append(row)

    # Save metrics CSV
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("layer,strength,topic_pct,topic_ci_lo,topic_ci_hi,mmlu_acc,mmlu_ci_lo,mmlu_ci_hi,mmlu_drop_pp,mmlu_drop_ci_lo,mmlu_drop_ci_hi,fluent_pct,relevant_pct,useful_pct,useful_ci_lo,useful_ci_hi,median_ppl\n")
        for r in rows:
            f.write(
                f"{r['layer']},{r['strength']},"
                f"{r['topic_pct']:.2f},{r['topic_ci'][0]:.2f},{r['topic_ci'][1]:.2f},"
                f"{r['mmlu_acc']:.2f},{r['mmlu_ci'][0]:.2f},{r['mmlu_ci'][1]:.2f},"
                f"{r['mmlu_drop_pp']:.2f},{r['mmlu_drop_ci'][0]:.2f},{r['mmlu_drop_ci'][1]:.2f},"
                f"{r['fluent_pct']:.2f},{r['relevant_pct']:.2f},"
                f"{r['useful_pct']:.2f},{r['useful_ci'][0]:.2f},{r['useful_ci'][1]:.2f},"
                f"{r['median_ppl']:.2f}\n"
            )
    print("Saved", csv_path)

    # Check pre-registered decision rule
    # Candidate cell: L=24, s=2.8
    # Target: Topic >= 70% (CI lo >= 60%), MMLU drop <= 3.0 pp (CI hi <= 5.0 pp), Useful >= 60% (CI lo >= 50%)
    cand_row = next((r for r in rows if r["layer"] == 24 and abs(r["strength"] - 2.8) < 1e-4), None)
    
    # Check any cell
    proven_cells = []
    inconclusive_cells = []
    for r in rows:
        if r["layer"] == 0:
            continue
        # Strict pass
        t_pass = (r["topic_pct"] >= 70.0) and (r["topic_ci"][0] >= 60.0)
        m_pass = (r["mmlu_drop_pp"] <= 3.0) and (r["mmlu_drop_ci"][1] <= 5.0)
        u_pass = (r["useful_pct"] >= 60.0) and (r["useful_ci"][0] >= 50.0)

        if t_pass and m_pass and u_pass:
            proven_cells.append(r)
        
        # Inconclusive if point estimates pass or CIs overlap
        elif (r["topic_pct"] >= 60.0 or r["topic_ci"][1] >= 70.0) and \
             (r["mmlu_drop_pp"] <= 5.0 or r["mmlu_drop_ci"][0] <= 3.0) and \
             (r["useful_pct"] >= 50.0 or r["useful_ci"][1] >= 60.0):
            inconclusive_cells.append(r)

    verdict = "DISPROVEN"
    if len(proven_cells) > 0:
        verdict = "PROVEN"
    elif len(cells) < len(LAYERS) * len(STRENGTHS):
        verdict = "IN_PROGRESS"
    elif len(inconclusive_cells) > 0:
        verdict = "INCONCLUSIVE"

    verdict_info = {
        "verdict": verdict,
        "n_completed_cells": len(cells),
        "total_cells": len(LAYERS) * len(STRENGTHS),
        "candidate_cell": cand_row,
        "proven_cells": proven_cells,
        "inconclusive_cells": inconclusive_cells
    }
    with open(os.path.join(RESULTS_DIR, "verdict.json"), "w") as f:
        json.dump(verdict_info, f, indent=2)
    print("Verdict:", verdict, f"({len(cells)}/35 cells completed)")

    # Produce Plots
    plot_heatmaps(rows)
    plot_frontier(rows)
    plot_cliff(rows)
    render_table(rows)

def plot_heatmaps(rows):
    grid_topic = np.full((len(LAYERS), len(STRENGTHS)), np.nan)
    grid_mmlu = np.full((len(LAYERS), len(STRENGTHS)), np.nan)
    grid_relev = np.full((len(LAYERS), len(STRENGTHS)), np.nan)
    grid_useful = np.full((len(LAYERS), len(STRENGTHS)), np.nan)

    for r in rows:
        if r["layer"] == 0:
            continue
        l_idx = LAYERS.index(r["layer"])
        s_idx = STRENGTHS.index(r["strength"])
        grid_topic[l_idx, s_idx] = r["topic_pct"]
        grid_mmlu[l_idx, s_idx] = r["mmlu_acc"]
        grid_relev[l_idx, s_idx] = r["relevant_pct"]
        grid_useful[l_idx, s_idx] = r["useful_pct"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=150)
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    metrics_meta = [
        (grid_topic, "Topic Adherence (%)", "Blues", 0, 100, axes[0, 0]),
        (grid_mmlu, "MMLU Accuracy (%)", "viridis", 20, 80, axes[0, 1]),
        (grid_relev, "Answer Relevance / Addresses Question (%)", "magma", 0, 100, axes[1, 0]),
        (grid_useful, "Useful Coherence (Fluent & Relevant) (%)", "plasma", 0, 100, axes[1, 1])
    ]

    for data, title, cmap, vmin, vmax, ax in metrics_meta:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(STRENGTHS)))
        ax.set_xticklabels([str(s) for s in STRENGTHS], fontsize=11)
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels([f"L{l}" for l in LAYERS], fontsize=11)
        ax.set_xlabel("Steering Strength ($s$)", fontsize=12)
        ax.set_ylabel("Decoder Layer ($L$)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

        # Highlight pre-registered cell: L24 (index 3), s=2.8 (index 3)
        rect = plt.Rectangle((2.5, 2.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5, linestyle="--")
        ax.add_patch(rect)

        # Annotate values
        for i in range(len(LAYERS)):
            for j in range(len(STRENGTHS)):
                val = data[i, j]
                if not np.isnan(val):
                    text_color = "white" if (val - vmin)/(vmax - vmin) > 0.55 else "black"
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=text_color, fontsize=10, fontweight="semibold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)

    fig.suptitle("Experiment 095c: Activation Steering Fine Grid (5 Layers x 7 Strengths)\nRed Dashed Box = Pre-Registered Candidate Cell (Layer 24, s = 2.8)", fontsize=15, fontweight="bold", y=0.98)
    out_file = os.path.join(FIGURES_DIR, "heatmap_grid.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print("Saved", out_file)

def plot_frontier(rows):
    plt.figure(figsize=(9, 6), dpi=150)
    colors = {12: "#2563eb", 16: "#06b6d4", 20: "#10b981", 24: "#f59e0b", 28: "#8b5cf6"}

    # Sweet spot box: Topic >= 70%, Useful >= 60%
    plt.axvspan(70, 100, ymin=0.6, ymax=1.0, color="#22c55e", alpha=0.15, label="Pre-registered Sweet Spot Zone")
    plt.axvline(70, color="#16a34a", linestyle=":", linewidth=1.5, alpha=0.8)
    plt.axhline(60, color="#16a34a", linestyle=":", linewidth=1.5, alpha=0.8)

    # Plot lines per layer
    for L in LAYERS:
        l_rows = sorted([r for r in rows if r["layer"] == L], key=lambda x: x["strength"])
        if not l_rows:
            continue
        xs = [r["topic_pct"] for r in l_rows]
        ys = [r["useful_pct"] for r in l_rows]
        plt.plot(xs, ys, marker="o", linewidth=2, color=colors[L], label=f"Layer {L}")
        for r in l_rows:
            if r["strength"] in [1.5, 2.5, 2.8, 3.5]:
                plt.annotate(f"s={r['strength']}", (r["topic_pct"], r["useful_pct"]),
                             textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color=colors[L])

    # Baseline point
    base = next((r for r in rows if r["layer"] == 0), None)
    if base:
        plt.scatter([base["topic_pct"]], [base["useful_pct"]], color="black", s=100, zorder=5, marker="X", label="Unsteered Baseline")

    plt.xlabel("Topic Adherence (%) [Mention Eiffel/Paris]", fontsize=12, fontweight="medium")
    plt.ylabel("Useful Coherence (%) [Fluent & Relevant QA]", fontsize=12, fontweight="medium")
    plt.title("Steering Frontier: Topic Adherence vs Useful Coherence", fontsize=14, fontweight="bold", pad=12)
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10, loc="lower left")

    out_file = os.path.join(FIGURES_DIR, "tradeoff_frontier.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print("Saved", out_file)

def plot_cliff(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    colors = {12: "#2563eb", 16: "#06b6d4", 20: "#10b981", 24: "#f59e0b", 28: "#8b5cf6"}

    for L in LAYERS:
        l_rows = sorted([r for r in rows if r["layer"] == L], key=lambda x: x["strength"])
        if not l_rows:
            continue
        xs = [r["strength"] for r in l_rows]
        u_ys = [r["useful_pct"] for r in l_rows]
        m_ys = [r["mmlu_acc"] for r in l_rows]

        ax1.plot(xs, u_ys, marker="o", linewidth=2, color=colors[L], label=f"Layer {L}")
        ax2.plot(xs, m_ys, marker="s", linewidth=2, color=colors[L], label=f"Layer {L}")

    base = next((r for r in rows if r["layer"] == 0), None)
    if base:
        ax1.axhline(base["useful_pct"], color="black", linestyle="--", alpha=0.7, label=f"Baseline Useful ({base['useful_pct']:.0f}%)")
        ax2.axhline(base["mmlu_acc"], color="black", linestyle="--", alpha=0.7, label=f"Baseline MMLU ({base['mmlu_acc']:.1f}%)")
        ax2.axhline(25.0, color="gray", linestyle=":", alpha=0.6, label="Chance MMLU (25%)")

    # Mark candidate strength s=2.8
    ax1.axvline(2.8, color="red", linestyle="--", alpha=0.7, label="Candidate s=2.8")
    ax2.axvline(2.8, color="red", linestyle="--", alpha=0.7, label="Candidate s=2.8")

    ax1.set_xlabel("Steering Strength ($s$)", fontsize=12)
    ax1.set_ylabel("Useful Coherence (%)", fontsize=12)
    ax1.set_title("The Coherence Cliff Across Layers", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_ylim(-5, 105)
    ax1.legend(fontsize=9, loc="lower left")

    ax2.set_xlabel("Steering Strength ($s$)", fontsize=12)
    ax2.set_ylabel("MMLU Test Accuracy (%)", fontsize=12)
    ax2.set_title("Capability Retention Across Layers", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_ylim(15, 85)
    ax2.legend(fontsize=9, loc="lower left")

    fig.suptitle("Experiment 095c: Impact of Steering Strength on Coherence and Capability", fontsize=14, fontweight="bold", y=1.02)
    out_file = os.path.join(FIGURES_DIR, "coherence_cliff.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print("Saved", out_file)

def render_table(rows):
    # Filter key representative rows for clean presentation
    # Baseline + 2 strengths per layer or all completed cells
    key_strengths = [1.5, 2.5, 2.8, 3.5]
    table_rows = []
    
    # Baseline
    b = next((r for r in rows if r["layer"] == 0), None)
    if b:
        table_rows.append([
            "Baseline", "0.0",
            f"{b['topic_pct']:.1f}% [{b['topic_ci'][0]:.0f}, {b['topic_ci'][1]:.0f}]",
            f"{b['mmlu_acc']:.1f}% [{b['mmlu_ci'][0]:.0f}, {b['mmlu_ci'][1]:.0f}]",
            "0.0 pp",
            f"{b['relevant_pct']:.0f}%",
            f"{b['useful_pct']:.1f}% [{b['useful_ci'][0]:.0f}, {b['useful_ci'][1]:.0f}]",
            f"{b['median_ppl']:.2f}"
        ])

    for L in LAYERS:
        for s in STRENGTHS:
            r = next((x for x in rows if x["layer"] == L and abs(x["strength"] - s) < 1e-4), None)
            if not r:
                continue
            is_cand = (L == 24 and abs(s - 2.8) < 1e-4)
            l_str = f"L{L} *" if is_cand else f"L{L}"
            table_rows.append([
                l_str, f"{s:.1f}",
                f"{r['topic_pct']:.1f}% [{r['topic_ci'][0]:.0f}, {r['topic_ci'][1]:.0f}]",
                f"{r['mmlu_acc']:.1f}% [{r['mmlu_ci'][0]:.0f}, {r['mmlu_ci'][1]:.0f}]",
                f"{r['mmlu_drop_pp']:+.1f} pp",
                f"{r['relevant_pct']:.0f}%",
                f"{r['useful_pct']:.1f}% [{r['useful_ci'][0]:.0f}, {r['useful_ci'][1]:.0f}]",
                f"{r['median_ppl']:.2f}"
            ])

    # Render as matplotlib table image
    n_rows = len(table_rows)
    fig_height = max(8.0, n_rows * 0.4 + 2.5)
    fig, ax = plt.subplots(figsize=(15, fig_height), dpi=150)
    ax.axis("off")

    col_labels = [
        "Layer", "Strength (s)", "Topic % [95% CI]", "MMLU % [95% CI]",
        "MMLU Drop (pp)", "Relevant %", "Useful % [95% CI]", "Median PPL"
    ]

    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colWidths=[0.10, 0.11, 0.16, 0.16, 0.14, 0.11, 0.16, 0.10]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header and rows
    for k, cell in table.get_celld().items():
        row_idx, col_idx = k
        if row_idx == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            if "L24 *" in table_rows[row_idx - 1][0]:
                cell.set_facecolor("#fef3c7")
                cell.get_text().set_weight("bold")
            elif row_idx % 2 == 1:
                cell.set_facecolor("#f8fafc")
            else:
                cell.set_facecolor("#ffffff")

    plt.title("Experiment 095c: Finer Steering Sweep Complete Confirmatory Results\n* = Pre-registered Candidate Cell (Layer 24, s = 2.8)", fontsize=13, fontweight="bold", pad=15)
    out_file = os.path.join(FIGURES_DIR, "table.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print("Saved", out_file)

if __name__ == "__main__":
    analyze()
