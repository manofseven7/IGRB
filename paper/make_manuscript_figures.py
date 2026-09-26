from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results" / "igrb_analysis"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(ROOT / name, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def topology():
    fig, ax = plt.subplots(figsize=(7.25, 3.45))
    ax.set_xlim(-0.4, 11.4)
    ax.set_ylim(-0.45, 3.35)
    ax.axis("off")
    supplier = (5.5, 2.9)
    dcs = [(1.8, 1.75), (5.5, 1.75), (9.2, 1.75)]
    terminals = [(0.4 + i * 0.95, 0.35) for i in range(12)]
    colors = {"s": "#2A5CAA", "d": "#E58E26", "t": "#25855A"}

    def node(xy, text, color, size=390):
        ax.scatter(*xy, s=size, c=color, ec="white", lw=1.2, zorder=4)
        ax.text(*xy, text, ha="center", va="center", color="white", weight="bold", zorder=5)

    node(supplier, "S", colors["s"], 510)
    for i, xy in enumerate(dcs, 1):
        node(xy, f"D{i}", colors["d"], 430)
    for i, xy in enumerate(terminals, 1):
        node(xy, f"T{i}", colors["t"], 250)

    for d in dcs:
        ax.add_patch(FancyArrowPatch(supplier, d, arrowstyle="-|>", mutation_scale=10,
                                     lw=1.5, color="#4A4A4A", shrinkA=15, shrinkB=15))
        ax.add_patch(FancyArrowPatch(d, supplier, arrowstyle="-|>", mutation_scale=9,
                                     lw=1.0, ls="--", color="#7A4EAB", shrinkA=15, shrinkB=15,
                                     connectionstyle="arc3,rad=.12"))
    for j, d in enumerate(dcs):
        for t in terminals[j * 4:(j + 1) * 4]:
            ax.add_patch(FancyArrowPatch(d, t, arrowstyle="-|>", mutation_scale=8,
                                         lw=1.2, color="#4A4A4A", shrinkA=14, shrinkB=10))
            ax.add_patch(FancyArrowPatch(t, d, arrowstyle="-|>", mutation_scale=7,
                                         lw=.8, ls="--", color="#7A4EAB", shrinkA=10, shrinkB=14,
                                         connectionstyle="arc3,rad=.08"))
    ax.plot([], [], color="#4A4A4A", lw=1.5, label="material and allocation flow")
    ax.plot([], [], color="#7A4EAB", lw=1, ls="--", label="orders and state information")
    ax.legend(loc="upper right", frameon=False)
    ax.text(5.5, -0.25, "One supplier, three distribution nodes and twelve demand terminals",
            ha="center", color="#333333")
    save(fig, "igrb_topology.pdf")


def fixed_performance():
    df = pd.read_csv(RESULTS / "main_summary.csv")
    df = df[df["shift"].eq("none")].copy()
    order = ["seasonal", "intermittent", "correlated", "retail"]
    df["case"] = pd.Categorical(df["case"], order, ordered=True)
    df = df.sort_values(["case", "method"])
    labels = ["Seasonal", "Intermittent", "Correlated", "Retail-driven"]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.1))
    for ax, metric, title, ylim in [
        (axes[0], "f1", "Macro F1", (0.60, 0.89)),
        (axes[1], "auprc", "Average precision", (0.40, 1.0)),
    ]:
        for k, (method, color, hatch) in enumerate([
            ("LocalXGBTuned", "#AAB7C4", ""), ("IGRB", "#2A5CAA", "///")
        ]):
            vals = [float(df[(df.case == c) & (df.method == method)][metric].iloc[0]) for c in order]
            ax.bar(x + (k - .5) * .34, vals, .34, color=color, edgecolor="white",
                   hatch=hatch, label="Local expert" if k == 0 else "IGRB")
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#DDDDDD", lw=.6)
        ax.set_axisbelow(True)
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.4)
    save(fig, "igrb_fixed_performance.pdf")


def independent_effects():
    df = pd.read_csv(RESULTS / "independent_effects_all.csv")
    df = df[df["case"].isin(["seasonal", "correlated"])].copy()
    order = [
        ("correlated", "none"), ("correlated", "lead"), ("correlated", "policy"),
        ("seasonal", "none"), ("seasonal", "lead"), ("seasonal", "policy"),
    ]
    labels = ["Correlated—unchanged", "Correlated—lead shift", "Correlated—policy shift",
              "Seasonal—unchanged", "Seasonal—lead shift", "Seasonal—policy shift"]
    rows = [df[(df.case == c) & (df["shift"] == s)].iloc[0] for c, s in order]
    y = np.arange(len(rows))[::-1]
    means = np.array([r.delta for r in rows])
    lo = np.array([r.ci_low for r in rows])
    hi = np.array([r.ci_high for r in rows])
    fig, ax = plt.subplots(figsize=(7.25, 3.25))
    colors = ["#2A5CAA"] * 3 + ["#E58E26"] * 3
    for yi, m, l, h, color in zip(y, means, lo, hi, colors):
        ax.errorbar(m, yi, xerr=[[m-l], [h-m]], fmt="o", color=color, ecolor=color,
                    elinewidth=1.6, capsize=3, ms=5)
    ax.axvline(0, color="#333333", lw=1, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired macro-F1 difference (IGRB minus local expert)")
    ax.grid(axis="x", color="#DDDDDD", lw=.6)
    ax.set_axisbelow(True)
    ax.text(.99, .03, "Bars: 95% paired bootstrap intervals over demand records",
            transform=ax.transAxes, ha="right", color="#555555", fontsize=7.5)
    fig.tight_layout()
    save(fig, "igrb_independent_effects.pdf")


def policy():
    df = pd.read_csv(RESULTS / "policy_summary.csv")
    df = df[df["case"].eq("correlated")]
    order = ["none", "lead", "policy"]
    labels = ["Unchanged", "Lead-time shift", "Policy shift"]
    local = np.array([df[(df["shift"] == s) & (df.method == "LocalXGBTuned")].cost.iloc[0] for s in order])
    igrb = np.array([df[(df["shift"] == s) & (df.method == "IGRB")].cost.iloc[0] for s in order])
    savings = 100 * (local - igrb) / local
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.25, 2.95))
    bars = ax.bar(x, savings, .55, color=["#7BAFD4", "#2A5CAA", "#163D73"])
    ax.axhline(0, color="#333333", lw=.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cost reduction relative to local expert (%)")
    ax.set_ylim(0, max(savings) * 1.25)
    ax.grid(axis="y", color="#DDDDDD", lw=.6)
    ax.set_axisbelow(True)
    for b, val in zip(bars, savings):
        ax.text(b.get_x() + b.get_width()/2, val + .06, f"{val:.2f}%", ha="center", weight="bold")
    ax.text(.99, .95, "Guardrail abstains in the other three demand cases",
            transform=ax.transAxes, ha="right", va="top", color="#555555")
    fig.tight_layout()
    save(fig, "igrb_policy_savings.pdf")


def all_baselines():
    """Fixed-realization comparison using the audited legacy benchmark plus IGRB."""
    legacy = {
        "Persistence": [0.623, 0.671, 0.770, 0.848],
        "Local balance": [0.438, 0.582, 0.673, 0.748],
        "XGBoost (original)": [0.608, 0.682, 0.786, 0.861],
        "Pooled LSTM": [0.610, 0.682, 0.778, 0.570],
        "Mean GNN": [0.611, 0.662, 0.766, 0.485],
        "Relational mean GNN": [0.627, 0.662, 0.786, 0.594],
        "Mean GNN + GRU": [0.609, 0.666, 0.745, 0.327],
        "Homogeneous attention": [0.606, 0.661, 0.786, 0.614],
        "Relational attention + GRU": [0.641, 0.662, 0.760, 0.416],
        "HetST-GNN": [0.640, 0.671, 0.780, 0.635],
    }
    summary = pd.read_csv(RESULTS / "main_summary.csv")
    summary = summary[summary["shift"].eq("none")]
    case_order = ["seasonal", "intermittent", "correlated", "retail"]
    for method, label in [("LocalXGBTuned", "Tuned local expert"), ("IGRB", "IGRB (proposed)")]:
        legacy[label] = [
            float(summary[(summary.case == case) & (summary.method == method)].f1.iloc[0])
            for case in case_order
        ]
    table = pd.DataFrame.from_dict(
        legacy, orient="index", columns=["Seasonal", "Intermittent", "Correlated", "Retail-driven"]
    )
    table.index.name = "Method"
    table.to_csv(RESULTS / "all_baseline_comparison.csv", float_format="%.6f")

    values = table.to_numpy()
    fig, ax = plt.subplots(figsize=(7.25, 5.15))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0.30, vmax=0.88, aspect="auto")
    ax.set_xticks(np.arange(table.shape[1]), table.columns)
    ax.set_yticks(np.arange(table.shape[0]), table.index)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(length=0)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            color = "white" if values[i, j] > 0.73 else "#1F2933"
            weight = "bold" if table.index[i] == "IGRB (proposed)" else "normal"
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center",
                    color=color, weight=weight, fontsize=7.8)
    maxima = values.max(axis=0)
    for j, maximum in enumerate(maxima):
        for i in np.flatnonzero(np.isclose(values[:, j], maximum, atol=5e-4)):
            ax.add_patch(plt.Rectangle((j-.49, i-.49), .98, .98, fill=False,
                                       ec="#C73E1D", lw=1.8))
    ax.axhline(9.5, color="white", lw=2.4)
    ax.axhline(10.5, color="#C73E1D", lw=1.2, ls="--")
    cbar = fig.colorbar(image, ax=ax, fraction=.035, pad=.025)
    cbar.set_label("Macro F1")
    ax.set_xlabel("Fixed unchanged benchmark; mean over five model seeds")
    ax.xaxis.set_label_position("bottom")
    fig.tight_layout()
    save(fig, "igrb_all_baselines.pdf")


if __name__ == "__main__":
    topology()
    fixed_performance()
    independent_effects()
    policy()
    all_baselines()
