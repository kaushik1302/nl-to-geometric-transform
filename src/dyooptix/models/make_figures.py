"""Report figures, drawn from the saved result files. Run after run_all.ps1."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "outputs/figures"
# One fixed colour per representation, the same in every figure
COLOURS = {"hand": "#2a78d6", "hand+minilm": "#eb6834", "minilm": "#1baf7a",
           "clip": "#eda100", "tfidf": "#e87ba4"}
GREY, DARK, TEXT = "#9a9893", "#2b2b29", "#52514e"
SPLIT_NAMES = {"test": "test", "test_unseen_templates": "unseen templates", "geobench": "GeoBench"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": TEXT, "axes.labelcolor": TEXT,
                     "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e5e1",
                     "axes.axisbelow": True})


def grouped_bars(ax, groups, series, values, colours, errors=None, fmt="{:.0f}"):
    """Groups along x; one bar per series in each group; the value printed on each bar."""
    width = 0.8 / len(series)
    x = np.arange(len(groups))
    for k, name in enumerate(series):
        err = errors.get(name) if errors else None
        bars = ax.bar(x - 0.4 + width * (k + 0.5), values[name], width, yerr=err, label=name,
                      color=colours[name], edgecolor="white", linewidth=1.5,
                      error_kw={"elinewidth": 1, "ecolor": TEXT, "capsize": 2})
        for i, bar in enumerate(bars):
            top = bar.get_height() + (err[i] if err is not None else 0)
            ax.text(bar.get_x() + bar.get_width() / 2, top, fmt.format(values[name][i]),
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, groups)


def finish(fig, ax, name, ylabel, title, ncol, legend_y=-0.12):
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10, color=DARK)
    ax.legend(ncol=ncol, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, legend_y))
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=200)
    plt.close(fig)
    print("saved", f"{OUT}/{name}")


def fig_representations():
    """Corner error of every MLP (mean ± std over 3 seeds) against the constant baseline and the oracle."""
    summary = pd.read_csv("outputs/mlp_summary.csv", header=[0, 1], index_col=[0, 1])
    baselines = pd.read_csv("outputs/baselines.csv")
    breakdown = pd.read_csv("outputs/breakdown.csv")
    splits = list(SPLIT_NAMES)
    reps = list(COLOURS)
    values = {"constant": [baselines.query("model == 'constant' and split == @s").corner_px.item() for s in splits]}
    errors = {}
    for rep in reps:
        values[rep] = [summary.loc[(f"mlp_{rep}", s), ("corner_px", "mean")] for s in splits]
        errors[rep] = [summary.loc[(f"mlp_{rep}", s), ("corner_px", "std")] for s in splits]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    grouped_bars(ax, list(SPLIT_NAMES.values()), ["constant"] + reps, values,
                 {"constant": GREY, **COLOURS}, errors)
    for i, s in enumerate(splits[:2]):  # the oracle exists only for the synthetic splits
        floor = breakdown.query("model == 'oracle' and split == @s and group == 'all'").corner_px.item()
        ax.hlines(floor, i - 0.4, i + 0.4, colors=DARK, linestyles="--", linewidth=1,
                  label="oracle (noise floor)" if i == 0 else None) # type: ignore
        ax.text(i - 0.42, floor, f"{floor:.0f}", ha="right", va="center", fontsize=6.5, color=DARK)
    finish(fig, ax, "fig1_representations.png", "corner error (px), lower is better",
           "Gated MLP by text representation (mean ± std, 3 seeds)", ncol=7)


def fig_ridge_vs_mlp():
    """Val corner error: linear (ridge) vs non-linear (MLP) on the same features."""
    summary = pd.read_csv("outputs/mlp_summary.csv", header=[0, 1], index_col=[0, 1])
    baselines = pd.read_csv("outputs/baselines.csv")
    reps = ["hand", "tfidf", "clip", "minilm"]
    values = {"ridge": [baselines.query("model == @m and split == 'val'").corner_px.item()
                        for m in [f"ridge_{r}" for r in reps]],
              "MLP": [summary.loc[(f"mlp_{r}", "val"), ("corner_px", "mean")] for r in reps]}
    errors = {"MLP": [summary.loc[(f"mlp_{r}", "val"), ("corner_px", "std")] for r in reps]}
    constant = baselines.query("model == 'constant' and split == 'val'").corner_px.item()

    fig, ax = plt.subplots(figsize=(6, 3.6))
    grouped_bars(ax, reps, ["ridge", "MLP"], values, {"ridge": GREY, "MLP": DARK}, errors)
    ax.axhline(constant, color=DARK, linestyle="--", linewidth=1, label=f"constant baseline ({constant:.0f})")
    finish(fig, ax, "fig2_ridge_vs_mlp.png", "val corner error (px)",
           "Same features, linear vs non-linear", ncol=3)


def fig_levels():
    """Test corner error by amount level (single edits) against the oracle, seed-0 models."""
    breakdown = pd.read_csv("outputs/breakdown.csv")
    levels = ["small", "medium", "large", "explicit"]
    models = {"hand": "mlp_hand", "hand+minilm": "mlp_hand+minilm", "minilm": "mlp_minilm", "oracle": "oracle"}
    rows = breakdown.query("split == 'test' and group == 'level'").set_index(["model", "value"])
    values = {name: [rows.loc[(m, lv), "corner_px"] for lv in levels] for name, m in models.items()}
    n = [int(rows.loc[("oracle", lv), "n"]) for lv in levels] # type: ignore

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    grouped_bars(ax, [f"{lv}\n(n={k})" for lv, k in zip(levels, n)], list(models), values,
                 {**COLOURS, "oracle": GREY})
    finish(fig, ax, "fig3_levels.png", "test corner error (px)",
           "Error by amount level, against the noise floor (oracle)", ncol=4, legend_y=-0.2)


def fig_human_direction():
    """Human test set: direction accuracy by how the direction was phrased."""
    edits = pd.read_csv("outputs/human_edits.csv")
    labels = pd.read_csv("data/human/human_instructions.csv", keep_default_na=False)
    labels["line"] = labels.index + 2
    m = edits.merge(labels[["line", "translate_dir"]], on="line")
    models = ["hand", "hand+minilm", "minilm"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True,
                                      gridspec_kw={"width_ratios": [2, 1]})
    t = m[m.kind == "translate"]
    dirs = ["left", "right", "up", "down"]
    n = t[t.model == "hand"].translate_dir.value_counts()
    acc = t.groupby(["model", "translate_dir"]).direction_ok.mean()
    grouped_bars(left, [f"{d}\n(n={n[d]})" for d in dirs], models,
                 {mod: [acc[(mod, d)] for d in dirs] for mod in models}, COLOURS, fmt="{:.2f}")
    left.set_title("Translate: by labelled direction", loc="left", fontsize=9, color=DARK)

    r = m[m.kind == "rotate"].copy()
    r["says"] = np.where(r.instruction.str.lower().str.contains("clockwise"), "clockwise words", "left/right/other")
    groups = ["clockwise words", "left/right/other"]
    n = r[r.model == "hand"].says.value_counts()
    acc = r.groupby(["model", "says"]).direction_ok.mean()
    grouped_bars(right, [f"{g}\n(n={n[g]})" for g in groups], models,
                 {mod: [acc[(mod, g)] for g in groups] for mod in models}, COLOURS, fmt="{:.2f}")
    right.set_title("Rotate: by phrasing", loc="left", fontsize=9, color=DARK)

    left.set_ylim(0, 1.1)
    left.set_ylabel("direction correct (fraction)")
    left.legend(ncol=3, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.24))
    fig.suptitle("Human-written instructions: direction accuracy", x=0.01, ha="left", fontsize=10, color=DARK)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_human_direction.png", dpi=200)
    plt.close(fig)
    print("saved", f"{OUT}/fig4_human_direction.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_representations()
    fig_ridge_vs_mlp()
    fig_levels()
    fig_human_direction()


if __name__ == "__main__":
    main()