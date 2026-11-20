"""Poster charts from the fresh-test-set CSVs: direction right, vague level right, and failure causes."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KINDS = ["translate", "rotate", "scale"]
INK, GREY = "#0b0b0b", "#52514e"                                # text colours
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]                     # one colour per group, same in both charts

a = pd.read_csv("outputs/fresh_human_fresh_a_edits.csv")
b = pd.read_csv("outputs/fresh_human_fresh_b_edits.csv")
b = b[~b["failed"]]                                             # drop the 2 objects that are not in the photo
GROUPS = {"Part A: whole image": a,
          "Part B: object, not clamped": b[~b["clamped"]],
          "Part B: object, clamped": b[b["clamped"]]}

# Failure causes: sentence lines grouped by hand after reading each wrong row (A = Part A, B = Part B).
CAUSES = {"Diagonal move kept on one axis": (14, 5),
          "No size direction in the sentence": (5, 0),
          '"upward" / "upper" read as down': (3, 1),
          "Idiom or missed clause": (4, 0),
          "Clamp changed the edit": (0, 4),
          "Exact factor inverted": (1, 1),
          "Typo in the sentence": (2, 0),
          "Left / right swapped": (0, 1)}


def grouped_bars(column, title, path, groups=GROUPS):
    """One bar per group for each edit type; each bar labelled with its % and n."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.26
    for i, (name, df) in enumerate(groups.items()):
        for k, kind in enumerate(KINDS):
            values = df.loc[df["kind"] == kind, column].dropna()
            if values.empty:
                continue
            x = k + (i - (len(groups) - 1) / 2) * width            # bars centred on the edit type
            ax.bar(x, values.mean(), width - 0.02, color=COLOURS[i], label=name if k == 0 else None)
            ax.text(x, values.mean() + 0.02, f"{values.mean():.0%}\nn={len(values)}",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(KINDS)), KINDS)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=GREY)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=len(groups), bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("saved", path)


def causes_chart(path):
    """Horizontal bars: number of wrong sentences per cause, split into Part A and Part B."""
    names = list(CAUSES)[::-1]                                  # biggest cause at the top
    part_a = np.array([CAUSES[n][0] for n in names])
    part_b = np.array([CAUSES[n][1] for n in names])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(names, part_a, color=COLOURS[0], label="Part A", edgecolor="white", linewidth=1.5)
    ax.barh(names, part_b, left=part_a, color=COLOURS[1], label="Part B", edgecolor="white", linewidth=1.5)
    for y, total in enumerate(part_a + part_b):
        ax.text(total + 0.3, y, str(total), va="center", fontsize=8, color=INK)
    ax.set_xticks(range(0, 21, 5))
    ax.set_xlabel("wrong sentences", color=GREY)
    ax.set_title("Why direction went wrong (fresh set)", loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=GREY)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("saved", path)


grouped_bars("direction_ok", "Direction right", "outputs/fig_fresh_direction.png")
not_clamped = dict(list(GROUPS.items())[:2])                    # clamped vague edits: only 1-2 sentences, left out
grouped_bars("level_hit", "Vague words (\"a little\", \"a lot\"): right size level", "outputs/fig_fresh_vague.png",
             not_clamped)
causes_chart("outputs/fig_fresh_causes.png")