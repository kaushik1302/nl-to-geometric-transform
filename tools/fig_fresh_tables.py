"""Poster/report table: fresh-test-set scores (Part A whole image, Part B objects), read from the saved CSVs only."""
import matplotlib.pyplot as plt
import pandas as pd

UNIT = {"translate": "{:.0f} px", "rotate": "{:.0f}°", "scale": "{:.2f} |log s|"}   # how each error is written
ORDER = ["translate", "rotate", "scale"]
HEAD = ["Edit", "n", "Edit found", "Direction right", "Exact: median error (n)", "Vague: level right (n)"]


def rows(edits, label=""):
    """One table row per edit type: counts and rates from the per-edit CSV."""
    out = []
    for kind in ORDER:
        g = edits[edits["kind"] == kind]
        exact, vague = g["error"].dropna(), g["level_hit"].dropna()
        out.append([f"{kind}{label}", len(g), f"{g['detected'].mean():.0%}", f"{g['direction_ok'].mean():.0%}",
                    f"{UNIT[kind].format(exact.median())} ({len(exact)})" if len(exact) else "–",
                    f"{vague.mean():.0%} ({len(vague)})" if len(vague) else "–"])
    return out


def table(ax, title, body):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    t = ax.table(cellText=body, colLabels=HEAD, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.4)


a = pd.read_csv("outputs/fresh_human_fresh_a_edits.csv")
b = pd.read_csv("outputs/fresh_human_fresh_b_edits.csv")
sa = pd.read_csv("outputs/fresh_human_fresh_a_sentences.csv")
sb = pd.read_csv("outputs/fresh_human_fresh_b_sentences.csv")
present = sb[~sb["failed"]]                                      # failed rows = the 2 objects not in the photo

body_b = (rows(b[~b["clamped"] & ~b["failed"]], " (no clamp)") + rows(b[b["clamped"]], " (clamped)"))

fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 4.6), gridspec_kw={"height_ratios": [1, 1.7]})
table(top, f"Part A: {len(sa)} whole-image sentences   (all edit flags right: {sa['presence_ok'].mean():.0%})",
      rows(a))
table(bottom, f"Part B: {len(sb)} object sentences on 10 photos\n(right object: "
      f"{present['object_ok'].sum()}/{len(present)}, absent objects refused: {int(sb['failed'].sum())}/2, "
      f"all edit flags right: {present['presence_ok'].mean():.0%})", body_b)
fig.text(0.01, 0.01, "Clamped = the edit was reduced to keep the object inside the photo. "
         "Exact error: translate in px of a 512-px image, rotate in degrees, scale in |log s|.", fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("outputs/fig_fresh_tables.png", dpi=200)
print("saved outputs/fig_fresh_tables.png")