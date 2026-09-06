import csv, glob, statistics
from collections import defaultdict
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

files = glob.glob('/home/users/aadsilva/resultados/*.csv')
by_filter = defaultdict(list)
for f in files:
    with open(f) as fh:
        for row in csv.DictReader(fh):
            by_filter[row['filter_name']].append(row)

def vals(rows, key):
    return [float(r[key]) for r in rows if r.get(key) not in (None, '')]

complete = {f: rows for f, rows in by_filter.items() if len(rows) == 10}

names, auc_m, dice_m = [], [], []
for f, rows in complete.items():
    names.append(f)
    auc_m.append(statistics.mean(vals(rows, 'auc_roc')))
    dice_m.append(statistics.mean(vals(rows, 'dice_mean')))

auc_m = np.array(auc_m)
dice_m = np.array(dice_m)
r, p = stats.pearsonr(auc_m, dice_m)
slope, intercept, r_lr, p_lr, se = stats.linregress(auc_m, dice_m)
baseline_idx = names.index('baseline')

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ACCENT = "#1c6e62"
ACCENT_LINE = "#b45309"
POINT_GRAY = "#4b4b4b"

fig, ax = plt.subplots(figsize=(3.35, 2.6), dpi=600)

ax.scatter(auc_m, dice_m, s=7, c=POINT_GRAY, alpha=0.4, linewidths=0, zorder=2)

xline = np.linspace(auc_m.min() - 0.005, auc_m.max() + 0.005, 100)
yline = slope * xline + intercept
ax.plot(xline, yline, color=ACCENT_LINE, linewidth=1.3, zorder=3)

ax.scatter([auc_m[baseline_idx]], [dice_m[baseline_idx]], s=32, marker="D",
           facecolor=ACCENT, edgecolor="white", linewidths=0.5, zorder=4)

ax.set_xlabel("AUC-ROC")
ax.set_ylabel("Dice (Grad-CAM)")

ax.annotate(
    f"$r$ = {r:.3f}\n$R^2$ = {r**2:.3f}\n$n$ = {len(auc_m)}",
    xy=(0.04, 0.97), xycoords="axes fraction",
    ha="left", va="top", fontsize=7,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="#cccccc", linewidth=0.5),
)

# compact legend via proxy artists, short labels, upper area is taken by stat box
# so place legend at lower right, small
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=POINT_GRAY,
           markeredgewidth=0, alpha=0.7, markersize=5, label='condições'),
    Line2D([0], [0], color=ACCENT_LINE, linewidth=1.3, label='regressão'),
    Line2D([0], [0], marker='D', color='none', markerfacecolor=ACCENT,
           markeredgecolor='white', markeredgewidth=0.5, markersize=6, label='baseline'),
]
leg = ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9,
                 handletextpad=0.4, borderpad=0.35, labelspacing=0.25,
                 fontsize=7)
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_edgecolor("#cccccc")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, linewidth=0.35, alpha=0.3, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(length=2.5, width=0.6)

fig.tight_layout(pad=0.35)

out = "/tmp/claude-2161/-home-users-aadsilva-ic-hcpa-retinopathy-hcpa-repository-hcpa/9d1ef0b4-ef74-4b81-b847-b9ff27890ef6/scratchpad/paper_f/figura1_auc_dice.pdf"
fig.savefig(out)
out_png = out.replace(".pdf", ".png")
fig.savefig(out_png, dpi=600)
print("saved:", out)
print(f"r={r:.4f} R2={r**2:.4f} slope={slope:.4f} intercept={intercept:.4f} n={len(auc_m)}")
