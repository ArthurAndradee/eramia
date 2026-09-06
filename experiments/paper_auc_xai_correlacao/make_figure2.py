"""
Figura 2: confirmação multi-método (10 condições) -- painel A: AUC-ROC x
Dice para Grad-CAM/LIME/Occlusion, com reta de regressão para os métodos
com correlação significativa; painel B: sanity check de Adebayo, rho de
Spearman médio (entre as 10 condições) por etapa de aleatorização, por
método -- destaca a condição em que o LIME falha o sanity check (rho não
cai para zero).

Le os CSVs reais em resultados/*.csv (reps 10-19) e os JSONs em
resultados/sanity_check/*.json (ver pipeline/run_sanity_check.py).
"""
import csv, glob, json, statistics
from collections import defaultdict
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

FILTERS = [
    "baseline", "BenGraham_MaxGreen2.0", "BenGraham_GreenChannel_CLAHE4.0_Otsu",
    "Gamma0.8_MaxGreen2.0_Otsu", "BenGraham_MaxGreen2.0_Canny", "Retinex_Gaussian",
    "Retinex_MaxGreen2.0_Gaussian", "Retinex_MaxGreen2.0_CLAHE4.0_Otsu",
    "Retinex_Grayscale_CLAHE4.0_Otsu", "LABNorm_MaxGreen2.0_Frangi",
]
METHODS = ["gradcam", "lime", "occlusion"]
METHOD_LABEL = {"gradcam": "Grad-CAM", "lime": "LIME", "occlusion": "Occlusion"}
METHOD_COLOR = {"gradcam": "#1c6e62", "lime": "#b45309", "occlusion": "#5b5ea6"}
METHOD_MARKER = {"gradcam": "o", "lime": "s", "occlusion": "^"}

# ---------- Painel A: AUC x Dice, 10 condicoes x 3 metodos ----------
rows_by_filter_method = defaultdict(list)
for f in FILTERS:
    for rep in range(10, 20):
        files = sorted(glob.glob(f"/home/users/aadsilva/resultados/{f}_{rep}_*.csv"))
        if not files:
            continue
        with open(files[-1]) as fh:
            for row in csv.DictReader(fh):
                rows_by_filter_method[(f, row["xai_method"])].append(row)

data = {m: {"auc": [], "dice": []} for m in METHODS}
for f in FILTERS:
    for m in METHODS:
        rows = rows_by_filter_method.get((f, m), [])
        if not rows:
            continue
        data[m]["auc"].append(statistics.mean(float(r["auc_roc"]) for r in rows))
        data[m]["dice"].append(statistics.mean(float(r["dice_mean"]) for r in rows))

stats_by_method = {}
for m in METHODS:
    auc = np.array(data[m]["auc"]); dice = np.array(data[m]["dice"])
    r, p = stats.pearsonr(auc, dice)
    slope, intercept, *_ = stats.linregress(auc, dice)
    stats_by_method[m] = dict(auc=auc, dice=dice, r=r, p=p, slope=slope, intercept=intercept)

# ---------- Painel B: sanity check, rho medio por etapa e metodo ----------
sanity_dir = "/home/users/aadsilva/resultados/sanity_check"
steps_by_method = defaultdict(lambda: defaultdict(list))  # method -> step -> [rho,...]
lime_outlier = None
lime_outlier_filter = None
for f in FILTERS:
    path = f"{sanity_dir}/{f}.json"
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        continue
    for m in METHODS:
        series = d.get(m)
        if not isinstance(series, dict):
            continue
        last_step = max(int(k) for k in series.keys())
        last_rho = series[str(last_step)]
        is_outlier = m == "lime" and last_rho == last_rho and last_rho > 0.5
        if is_outlier:
            lime_outlier = (f, last_step, last_rho)
            lime_outlier_filter = f
        for step_str, rho in series.items():
            step = int(step_str)
            if rho != rho:  # skip NaN
                continue
            # the one LIME-failure condition is excluded from the mean
            # trajectory (plotted separately below) so it does not visually
            # dilute the "typical" LIME behavior into a false middle ground
            if is_outlier and f == lime_outlier_filter:
                continue
            steps_by_method[m][step].append(rho)

mean_steps = {m: sorted(steps_by_method[m].keys()) for m in METHODS}
mean_rho = {m: [statistics.mean(steps_by_method[m][s]) for s in mean_steps[m]] for m in METHODS}

# ---------- estilo (mesmo da figura 1) ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.3, 2.3), dpi=600)

# Painel A
for m in METHODS:
    s = stats_by_method[m]
    axA.scatter(s["auc"], s["dice"], s=16, c=METHOD_COLOR[m], marker=METHOD_MARKER[m],
                alpha=0.75, linewidths=0, zorder=3)
    if s["p"] < 0.05:
        xline = np.linspace(s["auc"].min() - 0.01, s["auc"].max() + 0.01, 50)
        yline = s["slope"] * xline + s["intercept"]
        axA.plot(xline, yline, color=METHOD_COLOR[m], linewidth=1.0, alpha=0.8, zorder=2)

axA.set_xlabel("AUC-ROC")
axA.set_ylabel("Dice")
axA.set_title("(A) AUC-ROC × Dice, por método", fontsize=8, pad=4)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)
axA.grid(True, linewidth=0.35, alpha=0.3, zorder=0)
axA.set_axisbelow(True)
axA.tick_params(length=2.5, width=0.6)
leg_txt = "\n".join(
    f"{METHOD_LABEL[m]}: $r$={stats_by_method[m]['r']:.2f}"
    + ("*" if stats_by_method[m]["p"] < 0.05 else " (n.s.)")
    for m in METHODS
)
axA.annotate(leg_txt, xy=(0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
             fontsize=6.3, bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                                      edgecolor="#cccccc", linewidth=0.5))
# Handles de legenda separados dos pontos do grafico -- os pontos usam
# alpha=0.75 (varios se sobrepoem), mas a legenda em si deve ficar
# totalmente opaca/visivel, nao herdar essa transparencia (bug corrigido:
# antes o label= ia direto no scatter(), entao a legenda saia esbranquicada).
method_handles = [
    Line2D([0], [0], marker=METHOD_MARKER[m], color="none", markerfacecolor=METHOD_COLOR[m],
           markeredgewidth=0, markersize=5.5, alpha=1.0, label=METHOD_LABEL[m])
    for m in METHODS
]
axA.legend(handles=method_handles, loc="lower right", frameon=True, framealpha=0.9,
           handletextpad=0.3, borderpad=0.3, labelspacing=0.2, fontsize=6.3)

# Painel B
for m in METHODS:
    axB.plot(mean_steps[m], mean_rho[m], color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
              markersize=3.5, linewidth=1.1, alpha=0.9, label=METHOD_LABEL[m], zorder=3)

if lime_outlier:
    fname, step, rho = lime_outlier
    axB.scatter([step], [rho], s=50, facecolor="#b45309", edgecolor="white",
                linewidths=0.8, marker="X", zorder=5)
    axB.annotate(f"LIME, 1 condição\n$\\rho$={rho:.2f} (falha)", xy=(step, rho),
                 xytext=(3.1, 0.62), fontsize=6.2, color="#b45309", ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color="#b45309", linewidth=0.6,
                                  connectionstyle="arc3,rad=0.15"))

axB.axhline(0, color="#999999", linewidth=0.6, linestyle=":", zorder=1)
axB.set_xlabel("camadas de topo aleatorizadas")
axB.set_ylabel(r"$\rho$ de Spearman (média)")
axB.set_title("(B) Sanity check de Adebayo", fontsize=8, pad=4)
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.grid(True, linewidth=0.35, alpha=0.3, zorder=0)
axB.set_axisbelow(True)
axB.tick_params(length=2.5, width=0.6)
# sem legenda própria -- usa a mesma codificação de cor/marcador do painel A
# (ver legenda da Figura, indicado na legenda do LaTeX)

fig.tight_layout(pad=0.5, w_pad=1.6)

out = "/home/users/aadsilva/ic/hcpa-retinopathy/hcpa-repository/hcpa/experiments/paper_auc_xai_correlacao/figura2_multimetodo.pdf"
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=600)
print("saved:", out)
for m in METHODS:
    s = stats_by_method[m]
    print(f"{m}: r={s['r']:.4f} p={s['p']:.4f} n={len(s['auc'])}")
print("lime_outlier:", lime_outlier)
