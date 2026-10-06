#!/usr/bin/env python3
"""Analise final da varredura de 576 condicoes de pre-processamento
(EfficientNetV2S, FGADR, reps 0-9). Le ~/resultados/*.csv e grava tabelas,
figuras e resumo.json nesta pasta. Uso: python3 analise.py"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = Path(os.path.expanduser("~/resultados"))
MATRIX = REPO / "experiments" / "filter_matrix_template.json"
FIG = HERE / "figuras"

LAYERS = {
    "C1_iluminacao": ["BenGraham", "Gamma0.8", "Retinex", "HistogramEq", "LABNorm"],
    "C2_espectral": ["GreenChannel", "MaxGreen2.0", "Grayscale"],
    "C3_contraste_local": ["AHE40.0", "CLAHE4.0"],
    "C4_refinamento": ["Unsharp1.5", "Median", "Gaussian", "Morpho", "Frangi", "Canny", "Otsu"],
}
METRICS = ["auc_roc", "sensitivity", "specificity", "accuracy", "f1_score", "precision",
           "dice_mean", "iou_mean", "pointing_game_accuracy"]


def layer_choice(name, opts):
    toks = name.split("_")
    hit = [o for o in opts if o in toks]
    return hit[0] if hit else "nenhum"


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        prev = min(prev, p[idx] * m / rank)
        adj[idx] = prev
    return adj


def load_runs():
    frames = []
    for f in glob.glob(str(RESULTS / "*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "filter_name" not in d.columns or "repetition" not in d.columns:
            continue
        if "xai_method" in d.columns:
            d = d[d["xai_method"].isna() | (d["xai_method"] == "gradcam")]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["repetition"].between(0, 9)]
    # uma linha por (condicao, repeticao): a avaliacao mais recente
    df = df.sort_values("timestamp").groupby(["filter_name", "repetition"]).tail(1)
    return df


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    matrix = json.load(open(MATRIX))
    depth = {e["name"]: e["depth"] for e in matrix}

    runs = load_runs()
    runs = runs[runs["filter_name"].isin(depth)]
    nrep = runs.groupby("filter_name")["repetition"].nunique()
    complete = sorted(nrep[nrep == 10].index)
    incomplete = {n: int(nrep.get(n, 0)) for n in depth if nrep.get(n, 0) < 10}
    runs.to_csv(HERE / "execucoes_reps0-9.csv", index=False)

    r = runs[runs["filter_name"].isin(complete)]
    agg = r.groupby("filter_name")[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["auc_ci95"] = 1.96 * agg["auc_roc_std"] / np.sqrt(10)
    agg["depth"] = [depth[n] for n in agg.index]
    for lay, opts in LAYERS.items():
        agg[lay] = [layer_choice(n, opts) for n in agg.index]

    base = r[r["filter_name"] == "baseline"].set_index("repetition")["auc_roc"]
    pv, dd, dpair, ppair = [], [], [], []
    for n in agg.index:
        x = r[r["filter_name"] == n].set_index("repetition")["auc_roc"]
        if n == "baseline":
            pv.append(np.nan); dd.append(0.0); dpair.append(0.0); ppair.append(np.nan)
            continue
        pv.append(stats.ttest_ind(x, base, equal_var=False).pvalue)
        sp = np.sqrt((x.var(ddof=1) + base.var(ddof=1)) / 2)
        dd.append((x.mean() - base.mean()) / sp if sp > 0 else np.nan)
        # mesma seed => pareado por repeticao
        j = x.index.intersection(base.index)
        diff = x.loc[j] - base.loc[j]
        dpair.append(diff.mean())
        ppair.append(stats.ttest_rel(x.loc[j], base.loc[j]).pvalue)
    agg["delta_auc_vs_baseline"] = agg["auc_roc_mean"] - agg.loc["baseline", "auc_roc_mean"]
    agg["p_welch"] = pv
    agg["p_pareado_seed"] = ppair
    agg["cohen_d"] = dd
    mask = agg.index != "baseline"
    for col in ("p_welch", "p_pareado_seed"):
        agg.loc[mask, col + "_holm"] = holm(agg.loc[mask, col])
        agg.loc[mask, col + "_fdr"] = bh(agg.loc[mask, col])
    agg = agg.sort_values("auc_roc_mean", ascending=False)
    agg.insert(0, "rank", range(1, len(agg) + 1))
    agg.to_csv(HERE / "condicoes_agregadas.csv")

    # efeito marginal pareado de cada opcao de camada: compara cada condicao
    # que usa a opcao com a MESMA condicao sem nada naquela camada
    rows = []
    names = set(agg.index)
    for lay, opts in LAYERS.items():
        for o in opts:
            deltas = []
            for n in agg.index:
                toks = n.split("_")
                if o not in toks:
                    continue
                rest = [t for t in toks if t != o]
                cf = "_".join(rest) if rest else "baseline"
                if cf in names:
                    deltas.append(agg.loc[n, "auc_roc_mean"] - agg.loc[cf, "auc_roc_mean"])
            deltas = np.array(deltas)
            w = stats.wilcoxon(deltas).pvalue if len(deltas) > 5 else np.nan
            rows.append(dict(camada=lay, opcao=o, n_pares=len(deltas),
                             delta_auc_medio=deltas.mean(), delta_auc_mediana=np.median(deltas),
                             frac_pares_melhora=(deltas > 0).mean(), p_wilcoxon=w))
    eff = pd.DataFrame(rows)
    eff["p_wilcoxon_holm"] = holm(eff["p_wilcoxon"].fillna(1))
    eff.to_csv(HERE / "efeito_marginal_por_filtro.csv", index=False)

    # modelo aditivo AUC ~ C1 + C2 + C3 + C4 (dummies, referencia = nenhum)
    X = pd.get_dummies(agg[list(LAYERS)], drop_first=False)
    X = X[[c for c in X.columns if not c.endswith("_nenhum")]].astype(float)
    X.insert(0, "intercepto", 1.0)
    y = agg["auc_roc_mean"].values
    beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
    yhat = X.values @ beta
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    pd.Series(beta, index=X.columns).to_csv(HERE / "modelo_aditivo_coeficientes.csv", header=["coef"])

    by_depth = agg.groupby("depth")["auc_roc_mean"].agg(["count", "mean", "median", "min", "max"])
    by_depth.to_csv(HERE / "auc_por_profundidade.csv")
    rho_depth = stats.spearmanr(agg["depth"], agg["auc_roc_mean"])
    kw = stats.kruskal(*[g["auc_roc_mean"].values for _, g in agg.groupby("depth") if len(g) > 1])

    # condicoes praticamente identicas (operacao redundante na cadeia)
    near = []
    vals = agg[["auc_roc_mean", "auc_roc_std", "dice_mean_mean"]].round(4)
    for _, g in vals.groupby(list(vals.columns)):
        if len(g) > 1:
            near.append(list(g.index))

    rd = stats.pearsonr(agg["auc_roc_mean"], agg["dice_mean_mean"])
    rp = stats.pearsonr(agg["auc_roc_mean"], agg["pointing_game_accuracy_mean"])
    b = agg.loc["baseline"]
    resumo = dict(
        condicoes_matriz=len(depth), condicoes_completas=len(complete),
        condicoes_incompletas=incomplete, execucoes_reps0_9=int(len(runs)),
        baseline=dict(auc=b.auc_roc_mean, auc_sd=b.auc_roc_std, sens=b.sensitivity_mean,
                      spec=b.specificity_mean, dice=b.dice_mean_mean, rank=int(b["rank"])),
        melhor=dict(nome=agg.index[0], auc=agg.iloc[0].auc_roc_mean,
                    p_welch=agg.iloc[0].p_welch, p_pareado=agg.iloc[0].p_pareado_seed),
        acima_baseline=int((agg.loc[mask, "delta_auc_vs_baseline"] > 0).sum()),
        abaixo_baseline_sig_holm=int(((agg["delta_auc_vs_baseline"] < 0) & (agg["p_welch_holm"] < 0.05)).sum()),
        acima_baseline_sig_holm=int(((agg["delta_auc_vs_baseline"] > 0) & (agg["p_welch_holm"] < 0.05)).sum()),
        acima_baseline_sig_fdr=int(((agg["delta_auc_vs_baseline"] > 0) & (agg["p_welch_fdr"] < 0.05)).sum()),
        acima_baseline_p05_sem_correcao=int(((agg["delta_auc_vs_baseline"] > 0) & (agg["p_welch"] < 0.05)).sum()),
        spearman_profundidade_auc=dict(rho=rho_depth.statistic, p=rho_depth.pvalue),
        kruskal_profundidade=dict(H=kw.statistic, p=kw.pvalue),
        modelo_aditivo_r2=r2,
        pearson_auc_dice=dict(r=rd.statistic, p=rd.pvalue),
        pearson_auc_pointing=dict(r=rp.statistic, p=rp.pvalue),
        faixas=dict(auc=[agg.auc_roc_mean.min(), agg.auc_roc_mean.max()],
                    sens=[agg.sensitivity_mean.min(), agg.sensitivity_mean.max()],
                    spec=[agg.specificity_mean.min(), agg.specificity_mean.max()],
                    dice=[agg.dice_mean_mean.min(), agg.dice_mean_mean.max()]),
        grupos_quase_identicos=near,
    )
    json.dump(resumo, open(HERE / "resumo.json", "w"), indent=2, ensure_ascii=False, default=float)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "pdf.fonttype": 42})
    # 1. ranking
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    xs = np.arange(len(agg))
    ax.fill_between(xs, agg.auc_roc_mean - agg.auc_ci95, agg.auc_roc_mean + agg.auc_ci95,
                    color="0.8", lw=0)
    ax.plot(xs, agg.auc_roc_mean, color="0.2", lw=0.8)
    bi = list(agg.index).index("baseline")
    ax.axhline(b.auc_roc_mean, color="#b45309", lw=0.7, ls="--")
    ax.plot([bi], [b.auc_roc_mean], "o", color="#b45309", ms=4)
    ax.set_xlabel("condição (ordenada por AUC)"); ax.set_ylabel("AUC-ROC (média, IC95%)")
    ax.text(len(agg) * 0.6, b.auc_roc_mean + 0.006, "baseline", color="#b45309")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "ranking_auc.pdf"); fig.savefig(FIG / "ranking_auc.png", dpi=300)
    plt.close(fig)
    # 2. profundidade
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    groups = [agg[agg.depth == d].auc_roc_mean for d in sorted(agg.depth.unique())]
    ax.boxplot(groups, labels=sorted(agg.depth.unique()), widths=0.55, flierprops=dict(ms=2))
    ax.axhline(b.auc_roc_mean, color="#b45309", lw=0.7, ls="--")
    ax.set_xlabel("nº de filtros encadeados"); ax.set_ylabel("AUC-ROC")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "auc_por_profundidade.pdf"); fig.savefig(FIG / "auc_por_profundidade.png", dpi=300)
    plt.close(fig)
    # 3. efeito marginal pareado
    e = eff.sort_values("delta_auc_medio")
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    cols = ["#5b5ea6" if v < 0 else "#1c6e62" for v in e.delta_auc_medio]
    ax.barh(e.opcao, e.delta_auc_medio, color=cols)
    ax.axvline(0, color="0.3", lw=0.6)
    ax.set_xlabel("Δ AUC médio ao adicionar o filtro\n(pareado com a mesma cadeia sem ele)")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "efeito_marginal.pdf"); fig.savefig(FIG / "efeito_marginal.png", dpi=300)
    plt.close(fig)
    # 4. sensibilidade x especificidade
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    sc = ax.scatter(agg.specificity_mean, agg.sensitivity_mean, c=agg.auc_roc_mean, s=6, cmap="viridis")
    ax.plot(b.specificity_mean, b.sensitivity_mean, "*", color="#b45309", ms=9)
    ax.set_xlabel("especificidade"); ax.set_ylabel("sensibilidade")
    fig.colorbar(sc, ax=ax, label="AUC-ROC")
    fig.tight_layout(); fig.savefig(FIG / "sens_vs_spec.pdf"); fig.savefig(FIG / "sens_vs_spec.png", dpi=300)
    plt.close(fig)

    print(json.dumps({k: v for k, v in resumo.items() if k != "condicoes_incompletas"},
                     indent=1, ensure_ascii=False, default=float))
    print("incompletas:", len(incomplete))
    print(eff.round(4).to_string(index=False))
    print(by_depth.round(4))


if __name__ == "__main__":
    main()
