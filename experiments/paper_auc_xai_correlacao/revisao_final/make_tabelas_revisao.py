#!/usr/bin/env python3
"""Tabelas de apoio as revisoes do ERAMIA II (itens 1.2, 1.4 e 2.5).

Tabela A -- as 18 correlacoes da mini-campanha (10 condicoes x 3 tecnicas):
  Pearson r com IC95% bootstrap, Spearman rho, e Pearson sem a condicao
  colapsada (AUC < 0,6 -> so LABNorm_MaxGreen2.0_Frangi, AUC 0,556).
Tabela B -- Grad-CAM sobre a varredura inteira (todas as condicoes, nao so
  10): Pearson e Spearman por estrato de AUC (todas, AUC >= 0,6, AUC >= 0,8),
  para as 495 do artigo e para as 576 completas.

Uso: python3 make_tabelas_revisao.py   (gera CSVs e .tex nesta pasta)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
SWEEP = PAPER.parent / "analise_final_576" / "condicoes_agregadas.csv"
COLLAPSED_AUC = 0.6
N_BOOT = 10000

METHODS = [("gradcam", "Grad-CAM"), ("lime", "LIME"), ("occlusion", "Occlusion")]
PERF = [("auc", "AUC-ROC"), ("sens", "Sensibilidade"), ("spec", "Especificidade")]
FID = [("dice", "Dice"), ("pg", "Pointing Game")]


def ptbr(x, nd=3):
    return "--" if pd.isna(x) else f"{x:.{nd}f}".replace(".", ",")


def pfmt(p):
    return "<0,001" if p < 0.001 else ptbr(p)


def boot_ci(x, y, rng):
    n = len(x)
    rs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if np.std(x[i]) == 0 or np.std(y[i]) == 0:
            continue
        rs.append(np.corrcoef(x[i], y[i])[0, 1])
    return np.percentile(rs, [2.5, 97.5])


def tabela_a():
    d = pd.read_csv(PAPER / "resultados_minicampanha_repeticoes.csv")
    agg = (d.groupby(["filtro", "metodo_xai"])
             .agg(auc=("auc", "mean"), sens=("sensibilidade", "mean"),
                  spec=("especificidade", "mean"), dice=("dice", "mean"),
                  pg=("pointing_game", "mean"), n_reps=("repeticao", "nunique"))
             .reset_index())
    rng = np.random.default_rng(42)
    rows = []
    for m, mlab in METHODS:
        g = agg[agg.metodo_xai == m]
        ok = g[g.auc >= COLLAPSED_AUC]
        for fk, flab in FID:
            for pk, plab in PERF:
                x, y = g[fk].values, g[pk].values
                r, p = stats.pearsonr(x, y)
                lo, hi = boot_ci(x, y, rng)
                rho, prho = stats.spearmanr(x, y)
                r9, p9 = stats.pearsonr(ok[fk], ok[pk])
                rows.append(dict(tecnica=mlab, fidelidade=flab, desempenho=plab,
                                 n=len(g), r=r, p=p, ic95_lo=lo, ic95_hi=hi,
                                 spearman_rho=rho, spearman_p=prho,
                                 n_sem_colapso=len(ok), r_sem_colapso=r9, p_sem_colapso=p9))
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "tabela_A_correlacoes_minicampanha.csv", index=False)
    agg.to_csv(HERE / "dados_figura2_agregados.csv", index=False)

    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{Correlação entre desempenho e fidelidade nas dez condições "
             r"(Pearson com IC95\% \textit{bootstrap}, Spearman e Pearson sem a condição "
             r"colapsada, AUC $<$ 0,6; n = 9).}",
             r"\label{tab:correlacoes}", r"\footnotesize",
             r"\begin{tabular}{lllcccc}", r"\hline",
             r"Técnica & Fidelidade & Desempenho & $r$ (n=10) & IC95\% & $\rho$ & $r$ (n=9) \\", r"\hline"]
    for _, x in t.iterrows():
        star = lambda p: "*" if p < 0.05 else ""
        lines.append(f"{x.tecnica} & {x.fidelidade} & {x.desempenho} & "
                     f"{ptbr(x.r)}{star(x.p)} & [{ptbr(x.ic95_lo, 2)}; {ptbr(x.ic95_hi, 2)}] & "
                     f"{ptbr(x.spearman_rho)}{star(x.spearman_p)} & "
                     f"{ptbr(x.r_sem_colapso)}{star(x.p_sem_colapso)} \\\\")
    lines += [r"\hline", r"\multicolumn{7}{l}{\scriptsize * $p < 0{,}05$, sem correção para comparações múltiplas.}",
              r"\end{tabular}", r"\end{table}"]
    (HERE / "tabela_A_correlacoes_minicampanha.tex").write_text("\n".join(lines) + "\n")
    return t


def tabela_b():
    a = pd.read_csv(SWEEP, index_col=0)
    a = a.rename(columns={"auc_roc_mean": "auc", "sensitivity_mean": "sens",
                          "specificity_mean": "spec", "dice_mean_mean": "dice",
                          "pointing_game_accuracy_mean": "pg"})
    names495 = set(pd.read_csv(PAPER / "resultados_495_condicoes.csv").filtro)
    conjuntos = {"495 (artigo)": a[a.index.isin(names495)], "576 (completa)": a}
    estratos = [("todas", 0.0), ("AUC >= 0,6", 0.6), ("AUC >= 0,8", 0.8)]
    rows = []
    for cname, base in conjuntos.items():
        for elab, thr in estratos:
            g = base[base.auc >= thr]
            for fk, flab in FID:
                for pk, plab in PERF:
                    r, p = stats.pearsonr(g[fk], g[pk])
                    rho, prho = stats.spearmanr(g[fk], g[pk])
                    rows.append(dict(conjunto=cname, estrato=elab, n=len(g),
                                     fidelidade=flab, desempenho=plab,
                                     r=r, p=p, spearman_rho=rho, spearman_p=prho))
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "tabela_B_gradcam_varredura.csv", index=False)

    for cname, tag in [("495 (artigo)", "495"), ("576 (completa)", "576")]:
        s = t[t.conjunto == cname]
        ns = {e: s[s.estrato == e].n.iloc[0] for e, _ in estratos}
        lines = [r"\begin{table}[ht]", r"\centering",
                 rf"\caption{{Grad-CAM nas {tag} condições da varredura: correlação de Pearson "
                 r"($r$) e de Spearman ($\rho$) entre desempenho e fidelidade, por estrato de AUC.}",
                 rf"\label{{tab:gradcam{tag}}}", r"\footnotesize",
                 r"\begin{tabular}{llcccccc}", r"\hline",
                 r" & & \multicolumn{2}{c}{Todas (n=%d)} & \multicolumn{2}{c}{AUC $\geq$ 0,6 (n=%d)} & \multicolumn{2}{c}{AUC $\geq$ 0,8 (n=%d)} \\"
                 % (ns["todas"], ns["AUC >= 0,6"], ns["AUC >= 0,8"]),
                 r"Fidelidade & Desempenho & $r$ & $\rho$ & $r$ & $\rho$ & $r$ & $\rho$ \\", r"\hline"]
        for fk, flab in FID:
            for pk, plab in PERF:
                cells = []
                for e, _ in estratos:
                    x = s[(s.estrato == e) & (s.fidelidade == flab) & (s.desempenho == plab)].iloc[0]
                    star = lambda p: "*" if p < 0.05 else ""
                    cells += [ptbr(x.r) + star(x.p), ptbr(x.spearman_rho) + star(x.spearman_p)]
                lines.append(f"{flab} & {plab} & " + " & ".join(cells) + r" \\")
        lines += [r"\hline", r"\multicolumn{8}{l}{\scriptsize * $p < 0{,}05$.}",
                  r"\end{tabular}", r"\end{table}"]
        (HERE / f"tabela_B_gradcam_{tag}.tex").write_text("\n".join(lines) + "\n")
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    ta = tabela_a()
    print(ta.round(3).to_string(index=False))
    tb = tabela_b()
    print(tb.round(3).to_string(index=False))
