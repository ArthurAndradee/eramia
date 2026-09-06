"""
Figura principal do artigo curto (SBC/ERAMIA, 4 pag.): grade 3x2 relacionando
AUC-ROC / Sensibilidade / Especificidade (linhas) com Dice / Pointing Game
(colunas), para os 3 metodos XAI da mini-campanha (10 condicoes, medias por
repeticao). IoU foi deliberadamente excluido: r(Dice,IoU)=0.9994 (p=1.4e-42,
n=30) sobre estes mesmos 30 pontos agregados -- painel redundante com Dice.

ATENCAO (para a legenda/metodologia do artigo): BenGraham_MaxGreen2.0 tem
apenas 4/10 repeticoes completas na mini-campanha (as demais tem 9 ou 10 --
ver resultados_minicampanha_repeticoes.csv e o achado de bookkeeping
registrado em relatorio_completo.md). Nao foi descartado da agregacao, mas
seu ponto em cada painel e a media de só 4 execucoes, nao 10.

Retas de regressao com opacidade reduzida quando p>=0.05 (deliberado, pedido
explicito do usuario): com n=10 uma reta de minimos quadrados pode parecer
visualmente convincente mesmo sem suporte estatistico -- a maioria das retas
no painel de Especificidade cai nessa categoria (ver r/p impressos em cada
painel) e deve ser lida como tal.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CSV_PATH = "resultados_minicampanha_repeticoes.csv"

METHOD_ORDER = ["gradcam", "lime", "occlusion"]
METHOD_LABEL = {"gradcam": "Grad-CAM", "lime": "LIME", "occlusion": "Occlusion"}
METHOD_MARKER = {"gradcam": "o", "lime": "^", "occlusion": "s"}
METHOD_LINESTYLE = {"gradcam": "-", "lime": "--", "occlusion": ":"}
METHOD_COLOR = {"gradcam": "#1c6e62", "lime": "#b45309", "occlusion": "#5b5ea6"}

GROUP_ORDER = ["altera estrutura", "preserva estrutura", "outra"]
GROUP_FILL = {"altera estrutura": "0.15", "preserva estrutura": "0.65", "outra": "white"}
# Rotulo curto so para a legenda (o nome completo continua em GROUP_ORDER/
# classify_group, usado no texto/metodologia do artigo).
GROUP_LABEL_SHORT = {"altera estrutura": "altera", "preserva estrutura": "preserva", "outra": "outra"}
GROUP_ALTERA = {"Canny", "Otsu"}
GROUP_PRESERVA = {"Gaussian", "Median", "Morpho"}


def classify_group(filtro: str) -> str:
    stage = filtro.rsplit("_", 1)[-1] if "_" in filtro else filtro
    if stage in GROUP_ALTERA:
        return "altera estrutura"
    if stage in GROUP_PRESERVA:
        return "preserva estrutura"
    return "outra"


def fmt_ptbr(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}".replace(".", ",")


def main(fade_nonsignificant: bool = True, out_stem: str = "figura_multimetodo", show_groups: bool = True):
    """
    fade_nonsignificant=True (versao "normal"): retas com p>=0.05 ficam
    esmaecidas, para nao sugerir uma tendencia mais forte do que os dados
    sustentam (pedido explicito em turno anterior).
    fade_nonsignificant=False (segunda versao, pedido explicito): TODAS as
    retas -- Grad-CAM, LIME e Occlusion -- ficam totalmente visiveis, com
    o mesmo estilo/espessura, para que as 3 tenham reta de tendencia
    claramente vista (nao so pontos) sobre EXATAMENTE as mesmas 10
    condicoes escolhidas na selecao original (nao restringe a um
    subconjunto menor) -- os valores de r/p continuam identicos aos da
    versao normal, so a opacidade da linha muda.
    """
    df = pd.read_csv(CSV_PATH)
    n_reps = df.groupby(["filtro", "metodo_xai"]).size().rename("n_reps").reset_index()
    incomplete = n_reps[n_reps.n_reps < 10]
    print("Repeticoes por (filtro, metodo) abaixo de 10:")
    print(incomplete.to_string(index=False))

    agg = df.groupby(["filtro", "metodo_xai"], as_index=False).agg(
        auc=("auc", "mean"), dice=("dice", "mean"), iou=("iou", "mean"),
        pg=("pointing_game", "mean"),
        sens=("sensibilidade", "mean"), spec=("especificidade", "mean"),
    )
    agg["grupo"] = agg["filtro"].map(classify_group)
    n_conditions = agg["filtro"].nunique()

    r_iou, p_iou = pearsonr(agg["dice"], agg["iou"])
    print(f"\nVerificacao Dice x IoU: r={r_iou:.4f} (p={p_iou:.2e}, n={len(agg)}) -- IoU excluido do layout.")

    plt.rcParams.update({
        "font.family": "serif",
        # Times/Nimbus (o padrao do template SBC) nao esta instalado neste
        # ambiente -- so ha DejaVu Serif disponivel, e como pdf.fonttype=42
        # incorpora os glifos realmente usados no render, o PDF final sai em
        # DejaVu Serif de fato, nao Times, mesmo listando Times primeiro
        # aqui. Mantido consistente com figura1/figura2 (mesma limitacao).
        "font.serif": ["DejaVu Serif"],
        "font.size": 7.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Transposto a pedido: linhas = metrica X (Dice, Pointing Game),
    # colunas = metrica Y (AUC-ROC, Sensibilidade, Especificidade).
    rows = [("dice", "Dice"), ("pg", "Pointing Game")]
    cols = [("auc", "AUC-ROC"), ("sens", "Sensibilidade"), ("spec", "Especificidade")]

    fig, axes = plt.subplots(2, 3, figsize=(5.4, 4.1), dpi=600, sharex=False, sharey=False)

    # compartilha eixo Y dentro de cada coluna (mesma metrica clinica nas 2 linhas)
    for c in range(3):
        axes[1, c].sharey(axes[0, c])
    # compartilha eixo X dentro de cada linha (mesma metrica XAI nas 3 colunas)
    for r in range(2):
        for c in range(1, 3):
            axes[r, c].sharex(axes[r, 0])

    corr_table = []
    for ri, (xkey, xlabel) in enumerate(rows):
        # Faixa X fixa por LINHA (Dice/Pointing Game), calculada uma vez a
        # partir de todos os pontos daquela metrica -- usada para desenhar
        # TODAS as retas (nao so entre o primeiro/ultimo ponto de cada
        # metodo). Sem isso, uma reta de inclinacao rasa (Occlusion/LIME
        # em varios paineis) fica inteira "enterrada" dentro do
        # aglomerado de pontos e nunca aparece visivelmente, ao contrario
        # da reta do Grad-CAM (inclinacao maior, sai do aglomerado
        # naturalmente) -- confirmado reproduzindo o painel isolado.
        row_x_min, row_x_max = agg[xkey].min(), agg[xkey].max()
        row_margin = 0.08 * (row_x_max - row_x_min)
        row_xs = np.linspace(row_x_min - row_margin, row_x_max + row_margin, 40)
        for ci, (ykey, ylabel) in enumerate(cols):
            ax = axes[ri, ci]
            for method in METHOD_ORDER:
                sub = agg[agg.metodo_xai == method]
                x, y = sub[xkey].values, sub[ykey].values
                r, p = pearsonr(x, y)
                corr_table.append({"y": ykey, "x": xkey, "metodo": method, "r": r, "p": p})

                for _, row_pt in sub.iterrows():
                    facecolor = GROUP_FILL[row_pt["grupo"]] if show_groups else "white"
                    ax.scatter(
                        row_pt[xkey], row_pt[ykey],
                        marker=METHOD_MARKER[method], s=18,
                        facecolor=facecolor,
                        edgecolor=METHOD_COLOR[method], linewidths=0.8,
                        zorder=3,
                    )

                slope, intercept, *_ = linregress(x, y)
                ys = slope * row_xs + intercept
                significant = p < 0.05
                if fade_nonsignificant:
                    linewidth, line_alpha = (1.6 if significant else 0.9), (0.95 if significant else 0.3)
                else:
                    # Segunda versao: reta sempre totalmente visivel e mais
                    # grossa, mesmo estilo/espessura para os 3 metodos -- a
                    # significancia continua legivel no texto r/p, so a
                    # reta em si nao fica esmaecida nem fina.
                    linewidth, line_alpha = 2.0, 0.95
                ax.plot(
                    row_xs, ys, linestyle=METHOD_LINESTYLE[method], color=METHOD_COLOR[method],
                    linewidth=linewidth, alpha=line_alpha, zorder=2,
                )

            # Anotacao r/p redesenhada: sem caixa com borda pesada -- um
            # pequeno traco na cor/estilo do metodo (mesma linha usada na
            # reta de regressao) como "marcador" de leitura, texto direto
            # em cima de um fundo claro e discreto sem contorno, ancorado
            # embaixo (a faixa inferior do eixo 0-1 fica vazia em quase
            # todos os paineis).
            ax.add_patch(plt.Rectangle(
                (0.0, 0.0), 1.0, 0.255, transform=ax.transAxes,
                facecolor="white", alpha=0.82, edgecolor="none", zorder=4,
            ))
            for i, method in enumerate(METHOD_ORDER):
                row_c = [c for c in corr_table if c["y"] == ykey and c["x"] == xkey and c["metodo"] == method][0]
                y0 = 0.255 - (i + 0.5) * (0.255 / 3)
                sig = row_c["p"] < 0.05
                # Na versao "normal" (fade_nonsignificant=True), o traco e
                # o texto da anotacao ficam esmaecidos quando p>=0.05,
                # espelhando a reta principal do painel. Na segunda versao
                # (fade_nonsignificant=False) a reta principal SEMPRE fica
                # totalmente visivel -- a anotacao tem que acompanhar,
                # senao fica inconsistente (reta forte, legenda apagada,
                # exatamente o problema reportado: "legendas... ruins de
                # ler enquanto outras estao boas").
                dash_alpha = (0.95 if sig else 0.45) if fade_nonsignificant else 0.95
                text_color = ("0.15" if sig else "0.5") if fade_nonsignificant else "0.15"
                ax.plot(
                    [0.035, 0.115], [y0, y0], transform=ax.transAxes,
                    linestyle=METHOD_LINESTYLE[method], color=METHOD_COLOR[method],
                    linewidth=1.6, alpha=dash_alpha, zorder=5, clip_on=False,
                )
                sig_mark = "" if sig else " (n.s.)"
                txt = f"r={fmt_ptbr(row_c['r'])} (p={fmt_ptbr(row_c['p'])}){sig_mark}"
                ax.text(
                    0.15, y0, txt, transform=ax.transAxes,
                    fontsize=5.3, va="center", ha="left", zorder=5,
                    color=text_color,
                )

            ax.set_ylim(-0.03, 1.03)
            # Grafico multifacetado (estilo facet_grid, convencao pedida:
            # tira cinza no topo de cada coluna + rotulo de linha plano a
            # esquerda -- ver fig1_masking_timeline_matrix_en.pdf anexado
            # como referencia visual). A identidade de cada linha/coluna
            # aparece uma unica vez, nao repetida embaixo/do lado de cada
            # um dos 6 paineis.
            if ri == 0:
                strip = plt.Rectangle(
                    (0, 1.0), 1, 0.16, transform=ax.transAxes, clip_on=False,
                    facecolor="0.88", edgecolor="none", zorder=5,
                )
                ax.add_patch(strip)
                ax.text(
                    0.5, 1.08, ylabel, transform=ax.transAxes,
                    fontsize=7.6, va="center", ha="center", zorder=6,
                )
            if ci == 2:
                # Tira lateral (direita) para o rotulo de linha (Dice /
                # Pointing Game), MESMO estilo visual da tira de topo
                # (mesmo cinza, mesmo tamanho de fonte) -- so vertical.
                row_strip = plt.Rectangle(
                    (1.0, 0), 0.14, 1, transform=ax.transAxes, clip_on=False,
                    facecolor="0.88", edgecolor="none", zorder=5,
                )
                ax.add_patch(row_strip)
                ax.text(
                    1.07, 0.5, xlabel, transform=ax.transAxes, rotation=-90,
                    fontsize=7.6, va="center", ha="center", zorder=6,
                )
            # Moldura completa (todas as 4 bordas) em vez de so eixo aberto,
            # para que cada painel seja um "quadro" fechado -- pedido explicito.
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
            ax.grid(True, linewidth=0.3, alpha=0.3, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(length=2, width=0.5)
            if ci > 0:
                plt.setp(ax.get_yticklabels(), visible=False)

    # shared legend: method (marker+linestyle) and group (fill tone) --
    # handlelength maior para o tracejado/pontilhado do LIME/Occlusion
    # aparecer de verdade (o default do matplotlib deixa a amostra de
    # linha curta demais para mostrar o padrao "--"/":" de forma legivel,
    # o que fazia a legenda parecer "so tem linha para o Grad-CAM").
    method_handles = [
        Line2D([0], [0], marker=METHOD_MARKER[m], color=METHOD_COLOR[m], linestyle=METHOD_LINESTYLE[m],
               markerfacecolor="white", markeredgecolor=METHOD_COLOR[m], markersize=5, linewidth=1.3,
               label=METHOD_LABEL[m])
        for m in METHOD_ORDER
    ]
    legend_handles = method_handles
    if show_groups:
        legend_handles = legend_handles + [
            Patch(facecolor=GROUP_FILL[g], edgecolor="0.2", linewidth=0.5, label=GROUP_LABEL_SHORT[g])
            for g in GROUP_ORDER
        ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=6.2,
        handletextpad=0.4, columnspacing=1.0, handlelength=2.6,
    )

    # Sem titulo geral e sem legendas de eixo genericas (removidos a pedido
    # -- a identidade de linha/coluna ja esta nas tiras cinza, e o rotulo
    # de linha (Dice/Pointing Game) agora e a tira lateral de dentro do
    # loop acima, no mesmo estilo visual da tira de topo). Sem a legenda
    # "n = ... condicoes" tambem (removida a pedido).
    fig.tight_layout(rect=[0.055, 0.08, 0.955, 0.97], pad=0.25, h_pad=0.5, w_pad=0.5)

    fig.savefig(f"{out_stem}.pdf")
    fig.savefig(f"{out_stem}.png", dpi=300)
    print(f"\nSalvo: {out_stem}.pdf / {out_stem}.png (n={n_conditions} condicoes)")

    print("\nTabela de correlacoes (r, p) usada nas anotacoes:")
    for row_c in corr_table:
        print(f"  y={row_c['y']:5s} x={row_c['x']:4s} {row_c['metodo']:10s} r={row_c['r']:+.4f} p={row_c['p']:.4f}")


if __name__ == "__main__":
    # Versao "normal": retas esmaecidas quando p>=0.05, para nao sugerir
    # tendencia mais forte do que os dados sustentam (mesmas 10 condicoes
    # da selecao original em ambas as versoes -- Grad-CAM/LIME/Occlusion
    # sempre usam exatamente o mesmo n).
    main(fade_nonsignificant=True, out_stem="figura_multimetodo")
    print("\n" + "=" * 70)
    # Segunda versao pedida explicitamente: MESMAS 10 condicoes (nao um
    # subconjunto menor), mas as retas dos 3 metodos sempre totalmente
    # visiveis e mais grossas -- para que Grad-CAM, LIME e Occlusion
    # tenham reta de tendencia claramente vista, nao so pontos esparsos.
    # show_groups=False: sem a 4a dimensao visual (tom de preenchimento
    # por grupo de operacao final do filtro) -- nao ha, ate agora, nenhuma
    # afirmacao no texto do artigo sobre essa diferenca entre grupos, e
    # method (forma+cor+estilo de linha) ja e suficiente para distinguir
    # as 3 series; a simplificacao tambem ajuda a legibilidade das retas.
    main(fade_nonsignificant=False, out_stem="figura_multimetodo_retas_completas", show_groups=False)
