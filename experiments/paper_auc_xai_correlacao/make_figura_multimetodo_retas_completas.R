# make_figura_multimetodo_retas_completas.R
#
# Equivalente em R (ggplot2) de make_figura_multimetodo.py, SO da segunda
# versao (fade_nonsignificant=FALSE, show_groups=FALSE): grade 2x3
# relacionando Dice/Pointing Game (linhas) com AUC-ROC/Sensibilidade/
# Especificidade (colunas), 10 condicoes da mini-campanha, 3 metodos XAI
# (Grad-CAM/LIME/Occlusion) -- retas de regressao SEMPRE totalmente
# visiveis (nao esmaecidas por significancia), sem a 4a dimensao de
# tom/preenchimento por grupo de operacao final do filtro (ver
# make_figura_multimetodo.py para o raciocinio completo e para a versao
# "normal", que esmaece retas nao significativas e mantem o grupo).
#
# IoU deliberadamente excluido: r(Dice,IoU)=0.9994 sobre estes mesmos 30
# pontos agregados -- painel redundante com Dice.
#
# ATENCAO (para a legenda/metodologia do artigo): BenGraham_MaxGreen2.0
# tem apenas 4/10 repeticoes completas na mini-campanha (as demais tem 9
# ou 10 -- ver resultados_minicampanha_repeticoes.csv e o achado de
# bookkeeping registrado em relatorio_completo.md). Nao foi descartado da
# agregacao, mas seu ponto em cada painel e a media de so 4 execucoes,
# nao 10.
#
# `facet_grid(xmetric ~ ymetric)` faz nativamente a tira cinza no topo
# (por coluna) + rotulo de linha na lateral direita (por linha) que no
# script Python foi desenhado a mao com Rectangle/text -- e exatamente o
# layout ja usado como referencia (fig1_masking_timeline_matrix_en.pdf).
#
# Uso: Rscript make_figura_multimetodo_retas_completas.R
# Gera: figura_multimetodo_retas_completas.pdf / .png

suppressMessages(library(tidyverse))

CSV_PATH <- "resultados_minicampanha_repeticoes.csv"
OUT_STEM <- "figura_multimetodo_retas_completas"

METHOD_LEVELS <- c("gradcam", "lime", "occlusion")
METHOD_LABEL  <- c(gradcam = "Grad-CAM", lime = "LIME", occlusion = "Occlusion")
METHOD_COLOR  <- c(gradcam = "#1c6e62", lime = "#b45309", occlusion = "#5b5ea6")
METHOD_SHAPE  <- c(gradcam = 21, lime = 24, occlusion = 22)          # circulo/triangulo/quadrado, preenchiveis
METHOD_LINETYPE <- c(gradcam = "solid", lime = "dashed", occlusion = "dotted")

fmt_ptbr <- function(x, nd = 3) {
  # vírgula decimal, como no artigo (pt-BR)
  sub("\\.", ",", sprintf(paste0("%.", nd, "f"), x))
}

# ---------------------------------------------------------------------
# Dados: agregado por (filtro, metodo_xai), media entre repeticoes
# disponiveis (nem sempre 10 -- ver ATENCAO acima).
# ---------------------------------------------------------------------
df <- read_csv(CSV_PATH, show_col_types = FALSE)

n_reps <- df |>
  count(filtro, metodo_xai, name = "n_reps") |>
  filter(n_reps < 10)
cat("Repeticoes por (filtro, metodo) abaixo de 10:\n")
print(n_reps, n = Inf)

agg <- df |>
  group_by(filtro, metodo_xai) |>
  summarise(
    auc  = mean(auc), dice = mean(dice), iou = mean(iou),
    pg   = mean(pointing_game), sens = mean(sensibilidade), spec = mean(especificidade),
    .groups = "drop"
  )
n_conditions <- n_distinct(agg$filtro)

r_iou <- cor.test(agg$dice, agg$iou)
cat(sprintf(
  "\nVerificacao Dice x IoU: r=%.4f (p=%.2e, n=%d) -- IoU excluido do layout.\n",
  r_iou$estimate, r_iou$p.value, nrow(agg)
))

# ---------------------------------------------------------------------
# Formato longo: uma linha por (filtro, metodo, metrica-X, metrica-Y)
# -- assim facet_grid(xmetric ~ ymetric) desenha os 6 paineis de uma vez.
# ---------------------------------------------------------------------
x_metrics <- tibble(xkey = c("dice", "pg"), xlabel = c("Dice", "Pointing Game"))
y_metrics <- tibble(ykey = c("auc", "sens", "spec"), ylabel = c("AUC-ROC", "Sensibilidade", "Especificidade"))

long <- x_metrics |>
  cross_join(y_metrics) |>
  rowwise() |>
  reframe(
    xkey = xkey, xlabel = xlabel, ykey = ykey, ylabel = ylabel,
    filtro = agg$filtro, metodo_xai = agg$metodo_xai,
    xval = agg[[xkey]], yval = agg[[ykey]]
  ) |>
  mutate(
    xlabel = factor(xlabel, levels = x_metrics$xlabel),
    ylabel = factor(ylabel, levels = y_metrics$ylabel),
    metodo_xai = factor(metodo_xai, levels = METHOD_LEVELS)
  )

# ---------------------------------------------------------------------
# Correlacao (r, p) e coeficientes de reta por (xkey, ykey, metodo) --
# a MESMA tabela que o script Python imprime para conferencia cruzada.
# ---------------------------------------------------------------------
corr_table <- long |>
  group_by(xkey, xlabel, ykey, ylabel, metodo_xai) |>
  summarise(
    r = cor(xval, yval),
    p = cor.test(xval, yval)$p.value,
    slope = coef(lm(yval ~ xval))[["xval"]],
    intercept = coef(lm(yval ~ xval))[["(Intercept)"]],
    .groups = "drop"
  ) |>
  mutate(significant = p < 0.05)

cat("\nTabela de correlacoes (r, p) usada nas anotacoes:\n")
corr_table |>
  transmute(y = ykey, x = xkey, metodo = metodo_xai,
            r = sprintf("%+.4f", r), p = sprintf("%.4f", p)) |>
  print(n = Inf)

# ---------------------------------------------------------------------
# Anotacao r/p por painel: ancorada no canto inferior esquerdo (-Inf/-Inf
# com hjust/vjust), pra funcionar em qualquer escala X sem hardcode --
# SEMPRE totalmente visivel nesta segunda versao (nao esmaece por
# significancia; "(n.s.)" continua escrito por extenso quando aplicavel).
# ---------------------------------------------------------------------
ann <- corr_table |>
  mutate(
    # Sem nome do metodo por extenso (pedido explicito, igual ao
    # figura_multimetodo.pdf): a cor do proprio traco/texto ja identifica
    # o metodo. Sem "(n.s.)" tambem (pedido explicito) -- as 3 linhas
    # ficam com o MESMO comprimento de texto agora, entao usar coordenada
    # de dado fixa (em vez de -Inf/hjust) da o mesmo espacamento exato
    # para as 3, sem depender da largura variavel do texto.
    label = paste0("r=", fmt_ptbr(r), " (p=", fmt_ptbr(p), ")"),
    # empilha as 3 linhas de baixo pra cima dentro da caixa
    # step=1 -> topo da caixa (Grad-CAM primeiro, igual ao Python)
    vjust_step = as.numeric(factor(metodo_xai, levels = METHOD_LEVELS))
  )

# Amostra de traco colorido ao lado do texto r/p (igual ao Python: um
# tracinho na cor/estilo do metodo antes do texto, nao so o texto solto)
# -- mesma coordenada Y do geom_text correspondente, um pouco a esquerda.
ann <- ann |>
  mutate(ann_y = 0.19 - (vjust_step - 1) * 0.07)

# Legenda "fantasma": geom_abline SEMPRE desenha seu icone de legenda na
# propria inclinacao da reta (diagonal), e nem key_glyph="path" nem
# key_glyph=draw_key_path corrigem isso no ggplot2 4.0.3 (testado, sem
# efeito -- parece regressao da reescrita do sistema de Geom). Contorno:
# desliga a legenda do geom_abline (show.legend=FALSE) e usa uma camada
# geom_segment auxiliar, com dado FORA da area visivel (x=-1, recortado
# pelo coord_cartesian abaixo, nunca aparece no grafico), so pra fornecer
# o icone horizontal correto -- geom_segment usa draw_key_path por
# padrao, que E horizontal.
legend_dummy <- tibble(
  metodo_xai = factor(METHOD_LEVELS, levels = METHOD_LEVELS),
  x = -1, xend = -1.01, y = -1, yend = -1
)

p <- ggplot(long, aes(x = xval, y = yval)) +
  # retas de regressao -- geom_abline atravessa o painel INTEIRO
  # automaticamente (ao contrario de geom_smooth, que para no intervalo
  # dos dados) -- e exatamente a extensao "linhas mais visiveis" que o
  # script Python faz manualmente com uma faixa X fixa por linha.
  geom_abline(
    data = corr_table,
    aes(slope = slope, intercept = intercept, color = metodo_xai, linetype = metodo_xai),
    linewidth = 0.7, alpha = 0.95, show.legend = FALSE
  ) +
  geom_segment(
    data = legend_dummy,
    aes(x = x, xend = xend, y = y, yend = yend, color = metodo_xai, linetype = metodo_xai),
    linewidth = 0.9, inherit.aes = FALSE
  ) +
  # geom_point removido a pedido explicito -- so as retas de regressao
  # ficam visiveis agora, nenhum ponto de amostra individual.
  # geom_point(
  #   aes(shape = metodo_xai, color = metodo_xai), fill = "white",
  #   size = 1.5, stroke = 0.55
  # ) +
  # caixa de fundo da anotacao (fixa em 0-1 pq as 3 metricas Y sao todas
  # proporcoes 0-1) + texto sempre em tinta escura (nunca esmaecido) --
  # MESMO estilo do Python: retangulo branco semi-transparente, sem
  # borda, cobrindo a faixa inferior do painel.
  geom_rect(
    data = distinct(long, xlabel, ylabel),
    xmin = -Inf, xmax = Inf, ymin = -0.03, ymax = 0.20,
    inherit.aes = FALSE, fill = "white", alpha = 0.82
  ) +
  # Formato pedido: um TRACO de verdade (linha curta, nao caractere de
  # texto) na cor do metodo, seguido do texto "r=... (p=...)" em preto,
  # sem negrito. Coordenada de dado FIXA (nao -Inf/hjust): como as 3
  # linhas agora tem exatamente o mesmo comprimento de texto (sem nome do
  # metodo, sem "(n.s.)"), uma posicao fixa da o mesmo espacamento
  # horizontal E vertical para as 3 -- testado que cabe em ambas as
  # linhas (Dice ~0-0,10, Pointing Game ~0-0,11) sem cortar.
  geom_segment(
    data = ann,
    aes(x = 0.001, xend = 0.0095, y = ann_y, yend = ann_y, color = metodo_xai),
    linewidth = 0.9, lineend = "round", inherit.aes = FALSE, show.legend = FALSE
  ) +
  geom_text(
    data = ann,
    aes(x = 0.012, y = ann_y, label = label),
    hjust = 0, size = 1.85, color = "black", inherit.aes = FALSE,
    show.legend = FALSE
  ) +
  facet_grid(rows = vars(xlabel), cols = vars(ylabel)) +
  scale_color_manual(values = METHOD_COLOR, labels = METHOD_LABEL, name = NULL) +
  # scale_shape_manual removida junto com o geom_point (sem forma sem pontos) +
  scale_linetype_manual(values = METHOD_LINETYPE, labels = METHOD_LABEL, name = NULL) +
  scale_y_continuous(breaks = seq(0, 1, 0.2)) +
  scale_x_continuous(breaks = scales::breaks_pretty(n = 3)) +
  # coord_cartesian (nao scale limits) para RECORTAR a viewport sem
  # descartar dado nenhum -- geom_abline desenha a reta inteira por
  # construcao (nao tem x/y proprios), entao scale_y_continuous(limits=)
  # nao a recorta e ela "vazava" pra fora da moldura do painel; isso
  # corrige so a renderizacao, nenhuma informacao muda.
  # xlim explicito tambem: sem isso, o segmento fantasma em x=-1 (usado
  # so pra corrigir o icone da legenda, ver acima) entraria no calculo
  # automatico da faixa/quebras do eixo X e distorceria os ticks.
  coord_cartesian(xlim = c(-0.005, 0.115), ylim = c(-0.03, 1.03), clip = "on") +
  labs(x = NULL, y = NULL) +
  theme_bw(base_size = 8, base_family = "serif") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.3, color = "grey85"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
    strip.background = element_rect(fill = "grey88", color = NA),
    strip.text = element_text(size = 8.4, margin = margin(4, 4, 4, 4)),
    legend.position = "bottom",
    legend.key.size = unit(0.9, "lines"),
    legend.key.width = unit(1.3, "lines"),
    legend.text = element_text(size = 6.3),
    legend.spacing.x = unit(0.3, "lines"),
    legend.margin = margin(t = -4),
    legend.box.margin = margin(t = -6),
    axis.text = element_text(size = 6.6),
    plot.margin = margin(4, 4, 2, 2)
  ) +
  guides(color = guide_legend(override.aes = list(linewidth = 0.9)))

ggsave(paste0(OUT_STEM, ".pdf"), p, width = 5.4, height = 4.1, units = "in", device = cairo_pdf)
ggsave(paste0(OUT_STEM, ".png"), p, width = 5.4, height = 4.1, units = "in", dpi = 300)

cat(sprintf("\nSalvo: %s.pdf / %s.png (n=%d condicoes)\n", OUT_STEM, OUT_STEM, n_conditions))
