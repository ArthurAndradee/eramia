# Figura 2 (versao final, revisao 1.1/1.4): dispersao bruta das 10 condicoes
# x 3 tecnicas, SEM retas de regressao. Mesma grade e estilo de
# ../make_figura_multimetodo_retas_completas.R. A condicao colapsada
# (AUC < 0,6) aparece com marcador vazado. Anotacao por painel: Pearson r e
# p com n = 10 (os mesmos valores do artigo).
#
# Uso: Rscript make_figura2_dispersao.R  -> figura2_dispersao.pdf/.png

suppressMessages({library(dplyr); library(tidyr); library(tibble); library(readr); library(ggplot2)})

CSV_PATH <- "../resultados_minicampanha_repeticoes.csv"
OUT_STEM <- "figura2_dispersao"
COLLAPSED_AUC <- 0.6

METHOD_LEVELS <- c("gradcam", "lime", "occlusion")
METHOD_LABEL  <- c(gradcam = "Grad-CAM", lime = "LIME", occlusion = "Occlusion")
METHOD_COLOR  <- c(gradcam = "#1c6e62", lime = "#b45309", occlusion = "#5b5ea6")
METHOD_SHAPE  <- c(gradcam = 21, lime = 24, occlusion = 22)

fmt_ptbr <- function(x, nd = 3) sub("\\.", ",", sprintf(paste0("%.", nd, "f"), x))

agg <- read_csv(CSV_PATH, show_col_types = FALSE) |>
  group_by(filtro, metodo_xai) |>
  summarise(auc = mean(auc), dice = mean(dice), pg = mean(pointing_game),
            sens = mean(sensibilidade), spec = mean(especificidade), .groups = "drop") |>
  mutate(colapsada = auc < COLLAPSED_AUC)

x_metrics <- tibble(xkey = c("dice", "pg"), xlabel = c("Dice", "Pointing Game"))
y_metrics <- tibble(ykey = c("auc", "sens", "spec"), ylabel = c("AUC-ROC", "Sensibilidade", "Especificidade"))

long <- bind_rows(lapply(seq_len(nrow(x_metrics)), function(i) {
  bind_rows(lapply(seq_len(nrow(y_metrics)), function(j) {
    tibble(xlabel = x_metrics$xlabel[i], ylabel = y_metrics$ylabel[j],
           filtro = agg$filtro, metodo_xai = agg$metodo_xai, colapsada = agg$colapsada,
           xval = agg[[x_metrics$xkey[i]]], yval = agg[[y_metrics$ykey[j]]])
  }))
})) |>
  mutate(xlabel = factor(xlabel, levels = x_metrics$xlabel),
         ylabel = factor(ylabel, levels = y_metrics$ylabel),
         metodo_xai = factor(metodo_xai, levels = METHOD_LEVELS))

corr_table <- long |>
  group_by(xlabel, ylabel, metodo_xai) |>
  summarise(r = cor(xval, yval), p = cor.test(xval, yval)$p.value, .groups = "drop")
cat("Pearson r/p (n=10) usados nas anotacoes:\n"); print(corr_table, n = Inf)

# Escala X livre por linha (Dice e Pointing Game tem faixas diferentes);
# a anotacao e posicionada em fracao da largura do painel de cada linha,
# reproduzindo a expansao padrao de 5% do ggplot.
xr <- long |> group_by(xlabel) |>
  summarise(lo = min(xval), hi = max(xval), .groups = "drop") |>
  mutate(w = hi - lo, plo = lo - 0.05 * w, pw = 1.1 * w)

YLIM <- c(0.20, 1.0)
ann <- corr_table |>
  left_join(xr, by = "xlabel") |>
  mutate(label = paste0("r=", fmt_ptbr(r), " (p=", fmt_ptbr(p), ")"),
         step = as.numeric(metodo_xai),
         y = 0.335 - (step - 1) * 0.048,
         x0 = plo + 0.025 * pw, x1 = plo + 0.085 * pw, xt = plo + 0.11 * pw)

box <- distinct(long, xlabel, ylabel)

# ponto invisivel (size 0) dentro da faixa das duas linhas, so para gerar o
# item de legenda da condicao colapsada
legend_dummy <- tibble(x = 0.05, y = 0.5, k = "Condição colapsada (AUC < 0,6)")

p <- ggplot(long, aes(x = xval, y = yval)) +
  geom_rect(data = box, xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = 0.36,
            inherit.aes = FALSE, fill = "white", alpha = 0.85) +
  geom_point(data = filter(long, !colapsada),
             aes(shape = metodo_xai, color = metodo_xai, fill = metodo_xai),
             size = 1.6, stroke = 0.45, alpha = 0.9) +
  geom_point(data = filter(long, colapsada),
             aes(shape = metodo_xai, color = metodo_xai),
             fill = "white", size = 1.6, stroke = 0.7, show.legend = FALSE) +
  geom_point(data = legend_dummy, aes(x = x, y = y, alpha = k),
             shape = 21, fill = "white", color = "grey25", size = 0, stroke = 0,
             inherit.aes = FALSE) +
  geom_segment(data = ann, aes(x = x0, xend = x1, y = y, yend = y, color = metodo_xai),
               linewidth = 0.9, lineend = "round", inherit.aes = FALSE, show.legend = FALSE) +
  geom_text(data = ann, aes(x = xt, y = y, label = label),
            hjust = 0, size = 1.85, color = "black", inherit.aes = FALSE) +
  facet_grid(rows = vars(xlabel), cols = vars(ylabel), scales = "free_x") +
  scale_color_manual(values = METHOD_COLOR, labels = METHOD_LABEL, name = NULL) +
  scale_fill_manual(values = METHOD_COLOR, labels = METHOD_LABEL, name = NULL) +
  scale_shape_manual(values = METHOD_SHAPE, labels = METHOD_LABEL, name = NULL) +
  scale_alpha_manual(values = c("Condição colapsada (AUC < 0,6)" = 1), name = NULL) +
  scale_y_continuous(breaks = seq(0.2, 1, 0.2), labels = function(v) sub("\\.", ",", sprintf("%.1f", v))) +
  scale_x_continuous(breaks = scales::breaks_pretty(n = 3),
                     labels = function(v) sub("\\.", ",", format(v, trim = TRUE))) +
  coord_cartesian(ylim = YLIM, clip = "on") +
  labs(x = NULL, y = NULL) +
  theme_bw(base_size = 8, base_family = "serif") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.3, color = "grey85"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
    strip.background = element_rect(fill = "grey88", color = NA),
    strip.text = element_text(size = 8.4, margin = margin(4, 4, 4, 4)),
    legend.position = "bottom",
    legend.text = element_text(size = 6.3),
    legend.spacing.x = unit(0.3, "lines"),
    legend.margin = margin(t = 0),
    legend.box.margin = margin(t = -2),
    axis.text = element_text(size = 6.6),
    plot.margin = margin(4, 4, 2, 2)
  ) +
  guides(color = guide_legend(order = 1, override.aes = list(size = 2)),
         fill = guide_legend(order = 1), shape = guide_legend(order = 1),
         alpha = guide_legend(order = 2, override.aes = list(size = 1.6, stroke = 0.7)))

ggsave(paste0(OUT_STEM, ".pdf"), p, width = 5.4, height = 4.1, units = "in", device = cairo_pdf)
ggsave(paste0(OUT_STEM, ".png"), p, width = 5.4, height = 4.1, units = "in", dpi = 300)
cat("Salvo:", OUT_STEM, "\n")
