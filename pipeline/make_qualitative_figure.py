#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figura qualitativa para o artigo: 1 linha x 5 paineis --
(A) amostra real do FGADR, (B) anotacao clinica (mascara de lesao real,
uniao de todos os tipos -- ver utils_common.load_gt_mask), (C) Grad-CAM,
(D) LIME, (E) Occlusion Sensitivity, todos sobre a MESMA imagem e o MESMO
checkpoint real (filtro "baseline", a condicao sem pre-processamento,
10/10 repeticoes completas na mini-campanha -- ver
experiments/paper_auc_xai_correlacao/relatorio_completo.md).

A imagem escolhida e a de MAIOR area de lesao anotada entre as imagens do
split de teste que tem mascara (nao restrita a subamostra fixa de 100 da
mini-campanha) -- escolha deliberada para legibilidade visual da figura,
sem efeito nenhum sobre os resultados quantitativos reportados em outro
lugar (que nao dependem de qual imagem e mostrada aqui).

Uso (dentro do container, com GPU):
    python3 make_qualitative_figure.py --base-ssd $SSD_BASE --base-home $HOME_BASE
Gera:
    experiments/paper_auc_xai_correlacao/figura_qualitativa.pdf / .png
"""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_common import setup_logging, get_paths, load_label_map, load_gt_mask
from utils_gradcam import generate_gradcam

FILTER_NAME = "baseline"
IMG_SIZE = (299, 299)
OUT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "paper_auc_xai_correlacao"

# Mesmos hiperparametros usados no resto da mini-campanha (ver
# script3_avalia.py / run_sanity_check.py) -- consistencia entre figuras.
LIME_KWARGS = {"num_samples": 150, "num_segments": 50}
OCCLUSION_KWARGS = {"patch_size": 32, "stride": 16}


def find_checkpoint(checkpoint_dir: Path, filter_name: str, logger):
    for rep in range(10, 20):
        p = checkpoint_dir / f"{filter_name}-{rep}.keras"
        if p.exists():
            return p, rep
    logger.error(f"Nenhum checkpoint mantido encontrado para {filter_name} em reps 10-19")
    return None, None


def pick_best_image(filtered_dataset_dir: Path, paths: dict, logger):
    """Imagem do split de teste com a MAIOR area de mascara de lesao
    anotada -- escolha deliberada para legibilidade (ver docstring do
    modulo), nao um subconjunto quantitativo."""
    import cv2
    import numpy as np

    split_dir = filtered_dataset_dir / "test"
    image_files = sorted(list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")))
    if not image_files:
        logger.error(f"Nenhuma imagem em {split_dir}")
        return None, None, None

    label_map = load_label_map(paths["labels_csv"])
    best = None
    for f in image_files:
        if f.name not in label_map:
            continue
        mask = load_gt_mask(f.name, paths["mask_dirs"])
        if mask is None:
            continue
        area = float(mask.sum())
        if best is None or area > best[2]:
            best = (f, mask, area)

    if best is None:
        logger.error("Nenhuma imagem com mascara de lesao encontrada")
        return None, None, None

    img_file, mask, area = best
    logger.info(f"Imagem escolhida: {img_file.name} (area de mascara = {area:.0f} px)")

    img = cv2.imread(str(img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE).astype("float32")
    mask_resized = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    return img, mask_resized, img_file.name


def main():
    parser = argparse.ArgumentParser(description="Figura qualitativa (amostra + anotacao + 3 mapas XAI)")
    parser.add_argument("--base-ssd", type=str, default=None)
    parser.add_argument("--base-home", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true",
                         help="Recomputa os mapas XAI do zero em vez de reusar o cache salvo")
    args = parser.parse_args()

    logger = setup_logging("make_qualitative_figure")
    paths = get_paths(args.base_ssd, args.base_home)

    import numpy as np

    cache_path = OUT_DIR / "figura_qualitativa_cache.npz"
    if cache_path.exists() and not args.no_cache:
        # Reaproveita os mapas ja computados (custam inferencia real em
        # GPU + LIME/Occlusion, minutos) -- so refaz o RENDER (estilo,
        # cores, layout), pra iterar na figura sem gastar tempo de fila
        # do cluster de novo a cada ajuste visual. --no-cache forca
        # recomputar tudo (ex.: se a imagem/checkpoint escolhido mudar).
        logger.info(f"Usando cache: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        img, mask = cached["img"], cached["mask"]
        gradcam_map, lime_map, occlusion_map = cached["gradcam"], cached["lime"], cached["occlusion"]
        img_name, pred = str(cached["img_name"]), float(cached["pred"])
    else:
        import tensorflow as tf
        from utils_xai_extra import generate_lime, generate_occlusion_sensitivity

        ckpt_path, rep = find_checkpoint(paths["checkpoints"], FILTER_NAME, logger)
        if ckpt_path is None:
            sys.exit(1)
        logger.info(f"Carregando checkpoint: {ckpt_path} (rep {rep})")
        model = tf.keras.models.load_model(ckpt_path)

        filtered_dataset_dir = paths["base_ssd"] / "datasets_filtrados" / FILTER_NAME
        img, mask, img_name = pick_best_image(filtered_dataset_dir, paths, logger)
        if img is None:
            sys.exit(1)

        images = np.expand_dims(img, axis=0)  # (1, H, W, 3)

        pred = float(model(tf.convert_to_tensor(images, dtype=tf.float32), training=False).numpy()[0, 0])
        logger.info(f"Predicao do modelo para esta imagem: p(referavel) = {pred:.4f}")

        logger.info("Gerando Grad-CAM...")
        gradcam_map = generate_gradcam(model, images, target_size=IMG_SIZE)[0]
        logger.info("Gerando LIME...")
        lime_map = generate_lime(model, images, target_size=IMG_SIZE, **LIME_KWARGS)[0]
        logger.info("Gerando Occlusion Sensitivity...")
        occlusion_map = generate_occlusion_sensitivity(model, images, target_size=IMG_SIZE, **OCCLUSION_KWARGS)[0]

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, img=img, mask=mask, gradcam=gradcam_map, lime=lime_map,
                 occlusion=occlusion_map, img_name=img_name, pred=pred)
        logger.info(f"Cache salvo: {cache_path}")

    render_figure(img, mask, gradcam_map, lime_map, occlusion_map, img_name, pred, logger)


def render_figure(img, mask, gradcam_map, lime_map, occlusion_map, img_name, pred, logger):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],  # ver nota de fonte nos outros scripts da pasta
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Fonte dos titulos em caixa: serif em negrito, para casar com o
    # restante do artigo -- o sbc-template.sty do artigo carrega
    # \RequirePackage{times} (Times, uma serifada), entao o corpo do texto
    # do artigo inteiro e serifado, nao sans-serif. "DejaVu Serif" e a
    # mesma familia ja usada no resto desta figura e confirmada (via
    # BaseFont no PDF) como identica a usada em figura_multimetodo_retas_
    # completas -- sem acesso a Times real neste ambiente (fc-list).
    TITLE_FONT = {"family": "DejaVu Serif", "weight": "bold", "size": 7.8}

    def rgba_overlay(heat, cmap_name, max_alpha):
        # Opacidade PROPORCIONAL ao valor do heatmap (nao uma opacidade
        # fixa por cima do overlay inteiro) -- sem isso, ate as regioes de
        # importancia ~0 recebem uma tinta solida (azul do "jet", rosa
        # claro do "Reds"), lavando a imagem inteira e fazendo o LIME (que
        # e naturalmente um mosaico de superpixels) parecer "colorido por
        # toda parte" em vez de destacar so as regioes realmente
        # importantes. Padrao usado em figuras de XAI publicadas.
        import matplotlib.cm as cm
        heat = np.clip(heat, 0.0, 1.0)
        rgba = cm.get_cmap(cmap_name)(heat)
        rgba[..., 3] = heat * max_alpha
        return rgba

    img_u8 = img.astype("uint8")
    # Painel B: mascara CRUA (fundo preto, lesao em branco), do jeito que
    # o arquivo original do FGADR realmente e (confirmado lendo os PNGs
    # brutos de HardExudate/Hemohedge/Microaneurysms/... diretamente:
    # binario 0/255, sem sobrepor na foto) -- nao um overlay tingido em
    # cima da imagem, que era como estava antes.
    mask_raw = (np.clip(mask, 0, 1) * 255).astype("uint8")
    # Titulos curtos e de uma linha so -- com a caixa agora ocupando a
    # largura inteira do painel, nao precisa de quebra de linha nem repetir
    # "(mapa de calor)" em cada uma das 3 caixas de XAI (isso e dito uma
    # unica vez, em um rotulo de grupo acima delas, mais abaixo).
    panels = [
        ("Imagem original (FGADR)", "photo", img_u8, None, None, None),
        ("Anotação clínica", "mask", mask_raw, None, None, None),
        ("Grad-CAM", "photo", img_u8, gradcam_map, "jet", 0.75),
        ("LIME", "photo", img_u8, lime_map, "jet", 0.75),
        ("Occlusion Sensitivity", "photo", img_u8, occlusion_map, "jet", 0.75),
    ]

    # BOX_Y0=1.0 -- a base da caixa fica colada exatamente na borda de
    # cima da imagem, sem nenhum espaco entre elas.
    BOX_Y0, BOX_H = 1.0, 0.22  # em fracao dos eixos (axes fraction)

    fig, axes = plt.subplots(1, 5, figsize=(9.5, 2.55), dpi=600)
    for ax, (title, kind, base, overlay, cmap, max_alpha) in zip(axes, panels):
        if kind == "mask":
            ax.imshow(base, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(base)
            if overlay is not None:
                ax.imshow(rgba_overlay(overlay, cmap, max_alpha))
        ax.set_xticks([])
        ax.set_yticks([])
        # Sem borda ao redor da propria imagem (estilo da referencia: a
        # borda preta fica so na caixa do titulo, nao no painel da foto).
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Titulo em caixa com a MESMA LARGURA do painel da imagem (retangulo
        # desenhado de x=0 a x=1 em coordenadas do eixo, nao uma caixa de
        # texto que so abraca o proprio texto) -- reproduz o estilo da
        # figura de referencia.
        ax.add_patch(plt.Rectangle(
            (0.0, BOX_Y0), 1.0, BOX_H, transform=ax.transAxes, clip_on=False,
            facecolor="white", edgecolor="black", linewidth=1.0, zorder=5,
        ))
        ax.text(0.5, BOX_Y0 + BOX_H / 2, title, transform=ax.transAxes,
                 ha="center", va="center", fontdict=TITLE_FONT, zorder=6, clip_on=False)

    # tight_layout PRIMEIRO (reserva espaco no topo via rect para as caixas
    # + o rotulo de grupo), so DEPOIS lemos a posicao final dos eixos --
    # ver nota nas versoes anteriores deste script sobre essa ordem.
    fig.tight_layout(rect=[0, 0, 1, 0.77], pad=0.4, w_pad=0.6)
    fig.canvas.draw()

    # "Mapas de calor" dito uma unica vez, com elegancia, como rotulo de
    # grupo acima das 3 caixas de titulo dos paineis C/D/E (Grad-CAM/LIME/
    # Occlusion) -- em vez de repetir "(mapa de calor)" em cada caixa
    # individual. Um traco fino por baixo marca visualmente que o rotulo
    # se refere aos 3 paineis, nao a um so.
    pos_c, pos_e = axes[2].get_position(), axes[4].get_position()
    box_top_frac = BOX_Y0 + BOX_H
    label_y = pos_c.y0 + box_top_frac * pos_c.height + 0.035
    line_y = pos_c.y0 + box_top_frac * pos_c.height + 0.015
    group_x0, group_x1 = pos_c.x0, pos_e.x1
    group_xc = (group_x0 + group_x1) / 2
    fig.text(group_xc, label_y, "Mapas de calor", ha="center", va="bottom",
              fontsize=8.2, style="italic")
    fig.add_artist(plt.Line2D(
        [group_x0 + 0.01, group_x1 - 0.01], [line_y, line_y],
        transform=fig.transFigure, color="0.4", linewidth=0.6,
    ))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "figura_qualitativa.pdf"
    out_png = OUT_DIR / "figura_qualitativa.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    logger.info(f"Salvo: {out_pdf} / {out_png}")


if __name__ == "__main__":
    main()
