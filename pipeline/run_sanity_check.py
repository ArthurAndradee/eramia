#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone runner for the Adebayo et al. cascading model-randomization
sanity check (utils_sanity_check.py), against REAL trained checkpoints
kept from the XAI mini-campaign (see experiments/filter_matrix_xai_mini.json,
run_xai_mini_campaign.sh, --keep-checkpoint).

For each of the 10 mini-campaign filters, picks one kept checkpoint
(the lowest available repetition in 10-19), loads a small deterministic
subsample of test images with lesion masks (same selection logic as
script3_avalia.py's cross-method comparison, just a smaller n — this is
a diagnostic check, not a metric averaged over the full test set), and
runs the cascading randomization test with all 3 XAI methods
(gradcam, lime, occlusion).

Output: JSON per filter in resultados/sanity_check/{filter}.json, plus
a printed summary table.
"""

import sys
import os
import json
import argparse
import logging
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_common import setup_logging, get_paths, load_label_map, load_gt_mask
from utils_gradcam import generate_gradcam
from utils_sanity_check import cascading_randomization_test

IMG_SIZE = (299, 299)
N_IMAGES = 15  # "a handful of images, e.g. 10-20, is enough" — see utils_sanity_check.py
SUBSAMPLE_SEED = 20260818  # same seed as script3_avalia.py's XAI_SUBSAMPLE_SEED

FILTERS = [
    "baseline", "BenGraham_MaxGreen2.0", "BenGraham_GreenChannel_CLAHE4.0_Otsu",
    "Gamma0.8_MaxGreen2.0_Otsu", "BenGraham_MaxGreen2.0_Canny", "Retinex_Gaussian",
    "Retinex_MaxGreen2.0_Gaussian", "Retinex_MaxGreen2.0_CLAHE4.0_Otsu",
    "Retinex_Grayscale_CLAHE4.0_Otsu", "LABNorm_MaxGreen2.0_Frangi",
]


def find_checkpoint(checkpoint_dir: Path, filter_name: str, logger) -> tuple:
    """Lowest available kept checkpoint in reps 10-19 for this filter."""
    for rep in range(10, 20):
        p = checkpoint_dir / f"{filter_name}-{rep}.keras"
        if p.exists():
            return p, rep
    logger.error(f"No kept checkpoint found for {filter_name} in reps 10-19")
    return None, None


def load_subsample_images(filtered_dataset_dir: Path, paths: dict, n: int, logger):
    import cv2
    import random
    import numpy as np

    split_dir = filtered_dataset_dir / "test"
    image_files = sorted(list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")))
    if not image_files:
        logger.error(f"No test images found in {split_dir}")
        return None, None

    label_map = load_label_map(paths["labels_csv"])

    eligible = []
    for f in image_files:
        if f.name not in label_map:
            continue
        gt_mask = load_gt_mask(f.name, paths["mask_dirs"])
        if gt_mask is not None:
            eligible.append(f.name)
    eligible = sorted(eligible)
    if not eligible:
        logger.error("No lesion-annotated images found for subsample")
        return None, None

    chosen_names = set(random.Random(SUBSAMPLE_SEED).sample(eligible, min(n, len(eligible))))
    chosen_files = [f for f in image_files if f.name in chosen_names]

    images = []
    for img_file in chosen_files:
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        images.append(img.astype("float32"))

    return np.array(images), [f.name for f in chosen_files]


def summarize(results: dict) -> str:
    """rho at step 0 vs rho at the last (fully randomized) step, per method."""
    lines = []
    for method, per_step in results.items():
        steps = sorted(per_step.keys())
        rho0 = per_step[steps[0]]
        rho_last = per_step[steps[-1]]
        verdict = "PASSA (rho cai)" if (rho_last == rho_last and rho0 == rho0 and rho_last < 0.3 * rho0) else \
                  ("INCONCLUSIVO (NaN)" if rho_last != rho_last or rho0 != rho0 else "FALHA (rho nao cai)")
        lines.append(f"    {method:10s}: rho(0)={rho0:.3f} -> rho(fim)={rho_last:.3f}  [{verdict}]")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Adebayo sanity check on real mini-campaign checkpoints")
    parser.add_argument("--base-ssd", type=str, default=None)
    parser.add_argument("--base-home", type=str, default=None)
    parser.add_argument("--n-images", type=int, default=N_IMAGES)
    parser.add_argument("--n-steps", type=int, default=6,
                         help="Randomization steps to test (evenly spaced); None = every top-level layer")
    parser.add_argument("--filters", type=str, default=None,
                         help="Comma-separated subset of FILTERS to run (default: all 10) — for a quick validation pass")
    args = parser.parse_args()
    filters_to_run = args.filters.split(",") if args.filters else FILTERS

    logger = setup_logging("run_sanity_check")
    paths = get_paths(args.base_ssd, args.base_home)
    checkpoint_dir = paths["checkpoints"]
    out_dir = paths["base_home"] / "resultados" / "sanity_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    from utils_xai_extra import generate_lime, generate_occlusion_sensitivity
    method_funcs = {
        "gradcam": generate_gradcam,
        "lime": generate_lime,
        "occlusion": generate_occlusion_sensitivity,
    }

    all_summaries = {}
    for filter_name in filters_to_run:
        logger.info("=" * 70)
        logger.info(f"Filtro: {filter_name}")
        ckpt_path, rep = find_checkpoint(checkpoint_dir, filter_name, logger)
        if ckpt_path is None:
            all_summaries[filter_name] = {"error": "no checkpoint found"}
            continue

        filtered_dataset_dir = paths["base_ssd"] / "datasets_filtrados" / filter_name
        images, image_names = load_subsample_images(filtered_dataset_dir, paths, args.n_images, logger)
        if images is None:
            all_summaries[filter_name] = {"error": "no eligible images found"}
            continue

        try:
            logger.info(f"Loading checkpoint {ckpt_path} (rep {rep})")
            model = tf.keras.models.load_model(ckpt_path)
        except Exception as e:
            logger.error(f"Failed to load {ckpt_path}: {e}")
            all_summaries[filter_name] = {"error": f"checkpoint load failed: {e}"}
            continue

        filter_results = {"checkpoint_rep": rep, "n_images": len(images), "image_names": image_names}
        for method_name, func in method_funcs.items():
            logger.info(f"  Running cascading randomization test with {method_name}...")
            try:
                kwargs = {"num_samples": 60, "num_segments": 30} if method_name == "lime" else \
                         ({"patch_size": 48, "stride": 32} if method_name == "occlusion" else {})
                result = cascading_randomization_test(
                    model, images, func, target_size=IMG_SIZE,
                    n_steps=args.n_steps, **kwargs,
                )
                filter_results[method_name] = result
            except Exception as e:
                logger.error(f"  {method_name} failed: {e}\n{traceback.format_exc()}")
                filter_results[method_name] = {"error": str(e)}

        out_path = out_dir / f"{filter_name}.json"
        out_path.write_text(json.dumps(filter_results, indent=2, default=str))
        logger.info(f"  Saved: {out_path}")

        method_series = {m: filter_results[m] for m in method_funcs if isinstance(filter_results.get(m), dict)
                          and "error" not in filter_results[m]}
        # keys came back as ints from cascading_randomization_test but JSON round-trips them as str;
        # use the in-memory dict directly for the summary (not the reloaded JSON)
        clean_series = {m: {int(k): v for k, v in filter_results[m].items()} for m in method_series}
        all_summaries[filter_name] = clean_series
        logger.info(f"  Resumo:\n{summarize(clean_series)}")

        del model
        tf.keras.backend.clear_session()

    logger.info("=" * 70)
    logger.info("RESUMO FINAL — sanity check de Adebayo (todos os filtros)")
    for filter_name, series in all_summaries.items():
        logger.info(f"\n  {filter_name}:")
        if isinstance(series, dict) and "error" in series:
            logger.info(f"    ERRO: {series['error']}")
        else:
            logger.info(summarize(series))


if __name__ == "__main__":
    main()
