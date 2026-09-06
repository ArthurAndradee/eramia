#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 3 — Avaliação e Coleta de Métricas (por filtro + repetição)

Objetivo: Avaliar o modelo treinado e extrair métricas XAI + clínicas.

Entrada: 
  - Checkpoint do modelo (output de Script 2)
  - Dataset filtrado imagens (input para modelo)
  - Máscaras originais do FGADR (para validação anatômica)

Saída: 
  - CSV em $HOME/resultados/{filtro}_{repeticao}_{timestamp}.csv
  - Success marker em $HOME/resultados/_success_markers/{filtro}_{repeticao}.ok

Métricas: IoU, Dice, Pointing Game Accuracy, AUC-ROC, Sensibilidade, Especificidade

Granularidade: Uma vez por filtro + repetição (após Script 2).

Pós-condições:
  - CSV válido com todas as métricas obrigatórias
  - Marcador de sucesso criado APENAS após validação
"""

import os
import sys
import argparse
import csv
import logging
import traceback
from pathlib import Path
from datetime import datetime
import numpy as np

# Add hcpa to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_common import (
    setup_logging, get_paths, validate_paths,
    create_success_marker, check_success_marker, get_git_commit_hash,
    load_label_map, load_gt_mask
)
from utils_metrics import (
    dice_coefficient, iou_score, pointing_game_accuracy,
    sensitivity, specificity, compute_roc_auc, compute_accuracy, compute_f1_score,
    compute_precision
)
from utils_gradcam import generate_gradcam

logger = logging.getLogger(__name__)

IMG_SIZE = (299, 299)

# Fixed seed for the XAI cross-method comparison subsample (see
# _select_xai_subsample() below) — NOT the training seed (--repeticao).
# Kept as a module constant, not a CLI argument, because it must be
# identical across every filter/repetition/method invocation for the
# multi-method comparison to be apples-to-apples; making it configurable
# would risk an accidental mismatch between runs.
XAI_SUBSAMPLE_SEED = 20260818

_XAI_METHOD_FUNCS = {}


def _get_xai_method_funcs():
    """Lazily build the method-name -> generate_*(model, images, target_size)
    registry (avoids importing utils_xai_extra, and therefore TensorFlow,
    at module load time for callers — e.g. validate_csv() below — that
    don't need it)."""
    global _XAI_METHOD_FUNCS
    if not _XAI_METHOD_FUNCS:
        from utils_xai_extra import generate_lime, generate_occlusion_sensitivity
        _XAI_METHOD_FUNCS = {
            "gradcam": generate_gradcam,
            "lime": generate_lime,
            "occlusion": generate_occlusion_sensitivity,
        }
    return _XAI_METHOD_FUNCS


def _select_xai_subsample(image_files: list, gt_mask_lookup: dict, n: int, logger) -> set:
    """
    Deterministic subset of `n` filenames (from the images that have a
    lesion mask) used for the cross-method XAI comparison — Occlusion
    Sensitivity costs one extra forward pass per grid position per image,
    so evaluating it (and, for a fair comparison, the other methods too)
    on the full ~553-image test split across 10 filters x 10 repetitions
    x 3 methods was judged too expensive; see the XAI methodology review
    (experiments/paper_auc_xai_correlacao). Fixed seed (XAI_SUBSAMPLE_SEED)
    and fixed input ordering (sorted filenames) make the same `n` images
    get picked for every filter/repetition/method — required for the
    cross-method agreement metric to compare the same images.
    """
    import random
    eligible = sorted(f.name for f in image_files if f.name in gt_mask_lookup)
    if n is None or n >= len(eligible):
        return set(eligible)
    chosen = set(random.Random(XAI_SUBSAMPLE_SEED).sample(eligible, n))
    logger.info(f"XAI cross-method subsample: {len(chosen)}/{len(eligible)} images "
                f"(seed={XAI_SUBSAMPLE_SEED})")
    return chosen


def load_model_and_predict(model_path: Path, filtered_dataset_dir: Path,
                           split_name: str, paths: dict, logger,
                           xai_methods: list = None,
                           xai_subsample: int = None) -> tuple:
    """
    Load trained model and run predictions + XAI saliency map(s) on the
    test split.

    Args:
        model_path: Path to saved model checkpoint
        filtered_dataset_dir: Path to filtered dataset
        split_name: Which split to evaluate (must be 'test')
        paths: Dictionary of paths from get_paths() (labels_csv, mask_dirs)
        logger: Logger instance
        xai_methods: List of method names to compute saliency maps with,
            drawn from _get_xai_method_funcs()'s registry. Defaults to
            ["gradcam"] — the original, single-method behavior.
        xai_subsample: If given and there is more than one method (or the
            caller passes it explicitly with one method), restrict the
            XAI evaluation subset to this many images (see
            _select_xai_subsample()) instead of every image with a lesion
            mask. None (default) evaluates all of them, matching the
            original behavior.

    Returns:
        Tuple of (predictions_prob, predictions_binary, ground_truth,
                  saliency_by_method, gt_masks, model). saliency_by_method
        is a dict {method_name: saliency_maps_array}; gt_masks is shared
        across methods (same images, same masks). Both only include
        images that have at least one lesion mask annotated (further
        restricted to the fixed subsample when xai_subsample is set); the
        rest are logged as skipped (no silent cap).
    """
    xai_methods = xai_methods or ["gradcam"]
    try:
        import cv2
        import tensorflow as tf

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None, None, None, None, None, None

        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info(f"✓ Model loaded: {model}")

        # Load images from split
        split_dir = filtered_dataset_dir / split_name
        image_files = sorted(list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png")))

        if not image_files:
            logger.warning(f"No images found in {split_dir}")
            return None, None, None, None, None, None

        logger.info(f"Loading {len(image_files)} images for inference...")

        label_map = load_label_map(paths["labels_csv"])
        missing_labels = [f.name for f in image_files if f.name not in label_map]
        if missing_labels:
            logger.error(f"{len(missing_labels)} test image(s) have no ground-truth label in "
                         f"{paths['labels_csv']}: {missing_labels[:10]}{'...' if len(missing_labels) > 10 else ''}")
            return None, None, None, None, None, None

        images = []
        ground_truth = []
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            # Keep raw [0, 255] range — EfficientNetV2S bundles its own
            # Rescaling/Normalization as the first model layer (see
            # dr_hcpa_v2_2024.py build_model()/decode_image()); dividing by
            # 255 here would double-normalize and mismatch training-time
            # inputs.
            img = img.astype(np.float32)
            images.append(img)
            ground_truth.append(label_map[img_file.name])

        images_array = np.array(images)
        ground_truth = np.array(ground_truth)
        logger.info(f"Image array shape: {images_array.shape}")

        # Run inference
        logger.info("Running inference...")
        predictions_prob = model.predict(images_array, verbose=0)
        predictions_prob = np.squeeze(predictions_prob)
        predictions_binary = (predictions_prob > 0.5).astype(int)

        logger.info(f"Predictions shape: {predictions_prob.shape}")

        # Ground-truth lesion masks first (needed to pick the XAI subset
        # BEFORE running any saliency method, so a subsample restricts
        # which images every method is computed on, not just which ones
        # get kept afterward — avoids wasting compute on images we'd
        # discard anyway).
        gt_mask_lookup = {}
        skipped = 0
        for img_file in image_files:
            gt_mask = load_gt_mask(img_file.name, paths["mask_dirs"])
            if gt_mask is None:
                skipped += 1
                continue
            gt_mask_lookup[img_file.name] = cv2.resize(gt_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        subsample_names = _select_xai_subsample(image_files, gt_mask_lookup, xai_subsample, logger)
        keep_flags = [f.name in gt_mask_lookup and f.name in subsample_names for f in image_files]
        keep_mask = np.array(keep_flags)
        gt_masks = np.array([gt_mask_lookup[f.name] for f, keep in zip(image_files, keep_flags) if keep]) \
            if any(keep_flags) else None
        images_subset = images_array[keep_mask] if keep_mask.any() else images_array[:0]

        logger.info(
            f"XAI evaluation subset: {int(keep_mask.sum())}/{len(image_files)} images have a lesion "
            f"mask and are in the chosen subsample ({skipped} skipped entirely — typically ICDR "
            f"grade 0 images with no lesion annotation)"
        )

        method_funcs = _get_xai_method_funcs()
        saliency_by_method = {}
        for method in xai_methods:
            if method not in method_funcs:
                logger.error(f"Unknown XAI method '{method}' — skipping (known: {list(method_funcs)})")
                continue
            logger.info(f"Generating {method} saliency maps ({len(images_subset)} images)...")
            saliency_by_method[method] = method_funcs[method](model, images_subset, target_size=IMG_SIZE) \
                if len(images_subset) > 0 else None

        return predictions_prob, predictions_binary, ground_truth, saliency_by_method, gt_masks, model

    except Exception as e:
        logger.error(f"Failed to load model and predict: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None, None, None


def _base_metrics(filter_name: str, repetition: int, predictions_prob: np.ndarray,
                   ground_truth: np.ndarray, logger) -> dict:
    """Clinical metrics shared by every XAI method's row (classification
    doesn't depend on which saliency method is used to explain it)."""
    return {
        "filter_name": filter_name,
        "repetition": repetition,
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(logger),
        "num_samples": len(ground_truth),
        "accuracy": compute_accuracy(ground_truth, predictions_prob),
        "sensitivity": sensitivity(ground_truth, predictions_prob),
        "specificity": specificity(ground_truth, predictions_prob),
        "precision": compute_precision(ground_truth, predictions_prob),
        "f1_score": compute_f1_score(ground_truth, predictions_prob),
        "auc_roc": compute_roc_auc(ground_truth, predictions_prob),
    }


def compute_metrics_for_split(filter_name: str, repetition: int,
                              predictions_prob: np.ndarray,
                              predictions_binary: np.ndarray,
                              ground_truth: np.ndarray,
                              saliency_by_method: dict = None,
                              gt_masks: np.ndarray = None,
                              logger = None) -> list:
    """
    Compute metrics for a split — one row per XAI method in
    `saliency_by_method`, each sharing the same clinical metrics
    (accuracy/sensitivity/.../AUC don't depend on the explanation method)
    but with its own Dice/IoU/Pointing Game against `gt_masks`.

    Args:
        filter_name: Name of the filter
        repetition: Repetition number
        predictions_prob: Predicted probabilities
        predictions_binary: Predicted binary labels
        ground_truth: Ground truth labels
        saliency_by_method: {method_name: saliency_maps_array}, as returned
            by load_model_and_predict(). None/empty -> a single row with
            no XAI metrics (clinical metrics only).
        gt_masks: Ground truth spatial masks, shared across methods
        logger: Logger instance

    Returns:
        List of metric dictionaries (length = max(1, len(saliency_by_method))),
        each with an "xai_method" key identifying which saliency method (or
        None if no XAI metrics were computed at all).
    """
    if predictions_prob is None or ground_truth is None:
        logger.warning("Cannot compute metrics - missing predictions or ground truth")
        return []

    try:
        if not saliency_by_method or gt_masks is None or len(gt_masks) == 0:
            logger.error("No XAI subset available (no test images with lesion masks) — "
                         "Dice/IoU/Pointing Game cannot be computed for this run")
            metrics = _base_metrics(filter_name, repetition, predictions_prob, ground_truth, logger)
            metrics["xai_method"] = None
            return [metrics]

        rows = []
        for method, saliency_maps in saliency_by_method.items():
            metrics = _base_metrics(filter_name, repetition, predictions_prob, ground_truth, logger)
            metrics["xai_method"] = method

            if saliency_maps is None or len(saliency_maps) == 0:
                logger.error(f"Method '{method}' produced no saliency maps — skipping its XAI metrics")
                rows.append(metrics)
                continue

            dice_scores = [dice_coefficient(saliency_maps[i], gt_masks[i]) for i in range(len(saliency_maps))]
            iou_scores = [iou_score(saliency_maps[i], gt_masks[i]) for i in range(len(saliency_maps))]
            pointing_scores = [pointing_game_accuracy(saliency_maps[i], gt_masks[i]) for i in range(len(saliency_maps))]

            metrics["xai_num_samples"] = len(saliency_maps)
            metrics["dice_mean"] = float(np.mean(dice_scores))
            metrics["dice_std"] = float(np.std(dice_scores))
            metrics["iou_mean"] = float(np.mean(iou_scores))
            metrics["iou_std"] = float(np.std(iou_scores))
            metrics["pointing_game_accuracy"] = float(np.mean(pointing_scores))

            logger.info(f"Computed metrics for {filter_name}-{repetition} [{method}]:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
            rows.append(metrics)

        return rows

    except Exception as e:
        logger.error(f"Failed to compute metrics: {e}")
        logger.error(traceback.format_exc())
        return []


def save_results_csv(metrics_list: list, output_path: Path, logger) -> bool:
    """
    Save metrics to CSV file.
    
    Args:
        metrics_list: List of metric dictionaries
        output_path: Path to save CSV
        logger: Logger instance
    
    Returns:
        True if successful
    """
    try:
        if not metrics_list:
            logger.error("No metrics to save")
            return False
        
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Get all unique keys
        fieldnames = set()
        for metrics in metrics_list:
            fieldnames.update(metrics.keys())
        
        fieldnames = sorted(list(fieldnames))
        
        logger.info(f"Saving results to {output_path}")
        logger.info(f"Fieldnames: {fieldnames}")
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_list)
        
        logger.info(f"✓ CSV saved with {len(metrics_list)} rows")
        
        # Validate CSV
        if not validate_csv(output_path, logger):
            logger.error("CSV validation failed")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        return False


def validate_csv(csv_path: Path, logger) -> bool:
    """
    Validate that CSV has required fields and no NaN values.
    
    Args:
        csv_path: Path to CSV file
        logger: Logger instance
    
    Returns:
        True if CSV is valid
    """
    try:
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        
        # Check required fields (clinical + XAI metrics are both mandatory)
        required_fields = [
            "filter_name", "repetition", "accuracy",
            "sensitivity", "specificity", "auc_roc", "f1_score",
            "dice_mean", "iou_mean", "pointing_game_accuracy",
        ]
        
        for field in required_fields:
            if field not in df.columns:
                logger.error(f"Missing required field in CSV: {field}")
                return False
        
        # Check for NaN values in required fields
        for field in required_fields:
            if df[field].isna().any():
                logger.error(f"Found NaN values in required field: {field}")
                return False
        
        logger.info(f"✓ CSV validation passed: {len(df)} rows, {len(df.columns)} columns")
        return True
    
    except Exception as e:
        logger.error(f"CSV validation failed: {e}")
        return False


def sweep_orphaned_checkpoints(checkpoint_dir: Path, success_markers_dir: Path, logger) -> None:
    """
    Self-healing safety net (found necessary via a real orphaned checkpoint
    left on grace1's SSD during infra validation: an older code version had
    evaluated a filter+repetition successfully — success marker + CSV both
    present — but its .keras was never deleted, and nothing ever revisits a
    filter+repetition once its marker exists, so it would have sat there for
    the rest of the campaign).

    Scans checkpoint_dir for ANY *.keras whose filter+repetition already has
    a success marker (meaning it was already evaluated successfully and
    SHOULD have been deleted by delete_checkpoint_after_success() — this
    catches the case where that deletion was skipped for any reason: an
    older code version, a crash between model.save() and the delete call,
    etc.). Runs after every successful evaluation, so an orphan left behind
    by any past run on this node gets swept up the next time evaluation
    succeeds here — bounds the leak window to "until this node runs one more
    eval" instead of "forever".

    Never raises: this is a best-effort cleanup, not a correctness
    requirement — a failure here must not turn an already-successful
    evaluation into a failed script3 run.
    """
    try:
        candidates = list(checkpoint_dir.glob("*.keras"))
    except Exception as e:
        logger.warning(f"Could not scan {checkpoint_dir} for orphaned checkpoints: {e}")
        return

    for ckpt in candidates:
        try:
            stem = ckpt.stem  # "{filter_name}-{repetition}"
            if "-" not in stem:
                continue
            filter_name, rep_str = stem.rsplit("-", 1)
            if not rep_str.isdigit():
                continue
            marker_path = success_markers_dir / f"{filter_name}_{rep_str}.ok"
            if marker_path.exists():
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                ckpt.unlink()
                logger.warning(
                    f"Swept orphaned checkpoint (success marker already existed, so this "
                    f".keras should already have been deleted): {ckpt} ({size_mb:.1f} MB freed)"
                )
        except Exception as e:
            logger.warning(f"Could not check/sweep possible orphaned checkpoint {ckpt}: {e}")


def delete_checkpoint_after_success(model_path: Path, logger) -> None:
    """
    Delete the (~250MB) .keras checkpoint after a filter+repetition has been
    fully and successfully evaluated (CSV validated + success marker already
    created by the caller). Across the ~180-run experimental matrix this is
    the difference between ~0GB and ~45GB of otherwise-unneeded model files
    sitting on node-local SSD scratch.

    Only the .keras file itself is removed. The small sibling artifacts
    written by dr_hcpa_v2_2024.py (training_metadata_{filtro}_{rep}.json,
    the phase1/phase2 CSV logs, the thresholds CSV, the ROC PDF) are kept —
    they're KB-sized (not the ~45GB problem), useful for later audit/debug,
    and training_metadata_*.json in particular is still read by
    script2_treina.py's own idempotency check for the intermediate
    "trained but not yet evaluated" state (see run_training() in
    script2_treina.py, which now also checks the success marker directly so
    it correctly skips retraining even after this function deletes the
    checkpoint).

    Never raises: a failure here must not turn an already-successful
    evaluation (CSV + success marker already on disk) into a failed script3
    run — it's logged as a warning and left for manual cleanup instead.
    """
    try:
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            model_path.unlink()
            logger.info(f"✓ Deleted checkpoint after successful evaluation: {model_path} ({size_mb:.1f} MB freed)")
        else:
            logger.warning(f"Checkpoint not found at {model_path} — nothing to delete (already removed?)")
    except Exception as e:
        logger.warning(f"Could not delete checkpoint {model_path} after successful evaluation: {e} "
                        f"(evaluation itself succeeded — this is non-fatal, but the checkpoint will "
                        f"linger on disk until manually removed)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Script 3: Evaluate model and compute metrics"
    )
    parser.add_argument(
        "--filtro",
        type=str,
        required=True,
        help="Filter name/parameters"
    )
    parser.add_argument(
        "--repeticao",
        type=int,
        required=True,
        help="Repetition number (0-9)"
    )
    parser.add_argument(
        "--base-ssd",
        type=str,
        default=None,
        help="Base SSD path"
    )
    parser.add_argument(
        "--base-home",
        type=str,
        default=None,
        help="Base home path"
    )
    parser.add_argument(
        "--xai-methods",
        type=str,
        default="gradcam",
        help="Comma-separated XAI methods to evaluate (gradcam, lime, "
             "occlusion). Default: gradcam only — unchanged original behavior."
    )
    parser.add_argument(
        "--xai-subsample",
        type=int,
        default=None,
        help="If set, restrict XAI evaluation (all requested methods) to this many "
             "images (deterministic, see XAI_SUBSAMPLE_SEED), instead of every test "
             "image with a lesion mask. Default: no subsampling (original behavior)."
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Skip deleting the .keras checkpoint (and skip the orphan sweep) after "
             "a successful evaluation. Opt-in only — default is the original "
             "delete-after-eval behavior; used by the XAI methodology mini-campaign, "
             "which needs the checkpoint to still exist for the sanity check."
    )

    args = parser.parse_args()
    xai_methods = [m.strip() for m in args.xai_methods.split(",") if m.strip()]
    
    # Setup logging
    logger = setup_logging(f"script3_avalia_{args.filtro}_{args.repeticao}")
    
    logger.info("=" * 80)
    logger.info(f"Starting evaluation: filter={args.filtro}, repetition={args.repeticao}")
    logger.info("=" * 80)
    
    try:
        # Get paths
        paths = get_paths(args.base_ssd, args.base_home)
        
        # Validate paths
        if not validate_paths(paths, logger):
            logger.error("Path validation failed!")
            sys.exit(1)
        
        # Check for existing success marker
        marker_path = paths["success_markers"] / f"{args.filtro}_{args.repeticao}.ok"
        if check_success_marker(marker_path, logger):
            logger.info(f"✓ Success marker already exists: {marker_path}")
            logger.info("Skipping evaluation (already completed)")
            sys.exit(0)
        
        # Load filtered dataset directory
        filtered_dataset_dir = paths["datasets_filtrados"] / args.filtro
        if not filtered_dataset_dir.exists():
            logger.error(f"Filtered dataset not found: {filtered_dataset_dir}")
            sys.exit(1)
        
        # Load model checkpoint — dr_hcpa_v2_2024.py saves as "{dataset}-{exec}.keras"
        checkpoint_dir = paths["checkpoints"]
        model_path = checkpoint_dir / f"{args.filtro}-{args.repeticao}.keras"

        predictions_prob, predictions_binary, ground_truth, saliency_by_method, gt_masks, _ = load_model_and_predict(
            model_path, filtered_dataset_dir, "test", paths, logger,
            xai_methods=xai_methods, xai_subsample=args.xai_subsample
        )

        if predictions_prob is None or ground_truth is None:
            logger.error("Evaluation cannot proceed without model predictions and ground truth labels.")
            sys.exit(1)

        metrics_rows = compute_metrics_for_split(
            args.filtro, args.repeticao,
            predictions_prob, predictions_binary, ground_truth,
            saliency_by_method=saliency_by_method, gt_masks=gt_masks,
            logger=logger
        )

        if not metrics_rows:
            logger.error("Failed to compute metrics")
            sys.exit(1)

        # Save results — one CSV row per requested XAI method (see
        # compute_metrics_for_split()); metadata on the success marker uses
        # the first row, which is enough to identify the run (filter/rep are
        # identical across rows, only xai_method/Dice/IoU/PG differ).
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = paths["resultados"] / f"{args.filtro}_{args.repeticao}_{timestamp}.csv"

        if not save_results_csv(metrics_rows, csv_path, logger):
            logger.error("Failed to save results CSV")
            sys.exit(1)

        # Create success marker
        try:
            create_success_marker(marker_path, logger, metadata=metrics_rows[0])
            logger.info(f"✓ Created success marker: {marker_path}")
        except Exception as e:
            logger.error(f"Failed to create success marker: {e}")
            sys.exit(1)

        # Delete the checkpoint ONLY now — after predictions, metrics, CSV
        # (validated) and success marker have all succeeded. This is the
        # single point in the whole pipeline where a filter+repetition is
        # unambiguously "done" (script2's own idempotency check, script4's
        # verify_success_markers(), and the orchestrator's skip_completed
        # logic all key off this same marker), so deletion is gated on it
        # rather than on script3 merely running to its end. At ~250MB per
        # checkpoint x ~180 runs (~45GB), the .keras is the only artifact
        # deleted here — the small CSV/PDF/metadata siblings are kept (see
        # delete_checkpoint_after_success() docstring for why).
        #
        # --keep-checkpoint (opt-in, default off) skips both this and the
        # orphan sweep below — used only by the XAI mini-campaign, which
        # needs the checkpoint to still exist afterward for the sanity
        # check (weight randomization). Every other caller is unaffected.
        if args.keep_checkpoint:
            logger.info(f"--keep-checkpoint set: leaving {model_path} on disk")
        else:
            delete_checkpoint_after_success(model_path, logger)
            # Best-effort sweep of any OTHER orphaned checkpoint left on this
            # same node by a past run (see sweep_orphaned_checkpoints() docstring).
            sweep_orphaned_checkpoints(checkpoint_dir, paths["success_markers"], logger)

        logger.info("=" * 80)
        logger.info(f"✓ Evaluation completed successfully")
        logger.info("=" * 80)

        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
