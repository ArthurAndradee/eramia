#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 2 — Treinamento (por filtro + repetição)

Objetivo: Treinar o EfficientNetV2S (warm-up + fine-tuning, ver
dr_hcpa_v2_2024.py) sobre o dataset filtrado.

Entrada: Caminho do dataset filtrado (já criado pelo Script 1), número da repetição/seed

Saída: Checkpoint da repetição, logs de train/validation loss

Granularidade: Uma vez por filtro + repetição (10 repetições por filtro).

Execução: 1 GPU L40S por job, limite de 24h total.

Dependências: Requer Script 1 ter completado com sucesso (via SLURM --dependency).
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# Add hcpa to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_common import (
    setup_logging, get_paths, validate_paths, get_git_commit_hash
)


def run_training(filter_name: str, repetition: int, paths: dict, logger, use_class_weights: bool = True,
                  phase1_epochs: int = None, phase2_epochs: int = None) -> bool:
    """
    Wrapper to run the existing dr_hcpa_v2_2024.py training script.

    Args:
        filter_name: Name/parameters of the filter
        repetition: Repetition number (0-9)
        paths: Dictionary of paths from get_paths()
        logger: Logger instance
        use_class_weights: Default True as of the HPO run of 2026-07-09 (was
            opt-in/False pre-audit) — matches dr_hcpa_v2_2024.py's own
            adopted default. Explicitly passed as --use_class_weights or
            --no-use_class_weights either way (not just omitted when False),
            so this parameter's value is never silently overridden by the
            child script's own default. See experiments/MODEL_OPTIMIZATION_AUDIT.md.
        phase1_epochs/phase2_epochs: TEST-ONLY overrides passed through to
            dr_hcpa_v2_2024.py's own TEST-ONLY flags (default None -> not
            passed at all, so the official matrix is untouched). Only ever
            set these for infrastructure/integration smoke tests, never for
            the real 576-condition campaign.

    Returns:
        True if training completed successfully
    """
    logger.info("=" * 80)
    logger.info(f"Starting training: filter={filter_name}, repetition={repetition}")
    logger.info("=" * 80)
    
    try:
        # Determine paths
        filtered_dataset_dir = paths["datasets_filtrados"] / filter_name
        tfrec_dir = paths["tfrecords_filtrados"] / filter_name
        checkpoint_dir = paths["checkpoints"]

        # Idempotency: two distinct "already done" states now exist, because
        # script3_avalia.py deletes the ~250MB .keras checkpoint immediately
        # after a successful evaluation (to avoid ~45GB of dead weight across
        # the ~180-run matrix) — so "checkpoint exists" is no longer a valid
        # proxy for "fully done" on its own:
        #
        #   1. success marker exists -> filter+repetição already trained AND
        #      evaluated successfully end-to-end (script3 ran to completion
        #      and, as part of that, deleted the checkpoint). Nothing left
        #      to do here at all, regardless of whether the checkpoint file
        #      still happens to be present.
        #   2. checkpoint + training metadata exist but NO success marker ->
        #      training finished in a previous attempt but evaluation
        #      (script3) hasn't succeeded yet — skip retraining, let script3
        #      pick up the existing checkpoint.
        #
        # This matters because the orchestrator resubmits ALL 10 repetitions
        # when re-running a partially-completed filter (e.g. 9/10 succeeded,
        # 1 crashed) — without this check, the 9 already-done reps would be
        # retrained from scratch every time, wasting hours of GPU time each.
        checkpoint_path = checkpoint_dir / f"{filter_name}-{repetition}.keras"
        metadata_path = checkpoint_dir / f"training_metadata_{filter_name}_{repetition}.json"
        success_marker_path = paths["success_markers"] / f"{filter_name}_{repetition}.ok"

        if success_marker_path.exists():
            logger.info(
                f"✓ Success marker already exists for {filter_name}-{repetition} — "
                f"already trained AND evaluated (checkpoint was deleted after evaluation "
                f"succeeded) — skipping training entirely"
            )
            logger.info(f"  Success marker: {success_marker_path}")
            return True

        if checkpoint_path.exists() and metadata_path.exists():
            logger.info(
                f"✓ Checkpoint and training metadata already exist for "
                f"{filter_name}-{repetition} (not yet evaluated) — skipping training (reuse existing checkpoint)"
            )
            logger.info(f"  Checkpoint: {checkpoint_path}")
            return True

        if not filtered_dataset_dir.exists():
            logger.error(f"Filtered dataset not found: {filtered_dataset_dir}")
            return False

        tfrec_files = list(tfrec_dir.glob('train*.tfrec')) + list(tfrec_dir.glob('test*.tfrec'))
        if not tfrec_files:
            logger.warning(f"No TFRecord files found in {tfrec_dir} — generating them now via create-tfrecord.py")
            tfrecord_script = Path(__file__).parent / "create-tfrecord.py"
            result = subprocess.run(
                [
                    "python3", str(tfrecord_script),
                    "--filtro", filter_name,
                    "--base-ssd", str(paths["base_ssd"]),
                    "--base-home", str(paths["base_home"]),
                ],
                timeout=3600,
            )
            if result.returncode != 0:
                logger.error("create-tfrecord.py failed — cannot proceed with training")
                return False

            tfrec_files = list(tfrec_dir.glob('train*.tfrec')) + list(tfrec_dir.glob('test*.tfrec'))
            if not tfrec_files:
                logger.error(f"Still no TFRecord files found in {tfrec_dir} after generation attempt")
                return False

        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup seed for reproducibility
        seed = repetition  # Use repetition number as seed
        
        # The training script path
        training_script = Path(__file__).parent / "dr_hcpa_v2_2024.py"

        if not training_script.exists():
            logger.error(f"Training script not found: {training_script}")
            return False

        # Build command with proper arguments. Batch size / epoch counts per
        # phase are hardcoded inside dr_hcpa_v2_2024.py (paper-mandated
        # values, must stay identical across every filtro x repetição).
        cmd = [
            "python3",
            str(training_script),
            "--tfrec_dir", str(tfrec_dir),
            "--dataset", filter_name,
            "--results", str(checkpoint_dir),
            "--exec", str(repetition),
            "--img_sizes", "299",
            "--num_classes", "2",
            "--verbose", "1",
            "--seed", str(seed)
        ]
        cmd.append("--use_class_weights" if use_class_weights else "--no-use_class_weights")
        if phase1_epochs is not None:
            cmd.extend(["--phase1_epochs", str(phase1_epochs)])
        if phase2_epochs is not None:
            cmd.extend(["--phase2_epochs", str(phase2_epochs)])


        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Set environment variables
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        
        # Add seed for reproducibility (TensorFlow)
        env["TF_DETERMINISTIC_OPS"] = "1"
        env["PYTHONHASHSEED"] = str(seed)
        
        # Record start time
        start_time = time.time()
        
        # Run training script
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,
            text=True,
            timeout=86400  # 24 hour timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            logger.error(f"Training script exited with code {result.returncode}")
            return False
        
        logger.info(f"✓ Training completed successfully in {elapsed_time:.1f}s")
        
        # Save training metadata
        metadata = {
            "filter_name": filter_name,
            "repetition": repetition,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "git_commit": get_git_commit_hash(logger),
            "elapsed_time_seconds": elapsed_time,
            "training_script": str(training_script),
        }
        
        metadata_path = checkpoint_dir / f"training_metadata_{filter_name}_{repetition}.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved training metadata to {metadata_path}")
        except Exception as e:
            logger.warning(f"Could not save training metadata: {e}")
        
        logger.info("=" * 80)
        logger.info(f"✓ Training completed: filter={filter_name}, repetition={repetition}")
        logger.info("=" * 80)
        
        return True
    
    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out after 24 hours")
        return False
    
    except Exception as e:
        logger.error(f"Training failed with exception: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Script 2: Train EfficientNetV2S model on filtered dataset"
    )
    parser.add_argument(
        "--filtro",
        type=str,
        required=True,
        help="Filter name/parameters (e.g., 'AHE40.0_CLAHE4.0')"
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
        "--use-class-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inverse-class-frequency loss weighting. Default ON as of the HPO run of "
             "2026-07-09 (was opt-in/OFF pre-audit). Pass --no-use-class-weights to restore "
             "the pre-audit behavior. See experiments/MODEL_OPTIMIZATION_AUDIT.md"
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=None,
        help="TEST-ONLY: override phase-1 epoch count (default: None -> not passed, "
             "dr_hcpa_v2_2024.py uses the paper-mandated value). Never use for the real matrix."
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=None,
        help="TEST-ONLY: override phase-2 epoch count (default: None -> not passed, "
             "dr_hcpa_v2_2024.py uses the paper-mandated value). Never use for the real matrix."
    )

    args = parser.parse_args()

    # Validate repetition. 0-9 is the main campaign's range (10 seeds per
    # condition); 10-19 is reserved for the XAI methodology mini-campaign
    # (experiments/filter_matrix_xai_mini.json) re-running a handful of
    # ALREADY-COMPLETED filters under multiple XAI methods -- using the
    # main campaign's own 0-9 would collide with its existing success
    # markers/checkpoints (same filter name + repetition = same on-disk
    # identity) and silently no-op instead of training. --repeticao is
    # only ever an RNG seed (random.seed/np.random.seed/tf.random.set_seed
    # in dr_hcpa_v2_2024.py take it directly, no lookup table keyed by
    # range), so widening this bound changes no science, only which
    # on-disk identity a run claims.
    if not 0 <= args.repeticao <= 19:
        print("ERROR: Repetition must be between 0 and 19")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(f"script2_treina_{args.filtro}_{args.repeticao}")
    
    # Get paths
    paths = get_paths(args.base_ssd, args.base_home)
    
    # Validate paths
    if not validate_paths(paths, logger):
        logger.error("Path validation failed!")
        sys.exit(1)
    
    # Run training
    success = run_training(args.filtro, args.repeticao, paths, logger,
                            use_class_weights=args.use_class_weights,
                            phase1_epochs=args.phase1_epochs,
                            phase2_epochs=args.phase2_epochs)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
