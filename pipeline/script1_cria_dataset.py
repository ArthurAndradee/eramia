#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1 — Geração da Variação do Dataset (por filtro)

Objetivo: Criar uma cópia do dataset com um filtro de pré-processamento aplicado.

Entrada: data/Seg-set/Original_Images (FGADR original), split.json (split fixo 70/30), nome/parâmetros do filtro

Saída: datasets_filtrados/{filtro}/{train,test}/imagens (máscaras NUNCA são copiadas ou filtradas;
       permanecem apenas em data/Seg-set/*_Masks para uso exclusivo do script3 na avaliação XAI)

Granularidade: Uma vez por filtro, reutilizado pelas 10 repetições.

Execução: Sem GPU necessária (CPU-only partition recomendado).
"""

import os
import sys
import argparse
import shutil
import cv2
from pathlib import Path
from datetime import datetime

# Add hcpa to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_common import (
    setup_logging, get_paths, validate_paths, load_split_json,
    save_metadata, get_git_commit_hash, verify_images_valid,
    count_files_in_directory, check_split_fingerprint, filter_lock
)
from utils_preprocessing import get_filter, parse_filter_name


def create_dataset_variation(filter_name: str, paths: dict, logger, force: bool = False,
                              split_file=None) -> bool:
    """
    Acquire the per-filter lock, then generate the filtered dataset variation.
    The lock serializes any concurrent script1 invocation for the SAME filter
    (e.g. an accidental double-submission), preventing two processes from
    racing on the exists-check / write / metadata-save sequence below.

    split_file: optional override of paths["split_file"] — used ONLY by the
    hyperparameter-optimization stage (pipeline/hpo_search.py), which needs
    an internal train/val carve-out of the TRAINING images alone (never the
    real test split) so HPO decisions cannot leak from the test set. The
    576-filter campaign never passes this (always None -> paths["split_file"]).
    """
    with filter_lock(paths, filter_name, logger):
        return _create_dataset_variation_locked(filter_name, paths, logger, force, split_file)


def _create_dataset_variation_locked(filter_name: str, paths: dict, logger, force: bool = False,
                                      split_file=None) -> bool:
    """
    Main function to create a filtered dataset variation. Must only be
    called while holding the filter_lock() for `filter_name` (see caller).

    Args:
        filter_name: Name/parameters of the filter (e.g., 'AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0')
        paths: Dictionary of paths from get_paths()
        logger: Logger instance
        split_file: optional override of paths["split_file"] (see create_dataset_variation docstring)

    Returns:
        True if successful
    """
    split_file = split_file or paths["split_file"]
    logger.info("=" * 80)
    logger.info(f"Starting dataset generation for filter: {filter_name}")
    logger.info("=" * 80)
    
    try:
        # Parse filter name and parameters
        _, filter_params = parse_filter_name(filter_name)
        logger.info(f"Parsed filter parameters: {filter_params}")
        
        # Create output directories
        filtered_dataset_dir = paths["datasets_filtrados"] / filter_name
        logger.info(f"Output dataset directory: {filtered_dataset_dir}")

        # Load split (single source of truth, 70/30 train/test — or the
        # internal HPO train/val carve-out when split_file is overridden)
        split = load_split_json(split_file, logger)

        if filtered_dataset_dir.exists():
            logger.warning(f"Dataset directory already exists: {filtered_dataset_dir}")
            if check_split_fingerprint(filtered_dataset_dir, split, logger):
                logger.info("Split fingerprint matches current split.json — reusing existing dataset")
                return verify_dataset_integrity(filtered_dataset_dir, paths, logger, split_file)
            elif force:
                logger.warning("--force passed: regenerating dataset despite stale/missing fingerprint")
                shutil.rmtree(filtered_dataset_dir)
            else:
                logger.error(
                    f"{filtered_dataset_dir} exists but was generated from a different split "
                    f"(or has no recorded fingerprint). Refusing to silently reuse it. "
                    f"Re-run with --force to regenerate, or remove the directory manually."
                )
                return False

        os.makedirs(filtered_dataset_dir, exist_ok=True)

        # Create train/test subdirectories
        for split_name in ["train", "test"]:
            split_dir = filtered_dataset_dir / split_name
            os.makedirs(split_dir, exist_ok=True)
            logger.info(f"Created directory: {split_dir}")
        
        # Load and instantiate filter
        logger.info(f"Loading filter: {filter_name}")
        image_filter = get_filter(filter_name, filter_params)
        logger.info(f"Filter loaded: {image_filter}")
        
        # Process each split
        total_processed = 0
        for split_name, image_list in split.items():
            logger.info(f"\nProcessing {split_name} split ({len(image_list)} images)...")
            
            split_dir = filtered_dataset_dir / split_name
            
            for idx, image_filename in enumerate(image_list):
                # Load original image
                orig_img_path = paths["original_images"] / image_filename
                
                if not orig_img_path.exists():
                    logger.warning(f"Image not found: {orig_img_path}")
                    continue
                
                try:
                    # Load image
                    image = cv2.imread(str(orig_img_path))
                    if image is None:
                        logger.warning(f"Cannot read image: {orig_img_path}")
                        continue
                    
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Apply filter
                    filtered_image = image_filter.apply(image)
                    
                    # Convert back to BGR for saving
                    filtered_image_bgr = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR)
                    
                    # Save filtered image
                    output_path = split_dir / image_filename
                    os.makedirs(output_path.parent, exist_ok=True)
                    cv2.imwrite(str(output_path), filtered_image_bgr)
                    
                    if (idx + 1) % 100 == 0:
                        logger.info(f"  Processed {idx + 1}/{len(image_list)} images")
                    
                    total_processed += 1
                
                except Exception as e:
                    logger.error(f"Error processing image {image_filename}: {e}")
                    continue
            
            logger.info(f"Completed {split_name} split: {total_processed} images processed")
        
        logger.info(f"\n✓ Total images filtered and copied: {total_processed}")
        
        # Verify integrity
        if not verify_dataset_integrity(filtered_dataset_dir, paths, logger, split_file):
            logger.error("Dataset integrity verification failed!")
            return False
        
        # Save metadata
        metadata = {
            "filter_name": filter_name,
            "filter_config": image_filter.get_config(),
            "timestamp": datetime.now().isoformat(),
            "git_commit": get_git_commit_hash(logger),
            "original_images_location": str(paths["original_images"]),
            "split_file": str(paths["split_file"]),
            "total_images_processed": total_processed,
            "split_sizes": {k: len(v) for k, v in split.items()},
        }
        
        metadata_path = filtered_dataset_dir / "metadata.json"
        save_metadata(metadata, metadata_path, logger)
        
        logger.info("=" * 80)
        logger.info(f"✓ Successfully created filtered dataset: {filter_name}")
        logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to create dataset variation: {e}", exc_info=True)
        return False


def verify_dataset_integrity(filtered_dataset_dir: Path, paths: dict, logger, split_file=None) -> bool:
    """
    Verify integrity of filtered dataset.

    Checks:
    - All splits (train, test) exist
    - Image counts match expected split sizes
    - Sample images can be opened

    split_file: optional override of paths["split_file"] (see
    create_dataset_variation docstring — used only by hpo_search.py).

    Returns:
        True if all checks pass
    """
    logger.info("\nVerifying dataset integrity...")
    split_file = split_file or paths["split_file"]

    try:
        split = load_split_json(split_file, logger)

        for split_name in ["train", "test"]:
            split_dir = filtered_dataset_dir / split_name
            
            if not split_dir.exists():
                logger.error(f"Split directory does not exist: {split_dir}")
                return False
            
            # Count images
            image_count = count_files_in_directory(split_dir, ['.jpg', '.png'])
            expected_count = len(split.get(split_name, []))
            
            if image_count != expected_count:
                logger.error(
                    f"Image count mismatch in {split_name}: "
                    f"expected {expected_count}, got {image_count}"
                )
                return False
            
            logger.info(f"✓ {split_name} split: {image_count} images verified")
            
            # Verify sample images
            if not verify_images_valid(split_dir, logger, sample_size=min(5, image_count)):
                logger.error(f"Image verification failed for {split_name} split")
                return False
        
        logger.info("✓ Dataset integrity verification passed!")
        return True
    
    except Exception as e:
        logger.error(f"Integrity verification failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Script 1: Generate filtered dataset variation for a given filter"
    )
    parser.add_argument(
        "--filtro",
        type=str,
        required=True,
        help="Filter name/parameters (e.g., 'AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0')"
    )
    parser.add_argument(
        "--base-ssd",
        type=str,
        default=None,
        help="Base SSD path (defaults to $SSD_BASE env var, or the repo directory)"
    )
    parser.add_argument(
        "--base-home",
        type=str,
        default=None,
        help="Base home path (defaults to $HOME)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the filtered dataset even if a stale/mismatched version already exists"
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Override paths['split_file'] — used ONLY by hpo_search.py for the internal "
             "HPO train/val carve-out (never the real test split). Default: None -> "
             "paths['split_file'] (the official 70/30 split.json)."
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging("script1_cria_dataset")

    # Get paths
    paths = get_paths(args.base_ssd, args.base_home)

    # Validate paths
    if not validate_paths(paths, logger):
        logger.error("Path validation failed!")
        sys.exit(1)

    # Create dataset variation
    split_file_path = Path(args.split_file) if args.split_file else None
    success = create_dataset_variation(args.filtro, paths, logger, force=args.force,
                                        split_file=split_file_path)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
