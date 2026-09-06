#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 4 — Exclusão dos Artefatos Temporários do Filtro

Objetivo: Liberar espaço removendo os artefatos temporários (dataset filtrado
e TFRecords) de um filtro já avaliado — ambos são reproduzíveis
deterministicamente a partir de data/Seg-set + experiments/split.json + o
nome do filtro, então nenhum dos dois é necessário para manter o experimento
reproduzível. Nunca remove checkpoints/, ~/resultados/ (métricas) ou logs.

Pré-condição obrigatória: Existência dos 10 marcadores de sucesso (_success_markers)

Protocolo de segurança:
  1. Verificar que os 10 marcadores de sucesso existem e são válidos
  2. Excluir APENAS datasets_filtrados/{filtro}/ e data/tfrecords_filtrados/{filtro}/
     — nunca data/Seg-set (dataset bruto), checkpoints/ ou ~/resultados/
  3. Implementar --dry-run para testes
  4. Implementar soft-delete (mover para trash com timestamp)
  5. Logar em auditoria_exclusoes.log
  6. Adquirir o lock por filtro (utils_common.filter_lock) para toda a operação,
     evitando corrida com script1/create-tfrecord.py para o mesmo filtro

Granularidade: Uma vez por filtro, após todas as 10 repetições completarem.

Dependências: Via SLURM --dependency=afterok de todos os 10 jobs de repetição.
"""

import os
import sys
import argparse
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime

# Add hcpa to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_common import (
    setup_logging, get_paths, validate_paths, check_success_marker, filter_lock
)

# Directories eligible for cleanup once a filter's 10 repetitions all
# succeed. Both are reproducible byproducts (deterministically regenerable
# from data/Seg-set + experiments/split.json + the filter name), so neither
# is required to keep the experiment reproducible — only checkpoints,
# metrics (~/resultados), and logs are kept permanently.
CLEANUP_TARGETS = [
    ("datasets_filtrados", "filtered_images"),
    ("tfrecords_filtrados", "tfrecords"),
]

logger = logging.getLogger(__name__)


def verify_success_markers(filter_name: str, paths: dict, logger) -> bool:
    """
    Verify that all 10 success markers exist and are valid.
    
    Args:
        filter_name: Name of the filter
        paths: Dictionary of paths from get_paths()
        logger: Logger instance
    
    Returns:
        True if all 10 markers exist and are valid
    """
    logger.info(f"Verifying success markers for filter: {filter_name}")
    
    all_valid = True
    for repetition in range(10):
        marker_path = paths["success_markers"] / f"{filter_name}_{repetition}.ok"
        
        if check_success_marker(marker_path, logger):
            logger.info(f"  ✓ Marker {repetition}: valid")
        else:
            logger.error(f"  ✗ Marker {repetition}: missing or invalid at {marker_path}")
            all_valid = False
    
    if all_valid:
        logger.info(f"✓ All 10 success markers verified for {filter_name}")
    else:
        logger.error(f"✗ Some success markers are missing or invalid for {filter_name}")
    
    return all_valid


def get_directory_size(path: Path) -> int:
    """
    Calculate total size of directory in bytes.
    
    Args:
        path: Path to directory
    
    Returns:
        Total size in bytes
    """
    total_size = 0
    for filepath in path.rglob("*"):
        if filepath.is_file():
            total_size += filepath.stat().st_size
    return total_size


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def soft_delete_dataset(target_dir: Path, artifact_label: str, filter_name: str, paths: dict, logger) -> bool:
    """
    Soft delete (move to trash with timestamp) an artifact directory for a filter.

    Args:
        target_dir: Directory to remove (e.g. datasets_filtrados/{filtro})
        artifact_label: Short tag for logging/audit (e.g. "filtered_images")
        filter_name: Name of the filter
        paths: Dictionary of paths from get_paths()
        logger: Logger instance

    Returns:
        True if successful
    """
    try:
        if not target_dir.exists():
            logger.warning(f"[{artifact_label}] Directory does not exist: {target_dir}")
            return True

        # Calculate size before deletion
        size_bytes = get_directory_size(target_dir)
        size_str = format_bytes(size_bytes)

        logger.info(f"[{artifact_label}] Soft-deleting: {target_dir}")
        logger.info(f"  Size: {size_str}")

        # Create trash directory
        os.makedirs(paths["trash"], exist_ok=True)

        # Move to trash with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_path = paths["trash"] / f"{artifact_label}_{filter_name}_{timestamp}"

        logger.info(f"Moving to: {trash_path}")
        shutil.move(str(target_dir), str(trash_path))

        logger.info(f"✓ [{artifact_label}] Soft-deleted to trash: {trash_path}")

        # Log to auditoria
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "soft_delete",
            "artifact": artifact_label,
            "filter_name": filter_name,
            "original_path": str(target_dir),
            "trash_path": str(trash_path),
            "size_bytes": size_bytes,
            "size_human": size_str,
        }

        audit_log_path = paths["auditoria"]
        os.makedirs(audit_log_path.parent, exist_ok=True)
        with open(audit_log_path, 'a') as f:
            f.write(json.dumps(audit_entry) + "\n")

        logger.info(f"Logged to auditoria: {audit_log_path}")

        return True

    except Exception as e:
        logger.error(f"[{artifact_label}] Soft-delete failed: {e}")
        return False


def hard_delete_dataset(target_dir: Path, artifact_label: str, allowed_parent_name: str,
                        filter_name: str, paths: dict, logger) -> bool:
    """
    Permanently delete an artifact directory for a filter.

    Args:
        target_dir: Directory to remove (e.g. datasets_filtrados/{filtro})
        artifact_label: Short tag for logging/audit (e.g. "filtered_images")
        allowed_parent_name: Required parent directory name (safety check —
            this function refuses to delete anything whose immediate parent
            isn't this exact name, e.g. "datasets_filtrados")
        filter_name: Name of the filter
        paths: Dictionary of paths from get_paths()
        logger: Logger instance

    Returns:
        True if successful
    """
    try:
        if not target_dir.exists():
            logger.warning(f"[{artifact_label}] Directory does not exist: {target_dir}")
            return True

        # Calculate size before deletion
        size_bytes = get_directory_size(target_dir)
        size_str = format_bytes(size_bytes)

        logger.warning(f"[{artifact_label}] HARD-DELETING: {target_dir}")
        logger.warning(f"  Size: {size_str}")

        # Verify that parent matches the expected directory (the only place
        # this function is ever allowed to delete from)
        if target_dir.parent.name != allowed_parent_name:
            logger.error(f"ERROR: Unexpected parent directory: {target_dir.parent} (expected {allowed_parent_name})")
            return False

        # Delete
        shutil.rmtree(str(target_dir))

        logger.info(f"✓ [{artifact_label}] Hard-deleted: {target_dir}")

        # Log to auditoria
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "hard_delete",
            "artifact": artifact_label,
            "filter_name": filter_name,
            "path": str(target_dir),
            "size_bytes": size_bytes,
            "size_human": size_str,
        }

        audit_log_path = paths["auditoria"]
        os.makedirs(audit_log_path.parent, exist_ok=True)
        with open(audit_log_path, 'a') as f:
            f.write(json.dumps(audit_entry) + "\n")

        logger.info(f"Logged to auditoria: {audit_log_path}")

        return True

    except Exception as e:
        logger.error(f"[{artifact_label}] Hard-delete failed: {e}")
        return False


def delete_dataset(filter_name: str, paths: dict, logger,
                  dry_run: bool = False, soft_delete: bool = True) -> bool:
    """
    Main function to delete a filter's temporary artifacts (filtered images
    AND TFRecords — both are deterministically regenerable from
    data/Seg-set + experiments/split.json + the filter name, so neither is
    required to keep the experiment reproducible). Checkpoints, ~/resultados
    metrics, and logs are never touched by this function.

    Acquires the per-filter lock for the whole operation, so this can never
    race with a concurrent script1/create-tfrecord.py invocation for the
    same filter (e.g. a stray manual rerun).

    Args:
        filter_name: Name of the filter
        paths: Dictionary of paths from get_paths()
        logger: Logger instance
        dry_run: If True, only simulate deletion without actually deleting
        soft_delete: If True, move to trash; if False, permanently delete

    Returns:
        True if successful
    """
    logger.info("=" * 80)
    logger.info(f"Starting cleanup for filter: {filter_name}")
    if dry_run:
        logger.warning("DRY-RUN MODE: No actual deletion will occur")
    if soft_delete:
        logger.info("SOFT-DELETE MODE: Moving to trash, not permanent deletion")
    else:
        logger.warning("HARD-DELETE MODE: Permanent deletion!")
    logger.info("=" * 80)

    try:
        # Verify success markers
        if not verify_success_markers(filter_name, paths, logger):
            logger.error("Success marker verification failed - aborting deletion!")
            return False

        with filter_lock(paths, filter_name, logger):
            overall_success = True
            for paths_key, artifact_label in CLEANUP_TARGETS:
                target_dir = paths[paths_key] / filter_name

                if dry_run:
                    logger.info(f"DRY-RUN: Would delete [{artifact_label}] {target_dir}")
                    if target_dir.exists():
                        size_bytes = get_directory_size(target_dir)
                        logger.info(f"  Directory size: {format_bytes(size_bytes)}")
                    continue

                if soft_delete:
                    success = soft_delete_dataset(target_dir, artifact_label, filter_name, paths, logger)
                else:
                    success = hard_delete_dataset(target_dir, artifact_label, paths_key, filter_name, paths, logger)

                overall_success = overall_success and success

        if dry_run:
            return True

        if overall_success:
            logger.info("=" * 80)
            logger.info(f"✓ Cleanup completed successfully for {filter_name}")
            logger.info("=" * 80)
        else:
            logger.error("✗ Cleanup failed for one or more artifacts")

        return overall_success

    except Exception as e:
        logger.error(f"Deletion operation failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Script 4: Delete filtered dataset after all 10 repetitions complete"
    )
    parser.add_argument(
        "--filtro",
        type=str,
        required=True,
        help="Filter name/parameters"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate deletion without actually deleting"
    )
    parser.add_argument(
        "--hard-delete",
        action="store_true",
        help="Permanently delete (default: soft-delete to trash)"
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
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(f"script4_deleta_dataset_{args.filtro}")
    
    # Get paths
    paths = get_paths(args.base_ssd, args.base_home)
    
    # Validate paths
    if not validate_paths(paths, logger):
        logger.error("Path validation failed!")
        sys.exit(1)
    
    # Delete dataset
    success = delete_dataset(
        args.filtro, paths, logger,
        dry_run=args.dry_run,
        soft_delete=not args.hard_delete
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
