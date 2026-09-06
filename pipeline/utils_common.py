# -*- coding: utf-8 -*-
"""
Common utilities for the XAI preprocessing pipeline.
Provides path management, logging, file operations, and metadata handling.
"""

import os
import csv
import json
import logging
import hashlib
import sys
import fcntl
import contextlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import subprocess

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent  # this file lives in <repo>/pipeline/
DEFAULT_BINARIZE_THRESHOLD = 2  # ICDR grade >= 2 => referable DR (1)


def binarize_icdr_grade(grade: int, threshold: int = DEFAULT_BINARIZE_THRESHOLD) -> int:
    """
    Single source of truth for the project's binarization rule:
    ICDR grades 0, 1 -> 0 (non-referable DR)
    ICDR grades 2, 3, 4 -> 1 (referable DR)
    """
    return 1 if grade >= threshold else 0

MASK_DIR_NAMES = {
    "HardExudate": "HardExudate_Masks",
    "Hemohedge": "Hemohedge_Masks",
    "IRMA": "IRMA_Masks",
    "Microaneurysms": "Microaneurysms_Masks",
    "Neovascularization": "Neovascularization_Masks",
    "SoftExudate": "SoftExudate_Masks",
}


def setup_logging(script_name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Setup logging for a script.
    
    Args:
        script_name: Name of the script for log file
        log_dir: Directory to save logs (uses $HOME/logs_orquestracao if None)
    
    Returns:
        Configured logger instance
    """
    if log_dir is None:
        log_dir = os.path.expanduser("~/logs_orquestracao")
    
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_paths(base_ssd: Optional[str] = None, base_home: Optional[str] = None) -> Dict[str, Path]:
    """
    Get all standard paths for the pipeline.

    Args:
        base_ssd: Base SSD path (defaults to $SSD_BASE env var, or the repo
            directory when unset — e.g. for local/dev runs without SLURM)
        base_home: Base home path (defaults to $HOME)

    Returns:
        Dictionary of Path objects
    """
    if base_ssd is None:
        base_ssd = os.environ.get("SSD_BASE", str(REPO_ROOT))
    if base_home is None:
        base_home = os.path.expanduser("~")

    base_ssd = Path(base_ssd)
    base_home = Path(base_home)

    seg_set = base_ssd / "data" / "Seg-set"

    return {
        "base_ssd": base_ssd,
        "base_home": base_home,
        "seg_set": seg_set,
        "original_images": seg_set / "Original_Images",
        "mask_dirs": {name: seg_set / dirname for name, dirname in MASK_DIR_NAMES.items()},
        "labels_csv": seg_set / "DR_Seg_Grading_Label.csv",
        "datasets_filtrados": base_ssd / "datasets_filtrados",
        "tfrecords_filtrados": base_ssd / "data" / "tfrecords_filtrados",
        "checkpoints": base_ssd / "checkpoints",
        "logs_ssd": base_ssd / "logs",
        "experiments_dir": base_ssd / "experiments",
        "split_file": base_ssd / "experiments" / "split.json",
        "split_metadata_file": base_ssd / "experiments" / "split_metadata.json",
        "filter_matrix_file": base_ssd / "experiments" / "filter_matrix_template.json",
        "resultados": base_home / "resultados",
        "success_markers": base_home / "resultados" / "_success_markers",
        "logs_orquestracao": base_home / "logs_orquestracao",
        "auditoria": base_home / "auditoria_exclusoes.log",
        "trash": base_home / "trash_datasets",
        "locks_dir": base_ssd / "locks",
    }


@contextlib.contextmanager
def filter_lock(paths: Dict[str, Path], filter_name: str, logger: Optional[logging.Logger] = None):
    """
    Advisory file lock (flock) scoped to a single filter name. Serializes any
    concurrent script1 (dataset creation) / create-tfrecord.py (TFRecord
    generation) / script4 (deletion) invocations for the SAME filter, so two
    processes can never race on creating/reading/deleting the same
    datasets_filtrados/{filtro} or tfrecords_filtrados/{filtro} directory.

    Blocks until the lock is free (does not fail if already held) — a
    concurrent process just waits its turn, then re-checks whatever
    idempotency condition applies (e.g. "does the dataset already exist with
    a matching split fingerprint?") and typically finds there is nothing
    left to do.
    """
    locks_dir = paths["locks_dir"]
    os.makedirs(locks_dir, exist_ok=True)
    lock_path = locks_dir / f"{filter_name}.lock"

    with open(lock_path, "w") as f:
        if logger:
            logger.info(f"Acquiring lock for filter '{filter_name}' ({lock_path})...")
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        if logger:
            logger.info(f"Lock acquired for filter '{filter_name}'")
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def validate_paths(paths: Dict[str, Path], logger: logging.Logger) -> bool:
    """
    Validate that required paths exist or can be created.

    Args:
        paths: Dictionary of paths from get_paths()
        logger: Logger instance

    Returns:
        True if all critical paths are valid
    """
    required_readonly = [paths["original_images"], paths["labels_csv"], paths["split_file"]]
    required_writable = [paths["base_ssd"], paths["base_home"]]

    for path in required_readonly:
        if not path.exists():
            logger.error(f"Required path does not exist: {path}")
            return False

    for path in required_writable:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            logger.error(f"Cannot write to path {path}: {e}")
            return False

    return True


def load_label_map(labels_csv: Path, threshold: int = DEFAULT_BINARIZE_THRESHOLD) -> Dict[str, int]:
    """
    Load the ICDR grading CSV (image filename, grade 0-4) and binarize it
    via binarize_icdr_grade() (grades 0,1 -> 0 non-referable; 2,3,4 -> 1 referable).

    Args:
        labels_csv: Path to DR_Seg_Grading_Label.csv
        threshold: Minimum grade (inclusive) considered referable DR

    Returns:
        Dict mapping image filename -> binary label (0/1)
    """
    label_map = {}
    with open(labels_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            filename, grade = row[0], row[1]
            try:
                grade_int = int(float(grade))
            except ValueError:
                continue
            label_map[filename] = binarize_icdr_grade(grade_int, threshold)
    return label_map


def load_gt_mask(image_filename: str, mask_dirs: Dict[str, Path]) -> Optional[np.ndarray]:
    """
    Build a combined ground-truth spatial mask for an image by taking the
    union of all lesion-type masks that exist for it (HardExudate,
    Hemohedge, IRMA, Microaneurysms, Neovascularization, SoftExudate).

    Args:
        image_filename: Filename of the original image (e.g. '0000_1.png')
        mask_dirs: Dict of lesion-type -> mask directory (from get_paths())

    Returns:
        Binary mask (H, W) as float32 in [0, 1], or None if no lesion mask
        exists for this image (e.g. grade 0 images with no annotations).
    """
    import cv2

    combined = None
    for mask_dir in mask_dirs.values():
        mask_path = mask_dir / image_filename
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        binary = (mask > 0).astype(np.float32)
        combined = binary if combined is None else np.maximum(combined, _resize_like(binary, combined))

    return combined


def _resize_like(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Resize mask to match reference's shape if they differ (nearest-neighbor)."""
    if mask.shape == reference.shape:
        return mask
    import cv2
    resized = cv2.resize(mask, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized


def compute_split_fingerprint(split: Dict[str, list]) -> Dict[str, int]:
    """Return a small fingerprint of a split (sizes per key) for staleness checks."""
    return {k: len(v) for k, v in split.items()}


def check_split_fingerprint(filtered_dataset_dir: Path, split: Dict[str, list], logger: logging.Logger) -> bool:
    """
    Compare the split fingerprint recorded in a filtered dataset's
    metadata.json against the current split.json. Returns True if they
    match (safe to reuse), False if stale or missing (must not reuse).
    """
    metadata_path = filtered_dataset_dir / "metadata.json"
    if not metadata_path.exists():
        logger.warning(f"No metadata.json found in {filtered_dataset_dir}; cannot verify freshness")
        return False

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read metadata.json in {filtered_dataset_dir}: {e}")
        return False

    recorded = metadata.get("split_sizes", {})
    current = compute_split_fingerprint(split)

    if recorded != current:
        logger.error(
            f"Split fingerprint mismatch for {filtered_dataset_dir}: "
            f"recorded={recorded}, current={current}"
        )
        return False

    return True


def load_split_json(split_file: Path, logger: logging.Logger) -> Dict[str, list]:
    """
    Load the fixed train/val/test split from JSON file.
    
    Args:
        split_file: Path to split.json
        logger: Logger instance
    
    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing list of image filenames
    """
    try:
        with open(split_file, 'r') as f:
            split = json.load(f)
        logger.info(f"Loaded split from {split_file}")
        logger.info(f"  Train: {len(split.get('train', []))} images")
        logger.info(f"  Val: {len(split.get('val', []))} images")
        logger.info(f"  Test: {len(split.get('test', []))} images")
        return split
    except Exception as e:
        logger.error(f"Failed to load split file {split_file}: {e}")
        raise


def save_metadata(metadata: Dict[str, Any], output_path: Path, logger: logging.Logger):
    """
    Save metadata JSON for a dataset or result.
    
    Args:
        metadata: Metadata dictionary
        output_path: Path to save metadata JSON
        logger: Logger instance
    """
    try:
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata to {output_path}: {e}")
        raise


def compute_directory_hash(directory: Path, logger: logging.Logger) -> str:
    """
    Compute hash of directory contents for integrity verification.
    
    Args:
        directory: Path to directory
        logger: Logger instance
    
    Returns:
        Hex string of MD5 hash
    """
    hash_md5 = hashlib.md5()
    
    try:
        for filepath in sorted(directory.rglob("*")):
            if filepath.is_file():
                with open(filepath, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
        
        result = hash_md5.hexdigest()
        logger.debug(f"Computed hash for {directory}: {result}")
        return result
    except Exception as e:
        logger.warning(f"Could not compute directory hash: {e}")
        return ""


def get_git_commit_hash(logger: logging.Logger) -> str:
    """
    Get current Git commit hash for reproducibility tracking.
    
    Args:
        logger: Logger instance
    
    Returns:
        Short commit hash, or empty string if not in a Git repo
    """
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        logger.debug(f"Git commit hash: {result}")
        return result
    except Exception:
        logger.debug("Not in a Git repository or Git not available")
        return ""


def create_success_marker(marker_path: Path, logger: logging.Logger, metadata: Dict[str, Any] = None):
    """
    Create a success marker file to indicate job completion.
    
    Args:
        marker_path: Path to marker file
        logger: Logger instance
        metadata: Optional metadata to include in marker
    """
    try:
        os.makedirs(marker_path.parent, exist_ok=True)
        
        marker_data = {
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        with open(marker_path, 'w') as f:
            json.dump(marker_data, f, indent=2)
        
        logger.info(f"Created success marker: {marker_path}")
    except Exception as e:
        logger.error(f"Failed to create success marker {marker_path}: {e}")
        raise


def check_success_marker(marker_path: Path, logger: logging.Logger) -> bool:
    """
    Check if a success marker exists and is valid.
    
    Args:
        marker_path: Path to marker file
        logger: Logger instance
    
    Returns:
        True if marker exists and is valid JSON
    """
    if not marker_path.exists():
        return False
    
    try:
        with open(marker_path, 'r') as f:
            json.load(f)
        return True
    except Exception as e:
        logger.warning(f"Invalid success marker {marker_path}: {e}")
        return False


def verify_images_valid(image_dir: Path, logger: logging.Logger, sample_size: int = 10) -> bool:
    """
    Verify that images in a directory can be opened (random sample).
    
    Args:
        image_dir: Path to directory with images
        logger: Logger instance
        sample_size: Number of random images to check
    
    Returns:
        True if all sampled images are valid
    """
    import cv2
    import random
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not image_files:
        logger.warning(f"No image files found in {image_dir}")
        return False
    
    # Sample random images
    sample = random.sample(image_files, min(sample_size, len(image_files)))
    
    for img_file in sample:
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                logger.error(f"Cannot read image: {img_file}")
                return False
        except Exception as e:
            logger.error(f"Error reading image {img_file}: {e}")
            return False
    
    logger.info(f"Verified {len(sample)} sample images in {image_dir}")
    return True


def count_files_in_directory(directory: Path, extensions: list = None) -> int:
    """
    Count files in directory with specific extensions.
    
    Args:
        directory: Path to directory
        extensions: List of extensions to count (e.g., ['.jpg', '.png'])
    
    Returns:
        Number of matching files
    """
    if extensions is None:
        extensions = ['.jpg', '.png']
    
    count = 0
    for ext in extensions:
        count += len(list(directory.glob(f"*{ext}")))
    
    return count
