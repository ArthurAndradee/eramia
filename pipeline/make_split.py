#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_split.py — Gera o split oficial (fixo) train/test do dataset FGADR.

Regras do projeto:
- Split único, 70% treino / 30% teste, reutilizado por TODOS os filtros.
- Seed fixa (documentada em split_metadata.json).
- Nunca deve ser regenerado durante os experimentos: por padrão este
  script recusa sobrescrever um split.json existente (use --force).

Uso:
    python3 make_split.py [--base-ssd PATH] [--force] [--seed 42] [--test-size 0.3]
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_common import get_paths, get_git_commit_hash, setup_logging, binarize_icdr_grade

SEED = 42
TEST_SIZE = 0.3


def load_grades(labels_csv: Path):
    filenames, grades = [], []
    with open(labels_csv, 'r') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            filenames.append(row[0])
            grades.append(int(float(row[1])))
    return filenames, grades


def main():
    parser = argparse.ArgumentParser(description="Generate the fixed 70/30 train/test split for FGADR")
    parser.add_argument("--base-ssd", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--force", action="store_true", help="Overwrite existing split.json")
    args = parser.parse_args()

    logger = setup_logging("make_split")

    paths = get_paths(args.base_ssd)
    split_file = paths["split_file"]

    if split_file.exists() and not args.force:
        logger.error(
            f"split.json already exists at {split_file}. "
            f"Refusing to overwrite (the official split must never be regenerated "
            f"once experiments have started). Pass --force to override deliberately."
        )
        sys.exit(1)

    if not paths["labels_csv"].exists():
        logger.error(f"Labels CSV not found: {paths['labels_csv']}")
        sys.exit(1)

    filenames, grades = load_grades(paths["labels_csv"])
    logger.info(f"Loaded {len(filenames)} images with grades. Distribution: {sorted(Counter(grades).items())}")

    from sklearn.model_selection import train_test_split

    train_files, test_files, train_grades, test_grades = train_test_split(
        filenames, grades,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=grades,
    )

    split = {"train": sorted(train_files), "test": sorted(test_files)}

    # Backup existing split.json (and its stale derivatives) before overwriting
    if split_file.exists():
        os.makedirs(paths["trash"], exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = paths["trash"] / f"split.json.old_3way_{timestamp}"
        split_file.rename(backup_path)
        logger.info(f"Backed up previous split.json to {backup_path}")

    with open(split_file, 'w') as f:
        json.dump(split, f, indent=2)
    logger.info(f"Wrote new split.json: train={len(split['train'])}, test={len(split['test'])}")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(logger),
        "method": "sklearn.model_selection.train_test_split (stratified by ICDR grade)",
        "seed": args.seed,
        "test_size": args.test_size,
        "total_images": len(filenames),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "grade_distribution": {
            "train": sorted(Counter(train_grades).items()),
            "test": sorted(Counter(test_grades).items()),
        },
        "binary_label_distribution": {
            # Stratifying by the finer 5-class ICDR grade (above) is a strict
            # superset guarantee of balancing the derived binary label: if
            # every grade-stratum is split proportionally, their union (the
            # binary groups) is too. Reported explicitly here for transparency.
            "train": sorted(Counter(binarize_icdr_grade(g) for g in train_grades).items()),
            "test": sorted(Counter(binarize_icdr_grade(g) for g in test_grades).items()),
        },
        "note": (
            "Replaces a previous 3-way (train/val/test, ~70/15/15) split that "
            "did not match the project's 70/30 train/test requirement or "
            "dr_hcpa_v2_2024.py's train*/test* TFRecord convention."
        ),
    }
    with open(paths["split_metadata_file"], 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote split_metadata.json to {paths['split_metadata_file']}")


if __name__ == "__main__":
    main()
