# XAI-Guided Preprocessing Pipeline — Implementation Guide

## Overview

This implementation provides a complete pipeline for the research project: **"XAI-Guided Preprocessing: Optimizing Image Contrast Filters for Diabetic Retinopathy using Anatomical Saliency Maps"**

The pipeline consists of 5 main scripts that orchestrate:
1. **Script 1** — Create filtered dataset variations
2. **Script 2** — Train InceptionV3 models
3. **Script 3** — Evaluate models and compute XAI metrics
4. **Script 4** — Clean up (delete) filtered datasets
5. **Script 5** — Orchestrate all scripts via SLURM jobs

## Directory Structure

```
$SSD_BASE/                        # /ssd/aadsilva/ic/hcpa
├── dataset_bruto/                # Original FGADR dataset (never modified)
│   ├── imagens/
│   └── mascaras/
├── split.json                    # Fixed train/val/test split (generated once)
├── datasets_filtrados/           # Filtered dataset variations (created by Script 1)
│   └── {filtro_name}/
│       ├── train/
│       ├── val/
│       └── test/
├── checkpoints/                  # Model checkpoints (from Script 2)
│   └── {filtro}_{repeticao}.keras
└── logs/                          # SLURM job logs

$HOME/                             # User's home directory
├── resultados/                    # Results CSV files (from Script 3)
│   ├── {filtro}_{repeticao}_{timestamp}.csv
│   └── _success_markers/          # Success markers (one per repetition)
│       └── {filtro}_{repeticao}.ok
├── logs_orquestracao/             # Orchestrator logs
├── auditoria_exclusoes.log        # Deletion audit log
└── trash_datasets/                # Soft-deleted datasets (with timestamp)
```

## Utility Modules

### `utils_preprocessing.py`
Contains image preprocessing filters:
- `AHEFilter` — Adaptive Histogram Equalization
- `CLAHEFilter` — Contrast Limited Adaptive Histogram Equalization
- `GammaFilter` — Gamma correction
- `MaxGreenFilter` — Green channel suppression
- `CompositeFilter` — Chain multiple filters
- `parse_filter_name()` — Parse filter name into parameters

Example usage:
```python
from utils_preprocessing import get_filter

params = {"clip_limit": 40.0, "tile_grid_size": (8, 8)}
filter = get_filter("AHE40.0", params)
filtered_image = filter.apply(image)
```

### `utils_metrics.py`
Contains XAI and clinical evaluation metrics:
- `dice_coefficient()` — Dice coefficient between masks
- `iou_score()` — Intersection over Union
- `pointing_game_accuracy()` — Peak activation overlap with GT mask
- `sensitivity()` — True Positive Rate (Recall)
- `specificity()` — True Negative Rate
- `compute_roc_auc()` — Area under ROC curve
- `compute_f1_score()` — F1 score
- `compute_all_metrics()` — Compute all metrics at once

Example usage:
```python
from utils_metrics import compute_all_metrics
import numpy as np

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.1, 0.8, 0.9, 0.2, 0.6])

metrics = compute_all_metrics(y_true, y_pred)
print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
```

### `utils_common.py`
Contains common utilities:
- `setup_logging()` — Configure logging for a script
- `get_paths()` — Get all standard paths
- `validate_paths()` — Check paths exist and are writable
- `load_split_json()` — Load train/val/test split
- `save_metadata()` — Save JSON metadata
- `verify_images_valid()` — Verify images can be opened
- `create_success_marker()` — Create marker file
- `check_success_marker()` — Check if marker exists

## Script 1 — Dataset Generation

**Purpose:** Create filtered dataset variations (applies filters only to images, never masks)

**Granularity:** One per filter (reused by 10 repetitions)

**Resources:** CPU-only (no GPU needed)

**Usage:**
```bash
python3 script1_cria_dataset.py \
    --filtro "AHE40.0_CLAHE4.0_Gamma0.8" \
    --base-ssd /ssd/aadsilva/ic/hcpa \
    --base-home $HOME
```

**Key Features:**
- ✓ Filters applied ONLY to images (not masks)
- ✓ Validates output integrity (image count, sample opens)
- ✓ Saves metadata (filter config, git commit, timestamp)
- ✓ Skips if dataset already exists (reuses for 10 repetitions)

**Output:**
- Directory: `datasets_filtrados/{filtro}/train|val|test/`
- Metadata: `datasets_filtrados/{filtro}/metadata.json`

## Script 2 — Training

**Purpose:** Train InceptionV3 model on filtered dataset

**Granularity:** One per filter + repetition (10 repetitions per filter)

**Resources:** 1 GPU L40S, max 24h

**Usage:**
```bash
python3 script2_treina.py \
    --filtro "AHE40.0_CLAHE4.0_Gamma0.8" \
    --repeticao 0 \
    --base-ssd /ssd/aadsilva/ic/hcpa \
    --base-home $HOME
```

**Key Features:**
- ✓ Wraps existing `dr_hcpa_v2_2024.py` training script
- ✓ Sets seeds for reproducibility (seed = repetition number)
- ✓ Saves training metadata with git commit
- ✓ Automatic GPU detection and management

**Output:**
- Checkpoint: `checkpoints/{filtro}_{repeticao}.keras`
- Metadata: `checkpoints/training_metadata_{filtro}_{repeticao}.json`

## Script 3 — Evaluation & Metrics

**Purpose:** Evaluate trained model and compute all XAI + clinical metrics

**Granularity:** One per filter + repetition

**Resources:** GPU (for inference)

**Metrics Computed:**
- Clinical: Accuracy, Sensitivity, Specificity, F1, AUC-ROC
- XAI: Dice, IoU, Pointing Game Accuracy (when saliency maps provided)

**Usage:**
```bash
python3 script3_avalia.py \
    --filtro "AHE40.0_CLAHE4.0_Gamma0.8" \
    --repeticao 0 \
    --base-ssd /ssd/aadsilva/ic/hcpa \
    --base-home $HOME
```

**Key Features:**
- ✓ Validates CSV output (no NaN, required fields present)
- ✓ Creates success marker ONLY after validation passes
- ✓ Skips if marker already exists
- ✓ Placeholder for actual model inference (integrate your inference code)

**Output:**
- Results CSV: `$HOME/resultados/{filtro}_{repeticao}_{timestamp}.csv`
- Success Marker: `$HOME/resultados/_success_markers/{filtro}_{repeticao}.ok`

## Script 4 — Dataset Deletion

**Purpose:** Clean up (soft or hard delete) filtered datasets after evaluation

**Granularity:** One per filter (after all 10 repetitions complete)

**Pre-condition:** All 10 success markers must exist and be valid

**Resources:** CPU-only

**Usage:**
```bash
# Soft-delete (move to trash with timestamp)
python3 script4_deleta_dataset.py \
    --filtro "AHE40.0_CLAHE4.0_Gamma0.8" \
    --base-ssd /ssd/aadsilva/ic/hcpa \
    --base-home $HOME

# Dry-run (test without deleting)
python3 script4_deleta_dataset.py \
    --filtro "AHE40.0_CLAHE4.0_Gamma0.8" \
    --dry-run

# Hard-delete (permanent, careful!)
python3 script4_deleta_dataset.py \
    --filtro "AHE40.0_CLAHE4.0_Gamma0.8" \
    --hard-delete
```

**Key Features:**
- ✓ Verifies all 10 success markers exist (does NOT trust SLURM --dependency alone)
- ✓ Dry-run mode for testing
- ✓ Soft-delete to trash by default (timestamp-tagged)
- ✓ Hard-delete option for permanent deletion
- ✓ Audit logging of all deletions
- ✓ Safety checks (never deletes dataset_bruto)

**Output:**
- Soft-delete: `$HOME/trash_datasets/{filtro}_{timestamp}/`
- Audit Log: `$HOME/auditoria_exclusoes.log` (JSON lines)

## Script 5 — Orchestrator

**Purpose:** Generate and submit SLURM jobs with proper dependencies

**Execution Modes:**

### Mode 1: SBATCH (Production)
Submit all jobs to queue with SLURM dependencies. Best for running the full filter matrix.

```bash
python3 script5_orquestrador.py \
    --mode sbatch \
    --base-ssd /ssd/aadsilva/ic/hcpa \
    --base-home $HOME \
    --partition grace \
    --cpu-partition cpu
```

**Job Dependency Chain per Filter:**
```
Script 1 (no GPU, 2h)
    ↓ --dependency=afterok:ID1
10× Script 2+3 (1 GPU each, up to 10h)
    ↓ --dependency=afterok:ID2:ID3:...:ID11
Script 4 (no GPU, 30min)
```

### Mode 2: SALLOC (Pilot/Debug)
Run filters sequentially within an interactive GPU allocation. Best for piloting and debugging.

```bash
# Allocate 1 GPU interactively
salloc --partition grace --gres=gpu:1 --time=24:00:00

# Inside allocation, run orchestrator
python3 script5_orquestrador.py \
    --mode salloc \
    --base-ssd /ssd/aadsilva/ic/hcpa \
    --base-home $HOME
```

## SLURM Job Templates

Ready-to-use SLURM scripts provided in:
- `job_template_script1.sh` — Dataset creation
- `job_template_script2_3.sh` — Training + Evaluation
- `job_template_script4.sh` — Dataset deletion

### Example: Submit jobs manually

```bash
# 1. Submit Script 1 (dataset creation)
JOB1=$(sbatch --export=FILTRO="AHE40.0_CLAHE4.0" job_template_script1.sh | awk '{print $NF}')
echo "Job 1 ID: $JOB1"

# 2. Submit 10 training jobs with dependency on Job 1
for REP in {0..9}; do
    sbatch --export=FILTRO="AHE40.0_CLAHE4.0",REPETICAO=$REP \
           --dependency=afterok:$JOB1 \
           job_template_script2_3.sh
done

# 3. Submit Script 4 (deletion) after all training jobs complete
# (This is handled automatically by Script 5's sbatch mode)
```

## Filter Name Convention

Filter names follow the pattern: `FILTER1_FILTER2_FILTER3_...`

Examples:
- `baseline` — No filter
- `AHE40.0` — AHE with clip_limit=40
- `CLAHE4.0` — CLAHE with clip_limit=4
- `Gamma0.8` — Gamma correction with gamma=0.8
- `MaxGreen2.0` — Green channel suppression with factor=2.0
- `AHE40.0_CLAHE4.0_Gamma0.8_MaxGreen2.0` — Composite

## Getting Started

### Step 1: Prepare Dataset

Ensure your FGADR dataset is in place:
```
/ssd/aadsilva/ic/hcpa/dataset_bruto/
├── imagens/          # Original retinal images (.jpg)
└── mascaras/         # Binary masks (.png)
```

### Step 2: Create Split File

Generate a `split.json` file once and freeze it:
```python
import json
from pathlib import Path

split = {
    "train": ["image1.jpg", "image2.jpg", ...],   # 70% of images
    "val": ["image100.jpg", "image101.jpg", ...],    # 15% of images
    "test": ["image200.jpg", "image201.jpg", ...]    # 15% of images
}

with open("/ssd/aadsilva/ic/hcpa/split.json", 'w') as f:
    json.dump(split, f, indent=2)
```

### Step 3: Define Filter Matrix

Create a filter matrix JSON file (optional, uses defaults otherwise):
```json
[
    {
        "name": "baseline",
        "params": {},
        "description": "Baseline (no filter)"
    },
    {
        "name": "AHE40.0_CLAHE4.0_Gamma0.8",
        "params": {
            "ahe_clip_limit": 40.0,
            "clahe_clip_limit": 4.0,
            "gamma": 0.8
        },
        "description": "Composite: AHE + CLAHE + Gamma"
    }
]
```

### Step 4: Run Pilot

Before submitting the full matrix, run a single filter end-to-end to calibrate times:

```bash
# Option A: Use salloc mode (interactive)
salloc --partition grace --gres=gpu:1 --time=20:00:00
cd /home/aadsilva/ic/hcpa-retinopathy/hcpa-repository/hcpa
python3 script5_orquestrador.py --mode salloc

# Option B: Use sbatch mode with default matrix (first filter only)
python3 script5_orquestrador.py --mode sbatch
```

### Step 5: Monitor Jobs

```bash
# Check queue
squeue -u aadsilva

# Check specific job logs
tail -f logs/script1_*.out
tail -f logs/script2_3_*.out

# Check results
ls ~/resultados/*.csv
ls ~/resultados/_success_markers/*.ok
```

### Step 6: Run Full Matrix

Once pilot is complete and times calibrated:

```bash
python3 script5_orquestrador.py \
    --mode sbatch \
    --skip-completed
```

## Troubleshooting

### Script 1: Dataset creation fails
- Check `dataset_bruto/` exists and has `imagens/` and `mascaras/` subdirectories
- Verify `split.json` is valid and lists actual image filenames
- Check disk space on SSD: `df -h /ssd/`

### Script 2: Training fails
- Ensure GPU is available: `nvidia-smi`
- Check that filtered dataset was created successfully (Script 1 completed)
- Verify `dr_hcpa_v2_2024.py` still exists and has correct CLI interface
- Check memory: `nvidia-smi` should show <80% usage before job starts

### Script 3: Evaluation fails
- Check that success marker wasn't created for previous failed attempt
- Verify checkpoint file exists: `ls checkpoints/{filtro}_{repeticao}.keras`
- Check for corrupted CSV (missing fields): `head $HOME/resultados/*.csv`

### Script 4: Deletion fails
- Verify all 10 success markers exist: `ls $HOME/resultados/_success_markers/{filtro}_*.ok`
- Use `--dry-run` to test first: `script4 --filtro name --dry-run`
- Check disk space: `df -h /ssd/`

### Jobs stuck in queue
- Check user job limit: `sacctmgr show user aadsilva` (look at MaxSubmitJobs)
- Monitor queue: `squeue -u aadsilva`
- Cancel problematic jobs: `scancel JOB_ID`

## Reproducibility Checklist

- [ ] Git commit hash recorded in metadata (from `get_git_commit_hash()`)
- [ ] Fixed seed per repetition (0-9, same across all filters)
- [ ] Frozen split.json (train/val/test fixed before any filtering)
- [ ] Baseline included in filter matrix (10 repetitions, no preprocessing)
- [ ] Metadata saved for every dataset, training, and evaluation
- [ ] All CSV results in `$HOME/resultados/` with timestamps
- [ ] Success markers created only after validation passes
- [ ] Audit log of deletions in `$HOME/auditoria_exclusoes.log`

## Performance Estimates

Based on pilot runs (calibrate with your actual hardware):

| Component | Time | Notes |
|-----------|------|-------|
| Script 1 (create dataset, CPU) | 1-2h | Depends on dataset size and filter complexity |
| Script 2 (train, GPU) | 2-4h | Depends on model complexity and hardware |
| Script 3 (evaluate, GPU) | 0.5-1h | Inference + metric computation |
| Script 4 (delete, CPU) | 5-30min | Depends on dataset size |
| **Total per filter** | **3-8h** | |
| **Total for N filters** | **N × 3-8h** | Can parallelize 10 repetitions per filter |

## Support & Issues

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `~/logs_orquestracao/` and SLURM logs
3. Verify paths and permissions
4. Consult the Notion research plan document

---

**Last Updated:** June 2026
**Author:** Research Team
**Status:** In Development
