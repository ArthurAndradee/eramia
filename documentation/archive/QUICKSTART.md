# XAI Preprocessing Pipeline — Quick Start Guide

## 5-Minute Setup

### 1. Navigate to Project Directory
```bash
cd /home/aadsilva/ic/hcpa-retinopathy/hcpa-repository/hcpa
```

### 2. Verify Prerequisites
```bash
# Check SLURM is available
which sbatch
sinfo

# Check Python packages
python3 -c "import cv2, tensorflow, numpy, pandas, sklearn; print('✓ All packages available')"

# Check GPU
nvidia-smi
```

### 3. Prepare Dataset (One-Time Setup)
```bash
# Verify dataset exists
ls -la /ssd/aadsilva/ic/hcpa/dataset_bruto/imagens/ | head -5
ls -la /ssd/aadsilva/ic/hcpa/dataset_bruto/mascaras/ | head -5

# Generate split.json if it doesn't exist
# (See README_PIPELINE.md for Python code to generate it)
ls /ssd/aadsilva/ic/hcpa/split.json
```

### 4. Run Pilot (Single Filter, 10 Repetitions)

```bash
# In interactive session (salloc mode) — best for first-time testing
salloc --partition grace --gres=gpu:1 --time=20:00:00
cd /home/aadsilva/ic/hcpa-retinopathy/hcpa-repository/hcpa
python3 script5_orquestrador.py --mode salloc

# OR submit to queue (sbatch mode)
python3 script5_orquestrador.py --mode sbatch
```

### 5. Monitor Progress

```bash
# Check job queue
squeue -u aadsilva

# Check results (after ~1 hour)
ls ~/resultados/*.csv
ls ~/resultados/_success_markers/*.ok

# Check logs
tail -f ~/logs_orquestracao/*.log
```

## Common Commands

### List all available filters
```bash
python3 -c "from script5_orquestrador import FilterMatrix; m = FilterMatrix(); \
[print(f['name']) for f in m.get_filters()]"
```

### Run specific filter manually
```bash
# Create dataset for AHE40.0 filter
python3 script1_cria_dataset.py --filtro AHE40.0

# Train repetition 0
python3 script2_treina.py --filtro AHE40.0 --repeticao 0

# Evaluate repetition 0
python3 script3_avalia.py --filtro AHE40.0 --repeticao 0

# Delete after all 10 reps done
python3 script4_deleta_dataset.py --filtro AHE40.0
```

### Check dataset integrity
```bash
# Verify a filtered dataset
python3 -c "
from pathlib import Path
from utils_common import count_files_in_directory

dataset_dir = Path('/ssd/aadsilva/ic/hcpa/datasets_filtrados/AHE40.0')
for split in ['train', 'val', 'test']:
    count = count_files_in_directory(dataset_dir / split)
    print(f'{split}: {count} images')
"
```

### View results
```bash
# CSV results
head ~/resultados/AHE40.0_0_*.csv

# Success markers
ls ~/resultados/_success_markers/AHE40.0*.ok

# Audit log
tail ~/auditoria_exclusoes.log
```

## Dry-Run Test (No GPU Required)

Test the complete pipeline without GPUs:

```bash
# 1. Test Script 1 alone
python3 script1_cria_dataset.py --filtro baseline

# 2. Test all validation functions
python3 -c "
from utils_common import setup_logging, get_paths, validate_paths
from utils_preprocessing import get_filter
from utils_metrics import compute_all_metrics
import numpy as np

logger = setup_logging('test')
logger.info('✓ All imports successful')

# Test metrics
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.1, 0.8, 0.9, 0.2, 0.6])
metrics = compute_all_metrics(y_true, y_pred)
logger.info(f'✓ Metrics computed: AUC={metrics[\"auc_roc\"]:.4f}')
"
```

## Troubleshooting Quick Ref

| Problem | Solution |
|---------|----------|
| `SLURM not found` | Load module: `module load slurm` |
| `GPU not available` | Check queue: `squeue -u aadsilva` \| Stop other jobs |
| `Dataset not found` | Check path: `ls /ssd/aadsilva/ic/hcpa/dataset_bruto/` |
| `Out of memory` | Reduce batch size in Script 2 (modify `dr_hcpa_v2_2024.py`) |
| `Jobs stuck` | Check limits: `sacctmgr show user aadsilva` |
| `Success marker not created` | Check CSV: `head ~/resultados/*.csv` for issues |

## Full Matrix Submission (Production)

Once pilot is calibrated:

```bash
# Create filter matrix file (optional)
cat > filter_matrix.json << 'EOF'
[
    {"name": "baseline", "params": {}, "description": "No filter"},
    {"name": "AHE40.0", "params": {"clip_limit": 40.0}, "description": "AHE"},
    {"name": "CLAHE4.0", "params": {"clip_limit": 4.0}, "description": "CLAHE"},
    {"name": "Gamma0.8", "params": {"gamma": 0.8}, "description": "Gamma"}
]
EOF

# Submit all filters
python3 script5_orquestrador.py \
    --mode sbatch \
    --matrix-file filter_matrix.json \
    --skip-completed

# Monitor
watch squeue -u aadsilva
```

## Results Analysis

```bash
# Consolidate all results
python3 << 'EOF'
import pandas as pd
from pathlib import Path
import glob

# Load all CSV files
csv_files = glob.glob("/home/aadsilva/resultados/*.csv")
dfs = [pd.read_csv(f) for f in csv_files]
df_all = pd.concat(dfs, ignore_index=True)

# Group by filter
print("\n=== Results Summary ===\n")
for filtro, group in df_all.groupby('filter_name'):
    print(f"{filtro}:")
    for metric in ['accuracy', 'sensitivity', 'specificity', 'auc_roc']:
        if metric in group.columns:
            mean = group[metric].mean()
            std = group[metric].std()
            print(f"  {metric}: {mean:.4f} ± {std:.4f}")
    print()

# Save consolidated results
df_all.to_csv("/home/aadsilva/resultados_consolidados/master.csv", index=False)
print(f"✓ Consolidated results saved to master.csv ({len(df_all)} rows)")
EOF
```

## Next Steps

1. **Run Pilot**: Execute one filter with 10 repetitions to calibrate times
2. **Calibrate**: Update time estimates in the research plan
3. **Define Filters**: Finalize the filter matrix based on your research questions
4. **Full Run**: Submit the complete matrix to SLURM
5. **Analysis**: Use the results for XAI analysis (Grad-CAM, Dice, IoU, etc.)

## Documentation

- **Full Guide**: See `README_PIPELINE.md` for comprehensive documentation
- **Research Plan**: Refer to Notion document for experimental design
- **Logs**: Check `~/logs_orquestracao/` for detailed execution logs

## Support

For issues:
1. Check logs: `tail ~/logs_orquestracao/*.log`
2. Review SLURM output: `tail logs/script*_*.out`
3. Run in verbose mode: Add `--verbose 2` to scripts
4. Consult README_PIPELINE.md troubleshooting section

---

**Status**: ✓ Ready to use  
**Last Updated**: June 2026
