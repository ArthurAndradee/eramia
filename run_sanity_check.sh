#!/bin/bash
# Roda o sanity check de Adebayo (cascading model-randomization) nos
# checkpoints reais mantidos da mini-campanha de metodologia XAI.
# Ver pipeline/run_sanity_check.py e pipeline/utils_sanity_check.py.
#
# Uso: sbatch run_sanity_check.sh

#SBATCH --job-name=hcpa_sanity_check
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=02:00:00

set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$SUBMIT_DIR"
mkdir -p logs

export SSD_BASE="$SUBMIT_DIR"
export HOME_BASE="$HOME"
export GPU_IDX=0

SIF_PATH="$SUBMIT_DIR/singularity.sif"
SINGULARITY_ARGS=(exec --nv --bind "$SUBMIT_DIR:$SUBMIT_DIR" --bind "$HOME:$HOME" "$SIF_PATH")

echo "=========================================="
echo "HCPA — Sanity check de Adebayo (checkpoints reais)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "=========================================="

ALERT_LOG="$SUBMIT_DIR/logs/ALERTAS_PENDENTES.txt"
NOTIFY_EMAIL="botooooxgamer123@gmail.com"

if singularity "${SINGULARITY_ARGS[@]}" python3 "$SUBMIT_DIR/pipeline/run_sanity_check.py" \
    --base-ssd "$SSD_BASE" --base-home "$HOME_BASE" --n-images 15 --n-steps 6
then
    echo "✓ Sanity check completo"
else
    # Same notification convention as script5_orquestrador.py's
    # notify_operator(): guaranteed local alert file + best-effort email.
    # This is a one-shot job (not a daemon), so a bash-level fallback here
    # instead of importing the Python helper.
    MSG="[$(date '+%Y-%m-%d %H:%M:%S')] hcpa_sanity_check (job ${SLURM_JOB_ID:-manual}) falhou -- ver logs/hcpa_sanity_check_${SLURM_JOB_ID:-manual}.log/.err"
    echo -e "$MSG\n$(printf '%.0s-' {1..60})" >> "$ALERT_LOG"
    echo "$MSG" | mailx -s "[HCPA] sanity check falhou" "$NOTIFY_EMAIL" 2>/dev/null || true
    echo "✗ Sanity check falhou"
    exit 1
fi
