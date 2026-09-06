#!/bin/bash
# Gera a figura qualitativa (amostra FGADR + anotacao clinica + Grad-CAM +
# LIME + Occlusion Sensitivity) a partir de um checkpoint real mantido.
# Ver pipeline/make_qualitative_figure.py.
#
# Uso: sbatch run_qualitative_figure.sh

#SBATCH --job-name=hcpa_qualitative_figure
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:30:00

set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$SUBMIT_DIR"
mkdir -p logs

export SSD_BASE="$SUBMIT_DIR"
export HOME_BASE="$HOME"
export GPU_IDX=0

SIF_PATH="$SUBMIT_DIR/singularity.sif"
SINGULARITY_ARGS=(exec --nv --bind "$SUBMIT_DIR:$SUBMIT_DIR" --bind "$HOME:$HOME" "$SIF_PATH")

ALERT_LOG="$SUBMIT_DIR/logs/ALERTAS_PENDENTES.txt"
NOTIFY_EMAIL="botooooxgamer123@gmail.com"

echo "=========================================="
echo "HCPA — Figura qualitativa (amostra + anotacao + XAI)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "=========================================="

if singularity "${SINGULARITY_ARGS[@]}" python3 "$SUBMIT_DIR/pipeline/make_qualitative_figure.py" \
    --base-ssd "$SSD_BASE" --base-home "$HOME_BASE"
then
    echo "✓ Figura qualitativa gerada"
else
    MSG="[$(date '+%Y-%m-%d %H:%M:%S')] hcpa_qualitative_figure (job ${SLURM_JOB_ID:-manual}) falhou -- ver logs/hcpa_qualitative_figure_${SLURM_JOB_ID:-manual}.log/.err"
    echo -e "$MSG\n$(printf '%.0s-' {1..60})" >> "$ALERT_LOG"
    echo "$MSG" | mailx -s "[HCPA] figura qualitativa falhou" "$NOTIFY_EMAIL" 2>/dev/null || true
    echo "✗ Figura qualitativa falhou"
    exit 1
fi
