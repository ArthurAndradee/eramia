#!/bin/bash
# SLURM Job Template 1 — Script 1 (Dataset Creation)
#
# Objetivo: Gerar dataset filtrado para um filtro específico
# Recurso: CPU-only (sem GPU necessária)
# Frequência: Uma vez por filtro
# Tempo estimado: ~1-2 horas
#
# Uso: sbatch --export=FILTRO="filtro_name" job_template_script1.sh
#      ou simplemente: sbatch job_template_script1.sh (com FILTRO definido aqui)
# IMPORTANTE: submeter sempre a partir da raiz do repositório (hcpa/).
# CRITICO: SSD_BASE costuma ser disco local do nó (ex. /ssd/aadsilva/ic/hcpa).
#   Se estiver rodando as etapas de um MESMO filtro manualmente (fora do
#   orquestrador), sempre passe o MESMO --nodelist em todas elas, ex.:
#   sbatch --nodelist=grace1 --export=FILTRO=X ...
#   Caso contrário, uma etapa pode não enxergar os arquivos que a etapa
#   anterior gravou no disco local de outro nó.

#SBATCH --job-name=hcpa_dataset_creation
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --partition=grace

set -euo pipefail

# Repo layout: this template lives in slurm/, pipeline scripts live in
# pipeline/ — SLURM sets SLURM_SUBMIT_DIR to wherever `sbatch` was invoked
# from (repo root, per the documented submission convention).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-.}"
SIF="$SUBMIT_DIR/singularity.sif"
cd "$SUBMIT_DIR/pipeline"

# Configuration
export SSD_BASE="${SSD_BASE:-/ssd/aadsilva/ic/hcpa}"
export HOME_BASE="${HOME_BASE:-$HOME}"

# Filter name (can be overridden via --export)
FILTRO="${FILTRO:-AHE40.0_CLAHE4.0_Gamma0.8}"

echo "=========================================="
echo "SLURM Job Template 1: Dataset Creation"
echo "=========================================="
echo "Filter: $FILTRO"
echo "SSD Base: $SSD_BASE"
echo "HOME Base: $HOME_BASE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Create logs directory (relative to the submission dir, not pipeline/)
mkdir -p "$SUBMIT_DIR/logs"

# SSD_BASE (e.g. /ssd/aadsilva/ic/hcpa) is node-local scratch OUTSIDE both
# $SUBMIT_DIR and $HOME — this cluster's singularity.conf does not auto-bind
# it (confirmed: `singularity exec $SIF ls /ssd` fails with "No such file or
# directory" without an explicit --bind), so it must be bound explicitly or
# every path under it silently resolves inside the container's own ephemeral
# filesystem instead of the real host disk. mkdir first: --bind requires the
# source to already exist on the host.
mkdir -p "$SSD_BASE"

# Run Script 1 inside the project's Singularity container (bare `python3` on
# the host has none of tensorflow/opencv/etc. — see singularity.def, built
# specifically for this cluster's Grace ARM64 nodes).
if ! singularity exec \
        --bind "$SUBMIT_DIR:$SUBMIT_DIR" --bind "$HOME:$HOME" --bind "$SSD_BASE:$SSD_BASE" \
        "$SIF" \
        python3 script1_cria_dataset.py \
            --filtro "$FILTRO" \
            --base-ssd "$SSD_BASE" \
            --base-home "$HOME_BASE"
then
    echo "✗ Script 1 failed"
    exit 1
fi

echo "✓ Script 1 completed successfully"
