#!/bin/bash
# SLURM Job Template — TFRecord Generation (per filter)
#
# Objetivo: Converter datasets_filtrados/{filtro}/{train,test} em TFRecords
# Recurso: CPU-only (sem GPU necessária)
# Frequência: Uma vez por filtro, entre Script 1 e os jobs de treino (Script 2+3)
# Tempo estimado: ~15-60 min
#
# Uso: sbatch --export=FILTRO="filtro_name" job_template_tfrecord.sh
#      ou com --dependency=afterok:JOB_ID_FROM_SCRIPT1
# IMPORTANTE: submeter sempre a partir da raiz do repositório (hcpa/).
# CRITICO: SSD_BASE costuma ser disco local do nó (ex. /ssd/aadsilva/ic/hcpa).
#   Se estiver rodando as etapas de um MESMO filtro manualmente (fora do
#   orquestrador), sempre passe o MESMO --nodelist em todas elas, ex.:
#   sbatch --nodelist=grace1 --export=FILTRO=X ...
#   Caso contrário, uma etapa pode não enxergar os arquivos que a etapa
#   anterior gravou no disco local de outro nó.

#SBATCH --job-name=hcpa_tfrecord_generation
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --partition=grace

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-.}"
SIF="$SUBMIT_DIR/singularity.sif"
cd "$SUBMIT_DIR/pipeline"

export SSD_BASE="${SSD_BASE:-/ssd/aadsilva/ic/hcpa}"
export HOME_BASE="${HOME_BASE:-$HOME}"

FILTRO="${FILTRO:-AHE40.0_CLAHE4.0_Gamma0.8}"

echo "=========================================="
echo "SLURM Job Template: TFRecord Generation"
echo "=========================================="
echo "Filter: $FILTRO"
echo "SSD Base: $SSD_BASE"
echo "HOME Base: $HOME_BASE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

mkdir -p "$SUBMIT_DIR/logs"

# SSD_BASE is node-local scratch outside $SUBMIT_DIR/$HOME and is NOT
# auto-bound by this cluster's singularity.conf — must bind explicitly (see
# job_template_script1.sh for how this was confirmed).
mkdir -p "$SSD_BASE"

if ! singularity exec \
        --bind "$SUBMIT_DIR:$SUBMIT_DIR" --bind "$HOME:$HOME" --bind "$SSD_BASE:$SSD_BASE" \
        "$SIF" \
        python3 create-tfrecord.py \
            --filtro "$FILTRO" \
            --base-ssd "$SSD_BASE" \
            --base-home "$HOME_BASE"
then
    echo "✗ TFRecord generation failed"
    exit 1
fi

echo "✓ TFRecord generation completed successfully"
