#!/bin/bash
# SLURM Job Template 2 — Scripts 2+3 (Training + Evaluation)
#
# Objetivo: Treinar e avaliar modelo em dataset filtrado (uma repetição)
# Recurso: 1 GPU (nó Grace, --exclusive garante o nó inteiro)
# Frequência: Uma vez por filtro × repetição (10 repetições por filtro)
# Tempo estimado: Variável, ~2-8 horas (calibrar com piloto)
#
# Uso: sbatch --export=FILTRO="nome_filtro",REPETICAO=0 job_template_script2_3.sh
#      ou com --dependency=afterok:JOB_ID_FROM_SCRIPT1
# IMPORTANTE: submeter sempre a partir da raiz do repositório (hcpa/).
# CRITICO: SSD_BASE costuma ser disco local do nó (ex. /ssd/aadsilva/ic/hcpa).
#   Se estiver rodando as etapas de um MESMO filtro manualmente (fora do
#   orquestrador), sempre passe o MESMO --nodelist em todas elas, ex.:
#   sbatch --nodelist=grace1 --export=FILTRO=X ...
#   Caso contrário, uma etapa pode não enxergar os arquivos que a etapa
#   anterior gravou no disco local de outro nó.

#SBATCH --job-name=hcpa_train_and_eval
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-.}"
SIF="$SUBMIT_DIR/singularity.sif"
cd "$SUBMIT_DIR/pipeline"

export SSD_BASE="${SSD_BASE:-/ssd/aadsilva/ic/hcpa}"
export HOME_BASE="${HOME_BASE:-$HOME}"

FILTRO="${FILTRO:-AHE40.0_CLAHE4.0_Gamma0.8}"
REPETICAO="${REPETICAO:-0}"

echo "=========================================="
echo "SLURM Job Template 2+3: Training + Eval"
echo "=========================================="
echo "Filter: $FILTRO"
echo "Repetition: $REPETICAO"
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

echo "GPU status (host):"
nvidia-smi || echo "(nvidia-smi unavailable on host — checked again inside the container below)"

# Run Script 2 (Training) — --nv forwards the NVIDIA driver/GPU into the
# container; SLURM's --gres=gpu:1 already restricts which GPU is visible.
echo ""
echo "Running Script 2 (Training)..."
if ! singularity exec --nv \
        --bind "$SUBMIT_DIR:$SUBMIT_DIR" --bind "$HOME:$HOME" --bind "$SSD_BASE:$SSD_BASE" \
        "$SIF" \
        python3 script2_treina.py \
            --filtro "$FILTRO" \
            --repeticao "$REPETICAO" \
            --base-ssd "$SSD_BASE" \
            --base-home "$HOME_BASE"
then
    echo "✗ Script 2 failed"
    exit 1
fi
echo "✓ Script 2 completed successfully"

# Run Script 3 (Evaluation)
echo ""
echo "Running Script 3 (Evaluation)..."
if ! singularity exec --nv \
        --bind "$SUBMIT_DIR:$SUBMIT_DIR" --bind "$HOME:$HOME" --bind "$SSD_BASE:$SSD_BASE" \
        "$SIF" \
        python3 script3_avalia.py \
            --filtro "$FILTRO" \
            --repeticao "$REPETICAO" \
            --base-ssd "$SSD_BASE" \
            --base-home "$HOME_BASE"
then
    echo "✗ Script 3 failed"
    exit 1
fi
echo "✓ Script 3 completed successfully"

echo ""
echo "=========================================="
echo "✓ Scripts 2+3 completed successfully"
echo "=========================================="
