#!/bin/bash
# SLURM Job Template 4 — Script 4 (Cleanup: filtered dataset + TFRecords)
#
# Objetivo: Remover (soft ou hard) o dataset filtrado e os TFRecords do
#           filtro após todas as 10 repetições completarem com sucesso.
#           Mantém checkpoints/, ~/resultados/ e logs intocados.
# Recurso: CPU-only (sem GPU necessária)
# Frequência: Uma vez por filtro, após todas as 10 repetições completarem
# Tempo estimado: ~5-30 minutos
#
# Pré-condição: Todos os 10 marcadores de sucesso (_success_markers) devem existir
#
# Uso: sbatch --export=FILTRO="nome_filtro" job_template_script4.sh
#      com --dependency=afterok:ID1:ID2:...:ID10 (IDs dos 10 jobs de repetição)
# IMPORTANTE: submeter sempre a partir da raiz do repositório (hcpa/).
# CRITICO: SSD_BASE costuma ser disco local do nó (ex. /ssd/aadsilva/ic/hcpa).
#   Se estiver rodando as etapas de um MESMO filtro manualmente (fora do
#   orquestrador), sempre passe o MESMO --nodelist em todas elas, ex.:
#   sbatch --nodelist=grace1 --export=FILTRO=X ...
#   Caso contrário, uma etapa pode não enxergar os arquivos que a etapa
#   anterior gravou no disco local de outro nó.

#SBATCH --job-name=hcpa_cleanup
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --partition=grace

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-.}"
SIF="$SUBMIT_DIR/singularity.sif"
cd "$SUBMIT_DIR/pipeline"

export SSD_BASE="${SSD_BASE:-/ssd/aadsilva/ic/hcpa}"
export HOME_BASE="${HOME_BASE:-$HOME}"

FILTRO="${FILTRO:-AHE40.0_CLAHE4.0_Gamma0.8}"

# Deletion mode (soft-delete by default, can be overridden)
DELETION_MODE="${DELETION_MODE:-soft}"  # soft or hard

echo "=========================================="
echo "SLURM Job Template 4: Cleanup"
echo "=========================================="
echo "Filter: $FILTRO"
echo "Deletion Mode: $DELETION_MODE"
echo "SSD Base: $SSD_BASE"
echo "HOME Base: $HOME_BASE"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

mkdir -p "$SUBMIT_DIR/logs"

# SSD_BASE is node-local scratch outside $SUBMIT_DIR/$HOME and is NOT
# auto-bound by this cluster's singularity.conf — must bind explicitly (see
# job_template_script1.sh for how this was confirmed). Especially critical
# here: without it, script4's deletion would operate on paths resolved
# inside the container's own ephemeral filesystem, not the real files on
# the host SSD — the "cleanup" would silently do nothing on the host.
mkdir -p "$SSD_BASE"

SCRIPT_ARGS=(--filtro "$FILTRO" --base-ssd "$SSD_BASE" --base-home "$HOME_BASE")
if [ "$DELETION_MODE" = "hard" ]; then
    SCRIPT_ARGS+=(--hard-delete)
fi

SINGULARITY_ARGS=(exec --bind "$SUBMIT_DIR:$SUBMIT_DIR" --bind "$HOME:$HOME" --bind "$SSD_BASE:$SSD_BASE" "$SIF")

# First run in dry-run mode to verify
echo ""
echo "Running in DRY-RUN mode to verify..."
if ! singularity "${SINGULARITY_ARGS[@]}" python3 script4_deleta_dataset.py "${SCRIPT_ARGS[@]}" --dry-run; then
    echo "✗ Dry-run failed"
    exit 1
fi
echo "✓ Dry-run passed"

# Now run actual deletion
echo ""
echo "Running actual deletion..."
if ! singularity "${SINGULARITY_ARGS[@]}" python3 script4_deleta_dataset.py "${SCRIPT_ARGS[@]}"; then
    echo "✗ Deletion failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Script 4 (Cleanup) completed successfully"
echo "=========================================="
