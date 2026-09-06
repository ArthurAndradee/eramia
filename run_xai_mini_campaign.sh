#!/bin/bash
# Ponto de entrada da mini-campanha de metodologia XAI (10 filtros, ver
# experiments/paper_auc_xai_correlacao e experiments/filter_matrix_xai_mini.json).
#
# Uso (a partir da raiz do repositório):
#     sbatch run_xai_mini_campaign.sh
#
# Reaproveita o mesmo despachante da campanha principal
# (pipeline/script5_orquestrador.py) e toda a engenharia de confiabilidade
# já validada nela (retry limitado, diversificacao de no em retry,
# balanceamento entre grace1/grace2, cancelamento de jobs presos, limite
# de fila) -- so muda: a matriz (10 filtros, nao 576), os metodos de XAI
# avaliados (Grad-CAM + LIME + Occlusion, nao so Grad-CAM), a subamostra
# fixa de imagens para esses metodos, a retencao do checkpoint apos
# avaliacao (necessaria para o teste de sanidade de Adebayo depois), e o
# deslocamento de repeticao (10-19, nao 0-9 -- essas 10 condicoes ja tem
# 0-9 completos na campanha principal; ver REP_OFFSET no orquestrador).
#
# Roda na particao `shared`, mesma logica do wrapper principal: e so um
# despachante leve, nao ocupa nó grace com exclusividade.

#SBATCH --job-name=hcpa_xai_mini_orchestrator
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=shared
#SBATCH --time=23:50:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-.}"
cd "$SUBMIT_DIR"
mkdir -p logs

echo "=========================================="
echo "HCPA — Mini-campanha de metodologia XAI (run_xai_mini_campaign.sh)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node (dispatcher): $(hostname)"
echo "=========================================="

exec python3 "$SUBMIT_DIR/pipeline/script5_orquestrador.py" \
    --matrix-file "$SUBMIT_DIR/experiments/filter_matrix_xai_mini.json" \
    --xai-methods "gradcam,lime,occlusion" \
    --xai-subsample 100 \
    --keep-checkpoint \
    --rep-offset 10 \
    --wrapper-script run_xai_mini_campaign.sh
