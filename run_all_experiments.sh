#!/bin/bash
# Ponto de entrada ÚNICO para rodar toda a matriz experimental (~180 execuções).
#
# Uso (a partir da raiz do repositório):
#     sbatch run_all_experiments.sh
#
# Depois disso não é necessário rodar mais nenhum comando manual: este job
# submete e acompanha todos os demais jobs (Script 1 → TFRecord → Script 2 →
# Script 3 → Script 4) sozinho, para todos os filtros pendentes de
# experiments/filter_matrix_template.json, até que a matriz inteira esteja
# concluída (ou até que só restem falhas registradas em failed_jobs.json).
#
# Roda na partição `shared` (não `grace`): este processo é apenas um
# despachante leve (não usa GPU, não roda os experimentos em si) — se ele
# ocupasse um nó `grace` com exclusividade, bloquearia os próprios nós que
# precisa gerenciar. `grace1`/`grace2` ficam inteiramente livres para os
# jobs reais (Script 1-4), cuja escolha de nó é feita pelo próprio SLURM.
#
# Este job se auto-resubmete perto do limite de tempo da partição (24h) caso
# ainda haja trabalho pendente — não é necessário reenviar manualmente.

#SBATCH --job-name=hcpa_orchestrator
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
echo "HCPA — Orquestrador único (run_all_experiments.sh)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node (dispatcher): $(hostname)"
echo "=========================================="

# Lógica de orquestração vive em pipeline/script5_orquestrador.py — este
# wrapper só configura a alocação SLURM do próprio despachante e o invoca.
# python3 do host (não precisa do container: nenhuma dependência de
# TensorFlow/OpenCV é usada pela lógica de orquestração em si, só pelos
# scripts 1-4 que ele submete separadamente, cada um já dentro do Singularity).
exec python3 "$SUBMIT_DIR/pipeline/script5_orquestrador.py"
