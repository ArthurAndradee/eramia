# Avaliação multi-método da relação entre desempenho e explicabilidade em triagem de retinopatia diabética

Código e dados agregados do artigo homônimo (ERAMIA-RS). O estudo treina a
EfficientNetV2S no conjunto FGADR sob diferentes pré-processamentos e compara
o desempenho de classificação (AUC-ROC, sensibilidade, especificidade) com a
fidelidade das explicações de Grad-CAM, LIME e Occlusion Sensitivity (Dice e
Pointing Game frente às máscaras de lesão), além do teste de sanidade de
Adebayo et al. (aleatorização em cascata dos pesos).

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `pipeline/` | Pré-processamento, criação de TFRecords, treino, avaliação, XAI, teste de sanidade e orquestrador SLURM |
| `experiments/generate_filter_matrix.py` | Gera a grade de 576 condições (`filter_matrix_template.json`): 4 etapas, no máximo um filtro por etapa, qualquer etapa pode ser omitida, (5+1)×(3+1)×(2+1)×(7+1) = 576 |
| `experiments/filter_matrix_xai_mini.json` | As 10 condições avaliadas com as três técnicas de XAI |
| `experiments/analise_final_576/` | Resultados agregados e análise da varredura de 576 condições (Grad-CAM) |
| `experiments/paper_auc_xai_correlacao/` | Resultados da mini-campanha, scripts das figuras do artigo |
| `experiments/paper_auc_xai_correlacao/revisao_final/` | Figura 2 da versão final (dispersão sem retas) e correlações de apoio (Pearson com IC95% *bootstrap*, Spearman, sem a condição colapsada; Grad-CAM na varredura completa) |
| `documentation/` | Guia de execução e notas metodológicas |
| `slurm/`, `singularity.def`, `requirements.txt` | Ambiente de execução |

## Execução

Os experimentos rodaram em um cluster SLURM com GPUs, dentro do contêiner
definido em `singularity.def`. A sequência e as variáveis de ambiente
(`SSD_BASE` etc.) estão em `documentation/RUN_EXPERIMENTS.md`.

```bash
sbatch run_all_experiments.sh       # varredura de 576 condições (Grad-CAM)
sbatch run_xai_mini_campaign.sh     # 10 condições x Grad-CAM, LIME e Occlusion
sbatch run_sanity_check.sh          # teste de sanidade de Adebayo
sbatch run_qualitative_figure.sh    # Figura 1
```

Opcionalmente, defina `NOTIFY_EMAIL` para receber alertas do orquestrador
por e-mail; sem ela, os alertas ficam apenas em `logs/ALERTAS_PENDENTES.txt`.

As figuras e tabelas podem ser regeneradas a partir dos CSVs versionados,
sem retreinar:

```bash
cd experiments/paper_auc_xai_correlacao/revisao_final
Rscript make_figura2_dispersao.R    # Figura 2
python3 make_tabelas_revisao.py     # correlações de apoio
```

## Dados

As imagens e máscaras do FGADR não são redistribuídas aqui, exceto a amostra
reduzida (299×299) usada na Figura 1 (`figura_qualitativa_cache.npz`). O
acesso ao conjunto deve ser solicitado aos seus autores (Zhou et al.,
*IEEE TMI*, 2021).
