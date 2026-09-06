# XAI-Guided Preprocessing Pipeline — Guia do Pipeline

> Atualizado em 2026-07-07 após auditoria operacional completa (execução
> real via SLURM + Singularity no cluster, validando idempotência,
> concorrência e limpeza). A versão anterior (agora em
> `documentation/archive/README_PIPELINE.md`) descrevia uma arquitetura que
> nunca funcionou de ponta a ponta (`dataset_bruto/`, split 3-vias,
> `/ssd/aadsilva/ic/hcpa` hardcoded, sem geração real de TFRecords, sem
> Grad-CAM). Para as limitações metodológicas conhecidas, ver
> `METHODOLOGICAL_NOTES.md`. Para o guia passo a passo de como executar os
> ~180 experimentos em produção (preparação do SSD, comandos reais,
> monitoramento, recuperação de falhas), ver `RUN_EXPERIMENTS.md`.

## Estrutura do repositório

```
hcpa/
├── pipeline/            # Todo o código do pipeline (scripts + utilitários)
│   ├── script1_cria_dataset.py
│   ├── create-tfrecord.py
│   ├── script2_treina.py
│   ├── dr_hcpa_v2_2024.py     # treino: EfficientNetV2S, 2 fases (warm-up + fine-tuning)
│   ├── script3_avalia.py
│   ├── script4_deleta_dataset.py
│   ├── script5_orquestrador.py
│   ├── make_split.py
│   ├── utils_common.py        # get_paths(), binarização, labels, máscaras
│   ├── utils_preprocessing.py # filtros de imagem (AHE/CLAHE/Gamma/MaxGreen/...)
│   ├── utils_metrics.py       # Dice, IoU, Pointing Game, métricas clínicas
│   └── utils_gradcam.py       # Grad-CAM sobre o backbone EfficientNetV2S
├── slurm/               # Templates de submissão SLURM (ver "Execução no cluster")
├── experiments/         # split.json, split_metadata.json, filter_matrix_template.json
├── tools/                # Scripts auxiliares ainda compatíveis (não fazem parte do fluxo principal)
├── legacy/               # Scripts/dados órfãos de pipelines anteriores (não tocar)
├── documentation/
│   ├── README_PIPELINE.md        (este arquivo)
│   ├── METHODOLOGICAL_NOTES.md   (limitações e decisões metodológicas)
│   └── archive/                  (documentação histórica, desatualizada)
├── data/
│   ├── Seg-set/                  # dataset FGADR real (Original_Images + *_Masks + labels)
│   └── tfrecords_filtrados/      # TFRecords gerados por filtro (artefato, gitignored)
├── datasets_filtrados/    # imagens filtradas por filtro (artefato, gitignored)
├── checkpoints/           # modelos treinados (artefato, gitignored; .keras
│                          #   removido automaticamente após avaliação com
│                          #   sucesso — só os metadados/CSVs pequenos ficam)
└── requirements.txt, .gitignore, singularity.def/.sif, .venv/
```

## Dataset

`data/Seg-set/`:
- `Original_Images/` — 1842 imagens de fundo de olho, 1280×1280px, `.png`.
- `HardExudate_Masks/`, `Hemohedge_Masks/`, `IRMA_Masks/`, `Microaneurysms_Masks/`,
  `Neovascularization_Masks/`, `SoftExudate_Masks/` — máscaras de lesão por tipo
  (nem toda imagem tem todos os tipos; usadas **somente** por `script3_avalia.py`
  para métricas XAI, nunca filtradas).
- `DR_Seg_Grading_Label.csv` — grade ICDR (0-4) por imagem.

**Binarização (Referable DR):** grade ICDR ≥ 2 → 1 (referável); grades 0,1 → 0
(não-referável). Fonte única de verdade: `utils_common.binarize_icdr_grade()`,
usada por `create-tfrecord.py` e `script3_avalia.py`. O modelo é sempre um
classificador binário (`Dense(1, sigmoid)`); `dr_hcpa_v2_2024.py` falha
explicitamente (`assert`) se `--num_classes` for chamado com valor diferente de 2.

## Split

`experiments/split.json`: 70% treino (1289 imagens) / 30% teste (553 imagens),
estratificado pela grade ICDR (5 classes — superconjunto da estratificação
binária, ver `METHODOLOGICAL_NOTES.md` item 4), seed fixa (42),
`sklearn.model_selection.train_test_split`. Documentado em
`experiments/split_metadata.json` (contagens, distribuição de classes,
distribuição binária, commit git). **Gerado uma única vez** por
`pipeline/make_split.py` — reutilizado por todos os filtros, nunca
regenerado durante os experimentos (o script recusa sobrescrever sem
`--force`).

## Fluxo do pipeline (por filtro)

```
data/Seg-set/Original_Images/*.png
        │
        ▼
script1_cria_dataset.py --filtro X       (CPU) — aplica o filtro só nas imagens,
        │                                         nunca nas máscaras
        ▼
datasets_filtrados/X/{train,test}/*.png
        │
        ▼
create-tfrecord.py --filtro X            (CPU) — redimensiona p/ 299×299,
        │                                         binariza o rótulo, grava TFRecords
        ▼
data/tfrecords_filtrados/X/{train,test}_shard*.tfrec
        │
        ▼
script2_treina.py --filtro X --repeticao N   (GPU) — invoca dr_hcpa_v2_2024.py
        │                                             (EfficientNetV2S, 2 fases:
        │                                              warm-up 5 épocas + fine-tuning
        │                                              até 50 épocas; 10 repetições,
        │                                              seed = N; pula se já treinado
        │                                              E avaliado, ou se já treinado
        │                                              e só falta avaliar)
        ▼
checkpoints/X-N.keras   (pesos da MELHOR época por val_loss da Fase 2,
                         restaurados via EarlyStopping(restore_best_weights=True)
                         antes do save final — não a última época; REMOVIDO
                         automaticamente pelo Script 3 logo após a avaliação
                         suceder — ver abaixo)
        │
        ▼
script3_avalia.py --filtro X --repeticao N   (GPU) — inferência + Grad-CAM real
        │                                             vs. máscaras de data/Seg-set
        ▼
~/resultados/X_N_<timestamp>.csv   (métricas clínicas + XAI reais)
~/resultados/_success_markers/X_N.ok
checkpoints/X-N.keras É REMOVIDO aqui (só após CSV validado + marker criados;
                       em caso de falha na avaliação o checkpoint é preservado)
        │
        ▼ (após as 10 repetições)
script4_deleta_dataset.py --filtro X     (CPU) — apaga datasets_filtrados/X/ E
                                                   data/tfrecords_filtrados/X/
                                                   (ambos reproduzíveis a partir
                                                   de data/Seg-set + split.json);
                                                   nunca checkpoints/ nem ~/resultados/
```

Tudo isso é orquestrado por `pipeline/script5_orquestrador.py` (modos
`sbatch` para produção e `salloc` para piloto/debug), que já injeta a etapa
de TFRecord na cadeia de dependências SLURM e usa
`experiments/filter_matrix_template.json` como matriz padrão quando presente.

**Concorrência:** script1, create-tfrecord.py e script4 adquirem um lock
por filtro (`utils_common.filter_lock()`, baseado em `fcntl.flock`, arquivo
em `locks/{filtro}.lock`) antes de ler/escrever `datasets_filtrados/{filtro}`
ou `data/tfrecords_filtrados/{filtro}` — duas execuções concorrentes para o
mesmo filtro (ex.: resubmissão manual, ou o fallback de auto-geração de
TFRecords do script2) serializam em vez de corromper os mesmos arquivos.

**Idempotência:** reenviar um filtro parcialmente concluído (ex.: 9/10
repetições OK, 1 falhou) é seguro e eficiente — `script1`/`create-tfrecord.py`
pulam se o fingerprint do split já bate; `script2` pula o retreino se o
success marker já existir (treinado E avaliado — o checkpoint já foi
removido nesse caso) OU se `checkpoints/{filtro}-{repeticao}.keras` + o
`training_metadata` já existirem sem marker (treinado, ainda não avaliado);
`script3` já pulava via success marker. Só a repetição realmente faltante
roda de verdade.

**Distribuição entre nós:** desde 2026-07-08, `script5_orquestrador.py` fixa
cada filtro em um nó via hash estável (`sha256`) do nome do filtro — não
mais pela posição na lista (`filter_idx % len(nodes)`) — garantindo que
reordenar/remover filtros da matriz ou retomar um subconjunto nunca mande um
filtro para um nó diferente daquele onde seus dados já estão no SSD local.
Ver `documentation/RUN_EXPERIMENTS.md` seção 4.4.

## Matriz de filtros

`experiments/filter_matrix_template.json` — desenho experimental
definitivo, 18 condições (baseline + 12 isoladas + 4 pares + 1 composto
triplo), cada uma com `rationale` (justificativa) e `hypothesis` (hipótese
testada). Documentação completa:
- `experiments/FILTER_AUDIT.md` — auditoria de todos os filtros
  implementados (parâmetros, custo, impacto esperado, artefatos,
  compatibilidade, evidência de literatura, inclusão/exclusão).
- `experiments/LITERATURE_REVIEW.md` — revisão bibliográfica com fontes.
- `experiments/EXPERIMENTAL_DESIGN.md` — justificativa metodológica,
  autocrítica da matriz, e relatório final (cobertura, limitações,
  extensões futuras, custo computacional estimado).

18 filtros × 10 repetições = 180 execuções de treino+avaliação.

## Execução local (sem SLURM)

```bash
cd hcpa/pipeline
python3 make_split.py                       # uma única vez
python3 script1_cria_dataset.py --filtro CLAHE4.0
python3 create-tfrecord.py --filtro CLAHE4.0
python3 script2_treina.py --filtro CLAHE4.0 --repeticao 0
python3 script3_avalia.py --filtro CLAHE4.0 --repeticao 0
python3 script4_deleta_dataset.py --filtro CLAHE4.0
```

Ou tudo de uma vez, para 1 filtro, via orquestrador em modo interativo:
```bash
cd hcpa/pipeline
python3 script5_orquestrador.py --mode salloc
```

## Execução no cluster (SLURM)

**Submeter sempre a partir da raiz do repositório** (`hcpa/`) — os templates
em `slurm/` usam `$SLURM_SUBMIT_DIR` para localizar `pipeline/`:

```bash
cd hcpa
sbatch slurm/job_template_script1.sh
# ou produção completa:
cd hcpa/pipeline
python3 script5_orquestrador.py --mode sbatch --skip-completed --partition grace
```

Todos os jobs (CPU-only e GPU) rodam na partição `grace`, exclusivos
(`--exclusive`, política obrigatória do cluster). Cada job roda dentro do
container Singularity (`singularity.sif`, ver `singularity.def`) via
`singularity exec [--nv]` — não depende de nenhum `python3`/venv ativado
manualmente no nó.

**`SSD_BASE` (`/ssd/aadsilva/ic/hcpa` em produção) é armazenamento LOCAL de
cada nó Grace** (confirmado nesta auditoria: `grace1` e `grace2` têm cada um
sua própria cópia independente em `/ssd/aadsilva`). Por isso
`script5_orquestrador.py` (`--mode sbatch`) fixa um nó por filtro
(`--nodelist`, round-robin entre os nós da partição via `get_available_nodes()`)
e propaga o MESMO nó para todas as etapas daquele filtro — script1,
TFRecord, as 10 repetições de treino+avaliação, e a limpeza. Filtros
diferentes podem cair em nós diferentes (paralelismo entre filtros), mas
dentro de um mesmo filtro tudo roda sempre no mesmo `/ssd`. Ao submeter os
templates estáticos em `slurm/` manualmente (fora do orquestrador), passe o
mesmo `--nodelist` em todas as etapas do mesmo filtro:
`sbatch --nodelist=grace1 --export=FILTRO=... slurm/job_template_script1.sh`.

Validado nesta auditoria com submissões reais (`sbatch`, sem `--test-only`)
no cluster: `script1_cria_dataset.py` e `create-tfrecord.py` executando
dentro do container em `grace2`, produzindo `datasets_filtrados/{filtro}` e
`data/tfrecords_filtrados/{filtro}` com contagens corretas; `script4` também
validado (recusa corretamente sem os 10 markers, limpa os dois artefatos
quando presentes).

**Nomenclatura dos jobs:** `hcpa_dataset_{filtro}`, `hcpa_tfrecord_{filtro}`,
`hcpa_train_{filtro}_{repeticao}`, `hcpa_cleanup_{filtro}` — logs em
`logs/%x_%j.log` / `logs/%x_%j.err` (relativo à raiz do repo).

## Caminhos ($SSD_BASE / $HOME)

`utils_common.get_paths()` resolve `base_ssd` como `$SSD_BASE` (env var) ou,
se não definida, o diretório raiz do repositório (`hcpa/`) — não depende de
`/ssd/aadsilva/ic/hcpa` existir para rodar localmente/em dev. `base_home`
resolve para `$HOME` (onde ficam `~/resultados/`, `~/logs_orquestracao/`,
`~/trash_datasets/`).

## Reprodutibilidade

- Seed fixa por repetição (0-9), agora efetiva de verdade em
  `dr_hcpa_v2_2024.py` via `--seed` (`tf.random.set_seed`, `np.random.seed`,
  `random.seed`) — ver `METHODOLOGICAL_NOTES.md` item 5.
- O checkpoint salvo (`checkpoints/{filtro}-{repeticao}.keras`) reflete os
  pesos da MELHOR época por `val_loss` da Fase 2 (fine-tuning), restaurados
  em memória por `EarlyStopping(restore_best_weights=True)` antes do save
  final — nunca a última época.
- Split único, documentado, nunca regenerado.
- Git commit hash registrado em todo metadata.json/CSV/marker.
- Limitações conhecidas (vazamento teste/validação, threshold do Grad-CAM,
  Pointing Game): ver `METHODOLOGICAL_NOTES.md`.

## Ambiente de execução

- **Cluster (produção):** todos os `python3` rodam dentro do container
  Singularity (`singularity.sif`, TensorFlow 2.13 + opencv/pandas/sklearn
  compilados para NVIDIA Grace ARM64 — ver `singularity.def`). Os templates
  em `slurm/` e os scripts gerados por `script5_orquestrador.py` já
  encapsulam isso; não é preciso ativar nada manualmente.
- **Local/dev (sem GPU/SLURM):** os scripts também rodam com `python3`
  direto, desde que `tensorflow`/`opencv-python`/`scikit-learn`/`pandas`
  estejam instalados no ambiente (ver `requirements.txt` ou `.venv/`) — não
  requer Singularity.
