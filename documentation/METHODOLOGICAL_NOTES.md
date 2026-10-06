# Notas metodológicas — XAI-Guided Preprocessing Pipeline

Este documento registra decisões e limitações metodológicas identificadas em
auditoria (2026-07-07), no formato: **problema → impacto científico →
recomendação → classificação** (correção segura e já aplicada vs. decisão que
exige validação manual do pesquisador responsável).

---

## 1. Vazamento do conjunto de teste para a seleção de modelo — **requer decisão manual, não alterado**

**Problema:** em `dr_hcpa_v2_2024.py`, `ModelCheckpoint(monitor='val_loss',
save_best_only=True, ...)` é alimentado com `get_training_dataset(VALID_FILENAMES)`,
onde `VALID_FILENAMES = TFREC_DIR + '/test*.tfrec'` — ou seja, o conjunto
chamado de "teste" é usado tanto para escolher o melhor checkpoint durante o
treino quanto para a avaliação final em `script3_avalia.py`. Não existe um
terceiro conjunto de validação verdadeiramente isolado.

**Impacto científico:** a seleção do "melhor" checkpoint já é informada pelo
próprio conjunto de teste, então a métrica final relatada pode estar
otimisticamente enviesada. Mais importante para este estudo: se diferentes
filtros de pré-processamento se beneficiarem *desigualmente* desse
vazamento (por exemplo, um filtro que produz uma curva de loss mais ruidosa
no "teste" pode ter seu checkpoint escolhido em um ponto de sorte estatística
diferente de outro filtro mais estável), a comparação **entre filtros** deixa
de ser justa — o núcleo científico do estudo.

**Recomendação:** duas opções, ambas exigem tocar no desenho de split que o
usuário definiu explicitamente como 70/30 train/test (sem terceira fatia):
  - (a) manter como está e reportar essa limitação explicitamente na seção de
    métodos do artigo (é uma limitação comum e aceitável em muitos estudos
    com dataset pequeno, desde que declarada); ou
  - (b) reservar uma fatia de validação a partir do treino (ex.: 70% → ~55%
    treino + 15% validação, mantendo os 30% de teste intocados) e apontar
    `monitor='val_loss'` para essa validação real.

**Classificação:** requer decisão do pesquisador — **não foi alterado nesta
rodada** (mexeria no requisito de split 70/30 definido no início do projeto).

**Decisão (2026-10-01):** opção (a). O estudo mantém apenas treino e teste
(70/30), sem conjunto de validação separado, inclusive na conclusão das
condições pendentes (item 13), para que todas as 576 condições sejam
comparáveis entre si. O uso do teste como `validation_data` (early stopping
e ReduceLROnPlateau, `dr_hcpa_v2_2024.py`) deve ser declarado como
limitação na seção de métodos de qualquer artigo derivado.

---

## 2. Threshold fixo (0.5) para binarizar o heatmap do Grad-CAM em Dice/IoU — documentado, seguro

**Problema:** `utils_metrics.dice_coefficient()`/`iou_score()` binarizam o
heatmap de saliência (contínuo, normalizado [0,1]) com um corte fixo em 0.5,
sem calibração por filtro ou por imagem.

**Impacto científico:** filtros que sistematicamente produzem heatmaps com
distribuição de magnitude diferente (mais ou menos "espalhados") podem ter
Dice/IoU artificialmente mais altos ou mais baixos só por causa do threshold,
não por alinhamento anatômico real melhor ou pior.

**Recomendação:** para publicação, considerar reportar também Dice/IoU
integrados sobre múltiplos thresholds (área sob a curva, análogo a
AUC-ROC) como métrica complementar em trabalho futuro.

**Classificação:** segura — é uma limitação a documentar, não exige mudança
de código para esta rodada de experimentos.

---

## 3. Pointing Game Accuracy com lesões múltiplas e espalhadas — limitação conhecida do método

**Problema:** `pointing_game_accuracy()` verifica apenas se o pixel de argmax
do heatmap cai dentro de qualquer máscara de lesão — um único ponto por
imagem. Retinopatia diabética tipicamente produz múltiplas lesões pequenas e
espalhadas (microaneurismas, hemorragias, exsudatos); um Grad-CAM
corretamente ativado sobre várias lesões, mas cujo pico global caia a poucos
pixels de uma delas, é penalizado como "erro" total.

**Impacto científico:** a métrica pode subestimar sistematicamente a
qualidade da localização em imagens com muitas lesões pequenas — mas essa é
uma limitação conhecida e amplamente aceita do próprio método "Pointing
Game" na literatura de XAI, não um bug de implementação deste projeto.

**Recomendação:** interpretar Pointing Game Accuracy em conjunto com
Dice/IoU (que já capturam sobreposição de área, não só um ponto), e declarar
a limitação na seção de métodos.

**Classificação:** documentação apenas; nenhuma mudança de código necessária.

---

## 4. Estratificação do split por grade ICDR (5 classes), não pela classe binária diretamente

**Contexto:** `make_split.py` estratifica o `train_test_split` pela grade
ICDR de 5 classes (0-4), não diretamente pela classe binária derivada
(referable/non-referable). Isso é **estritamente mais forte**, não uma
inconsistência: se cada estrato de grade é dividido proporcionalmente
70/30, a união dos estratos (os dois grupos binários) também é — é um caso
particular garantido. `split_metadata.json` agora registra explicitamente
`binary_label_distribution` para tornar esse balanceamento auditável sem
precisar re-derivar a partir da distribuição de 5 classes.

**Classificação:** sem ação necessária além da documentação já adicionada.

---

## 5. Reprodutibilidade da seed — **corrigido nesta rodada**

**Problema (encontrado nesta auditoria):** `dr_hcpa_v2_2024.py` nunca chamava
`tf.random.set_seed()`/`np.random.seed()`/`random.seed()`. `script2_treina.py`
só setava as variáveis de ambiente `PYTHONHASHSEED` e `TF_DETERMINISTIC_OPS=1`,
que não fixam a inicialização de pesos nem a ordem de embaralhamento do
`dataset.shuffle(2048)` usados no treino. Ou seja: as "10 repetições com seed
0-9" documentadas no checklist original do projeto não eram, na prática,
reproduzíveis nem controladas — cada rerun do mesmo `--exec N` produziria
pesos iniciais e ordem de dados diferentes.

**Impacto científico:** compromete diretamente a reprodutibilidade exigida
pelos critérios de aceitação do projeto, e torna a variância entre as 10
repetições parcialmente não-controlada (mistura variância genuína de
inicialização com não-determinismo de execução, não separáveis a posteriori).

**Correção aplicada:** `dr_hcpa_v2_2024.py` agora aceita `--seed` e chama
`random.seed()`, `np.random.seed()`, `tf.random.set_seed()` logo após o
parse de argumentos. `script2_treina.py` passa `--seed str(repeticao)` ao
comando (antes só setava env vars, que permanecem como reforço adicional de
determinismo de operações, não como mecanismo de seed).

**Classificação:** correção segura, sem custo científico (só faz o que já
estava documentado como requisito funcionar de verdade) — **aplicada**.

---

## 6. `BenGrahamNormalization` não implementa o método real de Ben Graham — documentado, não corrigido

**Problema:** a classe `BenGrahamNormalization` em `utils_preprocessing.py`
implementa normalização por subtração da média de cor e divisão pelo
desvio-padrão. O método real de Ben Graham (popularizado na competição
Kaggle de retinopatia diabética) é diferente: borra a imagem com um filtro
Gaussiano e subtrai o borrão da imagem original para realçar bordas/textura
(`I' = αI + βG(θ)*I + γ`).

**Impacto científico:** baixo para os experimentos atuais — essa classe não
está na matriz de filtros curada (Tarefa 3) — mas o nome induz a erro
qualquer uso futuro que espere replicar a técnica clássica da literatura.

**Recomendação:** renomear a classe (ex. `MeanStdColorNormalization`) ou
reimplementar para bater com o método real, antes de qualquer uso científico
que cite "Ben Graham normalization" como referência.

**Classificação:** documentado apenas; fora do escopo desta rodada (classe
não usada na matriz de filtros ativa).

---

## 7. [SUPERADO por item 10] `batch_size=256` com backbone InceptionV3 totalmente treinável — risco não testável neste ambiente

**Problema:** `script2_treina.py` fixa `--batch_size 256`. O backbone
InceptionV3 (21.8M parâmetros) está, na prática, totalmente treinável (ver
item 8) — batch 256 com fine-tuning completo em 299×299 é uma combinação
pesada de memória de ativações, mais do que os batch sizes tipicamente
usados na literatura para esse cenário (comumente 32-64 numa GPU de
24-48GB). Este ambiente de desenvolvimento não tem GPU, então não foi
possível validar empiricamente se isso cabe numa L40S (48GB) real.

**Recomendação:** antes de submeter a matriz completa (120 jobs), rodar 1
repetição de 1 filtro no cluster real e observar `nvidia-smi` durante o
treino. Se houver OOM, reduzir `--batch_size` (script2_treina.py) é a
correção mais direta.

**Classificação:** requer validação manual no hardware real — não é algo que
eu possa corrigir corretamente sem acesso à GPU de produção.

---

## 8. [SUPERADO por item 10] Backbone InceptionV3 efetivamente não congelado — informativo, decisão pré-existente

**Confirmado (sessão anterior, reafirmado aqui):** em `build_model()`, o
padrão "congela tudo em `base.layers`, depois descongela via `model.layers`"
resulta em fine-tuning completo do backbone (treinável: 21.770.401 de
21.804.833 parâmetros — só as estatísticas não-treináveis de BatchNorm ficam
de fora), não em linear probing como a estrutura do código sugere à
primeira vista. Não é um bug introduzido por esta auditoria; é um
comportamento pré-existente do script "existente" que o projeto pede para
preservar. Mantido como está — mas vale confirmar com quem desenhou o
experimento se fine-tuning completo é realmente a intenção (afeta tempo de
treino, risco de overfitting com apenas ~1300 imagens de treino, e
comparabilidade com qualquer trabalho relacionado que use linear probing).

**Classificação:** informativo — decisão de modelagem pré-existente, não
alterada.

---

## 9. [SUPERADO por item 10] Checkpoint salvo era da última época, não da melhor — corrigido (auditoria operacional, 2026-07-07)

**Nota (2026-07-07, refatoração de arquitetura):** o mecanismo descrito abaixo
(`ModelCheckpoint` + `model.load_weights()` manual) foi removido e substituído
por `EarlyStopping(restore_best_weights=True)` — ver item 10. O problema e a
correção aqui descritos continuam corretos como registro histórico do que foi
encontrado e corrigido antes da troca de arquitetura, mas não descrevem mais o
mecanismo atual de seleção do melhor checkpoint.

**Problema (encontrado na auditoria operacional):** em `dr_hcpa_v2_2024.py`,
o callback `sv = ModelCheckpoint(..., monitor='val_loss', save_best_only=True,
save_weights_only=True, ...)` salva os PESOS da melhor época em um arquivo
`.weights.h5` separado durante o treino. Como não há early stopping (o
treino sempre roda até `--epochs` completo), e o `model.save(...)` final
usava o estado do modelo em memória logo após `model.fit()` terminar (ou
seja, a ÚLTIMA época), o checkpoint `.keras` realmente avaliado por
`script3_avalia.py` nunca correspondia ao "melhor" val_loss — era
simplesmente o que sobrou da última época, potencialmente pior por
overfitting.

**Impacto científico:** todas as métricas clínicas e XAI relatadas
poderiam estar sistematicamente subestimando a capacidade real de cada
filtro, e de forma não uniforme entre filtros (um filtro cuja curva de
val_loss piora mais nas épocas finais seria penalizado mais que outro mais
estável) — comprometendo a comparabilidade entre condições experimentais,
exatamente o tipo de viés que este estudo busca evitar.

**Correção aplicada:** logo após `model.fit()` e antes do `model.save()`
final, o script agora recarrega os pesos salvos em `.weights.h5`
(`model.load_weights(best_weights_path)`), garantindo que o `.keras`
persistido — e portanto tudo que `script3_avalia.py` avalia — reflita
realmente a época de menor `val_loss`.

**Classificação:** correção segura (restaura o comportamento que o próprio
código já sinalizava pretender ter via `save_best_only=True`, sem mudar
desenho experimental) — **aplicada** (posteriormente substituída, ver item 10).

---

## 10. Substituição de arquitetura: InceptionV3 → EfficientNetV2S (2026-07-07)

**Contexto:** a arquitetura de backbone e a metodologia de treino foram
substituídas por completo, replicando Kao & Lin (2024) adaptado para o
escopo binário deste projeto (referável DR) e para o tamanho de entrada
nativo do pipeline (299×299, não 256×256 como no artigo original).

**O que mudou (`pipeline/dr_hcpa_v2_2024.py`):**
- Backbone: `InceptionV3` → `EfficientNetV2S` (ambos `weights='imagenet'`,
  `include_top=False`), ~20.3M parâmetros (substitui os ~21.8M do InceptionV3
  citados nos itens 7/8 acima — os números desses itens não se aplicam mais).
- Topo: `GAP -> Dense(1, sigmoid)` (linear probing raso) substituído por
  `GAP -> Dropout(0.5) -> Dense(512, elu) -> Dropout(0.5) -> Dense(1,
  sigmoid)`.
- Treino: uma única fase de "fine-tuning completo desde o início" (item 8)
  substituída por duas fases explícitas — Fase 1 (backbone congelado,
  5 épocas, Adam lr=0.001, sem label smoothing) e Fase 2 (backbone
  inteiramente treinável, até 50 épocas, Adam lr=0.00002,
  `BinaryCrossentropy(label_smoothing=0.2)`). Isso resolve o item 8 (que
  apontava fine-tuning completo "acidental" desde a primeira época) —
  agora o warm-up do topo antes do fine-tuning é uma fase deliberada e
  documentada.
- `batch_size` fixo passou de 256 (item 7, nunca validado em GPU real) para
  16 (valor do artigo replicado) — reduz drasticamente o risco de OOM citado
  no item 7.
- Seleção do melhor checkpoint: `ModelCheckpoint(save_best_only=True)` +
  reload manual (item 9) substituído por
  `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
  + `ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)` — a
  restauração do melhor estado agora acontece via callback padrão do Keras,
  não por um reload manual de arquivo de pesos.
- Pré-processamento de entrada: sem alteração de comportamento — os pixels já
  eram passados em escala bruta [0, 255] (sem `/255.0`) para o InceptionV3
  (o que já era metodologicamente questionável para aquele backbone, que
  espera `[-1, 1]`); para EfficientNetV2S essa é na verdade a entrada
  **correta**, pois o modelo Keras já inclui uma camada interna de
  Rescaling/Normalization. Foi corrigido, porém, um ponto de inconsistência
  em `script3_avalia.py`, que dividia por 255.0 antes da inferência — isso
  teria feito o modelo receber pixels duplamente normalizados (fora da
  distribuição vista no treino) na hora da avaliação.

**Impacto na matriz experimental (`experiments/filter_matrix_template.json`):**
nenhum — a matriz de filtros de pré-processamento de imagem é ortogonal à
arquitetura do classificador; nenhuma condição precisou ser refeita ou
redesenhada por causa desta troca.

**Validação realizada:** análise estática (leitura de código) e checagem
sintática (`ast.parse`) de todos os arquivos alterados. **Não validado**:
nenhum treino real (GPU) foi executado nesta sessão para confirmar
convergência, tempo por época ou ausência de OOM com o novo backbone —
recomenda-se pilotar 1 filtro × 1 repetição no cluster real antes de
submeter a matriz completa.

**Classificação:** mudança solicitada explicitamente pelo usuário — não uma
correção de auditoria.

---

## 11. Auditoria de infraestrutura de produção e correções (2026-07-08)

Validação completa em condições reais de produção (SLURM real via `sbatch`,
SSD local dos nós `grace1`/`grace2`, GPU L40S). Achados e correções:

1. **`--bind` do `SSD_BASE` ausente em todos os templates SLURM e no
   orquestrador** — confirmado empiricamente que `/ssd` não é auto-montado
   pelo `singularity.conf` deste cluster. Corrigido em
   `slurm/job_template_*.sh` (4 arquivos) e `pipeline/script5_orquestrador.py`
   (geração de scripts + modo `salloc`). **Classificação:** correção
   segura, aplicada e validada com execução real.
2. **Dataset bruto precisa estar replicado em CADA nó do round-robin** —
   `/ssd` é local por nó; um filtro pinado em `grace1` falha se só `grace2`
   tiver `data/Seg-set`. Corrigido preparando ambos os nós (ver
   `RUN_EXPERIMENTS.md` seção 2). **Classificação:** achado operacional,
   documentado como pré-requisito permanente.
3. **Distribuição por índice na lista, não pelo nome do filtro** — corrigido
   com hash `sha256` estável do nome do filtro (`_stable_node_for_filter`),
   validado com os 18 filtros reais da matriz (mapeamento idêntico sob
   reordenação e subconjunto, distribuição 9/9 entre os 2 nós).
   **Classificação:** correção segura, aplicada e validada.
4. **Checkpoints (~250MB × 180 execuções ≈ 45GB) nunca eram removidos** —
   implementado `delete_checkpoint_after_success()` em `script3_avalia.py`,
   chamado só após CSV validado + success marker criados; idempotência de
   `script2_treina.py` ajustada para reconhecer o success marker como sinal
   de "já concluído" (antes dependia só da existência do checkpoint).
   **Classificação:** correção solicitada explicitamente, aplicada e
   validada (240.9MB liberados em execução real).
5. **Dois crashes de baixo nível (`std::bad_function_call`/SIGABRT e
   SIGBUS) no treino real, não relacionados à migração de arquitetura** —
   descobertos durante a revalidação: (a) o otimizador `layout_optimizer`
   do grappler falhava de forma determinística no primeiro passo de
   treino contra uma op de dropout do EfficientNetV2S nesta combinação de
   GPU/driver/TF 2.13, corrigido desabilitando só esse optimizer
   (`tf.config.optimizer.set_experimental_options({"layout_optimizer": False})`);
   (b) um bloco de diagnóstico pós-treino (reavaliação final + curva ROC/
   thresholds/PDF) nunca lido por `script3_avalia.py` ou qualquer outra
   etapa passou a travar o processo inteiro com um sinal de SO (não
   capturável por `try/except` em Python) — removido por ser não-essencial
   e a única fonte identificada do crash. **Classificação:** correções de
   infraestrutura/confiabilidade, não alteram arquitetura, hiperparâmetros
   ou métricas — validadas com 2 execuções reais completas e sem crash após
   ambas as correções.
6. **Código morto e imports não utilizados** removidos em todo `pipeline/`
   (ver `RUN_EXPERIMENTS.md` não cobre isso — puramente cosmético/seguro,
   sem efeito em comportamento): `Path` (create-tfrecord.py), `rmtree`/
   `rename`/`listdir`/`join`/`exists`/`isfile` (dr_hcpa_v2_2024.py, também
   `pd`/`plt` após remoção do bloco de ROC/thresholds/PDF, e as funções
   mortas `sensitivity()`/`specificity()`/`get_valid_dataset()`/
   `get_test_dataset()`/`read_unlabeled_tfrecord()`/`dataset_to_numpy_util()`),
   `numpy` (script1_cria_dataset.py), `load_split_json` (script3_avalia.py),
   `tempfile` (script4_deleta_dataset.py), `time`/`datetime`/`Tuple`/
   `check_success_marker` (script5_orquestrador.py), `Tuple`/`Any`/
   `confusion_matrix` (utils_metrics.py). **Classificação:** correção
   segura, verificada por `ast`-based unused-import check.
7. **`run_clinical.sh`/`run_clinical_array.sh`/`submit_all_clinical_jobs.sh`
   revisados**: confirmados corretamente arquivados em `legacy/`,
   incompatíveis por design com o pipeline atual (caminhos/estrutura
   pré-reorganização), sem nenhuma referência ativa em `pipeline/`/`slurm/`.
   Nenhuma ação necessária.

---

## 12. Auditoria de robustez pré-campanha (armazenamento, checkpoints, falha de nós, execução longa) (2026-07-09)

Auditoria solicitada explicitamente pelo usuário antes de disparar a
campanha completa (576 filtros x 10 repetições = 5760 execuções). Achados e
correções:

### 12.1 Checkpoint órfão real encontrado e removido

Durante a auditoria, `/ssd/aadsilva/ic/hcpa/checkpoints/Gamma0.8-0.keras`
(252.5MB) foi encontrado em `grace1`, apesar do success marker
`Gamma0.8_0.ok` já existir (ciência já coletada em `~/resultados/`) — ou
seja, um checkpoint que deveria ter sido apagado por
`delete_checkpoint_after_success()` mas não foi (piloto rodou com uma versão
de código anterior, antes dessa função existir/funcionar). Isso confirma
concretamente o risco: **uma vez que o success marker existe, nada revisita
aquele filtro+repetição de novo** — se a deleção falhar por qualquer motivo
(crash entre `model.save()` e a deleção, versão antiga de código, etc.), o
checkpoint fica ali para sempre.

**Correção implementada:** `sweep_orphaned_checkpoints()` em
`script3_avalia.py`, chamada ao final de toda avaliação bem-sucedida — varre
o diretório de checkpoints do nó atrás de QUALQUER `.keras` cujo
filtro+repetição já tenha success marker, e apaga (nunca lança exceção,
best-effort). Como roda a cada uma das ~5760 avaliações, qualquer órfão
deixado por qualquer causa em qualquer nó é varrido na próxima vez que
QUALQUER avaliação suceder naquele nó — janela de vazamento limitada, não
mais indefinida. Testado com casos sintéticos (órfão com marker → apagado;
checkpoint em andamento sem marker → preservado; nome malformado →
ignorado com segurança). O órfão real de 252.5MB foi removido manualmente
como parte desta auditoria.

**Classificação:** correção de robustez, aplicada e testada.

### 12.2 Resiliência do daemon a exceções não capturadas

O laço principal de `run_daemon()` não tinha proteção própria: uma exceção
não capturada em `check_node_health_and_migrate()` ou `write_progress_report()`
(ex.: falha transitória de I/O no NFS, disco cheio por um instante) mataria
o processo inteiro — diferente do caminho de auto-resubmissão perto do
limite de 24h, que só age perto do fim do tempo, não em caso de crash.
Para uma campanha de vários dias rodando sem supervisão, isso é um ponto
único de falha real.

**Correção implementada:** todo o corpo do laço (checagem de saúde de nós,
processamento de todos os filtros, escrita de relatório de progresso,
checagem de conclusão) agora está dentro de um `try/except` que loga o erro
e tenta de novo no próximo ciclo, em vez de propagar e matar o daemon.

**Classificação:** correção de robustez, aplicada.

### 12.3 Limite de filtros "em voo" simultâneos (risco de acúmulo de SSD)

Script 1 (criação de dataset) é submetido deliberadamente SEM `--nodelist`
(deixa o SLURM escolher o nó livre). Isso significa que, no pior caso de
agendamento, vários filtros podem ficar pinados (dataset já criado) no MESMO
nó antes de qualquer um deles terminar as 10 repetições + limpeza — cada um
ocupando ~6GB (dataset + TFRecords) no SSD local daquele nó até ser limpo.
Com apenas 2 nós físicos isso tende a se autolimitar na prática, mas nada
impedia estruturalmente um acúmulo maior.

**Correção implementada:** `MAX_INFLIGHT_FILTERS = 6` — antes de submeter
Script 1 para um novo filtro, o orquestrador conta quantos filtros já estão
"em voo" (pinados a um nó, mas ainda não concluídos+limpos nem
permanentemente falhos) via `count_inflight_filters()`; acima do limite, a
submissão é adiada para o próximo ciclo. Isso limita o pior caso de uso de
SSD a `MAX_INFLIGHT_FILTERS x ~6GB ≈ 36GB` por nó, independentemente de
como o SLURM decida agendar — folga enorme frente aos ~2.3-2.5TB livres
observados por nó.

**Classificação:** correção de robustez, aplicada e testada com casos
sintéticos (contagem correta antes/depois de marcar um filtro como
permanentemente falho).

### 12.4 Visibilidade de espaço em disco no relatório de progresso

O processo despachante roda na partição `shared`, não em `grace1`/`grace2` —
não consegue ver `$SSD_BASE` (scratch local de nó) diretamente. Já conseguia
ver `$HOME` (NFS), mas não expunha isso no relatório de progresso.

**Correção implementada:** `write_progress_report()` agora inclui uso de
disco de `$HOME` (via `shutil.disk_usage`, uma única chamada `statvfs`, não
uma varredura de diretório — `trash_datasets/` sozinho pode chegar à ordem
de TB ao longo da campanha, então somar tamanhos de diretório a cada ciclo
de 60s seria um problema de desempenho por si só).

**Classificação:** melhoria de observabilidade, aplicada.

### 12.5 Estimativa de armazenamento para a campanha completa (576 x 10 = 5760 execuções)

**SSD (local por nó, transitório):**

| Item | Tamanho | Observação |
|---|---|---|
| Dataset filtrado por filtro | ~4.2-5.8GB (medido: baseline=4.2GB, AHE40.0=5.8GB) | Existe só enquanto o filtro está "em voo" |
| TFRecords por filtro | ~250-341MB (medido) | Idem |
| Checkpoint `.keras` por repetição | ~250MB | Deletado automaticamente após cada avaliação (script3) — nunca mais de 1 por vez por nó em operação normal |
| **Pico por nó (com `MAX_INFLIGHT_FILTERS=6`)** | **~36GB** | Limite estrutural agora garantido pela correção 12.3 |
| Capacidade disponível medida | 2.3-2.5TB livres por nó (`/ssd`, 3.5TB total) | Folga de ~65x sobre o pico teórico — **sem risco de esgotamento** |

**HOME (NFS, cumulativo, nunca limpo automaticamente):**

| Item | Tamanho estimado (576x10) | Observação |
|---|---|---|
| CSVs de métricas (`~/resultados/*.csv`) | ~4MB | Permanente (é o resultado científico) |
| `training_metadata_*.json` + CSVs de fase 1/2 | ~58MB | Permanente, pequeno, útil para auditoria |
| Logs de job SLURM de treino+avaliação (`logs/*.log`,`*.err`) | ~8.6GB (medido: ~1.5MB por job de treino real de 55 épocas) | Cresce, nunca é limpo automaticamente, mas modesto |
| Logs de job SLURM de dataset/tfrecord/limpeza | ~0.1GB | Pequeno |
| `~/trash_datasets/` (soft-delete do Script 4) | **~3.46TB** (576 x ~6GB) | **Cresce indefinidamente — nunca purgado automaticamente por design (é a rede de segurança do soft-delete)** |
| **TOTAL projetado em `$HOME`** | **~3.5TB** | Frente a 34TB livres medidos (73TB total, 55% usado) — **folga de ~10x, sem risco** |

**Grad-CAM/XAI:** calculado inteiramente em memória (`utils_gradcam.py`
retorna um array numpy, nunca grava em disco) — confirmado por leitura de
código, sem crescimento de armazenamento associado.

**Recomendação (não implementada — decisão do usuário):** `trash_datasets/`
crescerá ~3.46TB ao longo da campanha e nunca é purgado automaticamente
(esse é o propósito do soft-delete: permitir recuperação manual). Como a
folga de espaço é confortável (~10x), nenhuma purga automática foi
implementada — mas recomenda-se que o usuário revise e apague manualmente
`~/trash_datasets/` periodicamente (ex.: após confirmar que os resultados de
um lote de filtros já foram copiados/analisados), em vez de deixar por
meses.

### 12.6 Robustez contra falha de nós — revisão (nenhuma correção necessária, já implementado)

Revisão de `script5_orquestrador.py` confirma que os seguintes cenários já
são tratados corretamente pelo design existente (ver docstring do módulo e
`check_node_health_and_migrate()`, `process_filter()`):

- **Nó DOWN/DRAIN/FAIL/MAINT:** detectado via `sinfo`; após
  `NODE_DOWN_TIMEOUT_DEFAULT` (1800s) contínuo nesse estado, os jobs do(s)
  filtro(s) associados são cancelados (`scancel`) e o marcador de nó é
  limpo — o filtro é automaticamente resubmetido sem pino na próxima
  iteração (o SLURM escolhe outro nó livre). **Nenhuma espera indefinida por
  um nó específico.**
- **Nó ALLOCATED por horas:** esperado/normal (treino real leva tempo); o
  próprio limite de tempo do job SLURM (`--time=10:00:00` no template de
  treino) garante que um job realmente travado seja morto pelo SLURM, e a
  ausência de evento SUCCESS/FAILED correspondente no `master.log` após o
  job sair da fila é detectada como "possível crash" e marcada como falha
  isolada (não trava o restante da campanha).
- **Perda temporária de comunicação (squeue/sinfo falhando):** todas as
  funções de introspecção (`squeue_job_names`, `squeue_busy_nodes`,
  `sinfo_node_states`) já capturam exceções e degradam para conjunto/dict
  vazio, sem derrubar o daemon (agora reforçado pelo `try/except` do laço
  principal, item 12.2).
- **Falha de uma execução específica:** isolada em `failed_jobs.json`
  (`filters`/`repetitions`/`cleanup` separados) — não impede as demais
  repetições do mesmo filtro nem outros filtros.
- **Cancelamento de job:** um job cancelado sai da fila sem gerar evento
  SUCCESS/FAILED — detectado e tratado como falha (mesmo mecanismo do
  "possível crash" acima).
- **Reinício do orquestrador:** todo o estado de reconciliação vive em
  disco (success markers, `master.log`, `node_assignments/`,
  `failed_jobs.json`) — nenhum estado apenas em memória. Reiniciar o daemon
  (manual ou via auto-resubmissão perto do limite de 24h) continua
  exatamente de onde parou.
- **Duas execuções do mesmo experimento simultaneamente:** improvável —
  `--exclusive` reserva o nó inteiro por job; pino de nó (`--nodelist`)
  garante que todas as etapas de um filtro rodem no mesmo nó; checagem de
  instância única do próprio daemon (via `squeue -n hcpa_orchestrator`)
  impede dois despachantes concorrentes; `queued_names` é reconsultado a
  cada ciclo antes de qualquer submissão.
- **Risco de um experimento nunca rodar:** o único cenário residual
  identificado é o daemon inteiro morrer sem conseguir se auto-resubmeter
  (ex.: `sbatch` falha no momento exato da resubmissão perto do limite de
  24h) — já era logado como erro antes desta auditoria; **mitigado, não
  eliminado**, pelo item 12.2 (try/except no laço principal reduz
  drasticamente a chance de morte por exceção não relacionada ao limite de
  tempo). Recomenda-se checar `logs/hcpa_orchestrator_*.log` periodicamente
  nos primeiros dias da campanha para confirmar que a auto-resubmissão está
  de fato ocorrendo.

**Classificação:** revisão confirma design já robusto; nenhuma correção
adicional necessária além dos itens 12.1-12.4.

---

## 13. Conclusão das condições pendentes da varredura de 576 (2026-10-01)

**Estado de partida:** 495 das 576 condições com 10/10 repetições; 80 com
2 a 9 repetições e 1 (`Retinex_MaxGreen2.0_Otsu`) com nenhuma — 194
repetições faltando no total.

**Causa das lacunas:** quase todas são repetições cujo treino caiu 3 vezes
seguidas e foram marcadas como desistidas em `failed_jobs.json`. Os crashes
são de baixo nível e intermitentes, não determinísticos: códigos de saída
-6/SIGABRT (990 ocorrências), -7/SIGBUS (415) e -11/SIGSEGV (212), em geral
nos primeiros segundos do processo, em ambos os nós; 22% de todas as
tentativas de treino da campanha original falharam assim, e a maioria
passou na tentativa seguinte. A exceção é `Retinex_MaxGreen2.0_Otsu`, cujo
Script 1 estourou o limite de 2h (Retinex em 1280×1280 é lento) com o
conjunto de teste pela metade (176/553).

**O que foi feito:**
- `failed_jobs.json` copiado para `failed_jobs.backup_20261001.json`;
  removidas dele as entradas `repetitions`/`filters` das 81 condições
  pendentes (entradas de `cleanup` e de outras condições mantidas).
- Nova opção `--attempts-since` em `script5_orquestrador.py`: o limite de
  `MAX_TRAIN_ATTEMPTS` (3) passa a contar só lançamentos registrados no
  `master.log` a partir dessa data, dando um novo orçamento de 3 tentativas
  sem reescrever o histórico.
- `experiments/filter_matrix_pending.json`: só as 81 condições pendentes.
- `run_complete_pending.sh`: mesmo dispatcher e mesmos padrões da campanha
  principal (reps 0-9, só Grad-CAM, sem subamostragem, checkpoint apagado
  após a avaliação), com a matriz pendente e `--attempts-since`.
- `Retinex_MaxGreen2.0_Otsu`: Script 1 ressubmetido à mão com `--force`
  (dataset parcial) e limite de 4h, pelo próprio gerador de scripts do
  orquestrador, para que os eventos STARTED/SUCCESS sejam registrados no
  `master.log` normalmente.

**Comparabilidade com as 495 condições já completas:** as repetições novas
reutilizam as mesmas imagens filtradas e TFRecords já gerados (exceto a
condição acima), a mesma configuração de treino e as mesmas seeds. As
alterações não commitadas em `pipeline/` presentes desde a campanha
original (vetorização do Frangi, suporte a reps 10-19, LIME/Occlusion
opcionais) não mudam o comportamento padrão. Diferença cosmética: os CSVs
novos de `script3_avalia.py` trazem a coluna extra `xai_method`
(= `gradcam`), tratada na análise.

**Desfecho (2026-10-02 22:22):** as 194 repetições pendentes foram
concluídas; 576/576 condições com 10/10 repetições (5.760 execuções), sem
nenhuma desistência. Das 225 tentativas de treino desta retomada, 31 (14%)
caíram pelo mesmo crash intermitente e passaram ao serem repetidas.
Análise final e relatório: `experiments/analise_final_576/RELATORIO.md`.

**Correções operacionais feitas durante a retomada:**
- **Nós presos a um nó ocupado por outro usuário.** O grace1 ficou parado
  por horas enquanto os 2 slots da campanha esperavam o grace2, ocupado por
  outro usuário. Como `SSD_BASE` resolve para o `/home` compartilhado (os
  dados são visíveis dos dois nós), as 53 condições da matriz pendente
  atribuídas ao grace2 em `experiments/node_assignments/` foram reatribuídas
  ao grace1 (2026-10-02 14:15). Nenhum dado foi movido.
- **Novas tentativas mandadas para um nó ocupado.** `script5_orquestrador.py`
  sempre enviava a nova tentativa de um treino para o outro nó
  (`other_grace_node`), mesmo quando ele estava ocupado por outro usuário,
  deixando a repetição presa por horas. Agora a nova tentativa só vai para
  o outro nó se ele não estiver rodando job de outro usuário
  (`nodes_held_by_other_users`); se estiver, fica no nó original.
- **Análise automática.** `run_final_analysis.sh` (partição `shared`)
  esperou a conclusão, rodou `experiments/analise_final_576/analise.py` e
  avisou por e-mail e em `logs/ALERTAS_PENDENTES.txt`.
- `run_complete_pending.sh` e `run_final_analysis.sh` excluem o nó `bali2`
  da partição `shared`, onde o primeiro lançamento do orquestrador falhou
  (`launch_failed_requeued_held`).
