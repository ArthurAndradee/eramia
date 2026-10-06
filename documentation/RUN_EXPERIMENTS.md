# Guia Operacional — Execução dos Experimentos (576 condições × 10 = 5.760 execuções)

> Documento definitivo para operar o pipeline em produção no cluster.
> Atualizado em 2026-07-08: ponto de entrada único (`run_all_experiments.sh`)
> — um `sbatch` inicia um daemon despachante que roda a matriz inteira sem
> nenhum comando manual adicional. Para detalhes de arquitetura do pipeline
> em si, ver `README_PIPELINE.md`; para limitações metodológicas, ver
> `METHODOLOGICAL_NOTES.md`.

---

## 1. Preparação

### 1.1 Acessar o cluster

A partir do nó de login (ex. `tsubasa`), a partir da raiz do repositório:

```bash
cd ~/ic/hcpa-retinopathy/hcpa-repository/hcpa
```

Você **não precisa** entrar em `grace1`/`grace2` para rodar os experimentos
— o `sbatch` do passo 3 é suficiente a partir daqui. `ssh grace1`/`grace2`
só é necessário para depuração manual pontual, e só funciona se você tiver
uma alocação SLURM ativa naquele nó (`pam_slurm_adopt`):

```bash
ssh grace2   # requer job ativo lá; para debug manual apenas
```

### 1.2 Verificar GPU

```bash
sinfo -p grace                     # deve mostrar grace1 e grace2 "idle" ou "alloc"
squeue -u $USER                    # jobs seus em andamento
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv   # dentro de um job/salloc ativo
```

Cada nó (`grace1`, `grace2`) tem 1x NVIDIA L40S (46GB). Se `memory.used` não
estiver em ~0 MiB antes de um job começar, outro processo pode estar preso.

### 1.3 Verificar SSD local e espaço disponível

```bash
df -h /ssd
```

Cada nó tem seu **próprio SSD local** (`/ssd/...`, não compartilhado entre
nós, não compartilhado com o `$HOME` em NFS). Confirme que há pelo menos
~20GB livres por nó antes de iniciar um lote grande (ver seção 6).

---

## 2. Preparação do SSD

### 2.1 Estrutura obrigatória por nó

Todo nó que o SLURM possa escolher (`grace1` e `grace2`) precisa ter,
**localmente**, sob `/ssd/aadsilva/ic/hcpa/` (o valor padrão de `SSD_BASE`):

```
/ssd/aadsilva/ic/hcpa/
├── data/
│   └── Seg-set/                       # dataset bruto FGADR (copiado do NFS; ~4.2GB)
│       ├── Original_Images/
│       ├── DR_Seg_Grading_Label.csv
│       └── *_Masks/                   # 6 diretórios de máscaras de lesão
├── experiments/
│   ├── split.json                     # split fixo train/test (copiado do NFS)
│   ├── split_metadata.json
│   └── filter_matrix_template.json
├── datasets_filtrados/                 # gerado pelo Script 1 (por filtro)
├── data/tfrecords_filtrados/           # gerado pelo create-tfrecord.py (por filtro)
├── checkpoints/                        # gerado pelo Script 2 (removido pelo Script 3 após avaliação — ver seção 6)
└── locks/                              # locks de concorrência por filtro (fcntl)
```

**Por que isso é necessário em CADA nó**: `/ssd` é armazenamento local do nó,
não uma montagem de rede compartilhada. O Script 1 de cada filtro é
submetido **sem** `--nodelist` — o próprio SLURM escolhe `grace1` ou
`grace2` conforme disponibilidade no momento (ver seção 4.4) — então
qualquer um dos dois pode ser escolhido, e ambos precisam ter `data/Seg-set`
e `experiments/` localmente, ou o Script 1 falha com
`Required path does not exist`.

### 2.2 Como copiar o dataset para um nó novo (ou node recém-provisionado)

A partir de uma sessão com job ativo NO nó de destino (ou via um job SLURM
avulso), rodar uma única vez:

```bash
SSD_BASE=/ssd/aadsilva/ic/hcpa
NFS_REPO=$HOME/ic/hcpa-retinopathy/hcpa-repository/hcpa

mkdir -p "$SSD_BASE/data" "$SSD_BASE/experiments"
rsync -a "$NFS_REPO/data/Seg-set/" "$SSD_BASE/data/Seg-set/"
cp "$NFS_REPO/experiments/split.json" \
   "$NFS_REPO/experiments/split_metadata.json" \
   "$NFS_REPO/experiments/filter_matrix_template.json" \
   "$SSD_BASE/experiments/"
```

Confirme com:

```bash
find "$SSD_BASE/data/Seg-set/Original_Images" -type f | wc -l   # deve ser 1842
```

**Atenção**: se `/ssd/aadsilva/ic/hcpa` já existir com conteúdo de uma
reorganização anterior do repositório (checkout antigo, flat, com
`create-tfrecord.py`/`results/`/`models/` na raiz), **não sobrescreva sem
investigar** — mova para um backup (`mv .../hcpa .../hcpa_legacy_backup_<data>`)
antes de recriar a estrutura acima.

---

## 3. Execução

### 3.1 Toda a matriz experimental — ponto de entrada único (produção)

```bash
cd ~/ic/hcpa-retinopathy/hcpa-repository/hcpa
sbatch run_all_experiments.sh
```

Isso é **tudo**. Um único job (`hcpa_orchestrator`) é submetido à partição
`shared` (leve, sem GPU — ele só despacha e monitora, nunca ocupa
`grace1`/`grace2` ele mesmo) e roda como um daemon de longa duração que:

1. Lê `experiments/filter_matrix_template.json` (576 condições × 10 repetições
   = 5.760 execuções de treino+avaliação, mais criação de dataset/TFRecord/
   limpeza por filtro).
2. Para cada filtro ainda não concluído, submete Script 1 **sem**
   `--nodelist` — o SLURM escolhe `grace1` ou `grace2` conforme
   disponibilidade real no momento (ver seção 4.4).
3. Descobre em qual nó caiu e fixa (`--nodelist`) TFRecord + as 10
   repetições de treino/avaliação + limpeza nesse mesmo nó.
4. Registra cada evento em `logs/master.log` e atualiza
   `logs/progress_report.txt` periodicamente.
5. Isola falhas por filtro/repetição em `failed_jobs.json`, sem travar os
   demais.
6. Ao concluir tudo (ou esgotar o que é automaticamente recuperável),
   escreve `logs/final_report.txt` e termina.
7. Se o próprio limite de tempo do daemon (24h, partição `shared`) estiver
   perto do fim e ainda houver trabalho, ele se auto-resubmete e sai — você
   não precisa reenviar nada.

**Depois de rodar `sbatch run_all_experiments.sh` uma vez, não é necessário
executar mais nenhum comando manual** — use a seção 4 apenas para
acompanhar o progresso (opcional).

### 3.2 Um subconjunto da matriz

```bash
python3 pipeline/script5_orquestrador.py --matrix-file caminho/para/matriz_reduzida.json
```

Mesmo schema de `experiments/filter_matrix_template.json`. Útil para rodar
só alguns filtros novos sem esperar a matriz inteira. Pode ser chamado
diretamente (roda em primeiro plano, útil para depuração) ou também via
`sbatch` com um wrapper próprio análogo a `run_all_experiments.sh`.

### 3.3 Um único filtro/repetição manualmente (depuração pontual)

Só para investigar um caso específico — o daemon da seção 3.1 já faz isso
automaticamente para toda a matriz. Sempre com o **mesmo `--nodelist`** em
todas as etapas de um mesmo filtro:

```bash
SSD_BASE=/ssd/aadsilva/ic/hcpa
NODE=grace2

sbatch --nodelist=$NODE --export=FILTRO=CLAHE4.0,SSD_BASE=$SSD_BASE,HOME_BASE=$HOME \
    slurm/job_template_script1.sh
sbatch --nodelist=$NODE --dependency=afterok:<jobid_script1> \
    --export=FILTRO=CLAHE4.0,SSD_BASE=$SSD_BASE,HOME_BASE=$HOME \
    slurm/job_template_tfrecord.sh
sbatch --nodelist=$NODE --dependency=afterok:<jobid_script1>:<jobid_tfrecord> \
    --export=FILTRO=CLAHE4.0,REPETICAO=0,SSD_BASE=$SSD_BASE,HOME_BASE=$HOME \
    slurm/job_template_script2_3.sh
```

**Cuidado**: se o daemon da seção 3.1 estiver rodando ao mesmo tempo, ele
pode também tentar agir sobre o mesmo filtro (é idempotente/seguro, mas
pode gerar confusão nos logs). Prefira rodar isso com o daemon parado, ou
apenas para um filtro que o daemon ainda não tocou.

---

## 4. Monitoramento

### 4.1 Acompanhar o daemon e os jobs

```bash
squeue -u $USER                          # todos os seus jobs, incluindo hcpa_orchestrator
cat logs/progress_report.txt             # resumo atualizado a cada ciclo do daemon (~60s)
tail -f logs/master.log                  # log mestre: um evento por linha, todas as etapas/filtros
tail -f logs/hcpa_orchestrator_<jobid>.log   # log do próprio daemon (decisões de despacho)
```

Exemplo de `progress_report.txt`:

```
Completed: 53 / 180
Remaining: 127
Failed repetitions: 2
Estimated remaining time: 18:24:00

Current node usage:
  grace1: alloc (GPU busy)
  grace2: idle (GPU free)

Per-filter status:
  baseline: COMPLETED
  CLAHE4.0: train/eval: submitted missing repetitions on grace2
  ...
```

Exemplo de linha de `master.log`:

```
[2026-07-09 02:13:10] filter=CLAHE4.0_Gamma0.8 node=grace2 gpu=0 stage=tfrecord rep=NA status=STARTED
[2026-07-09 02:14:05] filter=CLAHE4.0_Gamma0.8 node=grace2 gpu=0 stage=tfrecord rep=NA status=SUCCESS
[2026-07-09 02:14:06] filter=CLAHE4.0_Gamma0.8 node=grace2 gpu=0 stage=train rep=0 status=STARTED
```

### 4.2 Consultar logs por job

```bash
squeue -j <jobid>
scontrol show job <jobid>                # Dependency=... e Reason=...
tail -f logs/hcpa_train_<filtro>_<rep>_<jobid>.log
cat logs/hcpa_train_<filtro>_<rep>_<jobid>.err   # deve estar vazio ou só com warnings benignos do TF/cuDNN
```

Reasons comuns: `Dependency` (aguardando etapa anterior), `Resources`
(aguardando nó livre — normal, o daemon não falha por isso), `Priority`.

### 4.3 Cancelar jobs / parar a campanha

```bash
scancel -n hcpa_orchestrator -u $USER    # para o próprio daemon (jobs já em andamento continuam soltos)
scancel -u $USER -n hcpa_train_CLAHE4.0_0
scancel -u $USER                          # tudo (cuidado)
```

Parar o daemon (`scancel hcpa_orchestrator`) **não cancela** os jobs de
pipeline já submetidos — eles continuam rodando/enfileirados
independentemente. Para retomar o despacho automático depois, basta
`sbatch run_all_experiments.sh` de novo (seção 5).

### 4.4 Distribuição entre nós (dinâmica, delegada ao SLURM)

O Script 1 de cada filtro é submetido **sem** `--nodelist` — o próprio
scheduler do SLURM escolhe `grace1` ou `grace2` com base na disponibilidade
real no momento (fila, GPU livre, carga), que é exatamente o papel para o
qual o SLURM foi desenhado. O daemon descobre o nó escolhido lendo
`experiments/node_assignments/{filtro}.node` (o próprio job grava seu
`hostname` ali assim que começa a rodar) e fixa TFRecord/treino/avaliação/
limpeza desse MESMO filtro nesse MESMO nó (obrigatório: SSD local). Isso
significa: "grace1 ocupada" nunca causa falha, apenas atraso natural — o
próximo filtro pendente vai automaticamente para `grace2` se ela estiver
livre.

Não há mais round-robin fixo nem hash — a escolha é sempre a mais atual
possível, e cada filtro é decidido de forma independente no momento em que
seu Script 1 é submetido.

### 4.5 Verificar métricas e success markers

```bash
ls ~/resultados/*.csv
ls ~/resultados/_success_markers/
cat ~/resultados/_success_markers/baseline_0.ok
column -s, -t < ~/resultados/baseline_0_*.csv | less -S
```

Campos do CSV (17 colunas): `filter_name`, `repetition`, `timestamp`,
`git_commit`, `num_samples`, `accuracy`, `sensitivity`, `specificity`,
`precision`, `f1_score`, `auc_roc`, `xai_num_samples`, `dice_mean`,
`dice_std`, `iou_mean`, `iou_std`, `pointing_game_accuracy`.

---

## 5. Recuperação após falhas

### 5.1 O daemon foi interrompido (limite de tempo, nó reiniciado, cancelado por engano)

**Reenvie exatamente o mesmo comando:**

```bash
sbatch run_all_experiments.sh
```

O novo daemon reconstrói o estado a partir de 3 fontes sempre visíveis via
NFS — nunca depende de memória própria nem do histórico do SLURM (`sacct`
está indisponível neste cluster): success markers
(`~/resultados/_success_markers/`), marcadores de nó
(`experiments/node_assignments/`) e a fila atual (`squeue`). Ele nunca
resubmete um job que já está pendente/rodando, nunca re-treina uma
repetição que já tem success marker, e continua exatamente de onde parou.
Isso vale mesmo que jobs de pipeline (Script 1-4) já submetidos por uma
instância anterior do daemon ainda estejam rodando — eles são jobs SLURM
independentes, não filhos do processo do daemon.

### 5.2 Um nó caiu de verdade (down/drain por muito tempo)

O daemon já detecta isso sozinho (consulta `sinfo` periodicamente): se um
nó com um filtro atribuído fica em estado `down`/`drain`/`fail` por mais de
30 minutos (configurável via `--node-down-timeout`), ele cancela os jobs
pendentes daquele filtro naquele nó, limpa sua atribuição de nó, e volta a
submeter o Script 1 daquele filtro sem `--nodelist` — deixando o SLURM
escolher o nó (presumivelmente vivo) que restar. **Nenhuma ação manual ou
edição de código é necessária.** Isso cobre tanto "grace1 ocupada por
horas" (não é considerado down, só espera) quanto "grace1 fora do ar por
dias" (é migrado automaticamente).

### 5.3 Como verificar quais filtros já terminaram

```bash
cat logs/progress_report.txt   # já traz isso, atualizado periodicamente
# ou manualmente:
for f in $(python3 -c "import json; print(' '.join(x['name'] for x in json.load(open('experiments/filter_matrix_template.json'))))"); do
    n=$(ls ~/resultados/_success_markers/${f}_*.ok 2>/dev/null | wc -l)
    echo "$f: $n/10 repetições concluídas"
done
```

### 5.4 Um filtro/repetição falhou de verdade (não é questão de nó)

O daemon **nunca para a campanha** por causa disso — ele marca a
falha e segue com os demais. Ao final (ou a qualquer momento):

```bash
cat failed_jobs.json
```

Estrutura: `{"filters": {...}, "repetitions": {...}, "cleanup": {...}}` —
falhas em nível de dataset/TFRecord (bloqueiam o filtro inteiro), falhas de
treino/avaliação de uma repetição específica (as outras 9 continuam
normalmente), e falhas de limpeza (a ciência já está completa, só o disco
não foi liberado). Cada entrada tem `reason` e `log_hint` (caminho glob do
log relevante). O daemon **não** re-tenta automaticamente algo já marcado
como falho — corrija a causa raiz e então:

```bash
# reexecutar só o que falhou: remova a entrada relevante de failed_jobs.json
# (ou o arquivo inteiro, se quiser reexaminar tudo) e rode de novo:
sbatch run_all_experiments.sh
```

Se for um crash de baixo nível (`std::bad_function_call`/`SIGABRT`/`SIGBUS`
no `.err`, sem traceback Python), já foram encontrados e corrigidos 2 casos
distintos durante a validação desta infraestrutura (grappler
`layout_optimizer` e um bloco de pós-treino não essencial) — se reaparecer
em outro ponto, o padrão de correção é o mesmo: isolar/remover o trecho não
essencial, nunca o treino em si.

---

## 6. Organização dos resultados

| O quê | Onde | Sobrevive a qual limpeza? |
|---|---|---|
| CSVs de métricas | `~/resultados/{filtro}_{rep}_{timestamp}.csv` (NFS/HOME) | Permanente |
| Success markers | `~/resultados/_success_markers/{filtro}_{rep}.ok` (NFS/HOME) | Permanente |
| Log mestre (todas as execuções) | `logs/master.log` (NFS) | Permanente, append-only |
| Relatório de progresso | `logs/progress_report.txt` (NFS) | Sobrescrito a cada ciclo |
| Relatório final | `logs/final_report.txt` (NFS) | Gerado quando a matriz é concluída |
| Falhas estruturadas | `failed_jobs.json` (raiz do repo, NFS) | Permanente até edição manual |
| Marcadores de nó por filtro | `experiments/node_assignments/{filtro}.node` (NFS) | Permanente enquanto o filtro não conclui |
| Logs SLURM (por job) | `logs/{job}_{jobid}.log`/`.err` (raiz do repo, NFS) | Permanente (não rotacionado automaticamente) |
| Dataset filtrado (imagens) | `$SSD_BASE/datasets_filtrados/{filtro}/` (SSD local do nó) | Apagado pelo Script 4 após as 10 repetições |
| TFRecords | `$SSD_BASE/data/tfrecords_filtrados/{filtro}/` (SSD local do nó) | Apagado pelo Script 4 após as 10 repetições |
| Checkpoint `.keras` (~250MB) | `$SSD_BASE/checkpoints/{filtro}-{rep}.keras` (SSD local do nó) | **Removido automaticamente pelo Script 3 imediatamente após a avaliação daquela repetição ter sucesso** (não espera as 10 repetições) |
| Metadata/CSVs de treino (KB) | `$SSD_BASE/checkpoints/training_metadata_*.json`, `*-phase1_warmup.csv`, `*-phase2_finetune.csv` | Mantidos (pequenos, úteis para auditoria) |
| Backup soft-delete (Script 4) | `~/trash_datasets/` (NFS/HOME) | Permanente até limpeza manual |

**Sobre a remoção automática de checkpoints**
(`pipeline/script3_avalia.py::delete_checkpoint_after_success()`): o
checkpoint só é apagado **depois** que CSV validado + success marker já
foram gravados com sucesso — nunca antes. Se a avaliação falhar, o
checkpoint permanece intacto para depuração.

---

## 7. Checklist final de produção

Antes de rodar `sbatch run_all_experiments.sh` para a matriz completa,
confirme:

- [ ] `git status` limpo o suficiente / mudanças de infraestrutura
      commitadas (para que `git_commit` nos CSVs rastreie a versão real).
- [ ] `data/Seg-set` + `experiments/{split.json,split_metadata.json,filter_matrix_template.json}`
      presentes em **ambos** `grace1` e `grace2` sob `/ssd/aadsilva/ic/hcpa`
      (seção 2) — o SLURM pode escolher qualquer um dos dois para qualquer
      filtro.
- [ ] `df -h /ssd` com espaço livre suficiente em ambos os nós.
- [ ] `nvidia-smi` limpo em ambos os nós antes de começar.
- [ ] `logs/master.log`, `failed_jobs.json`,
      `experiments/node_assignments/` — decidir se algo de uma rodada
      anterior deve ser preservado (resume normal) ou limpo (campanha do
      zero). Success markers em `~/resultados/_success_markers/` são
      sempre respeitados (nunca re-treinados).
- [ ] Rodar 1 filtro isolado de ponta a ponta e confirmar: checkpoint some
      após avaliação, CSV com 17 colunas sem NaN, success marker criado,
      sem crashes no `.err`, e (se possível) deixar completar 10/10 para
      confirmar que o Script 4 também roda com sucesso.
- [ ] `singularity.sif` presente e íntegro na raiz do repo (~6.7GB).
- [ ] Ninguém mais com alocação `--exclusive` competindo pelos nós
      `grace1`/`grace2` no momento de submeter o lote (`squeue -p grace`).
- [ ] Confirmar que a partição `shared` está disponível para o daemon
      (`sinfo -p shared`) — é onde `run_all_experiments.sh` roda.

Quando todos os itens acima estiverem marcados, `sbatch
run_all_experiments.sh` é suficiente para rodar a matriz inteira sem
intervenção manual — inclusive across falhas de nó, jobs individuais, e
reinícios do próprio daemon.
