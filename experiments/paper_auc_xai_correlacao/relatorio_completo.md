# Acurácia e Explicabilidade em Retinopatia Diabética — Relatório Completo de Resultados

Todos os números por trás do artigo submetido ao ERAMIA/RS: as 495 condições de pré-processamento da campanha principal, os 10 filtros da mini-campanha multi-método, o sanity check de Adebayo, e um achado de bookkeeping descoberto ao montar este relatório.

**Arquivos que acompanham este relatório:**

- `resultados_495_condicoes.csv` — as 495 condições completas (AUC, Dice, IoU, Pointing Game, resíduo)
- `resultados_minicampanha_repeticoes.csv` — as 276 repetições reais da mini-campanha (10 filtros × até 10 reps × 3 métodos XAI)

## Resumo executivo

| | |
|---|---|
| Condições na campanha principal | 495 |
| Correlação AUC × Dice (Grad-CAM) | r=0,865, R²=0,749, p<0,001 |
| Filtros selecionados para mini-campanha | 10 |
| Repetições da mini-campanha concluídas | **92/100** (corrigido — ver achado operacional) |
| Sanity check de Adebayo aprovado | 26/29 combinações válidas (90%) |
| GPU-horas consumidas (mini-campanha) | 319,87h |

## Critério de seleção dos 10 filtros

Da campanha principal (495 condições, Grad-CAM), a subamostra de 10 foi escolhida para representar tanto o padrão típico quanto os desvios da correlação AUC×Dice:

- **Cobertura** (5 pontos): um ponto por percentil de AUC (95, 75, 50, 25, 5), escolhido minimizando `|AUC−alvo| + 2×|resíduo|` — garante espalhamento ao longo de toda a faixa de desempenho sem escolher outliers por acidente.
- **Sonda** (4 pontos): os 2 maiores resíduos positivos e os 2 maiores resíduos negativos da regressão Dice~AUC, restritos a AUC>0,70 — testa deliberadamente os casos em que a correlação é mais fraca.
- **Extremo** (1 ponto): a condição de menor AUC de toda a base (0,552) — regime de desempenho quase aleatório, teste de estresse do método.

Representatividade validada: cobertura de **98,7%** da faixa de AUC e **96,8%** da faixa de Dice das 495 originais; teste de Kolmogorov–Smirnov entre as duas amostras: D=0,149, p=0,959 (poder estatístico baixo dado n pequeno, ressalva registrada no artigo).

## Os 10 filtros selecionados

AUC/Dice/resíduo recalculados diretamente da base de 495 condições. "Reps mini-campanha" é o valor real verificado nos marcadores de sucesso — ver [achado operacional](#achado-operacional-bug-de-bookkeeping-descoberto-neste-relatório) abaixo.

| Filtro | Papel | AUC | Dice | Resíduo | Reps mini-campanha |
|---|---|---|---|---|---|
| baseline | cobertura (p~99.6) | 0.9122 | 0.0645 | +0.0008 | 10/10 |
| BenGraham_MaxGreen2.0 | cobertura (p~76) | 0.8888 | 0.0581 | -0.0005 | 4/10 ⚠️ |
| BenGraham_GreenChannel_CLAHE4.0_Otsu | cobertura (p~48) | 0.8695 | 0.0545 | +0.0000 | 10/10 |
| Gamma0.8_MaxGreen2.0_Otsu | cobertura (p~24) | 0.7941 | 0.0379 | -0.0002 | 10/10 |
| BenGraham_MaxGreen2.0_Canny | cobertura (p~5) | 0.7059 | 0.0185 | -0.0005 | 9/10 ⚠️ |
| Retinex_Gaussian | sonda + | 0.8895 | 0.0810 | +0.0222 | 10/10 |
| Retinex_MaxGreen2.0_Gaussian | sonda + | 0.8852 | 0.0819 | +0.0240 | 10/10 |
| Retinex_MaxGreen2.0_CLAHE4.0_Otsu | sonda - | 0.8611 | 0.0358 | -0.0168 | 9/10 ⚠️ |
| Retinex_Grayscale_CLAHE4.0_Otsu | sonda - | 0.8536 | 0.0327 | -0.0183 | 10/10 |
| LABNorm_MaxGreen2.0_Frangi | extremo | 0.5518 | 0.0221 | +0.0366 | 10/10 |

## Correlação AUC × fidelidade Grad-CAM (n=495)

| Métrica | r | R² | p |
|---|---|---|---|
| Dice | 0,865 | 0,749 | <0,001 |
| IoU | 0,862 | 0,744 | <0,001 |
| Pointing Game | 0,831 | 0,690 | <0,001 |

Reta ajustada por mínimos quadrados sobre Dice~AUC: slope=0,2169, intercepto=−0,1342.

## Distribuições (n=495)

**AUC-ROC** — percentis: p0=0,552, p5=0,707, p10=0,738, p25=0,800, p50=0,871, p75=0,888, p90=0,897, p95=0,902, p100=0,917. Concentra-se fortemente entre 0,86–0,90 (moda), com cauda longa até 0,55 (regime quase aleatório).

**Dice** — percentis: p0=0,016, p5=0,020, p10=0,024, p25=0,035, p50=0,052, p75=0,061, p90=0,068, p95=0,072, p100=0,082. Pico em 0,055–0,062, também com cauda para condições de fidelidade muito baixa — a mesma condição costuma estar em ambas as caudas.

## Outliers da regressão Dice~AUC

As 10 condições cujo Dice mais se desvia do previsto pela reta de regressão, em qualquer direção.

| Filtro | AUC | Dice | Resíduo | Na subamostra? |
|---|---|---|---|---|
| LABNorm_MaxGreen2.0_Frangi | 0.552 | 0.0221 | +0.0366 | ✅ sim |
| MaxGreen2.0_Frangi | 0.567 | 0.0208 | +0.0319 | — |
| BenGraham_MaxGreen2.0_AHE40.0_Canny | 0.663 | 0.0347 | +0.0251 | — |
| Retinex_MaxGreen2.0_Gaussian | 0.885 | 0.0819 | +0.0240 | ✅ sim |
| Retinex_Gaussian | 0.889 | 0.0810 | +0.0222 | ✅ sim |
| Retinex_Morpho | 0.876 | 0.0766 | +0.0207 | — |
| LABNorm_GreenChannel_Gaussian | 0.895 | 0.0799 | +0.0200 | — |
| GreenChannel_Gaussian | 0.897 | 0.0802 | +0.0198 | — |
| Gamma0.8_MaxGreen2.0_Median | 0.898 | 0.0805 | +0.0198 | — |
| BenGraham_Grayscale_AHE40.0_Canny | 0.677 | 0.0324 | +0.0197 | — |

## Tabela completa das 495 condições

Ver `resultados_495_condicoes.csv` (colunas: filtro, auc_mean, auc_std, dice_mean, dice_std, iou_mean, pointing_game_mean, residuo_dice_auc). Amostra das 10 condições de maior AUC:

| Filtro | AUC | Dice | IoU | Pointing Game |
|---|---|---|---|---|
| LABNorm | 0.917 | 0.0650 | 0.0363 | 0.0776 |
| Gamma0.8 | 0.913 | 0.0648 | 0.0361 | 0.0803 |
| baseline | 0.912 | 0.0645 | 0.0360 | 0.0801 |
| MaxGreen2.0_Gaussian | 0.911 | 0.0733 | 0.0409 | 0.0799 |
| LABNorm_Median | 0.910 | 0.0671 | 0.0375 | 0.0807 |
| Gamma0.8_MaxGreen2.0_Gaussian | 0.909 | 0.0800 | 0.0449 | 0.0940 |
| HistogramEq_GreenChannel_Gaussian | 0.909 | 0.0640 | 0.0356 | 0.0707 |
| Gamma0.8_Median | 0.908 | 0.0690 | 0.0385 | 0.0819 |
| LABNorm_MaxGreen2.0_Gaussian | 0.907 | 0.0744 | 0.0415 | 0.0819 |
| Median | 0.907 | 0.0685 | 0.0383 | 0.0792 |

## Mini-campanha — resultados por repetição

Ver `resultados_minicampanha_repeticoes.csv` para as 276 linhas completas (filtro, repetição, método XAI, AUC, Dice, IoU, Pointing Game, sensibilidade, especificidade). Média por filtro × método:

| Filtro | Método | AUC | Dice | IoU | PG |
|---|---|---|---|---|---|
| baseline | gradcam | 0.9137 | 0.0767 | 0.0439 | 0.1000 |
| baseline | lime | 0.9137 | 0.0405 | 0.0219 | 0.0310 |
| baseline | occlusion | 0.9137 | 0.0395 | 0.0212 | 0.0380 |
| BenGraham_MaxGreen2.0 | gradcam | 0.8843 | 0.0718 | 0.0408 | 0.0800 |
| BenGraham_MaxGreen2.0 | lime | 0.8843 | 0.0445 | 0.0240 | 0.0275 |
| BenGraham_MaxGreen2.0 | occlusion | 0.8843 | 0.0428 | 0.0232 | 0.0450 |
| BenGraham_GreenChannel_CLAHE4.0_Otsu | gradcam | 0.8630 | 0.0707 | 0.0394 | 0.1110 |
| BenGraham_GreenChannel_CLAHE4.0_Otsu | lime | 0.8630 | 0.0513 | 0.0285 | 0.0010 |
| BenGraham_GreenChannel_CLAHE4.0_Otsu | occlusion | 0.8630 | 0.0525 | 0.0288 | 0.0560 |
| Gamma0.8_MaxGreen2.0_Otsu | gradcam | 0.7913 | 0.0405 | 0.0217 | 0.0270 |
| Gamma0.8_MaxGreen2.0_Otsu | lime | 0.7913 | 0.0459 | 0.0249 | 0.0300 |
| Gamma0.8_MaxGreen2.0_Otsu | occlusion | 0.7913 | 0.0275 | 0.0147 | 0.0240 |
| BenGraham_MaxGreen2.0_Canny | gradcam | 0.7201 | 0.0171 | 0.0089 | 0.0211 |
| BenGraham_MaxGreen2.0_Canny | lime | 0.7201 | 0.0334 | 0.0178 | 0.0133 |
| BenGraham_MaxGreen2.0_Canny | occlusion | 0.7201 | 0.0365 | 0.0197 | 0.0311 |
| Retinex_Gaussian | gradcam | 0.8913 | 0.0968 | 0.0557 | 0.0900 |
| Retinex_Gaussian | lime | 0.8913 | 0.0466 | 0.0254 | 0.0330 |
| Retinex_Gaussian | occlusion | 0.8913 | 0.0487 | 0.0277 | 0.0480 |
| Retinex_MaxGreen2.0_Gaussian | gradcam | 0.8868 | 0.0949 | 0.0549 | 0.0990 |
| Retinex_MaxGreen2.0_Gaussian | lime | 0.8868 | 0.0473 | 0.0257 | 0.0320 |
| Retinex_MaxGreen2.0_Gaussian | occlusion | 0.8868 | 0.0430 | 0.0232 | 0.0410 |
| Retinex_MaxGreen2.0_CLAHE4.0_Otsu | gradcam | 0.8565 | 0.0380 | 0.0203 | 0.0489 |
| Retinex_MaxGreen2.0_CLAHE4.0_Otsu | lime | 0.8565 | 0.0439 | 0.0238 | 0.0078 |
| Retinex_MaxGreen2.0_CLAHE4.0_Otsu | occlusion | 0.8565 | 0.0327 | 0.0175 | 0.0356 |
| Retinex_Grayscale_CLAHE4.0_Otsu | gradcam | 0.8532 | 0.0367 | 0.0197 | 0.0330 |
| Retinex_Grayscale_CLAHE4.0_Otsu | lime | 0.8532 | 0.0469 | 0.0256 | 0.0150 |
| Retinex_Grayscale_CLAHE4.0_Otsu | occlusion | 0.8532 | 0.0398 | 0.0216 | 0.0330 |
| LABNorm_MaxGreen2.0_Frangi | gradcam | 0.5555 | 0.0183 | 0.0096 | 0.0110 |
| LABNorm_MaxGreen2.0_Frangi | lime | 0.5555 | 0.0337 | 0.0179 | 0.0190 |
| LABNorm_MaxGreen2.0_Frangi | occlusion | 0.5555 | 0.0449 | 0.0244 | 0.0350 |

## Correlação multi-método (mini-campanha, n=10)

| Método | Dice (r, p) | IoU (r, p) | Pointing Game (r, p) |
|---|---|---|---|
| Grad-CAM | 0,756 / 0,011 | 0,747 / 0,013 | 0,765 / 0,010 |
| LIME | 0,735 / 0,015 | 0,730 / 0,017 | 0,218 / 0,545 |
| Occlusion | 0,083 / 0,820 | 0,096 / 0,792 | 0,415 / 0,233 |

Grad-CAM e LIME replicam a correlação forte vista nas 495 condições; Occlusion Sensitivity não.

## Sanity check de Adebayo

Das 29 combinações válidas condição × método (uma indefinida por gradiente constante), **26 (90%)** mostram queda de ρ=1,0 (referência) para próximo de zero conforme o modelo é aleatorizado — evidência de dependência real aos pesos aprendidos. A exceção é o LIME em `BenGraham_GreenChannel_CLAHE4.0_Otsu`, cujo mapa permanece quase inalterado (ρ=0,92) mesmo com o modelo totalmente aleatorizado.

## Achado operacional: bug de bookkeeping descoberto neste relatório

**Bug real.** `success_marker_reps()` (`pipeline/script5_orquestrador.py`) usava `glob(f"{filtro}_*.ok")` — um padrão não ancorado por prefixo, que também casa com marcadores de filtros DIFERENTES cujo nome começa com o mesmo prefixo (ex.: contar `BenGraham_MaxGreen2.0_Canny_10.ok` como se fosse de `BenGraham_MaxGreen2.0`). Isso é comum neste projeto, já que os nomes de filtro são cadeias compostas.

**Consequência real:** `BenGraham_MaxGreen2.0` tem genuinamente apenas **4 de 10** repetições da mini-campanha completas (10, 16, 17, 19 — faltam 11–15 e 18, nunca tentadas de novo porque o despachante achava que já estavam prontas). O relatório final do pipeline (`final_report.txt`) chegou a declarar esse filtro "10/10 concluído".

**Corrigido** nesta sessão: a função agora exige que o texto após o prefixo exato `"{filtro}_"` seja só dígitos, validado com teste unitário reproduzindo exatamente essa colisão.

**Total real da mini-campanha, com a contagem corrigida: 92/100** (não 98/100, como relatórios anteriores desta sessão chegaram a indicar).

Como esta função é o único ponto de checagem de conclusão usado por praticamente todo o despachante (status de progresso, disparo de limpeza, contagem de "em andamento"), o mesmo tipo de colisão pode ter afetado a contabilidade da campanha principal (576 filtros, muitos com nomes prefixo-de-outro) — os CSVs de métricas em si não são afetados (todos os números deste relatório vêm diretamente deles), só a contagem de "quantas repetições este filtro tem".

## Figuras

Ver `figura1_auc_dice.pdf`/`.png` (AUC × Dice, 495 condições) e `figura2_multimetodo.pdf`/`.png` (confirmação multi-método + sanity check), na mesma pasta do artigo (`experiments/paper_auc_xai_correlacao/`).
