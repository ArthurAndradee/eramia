# Varredura de pré-processamento — 576 condições × EfficientNetV2S (FGADR)

Relatório final da campanha completa (concluída em 2026-10-02). Todos os
números vêm de `analise.py` sobre os CSVs em `~/resultados` (reps 0–9);
tabelas completas nesta pasta.

## 1. Desenho

- **Dados:** FGADR, 1.842 imagens de fundo de olho com máscaras de lesão.
  Classificação binária de RD referável (ICDR ≥ 2, 83% das amostras).
  Split único estratificado 70/30 (1.289 treino / 553 teste), seed 42.
- **Modelo:** EfficientNetV2S pré-treinada na ImageNet, 2 fases (warm-up de
  5 épocas + fine-tuning parcial de até 50 com early stopping), AdamW,
  LR 1e-4, label smoothing 0,1, pesos de classe, batch 32, sem augmentation
  (configuração da HPO em 2 estágios, `experiments/hpo_stage2_final_audit.md`).
- **Espaço de busca:** no máximo 1 filtro por camada, ordem fixa
  C1 → C2 → C3 → C4. 6 × 4 × 3 × 8 = **576 condições** × 10 repetições
  (seeds 0–9) = **5.760 execuções**, todas concluídas.

| Camada | Opções |
|---|---|
| C1 – iluminação | BenGraham, Gamma0.8, Retinex, HistogramEq, LABNorm |
| C2 – espectral | GreenChannel, MaxGreen2.0, Grayscale |
| C3 – contraste local | AHE40.0, CLAHE4.0 |
| C4 – refinamento | Unsharp1.5, Median, Gaussian, Morpho, Frangi, Canny, Otsu |

- **Métricas:** AUC-ROC, sensibilidade, especificidade (limiar 0,5) e
  fidelidade do Grad-CAM contra as máscaras (Dice, IoU, Pointing Game).
- **Estatística:** cada condição comparada ao baseline por teste de Welch
  (n = 10 vs. 10) e por teste t pareado por seed, com correção de Holm e
  FDR (Benjamini-Hochberg) sobre as 575 comparações.

## 2. Resultado principal

**Nenhuma das 575 combinações de pré-processamento supera o baseline (sem
filtro) de forma significativa.**

| Posição | Condição | AUC (média ± dp) | Sens. | Espec. | p (Welch) |
|---|---|---|---|---|---|
| 1 | LABNorm | 0,917 ± 0,008 | 0,83 | 0,85 | 0,23 |
| 2 | Gamma0.8 | 0,913 ± 0,008 | 0,82 | 0,86 | 0,79 |
| **3** | **baseline** | **0,912 ± 0,009** | **0,82** | **0,85** | — |
| 4 | MaxGreen2.0_Gaussian | 0,911 ± 0,007 | 0,90 | 0,67 | 0,66 |
| 5 | LABNorm_Median | 0,910 ± 0,007 | 0,88 | 0,72 | 0,57 |
| 576 | LABNorm_MaxGreen2.0_Frangi | 0,552 ± 0,070 | 0,70 | 0,34 | < 0,001 |

- Só 2 condições têm AUC média acima do baseline (LABNorm, +0,005; Gamma0.8,
  +0,001), e nenhuma é significativa, nem sem correção. Nem o teste pareado
  por seed da melhor delas chega a p < 0,05 (p = 0,064).
- **466 de 575 condições são significativamente piores** que o baseline
  após Holm. As outras 109 são estatisticamente indistinguíveis dele.
- Figura: `figuras/ranking_auc.pdf`.

## 3. Profundidade da cadeia

Quanto mais filtros encadeados, pior (Spearman ρ = −0,22, p < 10⁻⁶;
Kruskal-Wallis p < 10⁻⁶):

| Nº de filtros | Condições | AUC média | AUC mediana | Melhor |
|---|---|---|---|---|
| 0 | 1 | 0,912 | 0,912 | 0,912 |
| 1 | 17 | 0,872 | 0,899 | 0,917 |
| 2 | 101 | 0,853 | 0,885 | 0,911 |
| 3 | 247 | 0,844 | 0,871 | 0,910 |
| 4 | 210 | 0,838 | 0,868 | 0,900 |

Figura: `figuras/auc_por_profundidade.pdf`.

## 4. Efeito de cada filtro

Para cada filtro, compara-se cada condição que o usa com a **mesma cadeia
sem ele** (ex.: `Gamma0.8_CLAHE4.0` vs. `CLAHE4.0`), o que isola a
contribuição do filtro. Teste de Wilcoxon pareado, correção de Holm.

| Filtro | Pares | Δ AUC médio | Δ mediano | % pares que melhoram | p (Holm) |
|---|---|---|---|---|---|
| Frangi | 72 | −0,165 | −0,154 | 0% | < 0,001 |
| Canny | 72 | −0,121 | −0,115 | 0% | < 0,001 |
| Otsu | 72 | −0,036 | −0,031 | 13% | < 0,001 |
| Unsharp1.5 | 72 | −0,022 | −0,021 | 0% | < 0,001 |
| BenGraham | 96 | −0,014 | −0,016 | 27% | < 0,001 |
| MaxGreen2.0 | 144 | −0,013 | −0,011 | 17% | < 0,001 |
| Morpho | 72 | −0,005 | −0,005 | 28% | < 0,001 |
| Grayscale | 144 | −0,004 | −0,002 | 43% | 0,051 |
| Median | 72 | −0,000 | −0,001 | 44% | 1,0 |
| CLAHE4.0 | 192 | +0,001 | −0,003 | 42% | 0,33 |
| AHE40.0 | 192 | +0,001 | −0,005 | 42% | 0,54 |
| GreenChannel | 144 | +0,002 | −0,002 | 48% | 1,0 |
| LABNorm | 96 | +0,002 | +0,001 | 63% | 0,099 |
| HistogramEq | 96 | +0,005 | −0,001 | 49% | 1,0 |
| Gamma0.8 | 96 | +0,005 | +0,002 | 70% | < 0,001 |
| Gaussian | 72 | +0,006 | +0,005 | 64% | 0,019 |
| Retinex | 96 | +0,009 | −0,003 | 44% | 1,0 |

Figura: `figuras/efeito_marginal.pdf`.

- **Detecção de bordas e binarização (Frangi, Canny, Otsu) destroem o
  desempenho.** Elas trocam a imagem por um mapa de bordas ou uma máscara
  binária, eliminando a informação de cor e intensidade de que o
  classificador precisa.
- **Gamma0.8 e Gaussian são os únicos com efeito positivo consistente**,
  mas pequeno (+0,005 a +0,006 de AUC). Mesmo assim, nenhuma cadeia com eles
  supera o baseline sozinho.
- **Retinex e HistogramEq só parecem positivos na média.** A mediana é
  negativa: o ganho médio vem inteiramente de "resgatar" as cadeias com
  Frangi/Canny/Otsu (Retinex +0,069 antes de Frangi, +0,029 antes de Canny),
  enquanto **sem filtro na C4 eles pioram** (Retinex −0,010, HistogramEq
  −0,007). Ou seja, interação, não benefício.
- **CLAHE4.0, o filtro mais usado na literatura de RD**, não ajuda: isolado,
  é significativamente pior que o baseline (0,896, p Holm = 0,037), e o
  efeito pareado mediano é negativo. O mesmo vale para AHE40.0 (0,880).
- **MaxGreen2.0 e BenGraham pioram de forma consistente.** Isso contrasta
  com o resultado do CI-IA (InceptionV3 no HCPA, em que MaxGreen + Gamma
  ajudou), o que sugere que o efeito do pré-processamento depende da
  arquitetura e/ou do dataset.
- Um modelo aditivo (AUC ~ C1 + C2 + C3 + C4, sem interações) explica 88%
  da variância entre condições (R² = 0,876): os efeitos são em grande parte
  independentes entre camadas, com a C4 dominando.

## 5. Sensibilidade × especificidade

- Correlação negativa entre condições (r = −0,48): vários filtros mudam o
  ponto de operação no limiar fixo de 0,5 em vez da capacidade de
  discriminação. Exemplo: MaxGreen2.0_Gaussian tem AUC igual à do baseline
  (0,911), mas sensibilidade 0,90 e especificidade 0,67, contra 0,82 / 0,85
  do baseline.
- Esses ganhos de sensibilidade não são ganhos reais de modelo: com o
  baseline, um limiar diferente de 0,5 poderia produzir o mesmo trade-off.
  Para triagem, a AUC é a medida certa para comparar condições.
- Figura: `figuras/sens_vs_spec.pdf`.

## 6. Fidelidade do Grad-CAM

- Baixa em todas as condições: Dice 0,016–0,082 e Pointing Game
  0,014–0,100.
- Acompanha fortemente o desempenho entre condições: AUC × Dice r = 0,86 e
  AUC × Pointing Game r = 0,83 (n = 576). Isso se deve sobretudo às
  condições destrutivas (Frangi/Canny/Otsu), em que tanto a AUC quanto a
  fidelidade despencam juntas.

## 7. Condições redundantes

13 pares de condições têm métricas idênticas até a 4ª casa decimal (lista
em `resumo.json`, campo `grupos_quase_identicos`). Em todos, o par difere
apenas por `Grayscale` aplicado antes de Frangi, Canny ou Otsu. Esses três
filtros convertem a imagem para tons de cinza internamente, então o
Grayscale anterior não muda nada. Na prática, o espaço tem 563 condições
distintas, não 576. Isso não altera nenhuma conclusão, mas deve ser
mencionado se o número de condições for reportado.

## 8. Limitações

1. **Sem conjunto de validação (decisão do estudo).** O conjunto de teste é
   usado como `validation_data` para early stopping e ReduceLROnPlateau
   (`dr_hcpa_v2_2024.py`). As AUCs absolutas podem estar levemente
   otimistas. A comparação entre condições é feita sob o mesmo protocolo.
   Ver `documentation/METHODOLOGICAL_NOTES.md`, item 1.
2. **Um dataset e uma arquitetura.** Os resultados não se transferem
   necessariamente para outras bases ou CNNs (o CI-IA, com InceptionV3 no
   HCPA, encontrou o efeito oposto para MaxGreen + Gamma).
3. **Uma dose por filtro na matriz combinatória** (ex.: CLAHE só com clip
   4,0, Gamma só 0,8).
4. **Limiar fixo de 0,5** para sensibilidade e especificidade (seção 5).
5. `BenGrahamNormalization` não implementa exatamente o método original
   (`METHODOLOGICAL_NOTES.md`, item 6).
6. 22% das tentativas de treino da campanha caíram por crashes intermitentes
   de baixo nível (SIGABRT, SIGBUS, SIGSEGV) e foram repetidas
   automaticamente. Só entra na análise a avaliação de uma execução que
   terminou com sucesso, sempre com a seed daquela repetição; um crash não
   deixa resultado parcial. Por isso isso custa computação, mas não deve
   introduzir viés.

## 9. Conclusão

Para EfficientNetV2S no FGADR, **a imagem sem pré-processamento é a melhor
escolha ou empata com a melhor.** Nenhum filtro clássico, isolado ou
combinado, melhora a AUC de forma significativa; a maioria piora, e o
prejuízo cresce com o tamanho da cadeia. As exceções positivas (Gamma0.8,
Gaussian, LABNorm) têm efeito pequeno demais para justificar seu uso.

## 10. Arquivos

| Arquivo | Conteúdo |
|---|---|
| `analise.py` | Script que gera tudo abaixo |
| `condicoes_agregadas.csv` | 576 condições: média/dp de todas as métricas, ranking, p-valores (Welch, pareado, Holm, FDR), Cohen's d |
| `efeito_marginal_por_filtro.csv` | Tabela da seção 4 |
| `auc_por_profundidade.csv` | Tabela da seção 3 |
| `modelo_aditivo_coeficientes.csv` | Coeficientes do modelo aditivo |
| `execucoes_reps0-9.csv` | As 5.760 execuções individuais |
| `resumo.json` | Números-chave |
| `figuras/` | Ranking, profundidade, efeito marginal, sensibilidade × especificidade (PDF + PNG) |
