#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera experiments/filter_matrix_template.json como o espaço combinatorio
completo de filtros de pre-processamento, organizado nas 4 camadas
funcionais definidas na auditoria metodologica (ver
documentation/METHODOLOGICAL_NOTES.md e o histórico de conversa):

    Camada 1 - Normalizacao Global de Iluminacao e Fundo
    Camada 2 - Isolamento Espectral e Cromatico
    Camada 3 - Realce Local de Contraste
    Camada 4 - Refinamento Morfologico, Espacial e de Bordas

Design de espaco de busca: cada pipeline aplica NO MAXIMO 1 filtro por
camada (incluindo a opcao "nenhum"), sempre na ordem fixa Camada1 ->
Camada2 -> Camada3 -> Camada4. Essa escolha de design (nao uma poda em
tempo de execucao) e o que torna as 4 regras de exclusao "obrigatorias"
da auditoria estruturalmente satisfeitas antes mesmo de gerar qualquer
combinacao — ver bloco de verificacao abaixo e o relatorio impresso.

Doses/parametros: para manter o espaco combinatorio tratavel, cada
filtro parametrizado entra na matriz combinatoria com UM UNICO valor
representativo (o mais respaldado na literatura ou o ponto medio entre
os extremos testados isoladamente). Variacoes de dose adicionais
(ex.: CLAHE2.0, CLAHE8.0, Gamma1.2) permanecem disponiveis como
condicoes de profundidade 1 (filtro isolado) para ablacao de dose, mas
nao sao fanned-out atraves de toda a matriz combinatoria — isso nao e
uma seguranca de poda, e uma decisao de escopo documentada explicitamente.
"""

import json
from pathlib import Path
from itertools import product

OUTPUT_PATH = Path(__file__).parent / "filter_matrix_template.json"

# Cada camada: lista de (token, tipo_no_codigo, rationale_curta).
# token = string exata usada no nome composto (deve ser reconhecida por
# pipeline/utils_preprocessing.py: get_filter / _create_composite_filter).
LAYER1 = [
    ("BenGraham", "BenGrahamNormalization",
     "correcao de iluminacao/vinheta via subtracao de fundo borrado (metodo Kaggle DR 2015)"),
    ("Gamma0.8", "GammaFilter",
     "correcao de exposicao global (clareia), compensando subexposicao comum em fundoscopia"),
    ("Retinex", "MultiScaleRetinex",
     "correcao de iluminacao multiescala no dominio logaritmico"),
    ("HistogramEq", "HistogramEqualization",
     "equalizacao de histograma GLOBAL, comparador classico do realce local (Camada 3)"),
    ("LABNorm", "LABNormalization",
     "normalizacao do canal L (luminosidade) em espaco LAB, preservando crominancia"),
]

LAYER2 = [
    ("GreenChannel", "GreenChannelExtraction",
     "extracao pura do canal verde (maior contraste vaso/lesao em fundo de olho)"),
    ("MaxGreen2.0", "MaxGreenFilter",
     "supressao moderada do canal verde (fator 2.0), preservando parte do balanco cromatico"),
    ("Grayscale", "GrayscaleConversion",
     "colapso RGB->cinza; risco documentado de perda de informacao diagnostica (cor exsudato vs. cotton-wool)"),
]

LAYER3 = [
    ("AHE40.0", "AHEFilter",
     "equalizacao adaptativa NAO clipada (clip_limit=40); comparador direto do beneficio do clipping do CLAHE"),
    ("CLAHE4.0", "CLAHEFilter",
     "equalizacao adaptativa com limite de clip (clip_limit=4.0), valor mais replicado na literatura de DR"),
]

LAYER4 = [
    ("Unsharp1.5", "UnsharpMask",
     "nitidez morfologica moderada, realca bordas de microaneurismas/vasos"),
    ("Median", "MedianFilter",
     "remocao de ruido sal-e-pimenta preservando bordas"),
    ("Gaussian", "GaussianBlur",
     "suavizacao linear pura, controle de perda de alta frequencia"),
    ("Morpho", "MorphologicalOperations",
     "abertura/fechamento morfologico (kernel eliptico) para limpeza de pequenos artefatos"),
    ("Frangi", "FrangiVesselness",
     "resposta continua (Hessiano) para estruturas tubulares/vasculares"),
    ("Canny", "CannyEdgeDetection",
     "deteccao de bordas binaria (mapa de contornos)"),
    ("Otsu", "OtsuThresholding",
     "binarizacao global por limiar automatico"),
]

LAYERS = [
    ("Camada1_Iluminacao", LAYER1),
    ("Camada2_Espectral", LAYER2),
    ("Camada3_ContrasteLocal", LAYER3),
    ("Camada4_Refinamento", LAYER4),
]


def build_entries():
    entries = []
    # Cada camada contribui com None (ausente) ou uma de suas opcoes.
    choice_lists = [[None] + layer for _, layer in LAYERS]

    for combo in product(*choice_lists):
        tokens = [c[0] for c in combo if c is not None]
        types = [c[1] for c in combo if c is not None]
        rationales = [c[2] for c in combo if c is not None]

        name = "_".join(tokens) if tokens else "baseline"
        depth = len(tokens)

        if depth == 0:
            rationale = ("Controle negativo obrigatorio - nenhuma tecnica de "
                         "pre-processamento aplicada.")
        else:
            rationale = ("Pipeline camada-a-camada (ordem fixa Camada1->Camada2->"
                         "Camada3->Camada4): " + "; ".join(rationales) + ".")

        entries.append({
            "name": name,
            "depth": depth,
            "filters": types,
            "rationale": rationale,
        })

    return entries


def verify_mandatory_rules(entries):
    """
    Confirma (nao aplica em runtime) que as 4 regras de exclusao
    'obrigatorias' da auditoria metodologica ja sao estruturalmente
    impossiveis dado o design de camada-unica + ordem fixa. Levanta
    AssertionError se qualquer entrada violar isso, o que indicaria
    um bug no gerador, nao uma falha das regras em si.
    """
    violations = {1: [], 2: [], 3: [], 4: []}

    for e in entries:
        f = e["filters"]

        def idx(t):
            return f.index(t) if t in f else None

        # Regra 1: GammaFilter -> HistogramEqualization (mesma camada,
        # selecao unica por camada torna coexistencia impossivel).
        if "GammaFilter" in f and "HistogramEqualization" in f:
            violations[1].append(e["name"])

        # Regra 2: {Grayscale,GreenChannel,MaxGreen} -> LABNormalization
        # (Camada2 depois de Camada1 na ordem fixa; LAB so pode vir ANTES).
        i_lab = idx("LABNormalization")
        for chan in ("GrayscaleConversion", "GreenChannelExtraction", "MaxGreenFilter"):
            i_chan = idx(chan)
            if i_lab is not None and i_chan is not None and i_lab > i_chan:
                violations[2].append(e["name"])

        # Regra 3: Grayscale -> {GreenChannel, MaxGreen} (mesma camada).
        if "GrayscaleConversion" in f and (
            "GreenChannelExtraction" in f or "MaxGreenFilter" in f
        ):
            violations[3].append(e["name"])

        # Regra 4: {Otsu,Canny} -> {CLAHE,AHE} (Camada4 depois de Camada3
        # na ordem fixa; CLAHE/AHE so podem vir ANTES de Otsu/Canny).
        for edge in ("OtsuThresholding", "CannyEdgeDetection"):
            i_edge = idx(edge)
            for contrast in ("CLAHEFilter", "AHEFilter"):
                i_contrast = idx(contrast)
                if i_edge is not None and i_contrast is not None and i_edge < i_contrast:
                    violations[4].append(e["name"])

    return violations


def main():
    entries = build_entries()
    violations = verify_mandatory_rules(entries)

    total = len(entries)
    by_depth = {}
    for e in entries:
        by_depth[e["depth"]] = by_depth.get(e["depth"], 0) + 1

    print("=" * 78)
    print("RELATORIO DE GERACAO DA MATRIZ COMBINATORIA DE FILTROS")
    print("=" * 78)
    print(f"\nTotal de combinacoes geradas (pos-design, pre-verificacao): {total}")
    print("Distribuicao por profundidade (numero de filtros no pipeline):")
    for d in sorted(by_depth):
        print(f"  depth={d}: {by_depth[d]} combinacoes")

    print("\nVerificacao das 4 regras OBRIGATORIAS da auditoria metodologica:")
    labels = {
        1: "Regra 1 (Gamma -> HistogramEqualization)",
        2: "Regra 2 (canal-colapsado -> LABNormalization)",
        3: "Regra 3 (Grayscale -> Green/MaxGreen)",
        4: "Regra 4 (Otsu/Canny -> CLAHE/AHE)",
    }
    any_violation = False
    for rule_id, names in violations.items():
        status = "OK — nenhuma ocorrencia (impossivel por design)" if not names else f"VIOLADA em {len(names)} entradas!"
        print(f"  {labels[rule_id]}: {status}")
        if names:
            any_violation = True

    if any_violation:
        raise AssertionError(
            "Uma ou mais regras obrigatorias foram violadas pela matriz gerada — "
            "ver detalhes acima. O gerador NAO deve escrever o JSON nesse estado."
        )

    print(
        "\nConclusao: as 4 regras obrigatorias sao satisfeitas ESTRUTURALMENTE "
        "pelo design (camada unica por nivel + ordem fixa Camada1->2->3->4), "
        "nao por um filtro de exclusao em tempo de execucao. Nenhuma combinacao "
        "precisou ser removida da enumeracao completa."
    )

    with open(OUTPUT_PATH, "w") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)

    print(f"\nEscrito: {OUTPUT_PATH} ({total} entradas)")
    print("=" * 78)


if __name__ == "__main__":
    main()
