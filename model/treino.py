"""Treina a árvore de decisão de assobio e exporta ONNX + header C."""

import argparse

import numpy as np
from skl2onnx import to_onnx
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

CABECALHO_C = """// Gerado por model/treino.py -- não editar à mão.
#ifndef CLASSIFIER_WEIGHTS_H
#define CLASSIFIER_WEIGHTS_H

// Árvore de decisão de profundidade {profundidade}, {folhas} folhas.
// Retorna a probabilidade de assobio na folha alcançada.
static inline float classifier_tree_score(const float* f) {{
{corpo}}}

#endif
"""


def emitir_arvore(arvore, no=0, nivel=1):
    """Converte a árvore treinada em if/else aninhados em C."""
    ind = "  " * nivel
    if arvore.children_left[no] == -1:
        contagem = arvore.value[no][0]
        prob = contagem[1] / (contagem[0] + contagem[1])
        return f"{ind}return {prob:.6f}f;\n"
    corpo = f"{ind}if (f[{arvore.feature[no]}] <= {arvore.threshold[no]:.6f}f) {{\n"
    corpo += emitir_arvore(arvore, arvore.children_left[no], nivel + 1)
    corpo += f"{ind}}} else {{\n"
    corpo += emitir_arvore(arvore, arvore.children_right[no], nivel + 1)
    corpo += f"{ind}}}\n"
    return corpo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", default="dados.csv")
    parser.add_argument("--onnx", default="modelo.onnx")
    parser.add_argument(
        "--header",
        default="../detector-anomalias/lib/classifier/classifier_weights.h",
    )
    parser.add_argument(
        "--rms-min-assobio",
        type=float,
        default=600000.0,
        help=(
            "RMS mínimo (coluna 0) para uma linha rotulada 1 continuar sendo"
            " tratada como assobio; abaixo disso ela é rerrotulada como 0"
            " (veja o comentário acima da aplicação deste corte)."
        ),
    )
    args = parser.parse_args()

    tabela = np.genfromtxt(args.dados, delimiter=",", skip_header=1)
    X = tabela[:, :8].astype(np.float32)
    y = tabela[:, 8].astype(int)

    # A coleta (Task 8) rotulou sessões inteiras: durante os 30s em que o
    # usuário assobiava em rajadas (~2s assobiando, ~1s em silêncio), toda
    # linha da sessão recebeu label 1 -- inclusive as pausas silenciosas
    # entre rajadas, que são acusticamente idênticas ao ambiente. Isso não é
    # ruído aleatório: é um viés sistemático que empurra a fronteira de
    # decisão para dentro da região "silêncio", fazendo o modelo disparar em
    # ambientes quietos.
    #
    # Por isso, linhas rotuladas 1 cujo RMS não ultrapassa `--rms-min-assobio`
    # são rerrotuladas como 0 antes do treino: elas não são assobio, mas são
    # negativos legítimos e valiosos, pois capturam exatamente a condição
    # "usuário perto do microfone, não assobiando" que hoje gera falsos
    # positivos. Nenhuma linha é descartada -- apenas o rótulo é corrigido.
    #
    # O corte não é uma constante fixa: ele acompanha o piso de ruído do
    # ambiente de implantação, que já mudou uma vez (sala quieta -> sala com
    # conversa -> sala com ruído real de fundo) e pode mudar de novo. Cada
    # vez que o ambiente de coleta muda, o corte precisa ser reavaliado
    # contra a nova distribuição de RMS dos negativos, não mantido por
    # inércia. Com o dataset ampliado (mais sala ruidosa real, que passou a
    # ser a maior fonte única de negativos), 400000 ainda deixava 5.3% dos
    # negativos acima do corte -- ruído de sala contaminando a classe
    # positiva, exatamente a falha de falso positivo relatada em uso real.
    # 600000 reduz essa contaminação para 1.5%, à custa de manter menos
    # assobios como positivos (380). Como a queixa em produção é de falso
    # positivo, e não de assobio perdido, errar para o lado da pureza da
    # classe positiva é a escolha certa aqui.
    rotulos_corrigidos = (y == 1) & (X[:, 0] <= args.rms_min_assobio)
    print(
        f"Rerrotulando {rotulos_corrigidos.sum()} linha(s) de assobio com "
        f"RMS <= {args.rms_min_assobio:.0f} para label 0 (pausas silenciosas "
        "entre rajadas de assobio)."
    )
    y[rotulos_corrigidos] = 0

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Árvore rasa: separa assobio de grito sustentado, o que uma fronteira
    # linear não consegue. Invariante a escala, então sem StandardScaler.
    modelo = DecisionTreeClassifier(
        max_depth=4, class_weight="balanced", random_state=42
    ).fit(X_treino, y_treino)

    previsto = modelo.predict(X_teste)
    print(confusion_matrix(y_teste, previsto))
    print(classification_report(y_teste, previsto, digits=3))

    onnx = to_onnx(modelo, X_treino[:1].astype(np.float32))
    with open(args.onnx, "wb") as arquivo:
        arquivo.write(onnx.SerializeToString())

    with open(args.header, "w") as arquivo:
        arquivo.write(
            CABECALHO_C.format(
                profundidade=modelo.get_depth(),
                folhas=modelo.get_n_leaves(),
                corpo=emitir_arvore(modelo.tree_),
            )
        )

    print(f"ONNX em {args.onnx} e header em {args.header}")


if __name__ == "__main__":
    main()
