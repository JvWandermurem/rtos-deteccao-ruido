"""Treina a regressão logística de assobio e exporta ONNX + header C."""

import argparse

import numpy as np
from skl2onnx import to_onnx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CABECALHO_C = """// Gerado por model/treino.py -- não editar à mão.
#ifndef CLASSIFIER_WEIGHTS_H
#define CLASSIFIER_WEIGHTS_H

static const float CLASSIFIER_PESOS[8] = {{{pesos}}};
static const float CLASSIFIER_BIAS = {bias}f;
static const float CLASSIFIER_MEDIA[8] = {{{media}}};
static const float CLASSIFIER_ESCALA[8] = {{{escala}}};

#endif
"""


def formatar(vetor):
    return ", ".join(f"{v:.8f}f" for v in vetor)


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
        default=100000.0,
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
    # O corte de 100000 vem dos dados: é o vale do histograma bimodal de RMS
    # da sessão de assobio (densidade cai por volta de 93000 e volta a subir
    # por volta de 160000) e fica acima do percentil 99 da sessão ambiente
    # (84514).
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

    escalador = StandardScaler().fit(X_treino)
    modelo = LogisticRegression(max_iter=1000).fit(
        escalador.transform(X_treino), y_treino
    )

    previsto = modelo.predict(escalador.transform(X_teste))
    print(confusion_matrix(y_teste, previsto))
    print(classification_report(y_teste, previsto, digits=3))

    onnx = to_onnx(modelo, escalador.transform(X_treino)[:1].astype(np.float32))
    with open(args.onnx, "wb") as arquivo:
        arquivo.write(onnx.SerializeToString())

    with open(args.header, "w") as arquivo:
        arquivo.write(
            CABECALHO_C.format(
                pesos=formatar(modelo.coef_[0]),
                bias=f"{modelo.intercept_[0]:.8f}",
                media=formatar(escalador.mean_),
                escala=formatar(escalador.scale_),
            )
        )

    print(f"ONNX em {args.onnx} e header em {args.header}")


if __name__ == "__main__":
    main()
