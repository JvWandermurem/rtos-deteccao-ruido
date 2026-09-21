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
    args = parser.parse_args()

    tabela = np.genfromtxt(args.dados, delimiter=",", skip_header=1)
    X = tabela[:, :8].astype(np.float32)
    y = tabela[:, 8].astype(int)

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
