"""Grava features vindas da serial do ESP32 num CSV rotulado.

Uso: uma sessão por rótulo, por exemplo
    python coleta.py --label 1 --duracao 30   # assoviando
    python coleta.py --label 0 --duracao 30   # ruído ambiente
"""

import argparse
import csv
import os
import time

import serial

COLUNAS = [
    "rms", "centroid", "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5", "mfcc6",
    "label",
]


def coletar(porta, baud, label, duracao):
    linhas = []
    with serial.Serial(porta, baud, timeout=1) as ser:
        ser.reset_input_buffer()
        fim = time.time() + duracao
        while time.time() < fim:
            bruto = ser.readline().decode("ascii", errors="ignore").strip()
            if not bruto.startswith("FEAT "):
                continue
            partes = bruto.split()[1:]
            if len(partes) != 8:
                continue
            try:
                valores = [float(p) for p in partes]
            except ValueError:
                continue
            linhas.append(valores + [label])
    return linhas


def salvar(linhas, destino):
    existe = os.path.exists(destino)
    with open(destino, "a", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        if not existe:
            escritor.writerow(COLUNAS)
        escritor.writerows(linhas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--porta", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--label", type=int, required=True, choices=[0, 1])
    parser.add_argument("--duracao", type=float, default=30.0)
    parser.add_argument("--saida", default="dados.csv")
    args = parser.parse_args()

    print(f"Coletando {args.duracao}s com label={args.label}...")
    linhas = coletar(args.porta, args.baud, args.label, args.duracao)
    salvar(linhas, args.saida)
    print(f"{len(linhas)} amostras gravadas em {args.saida}")


if __name__ == "__main__":
    main()
