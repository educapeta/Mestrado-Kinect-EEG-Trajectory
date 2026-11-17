"""Analisa o log CSV do visualizador AHRS (item 149).

Le `ahrs_logs/ahrs_*.csv` (ou o caminho passado) e diz, por sensor:
- vetor inicial e final (mediana), angulo com -Z (chao) em cada um;
- quanto o vetor final ficou longe do inicial (se voce voltou a posicao
  paralela a mesa, deveria ser ~0);
- minimo/maximo/mediana do angulo com o chao durante a sessao;
- deriva do quaternion (rotacao relativa entre o primeiro e o ultimo quadro).

Uso:
  .venv\Scripts\python.exe tools\analisa_log_ahrs.py
  .venv\Scripts\python.exe tools\analisa_log_ahrs.py AHRS\\logs\\ahrs_....csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


def carregar(caminho: Path):
    linhas = []
    eventos = []
    with caminho.open("r", encoding="utf-8") as fh:
        for linha in fh:
            if linha.startswith("#EVENT"):
                eventos.append(linha.strip())
                continue
            if linha.startswith("#") or not linha.strip():
                continue
            linhas.append(linha)
    leitor = csv.DictReader(linhas)
    dados = {}
    for campo in leitor.fieldnames or ():
        dados[campo] = []
    for linha in leitor:
        for campo in dados:
            try:
                dados[campo].append(float(linha[campo]))
            except (TypeError, ValueError):
                dados[campo].append(np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in dados.items()}, eventos


def vetor_mediana(dados, lado, i0, i1):
    campos = [f"v{x}_{lado}" for x in "xyz"]
    if not all(c in dados for c in campos):
        return None
    mat = np.column_stack([dados[c][i0:i1] for c in campos])
    mat = mat[np.isfinite(mat).all(axis=1)]
    if len(mat) == 0:
        return None
    vetor = np.median(mat, axis=0)
    norma = np.linalg.norm(vetor)
    return vetor / norma if norma > 1e-9 else None


def angulo_com_baixo(vetor):
    if vetor is None:
        return None
    c = float(np.clip(np.dot(vetor, [0.0, 0.0, -1.0]), -1.0, 1.0))
    return math.degrees(math.acos(c))


def angulo_entre(a, b):
    if a is None or b is None:
        return None
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(c))


def quat_relativo_drift(dados, lado, i0, i1):
    campos = [f"{e}_{lado}" for e in ("qw", "qx", "qy", "qz")]
    if not all(c in dados for c in campos):
        return None
    q0 = np.array([dados[c][i0] for c in campos])
    q1 = np.array([dados[c][i1] for c in campos])
    if not np.isfinite(q0).all() or not np.isfinite(q1).all():
        return None
    n0 = np.linalg.norm(q0)
    n1 = np.linalg.norm(q1)
    if n0 < 1e-9 or n1 < 1e-9:
        return None
    q0 /= n0
    q1 /= n1
    # q_rel = q0^-1 * q1
    w0 = q0[0]
    v0 = -q0[1:]
    w, x, y, z = q1
    rel = np.array([
        w0 * w - np.dot(v0, q1[1:]),
        w0 * x + w * v0[0] + np.cross(v0, [x, y, z])[0],
        w0 * y + w * v0[1] + np.cross(v0, [x, y, z])[1],
        w0 * z + w * v0[2] + np.cross(v0, [x, y, z])[2],
    ])
    if rel[0] < 0:
        rel = -rel
    rel = rel / np.linalg.norm(rel)
    return math.degrees(2.0 * math.acos(float(np.clip(rel[0], -1.0, 1.0))))


def analisar(caminho: Path):
    dados, eventos = carregar(caminho)
    n = len(dados.get("t_mono", ()))
    if n == 0:
        print(f"ERRO: sem amostras em {caminho}")
        return 1
    duracao = float(dados["t_mono"][-1] - dados["t_mono"][0])
    print(f"ARQUIVO: {caminho}")
    print(f"amostras: {n} | duracao: {duracao:.1f} s | "
          f"taxa media: {n / max(duracao, 1e-9):.1f} Hz")
    print(f"eventos: {len(eventos)}")
    for evento in eventos[:12]:
        print(f"  {evento}")
    if len(eventos) > 12:
        print(f"  ... ({len(eventos) - 12} eventos)")
    print()

    i_ini = max(1, min(n, int(round(1.0 * n / max(duracao, 1e-9)))))
    i_fim = max(i_ini, n - max(2, int(round(1.5 * n / max(duracao, 1e-9)))))

    resultado = {}
    for lado, nome in (("left", "ESQUERDA 0x69"), ("right", "DIREITA 0x68")):
        vetor_ini = vetor_mediana(dados, lado, 0, i_ini)
        vetor_fim = vetor_mediana(dados, lado, i_fim, n)
        ang_ini = angulo_com_baixo(vetor_ini)
        ang_fim = angulo_com_baixo(vetor_fim)
        retorno = angulo_entre(vetor_ini, vetor_fim)
        angulos = dados.get(f"ang_{lado}_deg")
        if angulos is not None:
            angulos = angulos[np.isfinite(angulos)]
        q_drift = quat_relativo_drift(dados, lado, 0, n - 1)
        resultado[lado] = {
            "inicial": vetor_ini, "final": vetor_fim,
            "ang_inicial": ang_ini, "ang_final": ang_fim,
            "retorno": retorno, "q_drift": q_drift,
            "min": float(np.min(angulos)) if len(angulos) else None,
            "max": float(np.max(angulos)) if len(angulos) else None,
            "mediana": float(np.median(angulos)) if len(angulos) else None,
        }
        print(f"=== {nome} ===")
        print(f"  vetor inicial: {np.round(vetor_ini, 4) if vetor_ini is not None else None}"
              f" | ang. c/ -Z: {ang_ini:.2f} deg" if ang_ini is not None
              else f"  vetor inicial: {vetor_ini}")
        print(f"  vetor final:   {np.round(vetor_fim, 4) if vetor_fim is not None else None}"
              f" | ang. c/ -Z: {ang_fim:.2f} deg" if ang_fim is not None
              else f"  vetor final: {vetor_fim}")
        print(f"  retorno ao inicial: {retorno:.2f} deg" if retorno is not None
              else "  retorno ao inicial: None")
        print(f"  deriva do quaternion inicial->final: {q_drift:.2f} deg"
              if q_drift is not None else "  deriva do quaternion: None")
        print(f"  angulo com o chao: min {resultado[lado]['min']:.2f} / "
              f"mediana {resultado[lado]['mediana']:.2f} / "
              f"max {resultado[lado]['max']:.2f} deg"
              if resultado[lado]['min'] is not None else "  angulo: None")
        print()

    print("=== VEREDITO ===")
    for lado, nome in (("left", "ESQUERDA"), ("right", "DIREITA")):
        r = resultado[lado]
        if r["ang_final"] is None:
            print(f"{nome}: sem dados suficientes")
            continue
        alertas = []
        if r["ang_inicial"] is not None and r["ang_inicial"] > 5.0:
            alertas.append(f"inicio {r['ang_inicial']:.1f} graus longe de -Z")
        if r["ang_final"] > 5.0:
            alertas.append(f"final {r['ang_final']:.1f} graus longe de -Z")
        if r["retorno"] is not None and r["retorno"] > 10.0:
            alertas.append(f"nao voltou ao inicial ({r['retorno']:.1f} graus)")
        if r["max"] is not None and r["max"] < 1.0:
            alertas.append("o vetor nunca saiu de -Z (sensor nao foi movido "
                           "ou leitura travada)")
        if alertas:
            print(f"{nome}: ATENCAO -- " + "; ".join(alertas))
        else:
            print(f"{nome}: OK -- comecou/fechou perto de -Z e voltou ao "
                  f"inicial dentro de {r['retorno']:.1f} graus")
    return 0


def principal():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arquivo", nargs="?", default=None)
    args = parser.parse_args()
    if args.arquivo:
        caminho = Path(args.arquivo)
    else:
        pasta_logs = Path(__file__).resolve().parent.parent / "logs"
        logs = sorted(pasta_logs.glob("ahrs_*.csv"))
        if not logs:
            print(f"Nenhum log em {pasta_logs}/. Rode AHRS_grafico_log.bat antes.")
            return 1
        caminho = logs[-1]
        print(f"(usando o log mais recente: {caminho})")
    if not caminho.exists():
        print(f"ERRO: {caminho} nao existe")
        return 1
    return analisar(caminho)


if __name__ == "__main__":
    raise SystemExit(principal())
