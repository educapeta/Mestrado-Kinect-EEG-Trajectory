#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisa a trajetória suavizada de cada landmark 3D do Gate B.

Entrada: CSV longo gerado por
    tools/validacao_groundtruth_deslocamento.py --salvar-landmarks
Colunas: repeticoes, lado, estado, timestamp_s, metodo, landmark, x_m, y_m, z_m

Para cada repetição, método e landmark:
  - ordena por tempo;
  - suaviza x/y/z com Savitzky-Golay (ou média móvel se poucos frames);
  - calcula o deslocamento entre o hold inicial e o final;
  - calcula o comprimento do caminho integrado (soma das distâncias entre
    amostras suavizadas);
  - compara com 0,500 m.

Saída: tabela CSV por landmark + relatório markdown com mediana, IQR, p95 e
quantos landmarks ficaram dentro da tolerância.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

METODOS = ("kinect", "mv_aux0", "mv_aux1", "mv_3view")
DESLOCAMENTO_FISICO_M = 0.500
TOLERANCIA_CM = 2.0


def suaviza(arr: np.ndarray, janela: int = 7, ordem: int = 2) -> np.ndarray:
    """Suaviza (n,3); usa SG se houver frames, senão média móvel."""
    arr = np.asarray(arr, np.float64)
    if len(arr) < 3:
        return arr.copy()
    win = min(int(janela), len(arr))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return arr.copy()
    ordem = min(int(ordem), win - 1)
    out = arr.copy()
    for eixo in range(arr.shape[1]):
        out[:, eixo] = savgol_filter(arr[:, eixo], win, ordem, mode="interp")
    return out


def analisar(csv_path: Path, metodo_filtro: str = "kinect",
             janela: int = 7, ordem: int = 2,
             fisico: float = DESLOCAMENTO_FISICO_M,
             hold_frames: int = 8) -> dict:
    grupos = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if metodo_filtro != "todos" and row["metodo"] != metodo_filtro:
                continue
            try:
                pos = np.array([float(row["x_m"]), float(row["y_m"]),
                                float(row["z_m"])], np.float64)
            except ValueError:
                continue
            if not np.isfinite(pos).all():
                continue
            chave = (int(row["repeticoes"]), row["lado"], row["metodo"],
                     int(row["landmark"]))
            grupos[chave].append((float(row["timestamp_s"]), row["estado"], pos))

    resultados = []
    for (rep, lado, metodo, lm), amostras in sorted(grupos.items()):
        amostras.sort(key=lambda x: x[0])
        ts = np.array([a[0] for a in amostras])
        estados = [a[1] for a in amostras]
        pos = np.stack([a[2] for a in amostras])
        pos_suave = suaviza(pos, janela, ordem)

        # hold inicial/final: usa as amostras de cada estado; se não houver,
        # usa os primeiros/últimos 20% da trajetória.
        idx_ini = [i for i, e in enumerate(estados) if e == "hold_inicio"]
        idx_fim = [i for i, e in enumerate(estados) if e == "hold_fim"]
        if len(idx_ini) >= 3 and len(idx_fim) >= 3:
            # o 3-2-1 pode conter inicio de movimento; usa o FIM do hold inicial
            idx_ini = idx_ini[-hold_frames:]
            idx_fim = idx_fim[:hold_frames]
            p_ini = np.median(pos_suave[idx_ini], axis=0)
            p_fim = np.median(pos_suave[idx_fim], axis=0)
        else:
            n = max(3, len(pos_suave) // 5)
            p_ini = np.median(pos_suave[:n], axis=0)
            p_fim = np.median(pos_suave[-n:], axis=0)

        desloc = float(np.linalg.norm(p_fim - p_ini))
        # caminho integrado: só o trecho de MOVIMENTO (hold estático não conta)
        idx_mov = [i for i, e in enumerate(estados) if e == "movimento"]
        if len(idx_mov) >= 2:
            trecho = pos_suave[idx_mov]
            passos = np.linalg.norm(np.diff(trecho, axis=0), axis=1)
            caminho = float(passos.sum())
        else:
            passos = np.linalg.norm(np.diff(pos_suave, axis=0), axis=1)
            caminho = float(passos.sum()) if len(passos) else 0.0
        resultados.append({
            "repeticoes": rep, "lado": lado, "metodo": metodo,
            "landmark": lm, "n_frames": len(pos),
            "deslocamento_m": desloc,
            "erro_cm": abs(desloc - fisico) * 100.0,
            "caminho_integrado_m": caminho,
            "erro_caminho_cm": abs(caminho - fisico) * 100.0,
        })
    return {"resultados": resultados, "fisico_m": fisico}


def resumo_por_landmark(res: dict) -> list:
    por_lm = defaultdict(list)
    for r in res["resultados"]:
        por_lm[r["landmark"]].append(r)
    saida = []
    for lm, itens in sorted(por_lm.items()):
        d = np.array([i["deslocamento_m"] for i in itens])
        e = np.array([i["erro_cm"] for i in itens])
        c = np.array([i["caminho_integrado_m"] for i in itens])
        saida.append({
            "landmark": lm,
            "n": len(itens),
            "desloc_mediano_m": float(np.median(d)),
            "desloc_iqr_m": float(np.percentile(d, 75) - np.percentile(d, 25)),
            "erro_mediano_cm": float(np.median(e)),
            "erro_p95_cm": float(np.percentile(e, 95)),
            "caminho_mediano_m": float(np.median(c)),
            "dentro_tolerancia": bool(np.median(e) <= TOLERANCIA_CM),
        })
    return saida


def escreve_relatorio(res, resumo, saida_dir: Path, metodo: str):
    saida_dir.mkdir(parents=True, exist_ok=True)
    csv_path = saida_dir / f"trajetoria_landmarks_{metodo}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumo[0].keys()))
        w.writeheader()
        w.writerows(resumo)

    ok = sum(1 for r in resumo if r["dentro_tolerancia"])
    md = saida_dir / f"trajetoria_landmarks_{metodo}.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# Trajetória suavizada por landmark — {metodo}\n\n")
        f.write(f"Deslocamento físico: {res['fisico_m']*100:.1f} cm; "
                f"tolerância: {TOLERANCIA_CM:.1f} cm (mediana do erro).\n\n")
        f.write(f"Landmarks dentro da tolerância: **{ok}/{len(resumo)}**\n\n")
        f.write("| Landmark | Desloc. mediano | Erro mediano | Erro p95 | "
                "Caminho integrado | OK |\n")
        f.write("|---:|---:|---:|---:|---:|:---:|\n")
        for r in resumo:
            f.write(f"| {r['landmark']} | {r['desloc_mediano_m']:.4f} m | "
                    f"{r['erro_mediano_cm']:.2f} cm | "
                    f"{r['erro_p95_cm']:.2f} cm | "
                    f"{r['caminho_mediano_m']:.4f} m | "
                    f"{'SIM' if r['dentro_tolerancia'] else 'NÃO'} |\n")
        f.write("\n## Interpretação\n\n")
        f.write("- deslocamento = distância entre as medianas do hold inicial e final;\n")
        f.write("- caminho integrado = soma das distâncias entre amostras suavizadas;\n")
        f.write("- se o caminho for muito maior que 50 cm, há ruído/curvatura; se for "
                "muito menor, o movimento foi menor que o alvo.\n")
    return csv_path, md


def main():
    ap = argparse.ArgumentParser(description="Analisa trajetória suavizada por landmark")
    ap.add_argument("--csv", type=Path, required=True,
                    help="CSV longo gerado por --salvar-landmarks")
    ap.add_argument("--saida", type=Path, default=Path("analise_trajetoria"))
    ap.add_argument("--metodo", choices=["kinect", "mv_aux0", "mv_aux1",
                                         "mv_3view", "todos"], default="kinect")
    ap.add_argument("--janela", type=int, default=7, help="Janela Savitzky-Golay")
    ap.add_argument("--ordem", type=int, default=2)
    ap.add_argument("--fisico", type=float, default=DESLOCAMENTO_FISICO_M)
    ap.add_argument("--hold-frames", type=int, default=8,
                    help="Frames finais do hold inicial usados como posição de partida")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV não encontrado: {args.csv}")
    res = analisar(args.csv, args.metodo, args.janela, args.ordem,
                   args.fisico, args.hold_frames)
    if not res["resultados"]:
        raise SystemExit("Nenhum landmark encontrado no CSV.")
    resumo = resumo_por_landmark(res)
    csv_out, md_out = escreve_relatorio(res, resumo, args.saida, args.metodo)
    print(f"[OK] {len(resumo)} landmarks analisados")
    print(f"[OK] Tabela: {csv_out}")
    print(f"[OK] Relatório: {md_out}")


if __name__ == "__main__":
    main()
