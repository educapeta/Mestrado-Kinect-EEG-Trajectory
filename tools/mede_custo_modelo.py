"""Mede o custo do modelo por janela (FLOPs e tempo) em varias configuracoes.

Motivo (pergunta de 18/09): "se eu pegar contexto temporal mais longo, a previsao
nao fica muito lenta? e num microcontrolador?". Custo de entrada ~ canais x
amostras, entao **decimar a taxa** permite alongar o contexto quase de graca.
Este utilitario transforma isso em numeros: FLOPs (via FlopCounterMode do
torch) e tempo de inferencia em CPU para cada combinacao janela x taxa.

Uso:
    .venv\\Scripts\\python.exe tools\\mede_custo_modelo.py
    .venv\\Scripts\\python.exe tools\\mede_custo_modelo.py --repeticoes 20
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sand_trajectory_model as st  # noqa: E402

#: (rotulo, canais, amostras) -- amostras = janela(s) x taxa(Hz)
CONFIGURACOES = [
    ("2,0 s @ 500 Hz (atual)", 32, 1000),
    ("1,0 s @ 500 Hz", 32, 500),
    ("4,0 s @ 250 Hz", 32, 1000),
    ("4,0 s @ 100 Hz", 32, 400),
    ("6,0 s @ 100 Hz", 32, 600),
    ("9,6 s @ 100 Hz (SAND)", 32, 960),
    ("18,0 s @ 100 Hz", 32, 1800),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repeticoes", type=int, default=10,
                        help="Repeticoes do cronometro por configuracao")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--output-seq-len", type=int, default=30)
    return parser.parse_args()


def mede_flops(modelo, canais, amostras):
    """FLOPs de uma janela (se o torch tiver FlopCounterMode)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:                                   # torch antigo
        return None
    entrada = torch.zeros(1, canais, amostras)
    with FlopCounterMode(display=False) as contador:
        with torch.no_grad():
            modelo(entrada)
    return contador.get_total_flops()


def main():
    args = parse_args()
    print(f"torch {torch.__version__} | d_model={args.d_model} | "
          f"n_layers={args.n_layers} | saida {args.output_seq_len} pontos")
    print(f"{'configuracao':<26} {'amostras':>8} {'params':>9} {'MFLOPs':>9} "
          f"{'ms/janela':>10} {'B (INT8)':>9}")
    referencias = {}
    for rotulo, canais, amostras in CONFIGURACOES:
        modelo = st.SANDTrajectory(n_channels=canais, n_timesteps=amostras,
                                   d_model=args.d_model,
                                   n_layers=args.n_layers,
                                   output_seq_len=args.output_seq_len)
        modelo.eval()
        parametros = sum(p.numel() for p in modelo.parameters())
        flops = mede_flops(modelo, canais, amostras)
        entrada = torch.zeros(1, canais, amostras)
        tempos = []
        with torch.no_grad():
            for _ in range(args.repeticoes):
                inicio = time.perf_counter()
                modelo(entrada)
                tempos.append((time.perf_counter() - inicio) * 1e3)
        mediana = statistics.median(tempos)
        # tamanho do modelo quantizado em INT8 (1 byte por parametro)
        kb_int8 = parametros / 1024.0
        mflops = "n/d" if flops is None else f"{flops / 1e6:.1f}"
        print(f"{rotulo:<26} {amostras:>8} {parametros:>9,} {mflops:>9} "
              f"{mediana:>10.1f} {kb_int8:>8.1f}k")
        referencias[rotulo] = (mediana, flops, parametros)
    base = referencias[CONFIGURACOES[0][0]]
    print("\nComparacoes contra a configuracao atual "
          f"({CONFIGURACOES[0][0]}):")
    for rotulo, (mediana, flops, _p) in referencias.items():
        if rotulo == CONFIGURACOES[0][0]:
            continue
        rel_tempo = mediana / base[0]
        extra = ""
        if flops and base[1]:
            extra = f" | FLOPs x{flops / base[1]:.2f}"
        print(f"  {rotulo:<26} tempo x{rel_tempo:.2f}{extra}")
    print("\nLeitura: o custo de ENTRADA (~canais x amostras) domina; decimar a "
          "taxa permite janelas mais longas com custo parecido. Memoria do "
          "modelo em INT8 = 1 byte por parametro (cabe em MCU com 64-128 kB).")
    return 0


if __name__ == "__main__":
    sys.exit(main())