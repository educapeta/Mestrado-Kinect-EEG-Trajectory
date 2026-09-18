"""Compara a VARIABILIDADE do alvo concorrente x preditivo (armadilha do PCC).

Motivo: o PCC sozinho engana. Se o alvo preditivo (o futuro) tem pouca
variacao entre trials -- por exemplo, quando o punho ja esta' voltando ao home
-- um modelo que quase sempre responde "a media" obtem PCC alto sem ter
aprendido nada. Este utilitario imprime, por eixo, o desvio padrao do alvo em
cada configuracao, para que a comparacao de PCC seja honesta.

Uso:
    .venv\\Scripts\\python.exe tools\\compara_alvos.py --data gravacoes_sinteticas
    .venv\\Scripts\\python.exe tools\\compara_alvos.py --data gravacoes --canais 32
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sand_traj_treino as treino  # noqa: E402

CONFIGURACOES = [
    ("cue concorrente      (778, [0,0 , 2,0] s)",
     ["--event-code", "778", "--window-start-sec", "0.0", "--window-sec", "2.0"]),
    ("onset concorrente    (795, [-0,5 , 1,5] s)",
     ["--event-code", "795", "--window-start-sec", "-0.5", "--window-sec", "2.0"]),
    ("onset preditivo      (795, alvo [+1,5 , 2,5] s)",
     ["--event-code", "795", "--window-start-sec", "-0.5", "--window-sec", "2.0",
      "--target-start-sec", "1.5", "--target-end-sec", "2.5"]),
    ("onset preditivo longo(795, alvo [-0,5 , 1,5] s -> futuro [+0,5 , 2,5] s)",
     ["--event-code", "795", "--window-start-sec", "-0.5", "--window-sec", "2.0",
      "--target-start-sec", "0.5", "--target-end-sec", "2.5"]),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", default="gravacoes_sinteticas")
    parser.add_argument("--canal-count", type=int, default=32)
    parser.add_argument("--output-seq-len", type=int, default=30)
    parser.add_argument("--fs", type=float, default=500.0)
    parser.add_argument("--max-sessoes", type=int, default=0,
                        help="Usa apenas as N primeiras sessoes (0 = todas)")
    return parser.parse_args()


def pcc_por_eixo(previsao, alvo):
    """PCC por eixo entre duas series (N, seq, 3)."""
    resultados = []
    for eixo in range(3):
        a = np.asarray(previsao)[..., eixo].ravel()
        b = np.asarray(alvo)[..., eixo].ravel()
        if a.std() < 1e-9 or b.std() < 1e-9:
            resultados.append(0.0)
        else:
            resultados.append(float(np.corrcoef(a, b)[0, 1]))
    return resultados


def baseline_media(y, fracao_val=0.2):
    """PCC do alvo MEDIO do treino contra a validacao (baseline SEM EEG).

    Se o modelo nao superar isto, ele apenas decorou a trajetoria tipica e o
    PCC alto nao significa decodificacao.
    """
    n_val = max(1, int(y.shape[0] * fracao_val))
    media = y[:-n_val].mean(axis=0)                      # (seq, 3)
    repetida = np.repeat(media[None, :, :], n_val, axis=0)
    return pcc_por_eixo(repetida, y[-n_val:])


def main():
    args = parse_args()
    gravacoes = treino.find_recordings(args.data)
    if args.max_sessoes > 0:
        gravacoes = gravacoes[:args.max_sessoes]
    if not gravacoes:
        print(f"nenhuma gravacao em {args.data}")
        return 1
    print(f"{len(gravacoes)} sessao(oes) em {args.data}")
    print(f"{'configuracao':<52} {'N':>4} {'ampl. x':>8} {'ampl. y':>8} "
          f"{'ampl. z':>8} {'ampl. tot':>9} | "
          f"{'PCC media x':>11} {'y':>6} {'z':>6}")
    for rotulo, extra in CONFIGURACOES:
        guardado = sys.argv
        sys.argv = ["treino", "--data", args.data, "--fs", str(args.fs),
                    "--output-seq-len", str(args.output_seq_len),
                    "--channel-count", str(args.canal_count)] + extra
        try:
            cfg = treino.parse_args()
        finally:
            sys.argv = guardado
        _, y, _, _ = treino.load_all_epochs(gravacoes, cfg)
        amplitude = y.max(axis=1) - y.min(axis=1)      # por eixo, por janela
        media = amplitude.mean(axis=0)
        base = baseline_media(y)
        print(f"{rotulo:<52} {y.shape[0]:>4} {media[0]:>8.4f} {media[1]:>8.4f} "
              f"{media[2]:>8.4f} {media.mean():>9.4f} | "
              f"{base[0]:>11.3f} {base[1]:>6.3f} {base[2]:>6.3f}")
    print("\nRegra de leitura 1: so' compare PCC entre configuracoes com "
          "amplitude de alvo parecida.")
    print("Regra de leitura 2: o modelo so' decodifica de verdade se superar o "
          "PCC da coluna 'PCC media' (alvo medio do treino, baseline SEM EEG).")
    return 0


if __name__ == "__main__":
    sys.exit(main())