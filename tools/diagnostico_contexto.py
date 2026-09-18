"""Diagnostico de CONTEXTO: quanto de contexto causal cabe em cada fase do trial.

Motivo (pergunta de 18/09): "se eu usar contexto de 4-6 s, teria que aumentar os
movimentos de 4 s?" Nao -- a janela olha para TRAS. O que importa e' quanto
historico REPOUSO existe antes de cada ancora, e se a janela invade algo que nao
deveria. Dois problemas concretos:

  1. ME ancorada no onset: a janela estoura o inicio do trial e entra no trial
     anterior (o repouso antes do cue tem so' 2 s hoje);
  2. **MI (vazamento)**: a pausa entre a ME e a MI tem 2 s, entao qualquer
     contexto maior que ~2 s faz a janela da MI incluir a EXECUCAO do mesmo
     trial -- o modelo poderia "acertar" a MI decodificando o movimento
     executado. Precisa de pausa maior, mascara, ou contexto curto na MI.

Uso:
    .venv\\Scripts\\python.exe tools\\diagnostico_contexto.py --data gravacoes
    .venv\\Scripts\\python.exe tools\\diagnostico_contexto.py --data gravacoes_sinteticas
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eeg_motor_paradigm as emp  # noqa: E402

#: Contextos causais avaliados (s).
CONTEXTOS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
#: Deslocamento com que a janela comeca em relacao a ancora (s).
OFFSET_ME = -0.5          # janela ancorada no onset (planejamento motor)
OFFSET_MI = 0.0           # janela ancorada no inicio da MI


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", default="gravacoes",
                        help="Pasta com <stamp>.csv + <stamp>_eventos.json")
    return parser.parse_args()


def fase_do_evento(payload):
    """Amostra de cada evento em `eventos`, so' o que interessa.

    Devolve a lista de trials com as amostras de: inicio do trial (768),
    inicio/fim da ME (778/779), inicio/fim da MI (780/781) e onset (795).
    """
    codigos = {int(e["code"]): int(e["sample_index"])
               for e in payload.get("eventos", [])}
    fs = float(payload.get("fs", 500.0))
    trials, atual = [], None
    for evento in sorted(payload.get("eventos", []),
                         key=lambda e: int(e["sample_index"])):
        codigo, amostra = int(evento["code"]), int(evento["sample_index"])
        if codigo == emp.CODE_TRIAL:
            if atual:
                trials.append(atual)
            atual = {"inicio": amostra, "fs": fs, "onsets": []}
        elif atual is None:
            continue
        elif codigo == emp.CODE_ME_START:
            atual["me_inicio"] = amostra
        elif codigo == emp.CODE_ME_END:
            atual["me_fim"] = amostra
        elif codigo == emp.CODE_MI_START:
            atual["mi_inicio"] = amostra
        elif codigo == emp.CODE_MI_END:
            atual["mi_fim"] = amostra
        elif codigo == emp.CODE_MOVE_ONSET:
            atual["onsets"].append(amostra)
    if atual:
        trials.append(atual)
    return trials, fs, codigos


def main():
    args = parse_args()
    caminhos = sorted(glob.glob(os.path.join(args.data, "*_eventos.json")))
    if not caminhos:
        print(f"nenhum *_eventos.json em {args.data}")
        return 1
    trials, fs, _ = [], 0.0, None
    for caminho in caminhos:
        with open(caminho, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        lista, fs_arquivo, _ = fase_do_evento(payload)
        trials += lista
        fs = fs_arquivo
    completos = [t for t in trials
                 if all(chave in t for chave in ("me_inicio", "me_fim",
                                                 "mi_inicio", "mi_fim"))]
    print(f"{len(caminhos)} sessao(oes) | {len(trials)} trials "
          f"({len(completos)} com as 4 fases) | fs = {fs:g} Hz")

    repouso_antes_me, pausa_antes_mi = [], []
    for trial in completos:
        repouso_antes_me.append((trial["me_inicio"] - trial["inicio"]) / fs)
        pausa_antes_mi.append((trial["mi_inicio"] - trial["me_fim"]) / fs)
    if not completos:
        print("nenhum trial completo para analisar")
        return 1
    media_repouso = float(np.mean(repouso_antes_me))
    media_pausa = float(np.mean(pausa_antes_mi))
    print(f"historico antes da cue (ME): {media_repouso:.2f} s "
          f"| pausa entre ME e MI: {media_pausa:.2f} s")

    print(f"\n{'contexto':>9} | {'ME (ancora onset)':>26} | "
          f"{'MI (vazamento da ME)':>28}")
    for contexto in CONTEXTOS:
        # ME: a janela comeca em inicio_ME + OFFSET_ME - contexto
        limpo_me = media_repouso + OFFSET_ME          # suteis antes da cue
        estoura = max(0.0, contexto - limpo_me)
        # MI: a janela comeca em inicio_MI - contexto (offset 0)
        vaza = max(0.0, contexto - media_pausa)
        fracao_vaza = min(1.0, vaza / max(contexto, 1e-9))
        print(f"{contexto:>8.1f}s | "
              f"{('OK' if estoura <= 0 else f'invade {estoura:.2f} s'):>26} | "
              f"{('OK' if vaza <= 0 else f'{vaza:.2f} s ({fracao_vaza * 100:.0f}% da janela)'):>28}")
    print("\nLeitura: 'invade' na coluna ME = a janela entra no trial anterior "
          "(aumente o repouso antes do cue). 'vaza' na coluna MI = a janela da "
          "MI contem a EXECUCAO do mesmo trial (aumente a pausa antes da MI, "
          "mascare a ME na perda ou use contexto curto na MI).")
    return 0


if __name__ == "__main__":
    sys.exit(main())