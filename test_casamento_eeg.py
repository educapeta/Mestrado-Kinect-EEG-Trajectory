#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_casamento_eeg.py -- casamento groundtruth x EEG (funcoes puras).

Cobre `tools/casamento_eeg_groundtruth.py`:
  - mono_para_sample_index: interpolacao, clamp fora da ancora, erros;
  - ancora_dos_eventos: ordenacao, campos faltantes, formatos legados;
  - marcadores_do_csv: primeira amostra por codigo, filtro, 0 ignorado;
  - integracao com os arquivos REAIS de gravacoes/ (pulada se nao houver).

Rodar: .venv\\Scripts\\python.exe test_casamento_eeg.py
"""

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import casamento_eeg_groundtruth as ceg  # noqa: E402


class _Stream:
    def __init__(self):
        self.buf = io.StringIO()

    def write(self, s):
        self.buf.write(s)


def teste_mono_para_sample_index_interpolacao():
    ancoras = [(100.0, 0), (110.0, 5000)]
    si = ceg.mono_para_sample_index([100.0, 105.0, 110.0], ancoras)
    assert np.allclose(si, [0.0, 2500.0, 5000.0]), si


def teste_mono_para_sample_index_clamp():
    # fora da ancora: clamp na primeira/ultima (movimento antes/depois)
    ancoras = [(100.0, 0), (110.0, 5000)]
    si = ceg.mono_para_sample_index([90.0, 120.0], ancoras)
    assert np.allclose(si, [0.0, 5000.0]), si


def teste_mono_para_sample_index_exige_duas_ancoras():
    try:
        ceg.mono_para_sample_index([1.0], [(1.0, 0)])
    except ValueError:
        return
    raise AssertionError("deveria exigir 2+ ancoras")


def teste_ancora_dos_eventos_ordena_e_ignora_incompletos():
    eventos = [
        {"monotonic_s": 10.0, "sample_index": 500},
        {"monotonic_s": 5.0, "sample_index": 0},
        {"sample_index": 250},          # sem relogio
        {"monotonic_s": 7.0},            # sem sample_index
        "lixo",                          # nao-dict
    ]
    a = ceg.ancora_dos_eventos(eventos)
    assert a == [(5.0, 0), (10.0, 500)], a


def teste_ancora_aceita_t_mono_s_legado():
    a = ceg.ancora_dos_eventos([{"t_mono_s": 3.0, "sample_index": 7}])
    assert a == [(3.0, 7)], a


def teste_marcadores_primeira_amostra():
    marker = np.array([0, 0, 769, 769, 0, 770, 0])
    out = ceg.marcadores_do_csv(marker)
    assert out == {769: (2, 2), 770: (5, 1)}, out


def teste_marcadores_filtro_e_zero_ignorado():
    marker = np.array([0, 899, 0, 769, 898])
    out = ceg.marcadores_do_csv(marker, codigos={769, 898})
    assert out == {769: (3, 1), 898: (4, 1)}, out


def teste_integracao_dados_reais():
    pasta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gravacoes")
    if not os.path.isdir(pasta):
        print("  (sem gravacoes/ -- pulado)")
        return
    prefixos = sorted(f[:-len("_movimento.csv")]
                      for f in os.listdir(pasta)
                      if f.endswith("_movimento.csv"))
    if not prefixos:
        print("  (sem *_movimento.csv -- pulado)")
        return
    # so' sessoes com eventos.json tem ancora (sessoes interrompidas SEM
    # JSON nao casam; o JSON pode ser prefixo_eventos.json ou alinhado)
    def tem_eventos(p):
        fs = os.listdir(pasta)
        return any(f == p + "_eventos.json" or
                   (f.startswith(p) and f.endswith("_eventos.json"))
                   for f in fs)
    valida = [p for p in prefixos if tem_eventos(p)]
    if not valida:
        print("  (nenhuma sessao com eventos.json -- pulado)")
        return
    if not valida:
        print("  (nenhuma sessao com eventos.json -- pulado)")
        return
    s = _Stream()
    rc = ceg.diagnostico(valida[-1], pasta, stream=s)
    saida = s.buf.getvalue()
    assert rc == 0, rc
    # sessao interrompida (sem ancora) tem CASAMENTO: NAO -- o teste aceita
    # os dois vereditos; o que importa e' o diagnostico nao crashar e falar
    assert ("CASAMENTO: SIM" in saida) or ("CASAMENTO: NAO" in saida), saida
    if "CASAMENTO: SIM" in saida:
        # BUG INTERNO corrigido (19/10): o comentario acima diz que o teste
        # aceita OS DOIS vereditos, mas a assercao incondicional abaixo
        # exigia "sample_index equivalente" -- que so' existe quando houve
        # OVERLAP de relogio. Uma gravacao real de teste (11:42, movimento
        # parado antes da ancora do EEG) quebrou o teste por aqui.
        assert "sample_index equivalente" in saida, saida


def main():
    testes = [v for k, v in sorted(globals().items()) if k.startswith("teste_")]
    falhas = 0
    for t in testes:
        try:
            t()
            print(f"OK   {t.__name__}")
        except AssertionError as e:
            falhas += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"{len(testes) - falhas}/{len(testes)}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
