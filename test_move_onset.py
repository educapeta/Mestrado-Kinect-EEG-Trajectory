"""Teste do detector de INICIO DO MOVIMENTO (onset) e da coluna KT_onset.

Protocolo (motivo cientifico): a janela de EEG de interesse comeca ~0,5 s ANTES
do movimento, porque o planejamento motor comeca 500-1000 ms antes do movimento
visivel (potencial de prontidao) e ~500-350 ms antes o cortex pre-motor / area
motora suplementar organizam a sequencia. Isso exige detectar o onset pelo
Kinect de forma confiavel e marca-lo nos dados (marcador 795 + coluna KT_onset).

Casos:
1. mao parada (ruido abaixo do limiar) -> nao dispara;
2. movimento real -> dispara exatamente 1x, com indice plausivel;
3. pico de ruido isolado (deslocamento < minimo) -> nao dispara; o movimento
   seguinte dispara;
4. fora da fase ME -> nunca dispara;
5. amostras None/NaN -> nao travam e nao disparam;
6. reset() entre trials: cada trial dispara uma vez;
7. coluna KT_onset presente e valendo 1 apenas na amostra publicada;
8. DERIVA/ruido reais da bancada (mao parada, medidos em 21/10/2026) nao podem
   disparar -- o criterio antigo (5 mm / 0,03 m/s) disparava 11x em 6,2 s de
   mao apoiada e era a causa do video de priming comecar antes do movimento;
9. o onset devolvido e' RETROATIVO (inicio do movimento, nao o instante da
   confirmacao) e a trajetoria da trial fica guardada em `track` para o clipe.
"""
import sys

import numpy as np

import eeg_motor_paradigm as p

FS_TRACK = 30.0
DT = 1.0 / FS_TRACK
T0 = 1000.0
RU = np.random.default_rng(0)


def passo(t, posicao, detector, fase_me=True):
    return detector.update(t, None if posicao is None
                           else np.asarray(posicao, float), fase_me)


# 1) mao parada com ruido de 1 mm -> nao pode disparar
d = p.MovementOnsetDetector()
parado = [passo(T0 + i * DT, RU.normal(0, 0.001, 3), d) for i in range(60)]
assert not any(parado), "mao parada nao pode disparar onset"

# 2) movimento real -> dispara 1x
d = p.MovementOnsetDetector()
disparos = [passo(T0 + i * DT, RU.normal(0, 0.001, 3), d) for i in range(15)]
for i in range(30):
    disparos.append(passo(T0 + (15 + i) * DT, [0.02 * i, 0.0, 0.0], d))
assert sum(disparos) == 1, f"deveria disparar 1x (disparou {sum(disparos)})"
assert d.detected and d.onset_time is not None and d.armed
assert 15 <= d.onset_index <= 20, d.onset_index

# 3) pico isolado de 2 mm (0,06 m/s) mas deslocamento total < minimo -> nao conta
d = p.MovementOnsetDetector()
historico = [passo(T0 + i * DT, [0.0, 0.0, 0.0], d) for i in range(10)]
historico.append(passo(T0 + 10 * DT, [0.002, 0.0, 0.0], d))
historico += [passo(T0 + (11 + i) * DT, [0.002, 0.0, 0.0], d) for i in range(4)]
assert not any(historico), "pico isolado nao pode contar como onset"
for i in range(20):
    historico.append(passo(T0 + (15 + i) * DT, [0.002 + 0.02 * i, 0.0, 0.0], d))
assert sum(historico) == 1, sum(historico)

# 4) fora da fase ME -> nunca dispara
d = p.MovementOnsetDetector()
assert not any(passo(T0 + i * DT, [0.05 * i, 0.0, 0.0], d, fase_me=False)
               for i in range(20))

# 5) None/NaN nao travam
d = p.MovementOnsetDetector()
assert not any(passo(T0 + i * DT, None, d) for i in range(5))
assert not any(passo(T0 + i * DT, [np.nan, 0.0, 0.0], d) for i in range(5))
assert d.samples == 10 and not d.detected, (d.samples, d.detected)

# 6) reset entre trials
d = p.MovementOnsetDetector()
for trial in range(2):
    d.reset()
    for i in range(10):
        d.update(T0 + i * DT, [0.02 * i, 0.0, 0.0], True)
    assert d.detected and d.onset_index > 0, f"trial {trial} nao detectou"

# 7) coluna KT_onset
assert p.N_MOTION_COLS == 59, p.N_MOTION_COLS
assert p.MOTION_COLUMNS[-1] == "KT_onset", p.MOTION_COLUMNS[-1]
coluna = p.MOTION_COLUMNS.index("KT_onset")
base = dict(p.SharedState().motion)
assert base.get("onset") == 0
linha_1 = p.build_motion_row({**base, "onset": 1}, None, None)
linha_0 = p.build_motion_row({**base, "onset": 0}, None, None)
assert len(linha_1) == p.N_MOTION_COLS
assert linha_1[coluna] == 1.0 and linha_0[coluna] == 0.0
assert p.CODE_MOVE_ONSET in p.CODE_NAMES

# 8) DERIVA/ruido REAIS da bancada (medido em 21/10/2026, mao apoiada a 30 Hz:
#    passo mediano 3,3 mm; deriva lenta de +-1,5 cm) -> o criterio NOVO nao
#    pode disparar. O ANTIGO (0,03 m/s + 5 mm) disparava 11x em 6,2 s com a mao
#    parada -- e era essa a causa do video de priming comecar antes do
#    movimento. Este caso falha sem o patch.
d = p.MovementOnsetDetector()
RU2 = np.random.default_rng(7)


def mao_parada(t):
    decorrido = t - T0
    deriva = np.array([0.015 * np.sin(2 * np.pi * decorrido / 3.0),
                       0.010 * np.cos(2 * np.pi * decorrido / 4.0), 0.0])
    return deriva + RU2.normal(0, 0.0033, 3)


parado_real = [passo(T0 + i * DT, mao_parada(T0 + i * DT), d)
               for i in range(180)]                      # 6 s a 30 Hz
assert not any(parado_real), "mao parada (ruido real) nao pode disparar onset"

# 9) onset RETROATIVO: mao parada 1,5 s + alcance de 18 cm -> o instante
#    devolvido e' o INICIO do movimento (nao a amostra em que o deslocamento
#    minimo foi confirmado, ~0,2 s depois). Sem o criterio por janela o onset
#    sairia no primeiro jitter (~0,1 s).
d = p.MovementOnsetDetector()
RU3 = np.random.default_rng(3)
disparos = []
for i in range(45):                                       # 1,5 s parado
    disparos.append(passo(T0 + i * DT, RU3.normal(0, 0.002, 3), d))
for i in range(30):                                       # 0,6 s de alcance
    s = i / 29.0
    suave = 3 * s ** 2 - 2 * s ** 3
    disparos.append(passo(T0 + (45 + i) * DT,
                          np.array([0.0, 0.0, -0.18 * suave])
                          + RU3.normal(0, 0.002, 3), d))
assert sum(disparos) == 1, f"deveria disparar 1x (disparou {sum(disparos)})"
assert d.onset_time is not None
erro = (d.onset_time - T0) - 45 * DT                      # + = antes do real
assert -0.12 <= erro <= 0.25, f"onset {erro:+.2f} s fora do inicio real"
assert len(p.MovementOnsetDetector().track) == 0

print("MOVE_ONSET_OK: 9/9 casos (parada, movimento, pico de ruido, fase, NaN, "
      "reset entre trials, coluna KT_onset, ruido real de bancada, onset "
      "retroativo)")
sys.exit(0)
