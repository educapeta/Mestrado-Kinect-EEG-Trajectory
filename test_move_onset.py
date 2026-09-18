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
7. coluna KT_onset presente e valendo 1 apenas na amostra publicada.
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
assert p.N_MOTION_COLS == 43, p.N_MOTION_COLS
assert p.MOTION_COLUMNS[-1] == "KT_onset", p.MOTION_COLUMNS[-1]
coluna = p.MOTION_COLUMNS.index("KT_onset")
base = dict(p.SharedState().motion)
assert base.get("onset") == 0
linha_1 = p.build_motion_row({**base, "onset": 1}, None, None)
linha_0 = p.build_motion_row({**base, "onset": 0}, None, None)
assert len(linha_1) == p.N_MOTION_COLS
assert linha_1[coluna] == 1.0 and linha_0[coluna] == 0.0
assert p.CODE_MOVE_ONSET in p.CODE_NAMES

print("MOVE_ONSET_OK: 7/7 casos (parada, movimento, pico de ruido, fase, NaN, "
      "reset entre trials, coluna KT_onset)")
sys.exit(0)