"""Teste das MELHORIAS de 14/10/2026 (item 55 do historico).

1. CODE_TRIAL_PRACTICE (789) unico entre os codigos e com legenda.
2. build_practice_list cobre as 6 condicoes e respeita n (0 = nenhuma).
3. slides_instrucao: 4 slides com as palavras-chave do protocolo.
4. FIX DO BUG DA ESQUERDA: set_phase extrai o lado da mao do SUFIXO da
   condicao ("bola_esquerda" -> "left"); antes arm_cue_side ficava SEMPRE
   vazio e o MediaPipe escolhia a mao errada (trials da esquerda sem
   trajetoria).
5. SharedState: marcador de onset da MI (mark/take consome uma unica vez).
6. parse_args: defaults novos (--trials-teste 6; mi_* desligados).
7. Metadados da sessao registram os novos flags.
"""
import inspect
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eeg_motor_paradigm as p

# --------------------------------------------------------------------- 1)
assert p.CODE_TRIAL_PRACTICE == 789
outros = {p.CODE_TRIAL, p.CODE_TRIAL_CANCEL, p.CODE_SLOWMO_START,
          p.CODE_MOVE_ONSET, p.CODE_LINK_LOST, p.CODE_LINK_OK,
          p.CODE_SESSION, p.CODE_SESSION_END, p.CODE_BLOCK_START,
          p.CODE_BLOCK_END, p.CODE_PAUSE_START, p.CODE_PAUSE_END,
          p.CODE_ME_START, p.CODE_ME_END, p.CODE_MI_START, p.CODE_MI_END,
          *p.CODE_HAND.values(), *p.CODE_CONDITION.values()}
assert p.CODE_TRIAL_PRACTICE not in outros, "789 colide com outro codigo"
assert p.CODE_TRIAL_PRACTICE in p.CODE_NAMES

# --------------------------------------------------------------------- 2)
seis = p.build_practice_list(6, random.Random(1))
assert len(seis) == 6, seis
assert len({t["condicao"] for t in seis}) == 6, "pratica nao cobre as 6"
treze = p.build_practice_list(13, random.Random(2))
assert len(treze) == 13
assert len({t["condicao"] for t in treze}) == 6
assert all(t["condicao"] == f"{t['objeto']}_{t['mao']}" for t in treze)
assert all(t["objeto"] in p.OBJECTS and t["mao"] in p.HANDS for t in treze)
assert p.build_practice_list(0) == []

# --------------------------------------------------------------------- 3)
slides = p.slides_instrucao()
assert len(slides) == 4, slides
texto = " ".join(slides).upper()
for chave in ("ESPACO", "ESQUERDA", "DIREITA", "GARRAFA", "BOLA",
              "CANETA", "EXECUTE", "IMAGINE", "HOME"):
    assert chave in texto, f"slide sem a palavra {chave}"

# --------------------------------------------------------------------- 4)
estado = p.SharedState()
estado.set_phase(p.PHASE_ME, cue="bola_esquerda")
assert estado.arm_cue_side == "left", estado.arm_cue_side
estado.set_phase(p.PHASE_MI, cue="caneta_direita")
assert estado.arm_cue_side == "right", estado.arm_cue_side
estado.set_phase(p.PHASE_HOME, cue="")
assert estado.arm_cue_side == ""

# --------------------------------------------------------------------- 5)
outro = p.SharedState()
outro.set_mi_auto(True)
assert outro.mi_auto_onset is True
assert outro.take_mi_onset() is False       # nada marcado ainda
outro.mark_mi_onset()
assert outro.take_mi_onset() is True        # consome
assert outro.take_mi_onset() is False       # uma unica vez

# --------------------------------------------------------------------- 6)
_argv = sys.argv
sys.argv = ["eeg_motor_paradigm.py", "--source", "gnautilus"]
try:
    args = p.parse_args()
finally:
    sys.argv = _argv
assert args.trials_teste == 6, args.trials_teste
assert args.mi_auto_onset is False
assert args.mi_desenho is False
assert abs(args.mi_auto_timeout - 4.0) < 1e-9
assert args.sem_instrucoes is False

# --------------------------------------------------------------------- 7)
assinatura = inspect.signature(p._build_session_meta)
trials = p._ParadigmController.build_trial_list(50, 5, random.Random(0))
kw = {}
if "priming_dir" in assinatura.parameters:
    kw["priming_dir"] = None
meta = p._build_session_meta(args, trials, args.source,
                             "x.csv", "m.csv", "j.json", **kw)
assert meta["trials_teste_pratica"] == 6, meta["trials_teste_pratica"]
assert meta["mi_auto_onset"] is False
assert meta["mi_desenho"] is False
assert meta["instrucoes"] is True

print("MELHORIAS_14_10_OK",
      f"cod={p.CODE_TRIAL_PRACTICE}",
      f"pratica={len(seis)}", f"slides={len(slides)}")
