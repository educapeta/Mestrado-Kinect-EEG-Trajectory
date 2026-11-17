"""Suite do `tools/preview_kinect.py` -- o preview do Kinect RGB, sem hardware.

O QUE ESTA SUITE TRAVA (07/10/2026)
  O contrato do frame do pykinect2 com o OpenCV: buffer (h*w*4, uint8) em BGRA
  -> reshape -> BGR (o MESMO caminho do `coleta_poses` do freeze); o loop do
  `preview()` com um sensor FALSO (entrega / nao entrega frame) -- inclusive o
  veredito honesto "0 frames = sensor mudo", que e' o estado real da maquina
  em 07/10 (IsAvailable FALSE); o `main()` devolvendo 2 quando `sc.abre_kinect`
  levanta -- a recusa NAO pode virar traceback; e o teto de `--sem-janela`
  (sem GUI nao ha' ESC, o programa nao pode ficar para sempre).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
if str(RAIZ / "tools") not in sys.path:
    sys.path.insert(0, str(RAIZ / "tools"))

import numpy as np                                          # noqa: E402

import preview_kinect as pv                                 # noqa: E402

casos = 0


class _KinectFalso:
    """Mesmo recorte do runtime real: color_frame_desc + has/get/close."""

    def __init__(self, entrega=True, largura=64, altura=48):
        self.color_frame_desc = SimpleNamespace(Width=largura, Height=altura)
        self._entrega = entrega
        self.fechado = False

    def has_new_color_frame(self):
        return self._entrega

    def get_last_color_frame(self):
        altura, largura = self.color_frame_desc.Height, self.color_frame_desc.Width
        return np.arange(altura * largura * 4, dtype=np.uint8)

    def close(self):
        self.fechado = True


# --- 1) o contrato BGRA->BGR e' o caminho do freeze --------------------------
dados = np.zeros(4 * 4, dtype=np.uint8)
dados[0:4] = (1, 2, 3, 4)                       # B, G, R, A
quadro = pv.quadro_de_color(dados, 2, 2)
assert quadro.shape == (2, 2, 3), quadro.shape
assert quadro.dtype == np.uint8, quadro.dtype
assert tuple(quadro[0, 0]) == (1, 2, 3), tuple(quadro[0, 0])   # B,G,R preservados, A fora
casos += 1

# tamanho errado -> erro claro (nao um frame silencioso torto)
try:
    pv.quadro_de_color(np.zeros(7, dtype=np.uint8), 2, 2)
except ValueError:
    casos += 1
else:                                              # pragma: no cover
    raise AssertionError("tamanho errado deveria levantar")

# --- 2) desenha_status pinta em cima e devolve a mesma imagem ----------------
base = np.zeros((48, 64, 3), dtype=np.uint8)
pintado = pv.desenha_status(base.copy(), 30.0, 10, 1.5)
assert pintado.shape == (48, 64, 3), pintado.shape
assert pintado.sum() > 0, "status nao desenhado"
casos += 1

# --- 3) sensor ENTREGANDO: conta frames e respeita a duracao -----------------
falso = _KinectFalso()
agora = time.time()
n_frames, fps = pv.preview(falso, duracao_s=0.3, sem_janela=True,
                           _imprime=lambda *_: None)
decorrido = time.time() - agora
assert n_frames > 0, "sensor entregando deveria contar frame"
assert fps > 0.0, fps
assert decorrido >= 0.25, "saiu antes da duracao pedida (%.2f s)" % decorrido
casos += 1

# --- 4) sensor MUDO: n_frames = 0 em vez de espera infinita ------------------
mudo = _KinectFalso(entrega=False)
avisos = []
n_frames, fps = pv.preview(mudo, duracao_s=0.2, sem_janela=True,
                           _imprime=avisos.append)
assert n_frames == 0, n_frames
assert fps == 0.0, fps
assert any("primeiro frame" in a for a in avisos), avisos


# --- 5) --sem-janela sem duracao: teto de tempo (sem GUI nao ha' ESC) --------
original_limite = pv.LIMITE_SEM_JANELA_S
pv.LIMITE_SEM_JANELA_S = 0.25
try:
    avisos = []
    n_frames, _ = pv.preview(_KinectFalso(entrega=False), duracao_s=0.0,
                             sem_janela=True, _imprime=avisos.append)
finally:
    pv.LIMITE_SEM_JANELA_S = original_limite
assert n_frames == 0, n_frames
assert any("teto de" in a for a in avisos), avisos
casos += 1

# --- 6) main(): abre_kinect falhando -> exit 2 com instrucao (nao traceback) -
original_abre = pv.sc.abre_kinect
pv.sc.abre_kinect = lambda **_: (_ for _ in ()).throw(
    RuntimeError("Kinect mudo (IsAvailable FALSE)"))
try:
    saida = pv.main(["--sem-janela", "--duracao-s", "0.1"])
finally:
    pv.sc.abre_kinect = original_abre
assert saida == 2, saida
casos += 1

# --- 7) o fechamento honesto: 0 frames -> exit 1 -----------------------------
original_abre = pv.sc.abre_kinect
mudo_real = _KinectFalso(entrega=False)
pv.sc.abre_kinect = lambda **_: mudo_real
try:
    saida = pv.main(["--sem-janela", "--duracao-s", "0.15"])
finally:
    pv.sc.abre_kinect = original_abre
assert saida == 1, saida
assert mudo_real.fechado, "runtime deveria fechar em todo caminho"
casos += 1

print("test_preview_kinect: %d casos OK" % casos)

casos += 1
