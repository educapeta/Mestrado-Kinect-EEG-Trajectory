"""Teste do camera_uvc.trava_exposicao -- sem hardware (captura FALSA).

O contrato e' o RELATORIO honesto: cada set() e' lido de volta e a funcao diz o
que a camera OBEDECEU e o que IGNOROU (DirectShow ja' foi medido ignorando
BUFFERSIZE na Trust) e nunca lanca excecao (ela roda no caminho de abertura de
camera; nao e' ali que o programa pode quebrar).

    .venv\\Scripts\\python.exe test_camera_uvc.py
"""
import io
import math
import sys
from contextlib import redirect_stdout

import cv2

from camera_uvc import (_interp_tempo, parse_controle_por_camera,  # noqa: E402
                        trava_exposicao, valor_por_camera)

casos = 0


class CapturaFake:
    """Dupla set/get com comportamento configuravel (obedece ou ignora)."""

    def __init__(self, leitura_antes=None, obedece=True, com_erro=False):
        self.leituras = dict(leitura_antes or {})
        self.obedece = obedece
        self.com_erro = com_erro
        self.sets = []

    def set(self, prop, valor):
        if self.com_erro:
            raise RuntimeError("driver cascou")
        self.sets.append((prop, float(valor)))
        if self.obedece:
            self.leituras[prop] = float(valor)
        return self.obedece

    def get(self, prop):
        if self.com_erro:
            raise RuntimeError("driver cascou")
        if not self.obedece and prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            return 0.75                      # continua AUTO: ignorou o set
        return self.leituras.get(prop, 0.0)


# 1) Camera que OBEDECE: manual confirmado e tempo pedido aplicado ----------
cap = CapturaFake({cv2.CAP_PROP_AUTO_EXPOSURE: 0.75,
                   cv2.CAP_PROP_EXPOSURE: -4.0})
rel = trava_exposicao(cap, exposicao=-7, verbose=False)
casos += 1
assert rel["auto_exposicao"] <= 0.5, f"manual nao confirmado: {rel}"
assert rel["tempo"] == -7.0, f"tempo nao aplicado: {rel}"
assert (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) in cap.sets, cap.sets
assert (cv2.CAP_PROP_EXPOSURE, -7.0) in cap.sets, cap.sets
if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
    assert (cv2.CAP_PROP_AUTOFOCUS, 0.0) in cap.sets, "foco nao desligado"

# 2) Camera que IGNORA tudo: relatorio diz IGNORADA (nao afirma sucesso) ----
cap = CapturaFake(obedece=False)
rel = trava_exposicao(cap, verbose=False)
casos += 1
assert rel["auto_exposicao"] > 0.5, f"mentiu sobre a auto-expo: {rel}"

# 3) set()/get() lancando excecao: nunca propaga (caminho de abertura) ------
cap = CapturaFake(com_erro=True)
rel = trava_exposicao(cap, exposicao=-7, fps=60, verbose=False)
casos += 1
assert isinstance(rel, dict), rel

# 4) travar_foco=False nao toca no autofocus --------------------------------
cap = CapturaFake(obedece=True)
trava_exposicao(cap, travar_foco=False, verbose=False)
casos += 1
if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
    assert all(p != cv2.CAP_PROP_AUTOFOCUS for p, _ in cap.sets), cap.sets

# 5) Sem exposicao explicita: SO' le (nao seta o tempo) ---------------------
cap = CapturaFake({cv2.CAP_PROP_EXPOSURE: -5.0})
rel = trava_exposicao(cap, verbose=False)
casos += 1
assert all(p != cv2.CAP_PROP_EXPOSURE for p, _ in cap.sets), \
    "sem --exposicao o tempo nao pode ser setado"
assert rel["tempo"] == -5.0, rel

# 6) verbose: UMA linha com os marcadores do que pegou/ignorou --------------
buf = io.StringIO()
with redirect_stdout(buf):
    trava_exposicao(CapturaFake(obedece=True), rotulo="aux 1")
texto = buf.getvalue()
casos += 1
assert texto.count("\n") == 1, f"esperava 1 linha: {texto!r}"
assert "[exposicao aux 1]" in texto and "auto-expo->manual" in texto, texto
assert "~1/" in texto or "tempo" in texto, texto

# 7) Heuristica de tempo (log2 -> 1/N s) ------------------------------------
casos += 1
assert _interp_tempo(-7) == "~1/128 s", _interp_tempo(-7)
assert _interp_tempo(-4) == "~1/16 s", _interp_tempo(-4)
assert _interp_tempo(100.0) == "bruto", _interp_tempo(100.0)
assert _interp_tempo(float("nan")) == "?"

# 8) MJPEG: define_modo respeita a ORDEM (W/H ANTES do fourcc -- medido: a
#    ordem inversa renegocia e volta pra YUY2, caindo a 8-10 fps), le_fourcc
#    reporta o modo REAL e o relatorio mede o fps que a camera ENTREGA (a
#    propriedade mentiu: pediu 60, entregou 4-8 -- medido na bancada).
from camera_uvc import define_modo, le_fourcc, mede_fps  # noqa: E402


class CapturaComFrames(CapturaFake):
    """Captura falsa que 'entrega' frames (read sempre ok)."""

    def read(self):
        return True, None


cap = CapturaComFrames(obedece=True)
define_modo(cap, 1280, 720, mjpeg=True)
casos += 1
assert (cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")) in cap.sets, "MJPEG nao pedido"
props = [p for p, _ in cap.sets]
assert (props.index(cv2.CAP_PROP_FRAME_WIDTH)
        < props.index(cv2.CAP_PROP_FOURCC)), \
    "W/H tem de vir ANTES do fourcc (ordem inversa volta pra YUY2, medido)"
assert le_fourcc(cap) == "MJPG", le_fourcc(cap)
rel = trava_exposicao(cap, fps=60, verbose=False)
assert math.isfinite(rel["fps_real"]) and rel["fps_real"] > 0, rel
fps_medido = mede_fps(CapturaComFrames())
assert math.isfinite(fps_medido) and fps_medido > 0, fps_medido

# 9) Tempo LONGO no automatico -> AVISO (a Trust estava em -4 ~ 1/16 s) ------
cap = CapturaFake({cv2.CAP_PROP_EXPOSURE: -4.0})
buf = io.StringIO()
with redirect_stdout(buf):
    trava_exposicao(cap, rotulo="aux 1")
texto = buf.getvalue()
casos += 1
assert "LONGO" in texto and "~1/16 s" in texto, texto
# tempo curto (SIGMA ja' em -7) NAO avisa
buf = io.StringIO()
with redirect_stdout(buf):
    trava_exposicao(CapturaFake({cv2.CAP_PROP_EXPOSURE: -7.0}))
assert "LONGO" not in buf.getvalue(), buf.getvalue()

# 10) Exposicao/ganho POR CAMERA (as duas webcams pedem valores diferentes) --
mapa = parse_controle_por_camera(None)
casos += 1
assert mapa["todos"] is None and mapa["indices"] == {}, mapa
assert valor_por_camera(mapa, 0) is None
mapa = parse_controle_por_camera("-6")
assert mapa["todos"] == -6.0 and mapa["indices"] == {}, mapa
assert valor_por_camera(mapa, 0) == -6.0 and valor_por_camera(mapa, 1) == -6.0
mapa = parse_controle_por_camera("0:-6, 1:-4")
assert mapa["indices"] == {0: -6.0, 1: -4.0}, mapa
assert valor_por_camera(mapa, 0) == -6.0
assert valor_por_camera(mapa, 1) == -4.0
assert valor_por_camera(mapa, 2) is None
assert valor_por_camera(parse_controle_por_camera(5), 3) == 5.0
print("10) OK: exposicao/ganho POR CAMERA (valor unico, 'i:N,i:N', espacos e "
      "float; camera sem valor devolve None)")

print(f"CAMERA_UVC_OK: {casos}/10 casos (trava honesta, MJPEG, fps REAL, "
      "aviso de tempo longo e controle POR CAMERA)")
sys.exit(0)
