"""DEPOIS DO REINICIO: o Kinect passou a estar disponivel?

Rode isto assim que o Windows voltar (o Kinect Studio FECHADO):

    .venv\\Scripts\\python.exe tools\\pos_reinicio_kinect.py

Ele responde tres coisas, na ordem que importa:
  1. `sensor.IsAvailable` -- o numero que estava FALSE e e' a causa da mudez;
  2. o caminho do PROJETO (`sc.abre_kinect`, com reabertura) entregou frame?
  3. o mapa indice->NOME das webcams continua o mesmo (o indice do OpenCV e' a
     POSICAO no USB: desabilitar/reabilitar dispositivo de audio nao mexe nisso,
     mas conferir e' barato e evita calibrar a camera errada).

Codigo de saida: 0 = pode ir para a Fase 1 (freeze); 1 = ainda mudo.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import kinect_compat                                        # noqa: E402

kinect_compat.aplicar()

import stereo_calibration as sc                             # noqa: E402
import probe_kinect_runtime as pr                           # noqa: E402

FALHAS = []


def linha(titulo):
    print("\n" + "=" * 72)
    print(titulo)
    print("=" * 72)


linha("1) o numero que decidia tudo: sensor.IsAvailable")
sensor, hres_get, hres_open, motivo = pr.abre_sensor()
if sensor is None:
    print("  GetDefaultKinectSensor: %s" % pr._hresult(hres_get))
    print("  ERRO: %s" % motivo)
    print("  -> o sensor nem foi encontrado: driver/SDK, e nao link de stream.")
    FALHAS.append("sensor ausente")
    disponivel = False
else:
    print("  GetDefaultKinectSensor: %s" % pr._hresult(hres_get))
    print("  sensor.Open()         : %s" % pr._hresult(hres_open))
    disponivel = bool(sensor.IsAvailable)
    print("  sensor.IsAvailable    : %s  (antes do reinicio: FALSE)"
          % ("TRUE  <-- MUDOU" if disponivel else "FALSE <-- ainda mudo"))
    if not disponivel:
        FALHAS.append("IsAvailable FALSE")
        print("     Ainda FALSE. Se voce DESABILITOU o dispositivo de audio do")
        print("     Kinect, o teste NAO confirma a causa -- o suspeito continua")
        print("     sendo o link: outra porta USB 3.0 (de outra controladora),")
        print("     fonte propria ligada ANTES do USB, e depois reparar o SDK.")

if disponivel:
    linha("2) o caminho do PROJETO entrega frame?")
    entrega = False
    try:
        runtime = sc.abre_kinect(espera_s=20.0)
        entrega = True
        try:
            runtime.close()
        except Exception:                                   # noqa: BLE001
            pass
        print("  RESULTADO: ENTREGA frame.")
    except RuntimeError as exc:
        print("  RESULTADO: nao entregou.")
        print("  %s" % exc)
        FALHAS.append("abre_kinect nao entregou frame")
else:
    entrega = False

linha("3) webcams (indice -> nome)")
try:
    import camera_names
    for indice in range(4):
        nome = camera_names.nome_da_camera(indice)
        print("  indice %d: %s" % (indice, nome if nome else "(nada)"))
except Exception as exc:                                    # noqa: BLE001 - sonda
    print("  indisponivel: %s" % exc)

linha("RESUMO")
if entrega:
    print("O KINECT ESTA' ENTREGANDO FRAME.")
    print("Proximo passo (bancada), na ordem:")
    print("  1. tools\\checa_kinect.py  (tem de sair com 0)")
    print("  2. python freeze_kinect_direto.py   <- Fase 1")
    print("  3. tools\\conserta_intrinsecas_kinect.py --sim   (aposenta o paliativo)")
    print("  4. python stereo_calibration.py --aux-index 0  e  --aux-index 1")
    print("  5. tools\\diagnostico_calibracao.py  e  a tecla T no groundtruth tool")
    sys.exit(0)
print("AINDA NAO: %s" % "; ".join(FALHAS))
print("Volte para tools\\probe_kinect_runtime.py --observar 60 e desligue/religue")
print("o USB: se o IsAvailable NAO virar TRUE nem durante a religacao, o conserto")
print("e' de LINK (porta/controladora/fonte) ou do proprio SDK.")
sys.exit(1)
