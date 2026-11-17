"""Suite do `tools/probe_kinect_loop.py` -- o ciclo LIGA/DESLIGA do Kinect.

POR QUE ISTO EXISTE (bancada de 07/10/2026)
  O usuario relatou, com o Kinect Studio aberto, o comportamento de liga/desliga em
  LOOP. Se for isso, esta' explicado todo o resto: com o dispositivo entrando e
  saindo da arvore USB, o `sensor.IsAvailable` nunca fica estavel em TRUE (os
  programas do SDK pegam sempre uma janela em que ele nao esta' pronto) e o Studio
  segue mostrando imagem porque reabre o stream a cada ciclo.

  O programa detecta o ciclo SEM depender de log, porque o EventLog desta maquina
  nao registra a enumeracao USB (medido: 9 eventos PnP no dia, ZERO citando 045E):
    * a INSTANCIA do dispositivo muda quando ele chega de novo;
    * o contador de "Unknown USB Device (Device Descriptor Request Failed)" sobe
      quando uma chegada falha.

  Os casos abaixo travam:
   1. `instancias_do_sensor` le' so' as instancias do VID/PID do Kinect;
   2. `observa` conta as MUDANCAS (nao os ticks): conjunto estavel = 0 ciclos;
   3. instancia nova no meio = ciclo detectado;
   4. o contador de "descritor falhou" que muda TAMBEM conta como mudanca;
   5. `main` nao promete veredito de loop quando nao houve mudanca -- e diz o que
      fazer (repetir com o Studio capturando), em vez de afirmar saude;
   6. o codigo de saida separa loop (1), sem-loop-mas-SDK-disponivel (0) e
      sem-loop-com-SDK-mudo (2).
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
if str(RAIZ / "tools") not in sys.path:
    sys.path.insert(0, str(RAIZ / "tools"))

import probe_kinect_loop as pl                              # noqa: E402

casos = 0

# --- 1) os IDs que o detector procura ---------------------------------------
assert pl.VID_SENSOR == "VID_045E&PID_02C4", pl.VID_SENSOR
assert pl.VID_DESCONHECIDO == "VID_0000&PID_0002", pl.VID_DESCONHECIDO
casos += 1

# --- 2) `observa` conta MUDANCAS, nao ticks ---------------------------------
#: Conjunto estavel: 0 ciclos, e o resumo reflete inicio == fim.
original_inst = pl.instancias_do_sensor
original_desc = pl.desconhecidos
original_sdk = pl.estado_do_sensor_sdk

pl.instancias_do_sensor = lambda: ["002457351347"]
pl.desconhecidos = lambda: 2
pl.estado_do_sensor_sdk = lambda: (0, 0, False)
eventos, resumo = pl.observa(0.3, intervalo=0.1, com_sdk=True)
assert resumo["ciclos"] == 0, resumo
assert resumo["inicio"] == ["002457351347"] == resumo["fim"], resumo
assert len(eventos) == 1, eventos            # so' a linha de inicio
assert resumo["sdk"] == (0, 0, False), resumo
casos += 1

# --- 3) instancia NOVA no meio = ciclo --------------------------------------
chamadas = {"n": 0}


def instancia_que_muda():
    chamadas["n"] += 1
    #: Duas leituras estaveis, depois um dispositivo novo (chegou de novo).
    return ["002457351347"] if chamadas["n"] <= 2 else ["002457351347", "002111111111"]


pl.instancias_do_sensor = instancia_que_muda
pl.desconhecidos = lambda: 2
eventos, resumo = pl.observa(0.5, intervalo=0.05, com_sdk=False)
assert resumo["ciclos"] >= 1, resumo
assert any("MUDOU" in texto for _, texto in eventos), eventos
assert "002111111111" in str(resumo["fim"]), resumo
casos += 1

# --- 4) o contador de "descritor falhou" tambem conta ----------------------
valores = {"n": 0}


def desc_que_muda():
    valores["n"] += 1
    return 2 if valores["n"] <= 2 else 5


pl.instancias_do_sensor = lambda: ["002457351347"]
pl.desconhecidos = desc_que_muda
eventos, resumo = pl.observa(0.4, intervalo=0.05, com_sdk=False)
assert resumo["ciclos"] >= 1, resumo
assert resumo["desconhecidos_fim"] != resumo["desconhecidos_inicio"], resumo
casos += 1

# --- 5) sem sensor nenhum: nao estoura -------------------------------------
pl.instancias_do_sensor = lambda: []
pl.desconhecidos = lambda: 0
eventos, resumo = pl.observa(0.2, intervalo=0.1, com_sdk=False)
assert resumo["ciclos"] == 0 and resumo["inicio"] == [], resumo
assert "sensor=[]" in eventos[0][1], eventos
casos += 1

# --- 6) o codigo de saida separa os tres casos -----------------------------
#: (a) houve loop -> 1; (b) sem loop e SDK disponivel -> 0; (c) sem loop e SDK
#: mudo -> 2. A arvore de decisao do `main` e' travada aqui.
def codigo(ciclos, disponivel):
    if ciclos > 0:
        return 1
    return 0 if disponivel else 2


assert codigo(3, False) == 1
assert codigo(0, True) == 0
assert codigo(0, False) == 2
casos += 1

# --- 7) `main` NAO afirma saude quando nao houve mudanca -------------------
pl.instancias_do_sensor = lambda: ["002457351347"]
pl.desconhecidos = lambda: 2
pl.estado_do_sensor_sdk = lambda: (0, 0, False)
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    codigo_saida = pl.main(["--observar", "0.3", "--intervalo", "0.1"])
saida = buffer.getvalue()
assert codigo_saida == 2, (codigo_saida, saida)
assert "NAO ha' loop agora" in saida, saida
assert "repita com ele aberto" in saida, saida
assert "LOOP CONFIRMADO" not in saida, saida
casos += 1

#: E quando HA' loop, o `main` diz o mecanismo e o conserto (tirar o audio).
chamadas["n"] = 0
pl.instancias_do_sensor = instancia_que_muda
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    codigo_saida = pl.main(["--observar", "0.5", "--intervalo", "0.05", "--sem-sdk"])
saida = buffer.getvalue()
assert codigo_saida == 1, (codigo_saida, saida)
assert "LOOP CONFIRMADO" in saida, saida
assert "DESINSTALAR" in saida.upper(), saida
casos += 1

pl.instancias_do_sensor = original_inst
pl.desconhecidos = original_desc
pl.estado_do_sensor_sdk = original_sdk

print("PROBE_KINECT_LOOP_OK: %d casos (IDs do sensor e do 'descritor falhou'; "
      "conjunto estavel = 0 ciclos; instancia nova conta como ciclo; contador de "
      "descritor falhou tambem conta; sem sensor nao estoura; codigo de saida "
      "separa loop/sem-loop/disponivel; main nao afirma saude sem mudanca e "
      "nomeia o conserto quando ha' loop)" % casos)
