"""Suite do `tools/probe_kinect_runtime.py` -- o sensor que ABRE e nao ENTREGA.

O QUE ESTA SUITE TRAVA (bancada de 07/10/2026)
  O Kinect Studio mostrava imagem e todo programa do SDK ficava mudo. A sonda
  mediu o passo que o `PyKinectRuntime` NAO checa (ele guarda os retornos em
  `hres` e nunca compara):

      GetDefaultKinectSensor -> 0x00000000 (S_OK)
      sensor.Open()          -> 0x00000000 (S_OK)
      sensor.IsAvailable     -> FALSE        <-- a causa

  Sem esta distincao, "nao achei o Kinect" (HRESULT de erro) e "achei, abri, e
  ele nao esta' disponivel" (IsAvailable FALSE) viram a MESMA mensagem -- e os
  consertos sao completamente diferentes (driver ausente x link do sensor). Os
  casos abaixo travam:

   1. `_hresult` mostra o codigo em hexa -- e' o que separa S_OK de erro;
   2. `estado_do_servico` NAO depende do idioma: o Windows traduz os rotulos do
      `sc query` (ESTADO/TIPO em portugues) e deixa os VALORES
      (`RUNNING`/`STOPPED`). Procurar o rotulo "STATE" devolveu None na bancada
      de 07/10/2026 -- o parser procura o valor;
   3. servico inexistente devolve (None, None) em vez de estourar;
   4. `observa` reporta o HISTORICO das transicoes de `IsAvailable`, que e' o que
      separa "estado preso" de "link ruim": se ele nunca vira TRUE enquanto o
      operador desliga e religa o USB, o conserto e' de link/driver;
   5. sem sensor, `abre_sensor` devolve motivo claro (nao um ponteiro invalido);
   6. o codigo de saida distingue os tres estados (0 disponivel+frame, 1
      disponivel sem frame, 2 indisponivel, 3 nem abriu).
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
if str(RAIZ / "tools") not in sys.path:
    sys.path.insert(0, str(RAIZ / "tools"))

import probe_kinect_runtime as pr                           # noqa: E402

casos = 0

# --- 1) HRESULT em hexa -----------------------------------------------------
assert pr._hresult(0) == "0x00000000", pr._hresult(0)
assert pr._hresult(-2147024894) == "0x80070002", pr._hresult(-2147024894)
assert pr._hresult(0x80070490) == "0x80070490", pr._hresult(0x80070490)
casos += 1

# --- 2) o nome do servico e' o do SDK ---------------------------------------
assert pr.SERVICO == "KinectMonitor", pr.SERVICO
casos += 1

# --- 3) estado_do_servico: independente do IDIOMA ---------------------------
#: O `sc query` em portugues traz `ESTADO : 4  RUNNING` -- procurar "STATE"
#: falhava. O parser tem de achar RUNNING/STOPPED pelo VALOR.
import subprocess                                           # noqa: E402

original_run = subprocess.run


class _Proc:
    def __init__(self, texto):
        self.stdout = texto
        self.stderr = ""
        self.returncode = 0


ROTULOS = (
    #: (rotulo em portugues, rotulo em ingles) -- os valores sao os mesmos.
    ("NOME_DO_SERVICO: KinectMonitor \n    TIPO : 10  WIN32_OWN_PROCESS  \n"
     "    ESTADO : 4  RUNNING \n    AGUARDAR_DICA : 0x0\n"),
    ("SERVICE_NAME: KinectMonitor \n    TYPE : 10  WIN32_OWN_PROCESS  \n"
     "    STATE : 4  RUNNING \n    WAIT_HINT : 0x0\n"),
)
for texto in ROTULOS:
    subprocess.run = lambda *a, **k: _Proc(texto)
    status, inicio = pr.estado_do_servico()
    assert status == "RUNNING", (status, texto)
    subprocess.run = lambda *a, **k: _Proc(texto.replace("RUNNING", "STOPPED"))
    status, _ = pr.estado_do_servico()
    assert status == "STOPPED", status
casos += 1

#: Servico inexistente -> (None, None), sem excecao.
subprocess.run = lambda *a, **k: _Proc(
    "[SC] EnumQueryServicesStatus: OpenService FALHA 1060:\n"
    "O servico especificado nao existe como servico instalado.\n")
assert pr.estado_do_servico() == (None, None), pr.estado_do_servico()
subprocess.run = original_run
casos += 1

# --- 4) observa: reporta a TRANSICAO de IsAvailable -------------------------
class SensorFalso:
    def __init__(self, valores):
        self.valores = list(valores)
        self.i = 0

    @property
    def IsAvailable(self):
        valor = self.valores[min(self.i, len(self.valores) - 1)]
        self.i += 1
        return valor


class RuntimeFalso:
    """Runtime de cor falso: entrega frame a partir da n-esima consulta."""

    def __init__(self, quando=10 ** 9):
        self.quando = quando
        self.consultas = 0
        self.fechou = 0

    def has_new_color_frame(self):
        self.consultas += 1
        return self.consultas >= self.quando

    def close(self):
        self.fechou += 1


original_abre = pr.abre_sensor
original_runtime = pr.PyKinectRuntime

#: Caso A: IsAvailable nunca vira TRUE -> viu_disponivel False.
falso = SensorFalso([False] * 50)
pr.abre_sensor = lambda: (falso, 0, 0, None)
pr.PyKinectRuntime = lambda *a, **k: RuntimeFalso()
viu_disp, viu_frame, historico = pr.observa(0.3, intervalo=0.05)
assert viu_disp is False and viu_frame is False, (viu_disp, viu_frame)
assert any("IsAvailable=False" in o for _, o in historico), historico
casos += 1

#: Caso B: IsAvailable vira TRUE no meio e depois vem frame -> os dois True, e o
#: historico registra a transicao (e' o que diz "o sensor inicializou").
falso = SensorFalso([False, False, True])
runtime = RuntimeFalso(quando=3)
pr.abre_sensor = lambda: (falso, 0, 0, None)
pr.PyKinectRuntime = lambda *a, **k: runtime
viu_disp, viu_frame, historico = pr.observa(0.5, intervalo=0.05)
assert viu_disp is True, historico
assert viu_frame is True, historico
assert any("IsAvailable=True" in o for _, o in historico), historico
assert runtime.fechou == 1, "o runtime tem de ser FECHADO no fim"
casos += 1

#: Caso C: sem sensor -> motivo claro e nada de excecao.
pr.abre_sensor = lambda: (None, -2147024894, None,
                          "GetDefaultKinectSensor devolveu ponteiro NULO")
viu_disp, viu_frame, historico = pr.observa(0.1, intervalo=0.05)
assert viu_disp is False and viu_frame is False, (viu_disp, viu_frame)
assert historico and "ponteiro NULO" in historico[0][1], historico
casos += 1

pr.abre_sensor = original_abre
pr.PyKinectRuntime = original_runtime
casos += 1

# --- 5) a instancia do dispositivo e' lida sem estourar ---------------------
#: Ela e' o que separa "enumerou de novo e ficou mudo" de "o Windows nao viu
#: religacao nenhuma" (medido em 07/10/2026: instancia IGUAL depois do replug).
instancia, suspenso = pr.instancia_do_sensor()
if instancia is not None:
    assert isinstance(instancia, str) and instancia, instancia
    assert suspenso in (0, 1, None), suspenso
casos += 1

# --- 6) nesta maquina: o estado e' coerente com o que a sonda reporta --------
#: Nao trava "esta' quebrado" (a maquina muda), mas trava a COERENCIA: se o
#: sensor estiver indisponivel, `main` tem de sair com 2 -- e nao com 0.
import io                                                   # noqa: E402
import contextlib                                           # noqa: E402

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    codigo = pr.main(["--sem-frame"])
saida = buffer.getvalue()
assert "GetDefaultKinectSensor:" in saida, saida
assert "sensor.IsAvailable" in saida, saida
assert "instancia no registro" in saida, \
    "a identidade do dispositivo tem de aparecer (e' o teste do replug)"
assert codigo in (0, 2, 3), codigo
if "IsAvailable    : FALSE" in saida:
    assert codigo == 2, ("IsAvailable FALSE e codigo %d: os dois tem de casar"
                         % codigo)
    assert "IsAvailable FALSE" in saida or "nao esta' inicializado" in saida, saida
casos += 1

print("PROBE_KINECT_RUNTIME_OK: %d casos (HRESULT em hexa; servico pelo VALOR e "
      "nao pelo rotulo traduzido; servico inexistente sem excecao; historico das "
      "transicoes de IsAvailable; sem sensor com motivo; instancia do dispositivo "
      "lida sem estourar; codigo de saida casa com o estado)" % casos)
