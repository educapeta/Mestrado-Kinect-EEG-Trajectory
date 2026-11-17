"""Regressao: CALIBRACAO DOS IMUs no inicio de cada sessao (item 101).

Contexto medido (itens 99/100): o firmware calibra o bias do giroscopio UMA vez
no boot e assume a luva PARADA. Boot com a luva em movimento deixou -12,5 dps no
gy da direita = 434 graus/min de deriva de yaw; reiniciando com as luvas paradas
a deriva caiu para -0,20 graus/min (direita) e -0,04 (esquerda).

O que este teste protege:
  1. parse das linhas de calibracao do firmware ("[imu] bias MPU 0x68: (...)");
  2. deteccao da porta serial (CP210x) com porta preferida;
  3. avaliacao do bias medido no UDP (OK / aceitavel / ALTO);
  4. reset pela serial com uma serial FALSA (pulso RTS, captura do boot, leitura
     dos dois biases e descarte das linhas de dados);
  5. medida do bias pelo UDP com socket FALSO (media/std) e porta ocupada;
  6. ritual completo `calibra_imu` com hardware simulado (dois caminhos: OK e
     sem porta serial) -- e o resultado serializavel para o JSON da sessao;
  7. a integracao nos DOIS programas (groundtruth tool e paradigma): import,
     chamada do ritual no inicio, flags e o campo `imu_calibracao` no meta.

Nao usa hardware de verdade: tudo injetado.
"""
from __future__ import annotations

import io
import os
import socket as _socket
import sys

import imu_reset

FALHAS = []


def check(nome, condicao, detalhe=""):
    if condicao:
        print(f"   ok: {nome}")
    else:
        FALHAS.append(nome)
        print(f"   FALHOU: {nome} {detalhe}")


# ============================================================================
# 1) parse das linhas de bias do firmware
# ============================================================================
print("1) parse das linhas de calibracao do firmware")
linhas = [
    "rst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)",
    "[i2c] barramento vivo em SDA=22 SCL=21 (endereco 0x69)",
    "[imu] calibrando giroscopio MPU 0x68... mantenha parado",
    "[imu] bias MPU 0x68: (2.4400, 3.0268, -0.3868)",
    "[imu] calibrando giroscopio MPU 0x69... mantenha parado",
    "[imu] bias MPU 0x69: (1.1188, 0.0688, -0.6436)",
    "1,8023481,-1.13,2.89,0.00,-0.05,-0.02,1.06,0.17,-0.05,0.03",
]
biases = imu_reset.parse_bias_firmware(linhas)
check("achou o canal 0x68", "0X68" in biases or "0x68" in biases, biases)
check("achou o canal 0x69", "0X69" in biases or "0x69" in biases, biases)
canal68 = biases.get("0X68") or biases.get("0x68")
canal69 = biases.get("0X69") or biases.get("0x69")
check("valores do 0x68", canal68 == (2.44, 3.0268, -0.3868), canal68)
check("valores do 0x69", canal69 == (1.1188, 0.0688, -0.6436), canal69)
check("ignora linha de dados e texto sem bias",
      len(biases) == 2, biases)
check("ignora vetor malformado",
      imu_reset.parse_bias_firmware(["[imu] bias MPU 0x68: (a, b, c)"]) == {})

# ============================================================================
# 2) deteccao da porta serial (CP210x)
# ============================================================================
print("2) deteccao da porta serial do ESP32")
Porta = type("Porta", (), {})


def cria_porta(device, description):
    p = Porta()
    p.device = device
    p.description = description
    return p


portas = [cria_porta("COM1", "Porta de comunicacao (COM1)"),
          cria_porta("COM4", "Silicon Labs CP210x USB to UART Bridge (COM4)")]
check("acha o CP210x", imu_reset.encontrar_porta_serial(
    None, listar_portas=lambda: portas) == "COM4")
check("porta preferida tem prioridade", imu_reset.encontrar_porta_serial(
    "COM7", listar_portas=lambda: portas) == "COM7")
check("sem CP210x devolve None", imu_reset.encontrar_porta_serial(
    None, listar_portas=lambda: [cria_porta("COM1", "Serial")]) is None)

# ============================================================================
# 3) avaliacao do bias medido no UDP
# ============================================================================
print("3) avaliacao do bias parado (OK / aceitavel / ALTO)")
ok, mensagens = imu_reset.avaliar_bias({
    4210: {"media": (0.01, -0.02, 0.03), "std": (0.08, 0.08, 0.08),
           "amostras": 200},
    4211: {"media": (-0.05, 0.02, 0.01), "std": (0.1, 0.1, 0.1),
           "amostras": 200},
})
check("bias bom = OK", ok is True and any("OK" in m for m in mensagens),
      mensagens)
ok, mensagens = imu_reset.avaliar_bias({
    4210: {"media": (0.30, 0.05, 0.02), "std": (0.08, 0.08, 0.08),
           "amostras": 200}})
check("bias visivel = aceitavel (nao reprova)",
      ok is True and any("aceitavel" in m for m in mensagens), mensagens)
ok, mensagens = imu_reset.avaliar_bias({
    4210: {"media": (-12.5, 3.0, 0.5), "std": (0.09, 0.09, 0.09),
           "amostras": 200}})
check("bias alto = reprova e manda repetir",
      ok is False and any("ALTO" in m for m in mensagens), mensagens)
ok, mensagens = imu_reset.avaliar_bias({4210: {"erro": "sem amostras"}})
check("falha de medida nao reprova (so' avisa)",
      ok is True and any("nao medi" in m for m in mensagens), mensagens)
ok, mensagens = imu_reset.avaliar_bias({
    4210: {"media": (0.05, 0.02, 0.0), "std": (2.5, 0.2, 0.2),
           "amostras": 150}})
check("desvio alto = MEDIDA INVALIDA (luva se mexeu)",
      ok is False and any("INVALIDA" in m for m in mensagens), mensagens)


# ============================================================================
# 4) reset pela serial com serial FALSA
# ============================================================================
print("4) reset pela serial (serial falsa): pulso RTS + boot + biases")


class SerialFalsa:
    """Serial de mentira: entrega o log de boot em pedacos e linhas de dados."""

    def __init__(self, porta, baudrate, timeout=0.2):
        self.porta = porta
        self.baudrate = baudrate
        self.dtr = True
        self.rts = False
        self._pedacos = []
        self.fechado = False
        texto = (
            "ets Jun  8 2016 00:22:57\r\n"
            "rst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)\r\n"
            "[i2c] barramento vivo em SDA=22 SCL=21 (endereco 0x69)\r\n"
            "[imu] calibrando giroscopio MPU 0x68... mantenha parado\r\n"
            "[imu] bias MPU 0x68: (2.4400, 3.0268, -0.3868)\r\n"
            "[imu] calibrando giroscopio MPU 0x69... mantenha parado\r\n"
            "[imu] bias MPU 0x69: (1.1188, 0.0688, -0.6436)\r\n"
            "1,8023481,-1.13,2.89,0.00,-0.05,-0.02,1.06,0.17,-0.05,0.03\r\n"
        ).encode("utf-8")
        for i in range(0, len(texto), 60):
            self._pedacos.append(texto[i:i + 60])

    def reset_input_buffer(self):
        pass

    @property
    def in_waiting(self):
        return len(self._pedacos[0]) if self._pedacos else 0

    def read(self, n=1):
        if not self._pedacos:
            return b""
        return self._pedacos.pop(0)

    def close(self):
        self.fechado = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
        return False


resultado = imu_reset.reiniciar_luvas(
    porta="COM4", segundos=0.3,
    serial_factory=lambda *a, **k: SerialFalsa(*a, **k))
check("executou", resultado["executado"] is True, resultado)
check("viu o boot (reiniciou)", resultado["reiniciou"] is True, resultado)
check("leu os DOIS biases",
      len(resultado["biases"]) == 2, resultado["biases"])
check("descartou as linhas de dados",
      all(not l.startswith(("1,", "2,")) for l in resultado["linhas"]),
      resultado["linhas"][-1])
check("porta usada", resultado["porta"] == "COM4", resultado["porta"])

# sem porta serial: nao explode, devolve executado=False
resultado = imu_reset.reiniciar_luvas(porta=None, segundos=0.1,
                                     listar_portas=lambda: [])
check("sem porta: executado=False e motivo",
      resultado["executado"] is False and "serial" in resultado["motivo"],
      resultado)


# ============================================================================
# 5) medida do bias pelo UDP com socket FALSO
# ============================================================================
print("5) medida do bias pelo UDP (socket falso)")
gyros = {
    4210: [(0.1743, -0.0537, 0.0310), (-0.0331, -0.0540, -0.0905)],
    4211: [(0.0000, -0.0100, 0.0200), (0.0100, 0.0000, -0.0200)],
}


class SocketFalso:
    def __init__(self, *args, **kwargs):
        self.porta = None
        self._pacotes = []
        self.fechado = False

    def setsockopt(self, *args):
        pass

    def bind(self, endereco):
        self.porta = int(endereco[1])
        for gyro in gyros.get(self.porta, []):
            self._pacotes.append(
                ("3600000000,0,0,0,-0.05,-0.02,1.06,%.4f,%.4f,%.4f"
                 % gyro).encode("utf-8"))

    def settimeout(self, _valor):
        pass

    def recvfrom(self, _n):
        if not self._pacotes:
            raise _socket.timeout()
        return self._pacotes.pop(0), ("127.0.0.1", 1)

    def close(self):
        self.fechado = True


bias_udp = imu_reset.medir_bias_udp(
    portas=(4210, 4211), segundos=0.2,
    socket_factory=lambda *a, **k: SocketFalso(*a, **k))
check("mede as duas portas", set(bias_udp) == {4210, 4211}, bias_udp)
check("media do gyro da direita",
      abs(bias_udp[4210]["media"][0] - 0.0706) < 1e-6, bias_udp[4210])
check("conta amostras", bias_udp[4211]["amostras"] == 2, bias_udp[4211])


class SocketOcupado:
    def __init__(self, *args, **kwargs):
        pass

    def setsockopt(self, *args):
        pass

    def bind(self, _endereco):
        raise OSError("endereco em uso")

    def settimeout(self, _valor):
        pass

    def close(self):
        pass


bias_udp = imu_reset.medir_bias_udp(
    portas=(4210,), segundos=0.1,
    socket_factory=lambda *a, **k: SocketOcupado(*a, **k))
check("porta ocupada vira {'erro': ...} sem explodir",
      "erro" in bias_udp.get(4210, {}), bias_udp)

# ============================================================================
# 6) ritual completo com hardware simulado
# ============================================================================
print("6) ritual completo calibra_imu (hardware simulado)")
capturado = io.StringIO()


def criar_socket_falso():
    """A mesma fabrica serve para os dois sockets: cada bind escolhe a porta."""
    return SocketFalso()


resultado = imu_reset.calibra_imu(
    participante="P01", sessao="S1", portas=(4210, 4211),
    porta_serial="COM4", segundos_boot=0.3, segundos_udp=0.2,
    settle_s=0.0, confirmar=False,
    avisar=lambda m: capturado.write(str(m) + "\n"),
    serial_factory=lambda *a, **k: SerialFalsa(*a, **k),
    socket_factory=lambda *a, **k: criar_socket_falso())
check("executado", resultado["executado"] is True, resultado)
check("ok", resultado["ok"] is True, resultado["mensagens"])
check("participante/sessao no resultado",
      resultado["participante"] == "P01" and resultado["sessao"] == "S1")
check("bias do firmware no resultado",
      len(resultado["biases_firmware"]) == 2, resultado["biases_firmware"])
check("bias medido no UDP no resultado",
      len(resultado["bias_udp"]) == 2, resultado["bias_udp"])
texto = capturado.getvalue()
check("aviso mostra o participante e a sessao",
      "CALIBRACAO DOS IMUs" in texto and "P01" in texto, texto[:200])
check("aviso manda deixar as luvas paradas",
      "PARADAS" in texto, texto[:200])


def _serializa(obj):
    import json
    return isinstance(json.dumps(obj), str)


check("resultado serializavel em JSON",
      _serializa(resultado) is True)

# sem porta serial: ritual avisa e nao derruba (executado=False)
resultado = imu_reset.calibra_imu(
    participante="P02", portas=(4210, 4211), porta_serial=None,
    segundos_boot=0.1, segundos_udp=0.1, settle_s=0.0, confirmar=False,
    avisar=lambda _m: None, listar_portas=lambda: [],
    serial_factory=lambda *a, **k: SerialFalsa(*a, **k),
    socket_factory=lambda *a, **k: SocketFalso(*a, **k))
check("sem porta: executado=False, motivo preenchido",
      resultado["executado"] is False and resultado["motivo"],
      resultado)
check("sem porta: ainda mede o bias pelo UDP (socket falso)",
      len(resultado["bias_udp"]) == 2, resultado["bias_udp"])

# ============================================================================
# 7) integracao nos DOIS programas
# ============================================================================
print("7) integracao nos dois programas")
raiz = os.path.dirname(os.path.abspath(__file__))
fonte_tool = open(os.path.join(raiz, "tools", "kinect_groundtruth_tool.py"),
                  encoding="utf-8").read()
fonte_par = open(os.path.join(raiz, "eeg_motor_paradigm.py"),
                 encoding="utf-8").read()
check("tool importa imu_reset", "import imu_reset" in fonte_tool)
check("tool chama calibra_imu no inicio",
      "imu_reset.calibra_imu(" in fonte_tool)
check("tool tem --sem-calibra-imu", "--sem-calibra-imu" in fonte_tool)
check("tool tem --porta-imu-serial", "--porta-imu-serial" in fonte_tool)
check("tool tem --participante no aviso", '"--participante"' in fonte_tool)
check("paradigma importa imu_reset", "import imu_reset" in fonte_par)
check("paradigma chama calibra_imu",
      "imu_reset.calibra_imu(" in fonte_par)
check("paradigma tem --sem-calibra-imu", "--sem-calibra-imu" in fonte_par)
check("paradigma tem --porta-imu-serial", "--porta-imu-serial" in fonte_par)
check("paradigma grava imu_calibracao no meta",
      '"imu_calibracao"' in fonte_par)
check("modulo imu_reset esta' na raiz",
      os.path.exists(os.path.join(raiz, "imu_reset.py")))

print("8) tentativas automaticas quando a medida sai alta/invalida")
chamadas = {"n": 0}
medidor_original = imu_reset.medir_bias_udp


def medidor_falso(portas, segundos, socket_factory=None):
    chamadas["n"] += 1
    if chamadas["n"] == 1:
        return {4210: {"media": (-6.0, 0.1, 0.0),
                       "std": (0.1, 0.1, 0.1), "amostras": 150},
                4211: {"media": (0.9, 0.4, -1.2),
                       "std": (0.1, 0.1, 0.1), "amostras": 150}}
    return {4210: {"media": (0.01, -0.01, 0.0),
                   "std": (0.08, 0.08, 0.08), "amostras": 150},
            4211: {"media": (-0.02, 0.0, 0.01),
                   "std": (0.1, 0.1, 0.1), "amostras": 150}}


imu_reset.medir_bias_udp = medidor_falso
try:
    resultado_retry = imu_reset.calibra_imu(
        participante="P07", portas=(4210, 4211), porta_serial="COM4",
        segundos_boot=0.2, segundos_udp=0.05, settle_s=0.0, tentativas=2,
        confirmar=False, avisar=lambda _m: None,
        serial_factory=lambda *a, **k: SerialFalsa(*a, **k))
finally:
    imu_reset.medir_bias_udp = medidor_original
check("retry: medida alta na 1a tentativa e' repetida automaticamente",
      chamadas["n"] == 2, chamadas)
check("retry: terminou OK na 2a tentativa",
      resultado_retry["ok"] is True
      and resultado_retry.get("tentativas_usadas") == 2,
      (resultado_retry.get("ok"),
       resultado_retry.get("tentativas_usadas")))

if FALHAS:
    print()
    print(f"IMU_RESET_FALHOU: {len(FALHAS)} verificacao(oes): "
          + " | ".join(FALHAS))
    raise SystemExit(1)
print()
print("IMU_RESET_OK: parse dos biases | deteccao CP210x | avaliacao OK/alto | "
      "reset+UDP simulados | ritual completo e serializavel | integracao no "
      "tool e no paradigma")
