"""CALIBRACAO DOS IMUs no inicio de cada sessao de aquisicao (item 101).

POR QUE EXISTE (medido em 21/10/2026, itens 99/100): o firmware calibra o bias
do giroscopio UMA vez no boot e assume a luva PARADA. Se o ESP32 religa com a
luva em movimento, o "bias" vira a media do movimento e o yaw deriva:
  - antes: direita -12,5 dps = 434 graus/min de yaw;
  - depois de reiniciar com as luvas paradas: -0,20 graus/min (direita) e
    -0,04 (esquerda).
Este modulo automatiza o ritual que resolveu o problema e o liga no INICIO de
cada sessao, com aviso por PARTICIPANTE/SESSAO:

  1) avisa e pede confirmacao (luvas paradas na mesa);
  2) reinicia o ESP32 das luvas pela serial (pulso RTS -> EN; e' o pulso que
     funciona no CP210x desta placa -- o pulso do esptool derrubava a porta);
  3) espera o boot e le as linhas "[imu] bias MPU 0x68/0x69" (prova de que a
     calibracao rodou);
  4) mede o gyro parado pelo UDP (4210/4211, ~3 s) e avalia o bias residual;
  5) devolve um dicionario com o resultado (vai para o JSON da sessao no
     paradigma e para o console do groundtruth tool).

Falhas NAO derrubam a sessao: sem porta serial, sem UDP ou bias alto, ele avisa
e retorna `executado=False`/`ok=False` (o operador decide se repete).
"""
from __future__ import annotations

import re
import time

#: Media do gyro parado (por eixo) aceitavel para a sessao, em dps. Com ~0,5 dps
#: a deriva de yaw pode chegar a ~30 graus/min em posicoes desfavoraveis -- por
#: isso o limite e' apertado e o script de reinicio deve ser repetido.
BIAS_MAX_DPS = 0.50
#: Acima disso o resultado e' "aceitavel com aviso" (bias residual visivel).
BIAS_AVISO_DPS = 0.20
#: Regex das linhas que o firmware imprime no boot.
_RE_BIAS = re.compile(r"bias MPU\s+(0x[0-9A-Fa-f]{2})\s*:\s*\(([^)]*)\)")
_RE_BOOT = re.compile(r"rst:|POWERON_RESET|bias MPU|calibrando giroscopio")


def encontrar_porta_serial(preferida=None, listar_portas=None):
    """Porta serial do ESP32 (CP210x). `preferida` tem prioridade; None = nao achou.

    `listar_portas` e' injetavel para teste (default:
    `serial.tools.list_ports.comports()`).
    """
    if preferida:
        return str(preferida)
    if listar_portas is None:
        try:
            from serial.tools import list_ports
        except ImportError:
            return None
        listar_portas = list_ports.comports
    try:
        portas = list(listar_portas())
    except Exception:                      # noqa: BLE001 - ambiente sem serial
        return None
    candidatas = []
    for porta in portas:
        nome = str(getattr(porta, "device", "")) or str(porta)
        descricao = str(getattr(porta, "description", ""))
        if "CP210" in descricao.upper() or "CP210" in nome.upper():
            candidatas.append(nome)
    return candidatas[0] if candidatas else None


def parse_bias_firmware(linhas):
    """{canal: (x, y, z)} a partir das linhas "[imu] bias MPU 0x68: (...)".

    Vetores ilegiveis/NaN sao ignorados (o chamador ve' o canal ausente).
    """
    import math

    biases = {}
    for linha in linhas:
        achado = _RE_BIAS.search(str(linha))
        if not achado:
            continue
        try:
            valores = tuple(float(p) for p in achado.group(2).split(","))
        except ValueError:
            continue
        if len(valores) == 3 and all(math.isfinite(v) for v in valores):
            biases[achado.group(1).upper()] = valores
    return biases


def reiniciar_luvas(porta=None, segundos=10.0, serial_factory=None,
                    listar_portas=None):
    """Pulso de RESET por RTS e escuta do boot. Devolve dict com o que viu."""
    porta = encontrar_porta_serial(porta, listar_portas)
    if porta is None:
        return {"executado": False, "motivo": "sem porta serial CP210x",
                "porta": None, "linhas": [], "biases": {}, "reiniciou": False}
    if serial_factory is None:
        import serial
        serial_factory = serial.Serial
    linhas = []
    reiniciou = False
    try:
        with serial_factory(porta, 115200, timeout=0.2) as porta_serial:
            porta_serial.reset_input_buffer()
            # RTS = EN (ativo baixo); DTR alto para NAO entrar em download mode.
            porta_serial.dtr = False
            porta_serial.rts = True
            time.sleep(0.15)
            porta_serial.rts = False
            time.sleep(0.05)
            porta_serial.reset_input_buffer()
            t0 = time.monotonic()
            pendente = b""
            while time.monotonic() - t0 < segundos:
                pendente += porta_serial.read(porta_serial.in_waiting or 1)
                while b"\n" in pendente:
                    bruta, pendente = pendente.split(b"\n", 1)
                    texto = bruta.decode("utf-8", "replace").strip()
                    if not texto or texto[:2] in ("1,", "2,"):
                        continue                        # linha de dados
                    linhas.append(texto)
                    if _RE_BOOT.search(texto):
                        reiniciou = True
    except Exception as exc:                   # noqa: BLE001 - caminho de hardware
        return {"executado": False, "motivo": f"{type(exc).__name__}: {exc}",
                "porta": porta, "linhas": linhas, "biases": {}, "reiniciou": False}
    return {"executado": True, "porta": porta, "linhas": linhas,
            "biases": parse_bias_firmware(linhas), "reiniciou": reiniciou}


def medir_bias_udp(portas=(4210, 4211), segundos=3.0, socket_factory=None):
    """{porta: {"media": (x,y,z), "std": (x,y,z), "amostras": n}} do gyro parado.

    Porta ocupada por outro programa (ex.: o proprio tool ja' aberto) vira
    `{"erro": ...}`: SEM SO_REUSEADDR de proposito, para o ritual FALHAR CLARO
    em vez de roubar os pacotes de quem ja' esta' lendo os IMUs.
    """
    import math
    import socket as _socket

    estatisticas = {}
    fabrica = socket_factory or _socket.socket
    for porta_udp in portas:
        sock = fabrica(_socket.AF_INET, _socket.SOCK_DGRAM)
        try:
            sock.bind(("0.0.0.0", int(porta_udp)))
        except OSError as exc:
            try:
                sock.close()
            except OSError:
                pass
            estatisticas[int(porta_udp)] = {
                "erro": f"porta em uso/indisponivel (outro programa aberto? "
                        f"feche-o e repita): {exc}"}
            continue
        sock.settimeout(0.05)
        amostras = []
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < float(segundos):
                try:
                    dado, _ = sock.recvfrom(2048)
                except _socket.timeout:
                    continue
                campos = dado.decode("utf-8", "replace").strip().split(",")
                if len(campos) < 10:
                    continue
                try:
                    gyro = [float(campos[7]), float(campos[8]),
                            float(campos[9])]
                except ValueError:
                    continue
                if all(math.isfinite(v) for v in gyro):
                    amostras.append(gyro)
        finally:
            sock.close()
        if not amostras:
            estatisticas[int(porta_udp)] = {
                "erro": "sem amostras (o ESP32 nao esta' enviando UDP? "
                        "confira o Wi-Fi/hotspot e o boot das luvas)"}
            continue
        n = len(amostras)
        medias = [sum(a[i] for a in amostras) / n for i in range(3)]
        desvios = [math.sqrt(sum((a[i] - medias[i]) ** 2
                                 for a in amostras) / max(1, n - 1))
                   for i in range(3)]
        estatisticas[int(porta_udp)] = {"media": tuple(medias),
                                       "std": tuple(desvios),
                                       "amostras": n}
    return estatisticas


def avaliar_bias(bias_udp, max_dps=BIAS_MAX_DPS, aviso_dps=BIAS_AVISO_DPS,
                 max_std_dps=1.5):
    """(ok, mensagens) a partir do resultado de `medir_bias_udp`.

    Medida com desvio alto (`max_std_dps`) e' INVALIDA: significa que a
    luva estava se MEXENDO durante a janela -- nao e' bias de verdade e
    precisa repetir (o ritual faz isso automaticamente).
    """
    mensagens = []
    ok = True
    for porta_udp, dados in sorted(bias_udp.items()):
        if dados.get("erro"):
            mensagens.append(f"porta {porta_udp}: nao medi ({dados['erro']})")
            continue
        media = dados["media"]
        desvio = dados.get("std") or (0.0, 0.0, 0.0)
        if desvio and max(float(v) for v in desvio) > max_std_dps:
            ok = False
            mensagens.append(
                f"porta {porta_udp}: MEDIDA INVALIDA -- a luva se MEXEU "
                f"durante a janela (desvio {max(float(v) for v in desvio):.2f} "
                "dps); repita com as luvas paradas")
            continue
        maior = max(abs(v) for v in media)
        texto = (f"porta {porta_udp}: bias medio "
                 f"({media[0]:+.2f}, {media[1]:+.2f}, {media[2]:+.2f}) dps")
        if maior > max_dps:
            ok = False
            mensagens.append(texto + " -> ALTO: repita o reinicio com as luvas "
                                     "100 por cento paradas")
        elif maior > aviso_dps:
            mensagens.append(texto + " -> aceitavel, mas visivel (yaw pode "
                                     "derivar ~1-2 graus/min)")
        else:
            mensagens.append(texto + " -> OK")
    return ok, mensagens


def calibra_imu(participante="", sessao="", portas=(4210, 4211),
                porta_serial=None, segundos_boot=10.0, segundos_udp=3.0,
                settle_s=2.0, tentativas=2, confirmar=False, avisar=print,
                serial_factory=None, socket_factory=None,
                listar_portas=None):
    """Ritual COMPLETO de calibracao dos IMUs no inicio da aquisicao.

    `settle_s`: espera depois do boot antes de medir o bias (o filtro/WiFi
    terminam de assentar). `tentativas`: se a medida sair INVALIDA/ALTA
    (luva mexeu), reinicia e mede de novo AUTOMATICAMENTE -- antes o
    operador tinha de relancar o programa inteiro.
    """
    """Ritual COMPLETO de calibracao dos IMUs no inicio da aquisicao.

    Devolve dict serializavel (vai para o JSON da sessao):
      {"executado": bool, "ok": bool, "porta": str|None,
       "biases_firmware": {canal: (x,y,z)}, "bias_udp": {...},
       "mensagens": [...], "motivo": str (quando nao executou),
       "participante": ..., "sessao": ...}
    """
    resultado = {"executado": False, "ok": False, "porta": None,
                 "biases_firmware": {}, "bias_udp": {}, "mensagens": [],
                 "participante": participante, "sessao": sessao}
    avisar("")
    avisar("=" * 72)
    avisar("  CALIBRACAO DOS IMUs (inicio da sessao)")
    avisar(f"  participante: {participante or '(nao informado)'} | "
           f"sessao: {sessao or '(nao informada)'}")
    avisar("  O firmware calibra o bias do giroscopio NO BOOT; ele PRECISA que")
    avisar("  as luvas estejam PARADAS e DEITADAS na mesa (sem mao, sem "
             "mexer; uma face do modulo IMU paralela a mesa) -- o "
             "firmware calibra o acelerometro no boot (item 124).")
    avisar("=" * 72)
    if confirmar:
        try:
            resposta = input(
                "  Deixe as luvas PARADAS e DEITADAS e tecle ENTER para "
                "reiniciar (ou 'n' para pular): ").strip().lower()
        except EOFError:
            resposta = ""
            avisar("  (sem terminal interativo: seguindo automaticamente)")
        if resposta == "n":
            resultado["motivo"] = "calibracao pulada pelo operador"
            resultado["ok"] = True
            resultado["mensagens"].append("calibracao dos IMUs pulada pelo operador")
            avisar("  Calibracao dos IMUs PULADA a pedido do operador.")
            return resultado
    reinicio = reiniciar_luvas(porta=porta_serial, segundos=segundos_boot,
                               serial_factory=serial_factory,
                               listar_portas=listar_portas)
    if resultado.get("porta") is None:
        resultado["porta"] = reinicio.get("porta")
    resultado["porta"] = reinicio.get("porta")
    resultado["biases_firmware"] = reinicio.get("biases") or {}
    if reinicio.get("executado"):
        resultado["executado"] = True
        for linha in reinicio.get("linhas") or []:
            avisar("  | " + str(linha)[:150])
        if not reinicio.get("reiniciou"):
            resultado["mensagens"].append(
                "AVISO: nao vi o boot/calibracao no tempo escutado (aumente --segundos?)")
        if resultado["biases_firmware"]:
            for canal, valores in sorted(resultado["biases_firmware"].items()):
                avisar(f"  bias lido do firmware {canal}: "
                       f"({valores[0]:+.3f}, {valores[1]:+.3f}, {valores[2]:+.3f})")
    else:
        resultado["motivo"] = reinicio.get("motivo", "falha no reset")
        avisar(f"  AVISO: nao consegui reiniciar pela serial ({resultado['motivo']})")
        avisar("  Faca na mao: aperte EN/RST (ou desligue/religue o USB) com as")
        avisar("  luvas paradas e nao mexa por ~3 s.")

    tentativas = 1 if not reinicio.get("executado") else max(1, int(tentativas))
    bias_udp = {}
    mensagens_medida = []
    for tentativa in range(1, tentativas + 1):
        if tentativa > 1:
            avisar("  MEDIDA ALTA/INVALIDA: repetindo o reinicio "
                   f"automaticamente (tentativa {tentativa}/{tentativas}); "
                   "deixe as luvas PARADAS...")
            reinicio = reiniciar_luvas(porta=porta_serial,
                                       segundos=segundos_boot,
                                       serial_factory=serial_factory,
                                       listar_portas=listar_portas)
            resultado["biases_firmware"] = (reinicio.get("biases")
                                            or resultado["biases_firmware"])
        if settle_s and settle_s > 0.0:
            avisar(f"  aguardando {settle_s:g} s para o filtro assentar...")
            time.sleep(float(settle_s))
        bias_udp = medir_bias_udp(portas=portas, segundos=segundos_udp,
                                  socket_factory=socket_factory)
        resultado["bias_udp"] = bias_udp
        if not bias_udp:
            resultado["mensagens"].append("sem medida de bias pelo UDP")
            break
        ok, mensagens_medida = avaliar_bias(bias_udp)
        resultado["ok"] = ok
        resultado["mensagens"] = [m for m in resultado["mensagens"]
                                  if not m.startswith("porta ")]
        resultado["mensagens"].extend(mensagens_medida)
        if ok:
            break
    resultado["tentativas_usadas"] = tentativa
    for mensagem in mensagens_medida:
        avisar("  " + mensagem)
    avisar("  RESULTADO: " + ("OK -- IMUs calibrados para esta sessao"
                              if resultado["ok"] else
                              "ATENCAO -- bias ainda alto/invalido; rode "
                              "tools\\reinicia_luvas_serial.py com as "
                              "luvas paradas ou confira o console"
                              + (f" (tentativas: {resultado.get('tentativas_usadas')})"
                                 if resultado.get("tentativas_usadas", 1) > 1
                                 else "")))
    return resultado
