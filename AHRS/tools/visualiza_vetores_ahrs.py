"""Visualizador 3D dos DOIS vetores de orientacao do MPU6050 (item 142).

Le a serial do firmware `firmware/esp32_teste_ahrs` (quaternions corpo->mundo
dos dois MPUs) e desenha UM VETOR POR SENSOR. Cada vetor e' fixo no corpo do
sensor e e' "tareado" no primeiro quadro para comecar apontando em -Z (chao)
quando a face esta' paralela a mesa; girar o sensor gira o vetor.

Uso:
  .venv\Scripts\python.exe tools\visualiza_vetores_ahrs.py --port COM4
  .venv\Scripts\python.exe tools\visualiza_vetores_ahrs.py --simular
Teclas: Z = re-tarar agora; C = recalibrar no ESP (IMUs paradas/deitadas);
        R = reiniciar atitude no ESP; Q/ESC = sair.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ahrs_vetores_core as core  # noqa: E402


def listar_portas():
    from serial.tools import list_ports
    portas = list(list_ports.comports())
    if not portas:
        print("Nenhuma porta serial encontrada.")
        return []
    print("Portas seriais encontradas:")
    for p in portas:
        print(f"  {p.device} | {p.description} | {p.hwid}")
    return portas


def escolher_porta(porta: str | None):
    if porta:
        return porta
    from serial.tools import list_ports
    portas = list(list_ports.comports())
    for p in portas:
        descricao = (p.description or "").lower()
        if "cp210" in descricao or "usb" in descricao or "uart" in descricao:
            return p.device
    return portas[0].device if portas else None


class LeitorSerial(threading.Thread):
    """Le a serial e guarda o ultimo quadro `A,...` parseado."""

    def __init__(self, porta: str, baud: int = 115200, debug: bool = False):
        super().__init__(daemon=True)
        self.porta = porta
        self.baud = baud
        self.debug = debug
        self.parar = False
        self.erro = None
        self.lock = threading.Lock()
        self.ultimo = None
        self.ultimo_instante = 0.0
        self.serial = None
        self.linhas = 0
        self.linhas_validas = 0
        self.primeiras = []
        self.ultima_bruta = ""
        self.tem_firmware_ahrs = False
        self.tem_linhas_luvas = False

    def run(self):
        try:
            import serial
            self.serial = serial.Serial(self.porta, self.baud, timeout=0.2)
        except Exception as exc:  # noqa: BLE001
            self.erro = f"nao abri {self.porta}: {exc}"
            return
        while not self.parar:
            try:
                bruto = self.serial.readline()
            except Exception as exc:  # noqa: BLE001
                self.erro = f"erro de leitura serial: {exc}"
                break
            if not bruto:
                continue
            linha = bruto.decode("utf-8", "replace").strip()
            self.linhas += 1
            self.ultima_bruta = linha
            if (linha.startswith("# AHRS_VETOR")
                    or linha.startswith("[ahrs]")):
                self.tem_firmware_ahrs = True
            if linha.startswith("1,") or linha.startswith("2,"):
                self.tem_linhas_luvas = True
            if self.debug or len(self.primeiras) < 6:
                if len(self.primeiras) < 6:
                    self.primeiras.append(linha)
                if self.debug:
                    print(f"[serial] {linha}", flush=True)
            dados = core.parse_linha_ahrs(linha)
            if dados is not None:
                self.linhas_validas += 1
                with self.lock:
                    self.ultimo = dados
                    self.ultimo_instante = time.monotonic()
        try:
            if self.serial is not None:
                self.serial.close()
        except Exception:  # noqa: BLE001
            pass

    def copiar(self):
        with self.lock:
            return self.ultimo, self.ultimo_instante

    def diagnostico(self):
        with self.lock:
            return {
                "linhas": self.linhas,
                "validas": self.linhas_validas,
                "firmware_ahrs": self.tem_firmware_ahrs,
                "linhas_luvas": self.tem_linhas_luvas,
                "ultima": self.ultima_bruta,
                "primeiras": list(self.primeiras),
            }

    def enviar(self, comando: str):
        if self.serial is None:
            return False
        try:
            self.serial.write(comando.encode("ascii"))
            return True
        except Exception:  # noqa: BLE001
            return False


def dados_simulados(t: float) -> dict:
    q_esq, q_dir = core.quat_simulado(t)
    return {
        "t_us": int(t * 1.0e6),
        "t_s": t,
        "left": q_esq,
        "right": q_dir,
        "accel_left": np.array([0.0, 0.0, 1.0]),
        "accel_right": np.array([0.0, 0.0, 1.0]),
    }


def texto_estado(sample, vetores, v_body, instante, origem):
    if sample is None:
        return ("AGUARDANDO QUADROS A,... DO FIRMWARE DE TESTE -- confira o "
                "monitor serial / se gravou firmware/esp32_teste_ahrs "
                "(com --simular a janela funciona sem hardware)")
    idade = time.monotonic() - instante
    linhas = [f"idade do ultimo quadro: {idade * 1000.0:.0f} ms"]
    for lado, nome, cor in (("left", "ESQUERDA (0x69)",
                             "ciano"), ("right", "DIREITA (0x68)",
                                        "laranja")):
        q = sample[lado]
        v = vetores[lado]
        linhas.append(
            f"{nome} [{cor}]: v=({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}) | "
            f"ang. c/ chao={core.angulo_com_baixo(v):5.1f} graus | "
            f"q=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f})")
    return "\n".join(linhas)


class Cena:
    def __init__(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        self.plt = plt
        self.fig = plt.figure(figsize=(11.0, 7.5))
        self.fig.canvas.manager.set_window_title(
            "AHRS 2x MPU6050 -- vetores (item 142)")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim(-1.25, 1.25)
        self.ax.set_ylim(-1.25, 1.25)
        self.ax.set_zlim(-1.25, 1.25)
        self.ax.set_xlabel("X (direita da imagem)")
        self.ax.set_ylabel("Y (cima da imagem)")
        self.ax.set_zlabel("Z (para dentro; -Z = chao)")
        self.ax.view_init(elev=20, azim=-62)
        self.ax.set_box_aspect((1.0, 1.0, 1.0))
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03)
        # tripe do mundo (pequeno)
        for eixo, cor in ((np.array([0.45, 0, 0]), "tab:red"),
                          (np.array([0, 0.45, 0]), "tab:green"),
                          (np.array([0, 0, 0.45]), "tab:blue")):
            self.ax.quiver(0, 0, 0, *eixo, color=cor, linewidth=1.2,
                           arrow_length_ratio=0.12, alpha=0.75)
        self.ax.text(0.48, 0, 0, "X", color="tab:red")
        self.ax.text(0, 0.48, 0, "Y", color="tab:green")
        self.ax.text(0, 0, 0.50, "Z", color="tab:blue")
        # piso em z=-1.05 para dar referencia visual de "chao"
        for x in np.linspace(-1.0, 1.0, 6):
            self.ax.plot([x, x], [-1.0, 1.0], [-1.05, -1.05],
                         color="0.75", linewidth=0.7)
        for y in np.linspace(-1.0, 1.0, 6):
            self.ax.plot([-1.0, 1.0], [y, y], [-1.05, -1.05],
                         color="0.75", linewidth=0.7)
        self.ax.text(-1.05, -1.0, -1.05, "piso (chao)", color="0.45",
                     fontsize=8)
        self.artistas = {"left": [], "right": []}
        self.texto = self.ax.text2D(0.01, 0.985, "", transform=self.ax.transAxes,
                                    fontsize=8, family="monospace",
                                    va="top")
        self.fig.canvas.mpl_connect("key_press_event", self._tecla)
        self.fechar = False

    def _tecla(self, evento):
        teclas = getattr(self, "ao_teclar", None)
        if teclas is not None:
            teclas(evento)
        if evento.key in ("q", "escape"):
            self.fechar = True

    def desenhar(self, sample, vetores, texto):
        origens = {"left": np.array([-0.22, 0.0, 0.12]),
                   "right": np.array([0.22, 0.0, 0.12])}
        cores = {"left": "tab:cyan", "right": "tab:orange"}
        nomes = {"left": "ESQUERDA", "right": "DIREITA"}
        for lado in ("left", "right"):
            for artista in self.artistas[lado]:
                try:
                    artista.remove()
                except Exception:  # noqa: BLE001
                    pass
            self.artistas[lado] = []
            if sample is None:
                continue
            origem = origens[lado]
            vetor = np.asarray(vetores[lado], dtype=float)
            q = self.ax.quiver(*origem, *(vetor * 0.90), color=cores[lado],
                               linewidth=4.5, arrow_length_ratio=0.18)
            ponta = origem + vetor * 0.90
            pontos = self.ax.scatter(*ponta, color=cores[lado], s=35,
                                     depthshade=False)
            rotulo = self.ax.text(*(origem + vetor * 1.02), nomes[lado],
                                  color=cores[lado], fontsize=8)
            self.artistas[lado] = [q, pontos, rotulo]
        self.texto.set_text(texto)


def modo_texto(args):
    """Fallback sem janela: imprime os vetores no console."""
    leitor = None
    if not args.simular:
        porta = escolher_porta(args.port)
        if porta is None:
            print("ERRO: nenhuma porta serial encontrada.")
            return 1
        leitor = LeitorSerial(porta, args.baud, debug=args.debug)
        leitor.start()
        print(f"[texto] lendo {porta}; aguarde o boot (~6 s). "
              "Ctrl+C encerra.")
    inicio = time.monotonic()
    sample = None
    instante = 0.0
    v_body = {"left": np.array([0.0, 0.0, -1.0]),
              "right": np.array([0.0, 0.0, -1.0])}
    zerado = False
    ultimo_print = 0.0
    ultimo_status = 0.0
    try:
        while True:
            agora = time.monotonic()
            if args.duracao > 0.0 and agora - inicio > args.duracao:
                break
            if args.simular:
                sample = dados_simulados(agora - inicio)
                instante = agora
            elif leitor is not None:
                sample, instante = leitor.copiar()
                if leitor.erro:
                    print(f"[texto] ERRO: {leitor.erro}")
                    break
                if sample is None and agora - inicio > 3.0 \
                        and agora - ultimo_status > 3.0:
                    ultimo_status = agora
                    d = leitor.diagnostico()
                    print(f"[texto] sem quadros A ainda: linhas={d['linhas']} "
                          f"validas={d['validas']} ultima={d['ultima']!r}",
                          flush=True)
                    if d["linhas_luvas"]:
                        print("[texto] COM4 esta' no firmware das luvas; "
                              "grave esp32_teste_ahrs.", flush=True)
                    elif d["firmware_ahrs"]:
                        print("[texto] firmware de teste detectado; "
                              "aguardando boot/calibracao (~6 s)...",
                              flush=True)
            if sample is None:
                time.sleep(0.05)
                continue
            if not zerado:
                for lado in ("left", "right"):
                    R0 = core.quat_para_matriz(*sample[lado])
                    v_body[lado] = core.vetor_corpo_para_baixo(R0)
                zerado = True
                print("[texto] tara aplicada: vetores comecam em -Z (chao).")
            if agora - ultimo_print >= 0.25:
                ultimo_print = agora
                vetores = {}
                for lado in ("left", "right"):
                    R = core.quat_para_matriz(*sample[lado])
                    vetores[lado] = core.vetor_mundo(R, v_body[lado])
                print(
                    "L=({:+.3f},{:+.3f},{:+.3f}) ang={:5.1f} | "
                    "R=({:+.3f},{:+.3f},{:+.3f}) ang={:5.1f} | "
                    "qL=({:+.3f},{:+.3f},{:+.3f},{:+.3f}) | "
                    "qR=({:+.3f},{:+.3f},{:+.3f},{:+.3f})".format(
                        *vetores["left"], core.angulo_com_baixo(vetores["left"]),
                        *vetores["right"], core.angulo_com_baixo(vetores["right"]),
                        *sample["left"], *sample["right"]), flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[texto] encerrado pelo usuario (Ctrl+C).")
    finally:
        if leitor is not None:
            leitor.parar = True
            try:
                leitor.join(timeout=1.0)
            except KeyboardInterrupt:
                pass
    return 0


class LogAHRS:
    """Grava CSV de amostras/eventos do visualizador para analise posterior."""

    CAMPOS = [
        "t_mono", "t_us",
        "qw_left", "qx_left", "qy_left", "qz_left",
        "qw_right", "qx_right", "qy_right", "qz_right",
        "ax_left", "ay_left", "az_left",
        "ax_right", "ay_right", "az_right",
        "vx_left", "vy_left", "vz_left", "ang_left_deg",
        "vx_right", "vy_right", "vz_right", "ang_right_deg",
    ]

    def __init__(self, caminho, porta=None, baud=None):
        import numpy as np  # noqa: F401 (mantem dependencia explicita)
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.arquivo = self.caminho.open("w", encoding="utf-8", newline="")
        self.csv = csv.writer(self.arquivo)
        self.arquivo.write("# AHRS_LOG v1\n")
        self.arquivo.write(f"# inicio={datetime.now().isoformat(timespec='seconds')}\n")
        if porta:
            self.arquivo.write(f"# porta={porta} baud={baud}\n")
        self.arquivo.write("# evento=linha comecando com #EVENT\n")
        self.csv.writerow(self.CAMPOS)
        self.n = 0

    def evento(self, nome, **extras):
        partes = [f"#EVENT t_mono={time.monotonic():.6f}", f"evento={nome}"]
        for chave, valor in extras.items():
            partes.append(f"{chave}={valor}")
        self.arquivo.write(",".join(partes) + "\n")
        self.arquivo.flush()

    def amostra(self, sample, v_body, vetores):
        linha = [
            time.monotonic(), sample["t_us"],
            *[float(v) for v in sample["left"]],
            *[float(v) for v in sample["right"]],
            *[float(v) for v in sample["accel_left"]],
            *[float(v) for v in sample["accel_right"]],
            *[float(v) for v in vetores["left"]],
            core.angulo_com_baixo(vetores["left"]),
            *[float(v) for v in vetores["right"]],
            core.angulo_com_baixo(vetores["right"]),
        ]
        self.csv.writerow(linha)
        self.n += 1
        if self.n % 10 == 0:
            self.arquivo.flush()

    def fechar(self):
        self.evento("fim", amostras=self.n)
        try:
            self.arquivo.close()
        except OSError:
            pass


def _opencv_camera():
    """Bases da camera virtual (OpenCV) para projetar o mundo 3D na tela."""
    import numpy as np
    eye = np.array([2.8, -3.8, 2.2], dtype=float)
    alvo = np.zeros(3, dtype=float)
    cima = np.array([0.0, 0.0, 1.0], dtype=float)
    frente = alvo - eye
    frente /= np.linalg.norm(frente)
    direita = np.cross(frente, cima)
    direita /= np.linalg.norm(direita)
    cima2 = np.cross(direita, frente)
    return eye, direita, cima2, frente


def modo_opencv(args):
    """Janela 3D nativa via OpenCV highgui (sem matplotlib/Qt)."""
    import cv2
    import numpy as np

    largura, altura = 1100, 700
    foco = 900.0
    eye, direita, cima2, frente = _opencv_camera()

    def projetar(ponto):
        d = np.asarray(ponto, dtype=float) - eye
        z = float(d @ frente)
        if z <= 0.15:
            return None
        x = float(d @ direita)
        y = float(d @ cima2)
        return (int(largura / 2 + foco * x / z),
                int(altura / 2 - foco * y / z))

    janela = "AHRS 2x MPU6050 -- vetores 3D (item 142, OpenCV)"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, largura, altura)

    leitor = None
    porta = None
    logger = None
    if not args.simular:
        porta = escolher_porta(args.port)
        if porta is None:
            print("ERRO: nenhuma porta serial encontrada.")
            return 1
        leitor = LeitorSerial(porta, args.baud, debug=args.debug)
        leitor.start()
        print(f"[opencv] lendo {porta}; aguarde o boot (~6 s). "
              "Q/ESC fecha.", flush=True)
    if args.log:
        caminho_log = args.log
        if str(caminho_log).lower() in ("auto", ""):
            carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
            caminho_log = Path(__file__).resolve().parent.parent / "logs" / f"ahrs_{carimbo}.csv"
        try:
            logger = LogAHRS(caminho_log, porta=porta, baud=args.baud)
            print(f"[opencv] LOG: {logger.caminho}", flush=True)
        except OSError as exc:
            print(f"[opencv] nao consegui criar log: {exc}")
            logger = None
    inicio = time.monotonic()
    ultimo_status = 0.0
    sample = None
    instante = 0.0
    v_body = {"left": np.array([0.0, 0.0, -1.0]),
              "right": np.array([0.0, 0.0, -1.0])}
    zerado = False
    snapshot_salvo = False
    amostras = 0
    ultimo_amostra = 0.0

    origens = {"left": np.array([-0.22, 0.0, 0.12]),
               "right": np.array([0.22, 0.0, 0.12])}
    cores = {"left": (255, 255, 0), "right": (0, 165, 255)}  # BGR
    nomes = {"left": "ESQUERDA 0x69", "right": "DIREITA 0x68"}

    try:
        while True:
            agora = time.monotonic()
            if args.duracao > 0.0 and agora - inicio > args.duracao:
                break
            if args.simular:
                sample = dados_simulados(agora - inicio)
                instante = agora
            elif leitor is not None:
                sample, instante = leitor.copiar()
                if leitor.erro:
                    print(f"[opencv] ERRO: {leitor.erro}", flush=True)
                    break
                if sample is None and agora - inicio > 3.0 \
                        and agora - ultimo_status > 3.0:
                    ultimo_status = agora
                    d = leitor.diagnostico()
                    print(f"[opencv] sem quadros A: linhas={d['linhas']} "
                          f"validas={d['validas']}", flush=True)
                    if d["linhas_luvas"]:
                        print("[opencv] COM4 no firmware das luvas; grave "
                              "esp32_teste_ahrs.", flush=True)
                    elif d["firmware_ahrs"]:
                        print("[opencv] firmware de teste detectado; "
                              "aguardando calibracao (~6 s)...", flush=True)
            if sample is not None and not zerado:
                for lado in ("left", "right"):
                    R0 = core.quat_para_matriz(*sample[lado])
                    v_body[lado] = core.vetor_corpo_para_baixo(R0)
                zerado = True
                if logger is not None:
                    logger.evento("tara_inicial",
                                  quat_left=tuple(round(float(v), 6) for v in sample["left"]),
                                  quat_right=tuple(round(float(v), 6) for v in sample["right"]),
                                  vbody_left=tuple(round(float(v), 6) for v in v_body["left"]),
                                  vbody_right=tuple(round(float(v), 6) for v in v_body["right"]))
                print("[opencv] dados AHRS chegando; vetores tareados em "
                      "-Z (chao).", flush=True)
            vetores_log = {}
            if sample is not None:
                amostras += 1
                ultimo_amostra = agora
                for lado in ("left", "right"):
                    R = core.quat_para_matriz(*sample[lado])
                    vetores_log[lado] = core.vetor_mundo(R, v_body[lado])
                if logger is not None:
                    logger.amostra(sample, v_body, vetores_log)

            quadro = np.zeros((altura, largura, 3), dtype=np.uint8)
            # piso (z = -1.2) em cinza
            for valor in np.linspace(-1.2, 1.2, 7):
                a = projetar((valor, -1.2, -1.2))
                b = projetar((valor, 1.2, -1.2))
                if a is not None and b is not None:
                    cv2.line(quadro, a, b, (55, 55, 55), 1, cv2.LINE_AA)
                a = projetar((-1.2, valor, -1.2))
                b = projetar((1.2, valor, -1.2))
                if a is not None and b is not None:
                    cv2.line(quadro, a, b, (55, 55, 55), 1, cv2.LINE_AA)
            # eixos do mundo
            origem_px = projetar((0.0, 0.0, 0.0))
            if origem_px is not None:
                for ponta, cor, rotulo in (
                        ((0.85, 0, 0), (0, 0, 255), "X"),
                        ((0, 0.85, 0), (0, 255, 0), "Y"),
                        ((0, 0, 0.85), (255, 0, 0), "Z (cima)")):
                    px = projetar(ponta)
                    if px is not None:
                        cv2.arrowedLine(quadro, origem_px, px, cor, 2,
                                        cv2.LINE_AA, tipLength=0.12)
                        cv2.putText(quadro, rotulo, (px[0] + 6, px[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1,
                                    cv2.LINE_AA)
                cv2.circle(quadro, origem_px, 5, (220, 220, 220), -1,
                           cv2.LINE_AA)
                cv2.putText(quadro, "-Z = chao", (origem_px[0] - 40,
                                                  origem_px[1] + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (150, 150, 150), 1, cv2.LINE_AA)
                # vetores (um por sensor)
                for lado in ("left", "right"):
                    R = (core.quat_para_matriz(*sample[lado])
                         if sample is not None else np.eye(3))
                    vetor = core.vetor_mundo(R, v_body[lado]) \
                        if sample is not None else np.array([0, 0, -1.0])
                    inicio_v = origens[lado]
                    ponta_v = inicio_v + vetor * 0.9
                    p0 = projetar(inicio_v)
                    p1 = projetar(ponta_v)
                    if p0 is None or p1 is None:
                        continue
                    cv2.arrowedLine(quadro, p0, p1, cores[lado], 4,
                                    cv2.LINE_AA, tipLength=0.14)
                    cv2.circle(quadro, p1, 7, cores[lado], -1, cv2.LINE_AA)
                    x_rotulo = (max(5, p1[0] - 165) if lado == "left"
                                else p1[0] + 10)
                    cv2.putText(quadro, nomes[lado],
                                (x_rotulo, p1[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cores[lado], 1,
                                cv2.LINE_AA)
            # painel de texto
            idade = (time.monotonic() - instante) if sample is not None else None
            linhas_texto = []
            if sample is None:
                linhas_texto.append("AGUARDANDO QUADROS A,... DO FIRMWARE DE "
                                    "TESTE (boot/calibracao ~6 s)")
                if leitor is not None:
                    d = leitor.diagnostico()
                    linhas_texto.append(
                        f"serial: linhas={d['linhas']} validas={d['validas']} "
                        f"ultima={d['ultima'][:70]!r}")
            else:
                linhas_texto.append(f"amostras={amostras} | ultimo quadro: "
                                    f"{idade * 1000.0:.0f} ms")
                for lado in ("left", "right"):
                    R = core.quat_para_matriz(*sample[lado])
                    vetor = core.vetor_mundo(R, v_body[lado])
                    q = sample[lado]
                    linhas_texto.append(
                        f"{nomes[lado]}: v=({vetor[0]:+.3f},{vetor[1]:+.3f},"
                        f"{vetor[2]:+.3f}) ang.chao="
                        f"{core.angulo_com_baixo(vetor):5.1f} deg")
                    linhas_texto.append(
                        f"   q=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},"
                        f"{q[3]:+.3f})")
                linhas_texto.append(
                    f"accel L={np.round(sample['accel_left'], 2)} "
                    f"R={np.round(sample['accel_right'], 2)}")
            linhas_texto.append("Z=re-tarar | C=recalibrar | R=reiniciar | "
                                "Q/ESC=sair")
            y = 24
            for linha in linhas_texto:
                cor_texto = (0, 255, 255) if sample is None else (230, 230, 230)
                cv2.putText(quadro, linha[:120], (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_texto, 1,
                            cv2.LINE_AA)
                y += 20
            if args.snapshot and not snapshot_salvo:
                try:
                    cv2.imwrite(args.snapshot, quadro)
                    print(f"[opencv] primeiro quadro salvo em {args.snapshot}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[opencv] nao salvei snapshot: {exc}")
                snapshot_salvo = True
            cv2.imshow(janela, quadro)
            tecla = cv2.waitKey(25) & 0xFF
            if tecla in (27, ord("q"), ord("Q")):
                break
            if tecla in (ord("z"), ord("Z")):
                if sample is not None:
                    for lado in ("left", "right"):
                        R0 = core.quat_para_matriz(*sample[lado])
                        v_body[lado] = core.vetor_corpo_para_baixo(R0)
                    zerado = True
                    if logger is not None:
                        logger.evento("tara_Z",
                                      quat_left=tuple(round(float(v), 6) for v in sample["left"]),
                                      quat_right=tuple(round(float(v), 6) for v in sample["right"]))
                    print("[opencv] tara aplicada (vetores em -Z).")
            elif tecla in (ord("c"), ord("C")) and leitor is not None:
                if leitor.enviar("c"):
                    if logger is not None:
                        logger.evento("comando_c")
                    print("[opencv] enviei 'c' ao ESP (recalibrar).")
            elif tecla in (ord("r"), ord("R")) and leitor is not None:
                if leitor.enviar("r"):
                    if logger is not None:
                        logger.evento("comando_r")
                    print("[opencv] enviei 'r' ao ESP (reiniciar atitude).")
            try:
                if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        print("[opencv] encerrado pelo usuario (Ctrl+C).")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        if leitor is not None:
            leitor.parar = True
            try:
                leitor.join(timeout=1.0)
            except KeyboardInterrupt:
                pass
        if logger is not None:
            logger.fechar()
            print(f"[opencv] log fechado: {logger.caminho}", flush=True)
    return 0


def principal():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=None, help="ex.: COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--simular", action="store_true",
                        help="roda sem hardware (demo/sala)")
    parser.add_argument("--listar-portas", action="store_true")
    parser.add_argument("--duracao", type=float, default=0.0,
                        help="fecha sozinho depois de N segundos (0=infinito)")
    parser.add_argument("--debug", action="store_true",
                        help="mostra as linhas cruas recebidas da serial")
    parser.add_argument("--texto", action="store_true",
                        help="modo console: imprime os vetores sem abrir janela")
    parser.add_argument("--opencv", action="store_true",
                        help="janela 3D nativa do OpenCV (padrao)")
    parser.add_argument("--matplotlib", action="store_true",
                        help="usa a janela 3D antiga do matplotlib/Qt")
    parser.add_argument("--snapshot", default=None,
                        help="salva o primeiro quadro da janela OpenCV em PNG")
    parser.add_argument("--log", nargs="?", const="auto", default=None,
                        help="grava CSV de amostras/eventos; sem valor usa "
                             "AHRS/logs/ahrs_<timestamp>.csv")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "qtagg", "tkagg", "agg"],
                        help="backend do matplotlib (se a janela nao abrir, "
                             "tente tkagg)")
    args = parser.parse_args()

    if args.backend != "auto":
        import matplotlib
        matplotlib.use(args.backend, force=True)
        print(f"[visualizador] backend forcado: {args.backend}")
    portas = listar_portas()
    if args.listar_portas:
        return 0
    if args.texto:
        return modo_texto(args)
    # OpenCV highgui e' o PADRAO: nao depende de Qt/Tcl e e' a janela que o
    # resto do projeto ja' usa. --matplotlib mantem a versao antiga.
    if args.opencv or not args.matplotlib:
        return modo_opencv(args)
    porta = None
    if not args.simular:
        porta = escolher_porta(args.port)
        if porta is None:
            print("ERRO: nenhuma porta serial encontrada. Use --port COMx "
                  "(ou --simular).")
            return 1
    inicio = time.monotonic()
    import matplotlib
    print(f"[visualizador] backend matplotlib: {matplotlib.get_backend()}",
          flush=True)
    print("[visualizador] abrindo a janela 3D... (se ela nao aparecer, veja "
          "se abriu atras do PowerShell ou use --simular para testar)",
          flush=True)
    cena = Cena()
    try:
        cena.plt.show(block=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[visualizador] aviso: show(block=False): {exc}")
    leitor = None
    if not args.simular:
        leitor = LeitorSerial(porta, args.baud, debug=args.debug)
        leitor.start()
        print(f"[visualizador] lendo {porta} a {args.baud} baud; firmware "
              "esperado: esp32_teste_ahrs (linhas A,...).", flush=True)
    v_body = {"left": np.array([0.0, 0.0, -1.0]),
              "right": np.array([0.0, 0.0, -1.0])}
    zerado = {"left": False, "right": False}
    ultimo_t = 0.0
    ultimo_desenho = 0.0
    ultimo_status = 0.0
    sample = None
    instante = 0.0

    def aplicar_tara(sample_atual):
        if sample_atual is None:
            return
        for lado in ("left", "right"):
            R0 = core.quat_para_matriz(*sample_atual[lado])
            v_body[lado] = core.vetor_corpo_para_baixo(R0)
            zerado[lado] = True

    def ao_teclar(evento):
        if evento.key in ("z", "Z"):
            aplicar_tara(sample)
            print("[visualizador] tara aplicada: vetores agora partem de -Z.")
        elif evento.key in ("c", "C") and leitor is not None:
            if leitor.enviar("c"):
                print("[visualizador] enviado 'c' ao ESP: recalibrar "
                      "(IMUs PARADAS e DEITADAS).")
        elif evento.key in ("r", "R") and leitor is not None:
            if leitor.enviar("r"):
                print("[visualizador] enviado 'r' ao ESP: reiniciar atitude.")
    cena.ao_teclar = ao_teclar

    try:
        while not cena.fechar and cena.plt.fignum_exists(cena.fig.number):
            agora = time.monotonic()
            if args.duracao > 0.0 and agora - inicio > args.duracao:
                cena.fechar = True
                break
            if args.simular:
                t = agora - inicio
                sample = dados_simulados(t)
                instante = agora
            elif leitor is not None:
                sample, instante = leitor.copiar()
                if leitor.erro:
                    print(f"[visualizador] ERRO: {leitor.erro}")
                    break
                if sample is None and agora - inicio > 3.0 \
                        and agora - ultimo_status > 3.0:
                    ultimo_status = agora
                    diag = leitor.diagnostico()
                    print(f"[visualizador] sem quadros A: linhas="
                          f"{diag['linhas']} validas={diag['validas']} | "
                          f"ultima={diag['ultima']!r}", flush=True)
                    if diag["firmware_ahrs"] and diag["validas"] == 0:
                        print("[visualizador] firmware de teste detectado; "
                              "aguardando boot/calibracao (~6 s) e os quadros "
                              "A,... -- nao interrompa ainda.", flush=True)
                    elif diag["linhas_luvas"]:
                        print("[visualizador] a COM4 esta enviando linhas do "
                              "FIRMWARE DAS LUVAS (1,.../2,...). Grave o "
                              "firmware de teste: pio run -d "
                              "firmware\esp32_teste_ahrs -t upload",
                              flush=True)
                    elif not diag["firmware_ahrs"] and diag["linhas"] == 0:
                        print("[visualizador] nenhum byte chegou; confira se "
                              "o ESP esta' ligado/resetado e se a COM4 e' a "
                              "placa certa (fechando o monitor serial).",
                              flush=True)
                    elif not diag["firmware_ahrs"]:
                        print("[visualizador] recebi linhas, mas nenhuma do "
                              "firmware AHRS; primeiras: "
                              f"{diag['primeiras'][:3]}", flush=True)
            # tara automatica no primeiro quadro (vetor comeca em -Z/chao)
            if sample is not None and not (zerado["left"] and zerado["right"]):
                aplicar_tara(sample)
                print("[visualizador] dados AHRS chegando: dois vetores "
                      "tareados em -Z (chao).", flush=True)
            vetores = {"left": np.array([0.0, 0.0, -1.0]),
                       "right": np.array([0.0, 0.0, -1.0])}
            if sample is not None:
                for lado in ("left", "right"):
                    R = core.quat_para_matriz(*sample[lado])
                    vetores[lado] = core.vetor_mundo(R, v_body[lado])
            if agora - ultimo_desenho > 0.04:  # ~25 fps
                ultimo_desenho = agora
                texto = texto_estado(sample, vetores, v_body, instante, None)
                texto += "\nZ=re-tarar | C=recalibrar ESP | R=reiniciar | Q=sair"
                cena.desenhar(sample, vetores, texto)
                cena.fig.canvas.draw_idle()
                cena.plt.pause(0.001)
            else:
                cena.plt.pause(0.01)
    except KeyboardInterrupt:
        print("[visualizador] encerrado pelo usuario (Ctrl+C).")
    finally:
        if leitor is not None:
            leitor.parar = True
            try:
                leitor.join(timeout=1.0)
            except KeyboardInterrupt:
                pass
        try:
            cena.plt.close("all")
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
