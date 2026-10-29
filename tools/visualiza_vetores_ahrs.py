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
import math
import sys
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

    def __init__(self, porta: str, baud: int = 115200):
        super().__init__(daemon=True)
        self.porta = porta
        self.baud = baud
        self.parar = False
        self.erro = None
        self.lock = threading.Lock()
        self.ultimo = None
        self.ultimo_instante = 0.0
        self.serial = None

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
            dados = core.parse_linha_ahrs(bruto.decode("utf-8", "replace"))
            if dados is not None:
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
        return "AGUARDANDO DADOS do firmware (A,...)"
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


def principal():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=None, help="ex.: COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--simular", action="store_true",
                        help="roda sem hardware (demo/sala)")
    parser.add_argument("--listar-portas", action="store_true")
    parser.add_argument("--duracao", type=float, default=0.0,
                        help="fecha sozinho depois de N segundos (0=infinito)")
    args = parser.parse_args()

    portas = listar_portas()
    if args.listar_portas:
        return 0
    leitor = None
    if not args.simular:
        porta = escolher_porta(args.port)
        if porta is None:
            print("ERRO: nenhuma porta serial encontrada. Use --port COMx "
                  "(ou --simular).")
            return 1
        leitor = LeitorSerial(porta, args.baud)
        leitor.start()
        print(f"Lendo {porta} a {args.baud} baud. Firmware esperado: "
              "firmware/esp32_teste_ahrs (linhas A,...).")
    inicio = time.monotonic()
    cena = Cena()
    v_body = {"left": np.array([0.0, 0.0, -1.0]),
              "right": np.array([0.0, 0.0, -1.0])}
    zerado = {"left": False, "right": False}
    ultimo_t = 0.0
    ultimo_desenho = 0.0
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
                    print(f"[visualizador] {leitor.erro}")
                    break
            # tara automatica no primeiro quadro (vetor comeca em -Z/chao)
            if sample is not None and not (zerado["left"] and zerado["right"]):
                aplicar_tara(sample)
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
    finally:
        if leitor is not None:
            leitor.parar = True
            leitor.join(timeout=1.0)
        cena.plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
