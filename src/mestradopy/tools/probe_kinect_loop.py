"""Detecta o ciclo LIGA/DESLIGA do Kinect na arvore USB (o bug do loop).

POR QUE ISTO EXISTE (bancada de 07/10/2026)
  O usuario relatou, com o Kinect Studio aberto, o comportamento de LIGA E
  DESLIGA EM LOOP do sensor. Isso explica todo o resto sem precisar de mais nada:
  se o dispositivo entra e sai da arvore USB, o `sensor.IsAvailable` NUNCA fica
  estavel em TRUE -- os programas do SDK pegam sempre uma janela em que o sensor
  nao esta' pronto, e o Kinect Studio (que reabre o stream a cada ciclo) segue
  mostrando imagem. Este e' o mecanismo descrito para o bug de audio do Windows
  (o DSP do ponto de captura provoca reset de USB no dispositivo composto):
    https://learn.microsoft.com/en-ca/answers/questions/5609514/kinect-for-windows-11-has-broken-solve-found

  COMO DETECTA SEM DEPENDER DE LOG (o EventLog desta maquina nao registra a
  enumeracao USB -- medido: 5 eventos PnP no periodo, nenhum de USB):
    * a INSTANCIA do dispositivo no registro (`VID_045E&PID_02C4` -> nome da
      instancia, ex.: `002457351347`). O Windows cria instancia NOVA quando o
      dispositivo chega de novo; instancia que muda = o dispositivo saiu e voltou;
    * o numero de dispositivos "Unknown USB Device (Device Descriptor Request
      Failed)" (`VID_0000&PID_0002`), que e' o que sobra de uma chegada que falhou.

Uso:
    # observa por 60 s (deixe o Kinect Studio ABERTO, e' onde o loop aparece)
    .venv\\Scripts\\python.exe tools\\probe_kinect_loop.py --observar 60
    # e' o mesmo teste com o registro lido a cada 1 s
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import kinect_compat                                        # noqa: E402

kinect_compat.aplicar()

try:
    import winreg
except ImportError:                                         # pragma: no cover
    winreg = None

#: Raiz do registro onde o Windows lista os dispositivos USB enumerados.
RAIZ_USB = r"SYSTEM\CurrentControlSet\Enum\USB"
#: IDs do sensor (Kinect v2) e do "descritor falhou".
VID_SENSOR = "VID_045E&PID_02C4"
VID_DESCONHECIDO = "VID_0000&PID_0002"


def _subchaves(caminho):
    """Nomes das subchaves de `caminho` (lista vazia se nao existir)."""
    if winreg is None:
        return []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, caminho) as chave:
            nomes = []
            indice = 0
            while True:
                try:
                    nomes.append(winreg.EnumKey(chave, indice))
                except OSError:
                    break
                indice += 1
            return nomes
    except OSError:
        return []


def instancias_do_sensor():
    """Instancias enumeradas do Kinect (a identidade muda quando ele volta)."""
    return sorted(_subchaves(RAIZ_USB + "\\" + VID_SENSOR))


def desconhecidos():
    """Quantos 'Unknown USB Device (Device Descriptor Request Failed)' existem."""
    return len(_subchaves(RAIZ_USB + "\\" + VID_DESCONHECIDO))


def estado_do_sensor_sdk():
    """(hresult_get, hresult_open, is_available) do caminho cru do SDK."""
    from pykinect2 import PyKinectV2
    sensor = ctypes.POINTER(PyKinectV2.IKinectSensor)()
    hres = ctypes.windll.kinect20.GetDefaultKinectSensor(ctypes.byref(sensor))
    if not bool(sensor):
        return hres, None, None
    hres_open = sensor.Open()
    return hres, hres_open, bool(sensor.IsAvailable)


def observa(segundos, intervalo=1.0, com_sdk=True):
    """(eventos, resumo) observando a arvore USB e (opcional) o IsAvailable.

    `eventos` = lista de (segundo, texto) a cada mudanca. E' a mudanca que
    interessa: instancia nova = o dispositivo chegou de novo.
    """
    eventos = []
    base_sensor = instancias_do_sensor()
    base_desc = desconhecidos()
    eventos.append((0.0, "inicio: sensor=%s | descritor falhou=%d"
                    % (base_sensor, base_desc)))
    anterior = (tuple(base_sensor), base_desc)
    inicio = time.time()
    ciclos = 0
    while time.time() - inicio < segundos:
        agora = time.time() - inicio
        sensor = instancias_do_sensor()
        desc = desconhecidos()
        marca = (tuple(sensor), desc)
        if marca != anterior:
            ciclos += 1
            eventos.append((agora, "MUDOU: sensor=%s | descritor falhou=%d"
                            % (sensor, desc)))
            anterior = marca
        time.sleep(intervalo)

    resumo = {"ciclos": ciclos, "inicio": base_sensor,
              "fim": instancias_do_sensor(),
              "desconhecidos_inicio": base_desc,
              "desconhecidos_fim": desconhecidos()}
    if com_sdk:
        hres, hres_open, disponivel = estado_do_sensor_sdk()
        resumo["sdk"] = (hres, hres_open, disponivel)
    return eventos, resumo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observar", type=float, default=60.0,
                        help="segundos observando (padrao 60)")
    parser.add_argument("--intervalo", type=float, default=1.0,
                        help="passo da leitura do registro (s)")
    parser.add_argument("--sem-sdk", action="store_true",
                        help="nao abre o sensor pelo SDK no fim (so' o registro)")
    args = parser.parse_args(argv)

    print("probe_kinect_loop: o Kinect esta' LIGANDO e DESLIGANDO em loop?")
    print("Deixe o Kinect Studio ABERTO durante a observacao: e' onde o loop "
          "aparece.")
    print("Observando %.0f s (registro a cada %.1f s)...\n"
          % (args.observar, args.intervalo))

    eventos, resumo = observa(args.observar, args.intervalo,
                              com_sdk=not args.sem_sdk)
    for quando, texto in eventos:
        print("  %6.1f s  %s" % (quando, texto))

    print("\n" + "=" * 72)
    print("RESUMO")
    print("=" * 72)
    print("  instancias no inicio : %s" % resumo["inicio"])
    print("  instancias no fim    : %s" % resumo["fim"])
    print("  mudancas no periodo  : %d" % resumo["ciclos"])
    print("  'descritor falhou'   : %d -> %d"
          % (resumo["desconhecidos_inicio"], resumo["desconhecidos_fim"]))
    if "sdk" in resumo:
        hres, hres_open, disponivel = resumo["sdk"]
        print("  SDK no fim           : Get=0x%08X Open=%s IsAvailable=%s"
              % ((hres or 0) & 0xFFFFFFFF,
                 "n/d" if hres_open is None else "0x%08X" % (hres_open & 0xFFFFFFFF),
                 disponivel))

    print("")
    if resumo["ciclos"] > 0:
        print("  >>> LOOP CONFIRMADO: o dispositivo mudou %d vez(es) em %.0f s."
              % (resumo["ciclos"], args.observar))
        print("  Este e' o mecanismo do bug de audio do Windows: o DSP do ponto de")
        print("  captura reinicia o dispositivo COMPOSTO, e enquanto ele reinicia o")
        print("  `IsAvailable` nunca fica estavel em TRUE -- por isso o programa do")
        print("  SDK fica mudo e o Studio (que reabre o stream a cada ciclo) mostra")
        print("  imagem.")
        print("  Conserto: tirar o AUDIO DO KINECT do caminho, nao so' desmarcar as")
        print("  melhorias -- DESINSTALAR o dispositivo de audio do Kinect no")
        print("  Gerenciador de Dispositivos (o projeto nao usa audio nenhum).")
        return 1
    print("  Nenhuma mudanca no periodo: NAO ha' loop agora.")
    print("  Se o loop so' aparece COM o Studio capturando, repita com ele aberto")
    print("  e um stream ativo (Color ou Body) na tela do Studio.")
    if "sdk" in resumo and resumo["sdk"][2]:
        print("  E o SDK esta' vendo o sensor disponivel: va' para a bancada.")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
