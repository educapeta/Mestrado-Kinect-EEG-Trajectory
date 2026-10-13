"""Testa o mecanismo CANONICO do SDK: `SubscribeIsAvailableChanged`.

POR QUE EXISTE (bancada de 07/10/2026)
  Os dois jeitos de perguntar "o sensor esta' pronto?" que usamos ate' agora sao
  POLLING: `sensor.IsAvailable` (comtypes) e o mesmo pelo vtable cru. Os dois
  devolvem FALSE. Mas o SDK tem um caminho de BOOTSTRAP proprio, e ele e' o que
  os samples oficiais usam:

      sensor.Open()
      handle = sensor.SubscribeIsAvailableChanged()   # vtable [3]
      WaitForSingleObject(handle, timeout)            # o SDK AVISA quando muda
      -> SO' ENTAO o sensor esta' pronto

  Se o evento nunca dispara e o `IsAvailable` nunca vira TRUE, entao o SDK de
  fato nao considera o sensor pronto (e o problema e' de LINK/driver). Se ele
  dispara, o polling simples e' que estava errado -- e o conserto e' de codigo.

  ESTE PROGRAMA NAO DEPENDE DE INTERPRETACAO DE FLAG. Uma correcao de metodo que
  fica registrada: o `DeviceSelectiveSuspended=1` do registro NAO serviu como
  prova de "sensor suspenso" -- ele estava presente tambem quando o Kinect
  funcionava, e a chave que REALMENTE decide (`SelectiveSuspendEnabled`) nao
  existe nesta maquina. Foi o mesmo erro de antes (ler um numero e contar uma
  historia): o que vale e' o que o SDK responde.

Uso:
    .venv\\Scripts\\python.exe tools\\probe_kinect_isavailable_event.py
    .venv\\Scripts\\python.exe tools\\probe_kinect_isavailable_event.py --espera 30
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

import probe_kinect_vtable as pv                            # noqa: E402

#: Vtable do IKinectSensor (ordem lida de PyKinectV2.py:2532).
INDICE_SUBSCRIBE = 3
INDICE_UNSUBSCRIBE = 4
INDICE_EVENT_DATA = 5

INFINITE = 0xFFFFFFFF
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102


def _subscreve(ponteiro):
    """(hresult, handle) do `SubscribeIsAvailableChanged`."""
    _, funcoes = pv._funcoes(ponteiro)
    prototipo = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_void_p))
    handle = ctypes.c_void_p(0)
    hres = prototipo(funcoes[INDICE_SUBSCRIBE])(ponteiro, ctypes.byref(handle))
    return hres, handle.value


def _cancela(ponteiro, handle):
    _, funcoes = pv._funcoes(ponteiro)
    prototipo = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    return prototipo(funcoes[INDICE_UNSUBSCRIBE])(ponteiro, ctypes.c_void_p(handle))


def _le_evento(ponteiro, handle):
    """(hresult, ponteiro_do_evento) do `GetIsAvailableChangedEventData`."""
    _, funcoes = pv._funcoes(ponteiro)
    prototipo = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_void_p))
    evento = ctypes.c_void_p(0)
    hres = prototipo(funcoes[INDICE_EVENT_DATA])(ponteiro, ctypes.c_void_p(handle),
                                                 ctypes.byref(evento))
    return hres, evento.value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--espera", type=float, default=20.0,
                        help="quanto esperar o evento (s)")
    args = parser.parse_args(argv)

    print("probe_kinect_isavailable_event: o SDK AVISA que o sensor ficou pronto?")
    print("(o caminho de bootstrap dos samples oficiais -- nao e' polling)")

    sensor = ctypes.POINTER(pv.PyKinectV2.IKinectSensor)()
    hres = ctypes.windll.kinect20.GetDefaultKinectSensor(ctypes.byref(sensor))
    print("  GetDefaultKinectSensor: %s" % pv._hresult(hres))
    if not bool(sensor):
        print("  ERRO: ponteiro nulo")
        return 3
    base = ctypes.cast(sensor, ctypes.c_void_p)
    print("  Open()                : %s" % pv._hresult(sensor.Open()))

    hres_sub, handle = _subscreve(base)
    print("  SubscribeIsAvailableChanged: %s  handle=%s"
          % (pv._hresult(hres_sub), handle))
    if not handle:
        print("  ERRO: o SDK nao devolveu handle de evento.")
        return 4

    print("\n  esperando o evento por ate' %.0f s (o SDK avisa quando MUDA)..."
          % args.espera)
    kernel32 = ctypes.windll.kernel32
    disparou = False
    inicio = time.time()
    while time.time() - inicio < args.espera:
        espera_ms = 250
        resultado = kernel32.WaitForSingleObject(ctypes.c_void_p(handle), espera_ms)
        _, disponivel = pv._chama(base, pv.INDICE_IS_AVAILABLE, "bool")
        marca = time.time() - inicio
        if resultado == WAIT_OBJECT_0:
            disparou = True
            hres_ev, evento = _le_evento(base, handle)
            valor_evento = None
            if evento:
                #: IIsAvailableChangedEventArgs::get_IsAvailable (4a entrada:
                #: IUnknown 0..2 + a propriedade).
                _, valor_evento = pv._chama(ctypes.c_void_p(evento), 3, "bool")
            print("  %.1f s  EVENTO DISPAROU | get_IsAvailable do evento = %s"
                  % (marca, valor_evento))
            if disponivel:
                print("       e o sensor esta' DISPONIVEL: bootstrap concluido.")
                break
        else:
            print("  %.1f s  sem evento (wait=%s) | IsAvailable=%s"
                  % (marca, "TIMEOUT" if resultado == WAIT_TIMEOUT else resultado,
                     disponivel))
            if disponivel:
                print("       IsAvailable virou TRUE sem evento: o sensor esta' pronto.")
                disparou = True
                break

    _cancela(base, handle)

    print("\n  " + "=" * 68)
    _, disponivel = pv._chama(base, pv.INDICE_IS_AVAILABLE, "bool")
    print("  IsAvailable no fim: %s | evento disparou: %s" % (disponivel, disparou))
    if disponivel:
        print("  VEREDITO: o sensor ESTA' pronto -- siga para a bancada.")
        return 0
    if disparou:
        print("  VEREDITO: o evento do SDK DISPAROU mas o sensor nao esta' pronto:")
        print("  o SDK ve a transicao e nao sustenta o estado -- suspeito de LINK.")
        return 1
    print("  VEREDITO: nem o EVENTO do SDK nem o `IsAvailable` -- em %.0f s o SDK"
          % args.espera)
    print("  nunca considerou o sensor pronto. Nao e' polling errado: o sensor nao")
    print("  sai do estado 'indisponivel' pelo caminho do SDK.")
    print("  Proximo: desinstalar o AUDIO do Kinect (a interface MI_02 ainda carrega")
    print("  `usbaudio`) e remedir; e o Configuration Verifier da controladora.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
