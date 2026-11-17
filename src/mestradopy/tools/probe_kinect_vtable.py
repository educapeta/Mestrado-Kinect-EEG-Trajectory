"""Le o `IsAvailable` do Kinect SEM o comtypes -- vtable COM na mao (ctypes puro).

POR QUE ISTO EXISTE (bancada de 07/10/2026)
  O `Kinect Studio` mostra imagem, o Windows ve o sensor chegar e sair quando o
  cabo e' mexido, e MESMO ASSIM todo programa do SDK diz
  `sensor.IsAvailable = FALSE` -- e nada entrega frame. O `kinect_compat.py`
  existe por um problema de ESTRUTURA de 32/64 bits (`tagSTATSTG` de 72 x 80
  bytes), o que levanta a hipotese: o `comtypes` (usado pelo `pykinect2`, gerado
  para Python 32 bits) pode estar lendo a ENTRADA ERRADA do vtable e devolvendo
  um FALSE que nao e' do sensor.

  Este script nao usa `pykinect2` para LER o valor: pega o ponteiro de interface,
  anda no vtable e chama com `ctypes.WINFUNCTYPE` (a ABI do COM). Se aqui der TRUE
  e no `pykinect2` der FALSE, o problema esta' na BIBLIOTECA; se der FALSE nos
  dois, o proprio SDK reporta indisponivel.

  ORDEM DO VTABLE -- lida de `pykinect2/PyKinectV2.py`, `IKinectSensor._methods_`
  (linha 2532), e nao adivinhada. O erro de 07/10/2026 foi CHUTAR esta ordem:

      0..2  IUnknown: QueryInterface, AddRef, Release
      3     SubscribeIsAvailableChanged      -> INT_PTR (waitable handle)
      4     UnsubscribeIsAvailableChanged    (in) INT_PTR
      5     GetIsAvailableChangedEventData   (in) INT_PTR + evento
      6     Open                              -- NAO tem saida
      7     Close                             -- NAO tem saida
      8     IsOpen                            -> c_bool
      9     IsAvailable                       -> c_bool     <<< a que decide
      10    ColorFrameSource                  -> ponteiro
      ...

  O `c_bool` importa: o SDK devolve `BOOLEAN` (1 byte). Ler como `c_int` (4
  bytes) escreve lixo nos outros 3 -- foi o `valor=820` que apareceu quando o
  indice estava errado, e e' o que faria um "TRUE" falso parecer verdadeiro.

Uso:
    .venv\\Scripts\\python.exe tools\\probe_kinect_vtable.py
    .venv\\Scripts\\python.exe tools\\probe_kinect_vtable.py --todos
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import kinect_compat                                        # noqa: E402

kinect_compat.aplicar()

from pykinect2 import PyKinectV2                            # noqa: E402

#: (indice, nome, tipo) das entradas do `IKinectSensor`, na ordem REAL do
#: `_methods_`. `tipo`: "int_ptr" (handle/saida de 8 bytes), "bool" (BOOLEAN de
#: 1 byte), "sem_saida" (so' o HRESULT), "ulong", "ponteiro".
ENTRADAS = (
    (0, "QueryInterface", "sem_saida"),
    (1, "AddRef", "sem_saida"),
    (2, "Release", "sem_saida"),
    (3, "SubscribeIsAvailableChanged", "int_ptr"),
    (4, "UnsubscribeIsAvailableChanged", "sem_saida"),
    (5, "GetIsAvailableChangedEventData", "sem_saida"),
    (6, "Open", "sem_saida"),
    (7, "Close", "sem_saida"),
    (8, "IsOpen", "bool"),
    (9, "IsAvailable", "bool"),
    (10, "ColorFrameSource", "ponteiro"),
    (11, "DepthFrameSource", "ponteiro"),
    (12, "BodyFrameSource", "ponteiro"),
    (13, "BodyIndexFrameSource", "ponteiro"),
    (14, "InfraredFrameSource", "ponteiro"),
    (15, "LongExposureInfraredFrameSource", "ponteiro"),
    (16, "AudioSource", "ponteiro"),
    (17, "OpenMultiSourceFrameReader", "ponteiro"),
    (18, "CoordinateMapper", "ponteiro"),
    (19, "UniqueKinectId", "sem_saida"),
    (20, "KinectCapabilities", "ulong"),
)
#: Os indices que decidem -- nomeados, para nunca mais serem chutados.
INDICE_IS_OPEN = 8
INDICE_IS_AVAILABLE = 9


def _funcoes(ponteiro):
    vtable = ctypes.cast(ponteiro, ctypes.POINTER(ctypes.c_void_p))[0]
    return vtable, ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))


def _chama(ponteiro, indice, tipo):
    """(hresult, valor) da entrada `indice` do vtable, com o TIPO certo de saida."""
    _, funcoes = _funcoes(ponteiro)
    endereco = funcoes[indice]
    if not endereco:
        return None, None
    if tipo == "sem_saida":
        prototipo = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
        return prototipo(endereco)(ponteiro), None
    if tipo == "bool":
        #: BOOLEAN = 1 byte. Ler como c_int devolve lixo -- ver a docstring.
        saida = ctypes.c_bool(False)
        prototipo = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_bool))
        hres = prototipo(endereco)(ponteiro, ctypes.byref(saida))
        return hres, bool(saida.value)
    if tipo == "ulong":
        saida = ctypes.c_ulong(0)
        prototipo = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_ulong))
        return prototipo(endereco)(ponteiro, ctypes.byref(saida)), saida.value
    #: Handle (INT_PTR) e ponteiros de interface: 8 bytes no 64 bits.
    saida = ctypes.c_void_p(0)
    prototipo = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_void_p))
    hres = prototipo(endereco)(ponteiro, ctypes.byref(saida))
    return hres, saida.value


def _hresult(valor):
    return "0x%08X" % (int(valor or 0) & 0xFFFFFFFF)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--todos", action="store_true",
                        help="chama TODAS as entradas nomeadas e mostra a saida")
    args = parser.parse_args(argv)

    print("probe_kinect_vtable: le o IsAvailable SEM comtypes (vtable na mao)")
    print("ordem do vtable lida de PyKinectV2.py:2532 (nao chutada)")

    sensor = ctypes.POINTER(PyKinectV2.IKinectSensor)()
    hres = ctypes.windll.kinect20.GetDefaultKinectSensor(ctypes.byref(sensor))
    print("  GetDefaultKinectSensor: %s" % _hresult(hres))
    if not bool(sensor):
        print("  ERRO: ponteiro nulo")
        return 3
    base = ctypes.cast(sensor, ctypes.c_void_p)
    vtable, funcoes = _funcoes(base)
    print("  interface 0x%X | vtable 0x%X" % (base.value or 0, vtable or 0))

    print("\n  --- caminho do pykinect2 (comtypes) ---")
    try:
        print("     IsAvailable (comtypes) = %s" % bool(sensor.IsAvailable))
    except Exception as exc:                                # noqa: BLE001 - sonda
        print("     IsAvailable (comtypes) LEVANTOU: %s" % exc)

    if args.todos:
        print("\n  --- todas as entradas, pelo vtable cru ---")
        for indice, nome, tipo in ENTRADAS:
            try:
                hr, valor = _chama(base, indice, tipo)
                print("     [%2d] %-32s %s  saida=%s"
                      % (indice, nome, _hresult(hr), valor))
            except Exception as exc:                        # noqa: BLE001 - sonda
                print("     [%2d] %-32s EXC %s" % (indice, nome, exc))

    print("\n  --- as duas que decidem, pelo vtable cru ---")
    hr_aberto, aberto = _chama(base, INDICE_IS_OPEN, "bool")
    hr_disp, disponivel = _chama(base, INDICE_IS_AVAILABLE, "bool")
    print("     [%d] IsOpen      -> %s  valor=%s"
          % (INDICE_IS_OPEN, _hresult(hr_aberto), aberto))
    print("     [%d] IsAvailable -> %s  valor=%s"
          % (INDICE_IS_AVAILABLE, _hresult(hr_disp), disponivel))

    print("\n  " + "=" * 68)
    if hr_disp != 0:
        print("  INCONCLUSIVO: a chamada crua devolveu %s (nao S_OK)."
              % _hresult(hr_disp))
        return 4
    if disponivel and not bool(sensor.IsAvailable):
        print("  VEREDITO: o vtable cru diz IsAvailable = TRUE e o comtypes FALSE.")
        print("  -> o problema esta' na BIBLIOTECA (comtypes/pykinect2). Conserto de")
        print("     codigo: ler pelo vtable, ou corrigir a definicao do comtypes.")
        return 1
    if disponivel:
        print("  VEREDITO: TRUE nos DOIS caminhos -- o sensor esta' disponivel.")
        print("  Siga com tools\\checa_kinect.py e va' para a bancada.")
        return 0
    print("  VEREDITO: FALSE nos DOIS caminhos (comtypes e vtable cru).")
    print("  -> NAO e' a biblioteca: o proprio SDK reporta o sensor indisponivel.")
    print("     Mesmo com o Studio mostrando imagem, o caminho do SDK nao inicializa")
    print("     o sensor -- o suspeito e' o LINK (porta/controladora/banda) ou o")
    print("     estado do driver.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
