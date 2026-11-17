"""Verificacao ANTES/DEPOIS do reinicio do Kinect (o flag preso).

POR QUE EXISTE (bancada de 07/10/2026)
  O usuario tirou o Kinect do barramento cheio e ligou sozinho numa porta
  SuperSpeed (o Studio parou de travar), mas `IsAvailable` continuou FALSE. A
  MEDICAO mostrou por que: o `DeviceSelectiveSuspended` da interface do SENSOR
  (`MI_00`) ficou em **1** -- preso de antes da mudanca de porta. O composto
  estava limpo e o audio tambem: so' o sensor.

  Suspensao presa explica o sintoma inteiro: o dispositivo esta' na arvore USB,
  `Open()` devolve S_OK e `IsOpen` vira TRUE, mas o SDK nunca ve o sensor
  ACORDADO -- e `IsAvailable` so' vira TRUE com o sensor pronto. Um reinicio do
  Windows zera esse estado (nao da' para zerar sem elevacao: `MSPower_DeviceEnable`
  e `pnputil` devolvem "Acesso negado" para o agente).

  Uso -- rode UMA vez ANTES do reinicio e UMA vez DEPOIS:
      .venv\\Scripts\\python.exe tools\\verifica_kinect_antes_depois.py --marcar
      (reinicie o Windows, Kinect Studio FECHADO)
      .venv\\Scripts\\python.exe tools\\verifica_kinect_antes_depois.py

  Codigo de saida: 0 = disponivel (e ENTREGANDO frame, se --entrega); 1 = ainda
  mudo; 2 = nem abriu.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import kinect_compat                                        # noqa: E402

kinect_compat.aplicar()

import probe_kinect_runtime as pr                           # noqa: E402
import probe_kinect_vtable as pv                            # noqa: E402

#: Onde fica a marca do estado de antes. Ignorado pelo git (`_*.json`).
ARQUIVO = RAIZ / "_kinect_estado_antes.json"


def estado_atual():
    """(dicionario) do estado medido agora -- tudo o que decide, num lugar so'."""
    dados = {
        "quando": time.strftime("%Y-%m-%d %H:%M:%S"),
        "instancia": None,
        "suspenso_sensor": None,
        "is_available": None,
        "is_open": None,
        "hres_get": None,
        "hres_open": None,
    }
    dados["instancia"], dados["suspenso_sensor"] = pr.instancia_do_sensor()
    sensor = ctypes.POINTER(pv.PyKinectV2.IKinectSensor)()
    hres = ctypes.windll.kinect20.GetDefaultKinectSensor(ctypes.byref(sensor))
    dados["hres_get"] = int(hres)
    if not bool(sensor):
        return dados
    base = ctypes.cast(sensor, ctypes.c_void_p)
    dados["hres_open"] = int(sensor.Open())
    _, aberto = pv._chama(base, pv.INDICE_IS_OPEN, "bool")
    _, disponivel = pv._chama(base, pv.INDICE_IS_AVAILABLE, "bool")
    dados["is_open"] = aberto
    dados["is_available"] = disponivel
    return dados


def imprime(dados):
    print("  quando                  : %s" % dados["quando"])
    print("  instancia (registro)    : %s" % dados["instancia"])
    print("  MI_00 suspenso          : %s   (1 = PRESO suspenso)"
          % dados["suspenso_sensor"])
    print("  GetDefaultKinectSensor  : 0x%08X"
          % ((dados["hres_get"] or 0) & 0xFFFFFFFF))
    print("  sensor.Open()           : %s"
          % ("-" if dados["hres_open"] is None
             else "0x%08X" % (dados["hres_open"] & 0xFFFFFFFF)))
    print("  IsOpen                  : %s" % dados["is_open"])
    print("  IsAvailable             : %s   <-- o que decide"
          % dados["is_available"])


def compara(antes, depois):
    print("\n  --- ANTES x DEPOIS ---")
    for rotulo, chave in (("instancia", "instancia"),
                          ("MI_00 suspenso", "suspenso_sensor"),
                          ("IsOpen", "is_open"),
                          ("IsAvailable", "is_available")):
        print("     %-16s %-18s -> %s"
              % (rotulo, antes.get(chave), depois.get(chave)))
    if antes.get("suspenso_sensor") == 1 and depois.get("suspenso_sensor") == 1:
        print("     ATENCAO: o flag de suspensao CONTINUA preso depois do reinicio.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--marcar", action="store_true",
                        help="grava o estado de ANTES (rode antes de reiniciar)")
    parser.add_argument("--entrega", action="store_true",
                        help="se estiver disponivel, testa o caminho do projeto")
    args = parser.parse_args(argv)

    print("verifica_kinect_antes_depois: o flag preso e o veredito do reinicio")
    dados = estado_atual()

    if args.marcar:
        ARQUIVO.write_text(json.dumps(dados, indent=2), encoding="utf-8")
        print("\n=== ESTADO DE ANTES (gravado em %s) ===" % ARQUIVO.name)
        imprime(dados)
        print("\nAgora reinicie o Windows (com o Kinect Studio FECHADO) e rode este")
        print("mesmo comando SEM --marcar.")
        return 0 if dados["is_available"] else 1

    antes = None
    if ARQUIVO.exists():
        try:
            antes = json.loads(ARQUIVO.read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001 - sonda
            antes = None
    print("\n=== ESTADO DE AGORA ===")
    imprime(dados)
    if antes:
        compara(antes, dados)
    else:
        print("\n  (sem estado de antes: rode com --marcar antes de reiniciar para")
        print("   ter a comparacao)")

    if not dados["is_available"]:
        print("\nRESUMO: o sensor AINDA NAO esta' disponivel.")
        print("  NAO conclua pelo `suspenso` acima: MEDIDO em 07/10/2026, o")
        print("  `DeviceSelectiveSuspended=1` do registro aparece tambem quando o")
        print("  Kinect FUNCIONA, e a chave que realmente decide")
        print("  (`SelectiveSuspendEnabled`) nao existe nesta maquina. Ele e' um")
        print("  rotulo persistente, nao prova de suspensao vivo.")
        print("  O que decide de verdade, ja' medido: nem o `IsAvailable` (comtypes")
        print("  e vtable cru) nem o EVENTO `IsAvailableChanged` do SDK -- este")
        print("  ultimo testado com tools\\probe_kinect_isavailable_event.py (20 s,")
        print("  handle valido, ZERO eventos).")
        print("  Proximos passos (na ordem):")
        print("    1. desinstalar o AUDIO do Kinect (a interface MI_02 ainda carrega")
        print("       `usbaudio`) e reiniciar;")
        print("    2. KinectV2ConfigurationVerifier.exe (avalia a controladora USB);")
        print("    3. so' depois de (2) reprovar: placa PCIe USB 3.0 compativel.")
        return 1

    print("\nRESUMO: o sensor ESTA' DISPONIVEL.")
    if args.entrega:
        import stereo_calibration as sc                     # noqa: E402
        print("\n=== o caminho do projeto entrega frame? ===")
        try:
            runtime = sc.abre_kinect(espera_s=sc.ESPERA_KINECT_S)
            try:
                runtime.close()
            except Exception:                               # noqa: BLE001
                pass
            print("  ENTREGA frame -- va' para tools\\checa_kinect.py e a Fase 1.")
            return 0
        except RuntimeError as exc:
            print("  NAO entregou: %s" % exc)
            return 1
    print("  Rode com --entrega para testar o frame, e depois tools\\checa_kinect.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
