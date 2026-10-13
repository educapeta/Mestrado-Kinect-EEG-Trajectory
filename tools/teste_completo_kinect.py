"""Teste completo do Kinect: roda o veredito, o loop e a pre-checagem num so' lugar.

O QUE ELE RESPONDE (bancada de 07/10/2026)
  Depois de mexer no LINK (tirar o Kinect do barramento cheio e liga-lo sozinho
  numa porta SuperSpeed), o que interessa e' uma coisa so': o sensor ficou
  DISPONIVEL e ENTREGANDO frame? Este programa junta as tres medicoes que
  respondem isso, na ordem:
    1. `sensor.IsAvailable` pelo vtable CRU (sem comtypes) + a instancia no
       registro -- o passo que decide, e o que diz se o Windows chegou a ver o
       dispositivo mudar de porta;
    2. o caminho OFICIAL do projeto (`sc.abre_kinect`, com reabertura) entrega
       frame? (pode levar ate' 3 x 20 s quando esta' mudo -- `--rapido` corta
       para uma tentativa de 5 s);
    3. webcams por indice -> NOME (o indice do OpenCV e' a posicao no USB e muda
       quando se mexe nos cabos -- e' o que evita calibrar a camera errada).
  Codigo de saida: 0 = pode ir para a bancada; 1 = ainda mudo; 2 = nem abriu.

Uso:
    .venv\\Scripts\\python.exe tools\\teste_completo_kinect.py
    .venv\\Scripts\\python.exe tools\\teste_completo_kinect.py --rapido
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

import stereo_calibration as sc                             # noqa: E402
import probe_kinect_runtime as pr                           # noqa: E402
import probe_kinect_vtable as pv                            # noqa: E402


def titulo(texto):
    print("\n" + "=" * 72)
    print(texto)
    print("=" * 72)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rapido", action="store_true",
                        help="uma tentativa de 5 s no caminho do projeto")
    args = parser.parse_args(argv)

    print("teste_completo_kinect: o sensor esta' disponivel e ENTREGANDO?")
    print("compatibilidade:", kinect_compat.motivo())

    titulo("1) o passo que decide: sensor.IsAvailable")
    sensor = ctypes.POINTER(pv.PyKinectV2.IKinectSensor)()
    hres_get = ctypes.windll.kinect20.GetDefaultKinectSensor(ctypes.byref(sensor))
    print("  GetDefaultKinectSensor: %s" % pv._hresult(hres_get))
    if not bool(sensor):
        print("  ERRO: ponteiro nulo -- o SDK nao devolveu sensor nenhum.")
        return 2
    base = ctypes.cast(sensor, ctypes.c_void_p)
    hres_open = sensor.Open()
    print("  sensor.Open()         : %s" % pv._hresult(hres_open))
    hr_aberto, aberto = pv._chama(base, pv.INDICE_IS_OPEN, "bool")
    hr_disp, disponivel = pv._chama(base, pv.INDICE_IS_AVAILABLE, "bool")
    print("  IsOpen      (vtable cru): %s valor=%s" % (pv._hresult(hr_aberto), aberto))
    print("  IsAvailable (vtable cru): %s valor=%s"
          % (pv._hresult(hr_disp), disponivel))
    try:
        print("  IsAvailable (comtypes) : %s" % bool(sensor.IsAvailable))
    except Exception as exc:                                # noqa: BLE001 - sonda
        print("  IsAvailable (comtypes) : LEVANTOU %s" % exc)

    instancia, suspenso = pr.instancia_do_sensor()
    print("  instancia no registro   : %s | DeviceSelectiveSuspended=%s"
          % (instancia, suspenso))
    status, inicio = pr.estado_do_servico()
    print("  servico KinectMonitor   : %s | %s" % (status, inicio))

    if not disponivel:
        titulo("RESUMO")
        print("  O SENSOR AINDA NAO ESTA' DISPONIVEL (IsAvailable = FALSE).")
        print("  O registro mostra a instancia %s" % instancia)
        print("  -> se voce MUDOU o Kinect de porta e esta instancia nao mudou, o")
        print("     Windows NAO viu o dispositivo chegar de novo: confira se o cabo")
        print("     de dados esta' firme, se o LED branco esta' aceso, e se o som de")
        print("     'dispositivo conectado' tocou ao religar.")
        print("  Diagnostico do link: tools\\probe_kinect_loop.py --observar 60")
        return 1

    titulo("2) o caminho OFICIAL do projeto entrega frame?")
    if args.rapido:
        espera, tentativas = 5.0, 1
    else:
        espera, tentativas = sc.ESPERA_KINECT_S, None
    print("  espera %.0f s, tentativas %s" % (espera, tentativas or sc.TENTATIVAS_KINECT))
    entrega = False
    try:
        runtime = sc.abre_kinect(espera_s=espera, tentativas=tentativas)
        entrega = True
        try:
            runtime.close()
        except Exception:                                   # noqa: BLE001
            pass
        print("  RESULTADO: ENTREGA frame.")
    except RuntimeError as exc:
        print("  RESULTADO: nao entregou.")
        print("  %s" % exc)

    titulo("3) webcams (indice OpenCV -> nome do Windows)")
    try:
        import camera_names
        for indice in range(4):
            nome = camera_names.nome_da_camera(indice)
            print("  indice %d: %s" % (indice, nome if nome else "(nada)"))
    except Exception as exc:                                # noqa: BLE001 - sonda
        print("  indisponivel: %s" % exc)

    titulo("RESUMO")
    if entrega:
        print("  O KINECT ESTA' ENTREGANDO FRAME. Pode ir para a bancada:")
        print("    1. tools\\checa_kinect.py            (confirma tudo, sai com 0)")
        print("    2. python freeze_kinect_direto.py    <- Fase 1 (12+ poses)")
        print("    3. tools\\conserta_intrinsecas_kinect.py --sim   (aposenta o paliativo)")
        print("    4. python stereo_calibration.py --aux-index 0  e  --aux-index 1")
        print("    5. tools\\diagnostico_calibracao.py  e a tecla T no groundtruth tool")
        return 0
    print("  DISPONIVEL, mas o caminho do projeto NAO entregou frame.")
    print("  Repita com --rapido se quiser um teste curto, e rode")
    print("  tools\\probe_kinect_runtime.py --observar 60 com o Studio FECHADO.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
