"""Sonda do KINECT v2: por que o sensor nao ENTREGA frame (com prova, nao palpite).

MEDIDO na bancada de 07/10/2026 (depois do uso do Kinect Studio):

    GetDefaultKinectSensor -> HRESULT 0x00000000 (S_OK)
    sensor.Open()          -> HRESULT 0x00000000 (S_OK)
    sensor.IsAvailable     -> FALSE          <-- a causa
    ColorFrameSource.OpenReader() + SubscribeFrameArrived() -> handle valido,
        e NENHUM frame em 20 s (color, depth, body e os tres juntos)

  Ou seja: o pykinect2 abre o sensor e assina os eventos -- o SDK e' que diz que
  o sensor NAO esta' disponivel. `IsAvailable` e' a propriedade que o SDK usa
  para "o sensor esta' inicializado e entregando quadros"; quando ela fica FALSE
  com o Studio mostrando imagem, o estado e' do LINK (USB 3.0 / driver), nao do
  programa. E' por isso que `has_new_color_frame()` devolve False para sempre: o
  evento nunca chega porque o SDK nunca habilita o stream.

  Os dois HRESULT sao a diferenca entre "nao achei o Kinect" (0x8007xxxx) e "achei,
  abri, e ele nao esta' disponivel" -- o `PyKinectRuntime` do pykinect2 NAO checa
  nenhum dos dois (o codigo guarda o retorno em `hres` e nunca compara), entao
  sem esta sonda os dois casos viram o mesmo "nao entrega frame".

Uso:
    # 1) ver o estado AGORA (IsAvailable, servico, tentativas)
    .venv\\Scripts\\python.exe tools\\probe_kinect_runtime.py

    # 2) observar Enquanto VOCE desliga/religa o USB: diz se o IsAvailable
    #    chega a virar TRUE em algum momento (se nao virar, e' o link/driver)
    .venv\\Scripts\\python.exe tools\\probe_kinect_runtime.py --observar 60

    # 3) conserto padrao do sintoma (precisa de terminal ELEVADO):
    .venv\\Scripts\\python.exe tools\\probe_kinect_runtime.py --reiniciar-servico

Codigo de saida: 0 = o sensor esta' disponivel E entregou frame; 1 = disponivel
mas sem frame; 2 = indisponivel (IsAvailable FALSE); 3 = nem abriu.
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import kinect_compat                                        # noqa: E402

kinect_compat.aplicar()

from pykinect2 import PyKinectV2                            # noqa: E402
from pykinect2.PyKinectRuntime import PyKinectRuntime       # noqa: E402

#: Servico que o SDK instala. Ele e' quem inicializa o sensor para os programas;
#: o Kinect Studio fala com o sensor por outro caminho, e e' por isso que ele
#: pode mostrar imagem enquanto `IsAvailable` esta' FALSE para todo o resto.
SERVICO = "KinectMonitor"


def _hresult(valor):
    return "0x%08X" % (int(valor) & 0xFFFFFFFF)


def abre_sensor():
    """(sensor, hres_get, hres_open, motivo) do caminho cru do SDK.

    Devolve o sensor JA' aberto (sem `PyKinectRuntime`), para poder ler
    `IsAvailable` -- que o `PyKinectRuntime` nao expoe.
    """
    sensor = ctypes.POINTER(PyKinectV2.IKinectSensor)()
    hres_get = ctypes.windll.kinect20.GetDefaultKinectSensor(ctypes.byref(sensor))
    if not bool(sensor):
        return None, hres_get, None, "GetDefaultKinectSensor devolveu ponteiro NULO"
    hres_open = sensor.Open()
    return sensor, hres_get, hres_open, None


def instancia_do_sensor():
    """(instancia, suspenso) do sensor no registro -- a IDENTIDADE do dispositivo.

    POR QUE ISTO IMPORTA (medido em 07/10/2026)
      Desligar e religar o USB deveria forcar uma ENUMERACAO NOVA. Se o
      `IsAvailable` nao muda E a instancia continua a MESMA, o Windows nao viu
      religacao nenhuma -- o sensor nunca saiu da arvore USB. Isso separa
      "enumerou de novo e ficou mudo" de "o Windows nem percebeu", e os consertos
      sao diferentes (driver x cabo/fonte/porta). Em OUTRA porta o Windows cria
      instancia nova (numero diferente); na MESMA porta ele reaproveita.
    """
    base = r"SYSTEM\CurrentControlSet\Enum\USB\VID_045E&PID_02C4&MI_00"
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as chave:
            instancia = winreg.EnumKey(chave, 0)
            with winreg.OpenKey(chave, instancia + r"\Device Parameters") as params:
                try:
                    suspenso = winreg.QueryValueEx(params,
                                                   "DeviceSelectiveSuspended")[0]
                except Exception:                           # noqa: BLE001
                    suspenso = None
            return instancia, suspenso
    except Exception:                                       # noqa: BLE001 - sonda
        return None, None


def estado_do_servico():
    """(status, start_type) do `KinectMonitor`, ou (None, None) se nao existe.

    Os ROTULOS do `sc query` sao traduzidos pelo Windows (`ESTADO`, `TIPO` em
    portugues) mas os VALORES nao (`RUNNING`, `AUTO_START`) -- por isso o parser
    procura os valores, nunca os rotulos. Ele ja' falhou por isso em 07/10/2026.
    """
    try:
        proc = subprocess.run(["sc", "query", SERVICO], capture_output=True,
                              text=True, timeout=20)
    except Exception:                                       # noqa: BLE001 - sonda
        return None, None
    texto = ((proc.stdout or "") + (proc.stderr or "")).upper()
    status = start = None
    if "RUNNING" in texto:
        status = "RUNNING"
    elif "STOPPED" in texto:
        status = "STOPPED"
    elif "DOES_NOT_EXIST" in texto or "1060" in texto:
        return None, None
    for valor in ("AUTO_START", "DEMAND_START", "DISABLED"):
        if valor in texto:
            start = valor
            break
    if start is None:
        #: O rotulo do StartType tambem e' traduzido; o REGISTRO nao. Start=2 e'
        #: automatico, 3 sob demanda, 4 desabilitado (constantes do Windows).
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Services\%s" % SERVICO) as chave:
                start = {2: "AUTO_START", 3: "DEMAND_START",
                         4: "DISABLED"}.get(winreg.QueryValueEx(chave, "Start")[0],
                                            "Start=?")
        except Exception:                                   # noqa: BLE001 - sonda
            start = None
    return status, start


def reinicia_servico():
    """Tenta parar e iniciar o `KinectMonitor` (precisa de terminal ELEVADO)."""
    print("  reiniciando o servico %s (precisa de administrador)..." % SERVICO)
    for acao in ("stop", "start"):
        proc = subprocess.run(["sc", acao, SERVICO], capture_output=True,
                              text=True, timeout=60)
        texto = ((proc.stdout or "") + (proc.stderr or "")).strip()
        print("    sc %s %s -> codigo %d | %s"
              % (acao, SERVICO, proc.returncode,
                 texto.replace("\n", " ")[:160]))
    return True


def observa(segundos, intervalo=0.5):
    """(viu_disponivel, viu_frame, historico) observando `IsAvailable` no tempo.

    E' o teste que separa "estado preso" de "link ruim": se o `IsAvailable` NUNCA
    vira TRUE enquanto voce desliga e religa o USB, o sensor nao esta'
    inicializando em nenhuma janela -- e ai o conserto e' de driver/porta, nao de
    software. Se ele VIRA TRUE, o sensor inicializa e o problema e' outro.
    """
    sensor, hres_get, hres_open, motivo = abre_sensor()
    if sensor is None:
        return False, False, [(0.0, "sem sensor: %s" % motivo)]
    runtime = None
    try:
        runtime = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    except Exception as exc:                                # noqa: BLE001 - sonda
        print("  (aviso: PyKinectRuntime nao abriu: %s)" % exc)
    historico = []
    viu_disponivel = False
    viu_frame = False
    inicio = time.time()
    anterior = None
    try:
        while time.time() - inicio < segundos:
            try:
                disponivel = bool(sensor.IsAvailable)
            except Exception as exc:                        # noqa: BLE001 - sonda
                historico.append((time.time() - inicio,
                                  "IsAvailable levantou %s" % exc))
                break
            if disponivel:
                viu_disponivel = True
            if runtime is not None and runtime.has_new_color_frame():
                viu_frame = True
            marca = (disponivel, viu_frame)
            if marca != anterior:
                historico.append((time.time() - inicio,
                                  "IsAvailable=%s frame=%s"
                                  % (disponivel, viu_frame)))
                anterior = marca
            time.sleep(intervalo)
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:                               # noqa: BLE001
                pass
    return viu_disponivel, viu_frame, historico


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observar", type=float, default=0.0,
                        help="segundos observando IsAvailable (0 = so' o estado)")
    parser.add_argument("--intervalo", type=float, default=0.5,
                        help="passo da observacao (s)")
    parser.add_argument("--reiniciar-servico", action="store_true",
                        help="para e inicia o %s (terminal elevado)" % SERVICO)
    parser.add_argument("--sem-frame", action="store_true",
                        help="nao tenta abrir frame (so' o estado do SDK)")
    args = parser.parse_args(argv)

    print("probe_kinect_runtime: por que o sensor nao entrega frame")
    print("compatibilidade:", kinect_compat.motivo())

    print("\n" + "=" * 72)
    print("1) servico %s" % SERVICO)
    print("=" * 72)
    status, start = estado_do_servico()
    if status is None:
        print("  nao consegui consultar (sem permissao?) -- rode num terminal "
              "normal: sc query %s" % SERVICO)
    else:
        print("  estado: %s | inicio: %s" % (status, start))
        if "RUNNING" not in status.upper():
            print("  -> o servico NAO esta' rodando: e' a causa classica de "
                  "IsAvailable FALSE em todo programa do SDK. Inicie-o "
                  "(servicos do Windows ou, elevado: sc start %s)." % SERVICO)

    print("\n" + "=" * 72)
    print("2) o sensor, pelo SDK cru (o que o pykinect2 NAO checa)")
    print("=" * 72)
    sensor, hres_get, hres_open, motivo = abre_sensor()
    if sensor is None:
        print("  GetDefaultKinectSensor: %s" % _hresult(hres_get))
        print("  ERRO: %s" % motivo)
        print("  -> nenhum sensor foi encontrado: driver/SDK, nao link de stream.")
        return 3
    print("  GetDefaultKinectSensor: %s (0 = S_OK)" % _hresult(hres_get))
    print("  sensor.Open()         : %s (0 = S_OK)" % _hresult(hres_open))
    disponivel = bool(sensor.IsAvailable)
    print("  sensor.IsAvailable    : %s   <-- e' ISTO que decide"
          % ("TRUE" if disponivel else "FALSE"))
    instancia, suspenso = instancia_do_sensor()
    if instancia:
        print("  instancia no registro : %s | DeviceSelectiveSuspended=%s"
              % (instancia, suspenso))
        print("     (religue o USB e olhe DE NOVO: instancia igual = o Windows nao "
              "viu religacao nenhuma)")
    if not disponivel:
        print("     IsAvailable FALSE = o SDK achou e abriu o sensor, mas ele NAO "
              "esta' inicializado/entregando. O Kinect Studio continua mostrando "
              "imagem porque fala com o sensor por outro caminho -- por isso o "
              "sintoma e' 'no Studio funciona, no Python nao'.")

    if args.reiniciar_servico:
        print("\n" + "=" * 72)
        print("3) reinicio do servico (conserto padrao deste sintoma)")
        print("=" * 72)
        reinicia_servico()
        time.sleep(3.0)

    if args.observar > 0:
        print("\n" + "=" * 72)
        print("4) observando por %.0f s -- DESLIGUE e RELIGUE o USB AGORA"
              % args.observar)
        print("=" * 72)
        viu_disponivel, viu_frame, historico = observa(
            args.observar, args.intervalo)
        for quando, o_que in historico:
            print("  %6.1f s  %s" % (quando, o_que))
        if not historico:
            print("  (nada mudou no periodo)")
        if viu_frame:
            print("\n  RESULTADO: o sensor ENTREGOU frame -- pode ir para a bancada.")
            return 0
        if viu_disponivel:
            print("\n  RESULTADO: IsAvailable virou TRUE (o sensor inicializou), mas "
                  "nao veio frame de cor. Repita com --observar e o tabuleiro fora "
                  "do quadro; se persistir, e' o stream, nao o link.")
            return 1
        print("\n  RESULTADO: o IsAvailable NAO virou TRUE em nenhum momento, nem "
              "durante a religacao.\n"
              "  -> o sensor nao inicializa: o conserto e' de LINK/DRIVER, nao de "
              "software. Ordem: (a) porta USB 3.0 azul DIRETO na placa (troque de "
              "controladora se houver outra); (b) fonte propria ligada antes do USB; "
              "(c) `sc stop %s` + `sc start %s` num terminal elevado, ou reiniciar "
              "o Windows; (d) reparar/reinstalar o 'Kinect for Windows SDK 2.0' "
              "(o Studio continuar funcionando nao descarta isto: o instalador "
              "tambem recria o registro do driver do sensor)." % (SERVICO, SERVICO))
        return 2

    if args.sem_frame:
        return 0 if disponivel else 2

    print("\n" + "=" * 72)
    print("3) o caminho do PROJETO (PyKinectRuntime + evento de cor)")
    print("=" * 72)
    try:
        runtime = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    except Exception as exc:                                # noqa: BLE001 - sonda
        print("  PyKinectRuntime NAO abriu: %s" % exc)
        return 3
    inicio = time.time()
    achou = False
    while time.time() - inicio < 8.0:
        if runtime.has_new_color_frame():
            achou = True
            break
        time.sleep(0.02)
    try:
        runtime.close()
    except Exception:                                       # noqa: BLE001
        pass
    print("  primeiro frame de cor: %s"
          % ("SIM em %.1f s" % (time.time() - inicio) if achou else "NENHUM em 8 s"))
    if achou:
        return 0
    print("\nRESUMO: IsAvailable=%s e nenhum frame. Use --observar 60 com o USB "
          "sendo desligado/religado para saber se ele chega a inicializar."
          % ("TRUE" if disponivel else "FALSE"))
    return 1 if disponivel else 2


if __name__ == "__main__":
    sys.exit(main())
