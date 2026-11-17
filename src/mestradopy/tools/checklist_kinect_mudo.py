"""Checklist do Kinect MUDO: cada passo com o que medir e o comando exato.

ONDE ESTAMOS (medido em 07/10/2026, Windows 10 Pro 22H2 build 19045.6466)
  O sensor ABRE (`GetDefaultKinectSensor` = S_OK, `sensor.Open()` = S_OK) e o
  `sensor.IsAvailable` fica FALSE -- em 5 aberturas, nos 3 streams e depois de
  REINICIAR o Windows. Ja' foram DESCARTADOS por medicao:

    * hardware queimado / cabo fora: enumerado `USB\\VID_045E&PID_02C4` + MI_00 +
      MI_02, driver `KinectSensor.sys` 2.0.1410.18000 com `ConfigFlags=0`;
    * `Kinect20.dll` errado: unico e legitimo no System32 (2.0.1410.18000);
    * servico do SDK: `KinectMonitor` RUNNING / AUTO_START;
    * as MELHORIAS DE AUDIO (causa conhecida do Windows 11): o endpoint do
      Kinect nao tem `FxProperties` (a aba nem aparece) e o usuario DESABILITOU o
      dispositivo de audio inteiro + reiniciou -- `IsAvailable` continua FALSE;
    * o projeto nao usa audio (varredura: nenhum sounddevice/pyaudio/AudioBeam).

  O que SOBRA, em ordem de custo -- e' este o checklist que o programa imprime:

    1. SUSPENSAO SELETIVA DE USB. MEDIDO nesta maquina: plano "Equilibrado" com
       `Configuracao de suspensao seletiva USB = Ativado` (0x1 em CA e CC) e o
       proprio sensor marcado `DeviceSelectiveSuspended = 1` no registro. O v2
       nao volta a transmitir de forma confiavel depois de suspender. Desligar
       custa um comando (elevado).
    2. PORTA USB 3.0 DE VERDADE, DE OUTRA CONTROLADORA. O v2 precisa das DUAS
       faixas de banda; em porta 2.0 (ou hub/extensao/painel frontal) ele
       enumera e nao transmite. Foi visto num caso real de USB "nao bom"
       reportado pelo proprio verificador da Microsoft.
    3. FONTE PROPRIA ligada ANTES do USB.
    4. REINSTALAR O DISPOSITIVO no Gerenciador de Dispositivos (forca nova
       enumeracao; e' o que o "desligar/religar" faz pelo lado do software).
    5. VERIFICADOR OFICIAL ("Kinect Configuration Verifier", do SDK): ele
       valida a CONTROLADORA USB e diz em texto o que esta' fora.
    6. REPARAR/REINSTALAR o "Kinect for Windows SDK 2.0".

Uso:
    .venv\\Scripts\\python.exe tools\\checklist_kinect_mudo.py
    .venv\\Scripts\\python.exe tools\\checklist_kinect_mudo.py --estado
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import kinect_compat                                        # noqa: E402

kinect_compat.aplicar()

import probe_kinect_runtime as pr                           # noqa: E402

#: GUID da configuracao "suspensao seletiva de USB" (subgrupo de USB do Windows).
SUBGRUPO_USB = "2a737441-1930-4402-8d77-b2bebba308a3"
CONFIG_SUSPENSAO_SELETIVA = "48e6b7a6-50f5-4782-a5d4-53bb8f07e226"


def suspensao_seletiva():
    """(ca, cc) da suspensao seletiva de USB: 0 desligada, 1 ligada, None n/d.

    Le pelo `powercfg /query` (nao pelo registro) porque e' o mesmo numero que a
    UI de Opcoes de Energia mostra -- e o comando de conserto e' o inverso dele.
    """
    try:
        #: `text=True` sozinho usa o codec do console (cp1252/CP850 aqui) e
        #: MORRE com `UnicodeDecodeError` na saida em portugues do powercfg --
        #: medido em 07/10/2026 (byte 0x8d em "Correntes Continuas").
        proc = subprocess.run(
            ["powercfg", "/query", "SCHEME_CURRENT", SUBGRUPO_USB,
             CONFIG_SUSPENSAO_SELETIVA],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30)
        saida = proc.stdout or ""
    except Exception:                                       # noqa: BLE001 - sonda
        return None, None
    atual = {}
    for linha in saida.splitlines():
        baixo = linha.lower()
        if "correntes alternadas atuais" in baixo or "current ac" in baixo:
            atual["ca"] = 1 if "0x00000001" in linha else 0
        if "correntes contínuas atuais" in baixo or "correntes continuas atuais" in baixo \
                or "current dc" in baixo:
            atual["cc"] = 1 if "0x00000001" in linha else 0
    return atual.get("ca"), atual.get("cc")


def estado():
    """Imprime o estado medido agora (o que ainda nao foi descartado)."""
    print("=" * 72)
    print("ESTADO MEDIDO AGORA")
    print("=" * 72)
    sensor, hres_get, hres_open, motivo = pr.abre_sensor()
    if sensor is None:
        print("  sensor: NAO ENCONTRADO (%s)" % motivo)
    else:
        print("  GetDefaultKinectSensor: %s" % pr._hresult(hres_get))
        print("  sensor.Open()         : %s" % pr._hresult(hres_open))
        print("  sensor.IsAvailable    : %s"
              % ("TRUE" if sensor.IsAvailable else "FALSE"))
    status, inicio = pr.estado_do_servico()
    print("  servico KinectMonitor : %s | %s" % (status, inicio))
    ca, cc = suspensao_seletiva()
    rotulo = {0: "DESLIGADA", 1: "LIGADA", None: "nao consegui ler"}
    print("  suspensao seletiva USB: CA=%s CC=%s"
          % (rotulo.get(ca, ca), rotulo.get(cc, cc)))
    if ca == 1 or cc == 1:
        print("     -> LIGADA e' suspeito: o v2 nao volta a transmitir de forma "
              "confiavel depois de suspender.")
    return ca, cc


PASSOS = (
    ("1", "SUSPENSAO SELETIVA DE USB (1 minuto, elevado)",
     ["powercfg /setacvalueindex SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 "
      "48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0",
      "powercfg /setdcvalueindex SCHEME_CURRENT 2a737441-1930-4402-8d77-b2bebba308a3 "
      "48e6b7a6-50f5-4782-a5d4-53bb8f07e226 0",
      "powercfg /setactive SCHEME_CURRENT"],
     "Depois: DESLIGUE e RELIGUE a fonte do Kinect e rode o probe de novo. "
     "Voltar atras: mesmo comando com 1 no ultimo numero."),
    ("2", "PORTA USB 3.0 DE OUTRA CONTROLADORA (fisico)",
     [],
     "Olhe o conector: o v2 exige USB 3.0 (porta AZUL, geralmente com 'SS'). "
     "Hub, extensao, painel frontal e adaptador NAO servem. Se a maquina tiver "
     "mais de uma controladora (Intel + ASMedia/Renesas, por exemplo), use a "
     "OUTRA -- e' o passo com mais relatos de sucesso quando o IsAvailable fica "
     "FALSE e o Studio funciona."),
    ("3", "FONTE PROPRIA do Kinect (fisico)",
     [],
     "A fonte tem de estar ligada ANTES do USB e o LED branco do sensor tem de "
     "estar aceso. Sem ela o sensor enumera e fica mudo."),
    ("4", "REINSTALAR O DISPOSITIVO (Gerenciador de Dispositivos)",
     [],
     "Gerenciador de Dispositivos > Kinect sensor devices > 'WDF KinectSensor "
     "Interface 0' > botao direito > DESINSTALAR (sem marcar 'apagar driver') > "
     "menu Acao > 'Verificar se ha alteracoes de hardware'. Isso forca uma "
     "enumeracao nova -- e' o 'desligar/religar' pelo lado do software."),
    ("5", "VERIFICADOR OFICIAL DA MICROSOFT",
     [],
     "'Kinect Configuration Verifier' (vem com o SDK 2.0). Ele testa a "
     "CONTROLADORA USB e escreve em texto o que esta' fora -- se a controladora "
     "for reprovada, os passos 1-4 nao resolvem e o caminho e' placa PCIe USB 3.0 "
     "com chipset compativel."),
    ("6", "REPARAR/REINSTALAR o 'Kinect for Windows SDK 2.0'",
     [],
     "O Studio continuar funcionando NAO descarta isto: o instalador recria o "
     "registro do driver do sensor. Reparar antes de reinstalar."),
)


def imprime_checklist(ca, cc):
    print("")
    print("=" * 72)
    print("CHECKLIST -- na ordem (o agente nao pode executar nenhum destes)")
    print("=" * 72)
    for numero, titulo, comandos, explicacao in PASSOS:
        print("\n%s) %s" % (numero, titulo))
        for linha in explicacao.split(". "):
            if linha.strip():
                print("     %s" % linha.strip())
        for comando in comandos:
            print("     > %s" % comando)
        if numero == "1" and (ca == 1 or cc == 1):
            print("     (medido agora: LIGADA -- comece por aqui)")
    print("\n" + "=" * 72)
    print("DEPOIS DE CADA PASSO: .venv\\Scripts\\python.exe tools\\probe_kinect_runtime.py --sem-frame")
    print("O alvo e' ver `sensor.IsAvailable : TRUE`. So' entao:")
    print("  tools\\checa_kinect.py  ->  freeze_kinect_direto.py  ->  Fases 2-4")
    print("=" * 72)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estado", action="store_true",
                        help="so' o estado medido (sem o checklist)")
    args = parser.parse_args(argv)
    ca, cc = estado()
    if not args.estado:
        imprime_checklist(ca, cc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
