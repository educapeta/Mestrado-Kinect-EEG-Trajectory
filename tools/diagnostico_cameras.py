"""Diagnostico de CAMERAS da bancada (Kinect + webcams auxiliares).

Motivo: antes da calibracao stereo e' preciso saber QUAIS cameras existem, quais
entregam frame, em que resolucao e em que ordem (indice do OpenCV) -- e a falha
mais comum nao aparece no Python: a camera nao enumera no Windows por causa de
extensor USB sem fonte, cabo longo ou hub saturado (aparece como
"Dispositivo USB Desconhecido (Falha na Solicitacao de Descritor)" com ERRO 43).

Este script cobre a parte do OpenCV/Kinect (portatil, sem admin) e imprime, para
CADA indice, o NOME da webcam que o Windows tem ali (`camera_names.py`) -- sem
isso, "idx 0 abriu" nao diz nada: quando as webcams externas caem fora do USB, o
idx 0 passa a ser a webcam do LAPTOP e o programa abre o rosto de quem esta' no
laptop achando que abriu a auxiliar. Para a parte de USB do WINDOWS rode tambem,
no PowerShell:

    # camera faltando / com problema?
    Get-PnpDevice -PresentOnly -Class Camera | Select Status,FriendlyName
    Get-PnpDevice -PresentOnly | Where-Object { $_.Problem -ne 'CM_PROB_NONE' } |
        Select FriendlyName,Problem,InstanceId

    # em que porta/hub a camera esta pendurada
    Get-PnpDeviceProperty -InstanceId 'USB\\VID_0000&PID_0002\\...' -KeyName `
        DEVPKEY_Device_LocationInfo,DEVPKEY_Device_Address,DEVPKEY_Device_ProblemCode

Uso (da raiz do projeto):
    .venv\\Scripts\\python.exe tools\\diagnostico_cameras.py
    .venv\\Scripts\\python.exe tools\\diagnostico_cameras.py --salvar
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import camera_names  # noqa: E402  (nome real de cada indice; ver o modulo)

INDICES = (0, 1, 2, 3)
BACKENDS = ((cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW"))
LEITURAS = 8


def lista_nomes():
    """Mostra QUEM E' QUEM pelo NOME (o indice do OpenCV nao e' estavel).

    Isto e' o que responde a pergunta que o `--salvar` so' responde olhando PNG:
    qual indice e' a webcam do laptop e qual e' uma externa. O nome vem da
    enumeracao DirectShow, a MESMA ordem do backend CAP_DSHOW do OpenCV.
    """
    print("=== QUEM E' QUEM (nomes do Windows, ordem = indice do OpenCV) ===")
    for linha in camera_names.resumo():
        print("  " + linha)
    nomes = camera_names.nomes_das_cameras()
    if not nomes:
        print("  (nao deu para ler os nomes: comtypes/COM indisponivel)")
        return
    externas = camera_names.indices_fora_do_laptop()
    if externas:
        print("  cameras EXTERNAS (servem de auxiliar): %s" % (externas,))
        return
    print("  ATENCAO: NENHUMA camera externa nos nomes; a unica camera")
    print("  enumerada e' a do laptop. Nenhum indice pode ser 'a webcam")
    print("  auxiliar' enquanto as externas nao aparecerem -- confira cabo,")
    print("  hub USB com fonte e a porta (ver as dicas de USB no fim).")



def testa_indice(indice, backend, largura=1280, altura=720):
    """Abre o indice no backend e tenta UM frame. Devolve (frame|None, segundos)."""
    inicio = time.time()
    cap = cv2.VideoCapture(indice, backend)
    if not cap.isOpened():
        cap.release()
        return None, time.time() - inicio, None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    obtido = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    for _ in range(LEITURAS):
        ok, frame = cap.read()
        if ok:
            cap.release()
            return frame, time.time() - inicio, obtido
        time.sleep(0.05)
    cap.release()
    return None, time.time() - inicio, obtido


def testa_kinect():
    """RGB + profundidade do Kinect v2. Devolve True se os dois vieram."""
    try:
        from pykinect2 import PyKinectV2
        from pykinect2.PyKinectRuntime import PyKinectRuntime
    except Exception as exc:  # noqa: BLE001
        print(f"  Kinect: pykinect2 indisponivel ({exc})")
        return False
    try:
        rt = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color
                             | PyKinectV2.FrameSourceTypes_Depth)
    except Exception as exc:  # noqa: BLE001
        print(f"  Kinect: NAO abriu ({exc})")
        return False
    print("  Kinect: RGB {}x{} | depth {}x{}".format(
        rt.color_frame_desc.Width, rt.color_frame_desc.Height,
        rt.depth_frame_desc.Width, rt.depth_frame_desc.Height))
    fim = time.time() + 8
    cor = profundidade = False
    mediana = -1.0
    while time.time() < fim and not (cor and profundidade):
        if not cor and rt.has_new_color_frame():
            cor = True
        if not profundidade and rt.has_new_depth_frame():
            dados = rt.get_last_depth_frame()
            arr = dados.reshape((rt.depth_frame_desc.Height,
                                 rt.depth_frame_desc.Width))
            validos = arr[arr > 0]
            mediana = float(np.median(validos)) if validos.size else -1.0
            profundidade = True
        time.sleep(0.02)
    rt.close()
    print("  Kinect: cor={} profundidade={} (mediana {} mm)".format(
        cor, profundidade, "%.0f" % mediana if mediana > 0 else "-"))
    return cor and profundidade


def lista_calibracoes(indices):
    """Mostra a prontidao de calibracao por camera auxiliar."""
    try:
        import kinect_imu_groundtruth as gt
    except Exception as exc:  # noqa: BLE001
        print("  (nao deu para importar kinect_imu_groundtruth: %s)" % exc)
        return
    try:
        alinhamento = gt.CameraAlignment(indices)
    except Exception as exc:  # noqa: BLE001
        print("  (CameraAlignment falhou: %s)" % exc)
        return
    for indice in indices:
        stereo = gt.aux_stereo_file(indice)
        mao = gt.aux_hand_file(indice)
        real = camera_names.nome_da_camera(indice)
        # O nome do Windows manda: `gt.AUX_CAMERA_NAMES` e' um mapa ESTATICO
        # (apelido por indice) e ja' ficou desatualizado uma vez.
        rotulo = real or gt.AUX_CAMERA_NAMES.get(indice, "?")
        if camera_names.e_do_laptop(real):
            rotulo += "  [webcam do LAPTOP]"
        print("  camera %d (%s)" % (indice, rotulo))
        # A PLACA GRAVADA no npz e' mostrada de propositio: desde a troca do
        # tabuleiro (placa nova de 6x4 quadrados / 57 mm) uma calibracao antiga
        # e' recusada por INCOMPATIBILIDADE, nao por qualidade -- sem isto o
        # "pronto=False" pareceria problema de luz/postura e a acao (recalibrar
        # a camera) passaria escondida.
        placa = "?"
        if stereo is not None:
            try:
                with np.load(stereo) as dados:
                    if "checkerboard_size" in dados.files:
                        cols, rows = (int(v) for v in dados["checkerboard_size"])
                        lado = (float(dados["square_size_m"]) * 1000.0
                                if "square_size_m" in dados.files
                                else float("nan"))
                        placa = "%dx%d cantos @ %g mm" % (cols, rows, lado)
            except (OSError, ValueError, KeyError):
                placa = "npz ilegivel"
        estado, motivo = alinhamento.stereo_status(indice)
        print("     stereo : %-38s %s" % (
            stereo.name if stereo is not None
            else "(falta stereo_calibration_aux%d.npz)" % indice,
            estado))
        print("     placa  : %-38s %s" % (placa, motivo))
        print("     mao    : %-38s utilizavel=%s" % (
            mao.name if mao is not None
            else "(falta hand_landmark_calibration_aux%d.npz)" % indice,
            alinhamento.aux_usable(indice)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--salvar", action="store_true",
                        help="Salva um frame de cada camera que funcionar "
                             "(cam_idxN.png) para inspecao")
    args = parser.parse_args()

    print("OpenCV %s" % cv2.__version__)
    print("=== KINECT v2 ===")
    testa_kinect()

    lista_nomes()

    print("=== WEBCAMS AUXILIARES (pedindo 1280x720) ===")
    funcionais = []
    for indice in INDICES:
        achou = False
        for backend, nome in BACKENDS:
            frame, gasto, obtido = testa_indice(indice, backend)
            if frame is not None:
                # Sem o nome, "idx 0 abriu" nao diz nada: pode ser a webcam do
                # LAPTOP (foi o que aconteceu quando as externas cairam fora).
                nome_real = camera_names.nome_da_camera(indice) or "?"
                print("  idx %d via %-5s: OK %dx%d media=%.1f preto=%s (%.1fs)"
                      " | %s" % (indice, nome, frame.shape[1], frame.shape[0],
                                 float(frame.mean()),
                                 bool(frame.mean() < 2.0), gasto, nome_real))
                if camera_names.aviso_laptop(indice):
                    print("        %s" % camera_names.aviso_laptop(indice))
                funcionais.append(indice)
                if args.salvar:
                    caminho = "cam_idx%d.png" % indice
                    cv2.imwrite(caminho, frame)
                    print("        frame salvo em %s" % caminho)
                achou = True
                break
            if obtido is not None:
                print("  idx %d via %-5s: abriu (%dx%d) mas SEM frame (%.1fs)"
                      % (indice, nome, obtido[0], obtido[1], gasto))
            else:
                print("  idx %d via %-5s: NAO abriu (%.1fs)"
                      % (indice, nome, gasto))
            if gasto > 3.0:            # backend que trava: nao insiste
                break
        if not achou:
            print("  idx %d: INDISPONIVEL" % indice)

    print("=== CALIBRACAO (por camera auxiliar) ===")
    lista_calibracoes(INDICES)

    print("=== RESUMO ===")
    print("  cameras entregando frame: %s" % (funcionais or "nenhuma"))
    nomes = camera_names.nomes_das_cameras()
    externas = camera_names.indices_fora_do_laptop()
    if nomes and not externas:
        print("  VEREDITO: so' a webcam do LAPTOP esta' enumerada. Qualquer")
        print("  indice que abrir e' ELA -- nao ha' camera auxiliar possivel")
        print("  ate' as externas aparecerem no Windows (nome, nao indice).")
    elif externas:
        usaveis = [i for i in externas if i in funcionais]
        print("  VEREDITO: externas nos nomes: %s | entregando frame: %s"
              % (externas, usaveis or "nenhuma"))
        if not usaveis:
            print("  (as externas existem mas nao entregam frame: backend "
                  "ocupado ou hub com problema)")
    if not funcionais:
        print("  DICA (PowerShell), procure ERRO 43 / descritor:")
        print("    Get-PnpDevice -PresentOnly | "
              "Where-Object { $_.Problem -ne 'CM_PROB_NONE' } | "
              "Select FriendlyName,Problem,InstanceId")
    return 0


if __name__ == "__main__":
    sys.exit(main())
