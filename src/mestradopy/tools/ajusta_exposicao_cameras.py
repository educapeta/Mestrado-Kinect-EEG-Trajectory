"""Mede a SATURACAO das cameras e recomenda a exposicao certa (item 104).

POR QUE EXISTE: apos colocar mais luz na bancada (22/10/2026) a webcam Trust
ficou estourada e a calibracao da palma (teclas J/V) piorou. Estouro de luz nao
e' so' "feio": o MediaPipe perde a mao/landmarks, a normal do plano do dorso sai
ruidosa (RMS alto) e o gate de qualidade descarta amostras -- o refit e o
certificado falham mesmo com o IMU perfeito. O Kinect tem exposicao propria
(nao controlavel por flag) e tambem pode estourar com luz demais.

O que este programa faz:
  1) para cada webcam auxiliar (SIGMA/Trust), varre os valores de `--exposicao`
     (unidades do driver; no DirectShow e' log2 de segundos: -7 = 1/128 s),
     captura quadros em cada um e mede media/p99/% de pixels estourados/dark;
  2) escolhe a melhor exposicao (estouro <= ~2%, escuro <= ~5%, imagem clara) e
     imprime a LINHA DE COMANDO pronta para o groundtruth tool / paradigma;
  3) (--kinect) mede tambem o RGB do Kinect, que NAO tem flag de exposicao --
     se estiver estourado, a recomendacao e' fisica (menos luz, difusor, afastar
     a fonte da luz, ou reduzir a abertura do Kinect);
  4) (--salvar-preview) grava um PNG de cada camera na melhor exposicao para
     conferir depois.

Uso:
    .venv\\Scripts\\python.exe tools\\ajusta_exposicao_cameras.py
    .venv\\Scripts\\python.exe tools\\ajusta_exposicao_cameras.py --exposicoes "-11,-10,-9,-8,-7,-6,-5"
    .venv\\Scripts\\python.exe tools\\ajusta_exposicao_cameras.py --kinect --salvar-preview

Feche o groundtruth tool antes (as cameras ficam ocupadas enquanto ele roda).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from camera_uvc import define_modo, le_fourcc, trava_exposicao  # noqa: E402


def _metricas_quadro(quadro):
    """(media, p99, % estourados, % escuros) de um quadro BGR."""
    cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
    media = float(np.mean(cinza))
    p99 = float(np.percentile(cinza, 99))
    estourados = float(np.mean(np.any(quadro >= 250, axis=2)) * 100.0)
    escuros = float(np.mean(cinza <= 20) * 100.0)
    return media, p99, estourados, escuros


def _abre_camera(indice, largura, altura):
    """Abre a webcam como o tool faz (DSHOW+MJPEG, fallback MSMF). Devolve captura."""
    for backend, nome in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")):
        captura = cv2.VideoCapture(indice, backend)
        if not captura.isOpened():
            captura.release()
            continue
        define_modo(captura, largura, altura, mjpeg=True)
        captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        trava_exposicao(captura, exposicao=None, ganho=None, fps=None,
                        rotulo=f"aux {indice}")
        for _ in range(6):
            captura.read()
        ok, quadro = captura.read()
        if ok and quadro is not None:
            return captura, nome, le_fourcc(captura)
        captura.release()
    return None, None, None


def _mede_kinect(amostras=25):
    """(media, p99, % estourados, % escuros) do RGB do Kinect, ou None."""
    try:
        from pykinect2 import PyKinectRuntime, PyKinectV2
    except Exception:                                  # noqa: BLE001
        return None
    try:
        kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color)
    except Exception:                                  # noqa: BLE001
        return None
    try:
        quadros = []
        limite = time.monotonic() + 5.0
        while len(quadros) < amostras and time.monotonic() < limite:
            if kinect.has_new_color_frame():
                dados = kinect.get_last_color_frame()
                largura = kinect.color_frame_desc.Width
                altura = kinect.color_frame_desc.Height
                img = np.asarray(dados, dtype=np.uint8).reshape(
                    (altura, largura, 4))[:, :, :3].copy()
                quadros.append(img)
            else:
                time.sleep(0.01)
        if not quadros:
            return None
        metricas = np.array([_metricas_quadro(q) for q in quadros])
        return tuple(float(v) for v in np.median(metricas, axis=0))
    finally:
        try:
            kinect.close()
        except Exception:                              # noqa: BLE001
            pass


def _escolhe_melhor(resultados, limite_estouro=2.0, alvo_media=115.0):
    """Melhor exposicao: SEM estouro e com a media mais perto do alvo.

    O criterio antigo tentava punir "% escuro", mas a cena tem fundo escuro de
    verdade (parede/ambiente) -- isso reprovava exposicoes boas. O que importa
    para o MediaPipe e' nao ESTOURAR (pixels saturados) e ter luz suficiente:
    escolhe a exposicao com estouro <= limite cujo brilho medio fica mais perto
    do alvo (115 de 255).
    """
    candidatos = {exp: dados for exp, dados in resultados.items()
                  if dados["estourados"] <= limite_estouro}
    if candidatos:
        return min(candidatos,
                   key=lambda exp: abs(candidatos[exp]["media"] - alvo_media))
    return min(resultados, key=lambda exp: resultados[exp]["estourados"])


def main(argv=None):
    import kinect_imu_groundtruth as gt

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--indices", default=None,
                        help="indices OpenCV das webcams (padrao: as externas "
                             "detectadas por nome)")
    parser.add_argument("--exposicoes", default="-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2",
                        help="valores de exposicao a varrer (padrao "
                             "'-13..-2'; no DirectShow e' log2 de segundos, "
                             "-7 = 1/128 s)")
    parser.add_argument("--alvo-media", type=float, default=115.0,
                        help="brilho medio alvo (0-255) para escolher a melhor "
                             "exposicao (padrao 115)")
    parser.add_argument("--ganho", type=float, default=None,
                        help="se dado, tambem fixa o ganho nesse valor")
    parser.add_argument("--kinect", action="store_true",
                        help="mede tambem o RGB do Kinect (exposicao NAO e' "
                             "controlavel por flag)")
    parser.add_argument("--largura", type=int, default=1280)
    parser.add_argument("--altura", type=int, default=720)
    parser.add_argument("--salvar-preview", action="store_true")
    args = parser.parse_args(argv)

    if args.indices:
        indices = [int(p) for p in str(args.indices).replace(";", ",").split(",")
                   if p.strip()]
    else:
        indices = list(gt.indices_auxiliares_automaticos())
    if not indices:
        print("nenhuma webcam externa detectada -- confira o USB e use "
              "tools\\diagnostico_cameras.py --salvar")
        return 2
    exposicoes = [float(p) for p in str(args.exposicoes).replace(";", ",").split(",")
                  if p.strip()]

    recomendadas = {}
    for indice in indices:
        nome = gt.nome_auxiliar(indice) or f"camera {indice}"
        print(f"\n=== camera {indice} ({nome}) ===")
        captura, backend, fourcc = _abre_camera(indice, args.largura,
                                                args.altura)
        if captura is None:
            print("  NAO ABRIU (a camera pode estar ocupada -- feche o tool e "
                  "tente de novo)")
            continue
        print(f"  aberta em {backend} ({fourcc}); varrendo {len(exposicoes)} "
              "valores de exposicao...")
        resultados = {}
        try:
            for exposto in exposicoes:
                captura.set(cv2.CAP_PROP_EXPOSURE, float(exposto))
                if args.ganho is not None:
                    captura.set(cv2.CAP_PROP_GAIN, float(args.ganho))
                time.sleep(0.25)
                for _ in range(3):
                    captura.read()
                quadros = []
                for _ in range(6):
                    ok, quadro = captura.read()
                    if ok and quadro is not None:
                        quadros.append(quadro)
                if not quadros:
                    continue
                metricas = np.array([_metricas_quadro(q) for q in quadros])
                media, p99, estourados, escuros = (float(v) for v in
                                                  np.median(metricas, axis=0))
                resultados[float(exposto)] = {
                    "media": media, "p99": p99,
                    "estourados": estourados, "escuros": escuros}
        finally:
            captura.release()
        if not resultados:
            print("  nao consegui medir (sem quadros)")
            continue
        print("  exp   media   p99   estourados%  escuros%   veredito")
        for exposto, dados in resultados.items():
            if (dados["estourados"] <= 2.0 and dados["escuros"] <= 5.0
                    and dados["media"] >= 70.0):
                veredito = "OK"
            elif dados["estourados"] > 2.0:
                veredito = "ESTOURADO"
            elif dados["media"] < 70.0:
                veredito = "escuro"
            else:
                veredito = "aceitavel"
            print(f"  {exposto:5.0f} {dados['media']:7.1f} {dados['p99']:5.1f}"
                  f" {dados['estourados']:11.1f} {dados['escuros']:9.1f}   "
                  f"{veredito}")
        melhor = _escolhe_melhor(resultados, alvo_media=args.alvo_media)
        recomendadas[int(indice)] = melhor
        dados = resultados[melhor]
        print(f"  MELHOR: --exposicao {melhor:g} (media {dados['media']:.1f}, "
              f"{dados['estourados']:.1f}% estourado, "
              f"{dados['escuros']:.1f}% escuro)")
        if args.salvar_preview:
            captura, _backend, _fourcc = _abre_camera(indice, args.largura,
                                                      args.altura)
            if captura is not None:
                captura.set(cv2.CAP_PROP_EXPOSURE, float(melhor))
                time.sleep(0.3)
                for _ in range(4):
                    captura.read()
                ok, quadro = captura.read()
                captura.release()
                if ok and quadro is not None:
                    caminho = f"_exposicao_{indice}_{melhor:g}.png".replace(
                        "-", "m")
                    cv2.imwrite(caminho, quadro)
                    print(f"  preview salvo: {caminho}")

    if args.kinect:
        print("\n=== Kinect RGB (exposicao automatica, nao controlavel por flag) ===")
        metricas = _mede_kinect()
        if metricas is None:
            print("  nao consegui ler o Kinect (lib/modulo ausente ou ocupado)")
        else:
            media, p99, estourados, escuros = metricas
            print(f"  media {media:.1f} | p99 {p99:.1f} | {estourados:.1f}% "
                  f"estourado | {escuros:.1f}% escuro")
            if estourados > 2.0:
                print("  AVISO: RGB do Kinect ESTOURADO -- o MediaPipe perde "
                      "landmarks e a normal do plano fica ruidosa. Reduza a luz "
                      "fisica, use difusor ou afaste a luz do sensor.")
            elif media < 70.0:
                print("  AVISO: RGB do Kinect ESCURO -- aproxime/adicione luz.")
            else:
                print("  Kinect OK para o MediaPipe.")

    if recomendadas:
        texto = ",".join(f"{indice}:{valor:g}"
                         for indice, valor in sorted(recomendadas.items()))
        print("\nLinha de comando recomendada para o tool/paradigma:")
        print(f'  --exposicao "{texto}"'
              + (f" --ganho-cam {args.ganho:g}" if args.ganho is not None
                 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
