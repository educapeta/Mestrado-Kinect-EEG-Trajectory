"""Mede POR QUE o tabuleiro aparece e desaparece durante a calibracao.

Tres causas possiveis, e este script separa as tres com numeros:

1. CUSTO DO DETECTOR (video travado). findChessboardCornersSB custa ~O(px^2);
   com CALIB_CB_EXHAUSTIVE|CALIB_CB_ACCURACY em 1920x1080 ele sozinho passa de
   100 ms por frame. Como o codigo detecta a cada DETECTION_INTERVAL_S (0.1 s)
   e reusa cantos velhos no meio, um detector caro demais aparece como VIDEO
   TRAVADO, nao como erro.
2. CONTRASTE DA IMPRESSAO (pisca mesmo com o tabuleiro visivel). Impressao
   fraca, papel brilhante ou luz desigual deixam o detector na fronteira e ele
   oscila entre achar e nao achar. Aqui medimos o contraste do cinza e testamos
   CLAHE, que costuma resolver.
3. TAMANHO NA IMAGEM. Abaixo de MIN_BOARD_SPAN_FRACTION da diagonal o proprio
   codigo rejeita a amostra -- e o sintoma e' o mesmo "piscar". Medimos a
   fracao da diagonal que o tabuleiro ocupa.

Uso (da raiz do projeto, com o tabuleiro NA FRENTE da camera):
    .venv\\Scripts\\python.exe tools\\diagnostico_tabuleiro.py
    .venv\\Scripts\\python.exe tools\\diagnostico_tabuleiro.py --fonte aux --idx 1
    .venv\\Scripts\\python.exe tools\\diagnostico_tabuleiro.py --salvar

Saida: FPS de entrega da camera, contraste do cinza, e uma tabela de
tempo/deteccao para cada combinacao (escala x CLAHE x flags). Com --salvar
grava tambem o cinza COMO O DETECTOR O VE (tabuleiro_cinza_*.png), que mostra
na hora se a impressao esta fraca.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stereo_calibration as sc

CHECKERBOARD_SIZE = sc.CHECKERBOARD_SIZE
MIN_SPAN = sc.MIN_BOARD_SPAN_FRACTION
FRAMES = 30
REPETICOES = 3

#: (rotulo, escala, clip do CLAHE, flags do SB).
#: Em 1280 de largura: 0.75 = 960 px, 0.5 = 640 px, 0.33 = 427 px.
#: O DEFAULT do codigo e' a cascata (427, 960, 640) -- ver DETECTION_WIDTHS.
CONFIGS = (
    ("SB EXH+ACC  full (ANTIGO)", 1.0, 0.0,
     cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY),
    ("SB EXH+ACC  full +CLAHE", 1.0, 2.0,
     cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY),
    ("SB EXH+ACC  960 px", 0.75, 2.0,
     cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY),
    ("SB EXH      960 px +CLAHE", 0.75, 2.0, cv2.CALIB_CB_EXHAUSTIVE),
    ("SB EXH      960 px sem CLAHE", 0.75, 0.0, cv2.CALIB_CB_EXHAUSTIVE),
    ("SB EXH+ACC  640 px", 0.5, 2.0,
     cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY),
    ("SB EXH      640 px +CLAHE (PADRAO 2o)", 0.5, 2.0,
     cv2.CALIB_CB_EXHAUSTIVE),
    ("SB EXH      640 px sem CLAHE", 0.5, 0.0, cv2.CALIB_CB_EXHAUSTIVE),
    ("SB EXH      427 px +CLAHE (PADRAO 1o)", 1.0 / 3.0, 2.0,
     cv2.CALIB_CB_EXHAUSTIVE),
)


def prepara(frame, escala, clahe_clip):
    """Cinza como o detector ve: reduzido (escala) e com CLAHE."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if escala != 1.0:
        gray = cv2.resize(gray, None, fx=escala, fy=escala,
                          interpolation=cv2.INTER_AREA)
    if clahe_clip > 0:
        gray = cv2.createCLAHE(clipLimit=clahe_clip,
                               tileGridSize=(8, 8)).apply(gray)
    return gray


def mede(gray, flags, escala_deteccao):
    """(mediana_ms, achou, span_px, cantos_na_escala_do_frame).

    ATENCAO ao parametro: `escala_deteccao` e' o FATOR DE REDUCAO usado em
    `prepara` (0.5 = metade), NAO o fator de volta. Os cantos sao DIVIDIDOS por
    ele para voltarem a' escala do frame original.
    """
    tempos = []
    achou = False
    cantos = None
    for _ in range(REPETICOES):
        inicio = time.perf_counter()
        ok, encontrados = cv2.findChessboardCornersSB(gray, CHECKERBOARD_SIZE,
                                                      flags)
        tempos.append((time.perf_counter() - inicio) * 1000.0)
        if ok:
            achou = True
            cantos = encontrados
    span = None
    if achou:
        cantos = cantos.reshape(-1, 2) / escala_deteccao
        diffs = cantos[:, None, :] - cantos[None, :, :]
        span = float(np.max(np.linalg.norm(diffs, axis=2)))
    return float(np.median(tempos)), achou, span, cantos


def margens(cantos, forma):
    """Margens do tabuleiro na imagem (px) + lado do quadrado (px).

    E' o diagnostico de ENQUADRAMENTO. O detector precisa do padrao INTEIRO, e
    a fileira de quadrados que fica FORA dos cantos internos ainda ocupa ~1
    lado de quadrado em cada borda -- logo margem abaixo disso significa que o
    tabuleiro esta' CORTADO pela imagem e a deteccao falha.
    """
    altura, largura = forma[:2]
    c = np.asarray(cantos, np.float64).reshape(-1, 2)
    grade = c.reshape(CHECKERBOARD_SIZE[1], CHECKERBOARD_SIZE[0], 2)
    passo = float(np.linalg.norm(np.diff(grade, axis=1), axis=-1).mean())
    return {
        "passo": passo,
        "esquerda": float(c[:, 0].min()),
        "direita": largura - 1 - float(c[:, 0].max()),
        "topo": float(c[:, 1].min()),
        "base": altura - 1 - float(c[:, 1].max()),
    }


def captura(fonte, idx, largura, altura, espera):
    """(frames, segundos) -- mede tambem a taxa de ENTREGA da propria camera.

    A PRIMEIRA leitura e' lenta por natureza: o MSMF leva 20-40 s para abrir a
    webcam e o Kinect leva alguns segundos para comecar a transmitir. Por isso a
    espera e' generosa e o progresso e' impresso (senao parece travado).
    """
    frames = []
    if fonte == "kinect":
        from pykinect2 import PyKinectV2
        from pykinect2.PyKinectRuntime import PyKinectRuntime
        runtime = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        desc = runtime.color_frame_desc
        inicio = aviso = time.time()
        while len(frames) < FRAMES and time.time() - inicio < espera:
            if runtime.has_new_color_frame():
                bruto = runtime.get_last_color_frame()
                frames.append(cv2.cvtColor(
                    bruto.reshape((desc.Height, desc.Width, 4)),
                    cv2.COLOR_BGRA2BGR))
            else:
                time.sleep(0.002)
                if time.time() - aviso > 3.0:
                    print(f"  ... aguardando o Kinect ha' "
                          f"{time.time() - inicio:.0f} s")
                    aviso = time.time()
        runtime.close()
        return frames, time.time() - inicio

    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    inicio = aviso = time.time()
    while len(frames) < FRAMES and time.time() - inicio < espera:
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
        elif time.time() - aviso > 3.0:
            print(f"  ... webcam {idx} ainda nao entregou "
                  f"({time.time() - inicio:.0f} s)")
            aviso = time.time()
    cap.release()
    return frames, time.time() - inicio


def contraste(gray):
    """Estatisticas que denunciam impressao fraca / luz desigual."""
    baixo, alto = np.percentile(gray, (5.0, 95.0))
    return {
        "media": float(gray.mean()),
        "desvio": float(gray.std()),
        "faixa": float(alto - baixo),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fonte", choices=("kinect", "aux"), default="kinect",
                        help="Camera a medir.")
    parser.add_argument("--idx", type=int, default=1,
                        help="Indice da webcam auxiliar (com --fonte aux).")
    parser.add_argument("--largura", type=int, default=1280)
    parser.add_argument("--altura", type=int, default=720)
    parser.add_argument("--salvar", action="store_true",
                        help="Grava o cinza COMO O DETECTOR O VE.")
    parser.add_argument("--espera", type=float, default=60.0,
                        help="Segundos maximos aguardando os frames (a primeira"
                             " leitura da webcam pode levar 20-40 s).")
    return parser.parse_args()


def main():
    args = parse_args()
    total = CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1]
    print(f"Tabuleiro esperado: {CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]} "
          f"cantos internos ({total})")
    print(f"Capturando ate' {FRAMES} frames de '{args.fonte}'...")
    frames, segundos = captura(args.fonte, args.idx, args.largura,
                               args.altura, args.espera)
    if not frames:
        print("NENHUM frame recebido: camera ocupada, desconectada ou indice "
              "errado.")
        return 1
    altura, largura = frames[0].shape[:2]
    diagonal = float(np.hypot(largura, altura))
    fps = len(frames) / max(segundos, 1e-9)
    print(f"Frames: {len(frames)} em {segundos:.2f} s -> {fps:.1f} FPS "
          f"({largura}x{altura})")
    if fps < 12.0:
        print("  AVISO: a camera ja' entrega poucos FPS sozinha -- o travamento"
              " nao e' so' do detector.")

    amostra = frames[len(frames) // 2]
    info = contraste(cv2.cvtColor(amostra, cv2.COLOR_BGR2GRAY))
    print(f"Contraste do cinza: media {info['media']:.0f} | desvio "
          f"{info['desvio']:.1f} | faixa p5-p95 {info['faixa']:.0f}")
    if info["faixa"] < 120.0:
        print("  AVISO: faixa de cinza baixa -> impressao/luz fracas; o CLAHE"
              " e' o remedio (--clahe 3 ou 4).")

    print(f"\n{'configuracao':<28}{'ms':>9}{'achou':>7}{'span':>9}")
    print("-" * 53)
    achados = []
    for rotulo, escala, clahe_clip, flags in CONFIGS:
        gray = prepara(amostra, escala, clahe_clip)
        ms, achou, span, cantos = mede(gray, flags, escala)
        fracao = f"{span / diagonal:.1%}" if span else "-"
        print(f"{rotulo:<28}{ms:>9.1f}{'SIM' if achou else 'nao':>7}{fracao:>9}")
        if achou:
            achados.append((rotulo, ms, span / diagonal, cantos))

    if args.salvar:
        # O frame BRUTO (cor) mostra o que a tabela nao mostra: enquadramento,
        # luz/reflexo e borrao. E' o que permite dar diagnostico a distancia.
        cv2.imwrite("tabuleiro_frame_bruto.png", amostra)
        print("  gravado tabuleiro_frame_bruto.png (cor, como a camera ve)")
        for posicao, (rotulo, escala, clahe_clip, _) in enumerate(CONFIGS[:4], 1):
            nome = (f"tabuleiro_cinza_{posicao}_e{escala:.2f}"
                    f"_c{clahe_clip:.0f}.png")
            cv2.imwrite(nome, prepara(amostra, escala, clahe_clip))
            print(f"  gravado {nome}")

    print("\n=== COMO LER ===")
    print("  ms perto ou acima de 100 -> o DETECTOR e' o gargalo do video")
    print("     travado: baixe a --largura-deteccao ou use --rapido.")
    print("  'achou=nao' sem CLAHE e 'SIM' com CLAHE -> impressao de baixo")
    print("     contraste: suba o --clahe (2 -> 4) e/ou melhore a luz.")
    print("  A MESMA configuracao acerta numa rodada e falha na outra (640 e")
    print("     960 ja' fizeram isso) -> o detector esta' na FRONTEIRA por")
    print("     causa da imagem: pouca luz = exposicao longa = borrao de")
    print("     movimento + ruido de ISO. NAO adianta trocar de largura:")
    print("     acenda uma luz DIFUSA no tabuleiro e prenda-o numa base rigida.")
    print("  span abaixo de %.0f%% -> APROXIME o tabuleiro da camera."
          % (100 * MIN_SPAN))
    if achados:
        melhor = max(achados, key=lambda a: (a[2], -a[1]))
        print(f"\n  Melhor combinacao medida: {melhor[0]} "
              f"({melhor[1]:.1f} ms, span {melhor[2]:.0%})")
        # ENQUADRAMENTO: a causa numero 1 de "achou=nao" e' o tabuleiro estar
        # CORTADO pela imagem. Isto mostra exatamente qual borda esta' curta.
        m = margens(melhor[3], amostra.shape)
        print(f"\n=== ENQUADRAMENTO ({melhor[0]}) ===")
        print(f"  lado do quadrado: {m['passo']:.1f} px")
        apertadas = []
        for nome in ("esquerda", "direita", "topo", "base"):
            quadrados = m[nome] / m["passo"]
            if quadrados < 1.0:
                apertadas.append(nome)
            print(f"  {nome:<9} {m[nome]:6.0f} px = {quadrados:4.1f} quadrados  "
                  f"{'RISCO' if quadrados < 1.0 else 'ok'}")
        if apertadas:
            print(f"  AVISO: margem curta em {', '.join(apertadas)}. O detector")
            print("  precisa do padrao INTEIRO: a fileira de quadrados que fica")
            print("  fora dos cantos internos ainda ocupa ~1 quadrado em cada")
            print("  borda. AFASTE ou CENTRALIZE o tabuleiro.")
        else:
            print("  ok: o tabuleiro esta' inteiro na imagem, com folga.")
    else:
        print("\n  NENHUMA combinacao achou o tabuleiro. Confira, NESTA ORDEM:")
        print("   1. o tabuleiro esta' INTEIRO na imagem? As 4 bordas tem de")
        print("      aparecer -- inclusive a ultima fileira de quadrados;")
        print("   2. a mao nao esta' NA FRENTE de nenhum quadrado da borda?")
        print("      segure pelas LATERAIS, com os dedos ATRAS da placa;")
        print(f"   3. o tamanho esperado e' "
              f"{CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]} cantos internos;")
        print("   4. rode com --salvar: grava tabuleiro_frame_bruto.png, que")
        print("      mostra enquadramento, luz/reflexo e borrao.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
