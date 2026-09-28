"""Descobre o TAMANHO do padrao do tabuleiro por MEDICAO (nao por contagem).

Por que existe
--------------
`findChessboardCornersSB` exige o numero EXATO de CANTOS INTERNOS. Se o valor
nao casar com a placa, a deteccao devolve False -- sem erro e sem aviso -- e a
calibracao fica impossivel (o RMS nunca aparece, os 15 samples nunca fecham).
Tabuleiros impressos quase sempre tem quadrados CORTADOS na borda (para o
contorno fechar em preto/branco), entao "contar quadrados" nao diz sozinho
quantos cantos o detector enxerga:

    N quadrados por eixo   ->   (N - 1) cantos internos por eixo
    a coluna/linha cortada na borda PODE gerar cantos extras (ou nao)

Como usar
---------
    .venv\\Scripts\\python.exe tools\\probe_tabuleiro.py
    .venv\\Scripts\\python.exe tools\\probe_tabuleiro.py --fonte webcam --idx 1
    .venv\\Scripts\\python.exe tools\\probe_tabuleiro.py --auto-teste

Segure o tabuleiro de frente para a camera, INTEIRO dentro da imagem e ocupando
boa parte dela (>= 30% da largura). O script testa sozinho, a cada frame, todos
os tamanhos candidatos e imprime o que casou -- nao precisa apertar nada.
ESC ou Q encerra.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: (colunas, linhas) de CANTOS INTERNOS, do mais provavel para o menos.
#: PLACA ATUAL (impressa 23/09/2026, medida com regua): 6 x 4 QUADRADOS de 57 mm
#: -> 5 x 3 cantos internos. As placas antigas ficam como controle: se o detector
#: casar com uma delas, o tabuleiro na frente da camera NAO e' o novo.
CANDIDATOS = (
    (5, 3),    # PLACA ATUAL: 6x4 quadrados de 57 mm
    (6, 3),    # se a placa tiver uma coluna completa a mais
    (5, 4),    # se tiver uma linha completa a mais
    (3, 5),    # a MESMA placa girada 90 graus
    (11, 7),   # placa ANTIGA: 12x8 quadrados, 26 mm (2 colunas cortadas)
    (10, 7),   # a antiga, se as colunas cortadas nao contassem
    (8, 6),    # placa LEGADA: 9x7 quadrados, 25 mm -- controle
    (5, 7),    # 6x8
    (5, 5),    # 6x6
)
LARGURA_QUADRO_MM = 57.0
#: Piso de tamanho util: abaixo disso a deteccao e' instavel (poucos px/canto).
FRACAO_MINIMA = 0.15


def detecta(gray, size):
    """Tenta SB (rapido/subpixel) e depois o classico. -> (cantos|None, nome)."""
    if hasattr(cv2, "findChessboardCornersSB"):
        found, cantos = cv2.findChessboardCornersSB(
            gray, size, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if found:
            return cantos.reshape(-1, 2).astype(np.float64), "SB"
    found, cantos = cv2.findChessboardCorners(
        gray, size,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None, None
    cantos = cv2.cornerSubPix(
        gray, cantos, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    return cantos.reshape(-1, 2).astype(np.float64), "classico"


def mede(cantos, size):
    """Metricas da grade detectada.

    `razao_passo` e' o teste de sanidade mais forte disponivel sem saber a
    distancia: para quadrados QUADRADOS ela tem de ficar ~1.0. Se um tamanho
    errado "casar" por acaso, ela costuma sair bem fora de 1.0.
    """
    cols, rows = int(size[0]), int(size[1])
    grade = cantos.reshape(rows, cols, 2)
    passo_x = float(np.linalg.norm(np.diff(grade, axis=1), axis=-1).mean())
    passo_y = float(np.linalg.norm(np.diff(grade, axis=0), axis=-1).mean())
    span_x = float(np.linalg.norm(grade[0, -1] - grade[0, 0]))
    span_y = float(np.linalg.norm(grade[-1, 0] - grade[0, 0]))
    return {
        "passo_x": passo_x,
        "passo_y": passo_y,
        "span_x": span_x,
        "span_y": span_y,
        "razao_passo": passo_x / max(passo_y, 1e-9),
    }


def reporta(achados, forma_frame, largura_mm):
    """Imprime o que casou e recomenda os dois valores para o codigo."""
    altura, largura = forma_frame[:2]
    diagonal = float(np.hypot(largura, altura))
    print("\n--- %d tamanho(s) detectado(s) ---" % len(achados))
    for size, cantos, detector in achados:
        m = mede(cantos, size)
        fracao = float(np.hypot(m["span_x"], m["span_y"])) / diagonal
        quadrados = (size[0] + 1, size[1] + 1)
        mm_x = quadrados[0] * largura_mm
        mm_y = quadrados[1] * largura_mm
        print("  %2dx%-2d cantos internos  (%dx%d quadrados, %.0fx%.0f mm)  [%s]"
              % (size[0], size[1], quadrados[0], quadrados[1], mm_x, mm_y,
                 detector))
        print("      passo x/y: %6.2f / %6.2f px  (razao %.3f -> %s)"
              % (m["passo_x"], m["passo_y"], m["razao_passo"],
                 "OK quadrados quadrados" if abs(m["razao_passo"] - 1.0) < 0.06
                 else "SUSPEITO (quadrados nao quadrados?)"))
        print("      ocupacao da imagem: %.0f%%  %s"
              % (100.0 * fracao,
                 "" if fracao >= FRACAO_MINIMA
                 else "(APROXIME O TABULEIRO)"))
        px_por_mm = 0.5 * (m["passo_x"] + m["passo_y"]) / largura_mm
        print("      escala: %.3f px/mm -> a placa tem ~%.0f px de largura"
              % (px_por_mm, px_por_mm * mm_x))
    #: O maior padrao detectado e' o mais util (mais pontos = melhor solver).
    #: Em empate de area -- a MESMA placa vista deitada vs em pe' -- fica a que
    #: tem quadrados mais "quadrados" (razao de passo mais perto de 1).
    melhor = max(achados,
                 key=lambda a: (a[0][0] * a[0][1],
                                -abs(mede(a[1], a[0])["razao_passo"] - 1.0)))
    print("\n  ==> USE:  CHECKERBOARD_COLS = %d" % melhor[0][0])
    print("            CHECKERBOARD_ROWS = %d" % melhor[0][1])
    print("            SQUARE_SIZE_M      = %.3f" % (largura_mm / 1000.0))
    print("  (mantenha o tabuleiro NESTA orientacao na calibracao: o detector")
    print("   nao aceita a placa girada 90 graus entre as capturas)")
    return melhor


def renderiza_tabuleiro(quadrados_x, quadrados_y, lado_px=40, margem_px=70):
    """Tabuleiro sintetico (margem branca = quiet zone do detector)."""
    largura = quadrados_x * lado_px + 2 * margem_px
    altura = quadrados_y * lado_px + 2 * margem_px
    img = np.full((altura, largura), 255, np.uint8)
    for j in range(quadrados_y):
        for i in range(quadrados_x):
            if (i + j) % 2 == 0:
                y0 = margem_px + j * lado_px
                x0 = margem_px + i * lado_px
                img[y0:y0 + lado_px, x0:x0 + lado_px] = 0
    return img


def auto_teste():
    """Valida a deteccao sem hardware: tabuleiros sinteticos de tamanho sabido."""
    print("=== AUTO-TESTE (tabuleiros sinteticos) ===")
    falhas = 0
    for quadrados_x, quadrados_y in ((8, 10), (8, 12), (10, 12), (9, 7)):
        esperado = (quadrados_x - 1, quadrados_y - 1)
        img = renderiza_tabuleiro(quadrados_x, quadrados_y)
        achados = []
        for size in CANDIDATOS:
            cantos, detector = detecta(img, size)
            if cantos is not None:
                achados.append((size, cantos, detector))
        tamanhos = [a[0] for a in achados]
        ok = esperado in tamanhos
        falhas += 0 if ok else 1
        print("  %2d x %-2d quadrados -> esperado %-8s | detectados: %s  %s"
              % (quadrados_x, quadrados_y, "%dx%d" % esperado,
                 tamanhos or "nenhum", "OK" if ok else "FALHOU"))
    print("  resultado: %s" % ("TUDO OK" if falhas == 0
                               else "%d FALHA(S)" % falhas))
    return falhas


def renderiza_tabuleiro_cortado(quadrados_x, quadrados_y, fracao_corte=0.5,
                                lado_px=40, margem_px=70):
    """Simula a placa REAL: grade completa + colunas CORTADAS nas duas bordas.

    `quadrados_x` conta TODAS as colunas, inclusive as cortadas: as colunas 0 e
    `quadrados_x - 1` sao desenhadas com `fracao_corte` da largura (o resto da
    celula foi cortado pela borda do papel).
    """
    largura = quadrados_x * lado_px + 2 * margem_px
    altura = quadrados_y * lado_px + 2 * margem_px
    img = np.full((altura, largura), 255, np.uint8)
    corte = max(1, int(round(fracao_corte * lado_px)))
    for j in range(quadrados_y):
        for i in range(quadrados_x):
            if (i + j) % 2:
                continue
            largura_q = lado_px
            x0 = margem_px + i * lado_px
            if i == 0:                      # sobrou o lado INTERNO da celula
                x0 += lado_px - corte
                largura_q = corte
            elif i == quadrados_x - 1:      # idem, do outro lado
                largura_q = corte
            y0 = margem_px + j * lado_px
            img[y0:y0 + lado_px, x0:x0 + largura_q] = 0
    return img


def simula_cortado():
    """Simula a placa real: 8x10 COMPLETOS + 2 colunas cortadas = 8x12.

    Pergunta que respondeu: as colunas cortadas geram CANTOS EXTRAS para o
    detector? Resposta CONFIRMADA na bancada: SIM -- a placa da' 11 x 7.
    """
    print("=== SIMULACAO: placa com 2 colunas CORTADAS nas bordas ===")
    for cols, rows in ((12, 8), (8, 12)):
        for fracao in (0.5, 0.3, 0.15):
            img = renderiza_tabuleiro_cortado(cols, rows, fracao)
            achados = [s for s in CANDIDATOS if detecta(img, s)[0] is not None]
            print("  %2dx%-2d quadrados (2 cortadas), corte %.0f%% -> %s"
                  % (cols, rows, 100 * fracao, achados or "nenhum"))
    print("\n  Leitura: aparece o tamanho da grade CHEIA (11x7 / 7x11), o que")
    print("  confirma que as colunas cortadas CONTAM -- e' o valor do codigo.")
    return 0


def abre_fonte(args):
    """Devolve (funcao_que_le_um_frame, descricao)."""
    if args.fonte == "kinect":
        from pykinect2 import PyKinectV2
        from pykinect2.PyKinectRuntime import PyKinectRuntime
        rt = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        desc = rt.color_frame_desc

        def le():
            if not rt.has_new_color_frame():
                return None
            bruto = rt.get_last_color_frame()
            return cv2.cvtColor(
                bruto.reshape((desc.Height, desc.Width, 4)),
                cv2.COLOR_BGRA2BGR)

        return le, "Kinect RGB %dx%d" % (desc.Width, desc.Height)

    cap = cv2.VideoCapture(args.idx, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.altura)

    def le():
        ok, frame = cap.read()
        return frame if ok else None

    return le, "webcam idx %d" % args.idx


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fonte", choices=("kinect", "webcam"),
                        default="kinect", help="Camera a usar.")
    parser.add_argument("--idx", type=int, default=0,
                        help="Indice da webcam (so' com --fonte webcam).")
    parser.add_argument("--largura", type=int, default=1280)
    parser.add_argument("--altura", type=int, default=720)
    parser.add_argument("--lado-mm", type=float, default=LARGURA_QUADRO_MM,
                        help="Lado do quadrado em mm (medido com regua).")
    parser.add_argument("--espelho", action="store_true",
                        help="Espelha o frame (a auxiliar chega espelhada).")
    parser.add_argument("--salvar", action="store_true",
                        help="Salva o frame bom em probe_tabuleiro.png")
    parser.add_argument("--auto-teste", action="store_true",
                        help="Valida a deteccao com tabuleiros sinteticos e sai.")
    parser.add_argument("--simula-cortado", action="store_true",
                        help="Simula a placa com colunas cortadas nas bordas.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.auto_teste:
        return auto_teste()
    if args.simula_cortado:
        return simula_cortado()

    le, descricao = abre_fonte(args)
    print("Fonte: %s" % descricao)
    print("Testando %d tamanhos: %s"
          % (len(CANDIDATOS), ", ".join("%dx%d" % c for c in CANDIDATOS)))
    print("Segure o tabuleiro de frente, inteiro e grande na imagem.")
    print("ESC/Q encerra. Ctrl+C tambem.\n")

    ultima = None
    salvou = False
    try:
        while True:
            frame = le()
            if frame is None:
                time.sleep(0.02)
                continue
            if args.espelho:
                frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            achados = []
            for size in CANDIDATOS:
                cantos, detector = detecta(gray, size)
                if cantos is not None:
                    achados.append((size, cantos, detector))

            #: So' reporta quando o conjunto de tamanhos achados MUDA, para nao
            #: inundar o console a cada frame.
            assinatura = tuple(a[0] for a in achados)
            if achados and assinatura != ultima:
                melhor = reporta(achados, frame.shape, args.lado_mm)
                ultima = assinatura
                if args.salvar and not salvou:
                    vis = frame.copy()
                    cv2.drawChessboardCorners(
                        vis, melhor[0], melhor[1].reshape(-1, 1, 2), True)
                    cv2.imwrite("probe_tabuleiro.png", vis)
                    print("      frame salvo em probe_tabuleiro.png")
                    salvou = True
            elif not achados:
                ultima = None

            vis = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.putText(vis, "achados: %s" % (assinatura or "nenhum"),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if achados else (0, 0, 255), 2)
            cv2.imshow("probe_tabuleiro (ESC sai)", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
