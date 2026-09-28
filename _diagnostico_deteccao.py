# -*- coding: utf-8 -*-
"""Diagnostico de deteccao do tabuleiro SEM depender das cameras de pe.

Roda o MESMO pipeline de stereo_calibration.py sobre imagens salvas em disco
(os batimentos _hb_*.png, um crop salvo, uma foto do celular...) e varre
todas as combinacoes de escala x metodo x CLAHE, mostrando exatamente qual
combinacao detecta (ou provando que nenhuma detecta -- o que aponta para
padrao errado, tabuleiro fora de escala ou imagem problematica, e nao para
as cameras).

Uso tipico:
    python _diagnostico_deteccao.py
    python _diagnostico_deteccao.py --imagens _hb_kinect.png _hb_aux.png
    python _diagnostico_deteccao.py --checkerboard 5x3 --quadrado 57
    python _diagnostico_deteccao.py --clahe 0

Regras do padrao: o cv2 conta CANTOS internos, nao quadrados. O tabuleiro atual
(23/09/2026) tem 6x4 QUADRADOS de 57 mm -> 5x3 cantos. Os padroes de
--checkerboard/--quadrado ja' vem de stereo_calibration.py; confirme contando os
quadrados na imagem.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

import stereo_calibration as sc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnostico de deteccao do tabuleiro em imagens salvas.")
    parser.add_argument("--imagens", type=str, nargs="+",
                        default=["_hb_kinect.png", "_hb_aux.png"],
                        help="Imagens a testar (padrao: _hb_kinect.png "
                             "_hb_aux.png).")
    parser.add_argument("--checkerboard", type=str,
                        default="%dx%d" % sc.CHECKERBOARD_SIZE,
                        metavar="CxR",
                        help="Cantos internos 'COLSxROWS' (5x3 = 6x4 "
                             "quadrados). Padrao: o de stereo_calibration.py.")
    parser.add_argument("--quadrado", type=float,
                        default=sc.SQUARE_SIZE_M * 1000.0, metavar="MM",
                        help="Lado do quadrado em mm (so' informativo aqui; "
                             "afeta a geometria 3D, nao a deteccao). "
                             "Padrao: o de stereo_calibration.py.")
    parser.add_argument("--clahe", type=float, default=None, metavar="X",
                        help="Forca o CLAHE do pipeline (sobrescreve "
                             "DETECTION_CLAHE_CLIP do modulo). Sem --clahe, "
                             "testa com o valor do modulo E com 0.")
    parser.add_argument("--sintetico", action="store_true",
                        help="Gera um tabuleiro SINTETICO perfeito com o "
                             "padrao pedido e o inclui no teste. Se nem ele "
                             "detectar, o problema e' do script/pipeline; se "
                             "detectar, o padrao esta' certo e o problema e' "
                             "da imagem real. Autoteste do diagnostico.")
    return parser.parse_args()


def gera_sintetico(cols, rows, lado_px=80):
    """Tabuleiro (cols+1)x(rows+1) quadrados perfeito, branco em volta.

    Devolve o caminho do PNG salvo. Serve de controle: se o detector falhar
    AQUI, o erro esta' no codigo/padrao; imagens reais nunca sao mais faceis.
    """
    quadrados_x, quadrados_y = cols + 1, rows + 1
    margem = lado_px
    img = np.full((rows * lado_px + 2 * margem,
                   cols * lado_px + 2 * margem), 255, np.uint8)
    for y in range(quadrados_y):
        for x in range(quadrados_x):
            if (x + y) % 2 == 0:
                x0, y0 = margem + x * lado_px, margem + y * lado_px
                cv2.rectangle(img, (x0, y0),
                              (x0 + lado_px, y0 + lado_px), 0, -1)
    caminho = f"_teste_sintetico_{cols}x{rows}.png"
    cv2.imwrite(caminho, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return caminho


def aplica_config(args):
    """Espelha as constantes do modulo com o pedido na linha de comando."""
    texto = args.checkerboard.lower().replace(" ", "")
    for separador in ("x", ",", "*"):
        if separador in texto:
            cols_txt, rows_txt = texto.split(separador, 1)
            break
    else:
        print(f"ERRO: --checkerboard '{args.checkerboard}' precisa ser "
              "COLSxROWS (ex.: 11x7).")
        sys.exit(2)
    try:
        cols, rows = int(cols_txt), int(rows_txt)
    except ValueError:
        print(f"ERRO: --checkerboard '{args.checkerboard}' precisa ser "
              "COLSxROWS (ex.: 11x7).")
        sys.exit(2)
    if cols < 2 or rows < 2:
        print(f"ERRO: padrao {cols}x{rows} pequeno demais (minimo 2x2).")
        sys.exit(2)
    sc.CHECKERBOARD_COLS = cols
    sc.CHECKERBOARD_ROWS = rows
    sc.CHECKERBOARD_SIZE = (cols, rows)
    sc.SQUARE_SIZE_M = args.quadrado / 1000.0
    if args.clahe is not None:
        sc.DETECTION_CLAHE_CLIP = args.clahe
    clahe_modulo = sc.DETECTION_CLAHE_CLIP
    # Variantes de CLAHE a varrer: o valor vigente do modulo e (se diferente)
    # tambem sem CLAHE, para separar "falha por contraste" de "falha estrutural".
    variantes_clahe = [clahe_modulo]
    if clahe_modulo != 0.0:
        variantes_clahe.append(0.0)
    return cols, rows, variantes_clahe


def prepara_gray_local(frame, largura, clip):
    """Mesma preparacao de prepare_gray, com CLAHE parametrizavel."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    escala = 1.0
    if largura and gray.shape[1] > largura:
        fator = largura / float(gray.shape[1])
        gray = cv2.resize(gray, None, fx=fator, fy=fator,
                          interpolation=cv2.INTER_AREA)
        escala = 1.0 / fator
    if clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray, escala


def estatisticas(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brilho = float(gray.mean())
    nitidez = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brilho, nitidez


def px_por_quadrado(cantos, cols, rows):
    """Escala media do tabuleiro na imagem (px por lado de quadrado)."""
    diagonal = np.linalg.norm(cantos[-1] - cantos[0])
    return float(diagonal / np.hypot(cols - 1, rows - 1))


def salva_overlay(frame, cantos, nome_arquivo):
    overlay = frame.copy()
    cv2.drawChessboardCorners(overlay, sc.CHECKERBOARD_SIZE,
                              cantos.reshape(-1, 1, 2).astype(np.float32),
                              True)
    cv2.imwrite(nome_arquivo, overlay)
    print(f"        overlay salvo em {nome_arquivo}")


def tenta_producao(frame):
    """O caminho exato que a calibracao usa (cascata SB + fallback classico)."""
    inicio = time.perf_counter()
    try:
        cantos = sc.find_corners(frame.copy())
    finally:
        # find_corners guarda estado global (_ultima_largura_ok); restaura
        # para o teste nao enviesar as proximas imagens.
        sc._ultima_largura_ok = None
    return cantos, (time.perf_counter() - inicio) * 1000.0


def tenta_manual(gray, metodo):
    """Um metodo isolado numa imagem ja preparada. Devolve (found, cantos)."""
    if metodo == "SB-EXH":
        return cv2.findChessboardCornersSB(
            gray, sc.CHECKERBOARD_SIZE, sc.DETECTION_SB_FLAGS)
    if metodo == "SB":
        return cv2.findChessboardCornersSB(gray, sc.CHECKERBOARD_SIZE, 0)
    return cv2.findChessboardCorners(
        gray, sc.CHECKERBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)


def diagnostica_imagem(caminho, cols, rows, variantes_clahe):
    print(f"\n=== {caminho} ===")
    frame = cv2.imread(caminho)
    if frame is None:
        print("  ERRO: nao consegui ler a imagem.")
        return False
    brilho, nitidez = estatisticas(frame)
    print(f"  {frame.shape[1]}x{frame.shape[0]} px | brilho medio "
          f"{brilho:.0f}/255 | nitidez (var. Laplace) {nitidez:.0f}")
    if nitidez < 50.0:
        print("  ATENCAO: imagem possivelmente fora de foco (nitidez < 50).")
    stem = os.path.splitext(os.path.basename(caminho))[0]
    algum_ok = False

    cantos, ms = tenta_producao(frame)
    ok = cantos is not None
    algum_ok |= ok
    print(f"  [{'OK' if ok else '..'}] pipeline de producao (cascata SB + "
          f"fallback): {len(cantos) if ok else 'falhou'} cantos ({ms:.0f} ms)")
    if ok:
        escala = px_por_quadrado(cantos, cols, rows)
        print(f"        ~{escala:.1f} px/quadrado -- se ficar < ~12 px o "
              "detector pode precisar de largura maior; se > ~150 px, de menor.")
        salva_overlay(frame, cantos, f"_diag_{stem}_producao.png")

    larguras = (960, 640, 427, 0)
    for clip in variantes_clahe:
        rotulo_clahe = f"CLAHE {clip:g}" if clip > 0 else "sem CLAHE"
        for largura in larguras:
            gray, _ = prepara_gray_local(frame, largura, clip)
            for metodo in ("SB-EXH", "SB", "classico"):
                inicio = time.perf_counter()
                found, corners = tenta_manual(gray, metodo)
                ms = (time.perf_counter() - inicio) * 1000.0
                tamanho = f"{largura}px" if largura else "cheia"
                if found:
                    algum_ok = True
                    cantos = corners.reshape(-1, 2).astype(np.float64)
                    print(f"  [OK] {metodo:8s} {tamanho:6s} {rotulo_clahe:10s}"
                          f" -> {len(cantos)} cantos ({ms:.0f} ms)")
                    salva_overlay(
                        frame, cantos,
                        f"_diag_{stem}_{metodo}_{tamanho}_c{clip:g}.png")
                else:
                    print(f"  [..] {metodo:8s} {tamanho:6s} {rotulo_clahe:10s}"
                          f" -> falhou ({ms:.0f} ms)")
    return algum_ok


def main():
    args = parse_args()
    cols, rows, variantes_clahe = aplica_config(args)
    print(f"Diagnostico com padrao {cols}x{rows} cantos internos "
          f"(= {cols + 1}x{rows + 1} quadrados), quadrado de "
          f"{sc.SQUARE_SIZE_M * 1000:g} mm")
    imagens = list(args.imagens)
    if args.sintetico:
        imagens.insert(0, gera_sintetico(cols, rows))
    algum_ok = False
    for caminho in imagens:
        algum_ok |= diagnostica_imagem(caminho, cols, rows, variantes_clahe)
    print("\n=== RESUMO ===")
    if algum_ok:
        print("DETECTOU em pelo menos uma combinacao: o padrao e' VIAVEL. "
              "Veja qual combinacao passou e por que o pipeline de producao "
              "(ou a camera ao vivo) nao esta' passando nela.")
        return 0
    print("NENHUMA combinacao detectou. Nesta ordem, verifique:")
    print("  1) Padrao certo? conte QUADRADOS e subtraia 1 de cada lado "
          "(12x8 quadrados -> 11x7 cantos).")
    print("  2) O tabuleiro aparece INTEIRO na imagem? borda cortada = "
          "falha sempre.")
    print("  3) Escala: veja px/quadrado quando detecta (~12-150 px por "
          "lado funciona).")
    print("  4) Impressao fraca/ruidosa: aumente --clahe (2-4) ou reimprima "
          "com preto mais denso.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
