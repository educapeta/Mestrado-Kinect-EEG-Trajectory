"""Congela o modelo do Kinect RGB com o arranjo KINECT-ONLY (sem a auxiliar).

Recuperado do transcript da sessao do Cline: leitura numerada de 710 linhas
(todas as funcoes puras) + o LOOP de `coleta_poses` e o `main()`, que a leitura
cortou. O modulo em disco tinha 429 linhas e nao casava com
`test_freeze_kinect_direto.py` (846 linhas, 24 nomes exigidos, 5 achados).
"""
import argparse
import sys
import threading
import time

import cv2
import numpy as np

import stereo_calibration as sc

#: Nome da janela unica: o arranjo so' tem uma camera.
JANELA = "Freeze Kinect - so' o tabuleiro no Kinect"
#: Deslocamento MEDIO dos cantos (px, resolucao cheia) entre duas deteccoes
#: seguidas acima do qual o tabuleiro e' considerado EM MOVIMENTO e o tempo de
#: estabilidade recomeca.
#: Por que este portao existe AQUI e nao no caminho do par: la' o
#: `pair_planar_error` (homografia aux -> Kinect) reprova movimento, porque um
#: erro de par alto tem de ser descartado de qualquer forma. Aqui nao ha' par, e
#: sem este numero o cronometro de 2 s continuaria contando com o tabuleiro na
#: mao -- entrando amostras BORRADAS num modelo que vai ser congelado.
#: MEDIDO: com o tabuleiro parado e luz difusa a deteccao fica em ~0,7 px na
#: resolucao cheia (ver DETECTION_REFINO_CHEIO em stereo_calibration); 3 px deixa
#: folga para o tremor de quem segura sem admitir movimento de verdade.
MAX_DESLOCAMENTO_PX = 3.0
#: Teto de tempo total da captura quando `--sem-janela` (sem janela nao ha' ESC:
#: sem este teto um Kinect que nao ve' o tabuleiro deixaria o processo eterno).
LIMITE_SEM_JANELA_S = 600.0
#: fx NOMINAL do Kinect RGB 1080p (o mesmo da cena de teste de
#: test_intrinsecas_kinect.py, medido na bancada). So' serve para ESTIMAR a
#: distancia exibida na janela (pinhole): quando o modelo congelado ja' existir,
#: o fx DELE e' usado no lugar. Nao e' medicao -- e' o numero que poe o operador
#: nos 0,30-0,42 m antes de ele olhar o span (mire 0,35-0,40 m: a 0,45 m o
#: tabuleiro fica exatamente no portao de span).
FX_NOMINAL_PX = 1081.37
#: Escalas do DETECTOR neste arranjo (o tabuleiro ocupa 35-60% da diagonal, nunca
#: o caso de fronteira que a cascata existe para cobrir). O padrao do projeto
#: (`sc.DETECTION_WIDTHS`) comeca em 427 px, e o proprio `sc.find_corners` mediu o
#: preco: erro de canto de 0,67 px na escala CHEIA, 1,27 px a 640 px e 2,20 px a
#: 427 px -- este ultimo ja' acima do portao de RMS deste modelo. Com o tabuleiro
#: grande nao ha' motivo para detectar a 427 px: a bancada de 05/10/2026 mediu RMS
#: 1,56-1,77 px (portao 0,6 px) com a cascata padrao. 960 px primeiro e a escala
#: cheia como rede (o refinamento em resolucao cheia continua valendo para os
#: cantos com folga da borda -- ver `sc.refina_cantos`).
LARGURAS_DETECCAO_DIRETO = (960, 0)
#: Quantas poses as PIORES em erro de reprojecao o solver pode descartar para o
#: modelo passar no portao de RMS (0 desliga). O caminho do par ja' limpa
#: outliers; aqui (`sc.calibrate_single`) nao havia limpeza NENHUMA: UMA pose
#: borrada entre 15 boas levava o RMS do conjunto a ~1,5 px e a captura inteira
#: era perdida sem dizer qual pose era. O que sai e' impresso, e as poses que
#: ficam tem de passar os TRES portoes de novo (`_avalia`).
DESCARTES_MAX = 4
#: Cores (BGR) do overlay de poses na janela.
COR_POSE_SALVA = (0, 255, 0)           #: X de cada pose ja' aceita
COR_ANEL_SEPARACAO = (0, 165, 255)     #: anel de separacao minima (laranja)
COR_TABULEIRO_AGORA = (255, 255, 255)  #: centro do tabuleiro AGORA


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--saida", type=str,
                        default=str(sc.KINECT_INTRINSICS_FILE), metavar="ARQUIVO",
                        help="Onde gravar o modelo congelado. O padrao e' o "
                             "arquivo que TODA calibracao de par reusa; use "
                             "outro nome para ensaiar sem tocar no modelo bom "
                             "da bancada.")
    parser.add_argument("--checkerboard", type=str, default=None, metavar="CxR",
                        help="Cantos internos do tabuleiro, como 'CANTOSxCANTOS'"
                             " (7x5 = 8x6 quadrados). Padrao: as constantes de "
                             "stereo_calibration (%dx%d)."
                             % tuple(sc.CHECKERBOARD_SIZE))
    parser.add_argument("--quadrado", type=float, default=None, metavar="MM",
                        help="Tamanho do quadrado em mm. Padrao: a constante de "
                             "stereo_calibration (%g mm)."
                             % (sc.SQUARE_SIZE_M * 1000.0))
    parser.add_argument("--largura-deteccao", type=int, nargs="+",
                        default=list(LARGURAS_DETECCAO_DIRETO),
                        metavar="N",
                        help="Largura(s)-ALVO da deteccao, tentadas EM CASCATA "
                             "(0 = resolucao cheia). Padrao %s para ESTE "
                             "arranjo: o tabuleiro e' grande e o erro de canto a "
                             "427 px medido no projeto e' 2,20 px, acima do "
                             "portao de RMS." % (tuple(LARGURAS_DETECCAO_DIRETO),))
    parser.add_argument("--clahe", type=float, default=sc.DETECTION_CLAHE_CLIP,
                        help="Forca do CLAHE no cinza antes de detectar (0 "
                             "desliga).")
    parser.add_argument("--rapido", action="store_true",
                        help="Tira o CALIB_CB_EXHAUSTIVE do detector SB: mais "
                             "rapido, porem menos robusto com o tabuleiro "
                             "pequeno ou muito inclinado.")
    parser.add_argument("--deslocamento-minimo", type=float,
                        default=sc.MIN_POSE_SEPARATION, metavar="FRAC",
                        help="Separacao minima entre poses, em fracao do span do "
                             "tabuleiro (a mesma regra da calibracao de par).")
    parser.add_argument("--max-poses", type=int, default=sc.MAX_TOTAL_SAMPLES,
                        metavar="N", help="Teto de poses aceitas.")
    parser.add_argument("--max-descartes", type=int, default=DESCARTES_MAX,
                        metavar="N",
                        help="Quantas poses (as PIORES por erro de reprojecao) o "
                             "solver pode descartar para o modelo passar no "
                             "portao de RMS (0 desliga): o que saiu e' impresso "
                             "e as poses que ficam passam os tres portoes.")
    parser.add_argument("--sair-ao-gravar", action="store_true",
                        help="Sai assim que o modelo for gravado (implicito em "
                             "--sem-janela).")
    parser.add_argument("--sem-janela", action="store_true",
                        help="Nao abre janela nenhuma (SSH/CI): grava e sai "
                             "assim que os portoes passarem; sem ESC, vale o "
                             "teto de %g s." % LIMITE_SEM_JANELA_S)
    parser.add_argument("--limite-s", type=float, default=0.0, metavar="SEG",
                        help="Teto de TEMPO total da captura (0 = sem teto; com "
                             "janela o operador sai no ESC).")
    return parser.parse_args()


def print_guidance():
    """O que faz o span passar dos 35% e o RMS ficar abaixo de 0,6 px."""
    print("COMO CAPTURAR (arranjo Kinect-only -- aqui manda SO' o Kinect):")
    print("- CHEGUE PERTO, mas nao cole: mire 0,35-0,40 m (faixa util 0,30-0,42 m). "
          "A 0,45 m o tabuleiro fica EXATAMENTE no portao (%.1f%%) e qualquer pose "
          "um pouco mais longe derruba o mediano; a ~1 m (a distancia em que a "
          "auxiliar tambem ve' o tabuleiro) ele ocupa 11-13%% da diagonal e o "
          "portao recusa; COLADO (<0,28 m, acima de %.0f%%) o RGB sai de foco e o "
          "tabuleiro domina o ajuste de fx. E' por isso que este arranjo existe."
          % (sc.KINECT_INTRINSICS_MIN_SPAN * 100,
             sc.KINECT_INTRINSICS_MAX_SPAN * 100))
    print("- O PORTAO OLHA O SPAN *MEDIANO* DAS POSES, nao a melhor pose: TODAS "
          "contam. Uma pose a 0,50 m (span ~32%%) ou a 0,80 m (~20%%) DERRUBA o "
          "conjunto -- 12 poses boas nao salvam 6 poses longe (MEDIDO em "
          "05/10/2026: 18 poses, mediano 34%%, captura perdida). A 0,45 m o "
          "tabuleiro ja' fica no LIMITE (%.1f%%)."
          % (sc.KINECT_INTRINSICS_MIN_SPAN * 100))
    print("- QUADRO COBERTO: passe o tabuleiro pelas BORDAS e CANTOS da imagem, "
          "nas duas orientacoes. E' o que ensina a distorcao do Kinect -- sem "
          "isso k1/k2 saem do ajuste da regiao central.")
    print("- INCLINE 20-45 graus para os lados, para cima e para baixo, em poses "
          "diferentes; tabuleiro sempre de frente e plano deixa o foco "
          "impreciso.")
    print("- VARIE A DISTANCIA (0,30 a 0,42 m) entre as poses: e' a variacao de "
          "escala que separa fx/fy de tz -- sempre DENTRO da faixa. A cada pose "
          "aceita o console mostra o span MEDIANO e quantas poses cumprem a "
          "meta: e' esse par de numeros que decide se a captura grava.")
    print("- LUZ DIFUSA apontada para o tabuleiro (lampada com uma folha de "
          "papel na frente). Luz de teto cria sombra e reflexo; pouca luz sobe a "
          "exposicao e borra o quadro com o tremor da mao.")
    print("- FOCO/LUZ decidem o RMS (o portao deste modelo e' %.1f px). O erro de "
          "canto medido no projeto e' %.2f px na escala CHEIA contra %.2f px a "
          "427 px -- por isso este arranjo detecta a 960 px e cai para a escala "
          "cheia. RMS de ~1,5 px com TODAS as poses parecidas e' captura: apoie o "
          "tabuleiro (nao segure na mao), luz difusa, menos inclinacao. Pose "
          "borrada sozinha nao condena o conjunto: o programa descarta ate' %d "
          "poses (as piores, com o numero de cada uma) e diz QUAIS refazer."
          % (sc.KINECT_INTRINSICS_MAX_RMS, 0.67, 2.20, DESCARTES_MAX))
    print("- Tabuleiro PARADO (o cronometro de %.0f s so' comeca a contar quando "
          "os cantos param de andar na imagem)."
          % sc.STABLE_CAPTURE_SECONDS)


def quadro_de_aviso():
    """Cartao 'sem imagem', no tamanho final da janela (DISPLAY_SCALE)."""
    aviso = np.full((int(200 * sc.DISPLAY_SCALE), int(780 * sc.DISPLAY_SCALE), 3),
                    40, np.uint8)
    cv2.putText(aviso, "SEM IMAGEM do Kinect RGB", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(aviso, "confira o cabo USB 3.0 (direto, sem hub) e se",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(aviso, "outro programa nao esta' usando a camera | ESC sai",
                (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return aviso


def deslocamento_entre(cantos_a, cantos_b):
    """(px) deslocamento medio dos cantos entre duas deteccoes seguidas.

    Medida de MOVIMENTO, nao de erro: com o tabuleiro na mao os dois conjuntos de
    cantos saem em posicoes diferentes, e e' isso que diz que a amostra estaria
    borrada. Funcao pura -- recebe listas de cantos e devolve 'inf' quando um dos
    lados e' None ou tem outra forma.
    """
    if cantos_a is None or cantos_b is None:
        return float("inf")
    a = np.asarray(cantos_a, np.float64).reshape(-1, 2)
    b = np.asarray(cantos_b, np.float64).reshape(-1, 2)
    if a.shape != b.shape:
        return float("inf")
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def linha_de_status(poses, fracao, mediana, rms, motivo, dist_m=None,
                    ok_meta=None, total=None):
    """Texto unico da janela: e' ele que diz se a captura esta' indo bem.

    `dist_m` (opcional) e' a distancia estimada ao tabuleiro (pinhole, ver
    `distancia_aproximada_m`): e' o numero que o operador consulta para entrar
    na faixa 0,30-0,42 m ANTES de olhar o span.

    `ok_meta`/`total` (opcionais) sao quantas poses ja' cumprem a meta de span:
    o mediano sozinho nao diz se a captura esta' "quase" ou "longe" -- em
    05/10/2026 ele dizia 34,9% com 9 de 18 poses cumprindo a meta e o operador
    nao tinha como ver isso no console. Uma casa decimal no mediano pelo mesmo
    motivo (o portao reprova em 34,9% e o motivo arredonda para "35%").
    """
    rms_txt = "--" if rms is None else "%.3f px" % rms
    dist_txt = ("~%.2f m" % dist_m
                if dist_m is not None and np.isfinite(dist_m) else "-- m")
    conta_txt = (" %d/%d ok" % (ok_meta, total)
                 if total else " --/-- ok")
    return ("poses %d (min %d) | span agora %.1f%% (%s) | mediano %.1f%% "
            "(meta %.1f%%)%s | RMS %s | %s"
            % (poses, sc.KINECT_INTRINSICS_MIN_POSES, fracao * 100.0,
               dist_txt, mediana * 100.0,
               sc.KINECT_INTRINSICS_MIN_SPAN * 100.0, conta_txt, rms_txt, motivo))


def modelo_em_disco(caminho):
    """(rms, fx) do modelo congelado BOM em disco, ou (None, None).

    E' de onde `coleta_poses` tira o `melhor_rms` INICIAL: sem esta leitura a
    primeira pose que passasse os portoes sobrescrevia um arquivo melhor (a
    comparacao so' existia dentro da sessao). Um arquivo RECUSADO pela leitura
    (o paliativo de 30/09/2026, sem poses/span) devolve (None, None): ele nao
    pode travar a gravacao de um modelo novo. `carrega_intrinsecas_kinect` nunca
    levanta, entao esta funcao tambem nao.
    """
    matriz, _, info = sc.carrega_intrinsecas_kinect(caminho)
    if matriz is None:
        return None, None
    rms = info.get("rms")
    return (float(rms) if rms is not None else None), float(matriz[0, 0])


def placa_diagonal_m():
    """(m) diagonal do tabuleiro entre os cantos EXTERNOS -- derivada de sc.

    Fonte unica: a placa vem de `stereo_calibration` (a copia divergente de
    01/10/2026 foi o erro; aqui nao ha' constante de tabuleiro copiada).
    """
    return float(np.hypot((sc.CHECKERBOARD_COLS - 1) * sc.SQUARE_SIZE_M,
                          (sc.CHECKERBOARD_ROWS - 1) * sc.SQUARE_SIZE_M))


def distancia_aproximada_m(fracao, kinect_size, fx=FX_NOMINAL_PX):
    """Distancia (m) do tabuleiro ao Kinect pelo tamanho angular (pinhole).

    Z = fx * diagonal_da_placa / span_px, com span_px = fracao * diagonal do
    quadro. Com a placa de 324,5 mm e fx nominal 1081,37 px a 2203 px de
    diagonal, vale 0,159 / fracao: ~0,45 m aos 35% (portao do modelo) e ~0,27 m
    aos 60% (portao maximo de span) -- a faixa em que este arranjo foi
    desenhado. `fx` vem do modelo congelado quando existe; o padrao e' o nominal.

    ESTIMATIVA para o operador se posicionar, nao medicao: com o tabuleiro
    inclinado o span encolhe e a distancia sai superestimada.
    """
    if fracao is None:
        return float("nan")
    fracao = float(fracao)
    if not np.isfinite(fracao) or fracao <= 0.0:
        return float("nan")
    diagonal = float(np.hypot(*kinect_size))
    return float(fx) * placa_diagonal_m() / (fracao * diagonal)


def centros_das_poses(pontos):
    """(N,2) centros (px, resolucao cheia) das poses ja' aceitas."""
    if not pontos:
        return np.zeros((0, 2), np.float64)
    return np.array([np.asarray(item, np.float64).reshape(-1, 2).mean(axis=0)
                     for item in pontos])


def raio_anel_separacao(span_px, fracao_minima):
    """(px) raio do anel de separacao: e' literalmente o portao da captura.

    `aceita_pose_kinect` exige centro a >= `min_separacao` * span de TODAS as
    poses salvas; desenhar esse raio ao redor de cada uma mostra, antes do
    clique, onde a proxima pose seria recusada como "quase igual". O span usado
    e' o da pose ATUAL -- o mesmo que o portao usa.
    """
    return float(fracao_minima) * float(span_px)


def spans_das_poses(pontos, kinect_size):
    """(N,) fracao da diagonal que CADA pose ocupa -- o que o mediano esconde.

    Mesma conta por pose que `sc.span_do_tabuleiro` faz por dentro; existe para
    responder a pergunta que o MEDIANO nao responde: QUANTAS poses cumprem a
    meta. Foi esse numero escondido que deixou a bancada de 05/10/2026 recolher
    6 poses frageis (18 poses, mediano 34%) achando que ia bem.
    """
    diagonal = float(np.hypot(*kinect_size))
    if not pontos:
        return np.zeros(0, np.float64)
    return np.array([sc.board_span(np.asarray(p, np.float64).reshape(-1, 2))
                     for p in pontos], np.float64) / diagonal


def resumo_do_span(pontos, kinect_size):
    """(mediana, ok, total): o numero que DECIDE + quantas poses cumprem a meta.

    A mediana vem de `sc.span_do_tabuleiro` (a MESMA que o portao usa -- nao ha'
    segunda formula). `ok` conta, das poses ja' aceitas, as que ocupam pelo menos
    `sc.KINECT_INTRINSICS_MIN_SPAN` da diagonal.
    """
    fracao = spans_das_poses(pontos, kinect_size)
    if fracao.size == 0:
        return 0.0, 0, 0
    return (sc.span_do_tabuleiro(pontos, kinect_size),
            int((fracao >= sc.KINECT_INTRINSICS_MIN_SPAN).sum()), int(fracao.size))


def texto_do_span(pontos, kinect_size):
    """'span mediano 34.9% (meta 35.0%): 9/18 poses cumprem' -- ou '' sem poses.

    Uma casa decimal DE PROPOSITO: o portao reprova em 34,9% e o motivo do
    `sc.portoes_do_modelo_kinect` arredonda para "35%" -- a bancada de
    05/10/2026 leu "ocupou 35% da diagonal (minimo 35%)" e so' podia concluir
    que o programa estava quebrado.
    """
    mediana, ok, total = resumo_do_span(pontos, kinect_size)
    if total == 0:
        return ""
    return ("span mediano %.1f%% (meta %.1f%%): %d/%d poses cumprem"
            % (mediana * 100.0, sc.KINECT_INTRINSICS_MIN_SPAN * 100.0,
               ok, total))


def aviso_span_fraco(fracao, pontos, kinect_size):
    """Aviso (ou '') quando a pose ACEITA esta' abaixo da meta do mediano.

    O portao de CAPTURA aceita de `sc.MIN_BOARD_SPAN_FRACTION` (8%) para cima,
    mas o portao do MODELO olha o MEDIANO de todas as poses: aceitar uma pose a
    0,55 m (29%) ou a 0,8 m (18%) e' o caminho mais curto para o mediano cair
    abaixo de 35% e a captura inteira NAO gravar -- foi o que aconteceu em
    05/10/2026. O aviso sai no MOMENTO da pose, nao no fim.
    """
    if fracao is None or not np.isfinite(fracao) \
            or fracao >= sc.KINECT_INTRINSICS_MIN_SPAN:
        return ""
    mediana, ok, total = resumo_do_span(pontos, kinect_size)
    return ("ATENCAO: pose a %.1f%% (abaixo da meta %.1f%%): o portao do modelo "
            "olha o MEDIANO das %d poses (%.1f%%, %d cumprem a meta) e esta pose "
            "o PUXA PARA BAIXO -- volte para 0,35-0,40 m"
            % (fracao * 100.0, sc.KINECT_INTRINSICS_MIN_SPAN * 100.0,
               total, mediana * 100.0, ok))


def motivo_com_span(motivo, pontos, kinect_size):
    """Motivo do portao + o span mediano: o numero que o operador controla."""
    texto = texto_do_span(pontos, kinect_size)
    return motivo if not texto else "%s [%s]" % (motivo, texto)


def erros_por_pose(objetos, pontos, matriz, distorcao):
    """(N,) RMS (px) de reprojecao de CADA pose com o modelo JA' resolvido.

    `cv2.calibrateCamera` devolve UM RMS para o conjunto: com UMA pose borrada
    entre 15 boas ele fica em ~1,5 px (`sqrt((e_bad^2 + 15*0,3^2)/16)`) e nao diz
    QUAL pose quebrou -- foi assim que a bancada de 05/10/2026 perdeu a captura
    com "RMS 1.770 px" e nenhuma pista de onde olhar. Aqui cada pose e' resolvida
    sozinha (`cv2.solvePnP`) com as intrinsecas ja' resolvidas e o erro dela e'
    medido. `inf` quando o solve daquela pose falha (pose degenerada).
    """
    erros = np.zeros(len(objetos), np.float64)
    matriz = np.asarray(matriz, np.float64).reshape(3, 3)
    distorcao = np.asarray(distorcao, np.float64).reshape(-1, 1)
    for i, (obj, img) in enumerate(zip(objetos, pontos)):
        tabuleiro = np.asarray(obj, np.float64).reshape(-1, 1, 3)
        alvo = np.asarray(img, np.float64).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(tabuleiro, alvo.reshape(-1, 1, 2),
                                      matriz, distorcao)
        if not ok:
            erros[i] = float("inf")
            continue
        projetado = cv2.projectPoints(tabuleiro, rvec, tvec, matriz,
                                      distorcao)[0].reshape(-1, 2)
        erros[i] = float(np.sqrt(np.mean(np.sum((projetado - alvo) ** 2, axis=1))))
    return erros


def piores_poses(erros, quantas=3):
    """Indices das piores poses (maior erro primeiro) -- `[]` se nao houver."""
    erros = np.asarray(erros, np.float64).reshape(-1)
    if erros.size == 0:
        return []
    return [int(i) for i in np.argsort(-erros)[:max(0, int(quantas))]]


def descartes_sugeridos(erros, limite, maximo):
    """Indices que PODEM sair: erros CLARAMENTE acima do resto da coleta.

    Regra: `erro > max(3 x mediana, limite)`. Uma pose nesse estado nao esta'
    "pouco pior": a deteccao dela esta' estragada (movimento, reflexo, fora de
    foco). Sem esse corte o solver comecaria a descartar POSE BOA so' para baixar
    o numero -- o oposto de calibrar. Conjunto uniforme (todas iguais, todas
    ruins) devolve `[]`: a acao ali e' consertar a CAPTURA, nao apagar poses.
    """
    erros = np.asarray(erros, np.float64).reshape(-1)
    if erros.size == 0:
        return []
    corte = max(3.0 * float(np.median(erros)), float(limite))
    candidatos = [int(i) for i in np.argsort(-erros) if erros[i] > corte]
    return candidatos[:max(0, int(maximo))]


def veredito_do_rms(erros, limite):
    """Frase que separa POSE ISOLADA de CAPTURA inteira (a acao e' diferente).

    O corte e' o MESMO do descarte (`descartes_sugeridos`: pior > 3x mediana).
    Comparar a mediana com o limite NAO serve: quando uma pose estragada puxa o
    K/D, a mediana das poses boas SOBE junto (MEDIDO: 0,59 px com duas poses de
    8 px de ruido na cena) e o veredito mandaria o operador consertar o que esta'
    bom.
    """
    erros = np.asarray(erros, np.float64).reshape(-1)
    if erros.size == 0:
        return ""
    mediana = float(np.median(erros))
    if not descartes_sugeridos(erros, limite, 1):
        return ("as %d poses estao com erro PARECIDO (mediana %.2f px, limite "
                "%.2f px): o problema e' a CAPTURA INTEIRA -- luz difusa, "
                "tabuleiro APOIADO (nao na mao), menos inclinacao e detector em "
                "escala cheia (--largura-deteccao 0)"
                % (erros.size, mediana, float(limite)))
    lista = ", ".join("#%d %.2f px" % (i + 1, erros[i])
                      for i in piores_poses(erros, 3))
    return ("so' %d de %d poses puxam o erro (%s; mediana %.2f px, corte "
            "%.2f px): refaca SO' essas, do mesmo jeito que as boas que ja' "
            "entraram"
            % (int((erros > max(3.0 * mediana, float(limite))).sum()), erros.size,
               lista, mediana, max(3.0 * mediana, float(limite))))


def gancho_do_pykinect2(anterior=None, aviso=None):
    """`threading.excepthook` que transforma o wart do pykinect2 em UMA linha.

    `PyKinectRuntime.close()` derruba os handles enquanto a thread de frames ainda
    esta' esperando: ela levanta `AttributeError: 'NoneType' object has no
    attribute 'GetFrameArrivedEventData'` e o Python imprime um TRACEBACK inteiro
    depois do resultado -- na bancada de 05/10/2026 ele apareceu entre a recusa e
    o "NADA GRAVADO", parecendo parte do erro. So' esse caso e' filtrado;
    qualquer outra excecao de thread segue para o tratamento padrao.
    """
    anterior = anterior or threading.__excepthook__
    aviso = aviso or (lambda texto: print(texto))

    def gancho(args):
        if isinstance(getattr(args, "exc_value", None), AttributeError) \
                and "GetFrameArrivedEventData" in str(args.exc_value):
            aviso("  (aviso: o runtime do Kinect reclamou ao fechar -- wart "
                  "conhecido do pykinect2; nao muda nada do que foi gravado)")
            return
        anterior(args)

    return gancho


def desenha_marcadores(exibicao, pontos, cantos, fracao_minima):
    """Overlay das poses salvas: X + anel de separacao + centro ATUAL.

    Desenhado ja' na escala da JANELA (sc.DISPLAY_SCALE). O npz congelado nao
    guarda as coordenadas de cada pose, entao os marcadores sao os da SESSAO
    CORRENTE -- suficiente para enxergar o portao de separacao enquanto se
    captura. `cantos is None` (tabuleiro nao visto hoje) tira o anel e o ponto:
    sem span atual nao ha' raio nem centro a mostrar. Devolve `exibicao`
    (funcao pura de array: o teste chama direto, sem janela nenhuma).
    """
    escala = sc.DISPLAY_SCALE
    anel = None
    if cantos is not None:
        anel = raio_anel_separacao(sc.board_span(cantos),
                                   fracao_minima) * escala
    for centro in centros_das_poses(pontos):
        cx, cy = (np.asarray(centro, np.float64) * escala).astype(int)
        cx, cy = int(cx), int(cy)
        if anel is not None:
            cv2.circle(exibicao, (cx, cy), int(round(anel)),
                       COR_ANEL_SEPARACAO, 1)
        cv2.drawMarker(exibicao, (cx, cy), COR_POSE_SALVA,
                       cv2.MARKER_TILTED_CROSS, 16, 2)
    if cantos is not None:
        atual = np.asarray(cantos, np.float64).reshape(-1, 2).mean(axis=0)
        cx, cy = (atual * escala).astype(int)
        cv2.circle(exibicao, (int(cx), int(cy)), 7, COR_TABULEIRO_AGORA, 2)
    return exibicao


def avalia_conjunto(objetos, pontos, kinect_size):
    """(ok, rms, matriz, distorcao, poses, span, erros, motivo) das poses dadas.

    Os TRES portoes na ordem barata primeiro (poses/span antes do solve) e, se o
    solve rodou, o erro de reprojecao de CADA pose (`erros`). `matriz`/`distorcao`
    sao `None` quando o solve nem chegou a rodar; `erros` idem.
    """
    poses = len(objetos)
    span = sc.span_do_tabuleiro(pontos, kinect_size)
    ok, motivo = sc.portoes_do_modelo_kinect(poses, span)
    if not ok:
        return False, None, None, None, poses, span, None, motivo
    rms, matriz, distorcao, poses, span = sc.resolve_modelo_kinect(
        objetos, pontos, kinect_size)
    erros = erros_por_pose(objetos, pontos, matriz, distorcao)
    ok, motivo = sc.portoes_do_modelo_kinect(poses, span, rms)
    return ok, rms, matriz, distorcao, poses, span, erros, motivo


def tenta_gravar(objetos, pontos, kinect_size, saida, melhor_rms=None,
                 descartes_max=0):
    """Resolve o modelo das poses e GRAVA se os portoes passarem.

    Devolve (gravou, rms|None, motivo), com `gravou` em TRES estados:
    True = gravou agora | None = os portoes passam mas o arquivo em disco ja' e'
    melhor (nao mexi) | False = os portoes reprovaram.

    `melhor_rms` e' o RMS do arquivo que ja' esta' em disco: sem ele, mais poses
    quase iguais PODEM subir o RMS e sobrescrever um modelo melhor -- o caminho do
    par guarda o melhor (`if final["stereo_rms"] < best_rms`), e aqui vale a mesma
    regra. Nao imprime nada e nao abre camera nenhuma: quem chama decide o
    formato, entao esta funcao roda igual no teste, sem janela e sem hardware.

    `descartes_max` (0 = desligado): se o RMS reprovar, o solver pode DESCARTAR as
    poses piores (`descartes_sugeridos`) uma a uma, ate' esse numero, e gravar o
    conjunto que sobra -- desde que ELE passe nos tres portoes. E' o que faltava
    para UMA pose borrada nao custar a captura inteira (a bancada de 05/10/2026
    perdeu 16 poses boas com o RMS em 1,77 px); o que saiu vai no motivo, e o
    `poses` gravado no npz e' o numero REALMENTE usado.
    """
    try:
        ok, rms, matriz, distorcao, poses, span, erros, motivo = \
            avalia_conjunto(objetos, pontos, kinect_size)
    except cv2.error as exc:
        #: Geometria degenerada (todas as poses no mesmo plano/posicao): o cv2
        #: levanta. Nao perde a coleta -- o operador so' precisa variar.
        return False, None, ("calibrateCamera falhou (%s) -- varie as poses"
                             % str(exc).strip().splitlines()[-1])
    if not ok and erros is None:
        if poses < sc.KINECT_INTRINSICS_MIN_POSES:
            return False, None, ("%d/%d poses -- junte mais, variando posicao, "
                                 "inclinacao e distancia"
                                 % (poses, sc.KINECT_INTRINSICS_MIN_POSES))
        return False, None, motivo
    if not ok:
        texto = ("RMS %.3f px; %s | %s"
                 % (rms, motivo,
                    veredito_do_rms(erros, sc.KINECT_INTRINSICS_MAX_RMS)))
        recuperado = _poda_ate_passar(objetos, pontos, kinect_size, erros,
                                      int(descartes_max))
        if recuperado is None:
            return False, rms, texto
        rms_n, matriz, distorcao, poses, span, fora, _ = recuperado
        detalhe = "%s (%s; mediana %.2f px)" % (
            ", ".join("#%d" % (i + 1) for i in fora),
            " e ".join("%.2f px" % erros[i] for i in fora),
            float(np.median(erros)))
        if melhor_rms is not None and rms_n >= melhor_rms:
            return None, rms_n, ("sem as poses %s o RMS cai para %.3f px, mas o "
                                 "arquivo em disco (%.3f px) ainda e' melhor: nao "
                                 "regravei" % (detalhe, rms_n, melhor_rms))
        sc.salva_intrinsecas_kinect(matriz, distorcao, kinect_size, rms_n, poses,
                                    span, caminho=saida)
        return True, rms_n, ("gravado SEM as poses %s -- as %d restantes passam "
                             "nos tres portoes (RMS %.3f px). Se quiser TODAS, "
                             "refaca essas (tabuleiro parado, luz difusa)"
                             % (detalhe, poses, rms_n))
    if melhor_rms is not None and rms >= melhor_rms:
        return None, rms, ("portoes OK, mas o RMS %.3f px NAO melhorou o arquivo "
                           "em disco (%.3f px): nao regravei"
                           % (rms, melhor_rms))
    sc.salva_intrinsecas_kinect(matriz, distorcao, kinect_size, rms, poses, span,
                                caminho=saida)
    return True, rms, "gravado"


def _poda_ate_passar(objetos, pontos, kinect_size, erros, descartes_max):
    """Tira as piores poses UMA a uma e devolve o conjunto que passa -- ou None.

    Devolve `(rms, matriz, distorcao, poses, span, indices_fora, motivo)` com os
    indices ORIGINAIS das poses descartadas. Cada rodada REAVALIA os tres portoes
    (`avalia_conjunto`): o corte nao e' "baixar o numero", e' "o conjunto menor
    ainda cumpre o que o modelo congelado exige" (e ele pode parar de cumprir --
    por exemplo, as poses de borda que sustentavam o span mediano). Nada e'
    gravado aqui.
    """
    if descartes_max <= 0 or len(objetos) <= sc.KINECT_INTRINSICS_MIN_POSES:
        return None
    indices = list(range(len(objetos)))
    atuais = np.asarray(erros, np.float64)
    fora = []
    for _ in range(int(descartes_max)):
        sugestao = descartes_sugeridos(atuais, sc.KINECT_INTRINSICS_MAX_RMS, 1)
        if not sugestao or len(indices) <= sc.KINECT_INTRINSICS_MIN_POSES:
            return None
        fora.append(indices.pop(sugestao[0]))
        try:
            ok, rms, matriz, distorcao, poses, span, erros_novos, _ = \
                avalia_conjunto([objetos[i] for i in indices],
                                [pontos[i] for i in indices], kinect_size)
        except cv2.error:
            return None
        if ok:
            return rms, matriz, distorcao, poses, span, fora, None
        if erros_novos is None:
            return None
        atuais = erros_novos
    return None


def coleta_poses(kinect, kinect_size, args):
    """Loop de captura do arranjo Kinect-only. Devolve (gravou, rms, motivo)."""
    largura, altura = kinect_size
    diagonal = float(np.hypot(largura, altura))
    objetos, pontos = [], []
    estavel_desde = None
    ultima_deteccao = 0.0
    cantos_cache = None
    cantos_anteriores = None
    ultimo_motivo = "aguardando o tabuleiro"
    #: Motivo do PORTAO DO MODELO na ultima pose aceita (o motivo REAL de nao
    #: haver arquivo). Sem ele o `main()` reportava o ultimo estado do frame
    #: ("tabuleiro nao visto no Kinect") -- foi o que a bancada de 05/10/2026
    #: leu como causa, com 18 poses aceitas e o mediano de span em 34%.
    motivo_do_modelo = None
    #: RMS/fx do arquivo BOM em disco (o `None` de `tenta_gravar` -- "o disco ja'
    #: e' melhor" -- so' existe se este numero comeca' VERDADEIRO). Arquivo
    #: recusado pela leitura devolve (None, None) e nao trava o novo.
    melhor_rms, fx_disco = modelo_em_disco(args.saida)
    fx = fx_disco if fx_disco is not None else FX_NOMINAL_PX
    gravou = False
    #: O `rms` da linha de estado so e' atribuido dentro do `if aceita:` da
    #: captura; no primeiro frame nenhuma pose passou ainda e o bloco da
    #: janela o leu sem definicao (UnboundLocalError, crash sem capturar
    #: pose alguma - corrigido 08/10/2026). Inicializa com None e o
    #: `linha_de_status` imprime "--" enquanto nao ha' pose aceita.
    rms = None
    sem_frame_avisado = False
    inicio = time.time()
    limite_s = float(args.limite_s)
    if args.sem_janela and limite_s <= 0.0:
        limite_s = LIMITE_SEM_JANELA_S
        print("  --sem-janela sem --limite-s: vale o teto de %.0f s."
              % LIMITE_SEM_JANELA_S)

    while True:
        if limite_s > 0.0 and time.time() - inicio > limite_s:
            ultimo_motivo = "teto de tempo da captura atingido"
            print("  %s (%.0f s)." % (ultimo_motivo, limite_s))
            break
        quadro = None
        if kinect.has_new_color_frame():
            dados = kinect.get_last_color_frame()
            quadro = dados.reshape((altura, largura, 4))
            quadro = cv2.cvtColor(quadro, cv2.COLOR_BGRA2BGR)
        if quadro is None:
            if not sem_frame_avisado:
                print("  SEM FRAME do Kinect RGB (a janela mostra o aviso); "
                      "confira o cabo USB 3.0 direto na maquina e se outro "
                      "programa nao esta' usando a camera.")
                sem_frame_avisado = True
            if not args.sem_janela:
                cv2.imshow(JANELA, quadro_de_aviso())
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            time.sleep(0.05)
            continue
        sem_frame_avisado = False
        agora = cv2.getTickCount() / cv2.getTickFrequency()
        #: Detecta a ~10 Hz e reusa os cantos entre frames: o SB em 1080p a cada
        #: frame inviabiliza a janela (mesma razao do stereo_calibration).
        if agora - ultima_deteccao >= sc.DETECTION_INTERVAL_S:
            cantos_cache = sc.find_corners(quadro)
            ultima_deteccao = agora
        cantos = cantos_cache
        fracao = (sc.board_span(cantos) / diagonal) if cantos is not None else 0.0
        #: (mediana, quantas cumprem a meta, total): a mediana vem da MESMA
        #: funcao que o portao do modelo usa (`sc.span_do_tabuleiro`).
        mediana, ok_meta, total_poses = resumo_do_span(pontos, kinect_size)
        exibicao = sc.reduzir_janela(quadro.copy())
        if cantos is not None:
            _, canonico = sc.canonical_order(cantos)
            cv2.drawChessboardCorners(
                exibicao, sc.CHECKERBOARD_SIZE,
                (canonico * sc.DISPLAY_SCALE).astype(np.float32)
                .reshape(-1, 1, 2), True)

        # --- ESTABILIDADE: movimento zera o cronometro ----------------------
        if cantos is None:
            estavel_desde = None
            ultimo_motivo = "tabuleiro nao visto no Kinect"
        elif deslocamento_entre(cantos, cantos_anteriores) > MAX_DESLOCAMENTO_PX:
            estavel_desde = None
            ultimo_motivo = "tabuleiro EM MOVIMENTO (pare para eu contar 2 s)"
        else:
            if estavel_desde is None:
                estavel_desde = agora
                ultimo_motivo = "segure PARADO para a captura"
            if agora - estavel_desde >= sc.STABLE_CAPTURE_SECONDS:
                estavel_desde = agora   # re-avalia a cada 2 s, como o par
                aceita, motivo, fracao = sc.aceita_pose_kinect(
                    cantos, pontos, kinect_size,
                    min_separacao=args.deslocamento_minimo)
                if aceita:
                    _, canonico = sc.canonical_order(cantos)
                    objetos.append(sc.board_object_points().copy())
                    pontos.append(canonico.reshape(-1, 1, 2))
                    print("Pose %d aceita: span %.1f%%, inclinacao %.2f, "
                          "~%.2f m | %s"
                          % (len(objetos), fracao * 100.0,
                             sc.board_tilt_ratio(cantos),
                             distancia_aproximada_m(fracao, kinect_size, fx),
                             texto_do_span(pontos, kinect_size)))
                    #: A pose passou no portao de CAPTURA (8%+) mas esta' abaixo
                    #: da meta do MEDIANO: avisa na hora, porque e' ela que vai
                    #: derrubar a gravacao sem ninguem perceber (05/10/2026).
                    aviso_fraco = aviso_span_fraco(fracao, pontos, kinect_size)
                    if aviso_fraco:
                        print("  -> %s" % aviso_fraco)
                    gravou_agora, rms, ultimo_motivo = tenta_gravar(
                        objetos, pontos, kinect_size, args.saida, melhor_rms,
                        descartes_max=args.max_descartes)
                    #: A recusa leva o span MEDIANO no colchete: o motivo do
                    #: `sc.portoes_do_modelo_kinect` arredonda o span ("ocupou
                    #: 35% (minimo 35%)") e o operador so' pode concluir que o
                    #: programa esta' quebrado.
                    if gravou_agora is False:
                        ultimo_motivo = motivo_com_span(ultimo_motivo, pontos,
                                                       kinect_size)
                        motivo_do_modelo = ultimo_motivo
                    #: O motivo SAI sempre -- inclusive quando `rms is None` (o
                    #: portao reprovou ANTES do solve): era essa combinacao que
                    #: escondia a recusa por span na bancada de 05/10/2026.
                    print("  -> %s" % ultimo_motivo)
                    #: REGRESSAO da linha 319: `False` (portao recusou) NAO e'
                    #: sucesso. O antigo `is not None` aceitava o False, marcava
                    #: gravou=True e o programa imprimia PRONTO sem arquivo.
                    #: `None` = os portoes passam e o arquivo em disco ja' e' o
                    #: melhor: ha' modelo gravado, so' nao foi REESCRITO agora.
                    if gravou_agora is not False:
                        gravou = True
                        #: Ha' gravacao: o motivo que vale e' o DELA. Sem isto o
                        #: retorno preferia a ultima recusa antiga ("junte mais
                        #: poses") a um arquivo que acabou de ser escrito.
                        motivo_do_modelo = None
                        if gravou_agora:
                            melhor_rms = rms
                            print("  (modelo gravado agora)")
                        else:
                            print("  (o modelo JA' esta' gravado. Mais poses "
                                  "podem melhorar o RMS; ESC encerra quando "
                                  "estiver bom)")
                    if gravou and (args.sair_ao_gravar or args.sem_janela):
                        break
                    if len(objetos) >= int(args.max_poses):
                        ultimo_motivo = ("teto de %d poses atingido"
                                         % int(args.max_poses))
                        break
        cantos_anteriores = cantos

        # --- janela: marcadores, linha de estado e legenda -------------------
        if not args.sem_janela:
            centros = centros_das_poses(pontos)
            desenha_marcadores(exibicao, pontos, cantos,
                               args.deslocamento_minimo)
            cv2.putText(exibicao, linha_de_status(
                len(objetos), fracao, mediana, rms, ultimo_motivo,
                distancia_aproximada_m(fracao, kinect_size, fx), ok_meta,
                total_poses), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                COR_TABULEIRO_AGORA, 1, cv2.LINE_AA)
            cv2.putText(exibicao,
                        "X verde = pose ja' aceita | anel = separacao minima "
                        "| %.0f-%0.f m" % (0.30, 0.45),
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        COR_POSE_SALVA, 1, cv2.LINE_AA)
            cv2.imshow(JANELA, exibicao)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    #: O motivo reportado e' o do PORTAO (o real), nao o do ultimo frame: com 18
    #: poses aceitas e o mediano em 34% o `ultimo_motivo` do frame dizia
    #: "tabuleiro nao visto no Kinect" e a bancada leu isso como causa.
    if not gravou and motivo_do_modelo:
        return False, None, motivo_do_modelo
    return gravou, melhor_rms, ultimo_motivo


def main():
    args = parse_args()
    sc.DETECTION_WIDTHS = tuple(int(v) for v in args.largura_deteccao)
    sc.DETECTION_CLAHE_CLIP = float(args.clahe)
    print("Tabuleiro: %dx%d cantos internos, quadrado de %g mm | detector: "
          "larguras %s | CLAHE %g"
          % (sc.CHECKERBOARD_COLS, sc.CHECKERBOARD_ROWS, sc.SQUARE_SIZE_M * 1000,
             tuple(sc.DETECTION_WIDTHS), sc.DETECTION_CLAHE_CLIP))
    print("Portoes do modelo congelado: %d poses | span >= %.0f%% da diagonal | "
          "RMS <= %g px"
          % (sc.KINECT_INTRINSICS_MIN_POSES, sc.KINECT_INTRINSICS_MIN_SPAN * 100,
             sc.KINECT_INTRINSICS_MAX_RMS))
    print_guidance()
    try:
        kinect = sc.abre_kinect()
    except RuntimeError as exc:
        print("ERRO: %s" % exc)
        return 2
    try:
        largura = kinect.color_frame_desc.Width
        altura = kinect.color_frame_desc.Height
        print("Resolucao do Kinect RGB: %dx%d | destino: %s"
              % (largura, altura, args.saida))
        gravou, rms, motivo = coleta_poses(kinect, (largura, altura), args)
    finally:
        kinect.close()
        cv2.destroyAllWindows()
    print("")
    if gravou:
        print("PRONTO: %s | RMS %.3f px | ultimo estado: %s"
              % (args.saida, rms if rms is not None else float("nan"), motivo))
        print("Este modelo passa a valer para TODA calibracao de par. Agora "
              "recalibre cada auxiliar UMA vez contra ele (stereo_calibration.py, "
              "sem --salvar-intrinsecas-kinect) e valide com a tecla T.")
        return 0
    print("NADA GRAVADO: %s" % motivo)
    print("O span que decide e' o MEDIANO das poses (>= %.1f%% da diagonal): uma "
          "pose a 0,50-0,80 m (span ~32%%-20%%) DERRUBA o conjunto, mesmo com 12 "
          "poses boas. Mantenha TODAS entre 0,30 e 0,42 m (mire 0,35-0,40 m)."
          % (sc.KINECT_INTRINSICS_MIN_SPAN * 100.0))
    print("O arquivo de destino NAO foi tocado.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
