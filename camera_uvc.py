"""Controles UVC das webcams: fourcc/resolucao, exposicao/ganho/fps/foco.

DOIS problemas MEDIDOS na maquina da bancada (10/10/2026) e o que este modulo
faz por cada um:

1. **FPS**: no modo padrao (1280x720 YUY2, bruto) as DUAS webcams entregam só
   8-10 fps -- 100-125 ms entre quadros; a 300-950 px/s (movimento rapido ja'
   observado no overlay) sao **30-120 px de deslocamento POR QUADRO**: e' a
   espinha dos picos de movimento. `pede_mjpeg` resolve com UMA linha: 720p
   **MJPEG = 30 fps** nas duas cameras, na MESMA resolucao (nao perde canto do
   tabuleiro). `mede_fps` mede o fps REAL (a propriedade CAP_PROP_FPS MENTE:
   aceitou "60" e entregou 4-8 fps) e `trava_exposicao` imprime o medido.

2. **EXPOSICAO**: com auto-exposure em luz fraca o driver alonga o tempo de
   integracao e e' isso que borra o movimento (a Trust estava em **-4 ~ 1/16 s**
   automatica; a SIGMA ja' em -7 ~ 1/128 s). Congelar troca "varia durante a
   sessao" por "fixa"; para ENCURTAR o tempo (`exposicao`, ex. -7) exige LUZ
   (LED difuso) -- a linha imprime o tempo atual com heuristica de log2 e
   AVISA quando ele e' longo demais pro movimento. O desligamento do autofocus
   evita o foco "respirando" invalidar a calibracao no meio da sessao.

DirectShow/MSMF IGNORA propriedades (medido: BUFFERSIZE=1 ignorado pela Trust;
fps pedido 60 aceito na propriedade e NAO entregue), entao AQUI cada set() e'
lido de volta e o relatorio diz o que a camera OBEDIU, o que IGNOROU e o que
REALMENTE entregou. `exposicao`/`ganho` sao unidades DO DRIVER; sem valor
explicito a funcao SO' desliga o automatico (tempo fica no atual).

Modulo propositalmente separado: `camera_names.py` e' deliberadamente leve "sem
cv2" (ver o docstring dele); aqui so' cv2 + stdlib.
"""
from __future__ import annotations

import math
import time

import cv2

#: Tentativas, em ordem, para DESLIGAR a auto-exposicao no Windows: a convencao
#: do OpenCV/FFmpeg e' 0.25 = manual / 0.75 = auto; alguns drivers usam 0/1.
#: A leitura de volta e' quem decide se pegou (<= 0.5 = manual).
CANDIDATOS_MANUAL = (0.25, 0.0)

#: Fourcc que destrava 720p30 nas webcams desta bancada (medido: bruto 8-10
#: fps, MJPEG 30 fps na mesma resolucao).
MODO_MJPEG = "MJPG"

#: Tempo de medicao do fps real (s) dentro de `trava_exposicao`.
FPS_MEDICAO_S = 0.5


def _interp_tempo(valor):
    """Heuristica de tempo quando o driver devolve log2 de segundos (DSHOW).

    -7 -> 2**7 = 128 -> "~1/128 s". Valores fora de [-13, -1] (ex.: unidade
    100-us do MSMF) sao devolvidos CRUS -- a linha impressa mostra o bruto
    junto, entao nada aqui engana o usuario.
    """
    if valor is None or not math.isfinite(valor):
        return "?"
    if -13.0 <= valor <= -1.0:
        return f"~1/{int(2 ** (-valor))} s"
    return "bruto"


def define_modo(captura, largura, altura, mjpeg=True):
    """Resolve o modo da stream: RESOLUCAO primeiro e DEPOIS o FOURCC.

    A ORDEM importa (medido em 10/10/2026, nas DUAS webcams): pedir MJPEG ANTES
    de W/H faz o driver renegociar o formato ao setar o tamanho e VOLTAR para o
    bruto -- fourcc lido de volta 'YUY2' -> 8-10 fps; com W/H primeiro e MJPEG
    depois, o fourcc fica 'MJPG' e a 720p entrega 30 fps. (A trava de
    exposicao NAO afeta o fps -- medido com e sem.)
    """
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    if mjpeg:
        captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*MODO_MJPEG))


def le_fourcc(captura):
    """fourcc REAL da stream ('MJPG'/'YUY2'/...) via get(CAP_PROP_FOURCC).

    A leitura e' a unica forma honesta de saber o modo: ja' se mediu um
    chamador que PEDIU MJPEG e RECEBEU YUY2 (por ordem dos sets) e imprimiu
    "MJPEG" na cara -- confie no que a camera DEVOLVEU, nao no pedido.
    """
    try:
        valor = int(captura.get(cv2.CAP_PROP_FOURCC))
    except Exception:
        return "?"
    if valor <= 0:
        return "?"
    return "".join(chr((valor >> (8 * i)) & 0xFF) for i in range(4))


def mede_fps(captura, segundos=FPS_MEDICAO_S):
    """fps REAL entregue pela camera (a propriedade CAP_PROP_FPS MENTE).

    Descarta 2 quadros (aquecimento) e conta por `segundos`. Devolve nan
    quando a captura nao tem `read` (dupla fake) ou nao entrega quadro.
    """
    leitor = getattr(captura, "read", None)
    if not callable(leitor):
        return float("nan")
    try:
        for _ in range(2):
            leitor()
        n = 0
        t0 = time.time()
        while time.time() - t0 < segundos:
            ok, _ = leitor()
            if ok:
                n += 1
        dur = time.time() - t0
        return (n / dur) if dur > 0 and n else float("nan")
    except Exception:
        return float("nan")


def parse_controle_por_camera(texto):
    """'N' (todas as cameras) ou 'indice:N,indice:N' -> mapa para busca.

    Existe porque as DUAS webcams desta bancada pedem exposicoes DIFERENTES na
    MESMA cena (medido em 19/10/2026: com exposicao -7 a Trust ficou com media
    41 -- 46% dos pixels abaixo de 30 -- enquanto a SIGMA entregava media 138).
    Um valor unico para as duas escurece uma delas.

    Devolve {"todos": valor|None, "indices": {indice: valor}}.
    """
    mapa = {"todos": None, "indices": {}}
    if texto is None or texto == "":
        return mapa
    if isinstance(texto, (int, float)):
        mapa["todos"] = float(texto)
        return mapa
    pedacos = [pedaco.strip() for pedaco in str(texto).replace(";", ",").split(",")
               if pedaco.strip()]
    for pedaco in pedacos:
        if ":" not in pedaco:
            mapa["todos"] = float(pedaco)
            continue
        indice, valor = pedaco.split(":", 1)
        mapa["indices"][int(indice)] = float(valor)
    return mapa


def valor_por_camera(mapa, indice):
    """Valor do controle para a camera `indice` (override por indice > todas)."""
    if not mapa:
        return None
    indices = mapa.get("indices") or {}
    if int(indice) in indices:
        return indices[int(indice)]
    return mapa.get("todos")


def trava_exposicao(captura, exposicao=None, ganho=None, fps=None,
                    travar_foco=True, verbose=True, rotulo=""):
    """Congela exposicao/foco; opcionalmente fixa tempo/ganho/fps; mede fps real.

    `exposicao`/`ganho`: unidades DO DRIVER (a linha imprime o atual com a
    heuristica de log2). `fps` em quadros/s (pedido; a propriedade pode mentir
    -- o fps REAL medido vai na mesma linha). `travar_foco=False` nao mexe no
    autofocus.

    Devolve dict {propriedade: valor_lido}; `verbose` imprime UMA linha
    "[exposicao <rotulo>] ..." com o que pegou, o que ignorou e o fps real.
    NUNCA lanca excecao (roda no caminho de abertura de camera).
    """
    def _le(prop):
        try:
            valor = float(captura.get(prop))
            return valor if math.isfinite(valor) else float("nan")
        except Exception:
            return float("nan")

    def _set(prop, valor):
        try:
            captura.set(prop, float(valor))
        except Exception:
            pass
        return _le(prop)

    relatorio = {}
    partes = []

    # --- 1) auto-exposicao -> manual (a camera mantem o tempo atual) ---------
    manual = float("nan")
    for candidato in CANDIDATOS_MANUAL:
        manual = _set(cv2.CAP_PROP_AUTO_EXPOSURE, candidato)
        if math.isfinite(manual) and manual <= 0.5:
            break
    relatorio["auto_exposicao"] = manual
    if math.isfinite(manual) and manual <= 0.5:
        partes.append(f"auto-expo->manual ({manual:g})")
    else:
        partes.append(f"auto-expo IGNORADA (leu {manual:g})")

    # --- 2) tempo de exposicao: fixa se pedido; leitura SEMPRE --------------
    if exposicao is not None:
        tempo = _set(cv2.CAP_PROP_EXPOSURE, exposicao)
        partes.append(f"tempo {tempo:g} ({_interp_tempo(tempo)})")
    else:
        tempo = _le(cv2.CAP_PROP_EXPOSURE)
        partes.append(f"tempo atual {tempo:g} ({_interp_tempo(tempo)})")
    relatorio["tempo"] = tempo
    #: AVISO de tempo LONGO (heuristica log2): a 1/16 s um movimento rapido
    #: borra dentro do proprio quadro -- e' o caso da Trust no automatico.
    if math.isfinite(tempo) and -13.0 <= tempo <= -1.0 and tempo >= -6.0:
        partes.append("ATENCAO tempo LONGO: borrara o movimento "
                      "(encurtte, ex.: --exposicao -7 ~1/128 s, com luz)")

    # --- 3) ganho: fixa se pedido; leitura informativa ----------------------
    if ganho is not None:
        ganho_lido = _set(cv2.CAP_PROP_GAIN, ganho)
        partes.append(f"ganho {ganho_lido:g}")
    relatorio["ganho"] = _le(cv2.CAP_PROP_GAIN)

    # --- 4) fps pedido (a propriedade pode aceitar e NAO entregar) ----------
    if fps is not None:
        fps_lido = _set(cv2.CAP_PROP_FPS, fps)
        partes.append(f"fps pedido {fps_lido:g}")

    # --- 5) autofocus: desliga MANTENDO a posicao atual ----------------------
    #    (nao se mexe em CAP_PROP_FOCUS: unidade = posicao absoluta da lente,
    #    setar 0 DESFOCARIA a imagem em vez de congelar o foco.)
    prop_foco = getattr(cv2, "CAP_PROP_AUTOFOCUS", None)
    if travar_foco and prop_foco is not None:
        foco = _set(prop_foco, 0)
        relatorio["autofocus"] = foco
        if math.isfinite(foco) and foco <= 0.5:
            partes.append("foco auto off")
        else:
            partes.append(f"foco auto IGNORADO (leu {foco:g})")

    # --- 6) fps REAL (a propriedade mentiu: pediu 60, entregou 4-8) ---------
    fps_real = mede_fps(captura)
    relatorio["fps_real"] = fps_real
    if math.isfinite(fps_real):
        partes.append(f"fps REAL {fps_real:.1f}")

    if verbose:
        prefixo = f"[exposicao {rotulo}]" if rotulo else "[exposicao]"
        print(prefixo + " " + " | ".join(partes), flush=True)
    return relatorio
