#!/usr/bin/env python
"""Compara o rastreamento da mao: Kinect (depth/MediaPipe) x multi-view (stereo).

Janela UNICA (OpenCV) com quatro paineis:

    +--------------------------------+--------------------------------+
    | Kinect RGB                     | Auxiliar (espelhada)           |
    |  verde   = MediaPipe do Kinect |  verde   = MediaPipe da aux    |
    |  amarelo = 3D multi-view       |  ciano   = 3D multi-view       |
    |  vermelho= dorso (landmark 9)  |  magenta = 3D do DEPTH Kinect  |
    +--------------------------------+--------------------------------+
    | Vista 3D (X-Z): depth x stereo | Contadores e metricas ao vivo  |
    +--------------------------------+--------------------------------+

O que ele grava (dois CSV + um JSON de metadados, ao lado do relatorio):
- `<prefixo>_frames.csv`    : uma linha por (frame, lado) -- deteccao, 3D, erros,
                              queda de depth, tempos;
- `<prefixo>_landmarks.csv` : uma linha por (frame, lado, landmark 0..20) --
                              posicao 3D de cada metodo, reprojecao, confianca,
                              validade;
- `<prefixo>_meta.json`     : sessao, cameras, calibracao, versoes, limiares.

Ao final ele chama `tools/metricas_tracking_mao.py` e escreve o relatorio
(`<prefixo>_relatorio_tracking.md`) + PNGs. Nada de video/frame e' gravado.

Por que este arquivo existe: o `tools/multiview_janela.py` mostra o diagnostico
visual, mas nao guarda o dado nem mede cobertura/jitter/continuidade. Aqui o
objetivo e' DECIDIR COM DADOS qual dos dois caminhos e' mais robusto para o
dorso (landmark 9) e para os 21 landmarks, no mesmo frame e no mesmo
referencial (camera de cor do Kinect, metros, eixos OpenCV).

Uso (da raiz do projeto, com as cameras ligadas):
    .venv\\Scripts\\python.exe tools\\compara_tracking_mao.py --aux-index 0 --duracao 15
    .venv\\Scripts\\python.exe tools\\compara_tracking_mao.py --aux-index 0 --duracao 15 --sem-janela

Teclas na janela: ESC/Q sai; E marca fase ESTATICA; M marca MOVIMENTO; R marca
REPOUSO; D liga/desliga o desenho dos pontos de depth.

Se faltar Kinect, camera auxiliar ou calibracao stereo utilizavel o programa
FALHA com a mensagem do que fazer (nao roda "meio funcionando": sem os dois
caminhos nao ha' comparacao).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import metricas_tracking_mao as metricas  # noqa: E402  (irmao nesta pasta)

#: Destino padrao dos resultados (pedido da tese, fora do repositorio para nao
#: versionar dado pessoal de bancada sem querer).
SAIDA_PADRAO = Path(
    r"C:\Users\Usuario\OneDrive\Desktop\Mestrado_Projeto\Tese_Latex_Agente\Tracking")

IDX_DORSO = 9
PONTOS_DORSO = (0, 5, 9, 13, 17)

#: Extrinsica depth->color do PROPRIO Kinect (calibracao do sensor, nao do rig).
#: O 3D do depth sai de `MapDepthPointToCameraSpace` (frame da camera de
#: PROFUNDIDADE), mas a calibracao stereo das auxiliares vive no frame da camera
#: de COR. Sem esta correcao, projetar o 3D do depth na auxiliar injeta a
#: baseline depth->color (~6 cm) de forma sistematica -- medido em 18/11/2026:
#: `kd_aux` mediano caia de 84,6 -> 15,2 px (aux0) e 39,0 -> 10,1 px (aux1) na
#: sessao de 12/11. Arquivo gerado por
#: `tools/auditoria_multiview_offline.py --salvar-frame-cor ...`.
RAIZ_PROJETO = Path(__file__).resolve().parent.parent.parent
FRAME_COR_FILE = (RAIZ_PROJETO / "configs" / "calibracao"
                  / "kinect_depth_para_cor.npz")
_FRAME_COR = None
_FRAME_COR_LIDO = False


def _frame_cor() -> dict | None:
    """Le (uma vez) a extrinsica depth->color; None = correcao indisponivel."""
    global _FRAME_COR, _FRAME_COR_LIDO
    if _FRAME_COR_LIDO:
        return _FRAME_COR
    _FRAME_COR_LIDO = True
    try:
        if FRAME_COR_FILE.exists():
            dados = np.load(FRAME_COR_FILE, allow_pickle=True)
            _FRAME_COR = {
                "R": np.asarray(dados["rotation"], np.float64).reshape(3, 3),
                "t": np.asarray(dados["translation"], np.float64).reshape(3),
                "residuo_px": (float(dados["residuo_px"])
                               if "residuo_px" in dados.files else float("nan")),
                "arquivo": str(FRAME_COR_FILE),
            }
    except (OSError, KeyError, ValueError):
        _FRAME_COR = None
    return _FRAME_COR


def _corrige_frame_cor(xyz):
    """X_cor = R X_depth + t (frame da profondidade -> frame da cor).

    Identidade quando o npz de correcao nao existe (comportamento antigo).
    """
    cfg = _frame_cor()
    xyz = np.asarray(xyz, np.float64)
    if cfg is None:
        return xyz
    return xyz @ cfg["R"].T + cfg["t"].reshape(1, 3)


def _nota_frame_3d() -> str:
    """Texto do meta.json explicando em que frame o 3D foi medido."""
    cfg = _frame_cor()
    if cfg is None:
        return ("3D do depth no frame da CAMERA DE PROFUNDIDADE; as metricas que "
                "comparam esse 3D com a auxiliar (kd_aux, d9) ficam infladas pela "
                "baseline depth->color. Gere configs/calibracao/"
                "kinect_depth_para_cor.npz com tools/auditoria_multiview_offline.py "
                "--salvar-frame-cor")
    return ("3D do depth convertido para o frame da CAMERA DE COR por "
            f"{cfg['arquivo']} (residuo PnP {cfg['residuo_px']:.2f} px); "
            "kd_aux e d9 comparam no MESMO frame da calibracao stereo")

# --------------------------------------------------------------------------
# Schema do CSV (as colunas sao o contrato com metricas_tracking_mao.py)
# --------------------------------------------------------------------------
#: Uma linha por (frame, lado). Unidades: metros, milissegundos, pixels.
CABECALHO_FRAMES = [
    "seq",                  # indice do frame gravado (0..N-1)
    "t_unix",               # time.time() do inicio do processamento
    "t_mono",               # time.perf_counter() -- base das metricas temporais
    "origem",               # 'kinect_aux' (bancada) ou 'sintetico'
    "aux_index",            # indice OpenCV da camera auxiliar ativa
    "hand",                 # 'right'/'left'/'none' (none = frame sem mao)
    "side",                 # lado atribuido pelo pipeline (lado do Kinect)
    "side_aux",             # lado atribuido pelo MediaPipe da AUXILIAR
    "n_maos_k",             # quantas maos o Kinect detectou neste frame
    "n_maos_a",             # quantas maos a auxiliar detectou neste frame
    "pareamento",           # 'lado' | 'unico' | 'so_kinect' | 'so_aux' | ''
    "fase",                 # rotulo do operador: ''=auto, 'estatico','movimento','repouso'
    "ok_kinect_detect",     # 1 = o RGB do Kinect teve mao detectada
    "ok_aux_detect",        # 1 = a auxiliar teve mao detectada
    "mv_triangulou",        # 1 = triangulate_with_kinect devolveu pontos
    "mv_valid",             # 1 = pelo menos 5 landmarks 3D validos no multi-view
    "n_landmarks_k",        # quantos dos 21 tem ponto 3D do depth
    "n_landmarks_mv",       # quantos dos 21 tem ponto 3D do multi-view valido
    "palm9_k_valid",        # dorso (landmark 9) do Kinect valido?
    "palm9_k_x", "palm9_k_y", "palm9_k_z",
    "palm9_mv_valid",
    "palm9_mv_x", "palm9_mv_y", "palm9_mv_z",
    "reproj_mv_kin_med_px",  # mediana do erro de reprojecao no RGB do Kinect
    "reproj_mv_aux_med_px",  # mediana do erro na AUXILIAR (vista independente)
    #: Mediana de |projecao do 3D do DEPTH na auxiliar - pixel do MediaPipe na
    #: auxiliar|. E' o diagnostico que separa "triangulacao errada" de
    #: "correspondencia de pixels enviesada": a triangulacao pode ter residuo
    #: pequeno nos DOIS pixels (por construcao, usa os mesmos pixels) e ainda
    #: assim o 3D sair deslocado se os pixels de uma camera estiverem
    #: sistematicamente fora do lugar. Medido na bancada 11/11/2026: 59 px de
    #: mediana com o 3D do depth deslocado ~250 mm do multi-view.
    "kd_aux_err_med_px",
    "reproj_k_self_med_px",  # Kinect: ida-e-volta pelo mapper do SDK (sanidade)
    "map_fail",              # quantos landmarks sem pixel de depth (mapeamento)
    "depth_fail",            # quantos sem leitura de depth antes do resgate
    "depth_rescue",          # quantos precisaram do resgate por prior de Z
    "sync_color_depth_ms",   # delta entre a captura do RGB e do depth do Kinect
    "d_kin_mv_dorso_mm",     # |palm9 Kinect - palm9 multi-view| (mm)
    "aux_lag_ms",            # leitura da auxiliar - captura do RGB do Kinect
    "dt_s",                  # intervalo desde o frame anterior
    "fps_inst",              # 1/dt_s
    "t_kin_detect_ms",       # custo do MediaPipe no RGB do Kinect
    "t_aux_detect_ms",       # custo do MediaPipe na auxiliar
    "t_tri_ms",              # custo de 21x depth 3D + triangulacao + projecoes
    "t_ctrl_ms",             # custo total de processamento do frame
    "note",                 # texto livre (motivo de recusa, aviso)
]

#: Uma linha por (frame, lado, landmark). Vazio = nao medido/nao valido.
CABECALHO_LANDMARKS = [
    "seq", "t_unix", "t_mono", "origem", "hand", "landmark_id", "is_dorso",
    "k_valid",              # 1 = posicao 3D do Kinect valida
    "k_conf",               # 1.0 (depth medido) | 0.5 (resgatado) | 0
    "k_map_fail",           # nao achou pixel de depth para este landmark
    "k_depth_fail",         # depth sem leitura (antes do resgate)
    "k_depth_rescue",       # profundidade substituida pelo prior
    "k_depth_m",            # profundidade USADA (m): medida ou resgatada
    "k_x_m", "k_y_m", "k_z_m",
    "k_reproj_px",          # ida-e-volta pelo MAPPER do SDK (consistencia)
    "mv_valid",             # 1 = triangulacao plausivel e dentro do gate
    "mv_implausivel",       # 1 = triangulou mas foi recusado (z/gate)
    "mv_conf",              # 1 - reproj/gate, cortado em [0,1]
    "mv_x_m", "mv_y_m", "mv_z_m",
    "mv_reproj_kin_px",
    "mv_reproj_aux_px",
    "kin_px_x", "kin_px_y",  # pixel do landmark no RGB do Kinect
    "aux_px_x", "aux_px_y",  # pixel do landmark na auxiliar (espelhada)
    #: |projecao do 3D do depth na auxiliar - pixel do landmark na auxiliar|
    #: (px). Vazio quando uma das duas fontes nao existe naquele frame.
    "kd_aux_err_px",
]

#: Convivencia dos dois caminhos: quantos pontos sao necessarios para dizer que
#: o frame tem mao "de verdade".
MIN_LANDMARKS_3D = 5


def _n(valor, casas: int = 6) -> str:
    """Numero formatado para CSV; NaN/inf/None -> '' (nunca 0.0 disfarcado)."""
    if valor is None:
        return ""
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    return f"{v:.{casas}f}"


def _i(valor) -> str:
    return "1" if bool(valor) else "0"


def _txt(valor) -> str:
    """Texto sem quebra de linha (o CSV e' uma linha por registro)."""
    return " ".join(str(valor).split())


class EscritorCSV:
    """Escreve os dois CSV de forma incremental (flush por frame).

    Incremental de proposito: uma sessao de bancada pode ser interrompida com
    CTRL+C ou com a camera caindo, e perder 15 s de medida por causa disso e'
    pior do que pagar um flush por frame.
    """

    def __init__(self, pasta: Path, prefixo: str):
        self.pasta = Path(pasta)
        self.prefixo = prefixo
        self.caminho_frames = self.pasta / f"{prefixo}_frames.csv"
        self.caminho_landmarks = self.pasta / f"{prefixo}_landmarks.csv"
        self.caminho_meta = self.pasta / f"{prefixo}_meta.json"
        self._fh_frames = None
        self._fh_lm = None
        self.n_frames = 0
        self.n_landmarks = 0

    def abrir(self):
        self.pasta.mkdir(parents=True, exist_ok=True)
        self._fh_frames = self.caminho_frames.open("w", newline="",
                                                  encoding="utf-8")
        self._fh_lm = self.caminho_landmarks.open("w", newline="",
                                                  encoding="utf-8")
        import csv as _csv
        self._w_frames = _csv.writer(self._fh_frames)
        self._w_lm = _csv.writer(self._fh_lm)
        self._w_frames.writerow(CABECALHO_FRAMES)
        self._w_lm.writerow(CABECALHO_LANDMARKS)
        self._fh_frames.flush()
        self._fh_lm.flush()
        return self

    def frame(self, campos: dict):
        self._w_frames.writerow([campos.get(c, "") for c in CABECALHO_FRAMES])
        self._fh_frames.flush()
        self.n_frames += 1

    def landmark(self, campos: dict):
        self._w_lm.writerow([campos.get(c, "") for c in CABECALHO_LANDMARKS])
        self._fh_lm.flush()
        self.n_landmarks += 1

    def meta(self, dados: dict):
        self.caminho_meta.write_text(
            json.dumps(dados, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")

    def fechar(self):
        for fh in (self._fh_frames, self._fh_lm):
            try:
                if fh is not None:
                    fh.close()
            except OSError:
                pass
        self._fh_frames = None
        self._fh_lm = None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Comparacao Kinect (depth/MediaPipe) x multi-view da mao",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aux-index", type=int, default=None,
                   help="Indice OpenCV da camera auxiliar (padrao: a primeira "
                        "externa com stereo utilizavel)")
    p.add_argument("--duracao", type=float, default=15.0,
                   help="Duracao da gravacao em segundos (padrao 15)")
    p.add_argument("--espera", type=float, default=5.0,
                   help="Contagem regressiva antes de gravar (padrao 5 s; 0 = "
                        "comeca imediatamente)")
    p.add_argument("--sem-janela", action="store_true",
                   help="Nao abre janela (grava igual; util para teste rapido)")
    p.add_argument("--saida-dir", default=str(SAIDA_PADRAO),
                   help=f"Pasta dos resultados (padrao: {SAIDA_PADRAO})")
    p.add_argument("--prefixo", default="tracking_mao",
                   help="Prefixo dos arquivos (padrao: tracking_mao)")
    p.add_argument("--pareamento", choices=("lado", "unico", "ambos"),
                   default="ambos",
                   help="Como casar as maos do Kinect com as da auxiliar: "
                        "'lado' (o que o pipeline de producao faz), 'unico' "
                        "(casa quando ha' exatamente 1 mao em cada camera) ou "
                        "'ambos' (padrao: lado e, se nao casar, unico)")
    p.add_argument("--reproj-max", type=float, default=metricas.CRITERIOS["reproj_max_px"],
                   help="Gate de reprojecao do multi-view em px (padrao 6.0, o "
                        "mesmo TRI_ERR_MAX_PX do groundtruth)")
    p.add_argument("--salto-mm", type=float, default=50.0,
                   help="Limiar de salto frame-a-frame para as metricas (mm)")
    p.add_argument("--trava-exposicao", action="store_true",
                   help="Tenta travar exposicao/ganho da auxiliar "
                        "(camera_uvc.trava_exposicao)")
    p.add_argument("--sem-relatorio", action="store_true",
                   help="So' grava os CSV (nao gera relatorio/PNGs no fim)")
    p.add_argument("--sintetico", action="store_true",
                   help="NAO abre hardware: gera uma sessao FALSA so' para "
                        "validar CSV/metricas/relatorio (marcada como sintetica)")
    p.add_argument("--sintetico-dir", default=None,
                   help="Pasta do modo sintetico (padrao: %%TEMP%%\\tracking_"
                        "mao_sintetico; NUNCA a pasta da tese)")
    return p


# --------------------------------------------------------------------------
# Preflight: falhar cedo e com mensagem util
# --------------------------------------------------------------------------
@dataclass
class Contexto:
    tracker: object
    alignment: object
    aux_index: int
    cap: object
    view: object
    kinect_matrix: np.ndarray
    kinect_dist: np.ndarray
    kinect_intrinsics_nota: str
    aux_nome: str
    aux_resolucao: tuple
    calibracao: dict
    espelhamento: str
    versoes: dict


_PROBLEMAS: list[str] = []


def _parea_maos(maos_k, maos_a, modo="ambos"):
    """Casa as maos do Kinect com as da auxiliar. Devolve (pares, sobras_k, sobras_a).

    Cada par e' (lado, mao_k, mao_a, modo). Motivo de existir o modo 'unico':
    o pipeline de producao (`AuxStereoSource`) casa as maos PELO LADO, e quando
    o rotulo do MediaPipe sai trocado em uma das cameras (o bug de lateralidade
    de 05/10/2026, arquivo `aux_handedness_flip.json`) ele simplesmente nao
    casa nada e o multi-view fica mudo -- foi o que aconteceu no primeiro teste
    desta ferramenta na bancada (Kinect disse 'right', auxiliar disse 'left' na
    MESMA mao). Com uma mao por camera nao ha' ambiguidade geometrica, entao o
    par unico e' seguro e o modo fica registrado no CSV.
    """
    pares = []
    usados_k, usados_a = set(), set()
    if modo in ("lado", "ambos"):
        for lado in ("right", "left"):
            ik = next((i for i, m in enumerate(maos_k)
                       if m.get("side") == lado and i not in usados_k), None)
            ia = next((i for i, m in enumerate(maos_a)
                       if m.get("side") == lado and i not in usados_a), None)
            if ik is not None and ia is not None:
                pares.append((lado, maos_k[ik], maos_a[ia], "lado"))
                usados_k.add(ik)
                usados_a.add(ia)
    if (modo in ("unico", "ambos") and len(maos_k) == 1 and len(maos_a) == 1
            and 0 not in usados_k and 0 not in usados_a):
        lado = maos_k[0].get("side") or maos_a[0].get("side") or "right"
        pares.append((lado, maos_k[0], maos_a[0], "unico"))
        usados_k.add(0)
        usados_a.add(0)
    sobras_k = [m for i, m in enumerate(maos_k) if i not in usados_k]
    sobras_a = [m for i, m in enumerate(maos_a) if i not in usados_a]
    return pares, sobras_k, sobras_a


def _falha(mensagem: str):
    _PROBLEMAS.append(mensagem)


def _abre_camera(indice: int, largura: int, altura: int, trava: bool):
    """Abre a auxiliar em DSHOW e cai para MSMF; devolve (cap, frame) ou None.

    Mesma ordem do tool de groundtruth (DSHOW e' o backend da calibracao e abre
    em ~1 s).
    """
    import cv2
    from camera_uvc import define_modo, le_fourcc

    for backend, nome in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")):
        cap = None
        try:
            cap = cv2.VideoCapture(indice, backend)
            if not cap.isOpened():
                cap.release()
                continue
            define_modo(cap, largura, altura, mjpeg=True)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if trava:
                try:
                    from camera_uvc import trava_exposicao
                    trava_exposicao(cap, rotulo=f"aux {indice}")
                except Exception as exc:            # noqa: BLE001
                    print(f"[aviso] trava de exposicao falhou: {exc}")
            for _ in range(5):                      # descarta o buffer velho
                cap.read()
            ok, quadro = cap.read()
            if ok and quadro is not None and getattr(quadro, "size", 0):
                print(f"[ok] camera auxiliar {indice} via {nome}: "
                      f"{quadro.shape[1]}x{quadro.shape[0]} "
                      f"({le_fourcc(cap)})")
                return cap, quadro
            cap.release()
        except Exception as exc:                    # noqa: BLE001
            if cap is not None:
                cap.release()
            print(f"[aviso] camera {indice} via {nome}: {type(exc).__name__}: {exc}")
    return None


def prepara_contexto(args) -> Contexto:
    """Abre Kinect + auxiliar e CONFERE a calibracao. Falha com SystemExit."""
    global _PROBLEMAS
    _PROBLEMAS = []
    try:
        import cv2
    except ImportError:
        raise SystemExit(
            "OpenCV nao esta' disponivel neste Python.\n"
            "Use o `.venv` do projeto: .venv\\Scripts\\python.exe "
            "tools\\compara_tracking_mao.py ...")
    try:
        import mediapipe as mp
    except ImportError:
        raise SystemExit(
            "MediaPipe nao esta' disponivel neste Python.\n"
            "Use o `.venv` do projeto: .venv\\Scripts\\python.exe "
            "tools\\compara_tracking_mao.py ...")

    import kinect_compat
    kinect_compat.aplicar()
    try:
        import kinect_imu_groundtruth as gt
    except Exception as exc:                        # noqa: BLE001
        raise SystemExit(
            f"Nao consegui importar kinect_imu_groundtruth: {type(exc).__name__}: "
            f"{exc}\n"
            f"  compatibilidade do pykinect2: {kinect_compat.motivo()}\n"
            "  diagnostico: .venv\\Scripts\\python.exe "
            "tools\\diagnostico_cameras.py")

    # ---- camera auxiliar: indice e calibracao ---------------------------
    indices = tuple(int(i) for i in gt.indices_auxiliares_automaticos())
    if args.aux_index is not None:
        aux_index = int(args.aux_index)
        indices = (aux_index,)
    elif indices:
        aux_index = int(indices[0])
    else:
        _falha(
            "Nenhuma camera auxiliar EXTERNA foi enumerada pelo Windows.\n"
            "    Sem auxiliar nao existe multi-view para comparar.\n"
            "    Confira cabo/hub USB (extensor sem fonte derruba a camera) e "
            "rode:\n"
            "    .venv\\Scripts\\python.exe tools\\diagnostico_cameras.py\n"
            "    Se voce sabe o indice da camera, passe --aux-index N.")
        _mensagem_abortar()

    try:
        alignment = gt.CameraAlignment(aux_indices=tuple(sorted(set(indices + (aux_index,)))))
    except Exception as exc:                        # noqa: BLE001
        raise SystemExit(
            f"CameraAlignment falhou ({type(exc).__name__}: {exc}).\n"
            "Sem calibracao stereo nao ha' multi-view. Rode "
            "tools\\diagnostico_cameras.py para ver o estado das cameras.")

    estado, motivo = alignment.stereo_status(aux_index)
    calib_model = (alignment.aux_models or {}).get(aux_index) or {}
    arquivo = gt.aux_stereo_file(aux_index)
    calibracao = {
        "arquivo": arquivo.name if arquivo is not None else "(ausente)",
        "rms": calib_model.get("rms"),
        "tabuleiro": (f"{int(calib_model['board_size'][0])}x"
                      f"{int(calib_model['board_size'][1])}"
                      if calib_model.get("board_size") else "?"),
        "quadrado_mm": (round(float(calib_model["square_size_m"]) * 1000.0, 2)
                        if calib_model.get("square_size_m") else None),
        "estado": estado,
        "motivo": motivo,
        "aux_size": (list(calib_model.get("auxiliary_size") or []) or None),
        "kinect_size": (list(calib_model.get("kinect_size") or []) or None),
    }
    if not alignment.aux_usable(aux_index):
        _falha(
            f"A camera auxiliar {aux_index} NAO tem calibracao stereo "
            f"utilizavel ({estado}: {motivo}).\n"
            f"    arquivo esperado: "
            f"{arquivo if arquivo is not None else 'stereo_calibration_aux%d.npz (ausente)' % aux_index}"
            "\n"
            "    O multi-view precisa de intrinsecas + rotacao/translacao do par "
            "(Kinect, auxiliar).\n"
            "    Calibre com stereo_calibration.py (tabuleiro "
            f"{gt.CHECKERBOARD_SIZE[0]}x{gt.CHECKERBOARD_SIZE[1]} cantos, "
            f"{gt.SQUARE_SIZE_M*1000:.0f} mm) e confira com "
            "tools\\diagnostico_cameras.py.")
    elif str(estado).startswith("rejeitada"):
        # `aux_usable` olha so' o RMS; a placa/quadrado gravados no npz tambem
        # importam. Com o tabuleiro trocado, o RMS pode ser otimo e a geometria
        # NAO ser do par de agora -> aviso alto (o teste continua, mas o
        # relatorio registra o estado).
        print(f"[AVISO] calibracao da auxiliar {aux_index} esta' REJEITADA por "
              f"compatibilidade ({motivo}).\n"
              "        O RMS pode parecer bom e a geometria nao ser a do par "
              "atual: RECALIBRE antes de confiar nas posicoes.",
              flush=True)
        calibracao["aviso"] = f"rejeitada: {motivo}"

    # ---- tamanho calibrado da auxiliar ---------------------------------
    cal_w, cal_h = (calibracao["aux_size"] or [1280, 720])[:2]
    if _PROBLEMAS:
        _mensagem_abortar()

    aberto = _abre_camera(aux_index, int(cal_w), int(cal_h), args.trava_exposicao)
    if aberto is None:
        nomes = []
        try:
            import camera_names
            nomes = camera_names.resumo()
        except Exception:                           # noqa: BLE001
            pass
        _falha(
            f"Nao consegui abrir a camera auxiliar de indice {aux_index} "
            "(DSHOW nem MSMF).\n"
            + ("    cameras pelo nome: " + " | ".join(nomes) + "\n" if nomes else "")
            + "    Rode: .venv\\Scripts\\python.exe tools\\diagnostico_cameras.py --salvar\n"
            "    (ele mostra qual indice e' qual webcam e se a camera entrega frame)")
        _mensagem_abortar()
    cap, quadro = aberto
    aux_resolucao = (quadro.shape[1], quadro.shape[0])

    # ---- Kinect ---------------------------------------------------------
    try:
        tracker = gt.KinectHandTracker(aux_indices=(aux_index,))
    except Exception as exc:                        # noqa: BLE001
        cap.release()
        _falha(
            f"Kinect nao abriu: {type(exc).__name__}: {exc}\n"
            "    confira o Kinect for Windows SDK 2.0, o cabo e a fonte; rode\n"
            "    .venv\\Scripts\\python.exe tools\\diagnostico_cameras.py\n"
            "    e, se o RGB aparece e o depth nao, tools\\checklist_kinect_mudo.py")
        _mensagem_abortar()

    inicio = time.monotonic()
    cor_ok = dep_ok = False
    while time.monotonic() - inicio < 8.0:
        tracker.frames()
        cor_ok = tracker.last_color is not None
        dep_ok = tracker.last_depth is not None
        if cor_ok and dep_ok:
            break
        time.sleep(0.05)
    if not cor_ok or not dep_ok:
        tracker.close()
        cap.release()
        _falha(
            "O Kinect abriu mas nao entregou "
            + ("RGB" if not cor_ok else "")
            + (" e " if not cor_ok and not dep_ok else "")
            + ("depth" if not dep_ok else "")
            + " em 8 s.\n"
            "    depth morto em silencio e' o modo de falha conhecido: rode\n"
            "    .venv\\Scripts\\python.exe tools\\checklist_kinect_mudo.py\n"
            "    .venv\\Scripts\\python.exe tools\\pos_reinicio_kinect.py")
        _mensagem_abortar()

    # ---- intrinsecas do Kinect para as projecoes ------------------------
    kinect_matrix = alignment.kinect_matrix
    kinect_dist = alignment.kinect_dist
    nota = "calibracao stereo (kinect_matrix congelado)"
    if kinect_matrix is None or kinect_dist is None:
        kinect_matrix = gt.KINECT_V2_INTRINSICS_1080P.copy()
        kinect_dist = np.zeros(5, np.float64)
        nota = "nominais do Kinect v2 (npz sem intrinsecas do Kinect)"
    kinect_matrix = np.asarray(kinect_matrix, np.float64)
    kinect_dist = np.asarray(kinect_dist, np.float64)
    kw, kh = int(tracker.color_width), int(tracker.color_height)
    if calibracao["kinect_size"]:
        kcw, kch = calibracao["kinect_size"][:2]
        if (int(kcw), int(kch)) != (kw, kh):
            escala = (kw / float(kcw), kh / float(kch))
            kinect_matrix = kinect_matrix.copy()
            kinect_matrix[0, 0] *= escala[0]
            kinect_matrix[0, 2] *= escala[0]
            kinect_matrix[1, 1] *= escala[1]
            kinect_matrix[1, 2] *= escala[1]
            nota = (f"{nota}; matriz ESCALADA de {kcw}x{kch} para {kw}x{kh} "
                    "(calibracao e runtime com resolucoes diferentes)")

    try:
        import camera_names
        aux_nome = camera_names.nome_da_camera(aux_index) or "?"
    except Exception:                               # noqa: BLE001
        aux_nome = "?"

    try:
        view = alignment.active_view() or alignment.aux_view(aux_index)
    except Exception:                               # noqa: BLE001
        view = alignment.aux_view(aux_index)
    if view is None or not view.usable:
        tracker.close()
        cap.release()
        _falha(
            "A vista stereo da camera auxiliar nao ficou utilizavel "
            f"(aux {aux_index}).\n"
            f"    estado: {estado} ({motivo})")

    versoes = {
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "mediapipe": mp.__version__,
        "numpy": np.__version__,
    }
    espelhamento = (
        "AUXILIARY_MIRROR_HORIZONTAL=True: o frame da auxiliar e' espelhado "
        "ANTES da deteccao (kinect_imu_groundtruth.transform_auxiliary_frame), "
        "entao os pixels e a calibracao vivem na imagem espelhada"
        if gt.AUXILIARY_MIRROR_HORIZONTAL else
        "AUXILIARY_MIRROR_HORIZONTAL=False: frame da auxiliar usado como vem")
    if _PROBLEMAS:
        tracker.close()
        cap.release()
        _mensagem_abortar()
    return Contexto(tracker=tracker, alignment=alignment, aux_index=aux_index,
                    cap=cap, view=view, kinect_matrix=kinect_matrix,
                    kinect_dist=kinect_dist, kinect_intrinsics_nota=nota,
                    aux_nome=aux_nome, aux_resolucao=aux_resolucao,
                    calibracao=calibracao, espelhamento=espelhamento,
                    versoes=versoes)


def _mensagem_abortar():
    """Imprime TODOS os problemas de uma vez e sai com codigo 2."""
    print("\n=== NAO DA' PARA COMPARAR: falta pre-requisito ===", file=sys.stderr)
    for i, problema in enumerate(_PROBLEMAS, 1):
        print(f"\n[{i}] {problema}", file=sys.stderr)
    print("\nO script NAO roda 'meio funcionando': a comparacao Kinect x "
          "multi-view exige os dois caminhos e a calibracao stereo.\n",
          file=sys.stderr)
    raise SystemExit(2)


# --------------------------------------------------------------------------
# Utilidades de medicao por frame
# --------------------------------------------------------------------------
def _kinect_3d_21(tracker, landmarks, lado):
    """21 pontos 3D do depth do Kinect + diagnostico por landmark.

    Devolve (xyz (21,3) em eixos OpenCV com NaN onde falhou, info (21,) dicts).
    O sinal de Y e' invertido: o camera space do Kinect e' Y-para-CIMA e a
    triangulacao/OpenCV trabalha Y-para-BAIXO (mesma conversao de
    fusion/aux_vision.py).
    """
    xyz = np.full((21, 3), np.nan, np.float64)
    info = []
    for indice, landmark in enumerate(landmarks):
        diag = getattr(tracker, "diag", None) or {}
        antes = {k: int(diag.get(k, 0) or 0)
                 for k in ("mapeamento", "depth", "resgatados")}
        n_prof = len(diag.get("profundidades") or [])
        ponto = None
        try:
            ponto = tracker._landmark_camera_point(landmark, lado)
        except (AttributeError, IndexError, TypeError, ValueError):
            ponto = None
        diag = getattr(tracker, "diag", None) or {}
        depois = {k: int(diag.get(k, 0) or 0) for k in antes}
        profundidades = diag.get("profundidades") or []
        depth_m = (float(profundidades[-1]) / 1000.0
                   if len(profundidades) > n_prof else float("nan"))
        registro = {
            "map_fail": depois["mapeamento"] > antes["mapeamento"],
            "depth_fail": depois["depth"] > antes["depth"],
            "rescue": depois["resgatados"] > antes["resgatados"],
            "depth_m": depth_m,
        }
        if ponto is not None:
            arr = np.asarray(ponto, np.float64).reshape(3)
            if np.isfinite(arr).all() and arr[2] > 0.05:
                xyz[indice] = np.array([arr[0], -arr[1], arr[2]], np.float64)
        info.append(registro)
    return xyz, info


def _reprojeta(K, dist, xyz):
    """Projeta (N,3) com a matriz/dist do Kinect -> (N,2) px ou None."""
    import cv2
    pontos = np.asarray(xyz, np.float64).reshape(-1, 1, 3)
    if pontos.size == 0:
        return None
    try:
        proj, _ = cv2.projectPoints(pontos, np.zeros(3), np.zeros(3),
                                    np.asarray(K, np.float64),
                                    np.asarray(dist, np.float64))
    except (cv2.error, ValueError):
        return None
    return proj.reshape(-1, 2)


def _projeta_cor_sdk(tracker, xyz):
    """Projeta pontos 3D no RGB do Kinect usando o MAPPER do SDK.

    `xyz` vem em eixos OpenCV (Y para baixo); o mapper espera o camera space do
    SDK (Y para CIMA), entao o Y e' invertido de volta aqui. Usar o mapper (e
    nao uma matriz intrinseca) e' o que inclui a extrinseca depth->color do
    proprio sensor: medido nesta bancada, a projecao com intrinsecas nominais
    erra ~42 px de forma sistematica por causa da baseline entre os dois
    sensores, o que faria um teste de consistencia saudavel parecer defeito.
    Ida-e-volta medida pelo mapper: ~1,5 px de mediana.
    """
    from pykinect2 import PyKinectV2
    pontos = np.asarray(xyz, np.float64).reshape(-1, 3)
    saida = np.full((len(pontos), 2), np.nan)
    for i, p in enumerate(pontos):
        if not np.isfinite(p).all():
            continue
        camera = PyKinectV2._CameraSpacePoint()
        camera.x = float(p[0])
        camera.y = float(-p[1])
        camera.z = float(p[2])
        try:
            projetado = tracker.mapper.MapCameraPointToColorSpace(camera)
            if projetado is None:
                continue
            x, y = float(projetado.x), float(projetado.y)
        except (AttributeError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            saida[i] = (x, y)
    return saida


def _conf_mv(reproj_px, gate_px):
    """Confianca do multi-view a partir do erro de reprojecao (documentada).

    `conf = clip(1 - reproj/gate, 0, 1)`: 0 no gate e 1 em erro zero. E' a
    mesma informacao que o gate usa, explicitada como numero para o CSV --
    nao e' uma probabilidade calibrada e nao deve ser lida como tal.
    """
    if reproj_px is None:
        return 0.0
    try:
        valor = float(reproj_px)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(valor) or gate_px <= 0:
        return 0.0
    return float(np.clip(1.0 - valor / gate_px, 0.0, 1.0))


# --------------------------------------------------------------------------
# Desenho
# --------------------------------------------------------------------------
CORES = {
    "kinect": (0, 255, 0),        # verde: MediaPipe de cada camera
    "mv": (0, 255, 255),          # amarelo: multi-view no Kinect
    "mv_aux": (255, 255, 0),      # ciano: multi-view na auxiliar
    "depth_aux": (255, 0, 255),   # magenta: depth do Kinect visto na auxiliar
    "dorso": (0, 0, 255),         # vermelho: landmark 9
    "texto": (255, 255, 255),
    "alerta": (0, 165, 255),
    "erro": (0, 0, 255),
}


@dataclass
class Visual:
    """Estado de desenho/monitor (nao entra no CSV, so' na tela)."""

    mostrar_depth: bool = True
    fps: float = 0.0
    frames: int = 0
    det_k: int = 0
    det_a: int = 0
    mv_frames: int = 0
    quedas: int = 0
    reproj_kin: deque = field(default_factory=lambda: deque(maxlen=200))
    reproj_aux: deque = field(default_factory=lambda: deque(maxlen=200))
    d9_mm: deque = field(default_factory=lambda: deque(maxlen=60))
    kd_aux: deque = field(default_factory=lambda: deque(maxlen=200))
    lat_iguais: int = 0
    lat_trocados: int = 0
    pos_k: dict = field(default_factory=lambda: {"right": deque(maxlen=15),
                                                  "left": deque(maxlen=15)})
    pos_mv: dict = field(default_factory=lambda: {"right": deque(maxlen=15),
                                                   "left": deque(maxlen=15)})


def _pinta_mao(img, pts, cor, cor_linha, raio=4, esp=1, mx=0, my=0):
    """Desenha as conexoes + os 21 nos de UMA mao em `img`."""
    import cv2
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    pts = pts + np.array([mx, my], np.float64)
    for a, b in _conexoes():
        if a < len(pts) and b < len(pts):
            pa, pb = pts[a], pts[b]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                cv2.line(img, (int(round(pa[0])), int(round(pa[1]))),
                         (int(round(pb[0])), int(round(pb[1]))),
                         cor_linha, esp, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if not np.isfinite(p).all():
            continue
        centro = (int(round(p[0])), int(round(p[1])))
        cv2.circle(img, centro, raio, cor, -1, cv2.LINE_AA)
        if i == IDX_DORSO:
            cv2.circle(img, centro, raio + 4, CORES["dorso"], 2, cv2.LINE_AA)


def _conexoes():
    try:
        import kinect_imu_groundtruth as gt
        return gt.HAND_CONNECTIONS
    except Exception:                               # noqa: BLE001
        return []


def _escala(img, largura, altura):
    """Fatores de escala (sx, sy) de `img` para o painel."""
    h, w = img.shape[:2]
    return largura / max(w, 1), altura / max(h, 1)


def _texto(img, texto, x, y, cor=None, escala=0.55, esp=1):
    import cv2
    cv2.putText(img, str(texto), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                escala, cor or CORES["texto"], esp, cv2.LINE_AA)


def _painel_3d(img, area, dados, lado=None):
    """Vista X-Z (planta): depth (magenta) x multi-view (ciano)."""
    import cv2
    x0, y0, largura, altura = area
    cv2.rectangle(img, (x0, y0), (x0 + largura, y0 + altura), (45, 45, 45), 1)
    titulo = "Vista 3D (X-Z, eixos da camera do Kinect)"
    if lado:
        titulo += f" - {lado}"
    _texto(img, titulo, x0 + 10, y0 + 22, CORES["alerta"], 0.6)
    _texto(img, "Z (m) para cima | X (m) para a direita | so' os nos visiveis",
           x0 + 10, y0 + 44, (170, 170, 170), 0.45)
    # Limites: em volta do dorso multi-view (ou do depth) com 0,45 m de margem.
    referencia = None
    for metodo in ("mv", "k"):
        pontos = dados.get(metodo)
        if pontos is None:
            continue
        validos = np.asarray(pontos, float)[np.isfinite(pontos).all(axis=1)]
        if len(validos):
            referencia = (validos[IDX_DORSO] if len(validos) > IDX_DORSO
                          else validos[0])
            break
    if referencia is None:
        cx, cz = 0.0, 1.2
    else:
        cx, cz = float(referencia[0]), float(referencia[2])
    # Escala de perto: a mao ocupa ~0,2 m; com +-0,45 m os 21 nos ficavam
    # espremidos num canto e a vista 3D nao servia para nada.
    xmin, xmax = cx - 0.28, cx + 0.28
    zmin, zmax = cz - 0.28, cz + 0.28
    passo_grade = 0.1

    esq, dir_, topo, base = 56, 20, 76, 34
    largura_util = max(1, largura - esq - dir_)
    altura_util = max(1, altura - topo - base)

    def mapeia(p):
        px = x0 + esq + (float(p[0]) - xmin) / (xmax - xmin) * largura_util
        py = y0 + altura - base - (float(p[2]) - zmin) / (zmax - zmin) * altura_util
        return int(round(px)), int(round(py))

    # Grade com rotulos.
    for passo in np.arange(math.ceil(xmin / passo_grade) * passo_grade, xmax,
                           passo_grade):
        px, _ = mapeia((passo, 0, zmin))
        cv2.line(img, (px, y0 + topo - 6), (px, y0 + altura - base + 4),
                 (40, 40, 40), 1)
        _texto(img, f"{passo:.1f}", px - 13, y0 + altura - 12, (135, 135, 135),
               0.42)
    for passo in np.arange(math.ceil(zmin / passo_grade) * passo_grade, zmax,
                           passo_grade):
        _, py = mapeia((xmin, 0, passo))
        cv2.line(img, (x0 + esq - 4, py), (x0 + largura - dir_, py),
                 (40, 40, 40), 1)
        _texto(img, f"{passo:.1f}", x0 + 10, py + 5, (135, 135, 135), 0.42)

    for metodo, cor in (("k", CORES["depth_aux"]), ("mv", CORES["mv_aux"])):
        pontos = dados.get(metodo)
        if pontos is None:
            continue
        for i, p in enumerate(np.asarray(pontos, np.float64).reshape(-1, 3)):
            if not np.isfinite(p).all():
                continue
            centro = mapeia(p)
            if i == IDX_DORSO:
                cv2.circle(img, centro, 8, CORES["dorso"], -1, cv2.LINE_AA)
                _texto(img, "9", centro[0] + 10, centro[1] - 7, CORES["dorso"],
                       0.55)
            else:
                cv2.circle(img, centro, 4, cor, -1, cv2.LINE_AA)
    _texto(img, "o depth Kinect   o multi-view   * dorso (9)", x0 + 10,
           y0 + 66, (170, 170, 170), 0.45)
    return img


def _painel_status(img, area, linhas):
    import cv2
    x0, y0, largura, altura = area
    cv2.rectangle(img, (x0, y0), (x0 + largura, y0 + altura), (45, 45, 45), 1)
    _texto(img, "CONTADORES DA SESSAO", x0 + 10, y0 + 22, CORES["alerta"], 0.6)
    y = y0 + 48
    for linha in linhas:
        cor = CORES["texto"]
        if isinstance(linha, tuple):
            linha, cor = linha
        _texto(img, linha, x0 + 10, y, cor, 0.46)
        y += 21
        if y > y0 + altura - 12:
            break
    return img


def monta_canvas(frame_k, frame_a, resultado, visual, fase, teclas,
                 resolucao_k=None):
    """Monta o canvas 1600x900 com os quatro paineis."""
    import cv2
    largura_painel, altura_painel = 800, 450
    canvas = np.zeros((altura_painel * 2, largura_painel * 2, 3), np.uint8)

    # --- Kinect ----------------------------------------------------------
    if frame_k is not None:
        painel_k = cv2.resize(frame_k, (largura_painel, altura_painel))
        sx, sy = _escala(frame_k, largura_painel, altura_painel)
    else:
        painel_k = np.zeros((altura_painel, largura_painel, 3), np.uint8)
        sx = sy = 1.0
    titulo_k = "KINECT RGB"
    if resolucao_k:
        titulo_k += f" ({int(resolucao_k[0])}x{int(resolucao_k[1])})"
    _texto(painel_k, titulo_k, 12, 26, CORES["alerta"], 0.6)
    for chave, dados in (resultado or {}).items():
        lado = dados.get("lado", str(chave))
        if dados.get("kin_px") is not None:
            pts = np.asarray(dados["kin_px"], np.float64).copy()
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            _pinta_mao(painel_k, pts, CORES["kinect"], (0, 170, 0), 4, 1)
            _texto(painel_k, f"{lado}", 12, 50, CORES["kinect"], 0.55)
        if dados.get("mv_kin_px") is not None:
            pts = np.asarray(dados["mv_kin_px"], np.float64).copy()
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            _pinta_mao(painel_k, pts, CORES["mv"], (0, 170, 170), 3, 1)
        if dados.get("erro_mv_kin") is not None:
            _texto(painel_k, f"reproj MV={dados['erro_mv_kin']:.1f}px",
                   12, altura_painel - 14, CORES["mv"], 0.5)
        if dados.get("d9_mm") is not None:
            _texto(painel_k, f"d9 kin x MV={dados['d9_mm']:.0f}mm",
                   300, altura_painel - 14, CORES["dorso"], 0.5)
    canvas[0:altura_painel, 0:largura_painel] = painel_k

    # --- Auxiliar --------------------------------------------------------
    if frame_a is not None:
        painel_a = cv2.resize(frame_a, (largura_painel, altura_painel))
        sxa, sya = _escala(frame_a, largura_painel, altura_painel)
    else:
        painel_a = np.zeros((altura_painel, largura_painel, 3), np.uint8)
        sxa = sya = 1.0
    _texto(painel_a, "AUXILIAR (espelhada)", 12, 26, CORES["alerta"], 0.6)
    for chave, dados in (resultado or {}).items():
        if dados.get("aux_px") is not None:
            pts = np.asarray(dados["aux_px"], np.float64).copy()
            pts[:, 0] *= sxa
            pts[:, 1] *= sya
            _pinta_mao(painel_a, pts, CORES["kinect"], (0, 170, 0), 4, 1)
        if dados.get("mv_aux_px") is not None:
            pts = np.asarray(dados["mv_aux_px"], np.float64).copy()
            pts[:, 0] *= sxa
            pts[:, 1] *= sya
            _pinta_mao(painel_a, pts, CORES["mv_aux"], (170, 170, 0), 3, 1)
        if visual.mostrar_depth and dados.get("kd_aux_px") is not None:
            pts = np.asarray(dados["kd_aux_px"], np.float64).copy()
            pts[:, 0] *= sxa
            pts[:, 1] *= sya
            _pinta_mao(painel_a, pts, CORES["depth_aux"], (140, 0, 140), 3, 1)
        if dados.get("erro_mv_aux") is not None:
            _texto(painel_a, f"reproj aux={dados['erro_mv_aux']:.1f}px",
                   12, altura_painel - 14, CORES["mv_aux"], 0.5)
    _texto(painel_a, "verde=MediaPipe ciano=multi-view magenta=depth",
           12, altura_painel - 34, (170, 170, 170), 0.45)
    canvas[0:altura_painel, largura_painel:2 * largura_painel] = painel_a

    # --- 3D + status -----------------------------------------------------
    area_3d = (0, altura_painel, largura_painel, altura_painel)
    primeiro_chave, primeiro_dados = next(iter((resultado or {}).items()),
                                          (None, {}))
    _painel_3d(canvas, area_3d, {"k": primeiro_dados.get("k_xyz"),
                                 "mv": primeiro_dados.get("mv_xyz")},
               primeiro_dados.get("lado", primeiro_chave))

    linhas = [
        f"frames gravados: {visual.frames}   fps: {visual.fps:.1f}",
        f"deteccao Kinect: {_razao(visual.det_k, visual.frames)}   "
        f"auxiliar: {_razao(visual.det_a, visual.frames)}",
        f"multi-view valido: {_razao(visual.mv_frames, visual.frames)}   "
        f"quedas totais de depth: {visual.quedas}",
        f"reproj MV: Kinect {_mediana(visual.reproj_kin):.1f}px | "
        f"aux {_mediana(visual.reproj_aux):.1f}px",
        f"depth 3D na aux: {_mediana(visual.kd_aux):.1f}px "
        "(grande = correspondencia enviesada)",
        f"d9 Kinect x MV: {_mediana(visual.d9_mm):.0f} mm",
        f"jitter (ult. 15): Kinect {_jitter(visual.pos_k):.1f} mm | "
        f"MV {_jitter(visual.pos_mv):.1f} mm",
        f"lateralidade igual: "
        f"{_razao(visual.lat_iguais, visual.lat_iguais + visual.lat_trocados)}"
        f" ({visual.lat_trocados} trocados)",
        f"fase: {fase or 'auto (velocidade)'}",
        "",
        teclas,
    ]
    _painel_status(canvas, (largura_painel, altura_painel, largura_painel,
                            altura_painel), linhas)
    return canvas


def _razao(numerador, denominador) -> str:
    return f"{100.0 * numerador / denominador:.0f}%" if denominador else "--"


def _mediana(fila) -> float:
    if not fila:
        return float("nan")
    return float(np.median(list(fila)))


def _jitter(posicoes: dict) -> float:
    """Desvio-padrao 3D (mm) das ultimas posicoes do dorso, por lado.

    Serve so' para o operador ver o jitter CAIR quando ele para a mao -- a
    metrica do relatorio e' outra (calculada sobre os segmentos estaticos).
    """
    valores = []
    for fila in posicoes.values():
        if len(fila) >= 5:
            arr = np.asarray(list(fila), np.float64)
            valores.append(float(np.linalg.norm(np.std(arr, axis=0))))
    if not valores:
        return float("nan")
    return float(np.median(valores) * 1000.0)


# --------------------------------------------------------------------------
# Sessao de aquisicao
# --------------------------------------------------------------------------
def _comando_reproduzivel() -> str:
    """Comando para o usuario repetir a sessao (com o Python do .venv).

    `sys.executable` e' `...\\.venv\\Scripts\\python.exe`; mostrar o caminho
    absoluto nao ajuda, mas mostrar so' `python.exe` faz o usuario rodar o
    Python do Windows Store e quebrar por falta de cv2/mediapipe.
    """
    executavel = Path(sys.executable)
    if ".venv" in executavel.parts:
        prefixo = r".venv\Scripts\python.exe"
    else:
        prefixo = str(executavel)
    argumentos = [a for a in sys.argv[1:]]
    return " ".join([prefixo] + argumentos)


def _meta_sessao(ctx, args, origem, extra=None):
    meta = {
        "schema": 1,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "origem": origem,
        "comando": _comando_reproduzivel(),
        "aux_index": ctx.aux_index,
        "aux_camera_nome": ctx.aux_nome,
        "aux_resolucao_usada": list(ctx.aux_resolucao),
        "kinect_resolucao": [int(ctx.tracker.color_width),
                             int(ctx.tracker.color_height)],
        "espelhamento": ctx.espelhamento,
        "calibracao": ctx.calibracao,
        "kinect_intrinsics_nota": ctx.kinect_intrinsics_nota,
        "versoes": ctx.versoes,
        "duracao_alvo_s": args.duracao,
        "reproj_max_px": args.reproj_max,
        "salto_mm": args.salto_mm,
        "teclas": "ESC/Q sai | E estatico | M movimento | R repouso | D depth",
        "referencial": "camera de cor do Kinect, metros, eixos OpenCV "
                       "(X direita, Y para baixo, Z para frente)",
        "frame_3d": _nota_frame_3d(),
        "privacidade": "nenhum video/frame e' gravado; so' landmarks e 3D",
    }
    if extra:
        meta.update(extra)
    return meta


def _grava_frame_e_landmarks(esc: EscritorCSV, seq, t_unix, t_mono, origem,
                             aux_index, lado, dados, fase):
    """Escreve as 21 linhas de landmark + 1 linha de frame."""
    n_ok_k = int(np.isfinite(dados["k_xyz"]).all(axis=1).sum())
    n_ok_mv = int(np.asarray(dados["mv_mask"]).sum())
    mv_valid = bool(n_ok_mv >= MIN_LANDMARKS_3D and dados["ok_k"]
                    and dados["ok_aux"])
    k_self_med = dados.get("k_self_med", float("nan"))
    d9_mm = dados.get("d9_mm", float("nan"))
    for i in range(21):
        k_xyz = dados["k_xyz"][i]
        mv_xyz = dados["mv_xyz"][i]
        info = dados["k_info"][i]
        k_valid = bool(np.isfinite(k_xyz).all())
        mv_valido = bool(dados["mv_mask"][i])
        reproj_kin = dados["mv_reproj_kin"][i] if dados.get("mv_reproj_kin") is not None else float("nan")
        reproj_aux = dados["mv_reproj_aux"][i] if dados.get("mv_reproj_aux") is not None else float("nan")
        esc.landmark({
            "seq": seq,
            "t_unix": _n(t_unix, 6),
            "t_mono": _n(t_mono, 6),
            "origem": origem,
            "hand": lado,
            "landmark_id": i,
            "is_dorso": _i(i == IDX_DORSO),
            "k_valid": _i(k_valid),
            "k_conf": _n(0.5 if info["rescue"] else (1.0 if k_valid else 0.0), 3),
            "k_map_fail": _i(info["map_fail"]),
            "k_depth_fail": _i(info["depth_fail"]),
            "k_depth_rescue": _i(info["rescue"]),
            "k_depth_m": _n(info["depth_m"], 6),
            "k_x_m": _n(k_xyz[0]), "k_y_m": _n(k_xyz[1]), "k_z_m": _n(k_xyz[2]),
            "k_reproj_px": _n(dados["k_self_reproj"][i]
                              if dados.get("k_self_reproj") is not None
                              else float("nan"), 4),
            "mv_valid": _i(mv_valido),
            "mv_implausivel": _i(bool(dados["mv_triangulou"][i])
                                 and not mv_valido),
            "mv_conf": _n(dados["mv_conf"][i], 4),
            "mv_x_m": _n(mv_xyz[0]), "mv_y_m": _n(mv_xyz[1]),
            "mv_z_m": _n(mv_xyz[2]),
            "mv_reproj_kin_px": _n(reproj_kin, 4),
            "mv_reproj_aux_px": _n(reproj_aux, 4),
            "kin_px_x": _n(dados["kin_px"][i][0], 2)
            if dados.get("kin_px") is not None else "",
            "kin_px_y": _n(dados["kin_px"][i][1], 2)
            if dados.get("kin_px") is not None else "",
            "aux_px_x": _n(dados["aux_px"][i][0], 2)
            if dados.get("aux_px") is not None else "",
            "aux_px_y": _n(dados["aux_px"][i][1], 2)
            if dados.get("aux_px") is not None else "",
            "kd_aux_err_px": _n(dados["kd_aux_err"][i]
                                if dados.get("kd_aux_err") is not None
                                else float("nan"), 2),
        })
    palm_k = dados["k_xyz"][IDX_DORSO]
    palm_mv = dados["mv_xyz"][IDX_DORSO]
    esc.frame({
        "seq": seq,
        "t_unix": _n(t_unix, 6),
        "t_mono": _n(t_mono, 6),
        "origem": origem,
        "aux_index": aux_index,
        "hand": lado,
        "side": lado,
        "side_aux": dados.get("side_aux", ""),
        "n_maos_k": _n(dados.get("n_maos_k", 0), 0),
        "n_maos_a": _n(dados.get("n_maos_a", 0), 0),
        "pareamento": dados.get("pareamento", ""),
        "fase": fase,
        "ok_kinect_detect": _i(dados["ok_k"]),
        "ok_aux_detect": _i(dados["ok_aux"]),
        "mv_triangulou": _i(bool(dados.get("mv_triangulou") is not None
                                 and np.isfinite(dados["mv_xyz"]).any())),
        "mv_valid": _i(mv_valid),
        "n_landmarks_k": n_ok_k,
        "n_landmarks_mv": n_ok_mv,
        "palm9_k_valid": _i(np.isfinite(palm_k).all()),
        "palm9_k_x": _n(palm_k[0]), "palm9_k_y": _n(palm_k[1]),
        "palm9_k_z": _n(palm_k[2]),
        "palm9_mv_valid": _i(np.isfinite(palm_mv).all()),
        "palm9_mv_x": _n(palm_mv[0]), "palm9_mv_y": _n(palm_mv[1]),
        "palm9_mv_z": _n(palm_mv[2]),
        "reproj_mv_kin_med_px": _n(dados.get("erro_mv_kin"), 4),
        "reproj_mv_aux_med_px": _n(dados.get("erro_mv_aux"), 4),
        "kd_aux_err_med_px": _n(dados.get("kd_aux_err_med"), 3),
        "reproj_k_self_med_px": _n(k_self_med, 4),
        "map_fail": dados["n_map_fail"],
        "depth_fail": dados["n_depth_fail"],
        "depth_rescue": dados["n_rescue"],
        "sync_color_depth_ms": _n(getattr(dados["tracker"], "sync_color_depth_ms",
                                          None), 3),
        "d_kin_mv_dorso_mm": _n(d9_mm, 3),
        "aux_lag_ms": _n(dados["aux_lag_ms"], 3),
        "dt_s": _n(dados["dt_s"], 6),
        "fps_inst": _n(1.0 / dados["dt_s"] if dados["dt_s"] > 0 else float("nan"), 3),
        "t_kin_detect_ms": _n(dados["t_kin_detect_ms"], 3),
        "t_aux_detect_ms": _n(dados["t_aux_detect_ms"], 3),
        "t_tri_ms": _n(dados["t_tri_ms"], 3),
        "t_ctrl_ms": _n(dados["t_ctrl_ms"], 3),
        "note": _txt(dados.get("note", "")),
    })


def roda_sessao(ctx: Contexto, args, esc: EscritorCSV) -> dict:
    """Loop de aquisicao com janela + gravacao. Devolve estatisticas finais."""
    import cv2
    import kinect_imu_groundtruth as gt
    from fusion.aux_vision import _project_kinect_to_aux

    janela = "Comparacao tracking mao - Kinect | Auxiliar | 3D"
    if not args.sem_janela:
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(janela, 1600, 900)
        try:
            cv2.setWindowProperty(janela, cv2.WND_PROP_TOPMOST, 1)
        except Exception:                            # noqa: BLE001
            pass

    # Contagem regressiva: da' tempo de colocar a mao na frente das cameras.
    inicio_espera = time.monotonic()
    while time.monotonic() - inicio_espera < max(0.0, args.espera):
        restante = args.espera - (time.monotonic() - inicio_espera)
        ctx.tracker.frames()
        ctx.cap.grab()
        if not args.sem_janela:
            tela = np.zeros((420, 980, 3), np.uint8)
            _texto(tela, "Rastreamento da mao: Kinect x multi-view",
                   20, 60, CORES["alerta"], 0.9)
            _texto(tela, "A mao precisa aparecer NOS DOIS paineis (Kinect e auxiliar).",
                   20, 130, CORES["texto"], 0.65)
            _texto(tela, "Se so' um painel mostrar a mao, mova-a para o meio da cena.",
                   20, 170, CORES["texto"], 0.65)
            _texto(tela, "1) segure PARADA ~5 s (jitter)  2) mova devagar ~10 s",
                   20, 225, CORES["alerta"], 0.65)
            _texto(tela, f"gravacao em {restante:.1f} s  (ESC cancela)",
                   20, 320, CORES["dorso"], 0.9)
            _texto(tela, "teclas: E=estatico  M=movimento  R=repouso  D=depth",
                   20, 375, (160, 160, 160), 0.55)
            cv2.imshow(janela, tela)
            if (cv2.waitKey(30) & 0xFF) in (27, ord("q")):
                ctx.cap.release()
                ctx.tracker.close()
                cv2.destroyAllWindows()
                return {"cancelado": True}
        else:
            time.sleep(0.05)

    registro = {
        "frames": 0, "det_k": 0, "det_a": 0, "mv_frames": 0, "quedas": 0,
        "duracao_s": 0.0, "cancelado": False,
    }
    visual = Visual()
    fase = ""
    seq = 0
    t_inicio = time.perf_counter()
    t_frame_anterior = t_inicio
    ultimo_captura_color = -1.0
    try:
        while time.perf_counter() - t_inicio < max(0.5, args.duracao):
            t_frame = time.perf_counter()
            # 1) Kinect: so' processa quando CHEGOU RGB novo (senao a mesma
            #    imagem viraria varias linhas do CSV e a cobertura mentiria).
            resultado_frames = ctx.tracker.frames()
            captura_color = float(getattr(ctx.tracker, "_last_color_capture_time", 0) or 0)
            if captura_color and captura_color == ultimo_captura_color:
                ctx.cap.grab()
                if not args.sem_janela and (cv2.waitKey(1) & 0xFF) == 27:
                    break
                time.sleep(0.001)
                continue
            ultimo_captura_color = captura_color
            t_kin = time.perf_counter()
            if resultado_frames is None:
                time.sleep(0.01)
                continue
            color_bgra, depth = resultado_frames
            if color_bgra is None:
                time.sleep(0.01)
                continue

            # 2) Auxiliar: le o MAIS CEDO possivel depois do frame do Kinect
            #    (e' o que minimiza a diferenca de instante entre as vistas).
            aux_frame = None
            t_leitura_aux = time.perf_counter()
            ok, bruto = ctx.cap.read()
            if ok and bruto is not None:
                aux_frame = gt.transform_auxiliary_frame(bruto)
            # Idade do frame da auxiliar em relacao ao instante de CAPTURA do
            # RGB do Kinect (nao ao instante em que o laco o recebeu): e' a
            # diferenca temporal entre as duas vistas, que vira erro de
            # triangulacao quando a mao se move.
            if captura_color:
                aux_lag_ms = (t_leitura_aux - captura_color) * 1000.0
            else:
                aux_lag_ms = (t_leitura_aux - t_kin) * 1000.0

            # 3) MediaPipe nas duas cameras.
            t0 = time.perf_counter()
            detect = ctx.tracker.detect_hands_all(color_bgra, depth)
            t_kin_detect_ms = (time.perf_counter() - t0) * 1000.0
            if isinstance(detect, tuple) and len(detect) == 2:
                _, maos_k = detect
            else:
                maos_k = detect or []
            t0 = time.perf_counter()
            maos_a = []
            if aux_frame is not None:
                try:
                    maos_a = ctx.tracker.detect_auxiliary_hands_sides(
                        aux_frame, aux_index=ctx.aux_index)
                except Exception as exc:             # noqa: BLE001
                    maos_a = []
                    print(f"[aviso] deteccao na auxiliar falhou: {exc}")
            t_aux_detect_ms = (time.perf_counter() - t0) * 1000.0

            # 4) Vista stereo atualizada para o tamanho REAL do frame.
            view = None
            if aux_frame is not None:
                try:
                    ctx.alignment.set_active_aux(ctx.aux_index)
                    ctx.alignment.set_auxiliary_size(aux_frame)
                    view = ctx.alignment.active_view() or ctx.alignment.aux_view(
                        ctx.aux_index)
                except Exception as exc:             # noqa: BLE001
                    print(f"[aviso] vista stereo falhou: {exc}")
                    view = None

            maos_k = [m for m in maos_k if m.get("side")]
            maos_a = [m for m in maos_a if m.get("side")]
            n_maos_k = len(maos_k)
            n_maos_a = len(maos_a)
            if n_maos_k == 1 and n_maos_a == 1:
                if maos_k[0].get("side") == maos_a[0].get("side"):
                    visual.lat_iguais += 1
                else:
                    visual.lat_trocados += 1
            pares, sobras_k, sobras_a = _parea_maos(maos_k, maos_a,
                                                    args.pareamento)
            # Uma "linha" por mao: os pares (duas cameras) + as sobras (uma
            # camera so'). As sobras contam contra a cobertura do outro metodo,
            # que e' o comportamento certo.
            tarefas = [(lado, mk, ma, modo) for (lado, mk, ma, modo) in pares]
            tarefas += [(m.get("side"), m, None, "so_kinect") for m in sobras_k]
            tarefas += [(m.get("side"), None, m, "so_aux") for m in sobras_a]
            lados = [t[0] for t in tarefas]

            # 5) 3D de cada mao (depth do Kinect + triangulacao + reprojecoes).
            t0 = time.perf_counter()
            resultados = {}
            for indice, (lado, mao_k, mao_a, modo) in enumerate(tarefas):
                dados = _processa_lado(ctx, view, lado, mao_k, mao_a, aux_frame,
                                       _project_kinect_to_aux, args)
                dados["aux_lag_ms"] = aux_lag_ms
                dados["t_kin_detect_ms"] = t_kin_detect_ms
                dados["t_aux_detect_ms"] = t_aux_detect_ms
                dados["pareamento"] = modo
                dados["lado"] = lado
                dados["side_aux"] = str((mao_a or {}).get("side") or "")
                dados["n_maos_k"] = n_maos_k
                dados["n_maos_a"] = n_maos_a
                if modo == "unico":
                    dados["note"] = _txt(
                        dados.get("note", "") +
                        f" | pareado por mao unica (kinect={lado}, "
                        f"aux={dados['side_aux']}): rotulos de lateralidade "
                        "divergentes")
                resultados[indice] = dados
            t_tri_ms = (time.perf_counter() - t0) * 1000.0
            t_ctrl = (time.perf_counter() - t_frame) * 1000.0
            dt = t_frame - t_frame_anterior
            t_frame_anterior = t_frame

            # 6) Grava os CSV (com os tempos ja' medidos).
            if not tarefas:
                seq += 1
                esc.frame({
                    "seq": seq,
                    "t_unix": _n(time.time(), 6), "t_mono": _n(t_frame, 6),
                    "origem": "kinect_aux", "aux_index": ctx.aux_index,
                    "hand": "none", "side": "", "fase": fase,
                    "n_maos_k": _n(n_maos_k, 0), "n_maos_a": _n(n_maos_a, 0),
                    "ok_kinect_detect": "0", "ok_aux_detect": "0",
                    "mv_triangulou": "0", "mv_valid": "0",
                    "n_landmarks_k": "0", "n_landmarks_mv": "0",
                    "dt_s": _n(dt, 6), "t_ctrl_ms": _n(t_ctrl, 3),
                    "t_kin_detect_ms": _n(t_kin_detect_ms, 3),
                    "t_aux_detect_ms": _n(t_aux_detect_ms, 3),
                    "aux_lag_ms": _n(aux_lag_ms, 3),
                    "note": _txt(getattr(ctx.tracker, "last_hand_reason", "")),
                })
                registro["frames"] += 1
            for indice, (lado, mao_k, mao_a, _modo) in enumerate(tarefas):
                dados = resultados[indice]
                dados["dt_s"] = dt
                dados["t_tri_ms"] = t_tri_ms
                dados["t_ctrl_ms"] = t_ctrl
                seq += 1
                _grava_frame_e_landmarks(
                    esc, seq, time.time(), t_frame, "kinect_aux",
                    ctx.aux_index, lado, dados, fase)
                registro["frames"] += 1
                registro["det_k"] += int(dados["ok_k"])
                registro["det_a"] += int(dados["ok_aux"])
                n_ok_mv = int(np.asarray(dados["mv_mask"]).sum())
                registro["mv_frames"] += int(n_ok_mv >= MIN_LANDMARKS_3D
                                             and dados["ok_aux"] and dados["ok_k"])
                if dados["ok_k"] and n_ok_mv == 0 and \
                        int(np.isfinite(dados["k_xyz"]).all(axis=1).sum()) == 0:
                    registro["quedas"] += 1
                if dados.get("erro_mv_kin") is not None:
                    visual.reproj_kin.append(dados["erro_mv_kin"])
                if dados.get("erro_mv_aux") is not None:
                    visual.reproj_aux.append(dados["erro_mv_aux"])
                if dados.get("d9_mm") is not None:
                    visual.d9_mm.append(dados["d9_mm"])
                if dados.get("kd_aux_err_med") is not None and \
                        np.isfinite(dados.get("kd_aux_err_med", float("nan"))):
                    visual.kd_aux.append(dados["kd_aux_err_med"])
                palm_k = dados["k_xyz"][IDX_DORSO]
                palm_mv = dados["mv_xyz"][IDX_DORSO]
                if np.isfinite(palm_k).all():
                    visual.pos_k[lado].append(palm_k)
                if np.isfinite(palm_mv).all():
                    visual.pos_mv[lado].append(palm_mv)
            visual.frames = registro["frames"]
            visual.det_k = registro["det_k"]
            visual.det_a = registro["det_a"]
            visual.mv_frames = registro["mv_frames"]
            visual.quedas = registro["quedas"]
            if dt > 0:
                visual.fps = 1.0 / dt

            # 7) Janela.
            if not args.sem_janela:
                canvas = monta_canvas(
                    cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR),
                    aux_frame, resultados, visual, fase,
                    "ESC/Q sai | E estatico | M movimento | R repouso | D depth",
                    resolucao_k=(ctx.tracker.color_width,
                                 ctx.tracker.color_height))
                cv2.imshow(janela, canvas)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (27, ord("q")):
                    break
                if tecla == ord("e"):
                    fase = "estatico"
                elif tecla == ord("m"):
                    fase = "movimento"
                elif tecla == ord("r"):
                    fase = "repouso"
                elif tecla == ord("d"):
                    visual.mostrar_depth = not visual.mostrar_depth
        registro["duracao_s"] = time.perf_counter() - t_inicio
    except KeyboardInterrupt:
        print("\n[aviso] interrompido pelo usuario; CSV preservado.")
        registro["cancelado"] = True
    finally:
        try:
            ctx.cap.release()
        except Exception:                             # noqa: BLE001
            pass
        try:
            ctx.tracker.close()
        except Exception:                             # noqa: BLE001
            pass
        if not args.sem_janela:
            cv2.destroyAllWindows()
    return registro


def _processa_lado(ctx, view, lado, mao_k, mao_a, aux_frame, projeta_aux, args):
    """Todo o 3D de UM lado: depth do Kinect, triangulacao e reprojecoes."""
    entrada = {
        "tracker": ctx.tracker, "ok_k": mao_k is not None,
        "ok_aux": mao_a is not None, "kin_px": None, "aux_px": None,
        "k_xyz": np.full((21, 3), np.nan),
        "k_xyz_cor": np.full((21, 3), np.nan),
        "k_info": [{"map_fail": False, "depth_fail": False, "rescue": False,
                    "depth_m": float("nan")} for _ in range(21)],
        "mv_xyz": np.full((21, 3), np.nan), "mv_mask": np.zeros(21, bool),
        "mv_triangulou": np.zeros(21, bool), "mv_conf": np.zeros(21),
        "mv_reproj_kin": None, "mv_reproj_aux": None,
        "k_self_reproj": None, "k_self_med": float("nan"),
        "erro_mv_kin": None, "erro_mv_aux": None, "d9_mm": float("nan"),
        "kd_aux_err": None, "kd_aux_err_med": float("nan"),
        "n_map_fail": 0, "n_depth_fail": 0, "n_rescue": 0,
        "aux_lag_ms": float("nan"), "dt_s": 0.0,
        "t_kin_detect_ms": float("nan"), "t_aux_detect_ms": float("nan"),
        "t_tri_ms": float("nan"), "t_ctrl_ms": float("nan"), "note": "",
    }
    kw, kh = int(ctx.tracker.color_width), int(ctx.tracker.color_height)

    # --- Kinect depth ----------------------------------------------------
    if mao_k is not None:
        landmarks = mao_k.get("landmarks") or []
        if len(landmarks) == 21:
            xyz, info = _kinect_3d_21(ctx.tracker, landmarks, lado)
            entrada["k_xyz"] = xyz
            #: Frame da COR (compativel com a calibracao stereo das auxiliares).
            entrada["k_xyz_cor"] = _corrige_frame_cor(xyz)
            entrada["k_info"] = info
            entrada["kin_px"] = np.array(
                [(lm.x * kw, lm.y * kh) for lm in landmarks], np.float64)
            entrada["n_map_fail"] = int(sum(1 for r in info if r["map_fail"]))
            entrada["n_depth_fail"] = int(sum(1 for r in info if r["depth_fail"]))
            entrada["n_rescue"] = int(sum(1 for r in info if r["rescue"]))
            # Ida-e-volta color->depth->camera->color pelo MAPPER do SDK: mede
            # consistencia do mapeamento (nao exactidao) e denuncia convencao de
            # eixos errada (o Y errado daria centenas de px).
            proj = _projeta_cor_sdk(ctx.tracker, xyz)
            if proj is not None:
                dist = np.linalg.norm(proj - entrada["kin_px"], axis=1)
                entrada["k_self_reproj"] = dist
                entrada["k_self_med"] = metricas._nanmed(dist)
            entrada["note"] = "kinect ok"
        else:
            entrada["note"] = f"mediapipe kinect devolveu {len(landmarks)} nos"
    else:
        entrada["k_info"] = [{"map_fail": False, "depth_fail": False,
                              "rescue": False, "depth_m": float("nan")}
                            for _ in range(21)]
        entrada["note"] = "sem mao no kinect"

    # --- Multi-view ------------------------------------------------------
    if mao_a is not None and view is not None and entrada["kin_px"] is not None:
        landmarks_a = mao_a.get("landmarks") or []
        if len(landmarks_a) == 21:
            largura_a, altura_a = (view.size or (aux_frame.shape[1],
                                                 aux_frame.shape[0]))
            aux_px = np.array([(lm.x * largura_a, lm.y * altura_a)
                               for lm in landmarks_a], np.float64)
            entrada["aux_px"] = aux_px
            pontos, reproj = view.triangulate_with_kinect(aux_px,
                                                          entrada["kin_px"])
            if pontos is not None:
                pontos = np.asarray(pontos, np.float64).reshape(-1, 3)
                entraram = np.isfinite(pontos).all(axis=1)
                entrada["mv_triangulou"] = entraram
                gate = args.reproj_max
                valido = (entraram
                          & (pontos[:, 2] > 0.3) & (pontos[:, 2] < 4.0))
                if reproj is not None:
                    completo = np.isfinite(reproj) & (reproj <= gate)
                    valido &= completo
                    entrada["mv_reproj_kin"] = np.asarray(reproj, np.float64)
                    entrada["mv_conf"] = np.array(
                        [_conf_mv(r, gate) if v else 0.0
                         for r, v in zip(reproj, valido)], np.float64)
                entrada["mv_mask"] = valido
                entrada["mv_xyz"] = pontos
                # Projecoes para desenho/erro: Kinect (mesma calibracao) e
                # auxiliar (vista independente).
                validos = pontos[valido]
                proj_k = (_reprojeta(ctx.kinect_matrix, ctx.kinect_dist, validos)
                          if len(validos) else None)
                if proj_k is not None and entrada["kin_px"] is not None:
                    desvios = np.linalg.norm(
                        proj_k - entrada["kin_px"][valido], axis=1)
                    entrada["erro_mv_kin"] = metricas._nanmed(desvios)
                    entrada["mv_kin_px"] = _espalha(proj_k, valido, valido.sum())
                if len(validos) and view.usable:
                    try:
                        proj_a = projeta_aux(view, validos)
                        desvios_a = np.linalg.norm(
                            proj_a - aux_px[valido], axis=1)
                        entrada["erro_mv_aux"] = metricas._nanmed(desvios_a)
                        entrada["mv_aux_px"] = _espalha(proj_a, valido,
                                                        valido.sum())
                        entrada["mv_reproj_aux"] = _espalha(desvios_a, valido,
                                                            valido.sum())
                    except Exception:             # noqa: BLE001
                        pass
                # Distance between the two dorsum estimates (same frame).
                palm_k = entrada["k_xyz_cor"][IDX_DORSO]
                palm_mv = entrada["mv_xyz"][IDX_DORSO]
                if (np.isfinite(palm_k).all() and np.isfinite(palm_mv).all()
                        and entrada["mv_mask"][IDX_DORSO]):
                    entrada["d9_mm"] = float(
                        np.linalg.norm(palm_k - palm_mv) * 1000.0)
    elif mao_a is not None and entrada["kin_px"] is None:
        entrada["note"] = "sem profundidade 3D no kinect para triangular"
    elif mao_a is None:
        entrada["note"] = "sem mao na auxiliar"

    # Depth do Kinect visto pela auxiliar (diagnostico visual + erro de
    # correspondencia). Usa o 3D no frame da COR: a calibracao stereo da
    # auxiliar foi feita no frame da camera de cor do Kinect, e o 3D cru do
    # depth vem do frame da camera de profundidade (baseline ~6 cm).
    xyz_aux_frame = entrada.get("k_xyz_cor")
    if xyz_aux_frame is None:
        xyz_aux_frame = entrada["k_xyz"]
    if (mao_a is not None and view is not None and view.usable
            and entrada["aux_px"] is not None
            and np.isfinite(xyz_aux_frame).all(axis=1).any()):
        try:
            validos_k = np.isfinite(xyz_aux_frame).all(axis=1)
            proj_kd = projeta_aux(view, xyz_aux_frame[validos_k])
            entrada["kd_aux_px"] = _espalha(proj_kd, validos_k,
                                            int(validos_k.sum()))
            desvios = np.linalg.norm(proj_kd - entrada["aux_px"][validos_k],
                                     axis=1)
            entrada["kd_aux_err"] = _espalha(desvios, validos_k,
                                            int(validos_k.sum()))
            entrada["kd_aux_err_med"] = metricas._nanmed(desvios)
        except Exception:                             # noqa: BLE001
            pass
    return entrada


def _espalha(valores, mascara, n_validos):
    """Volta um array compacto (n_validos, K) ao espaco dos 21 landmarks."""
    if valores is None:
        return None
    arr = np.asarray(valores, np.float64)
    forma = (21,) + arr.shape[1:]
    saida = np.full(forma, np.nan, np.float64)
    if arr.shape[0] != n_validos:
        return None
    saida[np.asarray(mascara, bool)] = arr
    return saida


# --------------------------------------------------------------------------
# Modo sintetico (valida o codigo sem hardware; NUNCA e' resultado de bancada)
# --------------------------------------------------------------------------
def roda_sintetico(args) -> dict:
    """Gera uma sessao FALSA (com jitter/queda de depth) para validar o codigo.

    Existe porque o pipeline de metricas/relatorio precisa ser testado antes de
    depender da bancada (e porque, sem Kinect na maquina, ainda tem de ser
    possivel verificar CSV, tabelas e PNGs). A sessao e' marcada como
    `origem=sintetico` em TODAS as linhas e no relatorio.
    """
    import tempfile
    pasta = Path(args.sintetico_dir) if args.sintetico_dir else \
        Path(tempfile.gettempdir()) / "tracking_mao_sintetico"
    pasta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261105)
    fps = 25.0
    dt = 1.0 / fps
    duracao = max(4.0, float(args.duracao))
    n = int(duracao * fps)
    t0 = time.monotonic()
    esc = EscritorCSV(pasta, "SINTETICO_" + args.prefixo).abrir()
    # Forma da mao: 21 offsets fixos em volta do centro (metros).
    base = np.array([[0, 0, 0], [-0.03, -0.01, 0.012], [-0.05, -0.02, 0.02],
                     [-0.06, -0.03, 0.026], [-0.065, -0.04, 0.03],
                     [-0.02, -0.05, -0.004], [-0.025, -0.075, -0.006],
                     [-0.028, -0.09, -0.007], [-0.03, -0.1, -0.008],
                     [0.0, -0.055, 0.0], [0.0, -0.085, 0.0], [0.0, -0.1, 0.0],
                     [0.0, -0.11, 0.0], [0.02, -0.05, -0.004],
                     [0.022, -0.078, -0.006], [0.024, -0.09, -0.007],
                     [0.025, -0.098, -0.008], [0.038, -0.045, -0.01],
                     [0.042, -0.065, -0.013], [0.044, -0.075, -0.014],
                     [0.045, -0.082, -0.015]], np.float64)
    palm0 = np.array([0.02, 0.03, 1.05])
    d_kin = np.array([0.004, 0.010, 0.002])       # vies do depth (m)
    seq = 0
    for i in range(n):
        t = i * dt
        # Estatico de 2 a 5 s; movimento sinusoidal no resto, com ENVELOPE
        # suave nas transicoes: uma descontinuidade artificial entre a pausa e o
        # movimento poluiria justamente a metrica de continuidade/saltos.
        if t <= 1.6 or t >= 5.4:
            peso = 1.0
        elif t < 2.0:
            peso = 0.5 * (1.0 - math.cos(math.pi * (t - 1.6) / 0.4))
            peso = 1.0 - peso
        elif t <= 5.0:
            peso = 0.0
        else:
            peso = 0.5 * (1.0 - math.cos(math.pi * (t - 5.0) / 0.4))
        deslocamento = np.array([
            0.10 * math.sin(2 * math.pi * 0.35 * t),
            0.06 * math.sin(2 * math.pi * 0.5 * t + 0.7),
            0.05 * math.sin(2 * math.pi * 0.25 * t + 1.1)])
        centro = palm0 + peso * deslocamento
        seq += 1
        fase = "estatico" if 2.0 <= t <= 5.0 else ""
        # Kinect: cobertura 92% (queda de depth em 8% dos landmarks).
        k_xyz = np.full((21, 3), np.nan)
        k_info = []
        for j in range(21):
            ponto = centro + base[j] + d_kin
            falhou = rng.random() < 0.08
            if not falhou:
                k_xyz[j] = ponto + rng.normal(0, 0.0022, 3)
            k_info.append({"map_fail": bool(rng.random() < 0.02),
                           "depth_fail": falhou,
                           "rescue": bool(falhou and rng.random() < 0.5),
                           "depth_m": float(ponto[2])})
        # Multi-view: cobertura 97%, jitter menor.
        mv_xyz = np.full((21, 3), np.nan)
        mv_mask = np.zeros(21, bool)
        for j in range(21):
            if rng.random() < 0.03:
                continue
            mv_xyz[j] = (centro + base[j] * 1.01
                         + rng.normal(0, 0.0013, 3))
            mv_mask[j] = True
        reproj_kin = np.where(mv_mask, np.abs(rng.normal(1.4, 0.5, 21)), np.nan)
        reproj_aux = np.where(mv_mask, np.abs(rng.normal(2.0, 0.7, 21)), np.nan)
        ok_k = int(np.isfinite(k_xyz).all(axis=1).sum()) > 3
        ok_a = True
        kin_px = np.array([(960 + p[0] * 1000.0, 540 + p[1] * 1000.0)
                           for p in k_xyz], np.float64)
        aux_px = np.array([(640 + p[0] * 700.0, 360 + p[1] * 700.0)
                           for p in mv_xyz], np.float64)
        palm_k = k_xyz[IDX_DORSO]
        palm_mv = mv_xyz[IDX_DORSO]
        d9 = (float(np.linalg.norm(palm_k - palm_mv) * 1000.0)
              if np.isfinite(palm_k).all() and np.isfinite(palm_mv).all()
              else float("nan"))
        esc.frame({
            "seq": seq, "t_unix": _n(time.time() + t, 6),
            "t_mono": _n(t0 + t, 6), "origem": "sintetico",
            "aux_index": 0, "hand": "right", "side": "right", "fase": fase,
            "side_aux": "right", "n_maos_k": "1", "n_maos_a": "1",
            "pareamento": "lado",
            "ok_kinect_detect": _i(ok_k), "ok_aux_detect": _i(ok_a),
            "mv_triangulou": _i(mv_mask.any()), "mv_valid": _i(mv_mask.sum() >= 5),
            "n_landmarks_k": int(np.isfinite(k_xyz).all(axis=1).sum()),
            "n_landmarks_mv": int(mv_mask.sum()),
            "palm9_k_valid": _i(np.isfinite(palm_k).all()),
            "palm9_k_x": _n(palm_k[0]), "palm9_k_y": _n(palm_k[1]),
            "palm9_k_z": _n(palm_k[2]),
            "palm9_mv_valid": _i(np.isfinite(palm_mv).all()),
            "palm9_mv_x": _n(palm_mv[0]), "palm9_mv_y": _n(palm_mv[1]),
            "palm9_mv_z": _n(palm_mv[2]),
            "reproj_mv_kin_med_px": _n(metricas._nanmed(reproj_kin), 4),
            "reproj_mv_aux_med_px": _n(metricas._nanmed(reproj_aux), 4),
            "reproj_k_self_med_px": _n(abs(rng.normal(0.4, 0.15)), 4),
            "map_fail": int(sum(1 for r in k_info if r["map_fail"])),
            "depth_fail": int(sum(1 for r in k_info if r["depth_fail"])),
            "depth_rescue": int(sum(1 for r in k_info if r["rescue"])),
            "sync_color_depth_ms": _n(1.5, 3),
            "d_kin_mv_dorso_mm": _n(d9, 3),
            "kd_aux_err_med_px": _n(abs(rng.normal(3.0, 1.0)), 3),
            "aux_lag_ms": _n(12.0, 3), "dt_s": _n(dt, 6),
            "fps_inst": _n(fps, 3),
            "t_kin_detect_ms": _n(38.0, 3), "t_aux_detect_ms": _n(21.0, 3),
            "t_tri_ms": _n(6.0, 3), "t_ctrl_ms": _n(70.0, 3),
            "note": "sintetico (validacao de codigo)",
        })
        for j in range(21):
            info = k_info[j]
            k_val = bool(np.isfinite(k_xyz[j]).all())
            esc.landmark({
                "seq": seq, "t_unix": _n(time.time() + t, 6),
                "t_mono": _n(t0 + t, 6), "origem": "sintetico",
                "hand": "right", "landmark_id": j, "is_dorso": _i(j == 9),
                "k_valid": _i(k_val),
                "k_conf": _n(0.5 if info["rescue"] else (1.0 if k_val else 0.0), 3),
                "k_map_fail": _i(info["map_fail"]),
                "k_depth_fail": _i(info["depth_fail"]),
                "k_depth_rescue": _i(info["rescue"]),
                "k_depth_m": _n(info["depth_m"] if not info["rescue"]
                                else float("nan"), 6),
                "k_x_m": _n(k_xyz[j][0]), "k_y_m": _n(k_xyz[j][1]),
                "k_z_m": _n(k_xyz[j][2]),
                "k_reproj_px": _n(abs(rng.normal(0.4, 0.15)), 4),
                "mv_valid": _i(mv_mask[j]),
                "mv_implausivel": _i(not mv_mask[j] and rng.random() < 0.3),
                "mv_conf": _n(float(np.clip(1.0 - reproj_kin[j] / 6.0, 0.0, 1.0))
                              if mv_mask[j] else 0.0, 4),
                "mv_x_m": _n(mv_xyz[j][0]), "mv_y_m": _n(mv_xyz[j][1]),
                "mv_z_m": _n(mv_xyz[j][2]),
                "mv_reproj_kin_px": _n(reproj_kin[j], 4),
                "mv_reproj_aux_px": _n(reproj_aux[j], 4),
                "kin_px_x": _n(kin_px[j][0], 2), "kin_px_y": _n(kin_px[j][1], 2),
                "aux_px_x": _n(aux_px[j][0], 2), "aux_px_y": _n(aux_px[j][1], 2),
                "kd_aux_err_px": _n(abs(rng.normal(3.0, 1.0)), 2),
            })
    esc.fechar()
    meta = {
        "schema": 1,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "origem": "sintetico",
        "comando": _comando_reproduzivel(),
        "aux_index": 0, "aux_camera_nome": "(sintetica)",
        "aux_resolucao_usada": [1280, 720], "kinect_resolucao": [1920, 1080],
        "espelhamento": "(sintetico)",
        "calibracao": {"arquivo": "(nenhum)", "rms": None, "tabuleiro": "?",
                       "quadrado_mm": None, "estado": "sintetico",
                       "motivo": "modo --sintetico: nada de hardware foi usado",
                       "aux_size": [1280, 720], "kinect_size": [1920, 1080]},
        "kinect_intrinsics_nota": "(sintetico)",
        "versoes": {"python": sys.version.split()[0], "numpy": np.__version__},
        "duracao_alvo_s": duracao, "reproj_max_px": args.reproj_max,
        "salto_mm": args.salto_mm,
        "teclas": "(sintetico: sem janela)",
        "referencial": "(sintetico)",
        "privacidade": "dados gerados por RNG; nenhum dado de pessoa",
    }
    esc.meta(meta)
    return {"pasta": pasta, "prefixo": "SINTETICO_" + args.prefixo,
            "frames": esc.n_frames, "landmarks": esc.n_landmarks}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)

    if args.sintetico:
        print("[sintetico] gerando sessao FALSA para validar CSV/metricas/"
              "relatorio (nada de hardware).")
        info = roda_sintetico(args)
        print(f"[sintetico] {info['frames']} frames e {info['landmarks']} "
              f"linhas de landmark em {info['pasta']}")
        if not args.sem_relatorio:
            resultado = metricas.analisa(
                info["pasta"] / f"{info['prefixo']}_frames.csv",
                info["pasta"], prefixo=info["prefixo"])
            print(metricas.resumo_console(resultado))
            for nome, caminho in resultado["arquivos"].items():
                print(f"  {nome}: {caminho}")
        print("\n[sintetico] ESTES NUMEROS NAO SAO DE BANCADA: serviram para "
              "validar o codigo.")
        return 0

    ctx = prepara_contexto(args)
    pasta = Path(args.saida_dir)
    esc = EscritorCSV(pasta, args.prefixo).abrir()
    print(f"[ok] sessao: {args.duracao:g} s | aux {ctx.aux_index} "
          f"({ctx.aux_nome}) | Kinect {ctx.tracker.color_width}x"
          f"{ctx.tracker.color_height} | CSV em {pasta}")
    print("[ok] calibracao: " + str(ctx.calibracao.get("estado")) + " - "
          + str(ctx.calibracao.get("motivo")))
    registro = {}
    try:
        registro = roda_sessao(ctx, args, esc)
    finally:
        esc.fechar()
        esc.meta(_meta_sessao(ctx, args, "kinect_aux", {
            "frames_logados": esc.n_frames,
            "linhas_landmark": esc.n_landmarks,
            "duracao_real_s": registro.get("duracao_s"),
            "cancelado": registro.get("cancelado", False),
        }))
    print(f"[ok] gravados {esc.n_frames} frames e {esc.n_landmarks} linhas de "
          f"landmark")

    if registro.get("cancelado") and esc.n_frames < 5:
        print("[aviso] sessao cancelada sem dados suficientes; relatorio nao "
              "gerado.")
        return 0
    if args.sem_relatorio:
        print("[ok] --sem-relatorio: nada de relatorio gerado.")
        return 0
    if esc.n_frames == 0:
        print("[erro] nenhum frame gravado: o Kinect nao entregou RGB novo "
              "durante a sessao. Nada a relatar.\n"
              "    Veja tools\\checklist_kinect_mudo.py")
        return 3

    resultado = metricas.analisa(esc.caminho_frames, pasta, prefixo=args.prefixo)
    print(metricas.resumo_console(resultado))
    print("\n[ok] arquivos gerados:")
    for nome, caminho in resultado["arquivos"].items():
        print(f"  {nome}: {caminho}")
    sem_mao = resultado["cobertura"].get("frames_sem_mao", 0)
    if sem_mao and sem_mao == esc.n_frames:
        print("\n[aviso] NENHUMA mao foi detectada em nenhum frame: o relatorio "
              "sai com cobertura 0 e a conclusao e' 'sem teste fisico util'.\n"
              "    Coloque a mao na frente do Kinect E da auxiliar e repita.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
