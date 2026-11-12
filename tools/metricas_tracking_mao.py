#!/usr/bin/env python
"""Metricas de robustez do rastreamento da mao: Kinect (depth/MediaPipe) x multi-view.

Este modulo e' a parte PURA (sem hardware) da comparacao: ele le os CSV que
`tools/compara_tracking_mao.py` grava, calcula as metricas de robustez, compara
a suavizacao de trajetoria (causal x offline) e escreve o relatorio em Markdown
+ PNGs. Nada aqui abre camera, Kinect, serial ou janela -- por isso pode ser
rodado em qualquer maquina (inclusive a partir de um CSV antigo).

O que e' medido (e com que definicao):

1. COBERTURA: fracao dos frames "tentados" para aquele lado em que o landmark
   tem posicao 3D valida. Denominador = linhas do CSV de frames para o lado
   (o frame entra quando ALGUMA camera viu aquela mao), o que e' o denominador
   honesto: se o Kinect nao detectou a mao naquele frame, aquilo conta como
   perda de cobertura do Kinect, nao como frame que nao existiu.
2. FALHA/QUEDA DE DEPTH: por landmark, o pipeline do Kinect tenta
   pixel de cor -> pixel de depth -> mediana de depth -> camera space. Cada
   etapa pode falhar (`k_map_fail`), o depth pode nao ter leitura
   (`k_depth_fail`) e existe o RESGATE por profundidade plausivel
   (`k_depth_rescue`, ver `_landmark_camera_point`). Os tres sao contados.
3. ERRO DE REPROJECAO: multi-view = distancia (px do Kinect) entre o pixel
   MediaPipe e a reprojecao do ponto triangulado; na auxiliar = o mesmo medido
   nos pixels da auxiliar (vista INDEPENDENTE da que fecha a triangulacao, por
   isso e' o numero mais informativo). Kinect depth = ida-e-volta
   (pixel -> 3D -> pixel): mede consistencia do mapeamento, NAO e' um erro
   independente -- esta' no CSV e no relatorio com esse aviso.
4. JITTER EM POSE ESTATICA: desvio-padrao da posicao 3D (mm) dentro de
   segmentos estaticos, por landmark. Os segmentos vem de duas fontes:
   (a) rotulo de fase dado pelo operador na janela (coluna `fase`); ou
   (b) deteccao AUTOMATICA: velocidade da mao < `--limiar-estatico-mm-s`
   durante uma janela de 0,6 s, com duracao minima de 0,5 s.
5. CONTINUIDADE/OUTLIERS: saltos frame-a-frame acima de `--salto-mm`
   (padrao 50 mm) e o maior salto observado. Em ~30 fps, 50 mm/frame = 1,5 m/s:
   acima disso o movimento humano e' rapido, mas possivel -- por isso o
   relatorio mostra o limiar e a distribuicao, nao so' a contagem.
6. FPS/LATENCIA: dt entre frames e os tempos de cada etapa (deteccao Kinect,
   deteccao auxiliar, triangulacao, laco). A latencia ponta-a-ponta do
   MediaPipe/TFLite nao e' exposta pela API: o que se mede aqui e' o CUSTO de
   processamento por etapa, que e' o piso de latencia do laco.

Suavizacao (item 4 do pedido): o modulo aplica na trajetoria do dorso
(landmark 9) e nos 21 landmarks:
  - cru;
  - media movel causal (so' passado) e offline (centrada, usa futuro);
  - Savitzky-Golay causal (ajuste no passado, avaliado na amostra atual) e
    offline (centrado, `scipy.signal.savgol_filter`);
  - Savitzky-Golay PREDITIVO (ajuste no passado avaliado em i+1: atraso zero,
    mas extrapola nas inversoes);
  - EMA causal (o filtro online mais comum).
Mede jitter, atraso (por correlacao cruzada do perfil de velocidade) e erro de
forma depois de compensar o atraso. A secao 5 do relatorio explica por que a
suavizacao offline (a que o pos-processamento do E2T usa) nao e' diretamente
executavel online e qual e' o analogo causal medido aqui.

Uso (a partir da raiz do projeto):
    .venv\\Scripts\\python.exe tools\\metricas_tracking_mao.py --csv <dir>\\tracking_mao_frames.csv
    .venv\\Scripts\\python.exe tools\\metricas_tracking_mao.py --csv ... --saida ... --sem-png

Saida padrao: ao lado do CSV (mesmo diretorio), com prefixo `relatorio_tracking`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Nomes do MediaPipe HandLandmarker (21 nos). O 9 e' o MCP do dedo medio: e' o
# ponto central do DORSO usado como referencia da palma no projeto (o plano
# medio do dorso usa os 5 pontos 0,5,9,13,17).
LANDMARK_NOMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
IDX_DORSO = 9
PONTOS_DORSO = (0, 5, 9, 13, 17)

#: Limiares do veredito por criterio (explicitos de proposito: o veredito do
#: relatorio tem de ser auditavel -- mudar o criterio muda o veredito, e isso
#: tem de estar escrito).
CRITERIOS = {
    "cobertura_min": 0.90,       # fracao de frames com 3D valido
    "jitter_max_mm": 5.0,        # desvio-padrao em pose estatica
    "salto_max_frac": 0.01,      # fracao de transicoes com salto > limiar
    "reproj_max_px": 6.0,        # mesmo gate do groundtruth (TRI_ERR_MAX_PX)
}

#: Suavizacao: janelas padrao (amostras). `janela_sg` 7 e' ~0,23 s em 30 fps.
JANELA_MM = 5
JANELA_SG = 7
ORDEM_SG = 2
ALPHA_EMA = 0.30

FONTE_SINTETICA = "sintetico"


# --------------------------------------------------------------------------
# Leitura
# --------------------------------------------------------------------------
def _f(valor) -> float:
    """float do CSV, NaN quando vazio/ilegivel (nunca 0.0 disfarcado)."""
    texto = ("" if valor is None else str(valor)).strip()
    if not texto:
        return float("nan")
    try:
        return float(texto)
    except ValueError:
        return float("nan")


def _b(valor) -> bool:
    """'1'/'true'/'sim' -> True; '' -> False."""
    texto = ("" if valor is None else str(valor)).strip().lower()
    return texto in ("1", "true", "sim", "yes", "s", "t")


def le_csv(caminho) -> list[dict]:
    caminho = Path(caminho)
    with caminho.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def le_meta(caminho_csv) -> dict:
    """Le o `<base>_meta.json` ao lado do CSV (sessao/calibracao/versoes).

    Sem o sidecar o relatorio continua sendo gerado (o CSV basta para as
    metricas); ele so' perde a secao de setup/calibracao -- por isso devolve {}
    em vez de levantar.
    """
    base = Path(caminho_csv)
    nome = base.name
    sufixo = "_frames.csv"
    if nome.endswith(sufixo):
        meta = base.with_name(nome[: -len(sufixo)] + "_meta.json")
    else:
        meta = base.with_suffix("").with_suffix(".meta.json")
    if not meta.exists():
        return {}
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@dataclass
class Rastro:
    """Uma trajetoria (1 lado x 1 landmark) tirada do CSV de landmarks."""

    hand: str
    landmark_id: int
    seq: np.ndarray
    t: np.ndarray
    k_xyz: np.ndarray
    mv_xyz: np.ndarray
    k_valid: np.ndarray
    mv_valid: np.ndarray
    k_conf: np.ndarray
    mv_conf: np.ndarray
    k_map_fail: np.ndarray
    k_depth_fail: np.ndarray
    k_depth_rescue: np.ndarray
    k_depth_m: np.ndarray
    reproj_kin_px: np.ndarray
    reproj_aux_px: np.ndarray
    aux_px: np.ndarray
    kin_px: np.ndarray
    reproj_self_px: np.ndarray
    kd_aux_err_px: np.ndarray

    @property
    def nome(self) -> str:
        return LANDMARK_NOMES[self.landmark_id]

    @property
    def e_dorso(self) -> bool:
        return self.landmark_id == IDX_DORSO


@dataclass
class Sessao:
    frames: list[dict] = field(default_factory=list)
    rastros: dict = field(default_factory=dict)      # (hand, lid) -> Rastro
    meta: dict = field(default_factory=dict)
    origem: str = ""
    #: instante (t_mono) do primeiro frame: os rastros guardam tempo ABSOLUTO e
    #: as metricas trabalham em tempo RELATIVO; sem esta ancora a mascara de
    #: segmento nunca casaria com a trajetoria.
    t0: float = 0.0

    @property
    def lados(self) -> list[str]:
        return sorted({str(r.get("hand") or "") for r in self.frames} - {""})

    def tempo_relativo(self, rastro: "Rastro") -> np.ndarray:
        return rastro.t - self.t0

    @property
    def sintetica(self) -> bool:
        return (str(self.meta.get("origem") or self.origem).strip().lower()
                == FONTE_SINTETICA)


def carrega_sessao(caminho_frames) -> Sessao:
    """Le o CSV de frames + o de landmarks (mesmo prefixo) numa Sessao."""
    caminho_frames = Path(caminho_frames)
    nome = caminho_frames.name
    sufixo = "_frames.csv"
    if nome.endswith(sufixo):
        base = nome[: -len(sufixo)]
        caminho_lm = caminho_frames.with_name(base + "_landmarks.csv")
    else:                                  # tolera nome fora do padrao
        caminho_lm = caminho_frames.with_name(caminho_frames.stem + "_landmarks.csv")
    sessao = Sessao(frames=le_csv(caminho_frames), meta=le_meta(caminho_frames),
                    origem="")
    if not caminho_lm.exists():
        raise SystemExit(
            f"CSV de landmarks nao encontrado: {caminho_lm}\n"
            "Ele e' gravado junto com o de frames por tools\\compara_tracking_mao.py."
        )
    linhas = le_csv(caminho_lm)
    sessao.origem = str(sessao.meta.get("origem") or
                        (linhas[0].get("origem") if linhas else "") or "")
    tempos = np.array([_f(r.get("t_mono")) for r in sessao.frames], float)
    finitos = tempos[np.isfinite(tempos)]
    sessao.t0 = float(finitos[0]) if finitos.size else 0.0

    # Agrupa por (lado, landmark). np.array de listas vazias precisa do dtype
    # explicito, senao vira float64 (0,) e quebra o shape (N,3).
    grupos: dict = {}
    for linha in linhas:
        chave = (str(linha.get("hand") or ""), int(_f(linha.get("landmark_id"))))
        grupos.setdefault(chave, []).append(linha)

    for (hand, lid), itens in grupos.items():
        n = len(itens)
        seq = np.array([_f(i.get("seq")) for i in itens], float)
        t = np.array([_f(i.get("t_mono")) for i in itens], float)

        def _xyz(campo_x, campo_y, campo_z):
            arr = np.array([[_f(i.get(campo_x)), _f(i.get(campo_y)),
                             _f(i.get(campo_z))] for i in itens], float)
            return arr.reshape(n, 3)

        def _vec(campo):
            return np.array([_f(i.get(campo)) for i in itens], float)

        def _mask(campo):
            return np.array([_b(i.get(campo)) for i in itens], bool)

        rastro = Rastro(
            hand=hand,
            landmark_id=int(lid),
            seq=seq,
            t=t,
            k_xyz=_xyz("k_x_m", "k_y_m", "k_z_m"),
            mv_xyz=_xyz("mv_x_m", "mv_y_m", "mv_z_m"),
            k_valid=_mask("k_valid"),
            mv_valid=_mask("mv_valid"),
            k_conf=_vec("k_conf"),
            mv_conf=_vec("mv_conf"),
            k_map_fail=_mask("k_map_fail"),
            k_depth_fail=_mask("k_depth_fail"),
            k_depth_rescue=_mask("k_depth_rescue"),
            k_depth_m=_vec("k_depth_m"),
            reproj_kin_px=_vec("mv_reproj_kin_px"),
            reproj_aux_px=_vec("mv_reproj_aux_px"),
            aux_px=np.array([[_f(i.get("aux_px_x")), _f(i.get("aux_px_y"))]
                             for i in itens], float).reshape(n, 2),
            kin_px=np.array([[_f(i.get("kin_px_x")), _f(i.get("kin_px_y"))]
                             for i in itens], float).reshape(n, 2),
            reproj_self_px=_vec("k_reproj_px"),
            kd_aux_err_px=_vec("kd_aux_err_px"),
        )
        sessao.rastros[(hand, int(lid))] = rastro
    return sessao


# --------------------------------------------------------------------------
# Estatistica basica (sem pandas: o projeto ja' usa numpy em todo lugar)
# --------------------------------------------------------------------------
def _nanmed(valores) -> float:
    arr = np.asarray(valores, float).ravel()
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _nanpct(valores, pct=95.0) -> float:
    arr = np.asarray(valores, float).ravel()
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, pct)) if arr.size else float("nan")


def _fps_da_sessao(frames: list[dict]) -> float:
    """FPS medio do laco pelas colunas `dt_s` (0 quando nao da' para medir)."""
    dts = np.array([_f(r.get("dt_s")) for r in frames], float)
    dts = dts[np.isfinite(dts) & (dts > 0)]
    if dts.size == 0:
        return 0.0
    return float(1.0 / np.median(dts))


def _tempo_relativo(frames: list[dict]) -> np.ndarray:
    t = np.array([_f(r.get("t_mono")) for r in frames], float)
    if t.size == 0:
        return t
    finitos = t[np.isfinite(t)]
    if finitos.size == 0:
        return np.arange(t.size, dtype=float)
    t0 = float(finitos[0])
    return t - t0


# --------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------
def metricas_lateralidade(sessao: Sessao) -> dict:
    """Concordancia de lateralidade entre as duas cameras (rotulo x rotulo).

    Por que importa: o pipeline de producao (`AuxStereoSource`) casa as maos
    PELO LADO. Se o rotulo do MediaPipe sai trocado em uma camera, o 3D
    multi-view simplesmente nao existe -- e o sintoma (multi-view "mudo") nao
    diz a causa. Aqui a divergencia vira numero.
    """
    iguais = diferentes = 0
    modos: dict = {}
    exemplos = []
    for linha in sessao.frames:
        modo = str(linha.get("pareamento") or "").strip()
        if modo:
            modos[modo] = modos.get(modo, 0) + 1
        lado_k = str(linha.get("hand") or "")
        lado_a = str(linha.get("side_aux") or "")
        if not modo or modo == "so_kinect" or modo == "so_aux":
            continue
        if lado_k not in ("right", "left") or lado_a not in ("right", "left"):
            continue
        if lado_k == lado_a:
            iguais += 1
        else:
            diferentes += 1
            if len(exemplos) < 5:
                exemplos.append(f"{lado_k} (Kinect) x {lado_a} (aux)")
    total = iguais + diferentes
    return {
        "comparaveis": total,
        "iguais": iguais,
        "diferentes": diferentes,
        "fracao_igual": (iguais / total) if total else float("nan"),
        "pareamento": modos,
        "exemplos": exemplos,
    }


def metricas_cobertura(sessao: Sessao) -> dict:
    """Cobertura por metodo, por lado e por landmark (+ dorso e 21 nos)."""
    saida: dict = {"por_lado": {}, "por_landmark": {}, "frames_sem_mao": 0}
    # Frames "tentados" por lado: linhas do CSV de frames daquele lado.
    tentados = {lado: 0 for lado in sessao.lados}
    for linha in sessao.frames:
        hand = str(linha.get("hand") or "")
        if hand in tentados:
            tentados[hand] += 1
        elif not hand or hand == "none":
            saida["frames_sem_mao"] += 1

    for lado in sessao.lados:
        n = max(tentados.get(lado, 0), 1)
        linhas = [r for r in sessao.frames if str(r.get("hand") or "") == lado]
        # Deteccao 2D por camera (independe de 3D valido).
        det_k = sum(1 for r in linhas if _b(r.get("ok_kinect_detect")))
        det_a = sum(1 for r in linhas if _b(r.get("ok_aux_detect")))
        det_ambas = sum(1 for r in linhas
                        if _b(r.get("ok_kinect_detect")) and _b(r.get("ok_aux_detect")))
        mv_frames = sum(1 for r in linhas if _b(r.get("mv_valid")))
        k_frac = []
        mv_frac = []
        for lid in range(21):
            rastro = sessao.rastros.get((lado, lid))
            if rastro is None:
                k_frac.append(float("nan"))
                mv_frac.append(float("nan"))
                continue
            k_frac.append(float(np.mean(rastro.k_valid)))
            mv_frac.append(float(np.mean(rastro.mv_valid)))
        k_frac_arr = np.asarray(k_frac, float)
        mv_frac_arr = np.asarray(mv_frac, float)
        saida["por_lado"][lado] = {
            "frames_tentados": tentados.get(lado, 0),
            "det_kinect": det_k / n,
            "det_aux": det_a / n,
            "det_ambas": det_ambas / n,
            "cobertura_kinect": float(np.nanmean(k_frac_arr)) if np.isfinite(k_frac_arr).any() else float("nan"),
            "cobertura_mv": float(np.nanmean(mv_frac_arr)) if np.isfinite(mv_frac_arr).any() else float("nan"),
            "cobertura_kinect_dorso": float(k_frac[IDX_DORSO]) if np.isfinite(k_frac[IDX_DORSO]) else float("nan"),
            "cobertura_mv_dorso": float(mv_frac[IDX_DORSO]) if np.isfinite(mv_frac[IDX_DORSO]) else float("nan"),
            "cobertura_kinect_min": float(np.nanmin(k_frac_arr)) if np.isfinite(k_frac_arr).any() else float("nan"),
            "cobertura_mv_min": float(np.nanmin(mv_frac_arr)) if np.isfinite(mv_frac_arr).any() else float("nan"),
            "frames_mv_validos": mv_frames,
            "por_landmark_kinect": k_frac,
            "por_landmark_mv": mv_frac,
        }
    return saida


def metricas_depth(sessao: Sessao) -> dict:
    """Taxa de falha/queda de depth e de resgate, por lado e por landmark."""
    saida: dict = {}
    for lado in sessao.lados:
        total = 0
        map_fail = depth_fail = rescue = 0
        sem_ponto = 0
        for lid in range(21):
            rastro = sessao.rastros.get((lado, lid))
            if rastro is None:
                continue
            total += rastro.k_map_fail.size
            map_fail += int(rastro.k_map_fail.sum())
            depth_fail += int(rastro.k_depth_fail.sum())
            rescue += int(rastro.k_depth_rescue.sum())
            sem_ponto += int((~rastro.k_valid).sum())
        linhas = [r for r in sessao.frames if str(r.get("hand") or "") == lado]
        frames_queda_total = sum(
            1 for r in linhas
            if _b(r.get("ok_kinect_detect")) and int(_f(r.get("n_landmarks_k"))) == 0)
        saida[lado] = {
            "landmarks_tentados": total,
            "map_fail": map_fail / total if total else float("nan"),
            "depth_fail": depth_fail / total if total else float("nan"),
            "rescue": rescue / total if total else float("nan"),
            "sem_ponto_3d": sem_ponto / total if total else float("nan"),
            "frames_queda_total": frames_queda_total,
            "frames_com_mao": len(linhas),
            "sync_color_depth_ms": _nanmed([_f(r.get("sync_color_depth_ms"))
                                            for r in linhas]),
        }
    return saida


def metricas_reprojecao(sessao: Sessao) -> dict:
    """Erro de reprojecao (px) por metodo: mediana/p95/por landmark."""
    saida: dict = {"por_lado": {}, "por_landmark": {}}
    for lado in sessao.lados:
        mv_k, mv_a, k_self, kd_a = [], [], [], []
        por_lm = {"mv_kin": [], "mv_aux": [], "k_self": [], "kd_aux": []}
        for lid in range(21):
            rastro = sessao.rastros.get((lado, lid))
            if rastro is None:
                for chave in por_lm:
                    por_lm[chave].append(float("nan"))
                continue
            e_kin = rastro.reproj_kin_px[rastro.mv_valid]
            e_aux = rastro.reproj_aux_px[rastro.mv_valid]
            e_self = rastro.reproj_self_px[rastro.k_valid]
            e_kd = rastro.kd_aux_err_px[np.isfinite(rastro.kd_aux_err_px)]
            mv_k.extend(e_kin[np.isfinite(e_kin)].tolist())
            mv_a.extend(e_aux[np.isfinite(e_aux)].tolist())
            k_self.extend(e_self[np.isfinite(e_self)].tolist())
            kd_a.extend(e_kd.tolist())
            por_lm["mv_kin"].append(_nanmed(e_kin))
            por_lm["mv_aux"].append(_nanmed(e_aux))
            por_lm["k_self"].append(_nanmed(e_self))
            por_lm["kd_aux"].append(_nanmed(e_kd))
        saida["por_lado"][lado] = {
            "mv_kin_mediana": _nanmed(mv_k),
            "mv_kin_p95": _nanpct(mv_k, 95),
            "mv_aux_mediana": _nanmed(mv_a),
            "mv_aux_p95": _nanpct(mv_a, 95),
            "k_self_mediana": _nanmed(k_self),
            "k_self_p95": _nanpct(k_self, 95),
            "kd_aux_mediana": _nanmed(kd_a),
            "kd_aux_p95": _nanpct(kd_a, 95),
            "n_mv": len(mv_k),
            "n_kd_aux": len(kd_a),
        }
        saida["por_landmark"][lado] = por_lm
    return saida


def _referencia_palm9(sessao: Sessao, lado: str) -> Optional[np.ndarray]:
    """Trajetoria de referencia do lado: palm9 multi-view se cobrir >=50%.

    A referencia so' serve para decidir QUAIS segmentos sao estaticos por
    velocidade; ela nao entra em nenhuma comparacao de precisao. Preferir a
    multi-view evita que a queda de depth do Kinect "congele" a referencia e
    faca um movimento parecer estatico.
    """
    mv = sessao.rastros.get((lado, IDX_DORSO))
    if mv is not None and np.mean(mv.mv_valid) >= 0.5:
        return mv.mv_xyz.copy(), mv.mv_valid.copy()
    k = sessao.rastros.get((lado, IDX_DORSO))
    if k is not None and np.mean(k.k_valid) >= 0.5:
        return k.k_xyz.copy(), k.k_valid.copy()
    return None


def segmentos_estaticos(sessao: Sessao, lado: str, fps: float,
                        limiar_mm_s: float = 30.0,
                        janela_s: float = 0.3,
                        minimo_s: float = 0.4) -> list[tuple[float, float]]:
    """Segmentos (t0, t1) em segundos onde a mao ficou parada.

    Duas fontes, nesta ordem de prioridade:
    1. rotulo de fase do operador (coluna `fase` com 'estatico');
    2. deteccao automatica por velocidade da trajetoria de referencia.

    A velocidade automatica e' medida na trajetoria SUAVIZADA por mediana
    (janela de 0,4 s), nao na crua: com o jitter do depth (~14 mm medidos nesta
    bancada) a diferenca frame-a-frame de uma mao PARADA vale ~20 mm, que a
    14 fps da' ~280 mm/s -- acima de qualquer limiar util. Medir a velocidade na
    serie crua tornava impossivel detectar pausa em dado ruidoso (aconteceu no
    teste de 11/11/2026: duas sessoes seguidas com 'nenhum segmento estatico').
    A mediana de 6 amostras derruba esse ruido sem apagar uma pausa real.

    A deteccao automatica e' conservadora: exige velocidade abaixo do limiar em
    TODA a janela de `janela_s` em volta da amostra.
    """
    t = _tempo_relativo(sessao.frames)
    fases = [str(r.get("fase") or "").strip().lower() for r in sessao.frames]
    explicitos = [f in ("estatico", "static", "parado", "e") for f in fases]
    if any(explicitos) and fps > 0:
        # Frames marcados pelo operador: agrupa runs e converte para tempo.
        runs = _runs(explicitos)
        segs = []
        for ini, fim in runs:
            if fim - ini < 2:
                continue
            t0 = float(t[ini]) if np.isfinite(t[ini]) else ini / fps
            t1 = float(t[fim - 1]) if np.isfinite(t[fim - 1]) else (fim - 1) / fps
            if t1 - t0 >= minimo_s * 0.5:
                segs.append((t0, t1))
        if segs:
            return segs

    ref = _referencia_palm9(sessao, lado)
    if ref is None or fps <= 0:
        return []
    xyz, valido = ref
    n = len(valido)
    if n < 6:
        return []
    dt = 1.0 / fps
    idx = np.arange(n)
    bons = np.isfinite(xyz).all(axis=1) & valido
    if bons.sum() < 6:
        return []
    # Trajetoria continua (interpolada nos buracos) para poder filtrar.
    suave = xyz.copy()
    for eixo in range(3):
        suave[:, eixo] = np.interp(idx, idx[bons], xyz[bons, eixo])
    # Mediana por eixo (janela ~0,4 s): mata o ruido frame-a-frame e preserva
    # a pausa.
    meia = max(1, int(round(0.2 * fps)))
    if 2 * meia + 1 <= n:
        de = np.lib.stride_tricks.sliding_window_view(suave, 2 * meia + 1,
                                                      axis=0)
        mediana = np.median(de, axis=2)                    # (n-2meia, 3)
        cheio = np.empty_like(suave)
        cheio[meia:n - meia] = mediana
        cheio[:meia] = mediana[0]
        cheio[n - meia:] = mediana[-1]
        suave = cheio
    velocidade = np.zeros(n)
    for i in range(1, n):
        velocidade[i] = float(np.linalg.norm(suave[i] - suave[i - 1])) / dt
    limiar = limiar_mm_s / 1000.0
    estatico = velocidade < limiar
    # Exige a janela inteira parada (erosao por janela centrada).
    meia_janela = max(1, int(round(janela_s * fps / 2.0)))
    if meia_janela * 2 + 1 < n:
        kernel = np.ones(2 * meia_janela + 1, dtype=float)
        soma = np.convolve(estatico.astype(float), kernel, mode="same")
        estatico = soma > (2 * meia_janela + 0.5)   # nenhum False na janela
    runs = _runs(estatico)
    segs = []
    for ini, fim in runs:
        if (fim - ini) * dt < minimo_s:
            continue
        t0 = float(t[ini]) if np.isfinite(t[ini]) else ini * dt
        t1 = float(t[fim - 1]) if np.isfinite(t[fim - 1]) else (fim - 1) * dt
        segs.append((t0, t1))
    return segs


def _runs(mask: Iterable[bool]) -> list[tuple[int, int]]:
    """[(ini, fim_exclusivo)] dos trechos True consecutivos."""
    runs = []
    ini = None
    for i, valor in enumerate(mask):
        if valor and ini is None:
            ini = i
        elif not valor and ini is not None:
            runs.append((ini, i))
            ini = None
    if ini is not None:
        runs.append((ini, len(mask)))
    return runs


def _mascara_segmentos(t: np.ndarray, segmentos: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(len(t), bool)
    for t0, t1 in segmentos:
        mask |= (t >= t0) & (t <= t1)
    return mask


def metricas_jitter(sessao: Sessao, fps: float,
                    segmentos: dict) -> dict:
    """Jitter (mm) por metodo e por landmark dentro dos segmentos estaticos."""
    saida: dict = {"por_lado": {}, "segmentos": segmentos, "por_landmark": {}}
    for lado in sessao.lados:
        segs = segmentos.get(lado) or []
        resumo = {"jitter_kinect": float("nan"), "jitter_mv": float("nan"),
                  "dorso_kinect": float("nan"), "dorso_mv": float("nan"),
                  "std2_kinect": float("nan"), "std2_mv": float("nan"),
                  "n_amostras": 0, "fonte": "auto"}
        jitters_k, jitters_mv, std2_k, std2_mv = [], [], [], []
        for lid in range(21):
            rastro = sessao.rastros.get((lado, lid))
            if rastro is None:
                continue
            # A mascara e' POR RASTRO: as linhas de landmark so' existem nos
            # frames em que aquela mao apareceu, entao o vetor de tempo dela e'
            # mais curto que o de frames (e o tempo e' relativo ao mesmo t0).
            t_rastro = sessao.tempo_relativo(rastro)
            mask = (_mascara_segmentos(t_rastro, segs) if segs
                    else np.zeros(len(t_rastro), bool))
            jk = _std_mascara(rastro.k_xyz, rastro.k_valid & mask)
            jm = _std_mascara(rastro.mv_xyz, rastro.mv_valid & mask)
            sk = _std2_mascara(rastro.k_xyz, rastro.k_valid & mask)
            sm = _std2_mascara(rastro.mv_xyz, rastro.mv_valid & mask)
            jitters_k.append(jk)
            jitters_mv.append(jm)
            std2_k.append(sk)
            std2_mv.append(sm)
            if lid == IDX_DORSO:
                resumo["n_amostras"] = int(mask.sum())
            saida["por_landmark"].setdefault(lado, {})[lid] = {
                "jitter_kinect_mm": jk * 1000.0, "jitter_mv_mm": jm * 1000.0}
        resumo["jitter_kinect"] = _nanmed(jitters_k) * 1000.0
        resumo["jitter_mv"] = _nanmed(jitters_mv) * 1000.0
        dorso = saida["por_landmark"].get(lado, {}).get(IDX_DORSO, {})
        resumo["dorso_kinect"] = float(dorso.get("jitter_kinect_mm", float("nan")))
        resumo["dorso_mv"] = float(dorso.get("jitter_mv_mm", float("nan")))
        resumo["std2_kinect"] = _nanmed(std2_k) * 1000.0
        resumo["std2_mv"] = _nanmed(std2_mv) * 1000.0
        resumo["n_landmarks_kinect"] = int(np.isfinite(jitters_k).sum())
        resumo["n_landmarks_mv"] = int(np.isfinite(jitters_mv).sum())
        saida["por_lado"][lado] = resumo
    return saida


def _std_mascara(xyz: np.ndarray, mask: np.ndarray) -> float:
    """Desvio-padrao 3D (norma dos desvios por eixo) das amostras na mascara.

    Exige >= 5 amostras: com 2-3 amostras o desvio e' ruido numerico e fingir
    jitter com ele seria pior do que devolver NaN ("nao medido").
    """
    mask = mask & np.isfinite(xyz).all(axis=1)
    if int(mask.sum()) < 5:
        return float("nan")
    dados = xyz[mask]
    desvios = np.nanstd(dados, axis=0)
    return float(np.linalg.norm(desvios))


def _std2_mascara(xyz: np.ndarray, mask: np.ndarray) -> float:
    """Desvio-padrao da SEGUNDA diferenca (jitter de alta frequencia, mm).

    Estatistica classica de jitter: a segunda diferenca cancela a velocidade
    constante e sobra o ruido frame-a-frame. Ela NAO depende de a mao estar
    exatamente parada (movimento uniforme tambem cancela), por isso e' util
    como contraprova do desvio-padrao puro.
    """
    valores = []
    for i in range(1, len(xyz) - 1):
        if not (mask[i] and mask[i - 1] and mask[i + 1]):
            continue
        d = xyz[i - 1] - 2.0 * xyz[i] + xyz[i + 1]
        if np.isfinite(d).all():
            valores.append(float(np.linalg.norm(d)))
    if len(valores) < 5:
        return float("nan")
    return float(np.std(valores))


def metricas_saltos(sessao: Sessao, limiar_mm: float = 50.0) -> dict:
    """Continuidade: saltos frame-a-frame acima do limiar, por metodo."""
    saida: dict = {"limiar_mm": limiar_mm, "por_lado": {}, "por_landmark": {}}
    for lado in sessao.lados:
        resumo = {"kinect": {}, "mv": {}}
        for metodo, campo_xyz, campo_val in (("kinect", "k_xyz", "k_valid"),
                                             ("mv", "mv_xyz", "mv_valid")):
            saltos = 0
            pares = 0
            maior = 0.0
            por_lm = []
            for lid in range(21):
                rastro = sessao.rastros.get((lado, lid))
                if rastro is None:
                    por_lm.append(float("nan"))
                    continue
                xyz = getattr(rastro, campo_xyz)
                val = getattr(rastro, campo_val)
                n_s = 0
                n_p = 0
                m = 0.0
                for i in range(1, len(xyz)):
                    if not (val[i] and val[i - 1]):
                        continue
                    d = float(np.linalg.norm(xyz[i] - xyz[i - 1]))
                    if not np.isfinite(d):
                        continue
                    n_p += 1
                    if d * 1000.0 > limiar_mm:
                        n_s += 1
                    m = max(m, d)
                saltos += n_s
                pares += n_p
                maior = max(maior, m)
                por_lm.append((n_s / n_p) if n_p else float("nan"))
            resumo[metodo] = {
                "saltos": saltos,
                "pares": pares,
                "fracao": (saltos / pares) if pares else float("nan"),
                "maior_mm": maior * 1000.0,
                "por_landmark": por_lm,
            }
        saida["por_lado"][lado] = resumo
        saida["por_landmark"][lado] = {
            "kinect": resumo["kinect"]["por_landmark"],
            "mv": resumo["mv"]["por_landmark"],
        }
    return saida


def metricas_latencia(sessao: Sessao) -> dict:
    """Custo por etapa do laco (ms) e FPS; 'sem medicao' vira NaN."""
    campos = {
        "kinect_detect": "t_kin_detect_ms",
        "aux_detect": "t_aux_detect_ms",
        "triangulacao": "t_tri_ms",
        "laco": "t_ctrl_ms",
    }
    saida = {}
    for nome, campo in campos.items():
        valores = [_f(r.get(campo)) for r in sessao.frames]
        saida[nome] = {"mediana": _nanmed(valores), "p95": _nanpct(valores, 95),
                       "n": int(np.isfinite(valores).sum())}
    dts = [_f(r.get("dt_s")) for r in sessao.frames]
    saida["dt_s"] = {"mediana": _nanmed(dts), "p95": _nanpct(dts, 95)}
    saida["fps"] = _fps_da_sessao(sessao.frames)
    return saida


# --------------------------------------------------------------------------
# Suavizacao de trajetoria
# --------------------------------------------------------------------------
def _mascara_finitos(xyz: np.ndarray, val: np.ndarray) -> np.ndarray:
    return val & np.isfinite(xyz).all(axis=1)


def interpola_buracos(xyz: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Interpola buracos de validade para poder filtrar (o CSV guarda NaN).

    A interpolacao e' LINEAR e so' existe para a comparacao de suavizacao (um
    filtro nao aceita NaN). Ela nunca e' usada nas metricas de cobertura.
    """
    out = np.array(xyz, float, copy=True)
    mask = _mascara_finitos(out, val)
    if mask.sum() < 3:
        return out
    idx = np.arange(len(out))
    for eixo in range(3):
        out[:, eixo] = np.interp(idx, idx[mask], out[mask, eixo])
    return out


def media_movel(xyz: np.ndarray, janela: int, causal: bool) -> np.ndarray:
    """Media movel (uma passada, sem padding: bordas encurtam a janela).

    `causal=True` usa so' as amostras passadas (executavel online);
    `causal=False` e' centrada (zero-phase, usa futuro).
    """
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    janela = max(1, int(janela))
    if n == 0:
        return xyz.copy()
    if causal:
        saida = np.empty_like(xyz)
        soma = np.zeros(3)
        for i in range(n):
            soma += xyz[i]
            if i >= janela:
                soma -= xyz[i - janela]
            saida[i] = soma / min(i + 1, janela)
        return saida
    meio = janela // 2
    saida = np.empty_like(xyz)
    for i in range(n):
        ini = max(0, i - meio)
        fim = min(n, i + meio + 1)
        saida[i] = xyz[ini:fim].mean(axis=0)
    return saida


def _coef_sg(janela: int, ordem: int, avaliar_em: float) -> np.ndarray:
    """Coeficientes (tamanho `janela`) do ajuste polinomial avaliado em x."""
    x = np.arange(janela, dtype=float)
    ordem = max(0, min(int(ordem), janela - 1))
    A = np.vander(x, ordem + 1, increasing=True)
    pinv = np.linalg.pinv(A)                 # (ordem+1, janela)
    base = np.array([float(avaliar_em) ** k for k in range(ordem + 1)], float)
    return base @ pinv                       # (janela,)


def savgol(xyz: np.ndarray, janela: int, ordem: int, causal: bool,
           preditivo: bool = False) -> np.ndarray:
    """Savitzky-Golay causal/offline/preditivo.

    - causal: ajusta aos ultimos `janela` samples e avalia na amostra ATUAL
      (o atraso intrinseco e' (janela-1)/2, igual ao do filtro centrado, mas
      so' precisa do passado);
    - offline: `scipy.signal.savgol_filter(mode='interp')` (centrado);
    - preditivo: avalia em x=janela, isto e', um passo A FRENTE da ultima
      amostra da janela (atraso zero, extrapola -- perde a forma nas inversoes).

    Sem scipy, o caminho offline e' feito aqui com as mesmas equacoes (o
    resultado e' o mesmo polinomio, so' sem a implementacao otimizada).
    """
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    janela = max(3, int(janela))
    if janela % 2 == 0:
        janela += 1
    if janela > n:
        janela = n if n % 2 else max(3, n - 1)
    if janela < 3 or n < 3:
        return xyz.copy()
    if causal:
        saida = np.array(xyz, copy=True)
        avaliar = float(janela) if preditivo else float(janela - 1)
        c = _coef_sg(janela, ordem, avaliar)        # (janela,)
        janelas = np.lib.stride_tricks.sliding_window_view(xyz, janela, axis=0)
        valores = np.einsum("w,ncw->nc", c, janelas)
        if preditivo:
            # A janela que TERMINA em i-1 produz a amostra i: descarta a ultima
            # janela (ela terminaria em n-1 e preveria n, fora do sinal).
            saida[janela:] = valores[:-1]
        else:
            saida[janela - 1:] = valores           # janela TERMINA em i
        return saida
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(xyz, janela, ordem, axis=0, mode="interp")
    except ImportError:
        return _savgol_offline_sem_scipy(xyz, janela, ordem)


def _savgol_offline_sem_scipy(xyz: np.ndarray, janela: int, ordem: int) -> np.ndarray:
    n = len(xyz)
    meio = janela // 2
    saida = np.array(xyz, copy=True)
    for i in range(meio, n - meio):
        trecho = xyz[i - meio:i + meio + 1]
        saida[i] = _aplica_poly(trecho, ordem, meio)
    return saida


def _aplica_poly(trecho: np.ndarray, ordem: int, centro: int) -> np.ndarray:
    x = np.arange(len(trecho), dtype=float)
    ordem = min(ordem, len(trecho) - 1)
    A = np.vander(x, ordem + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(A, trecho, rcond=None)
    base = np.array([float(centro) ** k for k in range(ordem + 1)])
    return base @ coef


def ema(xyz: np.ndarray, alpha: float = ALPHA_EMA) -> np.ndarray:
    saida = np.empty_like(np.asarray(xyz, float))
    anterior = None
    for i, ponto in enumerate(np.asarray(xyz, float)):
        anterior = ponto if anterior is None else (1 - alpha) * anterior + alpha * ponto
        saida[i] = anterior
    return saida


def savgol_fixed_lag(xyz: np.ndarray, janela: int, ordem: int) -> np.ndarray:
    """SavGol CAUSAL com atraso fixo de (janela-1)//2 frames.

    E' o SavGol offline ATRASADO: `y[i] = offline[i - m]`, m = (janela-1)//2.
    Por construcao entrega o MESMO jitter do filtro offline -- o preco e' a
    latencia de m frames (o orcamento que o online teria de pagar). E' o
    "analogo causal" honesto da suavizacao offline; a versao avaliada no ultimo
    sample (`savgol(..., causal=True)`) e' o analogo de atraso zero.
    """
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    if n < 4:
        return xyz.copy()
    janela = max(3, int(janela))
    if janela % 2 == 0:
        janela += 1
    if janela > n:
        janela = n if n % 2 else max(3, n - 1)
    if janela < 3:
        return xyz.copy()
    atraso = (janela - 1) // 2
    offline = savgol(xyz, janela, ordem, causal=False)
    saida = np.array(xyz, copy=True)
    if janela - 1 < n:
        saida[janela - 1:] = offline[janela - 1 - atraso:n - atraso]
    return saida


def _detrend(xyz: np.ndarray) -> np.ndarray:
    """Remove media e tendencia linear (por eixo) antes de correlacionar.

    Sem isso, numa trajetoria com deriva a correlacao cruzada e' dominada pelo
    movimento lento e o atraso sai sempre zero.
    """
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    if n < 3:
        return xyz - xyz.mean(axis=0, keepdims=True)
    eixo = np.linspace(-1.0, 1.0, n)
    A = np.vstack([np.ones(n), eixo]).T
    coef, *_ = np.linalg.lstsq(A, xyz, rcond=None)
    return xyz - A @ coef


def _perfil_velocidade(xyz: np.ndarray) -> np.ndarray:
    """Norma da diferenca frame-a-frame (diagnostico; nao usada no atraso)."""
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return np.concatenate([[d[0] if d.size else 0.0], d])


def estima_atraso_frames(cru: np.ndarray, filtrado: np.ndarray,
                         max_atraso: int = 20) -> int:
    """Atraso (frames) do filtrado em relacao ao cru, por correlacao cruzada.

    Correlaciona a POSICAO com media e tendencia removidas (uma correlacao por
    eixo, normalizada, somada). Positivo = o filtrado esta' ATRASADO.

    Por que nao o perfil de velocidade: medido neste projeto, correlacionar a
    norma da diferenca (que e' um sinal retificado) enviesava o atraso em +2
    frames para a media movel causal (dava 4 onde a teoria e o teste com
    deslocamento conhecido dizem 2). A posicao detrendada recupera exatamente
    deslocamentos conhecidos (teste de 0 a 3 frames) e concorda com a teoria:
    media movel causal janela 5 -> 2 frames; SavGol centrado -> 0; SavGol
    endpoint -> 0 (atraso DC nulo); SavGol fixed-lag janela 7 -> 3.
    """
    a = _detrend(cru)
    b = _detrend(filtrado)
    n = min(len(a), len(b))
    if n < 10:
        return 0
    a, b = a[:n], b[:n]
    melhor_lag = 0
    melhor = -np.inf
    # Pelo menos 8 amostras sobrepostas: sem este limite, um `lag` maior que o
    # trecho faz o slice negativo "dar a volta" e as duas series ficam com
    # tamanhos diferentes (bug real, trecho curto de uma sessao de bancada).
    max_atraso = min(int(max_atraso), max(0, n - 8))
    for lag in range(-max_atraso, max_atraso + 1):
        if lag >= 0:
            x, y = a[:n - lag], b[lag:]
        else:
            x, y = a[-lag:], b[:n + lag]
        if len(x) != len(y) or len(x) < 8:
            continue
        corr = 0.0
        valido = 0
        for eixo in range(a.shape[1]):
            u = x[:, eixo] - x[:, eixo].mean()
            v = y[:, eixo] - y[:, eixo].mean()
            du, dv = float(np.std(u)), float(np.std(v))
            if du < 1e-12 or dv < 1e-12:
                continue
            corr += float(np.mean(u * v) / (du * dv))
            valido += 1
        if not valido:
            continue
        corr /= valido
        if corr > melhor:
            melhor = corr
            melhor_lag = lag
    return melhor_lag


def _atraso_dc(nome: str, janela_mm: int = JANELA_MM,
               janela_sg: int = JANELA_SG, ordem_sg: int = ORDEM_SG) -> float:
    """Atraso de grupo em DC (frames) TEORICO de cada filtro, pelos coeficientes.

    E' a resposta exata para um movimento lento (rampa): y[i] = x[i - atraso].
    Serve de contraprova do atraso medido por correlacao, que tem resolucao de
    1 frame e depende do conteudo de frequencia do trecho:
    - media movel causal: (N-1)/2;
    - EMA: (1-alpha)/alpha;
    - SavGol centrado/offline e previsao: 0;
    - SavGol avaliado no ULTIMO sample: 0 (reproduz rampa exatamente);
    - SavGol fixed-lag: (N-1)//2 (e' o offline atrasado).
    """
    if nome in ("cru", "mm_offline", "sg_offline"):
        return 0.0
    if nome == "mm_causal":
        return (janela_mm - 1) / 2.0
    if nome == "ema_causal":
        return (1.0 - ALPHA_EMA) / ALPHA_EMA
    if nome == "sg_fixed_lag":
        return float((janela_sg - 1) // 2)
    if nome == "sg_causal":
        c = _coef_sg(janela_sg, ordem_sg, float(janela_sg - 1))
        return float(np.sum((janela_sg - 1 - np.arange(janela_sg)) * c))
    if nome == "sg_pred":
        c = _coef_sg(janela_sg, ordem_sg, float(janela_sg))
        return float(np.sum((janela_sg - np.arange(janela_sg)) * c))
    return float("nan")


def compara_suavizacao(sessao: Sessao, fps: float, lado: str,
                       janela_mm: int = JANELA_MM,
                       janela_sg: int = JANELA_SG,
                       ordem_sg: int = ORDEM_SG) -> dict:
    """Tabela de suavizacao para o dorso (landmark 9) de UM lado.

    Metodos: cru, media movel causal/offline, SavGol causal/offline/preditivo,
    EMA causal. Metricas: jitter (std 2a diferenca, mm), reducao de jitter (%),
    atraso medido (frames e ms) e erro de forma depois de compensar o atraso
    (RMS contra o cru alinhado, mm).
    """
    rastro = sessao.rastros.get((lado, IDX_DORSO))
    if rastro is None:
        return {}
    # Escolhe a fonte com maior cobertura: a comparacao de FILTROS tem de ser
    # feita na trajetoria mais completa (a que menos interpola buracos).
    usa_mv = float(np.mean(rastro.mv_valid)) >= float(np.mean(rastro.k_valid))
    bruto = rastro.mv_xyz if usa_mv else rastro.k_xyz
    val = rastro.mv_valid if usa_mv else rastro.k_valid
    fonte = "multiview" if usa_mv else "kinect"
    if int(val.sum()) < 12:
        return {"fonte": fonte, "n": int(val.sum()), "metodos": {}}
    # O filtro precisa de serie continua; usar so' o maior run sem NENHUM
    # buraco escolhia um trecho de ~2 s e o atraso medido ficava dependente de
    # um pedaco curto. O bloco util tolera buracos de ate 5 frames (preenchidos
    # por interpolacao) e as metricas de jitter/forma so' usam os samples que
    # eram validos de verdade.
    ini, fim = _bloco_util(val, max_gap=5)
    bruto_trecho = bruto[ini:fim].copy()
    valido_trecho = np.asarray(val[ini:fim], bool).copy()
    trecho = interpola_buracos(bruto_trecho, valido_trecho)
    n = len(trecho)
    if n < 12:
        return {"fonte": fonte, "n": n, "metodos": {}}
    dt = 1.0 / fps if fps > 0 else float("nan")
    n_preenchidos = int((~valido_trecho).sum())

    metodos = {
        "cru": (trecho, True),
        "mm_causal": (media_movel(trecho, janela_mm, True), True),
        "mm_offline": (media_movel(trecho, janela_mm, False), False),
        "sg_causal": (savgol(trecho, janela_sg, ordem_sg, True), True),
        "sg_fixed_lag": (savgol_fixed_lag(trecho, janela_sg, ordem_sg), True),
        "sg_offline": (savgol(trecho, janela_sg, ordem_sg, False), False),
        "sg_pred": (savgol(trecho, janela_sg, ordem_sg, True, preditivo=True),
                    True),
        "ema_causal": (ema(trecho, ALPHA_EMA), True),
    }
    # Bordas fora da comparacao: os metodos causais comecam com amostra crua
    # (sem janela cheia) e, sem este corte, o aquecimento entraria como "jitter
    # do filtro" -- medido: o fixed-lag aparecia com 26% de reducao quando o
    # correto era 72% (o mesmo do offline).
    trim = max(janela_sg - 1, janela_mm)
    if n - 2 * trim < 8:
        trim = 0
    base = trecho[trim:n - trim] if trim else trecho
    valido_base = valido_trecho[trim:n - trim] if trim else valido_trecho
    jitter_cru = _jitter2_mascara(base, valido_base)
    tabela = {}
    for nome, (serie, causal) in metodos.items():
        avaliado = serie[trim:n - trim] if trim else serie
        atraso = estima_atraso_frames(base, avaliado)
        alinhado = _desloca(avaliado, atraso)
        desvio = np.sum((alinhado - base) ** 2, axis=1)
        bons = valido_base & np.isfinite(desvio)
        erro = (float(np.sqrt(np.mean(desvio[bons]))) if int(bons.sum()) >= 3
                else float("nan"))
        jitter = _jitter2_mascara(avaliado, valido_base)
        tabela[nome] = {
            "causal": causal,
            "jitter_mm": jitter * 1000.0,
            "reducao_pct": ((jitter_cru - jitter) / jitter_cru * 100.0
                            if jitter_cru and jitter_cru > 0 else float("nan")),
            "atraso_frames": atraso,
            "atraso_ms": atraso * dt * 1000.0,
            "atraso_dc_frames": _atraso_dc(nome, janela_mm, janela_sg,
                                           ordem_sg),
            "erro_forma_mm": erro * 1000.0,
            "curva": serie,
        }
    return {
        "fonte": fonte,
        "n": n,
        "n_validos": int(valido_trecho.sum()),
        "n_preenchidos": n_preenchidos,
        "trim": trim,
        "dt_s": dt,
        "janela_mm": janela_mm,
        "janela_sg": janela_sg,
        "ordem_sg": ordem_sg,
        "trecho": trecho,
        "t": sessao.tempo_relativo(rastro)[ini:fim],
        "metodos": tabela,
    }


def _maior_trecho_valido(val: np.ndarray) -> tuple[int, int]:
    melhor = (0, 0)
    for ini, fim in _runs(val):
        if fim - ini > melhor[1] - melhor[0]:
            melhor = (ini, fim)
    return melhor


def _bloco_util(val: np.ndarray, max_gap: int = 5) -> tuple[int, int]:
    """Maior bloco continuo de amostras validas, tolerando buracos curtos.

    O filtro precisa de uma serie continua, mas exigir 100% de validade
    escolhia um trecho de 2 s (o maior run sem nenhum buraco) e a comparacao
    de filtros ficava dependente de um pedaco curto. Buracos de ate
    `max_gap` frames sao preenchidos por interpolacao (e as metricas de jitter/
    forma so' usam as amostras ORIGINALMENTE validas).
    """
    util = np.asarray(val, bool).copy()
    for ini, fim in _runs(~util):
        if fim - ini <= max_gap:
            util[ini:fim] = True
    return _maior_trecho_valido(util)


def _jitter2_mascara(xyz: np.ndarray, mask: np.ndarray) -> float:
    """Jitter (RMS da norma da 2a diferenca) so' onde as 3 amostras sao validas."""
    xyz = np.asarray(xyz, float)
    mask = np.asarray(mask, bool)
    if len(xyz) < 4:
        return float("nan")
    d2 = xyz[:-2] - 2.0 * xyz[1:-1] + xyz[2:]
    bons = mask[:-2] & mask[1:-1] & mask[2:] & np.isfinite(d2).all(axis=1)
    if int(bons.sum()) < 3:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(d2[bons] ** 2, axis=1))))


def _desloca(serie: np.ndarray, atraso: int) -> np.ndarray:
    """Desloca `serie` de `atraso` frames para comparar com o cru alinhado."""
    atraso = int(atraso)
    if atraso == 0:
        return serie
    saida = np.array(serie, copy=True)
    if atraso > 0:
        saida[:-atraso] = serie[atraso:]
        saida[-atraso:] = serie[-1]
    else:
        saida[-atraso:] = serie[:atraso]
        saida[:-atraso] = serie[0]
    return saida


def _jitter2(xyz: np.ndarray) -> float:
    """Raiz da media da NORMA da segunda diferenca (m)."""
    xyz = np.asarray(xyz, float)
    if len(xyz) < 4:
        return float("nan")
    d2 = xyz[:-2] - 2.0 * xyz[1:-1] + xyz[2:]
    valores = np.linalg.norm(d2, axis=1)
    valores = valores[np.isfinite(valores)]
    if valores.size < 3:
        return float("nan")
    return float(np.sqrt(np.mean(valores ** 2)))


# --------------------------------------------------------------------------
# Graficos
# --------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def gera_pngs(sessao: Sessao, cobertura: dict, jitter: dict, reproj: dict,
              suavizacao: dict, dir_saida: Path, prefixo: str = "tracking") -> list[Path]:
    """4 PNGs: cobertura por landmark, jitter, reprojecao e suavizacao."""
    plt = _plt()
    dir_saida = Path(dir_saida)
    dir_saida.mkdir(parents=True, exist_ok=True)
    gerados: list[Path] = []
    lados = sessao.lados or ["right"]
    cores = {"right": "#1f77b4", "left": "#d62728"}

    # --- 1) cobertura por landmark ---------------------------------------
    # Linhas (e nao barras) porque a cobertura vive em 88-100%: barra com eixo
    # completo nao mostra diferenca e barra com eixo truncado mente no
    # comprimento. A linha com marcador + eixo declarado e' honesta nos dois
    # sentidos.
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    x = np.arange(21)
    valores_validos = []
    for k, lado in enumerate(lados):
        dados = cobertura["por_lado"].get(lado)
        if not dados:
            continue
        for campo, marcador, estilo, rotulo in (
            ("por_landmark_kinect", "o", "-", f"Kinect depth ({lado})"),
            ("por_landmark_mv", "s", "--", f"Multi-view ({lado})"),
        ):
            valores = np.asarray(dados[campo], float) * 100.0
            valores_validos += [v for v in valores if np.isfinite(v)]
            ax.plot(x, valores, estilo, marker=marcador, ms=4, lw=1.4,
                    color=cores.get(lado, "C0"),
                    alpha=0.55 if campo == "por_landmark_kinect" else 1.0,
                    label=rotulo)
    piso = 50.0
    if valores_validos:
        piso = min(50.0, math.floor(min(valores_validos) / 10.0) * 10.0)
    ax.axvline(IDX_DORSO, color="k", ls=":", lw=1.2)
    ax.annotate("dorso\n(landmark 9)", (IDX_DORSO, piso), textcoords="offset points",
                xytext=(6, 4), fontsize=8)
    ax.axhline(CRITERIOS["cobertura_min"] * 100, color="g", ls="--", lw=1,
               label=f"criterio {CRITERIOS['cobertura_min']*100:.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}:{n}" for i, n in enumerate(LANDMARK_NOMES)],
                       rotation=90, fontsize=6.5)
    ax.set_ylabel("cobertura (% frames com 3D valido)")
    ax.set_ylim(piso, 101.5)
    ax.set_title("Cobertura 3D por landmark - Kinect depth x multi-view\n"
                 f"(eixo vertical comeca em {piso:.0f}%)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    caminho = dir_saida / f"{prefixo}_cobertura.png"
    fig.savefig(caminho)
    plt.close(fig)
    gerados.append(caminho)

    # --- 2) jitter em pose estatica --------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.0), dpi=140)
    for k, lado in enumerate(lados):
        dados_lm = jitter["por_landmark"].get(lado, {})
        valores_k = [dados_lm.get(i, {}).get("jitter_kinect_mm", np.nan)
                     for i in range(21)]
        valores_mv = [dados_lm.get(i, {}).get("jitter_mv_mm", np.nan)
                      for i in range(21)]
        ax.plot(x, valores_k, "-o", ms=3, lw=1.1, alpha=0.6,
                color=cores.get(lado, "C0"), label=f"Kinect depth ({lado})")
        ax.plot(x, valores_mv, "-s", ms=3, lw=1.3,
                color=cores.get(lado, "C0"), label=f"Multi-view ({lado})")
    ax.axvline(IDX_DORSO, color="k", ls=":", lw=1.2)
    ax.annotate("dorso (9)", (IDX_DORSO, 0), textcoords="offset points",
                xytext=(6, 6), fontsize=8)
    ax.axhline(CRITERIOS["jitter_max_mm"], color="g", ls="--", lw=1,
               label=f"criterio {CRITERIOS['jitter_max_mm']:g} mm")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}:{n}" for i, n in enumerate(LANDMARK_NOMES)],
                       rotation=90, fontsize=6.5)
    ax.set_ylabel("jitter (desvio-padrao 3D, mm)")
    ax.set_title("Jitter em pose estatica por landmark")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    caminho = dir_saida / f"{prefixo}_jitter.png"
    fig.savefig(caminho)
    plt.close(fig)
    gerados.append(caminho)

    # --- 3) suavizacao ----------------------------------------------------
    # Tres paineis: trajetoria completa (contexto), zoom de 1,5 s no trecho
    # mais RAPIDO (e' onde o atraso do filtro causal aparece) e jitter por
    # metodo. O zoom e' o que mostra o efeito, porque o jitter (mm) e' pequeno
    # frente ao deslocamento (cm).
    fig = plt.figure(figsize=(11.5, 6.2), dpi=140)
    grade = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0],
                             height_ratios=[1, 1], hspace=0.55, wspace=0.22)
    ax_traj = fig.add_subplot(grade[0, 0])
    ax_zoom = fig.add_subplot(grade[1, 0])
    ax_bar = fig.add_subplot(grade[:, 1])
    desenhado = False
    primeiro = None
    for lado in lados:
        dados = suavizacao.get("por_lado", {}).get(lado)
        if dados and dados.get("metodos"):
            primeiro = (lado, dados)
            break
    for lado in lados:
        dados = suavizacao.get("por_lado", {}).get(lado)
        if not dados or not dados.get("metodos"):
            continue
        t = dados["t"]
        base_t = t[0] if np.isfinite(t[0]) else 0.0
        trecho = dados["trecho"]
        metodos = dados["metodos"]
        ax_traj.plot(t - base_t, trecho[:, 2] * 1000.0, color="0.55", lw=1.0,
                     label=f"cru - eixo Z ({lado})")
        if "sg_offline" in metodos:
            ax_traj.plot(t - base_t, metodos["sg_offline"]["curva"][:, 2] * 1000.0,
                         color="#2ca02c", lw=1.6,
                         label="SavGol offline (centrado)")
        if "sg_causal" in metodos:
            ax_traj.plot(t - base_t, metodos["sg_causal"]["curva"][:, 2] * 1000.0,
                         color="#d62728", lw=1.6, ls="--", label="SavGol causal")
        if "ema_causal" in metodos:
            ax_traj.plot(t - base_t, metodos["ema_causal"]["curva"][:, 2] * 1000.0,
                         color="#9467bd", lw=1.2, ls=":", label="EMA causal")
        desenhado = True
    if desenhado:
        ax_traj.set_xlabel("tempo no trecho analisado (s)")
        ax_traj.set_ylabel("Z do dorso (mm)")
        ax_traj.legend(fontsize=7)
        ax_traj.set_title("Trajetoria do dorso: cru x causal x offline")
        ax_traj.grid(alpha=0.25)
    else:
        ax_traj.text(0.5, 0.5, "sem dados suficientes", ha="center", va="center",
                     transform=ax_traj.transAxes)
        ax_traj.set_axis_off()

    # Zoom no trecho mais rapido (maior velocidade media em 1,5 s).
    if primeiro is not None:
        lado, dados = primeiro
        t = dados["t"]
        trecho = dados["trecho"]
        metodos = dados["metodos"]
        base_t = t[0] if np.isfinite(t[0]) else 0.0
        fps_local = 1.0 / dados["dt_s"] if dados.get("dt_s") else fps
        janela = max(8, int(round(1.5 * (fps_local or 25.0))))
        janela = min(janela, max(8, len(trecho) - 2))
        velocidade = _perfil_velocidade(trecho)
        inicio = 0
        melhor = -1.0
        for i in range(0, max(1, len(trecho) - janela)):
            media = float(np.mean(velocidade[i:i + janela]))
            if media > melhor:
                melhor = media
                inicio = i
        fim = inicio + janela
        eixo_t = (t - base_t)[inicio:fim]
        ax_zoom.plot(eixo_t, trecho[inicio:fim, 2] * 1000.0, color="0.45",
                     lw=1.2, label="cru")
        if "sg_offline" in metodos:
            ax_zoom.plot(eixo_t, metodos["sg_offline"]["curva"][inicio:fim, 2] * 1000.0,
                         color="#2ca02c", lw=1.8, label="SavGol offline")
        if "sg_fixed_lag" in metodos:
            ax_zoom.plot(eixo_t, metodos["sg_fixed_lag"]["curva"][inicio:fim, 2] * 1000.0,
                         color="#ff7f0e", lw=1.4, ls="-.", label="SavGol fixed-lag")
        if "sg_causal" in metodos:
            ax_zoom.plot(eixo_t, metodos["sg_causal"]["curva"][inicio:fim, 2] * 1000.0,
                         color="#d62728", lw=1.4, ls="--", label="SavGol causal")
        if "ema_causal" in metodos:
            ax_zoom.plot(eixo_t, metodos["ema_causal"]["curva"][inicio:fim, 2] * 1000.0,
                         color="#9467bd", lw=1.2, ls=":", label="EMA causal")
        ax_zoom.set_xlabel("tempo (s)")
        ax_zoom.set_ylabel("Z do dorso (mm)")
        ax_zoom.set_title(f"Zoom de 1,5 s no trecho mais rapido ({lado})")
        ax_zoom.legend(fontsize=6.5, ncol=2)
        ax_zoom.grid(alpha=0.25)
    else:
        ax_zoom.text(0.5, 0.5, "sem dados", ha="center", va="center",
                     transform=ax_zoom.transAxes)
        ax_zoom.set_axis_off()

    ax = ax_bar
    nomes = ["cru", "mm_causal", "mm_offline", "sg_causal", "sg_fixed_lag",
             "sg_offline", "sg_pred", "ema_causal"]
    deslocamento = 0.0
    for lado in lados:
        dados = suavizacao.get("por_lado", {}).get(lado)
        if not dados or not dados.get("metodos"):
            continue
        jitters = [dados["metodos"].get(n, {}).get("jitter_mm", np.nan)
                   for n in nomes]
        ax.bar(np.arange(len(nomes)) + deslocamento, jitters, 0.38,
               color=cores.get(lado, "C0"), alpha=0.75, label=lado)
        deslocamento += 0.38
    ax.set_xticks(np.arange(len(nomes)))
    ax.set_xticklabels(nomes, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("jitter (mm)")
    ax.set_title("Jitter por metodo de suavizacao")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(dir_saida / f"{prefixo}_suavizacao.png", bbox_inches="tight")
    plt.close(fig)
    gerados.append(dir_saida / f"{prefixo}_suavizacao.png")

    # --- 4) reprojecao ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=140)
    dados_plot = []
    rotulos = []
    for lado in lados:
        for lid in range(21):
            valores = []
            rastro = sessao.rastros.get((lado, lid))
            if rastro is not None and rastro.mv_valid.any():
                e = rastro.reproj_kin_px[rastro.mv_valid]
                valores = e[np.isfinite(e)].tolist()
            dados_plot.append(valores if valores else [np.nan])
            rotulos.append(f"{lado[0]}{lid}")
    try:
        ax.boxplot(dados_plot, showfliers=False, widths=0.6)
        ax.set_xticks(np.arange(1, len(rotulos) + 1))
        ax.set_xticklabels(rotulos, rotation=90, fontsize=6)
    except (ValueError, IndexError):
        ax.text(0.5, 0.5, "sem triangulacao valida", ha="center", va="center",
                transform=ax.transAxes)
    ax.axhline(CRITERIOS["reproj_max_px"], color="g", ls="--", lw=1,
               label=f"gate {CRITERIOS['reproj_max_px']:g} px")
    ax.set_ylim(-0.2, CRITERIOS["reproj_max_px"] * 1.08)
    ax.set_ylabel("reprojecao no Kinect (px)")
    ax.set_title("Erro de reprojecao do multi-view por landmark\n"
                 "(so' os pontos que passaram no gate entram no grafico)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    caminho = dir_saida / f"{prefixo}_reprojecao.png"
    fig.savefig(caminho)
    plt.close(fig)
    gerados.append(caminho)
    return gerados


# --------------------------------------------------------------------------
# Relatorio
# --------------------------------------------------------------------------
def _num(valor, casas=2, sufixo="", vazio="--") -> str:
    if valor is None:
        return vazio
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return str(valor)
    if not math.isfinite(v):
        return vazio
    return f"{v:.{casas}f}{sufixo}"


def _pct(valor, casas=1) -> str:
    return _num(valor * 100.0, casas, "%") if valor is not None else "--"


def _tabela(cabecalho: list[str], linhas: list[list[str]]) -> str:
    texto = ["| " + " | ".join(cabecalho) + " |",
             "| " + " | ".join(["---"] * len(cabecalho)) + " |"]
    for linha in linhas:
        texto.append("| " + " | ".join(linha) + " |")
    return "\n".join(texto)


def _veredito_criterios(cobertura: dict, jitter: dict, saltos: dict,
                        reproj: dict) -> list[dict]:
    """Uma linha por (metodo, criterio) com PASSA/NAO PASSA/NA.

    E' de proposito que nao exista "nota final": o pedido era decidir com
    dados, e um escore composto arbitrario esconderia justamente o criterio que
    falhou. O que existe e' a contagem de criterios atendidos.
    """
    linhas = []
    for lado in cobertura["por_lado"]:
        cob = cobertura["por_lado"][lado]
        jit = jitter["por_lado"].get(lado, {})
        sal = saltos["por_lado"].get(lado, {})
        rep = reproj["por_lado"].get(lado, {})
        for metodo, cob_v, jit_v, sal_v, rep_v in (
            ("kinect",
             cob.get("cobertura_kinect"), jit.get("jitter_kinect"),
             sal.get("kinect", {}).get("fracao"), rep.get("k_self_mediana")),
            ("multiview",
             cob.get("cobertura_mv"), jit.get("jitter_mv"),
             sal.get("mv", {}).get("fracao"), rep.get("mv_kin_mediana")),
        ):
            ok_cob = _atende(cob_v, CRITERIOS["cobertura_min"], maior_melhor=True)
            ok_jit = _atende(jit_v, CRITERIOS["jitter_max_mm"], maior_melhor=False)
            ok_sal = _atende(sal_v, CRITERIOS["salto_max_frac"], maior_melhor=False)
            # O criterio de reprojecao NAO se aplica ao Kinect: a reprojecao
            # dele e' ida-e-volta (o ponto 3D nasceu do mesmo pixel). Marcar
            # PASSA ali seria dar credito por uma medida que nao existe.
            ok_rep = ("NA" if metodo == "kinect"
                      else _atende(rep_v, CRITERIOS["reproj_max_px"],
                                   maior_melhor=False))
            for criterio, valor, ok in (
                ("cobertura", cob_v, ok_cob),
                ("jitter", jit_v, ok_jit),
                ("continuidade", sal_v, ok_sal),
                ("reprojecao", rep_v, ok_rep),
            ):
                linhas.append({"lado": lado, "metodo": metodo,
                               "criterio": criterio, "valor": valor,
                               "resultado": ok})
    return linhas


def _atende(valor, limite: float, maior_melhor: bool) -> str:
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(v):
        return "NA"
    if maior_melhor:
        return "PASSA" if v >= limite else "NAO PASSA"
    return "PASSA" if v <= limite else "NAO PASSA"


def _wh(valor) -> str:
    """'1280x720' a partir de [1280, 720] (ou '--')."""
    if not valor:
        return "--"
    try:
        return f"{int(valor[0])}x{int(valor[1])}"
    except (TypeError, ValueError, IndexError):
        return str(valor)


def _setup_markdown(sessao: Sessao) -> list[str]:
    meta = sessao.meta or {}
    linhas = []
    cal = meta.get("calibracao") or {}
    linhas.append(f"- **Camera auxiliar (indice OpenCV):** "
                  f"`{meta.get('aux_index', '?')}`"
                  f"{' - ' + str(meta.get('aux_camera_nome')) if meta.get('aux_camera_nome') else ''}")
    if cal:
        linhas.append(f"- **Arquivo de calibracao stereo:** `{cal.get('arquivo', '?')}`"
                      f" | RMS {_num(cal.get('rms'), 2, ' px')}"
                      f" | tabuleiro {cal.get('tabuleiro', '?')}"
                      f" | quadrado {_num(cal.get('quadrado_mm'), 1, ' mm')}"
                      f" | estado: {cal.get('estado', '?')} ({cal.get('motivo', '')})")
        linhas.append(f"- **Resolucao:** calibrada {_wh(cal.get('aux_size'))}"
                      f"; usada {_wh(meta.get('aux_resolucao_usada'))}"
                      f" | Kinect {_wh(meta.get('kinect_resolucao'))}"
                      f" (calibrado {_wh(cal.get('kinect_size'))})")
    versoes = meta.get("versoes") or {}
    if versoes:
        linhas.append("- **Versoes:** " + ", ".join(
            f"{k} {v}" for k, v in versoes.items()))
    if meta.get("kinect_intrinsics_nota"):
        linhas.append(f"- **Intrinsecas do Kinect:** {meta['kinect_intrinsics_nota']}")
    if meta.get("espelhamento"):
        linhas.append(f"- **Espelhamento da auxiliar:** {meta['espelhamento']}")
    if meta.get("comando"):
        linhas.append(f"- **Comando:** `{meta['comando']}`")
    return linhas


def gera_relatorio(sessao: Sessao, dir_saida: Path, arquivos: dict,
                   prefixo: str = "tracking") -> Path:
    """Escreve o relatorio em Markdown a partir dos CSV.

    Nome: `relatorio_tracking.md` (o pedido da tese). No modo SINTETICO o nome
    carrega o prefixo (`SINTETICO_..._relatorio_tracking.md`) para um relatorio
    de validacao de codigo nunca se confundir com o de bancada.
    """
    dir_saida = Path(dir_saida)
    dir_saida.mkdir(parents=True, exist_ok=True)
    fps = _fps_da_sessao(sessao.frames)
    cobertura = metricas_cobertura(sessao)
    lateralidade = metricas_lateralidade(sessao)
    depth = metricas_depth(sessao)
    reproj = metricas_reprojecao(sessao)
    segmentos = {lado: segmentos_estaticos(sessao, lado, fps)
                 for lado in sessao.lados}
    jitter = metricas_jitter(sessao, fps, segmentos)
    saltos = metricas_saltos(sessao, float(sessao.meta.get("salto_mm", 50.0)))
    latencia = metricas_latencia(sessao)
    suavizacao = {"por_lado": {}}
    for lado in sessao.lados:
        suavizacao["por_lado"][lado] = compara_suavizacao(sessao, fps, lado)

    # --- dados de sessao ---------------------------------------------------
    t_rel = _tempo_relativo(sessao.frames)
    duracao = float(np.nanmax(t_rel) - np.nanmin(t_rel)) if len(t_rel) else 0.0
    n_frames = len({str(r.get("seq")) for r in sessao.frames})
    linhas: list[str] = []
    origem = "SINTETICO (validacao de codigo -- NAO e' bancada)" \
        if sessao.sintetica else "bancada (hardware real)"
    linhas.append("# Relatorio de rastreamento da mao - Kinect (depth/MediaPipe) x multi-view")
    linhas.append("")
    linhas.append(f"- **Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    linhas.append(f"- **Origem dos dados:** {origem}")
    linhas.append(f"- **Arquivos lidos:** `{arquivos.get('frames', '?')}`, "
                  f"`{arquivos.get('landmarks', '?')}`")
    linhas.append(f"- **Frames:** {n_frames} em {_num(duracao, 1, ' s')}"
                  f" | FPS efetivo {_num(fps, 1)}")
    lados = ", ".join(sessao.lados) or "nenhum"
    linhas.append(f"- **Lados com mao detectada:** {lados}"
                  f" | frames sem mao: {cobertura.get('frames_sem_mao', 0)}")
    linhas.append("- **Referencial dos 3D:** camera de COR do Kinect, metros, eixos "
                  "OpenCV (X direita, Y para baixo, Z para frente). O Y do camera "
                  "space do Kinect foi invertido para casar com a triangulacao.")
    linhas.append("")
    linhas.append("## 1. Setup e calibracao")
    linhas.append("")
    setup = _setup_markdown(sessao)
    linhas.extend(setup if setup else ["- (sem arquivo `_meta.json` ao lado do CSV)"])
    linhas.append("")

    # --- 2. metodo --------------------------------------------------------
    linhas.append("## 2. O que foi medido")
    linhas.append("")
    linhas.append("Dois caminhos independentes para a MESMA mao:")
    linhas.append("")
    linhas.append("1. **Kinect depth**: MediaPipe no RGB do Kinect -> pixel do "
                  "landmark -> mapeamento cor->depth -> mediana de depth -> "
                  "`MapDepthPointToCameraSpace` (mesmo caminho de "
                  "`KinectHandTracker._landmark_camera_point`).")
    linhas.append("2. **Multi-view**: MediaPipe no RGB do Kinect E na auxiliar -> "
                  "triangulacao por DLT com a calibracao stereo "
                  "(`AuxCameraView.triangulate_with_kinect`, o mesmo metodo de "
                  "`AuxStereoSource` e de `tools/multiview_janela.py`).")
    linhas.append("")
    linhas.append("Denominador das metricas: os frames em que aquela mao apareceu "
                  "em PELO MENOS uma camera. Se o Kinect nao viu a mao, isso conta "
                  "como perda de cobertura do Kinect -- nao como frame inexistente.")
    linhas.append("")

    # --- 3. resultados ----------------------------------------------------
    linhas.append("## 3. Resultados")
    linhas.append("")
    linhas.append("### 3.0 Lateralidade e pareamento entre as cameras")
    linhas.append("")
    if lateralidade["comparaveis"]:
        linhas.append(_tabela(
            ["frames comparaveis", "rotulo igual", "rotulo trocado",
             "concordancia"],
            [[str(lateralidade["comparaveis"]), str(lateralidade["iguais"]),
              str(lateralidade["diferentes"]),
              _pct(lateralidade["fracao_igual"])]]))
    else:
        linhas.append("_sem frames com as duas cameras vendo a mao ao mesmo "
                      "tempo: nao da' para comparar lateralidade_")
    if lateralidade["pareamento"]:
        linhas.append("")
        linhas.append("Pareamento usado no CSV: " + ", ".join(
            f"`{modo}` {n}" for modo, n in sorted(lateralidade["pareamento"].items())))
    if lateralidade["diferentes"]:
        linhas.append("")
        aviso = (f"**Atencao:** em {lateralidade['diferentes']} frames as duas "
                 "cameras deram rotulos de lado DIFERENTES para a mesma mao")
        if lateralidade["exemplos"]:
            aviso += " (ex.: " + "; ".join(lateralidade["exemplos"]) + ")"
        aviso += (". O pareamento por lado do pipeline de producao "
                  "(`AuxStereoSource`) nao casaria NADA nesses frames e o "
                  "multi-view ficaria mudo sem explicar o motivo.")
        linhas.append(aviso)
        if lateralidade["fracao_igual"] < 0.9:
            linhas.append("")
            linhas.append("Como a concordancia ficou abaixo de 90%, o teste de "
                          "bancada precisou do pareamento por mao unica "
                          "(`--pareamento unico`/`ambos`): com UMA mao em cada "
                          "camera nao ha' ambiguidade geometrica. O numero de "
                          "cobertura do multi-view desta sessao depende disso e "
                          "esta' registrado na coluna `pareamento` do CSV.")
    linhas.append("")
    linhas.append("### 3.1 Cobertura 3D")
    linhas.append("")
    cab = ["lado", "frames", "Kinect detecta", "Aux detecta", "ambas",
           "cob. Kinect (21)", "cob. Kinect (dorso 9)", "cob. multi-view (21)",
           "cob. multi-view (dorso 9)", "pior landmark Kinect", "pior landmark MV"]
    corpo = []
    for lado, d in cobertura["por_lado"].items():
        corpo.append([lado, str(d["frames_tentados"]), _pct(d["det_kinect"]),
                      _pct(d["det_aux"]), _pct(d["det_ambas"]),
                      _pct(d["cobertura_kinect"]),
                      _pct(d["cobertura_kinect_dorso"]),
                      _pct(d["cobertura_mv"]),
                      _pct(d["cobertura_mv_dorso"]),
                      _pct(d["cobertura_kinect_min"]),
                      _pct(d["cobertura_mv_min"])])
    linhas.append(_tabela(cab, corpo) if corpo else "_nenhum lado com deteccao_")
    linhas.append("")
    for lado, d in cobertura["por_lado"].items():
        linhas.append(f"- `{lado}`: cobertura minima por landmark - Kinect "
                      f"{_pct(d['cobertura_kinect_min'])}, multi-view "
                      f"{_pct(d['cobertura_mv_min'])}.")
    linhas.append("")

    linhas.append("### 3.2 Falhas de depth e de deteccao (Kinect)")
    linhas.append("")
    cab = ["lado", "landmarks tentados", "sem pixel de depth (mapeamento)",
           "sem leitura de depth", "resgatados pelo prior", "sem ponto 3D final",
           "frames com queda total", "sync cor-depth (ms)"]
    corpo = []
    for lado, d in depth.items():
        corpo.append([lado, str(d["landmarks_tentados"]), _pct(d["map_fail"]),
                      _pct(d["depth_fail"]), _pct(d["rescue"]),
                      _pct(d["sem_ponto_3d"]),
                      f"{d['frames_queda_total']}/{d['frames_com_mao']}",
                      _num(d["sync_color_depth_ms"], 1)])
    linhas.append(_tabela(cab, corpo) if corpo else "_sem dados_")
    linhas.append("")
    linhas.append("`resgatados pelo prior` = profundidade substituida pelo z "
                  "plausivel da mao (`resgatar_profundidade`): o ponto existe, mas "
                  "o Z NAO foi medido naquele landmark -- e' uma inferencia.")
    linhas.append("")

    linhas.append("### 3.3 Erro de reprojecao")
    linhas.append("")
    cab = ["lado", "multi-view no Kinect (mediana)", "no Kinect (p95)",
           "multi-view na AUXILIAR (mediana)", "na AUXILIAR (p95)",
           "depth 3D na AUXILIAR (mediana)", "depth 3D na AUXILIAR (p95)",
           "Kinect ida-e-volta (mediana)", "n amostras MV"]
    corpo = []
    for lado, d in reproj["por_lado"].items():
        corpo.append([lado, _num(d["mv_kin_mediana"], 2, " px"),
                      _num(d["mv_kin_p95"], 2, " px"),
                      _num(d["mv_aux_mediana"], 2, " px"),
                      _num(d["mv_aux_p95"], 2, " px"),
                      _num(d.get("kd_aux_mediana"), 2, " px"),
                      _num(d.get("kd_aux_p95"), 2, " px"),
                      _num(d["k_self_mediana"], 2, " px"),
                      str(d["n_mv"])])
    linhas.append(_tabela(cab, corpo) if corpo else "_sem triangulacao valida_")
    linhas.append("")
    linhas.append("A reprojecao **na auxiliar** e' a mais informativa: a "
                  "triangulacao usa os pixels das DUAS cameras, mas o erro "
                  "reportado pelo pipeline e' medido nos pixels do Kinect; a "
                  "projecao na auxiliar mostra se o ponto 3D explica a vista que "
                  "ele nao otimizou. A coluna do Kinect (ida-e-volta) mede apenas "
                  "consistencia do mapeamento cor->depth->camera e serve de "
                  "sanity check (valor grande = convencao de eixos errada).")
    linhas.append("")
    linhas.append("A coluna **depth 3D na AUXILIAR** e' o diagnostico que separa "
                  "erro de triangulacao de CORRESPONDENCIA enviesada: ela pega o "
                  "ponto 3D medido pelo depth (independente do multi-view), "
                  "projeta na auxiliar e compara com o pixel que a MediaPipe deu "
                  "ali. Se esse numero e' grande enquanto a reprojecao da "
                  "triangulacao e' pequena, os pixels das duas cameras nao "
                  "descrevem o mesmo ponto fisico -- e nenhuma triangulacao "
                  "conserta isso.")
    linhas.append("")
    achados = []
    for lado, d in reproj["por_lado"].items():
        valor = d.get("kd_aux_mediana")
        if valor is None or not math.isfinite(float(valor)):
            continue
        if float(valor) > 20.0:
            achados.append((lado, float(valor), d.get("mv_kin_mediana")))
        elif float(valor) <= 20.0:
            linhas.append(
                f"- `{lado}`: o 3D do depth projetado na auxiliar cai a "
                f"{_num(valor, 1, ' px')} do landmark da auxiliar -- as duas vistas "
                "estao geometricamente consistentes nesta montagem (o esperado "
                "para uma calibracao boa e uma mao de ~10 cm de espessura).")
            linhas.append("")
    if achados:
        detalhe = "; ".join(
            f"`{lado}`: {_num(valor, 0, ' px')} (triangulacao com residuo de "
            f"{_num(res, 1, ' px')} nos proprios pixels)"
            for lado, valor, res in achados)
        linhas.append(
            "> **Achado deste teste.** O 3D do DEPTH projetado na auxiliar cai "
            f"a centenas de px de onde a MediaPipe ve o mesmo landmark -- {detalhe}. "
            "Isso NAO e' ruido: e' incompatibilidade geometrica entre as duas "
            "vistas (extrinseca desatualizada/montagem deslocada ou "
            "correspondencia de pixels sistematicamente enviesada). Enquanto esse "
            "numero nao cair para a ordem de ~10-20 px, **nenhuma metrica de "
            "posicao do multi-view pode ser lida como qualidade do metodo "
            "multi-view**: o 3D dele e' consistente com os proprios pixels e "
            "errado em relacao a cena. Check-list: (1) recalibrar "
            "`stereo_calibration` com o tabuleiro na MESMA distancia em que a mao "
            "trabalha (1,0-1,4 m); (2) repetir este tool e olhar `depth 3D na "
            "AUXILIAR`; (3) so' depois comparar cobertura/jitter entre metodos.")
        linhas.append("")

    linhas.append("### 3.4 Jitter em pose estatica")
    linhas.append("")
    cab = ["lado", "segmentos estaticos", "amostras", "jitter Kinect (mediana 21)",
           "jitter MV (mediana 21)", "jitter Kinect (dorso)", "jitter MV (dorso)",
           "std 2a dif. Kinect", "std 2a dif. MV"]
    corpo = []
    for lado, d in jitter["por_lado"].items():
        segs = jitter["segmentos"].get(lado) or []
        corpo.append([lado, str(len(segs)), str(d["n_amostras"]),
                      _num(d["jitter_kinect"], 2, " mm"),
                      _num(d["jitter_mv"], 2, " mm"),
                      _num(d["dorso_kinect"], 2, " mm"),
                      _num(d["dorso_mv"], 2, " mm"),
                      _num(d["std2_kinect"], 2, " mm"),
                      _num(d["std2_mv"], 2, " mm")])
    linhas.append(_tabela(cab, corpo) if corpo else "_sem segmentos estaticos_")
    linhas.append("")
    for lado, segs in jitter["segmentos"].items():
        if segs:
            texto = ", ".join(f"{a:.1f}-{b:.1f} s" for a, b in segs[:6])
            linhas.append(f"- `{lado}`: segmentos {texto}"
                          f"{' ...' if len(segs) > 6 else ''}")
        else:
            linhas.append(f"- `{lado}`: nenhum segmento estatico detectado "
                          "(movimento continuo ou mao ausente) -> jitter NAO MEDIDO.")
    fonte_fase = any(str(r.get("fase") or "").strip().lower() in
                     ("estatico", "static", "parado")
                     for r in sessao.frames)
    linhas.append("")
    linhas.append("Segmentos: "
                  + ("**rotulados pelo operador** (tecla E na janela)."
                     if fonte_fase else
                     "**detectados automaticamente** por velocidade da "
                     "trajetoria suavizada (limiar 30 mm/s, pausa minima 0,4 s) "
                     "-- o operador nao marcou a fase nesta sessao."))
    n_amostras = [d["n_amostras"] for d in jitter["por_lado"].values()]
    if n_amostras and max(n_amostras) and max(n_amostras) < 30:
        linhas.append("")
        linhas.append(f"**Atencao:** o maior segmento tem {max(n_amostras)} "
                      "amostras. Desvio-padrao com tao poucas amostras e' "
                      "estimativa FRACA (a incerteza relativa e' ~1/sqrt(2n)): "
                      "para o jitter ser conclusivo, rode uma sessao dedicada "
                      "segurando a mao parada por 5 s com a tecla `E` "
                      "pressionada.")
    linhas.append("")

    linhas.append("### 3.5 Continuidade e outliers")
    linhas.append("")
    cab = ["lado", "metodo", "pares consecutivos", "saltos",
           f"> {_num(saltos['limiar_mm'], 0)} mm", "maior salto"]
    corpo = []
    for lado, d in saltos["por_lado"].items():
        for metodo in ("kinect", "mv"):
            m = d[metodo]
            corpo.append([lado, metodo, str(m["pares"]), str(m["saltos"]),
                          _pct(m["fracao"]), _num(m["maior_mm"], 1, " mm")])
    linhas.append(_tabela(cab, corpo) if corpo else "_sem pares consecutivos_")
    linhas.append("")
    linhas.append(f"Limiar: {_num(saltos['limiar_mm'], 0)} mm entre frames "
                  f"consecutivos ({_num(saltos['limiar_mm']/1000.0*fps, 2)} m/s "
                  f"em {_num(fps, 1)} fps). Movimento rapido legitimo tambem passa "
                  "do limiar: a coluna 'maior salto' existe para separar pico "
                  "isolado de reaquisicao errada.")
    linhas.append("")

    linhas.append("### 3.6 FPS e custo por etapa")
    linhas.append("")
    cab = ["etapa", "mediana (ms)", "p95 (ms)", "n"]
    corpo = []
    for etapa in ("kinect_detect", "aux_detect", "triangulacao", "laco"):
        d = latencia.get(etapa, {})
        corpo.append([etapa, _num(d.get("mediana"), 1), _num(d.get("p95"), 1),
                      str(d.get("n", 0))])
    linhas.append(_tabela(cab, corpo))
    linhas.append("")
    linhas.append(f"FPS efetivo do laco: **{_num(fps, 1)}**. A API do MediaPipe "
                  "(VIDEO) nao expoe a latencia interna do modelo; `kinect_detect` "
                  "e `aux_detect` sao o tempo de parede de cada chamada "
                  "`detect_for_video`, que e' o piso da latencia ponta-a-ponta.")
    linhas.append("")

    # --- 4. dorso x 21 ----------------------------------------------------
    linhas.append("## 4. Dorso (landmark 9) x os 21 landmarks")
    linhas.append("")
    for lado in sessao.lados:
        cob = cobertura["por_lado"].get(lado, {})
        jit = jitter["por_lado"].get(lado, {})
        sal = saltos["por_lado"].get(lado, {})
        rep = reproj["por_lado"].get(lado, {})
        por_lm = reproj["por_landmark"].get(lado, {})
        valores_mv = [v for v in por_lm.get("mv_aux", []) if np.isfinite(v)]
        valores_k = [v for v in por_lm.get("k_self", []) if np.isfinite(v)]
        linha = (f"- `{lado}`: cobertura do dorso Kinect "
                 f"{_pct(cob.get('cobertura_kinect_dorso'))} vs media 21 "
                 f"{_pct(cob.get('cobertura_kinect'))}; multi-view "
                 f"{_pct(cob.get('cobertura_mv_dorso'))} vs "
                 f"{_pct(cob.get('cobertura_mv'))}.")
        linha += (f" Jitter do dorso: Kinect {_num(jit.get('dorso_kinect'), 2, ' mm')}"
                  f", MV {_num(jit.get('dorso_mv'), 2, ' mm')}"
                  f" (mediana 21: {_num(jit.get('jitter_kinect'), 2)} / "
                  f"{_num(jit.get('jitter_mv'), 2)} mm).")
        linha += (f" Saltos: Kinect {_num(sal.get('kinect', {}).get('maior_mm'), 0, ' mm')}"
                  f" vs MV {_num(sal.get('mv', {}).get('maior_mm'), 0, ' mm')}.")
        if valores_mv:
            linha += (f" Reprojeccao na auxiliar: dorso "
                      f"{_num(por_lm['mv_aux'][IDX_DORSO], 2, ' px')} vs mediana "
                      f"{_num(_nanmed(valores_mv), 2, ' px')}.")
        linhas.append(linha)
    linhas.append("")
    linhas.append("O landmark 9 (MCP do dedo medio) e' o ponto central do dorso "
                  "usado como referencia da palma; os dedos (4,8,12,16,20) sao os "
                  "nos mais frageis dos dois metodos (contorno fino no depth e "
                  "oclusao entre dedos no RGB).")
    linhas.append("")

    # --- 5. suavizacao ----------------------------------------------------
    linhas.append("## 5. Suavizacao de trajetoria: offline x causal")
    linhas.append("")
    linhas.append("### 5.1 Por que a suavizacao offline do E2T nao e' "
                  "diretamente possivel no online")
    linhas.append("")
    linhas.append("O pos-processamento offline aplica um filtro **centrado / de "
                  "fase zero** (media movel centrada ou Savitzky-Golay com "
                  "`mode='interp'`/`filtfilt`) na trajetoria inteira do trial. "
                  "Para produzir a amostra `t`, esse filtro usa amostras de "
                  "`t + (N-1)/2` -- ou seja, **precisa do futuro**. Num sistema "
                  "online isso so' existiria esperando esses frames: a "
                  "suavizacao 'de graca' do offline e' exatamente o atraso que o "
                  "online nao pode pagar. E' a mesma assimetria que o projeto ja' "
                  "registrou no ramo longo do E2T/SAND: o contexto longo so' esta' "
                  "disponivel depois do trial (`docs/"
                  "ARQUITETURA_MULTIESCALA_E2T_SAND.md`, secao sobre causalidade).")
    linhas.append("")
    linhas.append("Numeros desta sessao (dorso, landmark 9, maior trecho continuo "
                  "da trajetoria -- buracos de ate 5 frames preenchidos por "
                  "interpolacao; o jitter AQUI e' a RMS da 2a diferenca no MESMO "
                  "trecho, para comparar FILTROS entre si, e nao o jitter de pose "
                  "estatica da secao 3.4):")
    linhas.append("")
    cab = ["metodo", "causal?", "jitter (mm)", "reducao do jitter",
           "atraso medido (frames)", "atraso DC teorico (frames)",
           "atraso medido (ms)",
           "erro de forma apos compensar atraso (mm)"]
    corpo = []
    for lado in sessao.lados:
        dados = suavizacao["por_lado"].get(lado)
        if not dados or not dados.get("metodos"):
            continue
        for nome in ("cru", "mm_causal", "mm_offline", "sg_causal",
                     "sg_fixed_lag", "sg_offline", "sg_pred", "ema_causal"):
            m = dados["metodos"].get(nome)
            if not m:
                continue
            corpo.append([f"{lado}: {nome}", "sim" if m["causal"] else "nao",
                          _num(m["jitter_mm"], 2),
                          _pct(m["reducao_pct"] / 100.0),
                          str(m["atraso_frames"]),
                          _num(m.get("atraso_dc_frames"), 2),
                          _num(m["atraso_ms"], 0),
                          _num(m["erro_forma_mm"], 2)])
    linhas.append(_tabela(cab, corpo) if corpo else
                  "_sem trecho continuo valido o bastante para filtrar_")
    linhas.append("")
    linhas.append("O *atraso medido* vem da correlacao cruzada da posicao "
                  "detrendada (resolucao de 1 frame); o *DC teorico* e' o atraso "
                  "de grupo do filtro em movimento lento, calculado dos "
                  "coeficientes -- os dois devem concordar dentro de ~1 frame, e "
                  "a divergencia e' informacao sobre o conteudo de frequencia do "
                  "trecho, nao erro de um dos dois.")
    linhas.append("")
    linhas.append("### 5.2 Qual e' o analogo causal")
    linhas.append("")
    linhas.append("Tres respostas possiveis, todas medidas acima:")
    linhas.append("")
    linhas.append("- **mesmo jitter, pagando o atraso (fixed-lag)**: `sg_fixed_lag` "
                  "e' o SavGol offline ATRASADO de (N-1)/2 frames. Ele entrega "
                  "exatamente o jitter do offline porque *e'* o filtro offline "
                  "com a informacao do futuro trocada por espera.")
    linhas.append("- **atraso zero, mais jitter (endpoint)**: `sg_causal` avalia o "
                  "polinomio no ultimo sample. Ele reproduz exatamente um "
                  "movimento linear (atraso DC nulo), mas sobra mais ruido.")
    linhas.append("- **barato e previsivel (EMA)**: um polo, atraso ~(1-alpha)/alpha "
                  "frames, facil de justificar em tempo real.")
    linhas.append("")
    for lado in sessao.lados:
        dados = suavizacao["por_lado"].get(lado)
        if not dados or not dados.get("metodos"):
            continue
        sg_c = dados["metodos"].get("sg_causal", {})
        sg_fl = dados["metodos"].get("sg_fixed_lag", {})
        sg_o = dados["metodos"].get("sg_offline", {})
        mm_c = dados["metodos"].get("mm_causal", {})
        ema_c = dados["metodos"].get("ema_causal", {})
        sg_p = dados["metodos"].get("sg_pred", {})
        janela = dados.get("janela_sg", JANELA_SG)
        atraso_teorico = (janela - 1) / 2.0 / max(fps, 1e-9) * 1000.0
        linhas.append(
            f"- `{lado}` (**SavGol offline**): reduz o jitter em "
            f"{_pct(sg_o.get('reducao_pct', float('nan'))/100.0)} com atraso "
            f"{_num(sg_o.get('atraso_ms'), 0, ' ms')} -- mas ele usou "
            f"{(janela-1)/2:.0f} frames de FUTURO ({(janela-1)/2/fps*1000:.0f} ms"
            f" em {_num(fps, 1)} fps), que num sistema online nao existem.")
        linhas.append(
            f"  - **`sg_fixed_lag`** (o mesmo filtro, atrasado): jitter "
            f"{_num(sg_fl.get('jitter_mm'), 2, ' mm')} "
            f"({_pct(sg_fl.get('reducao_pct', float('nan'))/100.0)} de reducao) "
            f"com atraso {_num(sg_fl.get('atraso_ms'), 0, ' ms')} "
            f"(DC teorico {_num(sg_fl.get('atraso_dc_frames'), 1, ' frames')} = "
            f"{_num(atraso_teorico, 0, ' ms')}). Essa e' a forma honesta de ter "
            "o resultado do offline em tempo real.")
        linhas.append(
            f"  - **`sg_causal`** (avaliado no ultimo sample): jitter "
            f"{_num(sg_c.get('jitter_mm'), 2, ' mm')} "
            f"({_pct(sg_c.get('reducao_pct', float('nan'))/100.0)}), atraso "
            f"{_num(sg_c.get('atraso_ms'), 0, ' ms')} -- atraso DC nulo porque "
            "um polinomio de ordem 2 reproduz uma rampa exatamente; o preco e' "
            "menos reducao de jitter.")
        linhas.append(
            f"  - **EMA causal** (alpha {ALPHA_EMA:g}): jitter "
            f"{_num(ema_c.get('jitter_mm'), 2, ' mm')}, atraso "
            f"{_num(ema_c.get('atraso_ms'), 0, ' ms')}; e' o filtro online mais "
            "barato e o mais facil de justificar em tempo real, ao custo de "
            "atraso proporcional a 1/alpha.")
        linhas.append(
            f"  - **SavGol preditivo** (avalia o polinomio um passo a frente): "
            f"atraso {_num(sg_p.get('atraso_ms'), 0, ' ms')} e erro de forma "
            f"{_num(sg_p.get('erro_forma_mm'), 2, ' mm')} -- tende a liderar "
            "levemente, mas extrapola nas inversoes de direcao.")
        linhas.append(
            f"  - **Media movel causal** (janela {dados.get('janela_mm', JANELA_MM)}): "
            f"jitter {_num(mm_c.get('jitter_mm'), 2, ' mm')}, atraso "
            f"{_num(mm_c.get('atraso_ms'), 0, ' ms')}.")
    linhas.append("")
    linhas.append("Regra pratica que sai destes numeros: no online, **escolha "
                  "explicitamente entre atraso e suavidade**. Se o orcamento de "
                  "latencia permite (N-1)/2 frames (~100 ms com janela 7 a 30 fps), "
                  "`sg_fixed_lag` reproduz o resultado do pos-processamento "
                  "offline; se nao permite, `sg_causal`/EMA suavizam menos, "
                  "porem sem atraso. A suavizacao offline continua valida como "
                  "**professor/rotulo** de treino e como teto de comparacao -- o "
                  "que ela nao pode ser e' um passo do pipeline online.")
    linhas.append("")

    # --- 6. criterios -----------------------------------------------------
    linhas.append("## 6. Criterios de decisao")
    linhas.append("")
    linhas.append("Limiares fixados ANTES de olhar o resultado "
                  "(`tools/metricas_tracking_mao.py::CRITERIOS`):")
    linhas.append("")
    linhas.append(f"- cobertura >= {CRITERIOS['cobertura_min']*100:.0f}% dos frames "
                  "com 3D valido;")
    linhas.append(f"- jitter em pose estatica <= {CRITERIOS['jitter_max_mm']:g} mm "
                  "(desvio-padrao 3D);")
    linhas.append(f"- saltos > limiar em <= {CRITERIOS['salto_max_frac']*100:.1f}% "
                  "das transicoes consecutivas;")
    linhas.append(f"- reprojecao <= {CRITERIOS['reproj_max_px']:g} px (multi-view; "
                  "o mesmo gate do groundtruth) -- NAO se aplica ao Kinect depth, "
                  "cuja reprojecao e' ida-e-volta.")
    linhas.append("")
    tabela = _veredito_criterios(cobertura, jitter, saltos, reproj)
    cab = ["lado", "metodo", "criterio", "valor medido", "resultado"]
    corpo = []
    for linha in tabela:
        valor = linha["valor"]
        if linha["criterio"] in ("cobertura", "continuidade"):
            texto = _pct(valor)
        elif linha["criterio"] == "jitter":
            texto = _num(valor, 2, " mm")
        else:
            texto = _num(valor, 2, " px")
        corpo.append([linha["lado"], linha["metodo"], linha["criterio"], texto,
                      str(linha["resultado"])])
    linhas.append(_tabela(cab, corpo) if corpo else "_sem dados para julgar_")
    linhas.append("")

    # --- 7. decisao -------------------------------------------------------
    linhas.append("## 7. Decisao preliminar")
    linhas.append("")
    resumo: dict = {}
    for linha in tabela:
        chave = (linha["lado"], linha["metodo"])
        resumo.setdefault(chave, {"PASSA": 0, "NAO PASSA": 0, "NA": 0})
        resumo[chave][linha["resultado"]] = resumo[chave].get(linha["resultado"], 0) + 1
    for (lado, metodo), contagem in resumo.items():
        positivos = contagem.get("PASSA", 0)
        negativos = contagem.get("NAO PASSA", 0)
        aplicaveis = positivos + negativos
        linhas.append(f"- `{lado}` / **{metodo}**: {positivos} de {aplicaveis} "
                      f"criterios aplicaveis atendidos"
                      f"{', ' + str(contagem.get('NA') or 0) + ' NA' if contagem.get('NA') else ''}.")
    linhas.append("")
    # Comparacao direta de cobertura do dorso, que e' o alvo do pedido.
    for lado, d in cobertura["por_lado"].items():
        cob_k = d.get("cobertura_kinect_dorso", float("nan"))
        cob_mv = d.get("cobertura_mv_dorso", float("nan"))
        if not (math.isfinite(cob_k) and math.isfinite(cob_mv)):
            continue
        vencedor = "Kinect depth" if cob_k >= cob_mv else "multi-view"
        linhas.append(f"Para o **dorso (landmark 9)** do lado `{lado}`, a maior "
                      f"cobertura foi de **{vencedor}** "
                      f"({_pct(max(cob_k, cob_mv))} vs {_pct(min(cob_k, cob_mv))}); "
                      f"nos 21 landmarks a media foi "
                      f"{_pct(d.get('cobertura_kinect'))} (Kinect) x "
                      f"{_pct(d.get('cobertura_mv'))} (multi-view).")
    # Continuidade: e' onde o caminho de DEPTH perde -- vale dizer por extenso.
    for lado, d in saltos["por_lado"].items():
        k = d.get("kinect", {})
        m = d.get("mv", {})
        frac_k = k.get("fracao")
        frac_m = m.get("fracao")
        if frac_k is None or not math.isfinite(float(frac_k)):
            continue
        linhas.append(
            f"Continuidade do lado `{lado}`: o Kinect depth salta "
            f"{_pct(frac_k)} das transicoes (maior salto "
            f"{_num(k.get('maior_mm'), 0, ' mm')}) contra "
            f"{_pct(frac_m)} do multi-view (maior salto "
            f"{_num(m.get('maior_mm'), 0, ' mm')}). O salto do depth e' o modo de "
            "falha classico do sensor no contorno da mao/na mesa: por um frame o "
            "ponto cai no FUNDO e volta. Nao e' erro de suavizacao: e' outlier de "
            "medida, e um filtro causal so' o amortece depois de alguns frames. "
            "(As duas fracoes NAO tem o mesmo denominador: o multi-view so' entra "
            "nos frames em que triangulou.)")
    linhas.append("")
    linhas.append("**Isto e' preliminar.** Uma sessao curta, um sujeito, sem "
                  "verdade externa (jig A4/nivel) e sem repeticao entre dias nao "
                  "autoriza arquivar um dos metodos. O veredito de cobertura e "
                  "reprojecao e' o mais estavel (nao depende de a mao ter se "
                  "movido); o de jitter so' vale se houve segmento estatico de "
                  "verdade. Para decidir de fato: 3 sessoes em 3 dias, com o jig "
                  "como referencia de posicao e o tabuleiro como referencia de "
                  "calibracao.")
    linhas.append("")
    # Trava de interpretacao: sem consistencia geometrica, cobertura do
    # multi-view nao mede o METODO, mede a calibracao.
    kd_med = [d.get("kd_aux_mediana") for d in reproj["por_lado"].values()]
    kd_med = [float(v) for v in kd_med if v is not None and math.isfinite(float(v))]
    if kd_med and max(kd_med) > 20.0:
        linhas.append("**Antes de comparar os metodos:** o erro `depth 3D na "
                      f"AUXILIAR` chegou a {_num(max(kd_med), 0, ' px')} nesta "
                      "sessao. Isso indica que a geometria entre Kinect e auxiliar "
                      "nao esta' consistente com a montagem de HOJE. Nesse estado, "
                      "o multi-view mede a calibracao (e o enviesamento de "
                      "correspondencia), nao a qualidade do metodo -- qualquer "
                      "comparacao de posicao/jitter entre Kinect e multi-view fica "
                      "invalida dos dois lados. A decisao defensavel HOJE e': "
                      "**usar o Kinect depth como fonte primaria** (como ja' "
                      "registrado em `docs/BIFURCACAO_VISAO.md`) e tratar o "
                      "multi-view como experimental ate' recalibrar e ver este "
                      "numero na casa de 10-20 px.")
        linhas.append("")

    # --- 8. limitacoes ----------------------------------------------------
    linhas.append("## 8. Limitacoes")
    linhas.append("")
    limitacoes = [
        "Sincronismo: os frames da auxiliar e do Kinect sao lidos em sequencia "
        "no mesmo laco, sem trigger de hardware. A diferenca de instante entre as "
        "duas vistas (dezenas de ms) aparece como erro de reprojecao e como jitter "
        "extra na triangulacao quando a mao se move.",
        "A cabeca de depth do Kinect e o RGB tem baselines/instantes diferentes: "
        "o ponto 3D do landmark depende do mapeamento cor->depth (mediana de "
        "patch 11x11) e do resgate por prior de profundidade; nos dedos o depth "
        "e' ruidoso e cai para o fundo.",
        "O `resgate` de profundidade faz o ponto EXISTIR sem medir Z: aumenta a "
        "cobertura do Kinect e pode mascarar erro de posicao. Ele esta' contado "
        "separadamente justamente por isso.",
        "A reprojecao do multi-view nao e' uma verdade independente: a mesma "
        "calibracao stereo que triangula tambem projeta. O tabuleiro (tecla T de "
        "`tools/kinect_groundtruth_tool.py`) e' a verificacao independente.",
        "O erro de reprojecao do Kinect e' ida-e-volta e por construcao pequeno: "
        "ele NAO diz que a posicao 3D esta' certa, so' que o mapeamento e' "
        "consistente.",
        "A calibracao stereo de cada auxiliar vale para o par (Kinect, aquela "
        "camera) na montagem em que foi feita; qualquer movimento de camera "
        "invalida os numeros sem aviso.",
        "Segmentos estaticos: a deteccao automatica por velocidade pode marcar "
        "como 'parado' um trecho de movimento muito lento (ou perder uma pausa "
        "curta). Quando o operador marca a fase na janela, o rotulo dele manda.",
        "Latencia: a API do MediaPipe em VIDEO mode nao devolve o tempo interno; "
        "mede-se o custo de parede por chamada, que e' um piso.",
        "Sem verdade externa de posicao (jig A4/nivel) nao ha' erro absoluto em "
        "mm -- so' comparacao relativa entre os dois metodos e auto-consistencia.",
        "A comparacao de POSICAO entre os metodos so' vale se a secao 3.3 "
        "(depth 3D na AUXILIAR) estiver na casa de dezenas de px: com centenas "
        "de px a geometria das duas vistas nao descreve a mesma cena e o "
        "multi-view fica deslocado por mais que a triangulacao seja 'limpa'.",
        "Nao foi gravado video nem frame algum (privacidade): o CSV tem apenas "
        "landmarks, posicoes 3D e metricas.",
    ]
    if sessao.sintetica:
        limitacoes.insert(0, "OS DADOS DESTE RELATORIO SAO SINTETICOS: eles "
                             "validam o codigo (CSV, metricas, relatorio, PNGs), "
                             "nao medem o hardware. Nenhum numero aqui descreve a "
                             "bancada.")
    for item in limitacoes:
        linhas.append(f"- {item}")
    linhas.append("")

    # --- 9. reproducao ----------------------------------------------------
    linhas.append("## 9. Como reproduzir")
    linhas.append("")
    linhas.append("```powershell")
    linhas.append(meta_comando(sessao))
    linhas.append("```")
    linhas.append("")
    linhas.append("Se o comando acima aparecer como `python.exe`, rode-o com "
                  "`.venv\\Scripts\\python.exe` (o Python do sistema nao tem "
                  "cv2/mediapipe).")
    linhas.append("")
    linhas.append("Teclas na janela: `ESC`/`Q` sai; `E` marca fase estatica; "
                  "`M` marca movimento; `R` marca repouso; `D` liga/desliga o "
                  "desenho dos pontos de depth. O CSV e o relatorio sao gravados "
                  "ao final da sessao.")
    linhas.append("")

    # --- 10. anexo: colunas -----------------------------------------------
    linhas.append("## 10. Anexo: colunas dos CSV")
    linhas.append("")
    linhas.append("O cabecalho e' o contrato com este relatorio; a documentacao "
                  "de cada coluna (unidade e significado) vive nos comentarios de "
                  "`CABECALHO_FRAMES` e `CABECALHO_LANDMARKS` em "
                  "`tools/compara_tracking_mao.py`. Resumo das unidades: metros "
                  "(posicoes), mm (distancias), px (reprojecao), ms (tempos), "
                  "1/0 (booleanos) e vazio = nao medido.")
    linhas.append("")
    if sessao.frames:
        linhas.append("`" + Path(arquivos.get("frames", "frames")).name + "`:")
        linhas.append("")
        linhas.append("```text")
        linhas.append(",".join(sessao.frames[0].keys()))
        linhas.append("```")
        linhas.append("")
    algum = next(iter(sessao.rastros.values()), None)
    if algum is not None:
        linhas.append("`" + Path(arquivos.get("landmarks", "landmarks")).name + "`:")
        linhas.append("")
        linhas.append("```text")
        linhas.append(
            "seq,t_unix,t_mono,origem,hand,landmark_id,is_dorso,k_valid,k_conf,"
            "k_map_fail,k_depth_fail,k_depth_rescue,k_depth_m,k_x_m,k_y_m,k_z_m,"
            "k_reproj_px,mv_valid,mv_implausivel,mv_conf,mv_x_m,mv_y_m,mv_z_m,"
            "mv_reproj_kin_px,mv_reproj_aux_px,kin_px_x,kin_px_y,aux_px_x,"
            "aux_px_y,kd_aux_err_px")
        linhas.append("```")
        linhas.append("")

    # --- arquivos ---------------------------------------------------------
    linhas.append("## 11. Arquivos gerados")
    linhas.append("")
    for nome, caminho in arquivos.items():
        linhas.append(f"- **{nome}**: `{caminho}`")
    linhas.append("")

    texto = "\n".join(linhas) + "\n"
    nome = ("relatorio_tracking.md" if not sessao.sintetica
            else f"{prefixo}_relatorio_tracking.md")
    destino = dir_saida / nome
    destino.write_text(texto, encoding="utf-8")
    return destino


def meta_comando(sessao: Sessao) -> str:
    meta = sessao.meta or {}
    comando = meta.get("comando")
    if comando:
        return str(comando)
    aux = meta.get("aux_index", 0)
    dur = meta.get("duracao_alvo_s", 15)
    return (r".venv\Scripts\python.exe tools\compara_tracking_mao.py "
            f"--aux-index {aux} --duracao {dur}")


def analisa(caminho_frames, dir_saida=None, prefixo: str = "tracking",
            gerar_png: bool = True) -> dict:
    """Pipeline completo: CSV -> metricas -> relatorio (+PNGs). Devolve resumo.

    Devolvido para o programa de aquisicao poder imprimir os numeros-chave na
    mesma sessao em que gravou (sem obrigar a rodar de novo a mao no CSV).
    """
    sessao = carrega_sessao(caminho_frames)
    dir_saida = Path(dir_saida) if dir_saida else Path(caminho_frames).parent
    fps = _fps_da_sessao(sessao.frames)
    cobertura = metricas_cobertura(sessao)
    lateralidade = metricas_lateralidade(sessao)
    depth = metricas_depth(sessao)
    reproj = metricas_reprojecao(sessao)
    segmentos = {lado: segmentos_estaticos(sessao, lado, fps)
                 for lado in sessao.lados}
    jitter = metricas_jitter(sessao, fps, segmentos)
    saltos = metricas_saltos(sessao, float(sessao.meta.get("salto_mm", 50.0)))
    latencia = metricas_latencia(sessao)
    suavizacao = {"por_lado": {lado: compara_suavizacao(sessao, fps, lado)
                               for lado in sessao.lados}}
    nome_base = Path(caminho_frames).name
    sufixo = "_frames.csv"
    base = nome_base[: -len(sufixo)] if nome_base.endswith(sufixo) else Path(
        caminho_frames).stem
    arquivos = {
        "frames": str(Path(caminho_frames).resolve()),
        "landmarks": str(Path(caminho_frames).with_name(base + "_landmarks.csv").resolve()),
        "meta": str(Path(caminho_frames).with_name(base + "_meta.json").resolve()),
    }
    if gerar_png:
        for png in gera_pngs(sessao, cobertura, jitter, reproj, suavizacao,
                             dir_saida, prefixo):
            arquivos[Path(png).stem] = str(Path(png).resolve())
    relatorio = gera_relatorio(sessao, dir_saida, arquivos, prefixo)
    arquivos["relatorio"] = str(relatorio.resolve())
    return {
        "sessao": sessao,
        "fps": fps,
        "cobertura": cobertura,
        "lateralidade": lateralidade,
        "depth": depth,
        "reprojecao": reproj,
        "jitter": jitter,
        "saltos": saltos,
        "latencia": latencia,
        "suavizacao": suavizacao,
        "relatorio": relatorio,
        "arquivos": arquivos,
    }


def resumo_console(resultado: dict) -> str:
    """Resumo curto das metricas para imprimir no terminal."""
    sessao: Sessao = resultado["sessao"]
    linhas = []
    fps = resultado["fps"]
    linhas.append(f"[metricas] fps={_num(fps, 1)} | "
                  f"lados={','.join(sessao.lados) or 'nenhum'}")
    lat = resultado.get("lateralidade") or {}
    if lat.get("comparaveis"):
        linhas.append(
            f"[metricas] lateralidade: {_pct(lat.get('fracao_igual'))} de "
            f"concordancia em {lat['comparaveis']} frames "
            f"({lat['diferentes']} trocados)")
    if lat.get("pareamento"):
        linhas.append("[metricas] pareamento: " + ", ".join(
            f"{modo}={n}" for modo, n in sorted(lat["pareamento"].items())))
    for lado, d in resultado["cobertura"]["por_lado"].items():
        linhas.append(
            f"[metricas] {lado}: cobertura Kinect {_pct(d['cobertura_kinect'])} "
            f"(dorso {_pct(d['cobertura_kinect_dorso'])}) | multi-view "
            f"{_pct(d['cobertura_mv'])} (dorso {_pct(d['cobertura_mv_dorso'])})")
        rep = resultado["reprojecao"]["por_lado"].get(lado, {})
        linhas.append(
            f"[metricas] {lado}: reproj MV no Kinect "
            f"{_num(rep.get('mv_kin_mediana'), 2, ' px')} | na auxiliar "
            f"{_num(rep.get('mv_aux_mediana'), 2, ' px')} | depth 3D na auxiliar "
            f"{_num(rep.get('kd_aux_mediana'), 2, ' px')}")
        jit = resultado["jitter"]["por_lado"].get(lado, {})
        linhas.append(
            f"[metricas] {lado}: jitter Kinect {_num(jit.get('jitter_kinect'), 2, ' mm')}"
            f" | MV {_num(jit.get('jitter_mv'), 2, ' mm')}"
            f" | dorso {_num(jit.get('dorso_kinect'), 2)} / "
            f"{_num(jit.get('dorso_mv'), 2)} mm")
    linhas.append(f"[metricas] relatorio: {resultado['relatorio']}")
    return "\n".join(linhas)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Metricas e relatorio do rastreamento da mao (Kinect x multi-view)")
    parser.add_argument("--csv", required=True,
                        help="CSV de frames gravado por tools/compara_tracking_mao.py")
    parser.add_argument("--saida", default=None,
                        help="Diretorio de saida (padrao: o do CSV)")
    parser.add_argument("--prefixo", default="tracking",
                        help="Prefixo dos arquivos gerados (padrao: tracking)")
    parser.add_argument("--sem-png", action="store_true",
                        help="Nao gera os PNGs (so' o relatorio)")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    resultado = analisa(args.csv, args.saida, args.prefixo,
                        gerar_png=not args.sem_png)
    print(resumo_console(resultado))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
