#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Treino L/R com feedback visual ADAPTATIVO + sessao final com decoder CONGELADO.

Objetivo (task-6): tratar BCI illiteracy com TREINO (nao so trocar classificador),
em desenho gradual ME->MI inspirado no E2T: o participante executa a ME e, no
mesmo trial, tenta a MI vendo o feedback continuo do decoder L/R.

O que este modulo faz:
  1. Calibracao inicial CSP+LDA (sklearn/mne; mesmo pipeline de
     `tools/csp_lda_duas_sessoes.py`).
  2. Score continuo L/R: janelas de 2 s deslizando dentro da fase MI
     (PASSO 0,25 s); score = 2*P(direita) - 1 em [-1, +1].
  3. Feedback visual: barra/cursor + mao virtual (OpenCV opcional; no modo
     `--smoke` o feedback e' simulado e pode ser renderizado em PNG).
  4. Adaptacao POR BLOCO com os labels do cue (aprendizado supervisionado
     offline do bloco anterior + recalibracao do limiar para equilibrar L/R).
  5. Dificuldade adaptativa: tolerancia (|score| minimo para "decidir") e ganho
     do cursor mudam com o desempenho do bloco.
  6. Sessao final `--modo teste-congelado`: decoder CONGELADO, sem adaptacao,
     metricas honestas (BA + IC bootstrap 95%, ERD mu/beta, nao-respondedor).
  7. Modo `--smoke`: sessoes sinteticas (3 sessoes = 60 trials/classe no total),
     sem hardware, para validar o pipeline e gerar relatorio.

Protocolo (fixado em `docs/guias/PROTOCOLO_TREINO_LR_ADAPTATIVO.md`):
  - 20 trials/classe por SESSAO, em 2 blocos de 10/classe (pausa entre blocos);
  - 3 sessoes -> 60 trials/classe (sessoes 1-2 treino adaptativo, sessao 3
    teste congelado) - nunca 60 em sequencia;
  - trial ME->MI: cue 1 s -> ME 4 s -> transicao 1 s -> MI 4 s -> ITI 1-2 s;
  - janela de decoder SEMPRE dentro da MI (nunca atravessa ME->MI).

Uso:
    # smoke sintetico (sem hardware)
    .venv\\Scripts\\python.exe src\\mestradopy\\tools\\feedback_adaptativo.py --smoke

    # plano do protocolo (3 sessoes) e validacao das restricoes
    .venv\\Scripts\\python.exe src\\mestradopy\\tools\\feedback_adaptativo.py --plano

    # sessao real de treino adaptativo (CSV+JSON de validacao_eeg_3classes.py)
    .venv\\Scripts\\python.exe src\\mestradopy\\tools\\feedback_adaptativo.py ^
        --sessao-dir outputs\\eeg\\piloto_me_mi_s01 --modo treino ^
        --salvar-modelo outputs\\visao\\lr_adaptativo\\modelo_s01.joblib

    # sessao final congelada
    .venv\\Scripts\\python.exe src\\mestradopy\\tools\\feedback_adaptativo.py ^
        --sessao-dir outputs\\eeg\\piloto_me_mi_s02 --modo teste-congelado ^
        --modelo outputs\\visao\\lr_adaptativo\\modelo_s01.joblib

Saidas (default: pasta temporaria; use --saida): relatorio.md, resultados.json,
modelo (se --salvar-modelo), questionario_estrategia.json, frames do feedback.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

FS = 128.0                    # taxa de trabalho (mesma do csp_lda_duas_sessoes.py)
BANDA = (8.0, 30.0)           # mu+beta
JANELA_S = 2.0                # janela de decisao do decoder
PASSO_S = 0.25                # passo da janela (score continuo)
TOL_INICIAL = 0.35            # |score| minimo para "decidir" (dificuldade easy)
GANHO_INICIAL = 1.0           # px de cursor por unidade de score
CLASSES = ("esquerda", "direita")

#: Codigos de marcador do projeto (ver docstring de eeg_motor_paradigm.py e
#: validacao_eeg_3classes.py).
MARK = {"trial_start": 768, "cue_esq": 769, "cue_dir": 770,
        "me_ini": 778, "me_fim": 779, "mi_ini": 780, "mi_fim": 781,
        "bloco_ini": 790, "bloco_fim": 791}

#: Plano do protocolo (task-6). Valores fixados ANTES da coleta.
PROTOCOLO = {
    "sessoes": 3,
    "trials_por_classe_por_sessao": 20,
    "blocos_por_sessao": 2,
    "trials_por_classe_por_bloco": 10,
    "pausa_entre_blocos_s": 120,
    "cue_s": 1.0,
    "me_s": 4.0,
    "transicao_s": 1.0,
    "mi_s": 4.0,
    "iti_s": (1.0, 2.0),
    "sessoes_treino_adaptativo": (1, 2),
    "sessao_teste_congelado": 3,
    "total_trials_por_classe": 60,
    "limiar_respondedor_ba": 0.55,
    "limiar_sucesso_ba": 0.65,
}


# ==========================================================================
# 1. Dados: sintetico (smoke) e sessoes reais
# ==========================================================================
def _passa_faixa(X, fs=FS, banda=BANDA):
    """Filtro passa-banda zero-fase (offline; aqui NAO e' tempo real)."""
    from scipy.signal import butter, sosfiltfilt
    lo, hi = banda
    sos = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="bandpass", output="sos")
    return sosfiltfilt(sos, np.asarray(X, np.float64), axis=-1)


def _resample(X, fs_in, fs_out=FS):
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(X, np.float64)
    from scipy.signal import resample_poly
    up = int(round(fs_out))
    down = int(round(fs_in))
    return resample_poly(np.asarray(X, np.float64), up, down, axis=-1)


def _janelas_dentro(segmento, janela_n, passo_n):
    """Janelas (n_ch, janela_n) estritamente dentro do segmento (MI).

    Invariante do protocolo: a janela nunca atravessa a fronteira ME->MI, porque
    `segmento` ja' e' a fase MI recortada pelos marcadores 780/781.
    """
    seg = np.asarray(segmento, np.float64)
    n = seg.shape[-1]
    if n < janela_n:
        return np.empty((0, seg.shape[-2], janela_n), np.float64)
    inicios = list(range(0, n - janela_n + 1, passo_n))
    if not inicios:
        return np.empty((0, seg.shape[-2], janela_n), np.float64)
    return np.stack([seg[:, i:i + janela_n] for i in inicios]).astype(np.float64)


def gerar_sessao_sintetica(sessao, trials_por_classe=20, blocos=2, seed=7,
                           erd_forte=True):
    """Sessao sintetica no formato do piloto (3 canais C3/Cz/C4, 4 s MI).

    O sinal tem ERD contralateral real (direita -> C3; esquerda -> C4) na banda
    mu/beta, ruido, deriva lenta e jitter de latencia. Serve para validar TODO o
    pipeline (CSP+LDA, score, adaptacao, congelamento, metricas) sem hardware;
    NUNCA e' resultado de bancada (marcado como `origem=sintetico`).
    """
    rng = np.random.default_rng(seed)
    n_por_bloco = int(trials_por_classe / blocos) if blocos else trials_por_classe
    trials = []
    for bloco in range(blocos):
        for classe in (0, 1):
            for _ in range(n_por_bloco):
                trials.append(_trial_sintetico(rng, classe, sessao, bloco,
                                               erd_forte=erd_forte))
    rng.shuffle(trials)
    for i, t in enumerate(trials):
        t["ordem"] = i
    return trials


def _trial_sintetico(rng, classe, sessao, bloco, erd_forte=True):
    """Um trial: dict com X_base/X_me/X_mi (3 canais) e metadados."""
    n_base = int(1.0 * FS)
    n_fase = int(4.0 * FS)
    t = np.arange(n_fase * 3 + n_base) / FS
    # ruido 1/f + branco
    branco = rng.normal(0, 1.0, (3, t.size))
    ruido = branco
    for i in range(1, 4):        # deriva lenta
        ruido += 0.6 * np.sin(2 * np.pi * (0.05 * i) * t +
                              rng.uniform(0, 2 * np.pi))
    # potencia mu/beta por canal; ERD contralateral
    base_mu = np.array([1.0, 0.9, 1.0])            # C3, Cz, C4
    erd = np.ones((3, 1))
    if erd_forte:
        if classe == 1:                            # direita -> C3 desincroniza
            erd[0, 0] = 0.68
            erd[2, 0] = 1.0
        else:                                      # esquerda -> C4 desincroniza
            erd[2, 0] = 0.68
            erd[0, 0] = 1.0
    else:                                          # nao-respondedor (illiteracy)
        erd[:] = 1.0
    # mu (10 Hz) + beta (20 Hz) modulados na fase MI (ultimos 4 s)
    sinal = np.zeros((3, t.size))
    fase_mi = (t >= 2.0).astype(np.float64)        # base 1s + ME 1s (sintetico)
    for canal in range(3):
        mod = 1.0 + (erd[canal, 0] - 1.0) * fase_mi
        sinal[canal] += mod * base_mu[canal] * np.sin(
            2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi))
        sinal[canal] += 0.5 * mod * np.sin(
            2 * np.pi * 20.0 * t + rng.uniform(0, 2 * np.pi))
    sinal += 1.8 * ruido
    sinal = sinal * 1e-5                         # ~uV
    # posiciona as janelas: base 1 s (primeiro), ME 4 s (meio), MI 4 s (fim)
    i_base = slice(0, n_base)
    i_me = slice(n_base, n_base + n_fase)
    i_mi = slice(n_base + n_fase, n_base + 2 * n_fase)
    return {
        "sessao": int(sessao), "bloco": int(bloco), "classe": int(classe),
        "X_base": sinal[:, i_base], "X_me": sinal[:, i_me],
        "X_mi": sinal[:, i_mi], "origem": "sintetico",
        "erd_embutido": bool(erd_forte),
    }


def carrega_sessao_eeg(pasta, fs_alvo=FS):
    """Le uma sessao real do piloto (CSV+JSON de `validacao_eeg_3classes.py`).

    Formato: CSV com Time, 32 canais EEG (uV), Marker; JSON com fs, nomes de
    canais e `block_orders`. Recorta por marcador:
      768 inicio do trial | 769/770 cue L/R | 778/779 ME | 780/781 MI
      790/791 bloco.  As janelas do decoder sao as do segmento MI (nunca cruzam
      ME->MI); o baseline (ERD) e' 1 s imediatamente antes do cue.
    """
    pasta = Path(pasta)
    csvs = sorted(pasta.glob("eeg_mi3c*.csv")) or sorted(pasta.glob("*.csv"))
    jsons = sorted(pasta.glob("eventos*.json"))
    if not csvs:
        raise SystemExit(f"nenhum CSV de EEG em {pasta}")
    caminho_csv = csvs[-1]
    meta = (json.load(open(jsons[-1], encoding="utf-8")) if jsons else {})
    fs = float(meta.get("fs", 500.0))
    nomes = [str(n) for n in meta.get("channel_names", [])]
    canais = []
    for alvo in ("C3", "Cz", "C4"):
        try:
            canais.append(nomes.index(alvo))
        except ValueError:
            canais.append({"C3": 14, "Cz": 15, "C4": 16}[alvo])
    linhas = []
    with open(caminho_csv, newline="", encoding="utf-8") as h:
        leitor = csv.reader(h)
        cabecalho = None
        for r in leitor:
            if not r:
                continue
            if r[0].startswith("#"):
                cabecalho = r
                continue
            if r[0].strip().lower() in ("time", "tempo"):
                cabecalho = r
                continue
            if cabecalho is None:
                continue
            try:
                linhas.append([float(x) for x in r])
            except ValueError:
                continue
    arr = np.asarray(linhas, np.float64)
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise SystemExit(f"CSV inesperado: shape {arr.shape}")
    n_ch = len(nomes) if nomes else (arr.shape[1] - 2)
    n_ch = min(n_ch, arr.shape[1] - 2)
    eeg = arr[:, 1:1 + n_ch][:, canais]
    markers = arr[:, -1].astype(np.int64)
    eeg = _passa_faixa(eeg.T, fs).T
    eeg = _resample(eeg, fs, fs_alvo)
    escala = fs_alvo / fs
    pos = {cod: np.where(markers == cod)[0] for cod in set(MARK.values())}
    # blocos: 790..791; se ausente, infere 1 bloco
    blocos = []
    if len(pos[MARK["bloco_ini"]]) and len(pos[MARK["bloco_fim"]]):
        for a, b in zip(pos[MARK["bloco_ini"]], pos[MARK["bloco_fim"]]):
            blocos.append((int(a), int(b)))
    else:
        blocos = [(0, len(markers) - 1)]
    ordem_bloco = meta.get("block_orders") or []
    trials = []
    for ib, (a_bloco, b_bloco) in enumerate(blocos):
        dentro = {cod: pos[cod][(pos[cod] >= a_bloco) & (pos[cod] <= b_bloco)]
                  for cod in MARK.values()}
        for k, (mi_a, mi_b) in enumerate(zip(dentro[MARK["mi_ini"]],
                                             dentro[MARK["mi_fim"]])):
            me_ini = dentro[MARK["me_ini"]][dentro[MARK["me_ini"]] < mi_a]
            me_fim = dentro[MARK["me_fim"]][dentro[MARK["me_fim"]] < mi_a]
            cues_esq = dentro[MARK["cue_esq"]][dentro[MARK["cue_esq"]] < me_ini[-1]]\
                if len(me_ini) else np.array([], int)
            cues_dir = dentro[MARK["cue_dir"]][dentro[MARK["cue_dir"]] < me_ini[-1]]\
                if len(me_ini) else np.array([], int)
            if len(cues_esq) == 0 and len(cues_dir) == 0:
                continue
            classe = 0 if (len(cues_esq) and (len(cues_dir) == 0 or
                                              cues_esq[-1] > cues_dir[-1])) else 1
            a = int(round(mi_a * escala))
            b = int(round(mi_b * escala))
            seg_mi = eeg[a:b].T
            if seg_mi.shape[-1] < int(1.0 * fs_alvo):
                continue
            if len(me_ini) and len(me_fim):
                x0 = int(round(me_ini[-1] * escala))
                x1 = int(round(me_fim[-1] * escala))
                seg_me = eeg[x0:x1].T
            else:
                seg_me = np.zeros((3, int(1.0 * fs_alvo)))
            cue_pos = (max(cues_esq[-1] if len(cues_esq) else -1,
                           cues_dir[-1] if len(cues_dir) else -1))
            c = int(round(cue_pos * escala))
            seg_base = eeg[max(0, c - int(1.0 * fs_alvo)):c].T
            if seg_base.shape[-1] < int(0.5 * fs_alvo):
                seg_base = np.zeros((3, int(1.0 * fs_alvo)))
            trials.append({"sessao": 1, "bloco": int(ib), "classe": int(classe),
                           "X_base": seg_base, "X_me": seg_me, "X_mi": seg_mi,
                           "origem": "bancada"})
    if not trials:
        raise SystemExit(f"nenhum trial recortado em {pasta} (marcadores?)")
    return trials, meta


# ==========================================================================
# 2. Decoder CSP+LDA e score continuo
# ==========================================================================
def cria_pipeline(n_componentes=2):
    """CSP+LDA identico ao `csp_lda_duas_sessoes.make_csp()` (mne/sklearn)."""
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    try:
        import mne
        mne.set_log_level("ERROR")
    except (ImportError, AttributeError):
        pass

    def _finite(Z):
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    return Pipeline([
        ("csp", CSP(n_components=n_componentes, reg="ledoit_wolf", log=True)),
        ("finite", FunctionTransformer(_finite)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def janelas_do_trial(trial):
    """Janelas de decisao do trial (SOMENTE dentro da fase MI)."""
    janela_n = int(round(JANELA_S * FS))
    passo_n = int(round(PASSO_S * FS))
    return _janelas_dentro(trial["X_mi"], janela_n, passo_n)


def treina_decoder(trials, n_componentes=2):
    """Treina CSP+LDA nas janelas dos trials (labels do cue)."""
    xs, ys = [], []
    for t in trials:
        w = janelas_do_trial(t)
        if w.shape[0] == 0:
            continue
        xs.append(w)
        ys.extend([t["classe"]] * w.shape[0])
    if not xs:
        raise SystemExit("sem janelas para treinar o decoder")
    X = np.concatenate(xs).astype(np.float64)
    y = np.asarray(ys, np.int64)
    pipeline = cria_pipeline(n_componentes)
    pipeline.fit(X, y)
    modelo = {"pipeline": pipeline, "n_janelas": int(len(y)),
              "origem": trials[0].get("origem", "?")}
    modelo["limiar"] = float(calibra_limiar(modelo, trials))
    return modelo


def pontua_janelas(modelo, janelas):
    """Score continuo L/R por janela: 2*P(direita) - 1 em [-1, +1]."""
    if janelas.shape[0] == 0:
        return np.array([], np.float64), np.array([], np.float64)
    proba = modelo["pipeline"].predict_proba(janelas)
    p_dir = proba[:, 1]
    return 2.0 * p_dir - 1.0, p_dir


def pontua_trial(modelo, trial):
    """Score do trial = media dos scores das janelas dentro da MI (soft vote)."""
    janelas = janelas_do_trial(trial)
    scores, p_dir = pontua_janelas(modelo, janelas)
    if scores.size == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(scores)), float(np.mean(p_dir)), int(scores.size)


def calibra_limiar(modelo, trials):
    """Limiar de decisao que equilibra as classes nos trials de calibracao.

    Reduz o vies sistematico do LDA (a saida probabilistica nao e' calibrada e
    um desequilibrio L/R rouba trials do participante). Retorna o ponto medio
    entre as medianas dos scores de cada classe.
    """
    esq, dir_ = [], []
    for t in trials:
        s, _, n = pontua_trial(modelo, t)
        if n == 0 or not math.isfinite(s):
            continue
        (esq if t["classe"] == 0 else dir_).append(s)
    if not esq or not dir_:
        return 0.0
    return float(np.median([np.median(esq), np.median(dir_)]))


def decide(score, limiar):
    """Classe decidida (+1 direita / -1 esquerda) e se foi 'decisao' (>0)."""
    if not math.isfinite(score):
        return 0, False
    return (1 if score >= limiar else -1), True


# ==========================================================================
# 3. Metricas (BA + IC bootstrap), ERD e nao-respondedor
# ==========================================================================
def bootstrap_ci_ba(y, pred, n_boot=2000, seed=11):
    """IC 95% bootstrap da acuracia balanceada (BA)."""
    y = np.asarray(y, np.int64)
    pred = np.asarray(pred, np.int64)
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0:
        return {"ba": float("nan"), "ic95": [float("nan"), float("nan")]}
    base = _ba(y, pred)
    amostras = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        amostras.append(_ba(y[idx], pred[idx]))
    lo, hi = np.percentile(amostras, [2.5, 97.5])
    return {"ba": float(base), "ic95": [float(lo), float(hi)]}


def _ba(y, pred):
    recalls = []
    for c in (0, 1):
        m = y == c
        if m.sum():
            recalls.append(float(np.mean(pred[m] == c)))
    return float(np.mean(recalls)) if recalls else float("nan")


def metricas_trials(y, pred, scores=None, limiar=0.0):
    out = bootstrap_ci_ba(y, pred)
    out["n"] = int(len(y))
    out["confusao"] = _confusao(y, pred)
    if scores is not None and len(scores):
        s = np.asarray(scores)
        out["score_medio_abs"] = float(np.nanmean(np.abs(s)))
        out["decisoes"] = int(np.nansum(np.abs(s) >= 0))
    return out


def _confusao(y, pred):
    cm = np.zeros((2, 2), int)
    for a, b in zip(y, pred):
        if 0 <= a < 2 and 0 <= b < 2:
            cm[a, b] += 1
    return cm.tolist()


def erd_por_canal(trials, faixas=((8.0, 13.0), (13.0, 30.0))):
    """ERD/ERS (%) da MI relativo ao baseline, por classe e canal.

    ERD% = 100*(P(MI)/P(base) - 1): negativo = dessincronizacao (esperado
    contralateral ao lado imaginado). Mede se o participante ENGajou a MI.
    """
    from scipy.signal import welch
    saida = {}
    for (lo, hi) in faixas:
        nome = f"{lo:.0f}-{hi:.0f}Hz"
        por_classe = {}
        for classe in (0, 1):
            erds = []
            for t in trials:
                if t["classe"] != classe:
                    continue
                # mesmo nperseg para baseline e MI (frequencias comparaveis)
                nperseg = int(min(128, t["X_base"].shape[-1],
                                  t["X_mi"].shape[-1]))
                if nperseg < 16:
                    continue
                f, p_base = welch(t["X_base"], fs=FS, nperseg=nperseg, axis=-1)
                _, p_mi = welch(t["X_mi"], fs=FS, nperseg=nperseg, axis=-1)
                banda = (f >= lo) & (f <= hi)
                if not banda.any():
                    continue
                erd = 100.0 * (p_mi[:, banda].mean(axis=1) /
                               np.maximum(p_base[:, banda].mean(axis=1), 1e-20) - 1.0)
                erds.append(erd)
            if erds:
                por_classe[CLASSES[classe]] = np.median(np.array(erds), axis=0).tolist()
        saida[nome] = por_classe
    return saida


def avalia_nao_respondedor(metricas, limiar_ba=None):
    """Criterio pre-fixado de nao-respondedor (BCI illiteracy)."""
    lim = PROTOCOLO["limiar_respondedor_ba"] if limiar_ba is None else limiar_ba
    ic_lo = metricas.get("ic95", [float("nan"), float("nan")])[0]
    if not math.isfinite(ic_lo):
        return {"nao_respondedor": None, "motivo": "sem metricas"}
    nao = bool(ic_lo < lim)
    return {
        "nao_respondedor": nao,
        "criterio": f"IC95% inferior da BA < {lim:.2f} apos o treino",
        "ba": metricas.get("ba"), "ic95": metricas.get("ic95"),
        "motivo": ("respondeu ao treino" if not nao else
                   "nao respondeu: revisar estrategia/ERD antes de continuar"),
    }


# ==========================================================================
# 4. Adaptacao, dificuldade e feedback
# ==========================================================================
def aplica_dificuldade(tol, ganho, ba_bloco):
    """Regra de dificuldade adaptativa (fixada antes da coleta)."""
    if not math.isfinite(ba_bloco):
        return tol, ganho, "sem_dados"
    if ba_bloco >= 0.80:
        return max(0.12, tol * 0.85), min(2.5, ganho * 1.10), "mais_dificil"
    if ba_bloco <= 0.55:
        return min(0.80, tol * 1.15), max(0.40, ganho * 0.90), "mais_facil"
    return tol, ganho, "mantido"


class FeedbackVisual:
    """Feedback do trial: barra, cursor e mao virtual (OpenCV opcional).

    Sem janela (`janela=False`) so' acumula a trajetoria (usada no render PNG e
    no log). O alvo e' o lado do cue: a barra tem L na esquerda e R na direita;
    a mao virtual anda para o lado que o decoder "puxa", e o participante deve
    empurra-la para o lado do cue.
    """

    def __init__(self, largura=640, altura=360, janela=False):
        self.largura, self.altura = largura, altura
        self.janela = bool(janela)
        self.trajetoria = []
        self.cv2 = None
        if janela:
            try:
                import cv2
                self.cv2 = cv2
            except ImportError:
                self.janela = False

    def atualiza(self, score, tolerancia, ganho, classe):
        alvo = -1.0 if classe == 0 else 1.0
        pos = float(np.clip(score * ganho, -1.0, 1.0))
        decidiu = abs(score) >= tolerancia
        acerto = (pos >= 0) == (alvo > 0) if decidiu else None
        registro = {"score": score, "pos": pos, "decidiu": bool(decidiu),
                    "acerto": acerto, "tolerancia": tolerancia,
                    "ganho": ganho, "classe": int(classe)}
        self.trajetoria.append(registro)
        return registro

    def desenha(self, registro):
        if not self.janela or self.cv2 is None:
            return None
        cv2 = self.cv2
        img = np.full((self.altura, self.largura, 3), 30, np.uint8)
        cx, cy = self.largura // 2, self.altura // 2
        # barra central
        cv2.line(img, (80, cy), (self.largura - 80, cy), (200, 200, 200), 2)
        cv2.putText(img, "L", (60, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 180, 80), 2)
        cv2.putText(img, "R", (self.largura - 60, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 80), 2)
        # tolerancia (zona morta)
        tol_px = int(registro["tolerancia"] * 200)
        cv2.rectangle(img, (cx - tol_px, cy - 14), (cx + tol_px, cy + 14),
                      (80, 80, 80), -1)
        # mao virtual: palma + 5 dedos
        px = int(cx + registro["pos"] * 200)
        cv2.circle(img, (px, cy), 18, (80, 200, 240), -1)
        for k in range(5):
            ang = -0.9 + 0.45 * k
            dx, dy = int(24 * math.cos(ang)), int(-24 * math.sin(ang))
            cv2.line(img, (px, cy - 6), (px + dx, cy - 6 + dy),
                     (80, 200, 240), 3)
        estado = ("ACERTO" if registro["acerto"] else
                  "ERRO" if registro["decidiu"] else "AGUARDE")
        cor = ((80, 255, 80) if estado == "ACERTO" else
               (80, 80, 255) if estado == "ERRO" else (200, 200, 200))
        cv2.putText(img, f"{estado}  score={registro['score']:+.2f}  "
                         f"tol={registro['tolerancia']:.2f}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
        return img

    def salva_png(self, caminho, registro):
        img = self.desenha(registro)
        if img is not None:
            self.cv2.imwrite(str(caminho), img)
            return True
        return False


def roda_treino_adaptativo(sessoes, modelo=None, janela=False, render_dir=None,
                           log=print):
    """Sessoes de treino: calibra no 1o bloco e adapta a cada bloco seguinte.

    Adaptacao (supervisionada, labels do cue conhecidos):
      - re-treina CSP+LDA com TODOS os trials ja' rotulados (inclui o bloco que
        acabou) e recalibra o limiar para equilibrar L/R;
      - a dificuldade (tolerancia/ganho) e' atualizada pela BA do bloco.
    Retorna dict com historico por bloco e o modelo final.
    """
    historico = []
    tol, ganho = TOL_INICIAL, GANHO_INICIAL
    todos = []
    for sessao in sessoes:
        blocos = sorted({t["bloco"] for t in sessao})
        for bloco in blocos:
            trials = [t for t in sessao if t["bloco"] == bloco]
            if modelo is None:
                modelo = treina_decoder(trials)
                papel = "calibracao"
            else:
                papel = "adaptacao"
            y, pred, scores, decisoes = [], [], [], 0
            feedback = FeedbackVisual(janela=janela)
            for t in trials:
                score, p_dir, n_win = pontua_trial(modelo, t)
                scores.append(score)
                classe_pred, _ = decide(score, modelo["limiar"])
                # posicao/feedback da mao virtual (para o participante)
                for k in range(n_win):
                    s_k = (2.0 * p_dir - 1.0) if math.isfinite(p_dir) else 0.0
                    reg = feedback.atualiza(s_k, tol, ganho, t["classe"])
                    decisoes += int(reg["decidiu"])
                    if render_dir is not None and k < 3:
                        feedback.salva_png(
                            Path(render_dir) / f"s{t['sessao']}_b{bloco}_t{t['ordem'] if 'ordem' in t else 0}_{k}.png",
                            reg)
                y.append(t["classe"])
                pred.append(1 if classe_pred > 0 else 0)
            todos.extend(trials)
            met = metricas_trials(y, pred, scores, modelo["limiar"])
            met["decisoes"] = int(decisoes)
            tol, ganho, acao = aplica_dificuldade(tol, ganho, met["ba"])
            historico.append({
                "sessao": int(trials[0]["sessao"]), "bloco": int(bloco),
                "papel": papel, "n_trials": len(trials),
                "ba": met["ba"], "ic95": met["ic95"],
                "score_medio_abs": met.get("score_medio_abs"),
                "decisoes": met.get("decisoes"),
                "tolerancia": tol, "ganho": ganho, "acao_dificuldade": acao,
            })
            log(f"  sessao {trials[0]['sessao']} bloco {bloco + 1}: {papel:10s} "
                f"BA={met['ba']:.3f} IC95=[{met['ic95'][0]:.3f},{met['ic95'][1]:.3f}] "
                f"tol={tol:.2f} ganho={ganho:.2f} ({acao})")
            if papel == "adaptacao":
                modelo = treina_decoder(todos)
    return {"modelo": modelo, "historico": historico, "trials": todos,
            "tolerancia_final": tol, "ganho_final": ganho}


def roda_teste_congelado(trials, modelo, log=print):
    """Sessao final: decoder CONGELADO (sem adaptacao) e metricas honestas."""
    y, pred, scores = [], [], []
    for t in trials:
        score, _, _ = pontua_trial(modelo, t)
        scores.append(score)
        classe_pred, _ = decide(score, modelo["limiar"])
        y.append(int(t["classe"]))
        pred.append(1 if classe_pred > 0 else 0)
    met = metricas_trials(y, pred, scores, modelo["limiar"])
    erd = erd_por_canal(trials)
    veredito = avalia_nao_respondedor(met)
    log(f"  teste congelado: BA={met['ba']:.3f} "
        f"IC95=[{met['ic95'][0]:.3f},{met['ic95'][1]:.3f}] n={met['n']} "
        f"-> {'NAO-RESPONDEDOR' if veredito['nao_respondedor'] else 'respondedor'}")
    return {"metricas": met, "erd": erd, "veredito": veredito}


# ==========================================================================
# 5. Plano/protocolo, relatorio e CLI
# ==========================================================================
def plano_protocolo(trials_por_classe=None, blocos_por_sessao=None):
    """Valida e descreve o plano de 3 sessoes (60 trials/classe)."""
    p = dict(PROTOCOLO)
    if trials_por_classe is not None:
        p["trials_por_classe_por_sessao"] = int(trials_por_classe)
    if blocos_por_sessao is not None:
        p["blocos_por_sessao"] = int(blocos_por_sessao)
    atpc = p["trials_por_classe_por_sessao"]
    blocos = p["blocos_por_sessao"]
    if atpc > 20:
        raise SystemExit("protocolo: no maximo 20 trials/classe por sessao")
    if blocos < 1 or atpc % blocos:
        raise SystemExit("protocolo: trials/classe/sessao deve ser divisivel pelos blocos")
    p["trials_por_classe_por_bloco"] = atpc // blocos
    p["trials_por_sessao"] = 2 * atpc
    p["total_trials_por_classe"] = p["sessoes"] * atpc
    p["total_trials"] = 2 * p["total_trials_por_classe"]
    p["duracao_trial_s"] = (p["cue_s"] + p["me_s"] + p["transicao_s"] +
                            p["mi_s"] + sum(p["iti_s"]) / 2.0)
    p["duracao_treino_min"] = round(
        (p["sessoes"] - 1) * p["trials_por_sessao"] * p["duracao_trial_s"] / 60.0, 1)
    if p["total_trials_por_classe"] != 60:
        raise SystemExit(
            f"protocolo: total de trials/classe deveria ser 60, deu "
            f"{p['total_trials_por_classe']} (3 sessoes x 20/classe)")
    return p


def imprime_plano(p):
    print("=== PLANO DO PROTOCOLO DE TREINO L/R ADAPTATIVO ===")
    print(f"  sessoes: {p['sessoes']} (treino adaptativo: {p['sessoes_treino_adaptativo']}, "
          f"teste congelado: {p['sessao_teste_congelado']})")
    print(f"  trials/classe por sessao: {p['trials_por_classe_por_sessao']} "
          f"em {p['blocos_por_sessao']} blocos de "
          f"{p['trials_por_classe_por_bloco']}/classe (pausa {p['pausa_entre_blocos_s']} s)")
    print(f"  total: {p['total_trials']} trials = {p['total_trials_por_classe']} "
          f"trials/classe distribuidos em 3 sessoes (nunca em sequencia)")
    print(f"  trial: cue {p['cue_s']} s -> ME {p['me_s']} s -> transicao "
          f"{p['transicao_s']} s -> MI {p['mi_s']} s -> ITI {p['iti_s']} s")
    print(f"  duracao estimada da sessao de treino: "
          f"{p['trials_por_sessao'] * p['duracao_trial_s'] / 60.0:.1f} min (+ pausas)")
    print(f"  criterios: respondendor BA>={p['limiar_respondedor_ba']:.2f} (IC95% "
          f"inferior); sucesso BA>={p['limiar_sucesso_ba']:.2f}")
    print("  3 sessoes x 20/classe = 60/classe; a sessao 3 usa o decoder CONGELADO")


def relatorio_md(resultado, plano):
    l = ["# Treino L/R adaptativo + sessao final congelada — relatorio", ""]
    l.append(f"- origem: **{resultado.get('origem')}**")
    l.append(f"- plano: {plano['sessoes']} sessoes x "
             f"{plano['trials_por_classe_por_sessao']} trials/classe "
             f"({plano['total_trials_por_classe']}/classe no total)")
    l.append("")
    l.append("## Treino adaptativo (bloco a bloco)")
    l.append("")
    l.append("| sessao | bloco | papel | trials | BA | IC95% | tol | ganho | acao |")
    l.append("|---|---:|---|---:|---:|---|---:|---:|---|")
    for h in resultado.get("historico", []):
        l.append(f"| {h['sessao']} | {h['bloco'] + 1} | {h['papel']} | "
                 f"{h['n_trials']} | {h['ba']:.3f} | "
                 f"[{h['ic95'][0]:.3f}, {h['ic95'][1]:.3f}] | "
                 f"{h['tolerancia']:.2f} | {h['ganho']:.2f} | {h['acao_dificuldade']} |")
    l.append("")
    teste = resultado.get("teste_congelado")
    if teste:
        m = teste["metricas"]
        v = teste["veredito"]
        l.append("## Sessao final — decoder congelado")
        l.append("")
        l.append(f"- BA = **{m['ba']:.3f}** (IC95% "
                 f"[{m['ic95'][0]:.3f}, {m['ic95'][1]:.3f}], n={m['n']})")
        l.append(f"- confusao (linhas=verdade L/R): {m['confusao']}")
        l.append(f"- |score| medio: {m.get('score_medio_abs', float('nan')):.3f}")
        l.append(f"- **nao-respondedor: {v['nao_respondedor']}** "
                 f"({v['criterio']}; {v['motivo']})")
        l.append("")
        l.append("## ERD/ERS mu-beta da MI (relativo ao baseline, %; "
                 "negativo = dessincronizacao esperada)")
        l.append("")
        for banda, por_classe in teste["erd"].items():
            for classe, valores in por_classe.items():
                l.append(f"- {banda} / MI {classe}: C3={valores[0]:+.1f}%, "
                         f"Cz={valores[1]:+.1f}%, C4={valores[2]:+.1f}%")
        l.append("")
    l.append("## Criterios pre-fixados")
    l.append("")
    l.append(f"- respondedor: BA >= {plano['limiar_respondedor_ba']:.2f} "
             "(IC95% inferior);")
    l.append(f"- sucesso: BA >= {plano['limiar_sucesso_ba']:.2f};")
    l.append("- questionario de estrategia preenchido ao fim de cada sessao.")
    l.append("")
    return "\n".join(l)


def _salva_modelo(modelo, caminho):
    import joblib
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": modelo["pipeline"], "limiar": modelo["limiar"],
                 "meta": {k: v for k, v in modelo.items()
                          if k not in ("pipeline",)}}, caminho)
    return caminho


def _carrega_modelo(caminho):
    import joblib
    d = joblib.load(caminho)
    return {"pipeline": d["pipeline"], "limiar": float(d["limiar"]),
            **{k: v for k, v in (d.get("meta") or {}).items()}}


def questionario_estrategia(caminho):
    """Template do questionario de estrategia da MI (preenchido pelo operador)."""
    itens = [
        ("o_que_imagina", "O que voce imaginou durante a MI? (movimento, sensacao, outro)"),
        ("sensacao_muscular", "Sentiu algum movimento/contracao muscular? (0-4)"),
        ("esforco", "Esforco percebido na MI (0-4)"),
        ("fadiga", "Fadiga (0-4)"),
        ("confianca", "Quao confiante estava no lado imaginado (0-4)"),
        ("visualizacao", "Gravidade da estrategia visual (1a pessoa x 3a pessoa)"),
        ("musica_ruido", "Fatores externos (ruido, luz, desconforto)"),
        ("observacoes", "Observacoes do experimentador"),
    ]
    dados = {"sessao": None, "data": None,
             "itens": [{"id": i, "pergunta": q, "resposta": None}
                       for i, q in itens]}
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(dados, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    return caminho


# ==========================================================================
# 6. main
# ==========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="roda 3 sessoes sinteticas (sem hardware) e gera relatorio")
    ap.add_argument("--plano", action="store_true",
                    help="mostra/valida o plano do protocolo e sai")
    ap.add_argument("--sessao-dir", type=Path, default=None,
                    help="pasta de uma sessao real (CSV+JSON do validacao_eeg_3classes.py)")
    ap.add_argument("--modo", choices=("treino", "teste-congelado"), default="treino")
    ap.add_argument("--modelo", type=Path, default=None,
                    help="modelo .joblib para continuar/avaliar")
    ap.add_argument("--salvar-modelo", type=Path, default=None)
    ap.add_argument("--saida", type=Path, default=None,
                    help="pasta de saida (default: temporaria)")
    ap.add_argument("--trials-por-classe", type=int, default=20)
    ap.add_argument("--blocos-por-sessao", type=int, default=2)
    ap.add_argument("--render", type=Path, default=None,
                    help="pasta para salvar frames do feedback (PNG)")
    ap.add_argument("--questionario", type=Path, default=None,
                    help="caminho do JSON do questionario de estrategia a criar")
    ap.add_argument("--sujeito-illiterado", action="store_true",
                    help="(smoke) simula participante sem ERD (nao-respondedor)")
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plano = plano_protocolo(args.trials_por_classe, args.blocos_por_sessao)
    if args.plano and not args.smoke and args.sessao_dir is None:
        imprime_plano(plano)
        return 0
    saida = Path(args.saida) if args.saida else (
        Path(tempfile.gettempdir()) / "feedback_adaptativo")
    saida.mkdir(parents=True, exist_ok=True)
    render = Path(args.render) if args.render else None
    if render:
        render.mkdir(parents=True, exist_ok=True)
    resultado = {"origem": "sintetico" if args.smoke else "bancada",
                 "plano": plano}

    if args.smoke:
        print("=== SMOKE: 3 sessoes sinteticas (SEM hardware, NAO e' resultado de bancada) ===")
        sessoes = []
        for s in (1, 2, 3):
            sessoes.append(gerar_sessao_sintetica(
                s, trials_por_classe=plano["trials_por_classe_por_sessao"],
                blocos=plano["blocos_por_sessao"], seed=args.seed + s,
                erd_forte=not args.sujeito_illiterado))
        modelo_inicial = None
        if args.modelo is not None:
            modelo_inicial = _carrega_modelo(args.modelo)
        print("-- sessoes 1-2 (treino adaptativo) --")
        treino = roda_treino_adaptativo(sessoes[:2], modelo=modelo_inicial,
                                        janela=False, render_dir=render)
        resultado["historico"] = treino["historico"]
        resultado["tolerancia_final"] = treino["tolerancia_final"]
        resultado["ganho_final"] = treino["ganho_final"]
        modelo = treino["modelo"]
        if args.salvar_modelo:
            print("modelo salvo em", _salva_modelo(modelo, args.salvar_modelo))
            resultado["modelo_salvo"] = str(args.salvar_modelo)
        print("-- sessao 3 (teste congelado) --")
        resultado["teste_congelado"] = roda_teste_congelado(sessoes[2], modelo)
        # progresso da sessao 3 sem adaptacao (feedback ainda roda)
        prog = []
        for t in sessoes[2]:
            score, _, _ = pontua_trial(modelo, t)
            prog.append({"classe": t["classe"], "score": score,
                         "acerto": (score >= modelo["limiar"]) == (t["classe"] == 1)})
        resultado["teste_progresso"] = prog
    elif args.sessao_dir is not None:
        print(f"=== sessao real: {args.sessao_dir} | modo {args.modo} ===")
        trials, meta = carrega_sessao_eeg(args.sessao_dir)
        print(f"  {len(trials)} trials, fs={meta.get('fs', '?')} Hz, "
              f"blocos={len({t['bloco'] for t in trials})}")
        resultado["sessao_dir"] = str(args.sessao_dir)
        if args.modo == "treino":
            modelo = _carrega_modelo(args.modelo) if args.modelo else None
            treino = roda_treino_adaptativo([trials], modelo=modelo,
                                            janela=False, render_dir=render)
            resultado["historico"] = treino["historico"]
            resultado["tolerancia_final"] = treino["tolerancia_final"]
            resultado["ganho_final"] = treino["ganho_final"]
            modelo = treino["modelo"]
            if args.salvar_modelo:
                _salva_modelo(modelo, args.salvar_modelo)
                resultado["modelo_salvo"] = str(args.salvar_modelo)
        else:
            if args.modelo is None:
                raise SystemExit("--modo teste-congelado exige --modelo .joblib")
            modelo = _carrega_modelo(args.modelo)
            resultado["teste_congelado"] = roda_teste_congelado(trials, modelo)
    else:
        imprime_plano(plano)
        print("\nNada executado: use --smoke, --sessao-dir ou --plano.")
        return 0

    texto = relatorio_md(resultado, plano)
    (saida / "relatorio.md").write_text(texto + "\n", encoding="utf-8")
    (saida / "resultados.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    if args.questionario:
        questionario_estrategia(args.questionario)
        print("questionario em", args.questionario)
    print("\n" + texto)
    print(f"\nSaidas em {saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
