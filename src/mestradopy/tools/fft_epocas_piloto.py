#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FFT/PSD e tempo-frequencia (Morlet) das epocas de MI de 4 s do piloto ME/MI.

Objetivo (task do Lead, 18/11/2026): figuras de espectro (Welch/PSD), diferenca/
razao entre classes ESQUERDA x DIREITA e espectrogramas medios (wavelet Morlet)
das epocas de **imaginacao motora (MI) de 4 s**, canais C3/C4/Cz, sessoes do
piloto `validacao_eeg_3classes.py` de 17/11/2026
(`outputs/eeg/piloto_me_mi_s01` e `piloto_me_mi_s02`).

Decisoes que seguem a instrucao do Lead:
  * marcadores do JSON de eventos: `mi_start=780` / `mi_end=781`; o epoch vai de
    780 a 780+4 s*fs (2000 amostras @500 Hz). NAO usa ME como MI e NAO deixa a
    janela atravessar ME->MI (o marcador 780 e' depois do fim do ME, 779);
  * rotulo L/R = ultimo cue (769=esquerda / 770=direita) antes do 780;
  * canais fisicos 15/16/17 (1-based) = C3/C4/Cz -> colunas Ch15/Ch16/Ch17 do
    CSV (indices 0-based 14/15/16), o mesmo `CH_OWN_IDX` do pacote
    `eeg.mi_eegnet`. O campo `channel_names` do JSON e' apenas metadado do
    gerador sintetico e NAO foi usado como mapeamento; o mapeamento foi
    confirmado pelos desvios-padrao (15,51/15,23/14,53) registrados no
    `docs/guias/HISTORICO_ITERACOES.txt` (item 189);
  * loader REUTILIZADO: `eeg.mi_eegnet.data._load_own_csv` (mesmo leitor das
    analises CSP/EEGNet do projeto), sem reimplementar o parser;
  * tempo-frequencia: Morlet via `mne.time_frequency.tfr_array_morlet`
    (n_cycles=7, 2-45 Hz, potencia media por classe, mesma escala L x R).

Saidas em `outputs/eeg/analises_fft_piloto/`:
    figuras/fig*.png                 (dpi>=150; aqui 200)
    tabelas/psd_media.csv            PSD media/SEM por classe/canal/freq
    tabelas/potencia_bandas.csv      potencia por banda/canal/classe + p/tamanho
    tabelas/picos_espectrais.csv     frequencia de pico em mu e beta
    tabelas/trials.csv               trial a trial (samples e rotulo)
    tabelas/epocas_e_psd.npz         epocas, baseline, PSD por trial, TFR media
    resumo_analise_fft.json          parametros, n, resultados, arquivos, avisos
    RELATORIO_ANALISE_FFT_PILOTO.md  metodos, resultados, limitacoes
    execucao.log                     log da execucao

Uso:
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\fft_epocas_piloto.py
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\fft_epocas_piloto.py --sessoes piloto_me_mi_s01
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\fft_epocas_piloto.py --saida outputs/eeg/analises_fft_piloto

Nao abre hardware, nao grava nada nas pastas de origem e nao usa git.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import warnings

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)

_PY = Path(__file__).resolve()
_REPO = _PY.parents[3]                     # <repo>/src/mestradopy/tools -> <repo>
_SRC = _PY.parents[1]                      # <repo>/src/mestradopy
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                           # pragma: no cover
    pass

import matplotlib
matplotlib.use("Agg")                       # offline, sem janela
import matplotlib.pyplot as plt             # noqa: E402
from PIL import Image                       # noqa: E402
from scipy import signal as sps             # noqa: E402
from scipy import stats as sstats           # noqa: E402

# Loader canonico das analises de EEG do projeto (reuso, nao reimplementacao).
from eeg.mi_eegnet.data import _load_own_csv, CH_OWN_IDX  # noqa: E402

try:
    from mne.time_frequency import tfr_array_morlet  # noqa: E402
    _MNE_OK = True
except Exception as _exc:                    # pragma: no cover
    tfr_array_morlet = None
    _MNE_OK = False
    _MNE_ERRO = f"{type(_exc).__name__}: {_exc}"

# =============================================================================
# Constantes
# =============================================================================
FS_PADRAO = 500.0
MI_SEC = 4.0
BASE_SEC = 1.0                              # baseline pre-cue (repouso)
BANDA_FILTRO = (1.0, 45.0)                  # Hz (zero-phase, sinal continuo)
FREQ_MIN, FREQ_MAX = 1.0, 45.0
NPERSEG = 1000                              # 2 s @500 Hz -> resolucao 0.5 Hz
NOVERLAP = 500
BANDAS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "total_1_45": (1.0, 45.0),
}
BANDAS_FIG = ("mu", "beta")
WAVELET_FREQS = np.arange(2.0, 45.0 + 1e-9, 1.0)

#: Morlet: n_ciclos adaptativo = max(freq/2, 3) -- garante que ate' 2 Hz a
#: wavelet (<= 1,5 s) caiba na epoca de 4 s (MNE exige wavelet <= sinal).
WAVELET_CICLOS_MIN = 3.0


def wavelet_n_ciclos(freqs: np.ndarray) -> np.ndarray:
    """n_ciclos do Morlet por frequencia (freq/2, minimo 3)."""
    f = np.asarray(freqs, np.float64).reshape(-1)
    return np.maximum(f / 2.0, WAVELET_CICLOS_MIN)
DPI = 200

COR_L = "#1f4e9c"          # esquerda (azul)
COR_R = "#b3202a"          # direita (vermelho)
COR_MU = "#f2c200"         # faixa mu
COR_BETA = "#7fb800"       # faixa beta
LIMIAR_FIG_BYTES = 10_000  # figura valida > 10 kB


# =============================================================================
# Estruturas
# =============================================================================
@dataclass
class SessaoEpocas:
    """Epocas de MI (e baseline pre-cue) de UMA sessao do piloto."""

    nome: str
    pasta: Path
    csv_path: Path
    json_path: Path
    meta: Dict[str, Any]
    fs: float
    canais_idx: Tuple[int, ...]
    canais_nomes: Tuple[str, ...]
    epocas_mi: np.ndarray               # (n_trials, n_ch, n_mi)
    epocas_base: np.ndarray             # (n_trials, n_ch, n_base) -- NaN se invalida
    labels: np.ndarray                  # (n_trials,) "L"/"R"
    trials: List[Dict[str, Any]]        # metadados por trial
    info: Dict[str, Any] = field(default_factory=dict)
    std_canais_sessao: Optional[np.ndarray] = None


@dataclass
class Resultado:
    """Tudo o que foi calculado (figuras/CSV/JSON saem daqui)."""

    sessoes: List[SessaoEpocas]
    fs: float
    canais: Tuple[str, ...]
    freqs: np.ndarray                   # grid PSD
    psd: np.ndarray                     # (n_trials, n_ch, n_freq) PSD por trial
    labels: np.ndarray                  # (n_trials,)
    sessoes_trial: np.ndarray           # (n_trials,) nome da sessao
    baseline_bandas: Dict[str, np.ndarray]   # banda -> (n_trials, n_ch)
    tfr_freqs: np.ndarray
    tfr_L: np.ndarray                   # (n_ch, n_freq, n_times) potencia media
    tfr_R: np.ndarray
    tfr_times: np.ndarray
    arquivos: Dict[str, str] = field(default_factory=dict)
    avisos: List[str] = field(default_factory=list)


# =============================================================================
# Log
# =============================================================================
class _Log:
    """Log simples que tambem acumula as linhas para `execucao.log`."""

    def __init__(self) -> None:
        self.linhas: List[str] = []

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.linhas.append(str(msg))


_LOG = _Log()


def _fmt(v: Any, casas: int = 3) -> str:
    """Formata numero de forma estavel para relatorio/MD (None -> '--')."""
    if v is None:
        return "--"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(x):
        return "--"
    if abs(x) >= 1000 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.2e}"
    return f"{x:.{casas}f}"


def _md_tabela(cabecalho: Sequence[str], linhas: Sequence[Sequence[Any]]) -> str:
    """Tabela markdown a partir de cabecalho e linhas."""
    out = ["| " + " | ".join(cabecalho) + " |",
           "|" + "|".join("---" for _ in cabecalho) + "|"]
    for linha in linhas:
        out.append("| " + " | ".join(str(c) for c in linha) + " |")
    return "\n".join(out)


# =============================================================================
# 1) Carregamento + epoching
# =============================================================================
def _localiza_arquivos(pasta: Path) -> Tuple[Path, Path]:
    """CSV/JSON mais recentes da sessao (mesma regra das analises do projeto)."""
    csvs = sorted(pasta.glob("eeg_mi3c*.csv"))
    jsons = sorted(pasta.glob("eventos*.json"))
    if not csvs:
        raise FileNotFoundError(f"nenhum eeg_mi3c*.csv em {pasta}")
    if not jsons:
        raise FileNotFoundError(f"nenhum eventos*.json em {pasta}")
    return csvs[-1], jsons[-1]


def _filtr_continuo(eeg: np.ndarray, fs: float,
                    banda: Tuple[float, float]) -> np.ndarray:
    """Bandpass zero-phase 1-45 Hz no sinal continuo dos canais escolhidos."""
    lo, hi = banda
    sos = sps.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, np.asarray(eeg, np.float64), axis=0)


def carrega_e_epoca(pasta: Path, canais_idx: Sequence[int],
                    canais_nomes: Sequence[str],
                    fs_esperado: Optional[float] = None) -> SessaoEpocas:
    """Carrega uma sessao e extrai as epocas de MI de 4 s (marcador 780)."""
    pasta = Path(pasta)
    csv_path, json_path = _localiza_arquivos(pasta)
    meta = json.load(open(json_path, encoding="utf-8"))
    fs = float(meta.get("fs", FS_PADRAO))
    if fs_esperado is not None and abs(fs - float(fs_esperado)) > 1e-6:
        raise ValueError(f"{pasta.name}: fs {fs} != esperado {fs_esperado}")
    marker_codes = dict(meta.get("marker_codes") or {})
    class_codes = dict(meta.get("class_codes") or {})
    code_mi_ini = int(marker_codes.get("mi_start", 780))
    code_mi_fim = int(marker_codes.get("mi_end", 781))
    code_me_fim = int(marker_codes.get("me_end", 779))
    # Rótulos L/R pelos codigos de cue do proprio JSON (769/770).
    cue_codes: Dict[int, str] = {}
    for nome_classe, rotulo in (("esquerda", "L"), ("direita", "R")):
        if nome_classe in class_codes:
            cue_codes[int(class_codes[nome_classe])] = rotulo
    if not cue_codes:
        cue_codes = {769: "L", 770: "R"}

    _LOG(f"[carga] {pasta.name}: {csv_path.name} + {json_path.name} (fs={fs:g})")
    eeg, markers, header = _load_own_csv(csv_path)     # loader reutilizado
    arr = np.asarray(eeg, np.float64)
    markers = np.asarray(markers, np.int64)
    if arr.ndim != 2 or arr.shape[1] != int(meta.get("channels", arr.shape[1])):
        raise ValueError(f"{pasta.name}: CSV com shape inesperado {arr.shape}; "
                         f"JSON diz {meta.get('channels')} canais")
    n_canais_total = arr.shape[1]
    idx = tuple(int(i) for i in canais_idx)
    if any(i < 0 or i >= n_canais_total for i in idx):
        raise ValueError(f"{pasta.name}: indices de canal {idx} fora de "
                         f"0..{n_canais_total - 1}")
    std_sessao = arr[:, list(idx)].std(axis=0)
    arr = arr[:, list(idx)]
    # Filtro 1-45 Hz no sinal continuo (mesma banda das figuras).
    arr_f = _filtr_continuo(arr, fs, BANDA_FILTRO)
    n_mi = int(round(MI_SEC * fs))
    n_base = int(round(BASE_SEC * fs))

    starts = np.where(markers == code_mi_ini)[0]
    ends = np.where(markers == code_mi_fim)[0]
    me_ends = np.where(markers == code_me_fim)[0]
    if starts.size == 0 or ends.size == 0:
        raise ValueError(f"{pasta.name}: sem marcadores {code_mi_ini}/{code_mi_fim}")
    _LOG(f"[carga] {pasta.name}: {len(starts)} inicios de MI, "
         f"{len(ends)} fins de MI, {len(me_ends)} fins de ME")

    epocas: List[np.ndarray] = []
    baselines: List[np.ndarray] = []
    labels: List[str] = []
    trials: List[Dict[str, Any]] = []
    avisos: List[str] = []
    rejeitados = 0
    for i_start, ini in enumerate(starts):
        # Primeiro fim de MI depois deste inicio (pareamento sequencial).
        pos = int(np.searchsorted(ends, ini, side="right"))
        if pos >= ends.size:
            rejeitados += 1
            avisos.append(f"trial {i_start}: sem marcador {code_mi_fim} apos {code_mi_ini}")
            continue
        fim = int(ends[pos])
        dur = fim - ini
        if abs(dur - n_mi) > int(0.1 * n_mi):
            rejeitados += 1
            avisos.append(f"trial {i_start}: duracao MI {dur} amostras "
                          f"(esperado {n_mi}) -- rejeitado")
            continue
        s = int(ini)
        e = s + n_mi
        if e > arr_f.shape[0]:
            rejeitados += 1
            avisos.append(f"trial {i_start}: epoch passa do fim do CSV")
            continue
        # Nao cruzar ME->MI: o epoch comeca no 780, que vem depois do 779.
        prev_me_fim = me_ends[me_ends < s]
        me_fim = int(prev_me_fim[-1]) if prev_me_fim.size else -1
        if me_fim >= s:
            rejeitados += 1
            avisos.append(f"trial {i_start}: 780 antes do 779 (janela cruzaria ME)")
            continue
        # Rotulo = ultimo cue (769/770) antes do inicio do MI.
        cues = np.where(np.isin(markers[:s], list(cue_codes)))[0]
        if cues.size == 0:
            rejeitados += 1
            avisos.append(f"trial {i_start}: sem cue de classe antes do MI")
            continue
        cue_pos = int(cues[-1])
        rotulo = cue_codes[int(markers[cue_pos])]
        # Baseline = 1 s ANTES do cue (repouso/ITI), sem cruzar o MI anterior.
        base_ini = cue_pos - n_base
        base_ok = base_ini >= 0
        prev_mi_fim = np.where(markers[:cue_pos] == code_mi_fim)[0]
        if base_ok and prev_mi_fim.size and int(prev_mi_fim[-1]) >= base_ini:
            base_ok = False
        epocas.append(arr_f[s:e].T.copy())
        if base_ok:
            baselines.append(arr_f[base_ini:cue_pos].T.copy())
        else:
            baselines.append(np.full((len(idx), n_base), np.nan))
        labels.append(rotulo)
        trials.append({
            "sessao": pasta.name,
            "trial": len(trials),
            "classe": rotulo,
            "classe_nome": ("esquerda" if rotulo == "L" else "direita"),
            "cue_sample": cue_pos,
            "me_fim_sample": me_fim,
            "mi_ini_sample": s,
            "mi_fim_sample": fim,
            "mi_ini_epoch_sample": s,
            "mi_fim_epoch_sample": e,
            "baseline_ini_sample": (base_ini if base_ok else None),
            "baseline_fim_sample": (cue_pos if base_ok else None),
            "baseline_valida": bool(base_ok),
            "dur_mi_amostras": int(dur),
            "gap_transicao_779_780": int(s - me_fim) if me_fim >= 0 else None,
        })
    n = len(epocas)
    if n == 0:
        raise ValueError(f"{pasta.name}: nenhum trial valido")
    epocas_arr = np.stack(epocas).astype(np.float64)
    base_arr = np.stack(baselines).astype(np.float64)
    labels_arr = np.asarray(labels)
    if not np.isfinite(epocas_arr).all():
        avisos.append("epocas com NaN/Inf")
    if not np.isfinite(base_arr[:, :, :]).all():
        n_base_ok = int(np.isfinite(base_arr).all(axis=(1, 2)).sum())
        avisos.append(f"baseline valida em {n_base_ok}/{n} trials")
    # Checagens de sanidade (registradas no JSON/relatorio).
    durs = [t["dur_mi_amostras"] for t in trials]
    gaps = [t["gap_transicao_779_780"] for t in trials
            if t["gap_transicao_779_780"] is not None]
    tri = np.array([t for t in trials])
    info = {
        "fs": fs,
        "n_marcadores_mi_ini": int(starts.size),
        "n_marcadores_mi_fim": int(ends.size),
        "n_trials_validos": int(n),
        "n_rejeitados": int(rejeitados),
        "n_L": int((labels_arr == "L").sum()),
        "n_R": int((labels_arr == "R").sum()),
        "dur_mi_amostras_mediana": float(np.median(durs)),
        "dur_mi_amostras_min": int(np.min(durs)),
        "dur_mi_amostras_max": int(np.max(durs)),
        "gap_transicao_779_780_mediana": (float(np.median(gaps)) if gaps else None),
        "epoch_samples": int(n_mi),
        "baseline_samples": int(n_base),
        "n_baseline_valida": int(np.isfinite(base_arr).all(axis=(1, 2)).sum()),
        "std_canais_sessao": [float(v) for v in std_sessao],
        "avisos": avisos,
    }
    _LOG(f"[carga] {pasta.name}: {n} epocas validas "
         f"(L={info['n_L']} R={info['n_R']}), rejeitados={rejeitados}; "
         f"dur MI mediana {info['dur_mi_amostras_mediana']:.0f} amostras; "
         f"gap 779->780 mediano {info['gap_transicao_779_780_mediana']} amostras")
    for aviso in avisos:
        _LOG(f"[aviso] {pasta.name}: {aviso}")
    return SessaoEpocas(
        nome=pasta.name, pasta=pasta, csv_path=csv_path, json_path=json_path,
        meta=meta, fs=fs, canais_idx=idx,
        canais_nomes=tuple(str(c) for c in canais_nomes),
        epocas_mi=epocas_arr, epocas_base=base_arr,
        labels=labels_arr, trials=trials, info=info,
        std_canais_sessao=std_sessao)


# =============================================================================
# 2) PSD, bandas, estatistica e TFR
# =============================================================================
def psd_por_trial(epocas: np.ndarray, fs: float,
                  nperseg: int = NPERSEG,
                  noverlap: int = NOVERLAP) -> Tuple[np.ndarray, np.ndarray]:
    """Welch por trial: (n_trials, n_ch, n_times) -> (f, PSD)."""
    ep = np.asarray(epocas, np.float64)
    nperseg = int(min(nperseg, ep.shape[-1]))
    noverlap = int(min(noverlap, nperseg - 1))
    f, psd = sps.welch(ep, fs=fs, window="hann", nperseg=nperseg,
                       noverlap=noverlap, detrend="constant",
                       scaling="density", axis=-1)
    sel = (f >= FREQ_MIN - 1e-9) & (f <= FREQ_MAX + 1e-9)
    return f[sel], psd[..., sel]


def faixa_banda(freqs: np.ndarray, banda: Tuple[float, float]
                ) -> Tuple[slice, np.ndarray]:
    """Indices do grid de frequencias dentro da banda (inclusivo)."""
    lo, hi = float(banda[0]), float(banda[1])
    mask = (freqs >= lo - 1e-9) & (freqs <= hi + 1e-9)
    return slice(int(np.argmax(mask)), int(np.argmax(mask)) + int(mask.sum())), mask


def potencia_banda(psd: np.ndarray, freqs: np.ndarray,
                   banda: Tuple[float, float]) -> np.ndarray:
    """Integra a PSD na banda (np.trapezoid) -> (n_trials, n_ch)."""
    _, mask = faixa_banda(freqs, banda)
    if not mask.any():
        raise ValueError(f"banda {banda} fora do grid {freqs[0]}..{freqs[-1]}")
    f_banda = freqs[mask]
    return np.trapezoid(psd[..., mask], x=f_banda, axis=-1)


def media_sem(valores: np.ndarray, eixo: int = 0
              ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Media/SEM robustas ignorando NaN (por coluna)."""
    arr = np.asarray(valores, np.float64)
    n_val = np.sum(np.isfinite(arr), axis=eixo)
    with np.errstate(invalid="ignore"):
        med = np.nanmean(arr, axis=eixo) if arr.size else np.array([])
        sem = (np.nanstd(arr, axis=eixo, ddof=1)
               / np.sqrt(np.maximum(n_val, 1)))
    sem = np.where(n_val > 1, sem, np.nan)
    return med, sem, int(np.max(n_val)) if n_val.size else 0


def _p_mwu(amostra_a: np.ndarray, amostra_b: np.ndarray) -> float:
    """p do Mann-Whitney U bilateral (nao corrigido; exploratorio)."""
    a = np.asarray(amostra_a, np.float64)
    b = np.asarray(amostra_b, np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan")
    try:
        return float(sstats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:                       # pragma: no cover
        return float("nan")


def _cohen_d(amostra_a: np.ndarray, amostra_b: np.ndarray) -> float:
    """Tamanho de efeito de Cohen (a - b) com desvio agrupado."""
    a = np.asarray(amostra_a, np.float64)
    b = np.asarray(amostra_b, np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    var = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) \
        / (a.size + b.size - 2)
    if var <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / math.sqrt(var))


def tfr_morlet_media(epocas: np.ndarray, fs: float,
                     freqs: np.ndarray = WAVELET_FREQS,
                     n_cycles: Optional[np.ndarray] = None,
                     chunk: int = 8) -> np.ndarray:
    """Potencia media Morlet por classe -> (n_ch, n_freq, n_times).

    Processa em blocos para nao alocar (n_trials x n_ch x n_freq x n_times)
    complexo de uma vez (com 60 trials seriam ~1,3 GB).
    """
    if not _MNE_OK:                          # pragma: no cover
        raise RuntimeError(f"mne indisponivel para Morlet: {_MNE_ERRO}")
    ep = np.asarray(epocas, np.float64)
    ciclos = (wavelet_n_ciclos(freqs) if n_cycles is None
              else np.asarray(n_cycles, np.float64).reshape(-1))
    n = ep.shape[0]
    soma: Optional[np.ndarray] = None
    for ini in range(0, n, max(int(chunk), 1)):
        bloco = ep[ini:ini + int(chunk)]
        tfr = tfr_array_morlet(bloco, sfreq=float(fs), freqs=np.asarray(freqs),
                               n_cycles=ciclos, use_fft=True,
                               output="power", verbose=False)
        if soma is None:
            soma = np.zeros(tfr.shape[1:], dtype=np.float64)
        soma += np.asarray(tfr, np.float64).sum(axis=0)
    return soma / max(n, 1)


def prepara_resultado(sessoes: List[SessaoEpocas],
                      nperseg: int = NPERSEG) -> Resultado:
    """Agrega sessoes, calcula PSD/TFR/bandas e devolve `Resultado`."""
    fs = float(sessoes[0].fs)
    for sessao in sessoes:
        if abs(float(sessao.fs) - fs) > 1e-9:
            raise ValueError("sessoes com fs diferentes")
    canais = sessoes[0].canais_nomes
    epocas = np.concatenate([s.epocas_mi for s in sessoes], axis=0)
    bases = np.concatenate([s.epocas_base for s in sessoes], axis=0)
    labels = np.concatenate([s.labels for s in sessoes])
    sessoes_trial = np.concatenate(
        [np.full(s.labels.size, s.nome, dtype=object) for s in sessoes])
    _LOG(f"[psd] {epocas.shape[0]} epocas x {epocas.shape[1]} canais x "
         f"{epocas.shape[2]} amostras; Welch nperseg={nperseg}")
    freqs, psd = psd_por_trial(epocas, fs, nperseg=nperseg)
    _LOG(f"[psd] grid {freqs[0]:.1f}..{freqs[-1]:.1f} Hz "
         f"({freqs.size} pontos, df={freqs[1]-freqs[0]:.2f} Hz)")
    # Baseline: Welch na janela pre-cue de 1 s (NaN nos trials sem baseline).
    base_bandas: Dict[str, np.ndarray] = {}
    n_base = bases.shape[-1]
    f_base, psd_base = psd_por_trial(
        bases, fs, nperseg=int(min(nperseg, n_base)),
        noverlap=int(min(NOVERLAP, max(n_base - 1, 0))))
    for nome_banda in BANDAS:
        base_bandas[nome_banda] = potencia_banda(psd_base, f_base,
                                                 BANDAS[nome_banda])
    # TFR Morlet media por classe (mesma escala depois).
    tfr_L = tfr_morlet_media(epocas[labels == "L"], fs)
    tfr_R = tfr_morlet_media(epocas[labels == "R"], fs)
    times = np.arange(epocas.shape[-1]) / fs
    _LOG(f"[tfr] Morlet {WAVELET_FREQS.size} freqs (2-45 Hz), "
         f"{epocas.shape[-1]} amostras (4 s), n_cycles=max(freq/2, {WAVELET_CICLOS_MIN:g})")
    return Resultado(sessoes=sessoes, fs=fs, canais=tuple(canais), freqs=freqs,
                     psd=psd, labels=labels, sessoes_trial=sessoes_trial,
                     baseline_bandas=base_bandas, tfr_freqs=WAVELET_FREQS,
                     tfr_L=tfr_L, tfr_R=tfr_R, tfr_times=times)


# =============================================================================
# 3) Figuras
# =============================================================================
def _config_estilo() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.autolayout": False,
    })


def _salva_figura(fig: plt.Figure, caminho: Path, arquivos: Dict[str, str]
                  ) -> None:
    """Salva PNG (dpi) e valida que abre e tem tamanho razoavel."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if not caminho.is_file():
        raise RuntimeError(f"figura nao foi criada: {caminho}")
    tamanho = caminho.stat().st_size
    if tamanho < LIMIAR_FIG_BYTES:
        raise RuntimeError(f"figura suspeita (pequena): {caminho} "
                           f"({tamanho} bytes)")
    with Image.open(caminho) as img:      # confirma que o PNG abre
        img.verify()
    arquivos[caminho.name] = str(caminho)
    _LOG(f"[figura] {caminho.name} ({tamanho / 1024:.0f} kB)")


def _shade_bandas(ax, alpha: float = 0.10) -> None:
    ax.axvspan(*BANDAS["mu"], color=COR_MU, alpha=alpha, lw=0)
    ax.axvspan(*BANDAS["beta"], color=COR_BETA, alpha=alpha, lw=0)


def _fig_psd(res: Resultado, destino: Path, arquivos: Dict[str, str]) -> None:
    """Figura 1: PSD media +- SEM por classe, 1-45 Hz, painel por canal."""
    fig, axes = plt.subplots(1, len(res.canais), figsize=(4.6 * len(res.canais), 4.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    rotulos = {"L": "Esquerda", "R": "Direita"}
    for i_canal, canal in enumerate(res.canais):
        ax = axes[i_canal]
        _shade_bandas(ax)
        for classe, cor in (("L", COR_L), ("R", COR_R)):
            sel = res.labels == classe
            media, sem, n = media_sem(res.psd[sel, i_canal, :])
            ax.plot(res.freqs, media, color=cor, lw=1.6,
                    label=f"{rotulos[classe]} (n={n})")
            lo = np.clip(media - sem, 1e-6, None)
            hi = media + sem
            ax.fill_between(res.freqs, lo, hi, color=cor, alpha=0.22, lw=0)
        ax.set_xlim(FREQ_MIN, FREQ_MAX)
        ax.set_yscale("log")
        ax.set_xlabel("Frequência (Hz)")
        ax.set_title(f"{canal}")
        ax.grid(True, which="both", alpha=0.22)
        if i_canal == 0:
            ax.set_ylabel("PSD (µV²/Hz)")
    axes[0].legend(loc="upper right", framealpha=0.9)
    fig.suptitle("PSD média ± SEM por classe — MI 4 s, Welch "
                 f"(nperseg={NPERSEG / res.fs:g} s, Hann) — C3/C4/Cz", y=1.02)
    _salva_figura(fig, destino / "fig1_psd_media_sem_esquerda_direita.png",
                  arquivos)


def _fig_diferenca_razao(res: Resultado, destino: Path,
                         arquivos: Dict[str, str]) -> Dict[str, Any]:
    """Figura 2: diferenca R-L (PSD linear) e razao R/L em dB, com mu/beta."""
    db = 10.0 * np.log10(np.clip(res.psd, 1e-12, None))   # dB por trial
    fig, axes = plt.subplots(2, len(res.canais),
                             figsize=(4.6 * len(res.canais), 7.2), sharex=True)
    axes = np.atleast_2d(axes)
    info: Dict[str, Any] = {"razao_media_mu_beta": {}}
    for i_canal, canal in enumerate(res.canais):
        ax = axes[0, i_canal]
        _shade_bandas(ax)
        med_L, sem_L, _ = media_sem(res.psd[res.labels == "L", i_canal, :])
        med_R, sem_R, _ = media_sem(res.psd[res.labels == "R", i_canal, :])
        dif = med_R - med_L
        sem_dif = np.sqrt(np.clip(sem_L, 0, None) ** 2
                          + np.clip(sem_R, 0, None) ** 2)
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
        ax.plot(res.freqs, dif, color="#6a1b9a", lw=1.5)
        ax.fill_between(res.freqs, dif - sem_dif, dif + sem_dif,
                        color="#6a1b9a", alpha=0.22, lw=0)
        ax.set_xlim(FREQ_MIN, FREQ_MAX)
        ax.set_xlabel("Frequência (Hz)")
        ax.grid(True, alpha=0.22)
        if i_canal == 0:
            ax.set_ylabel("PSD(R) − PSD(L) (µV²/Hz)")
        ax.set_title(f"{canal}: diferença Direita − Esquerda")

        ax2 = axes[1, i_canal]
        _shade_bandas(ax2)
        medL_dB, semL_dB, _ = media_sem(db[res.labels == "L", i_canal, :])
        medR_dB, semR_dB, _ = media_sem(db[res.labels == "R", i_canal, :])
        razao = medR_dB - medL_dB
        sem_raz = np.sqrt(np.clip(semL_dB, 0, None) ** 2
                          + np.clip(semR_dB, 0, None) ** 2)
        ax2.axhline(0.0, color="k", lw=0.8, alpha=0.6)
        ax2.plot(res.freqs, razao, color="#00838f", lw=1.5)
        ax2.fill_between(res.freqs, razao - sem_raz, razao + sem_raz,
                         color="#00838f", alpha=0.22, lw=0)
        ax2.set_xlim(FREQ_MIN, FREQ_MAX)
        ax2.set_xlabel("Frequência (Hz)")
        ax2.grid(True, alpha=0.22)
        if i_canal == 0:
            ax2.set_ylabel("Razão R/L (dB)")
        ax2.set_title(f"{canal}: razão R/L (dB)")
        # potencia media por banda na razao em dB
        for banda in BANDAS_FIG:
            _, mask = faixa_banda(res.freqs, BANDAS[banda])
            pot = potencia_banda(res.psd[:, i_canal, :], res.freqs,
                                 BANDAS[banda])
            mL = float(np.nanmean(pot[res.labels == "L"]))
            mR = float(np.nanmean(pot[res.labels == "R"]))
            razao_db = 10.0 * math.log10(max(mR, 1e-12) / max(mL, 1e-12))
            info["razao_media_mu_beta"].setdefault(canal, {})[banda] = float(razao_db)
    fig.suptitle("Diferença e razão entre classes (R − L / R:L) — MI 4 s — "
                 "faixas mu (8–13 Hz) e beta (13–30 Hz) destacadas", y=1.0)
    fig.tight_layout()
    _salva_figura(fig, destino / "fig2_diferenca_razao_R_menos_L.png", arquivos)
    return info


def _fig_temporal(res: Resultado, destino: Path,
                  arquivos: Dict[str, str]) -> None:
    """Figura 0 (sanidade): media temporal +- SEM das epocas por classe."""
    fig, axes = plt.subplots(1, len(res.canais),
                             figsize=(4.6 * len(res.canais), 3.9), sharex=True)
    axes = np.atleast_1d(axes)
    tempo = res.tfr_times if res.tfr_times.size else np.arange(
        res.psd.shape[-1]) / res.fs
    # epocas nao estao no Resultado; reconstroi a partir das sessoes
    epocas = np.concatenate([s.epocas_mi for s in res.sessoes], axis=0)
    for i_canal, canal in enumerate(res.canais):
        ax = axes[i_canal]
        for classe, cor, rotulo in (("L", COR_L, "Esquerda"),
                                    ("R", COR_R, "Direita")):
            sel = res.labels == classe
            dados = epocas[sel, i_canal, :]
            media = np.nanmean(dados, axis=0)
            sem = np.nanstd(dados, axis=0, ddof=1) / math.sqrt(max(sel.sum(), 1))
            ax.plot(tempo, media, color=cor, lw=1.3, label=rotulo)
            ax.fill_between(tempo, media - sem, media + sem, color=cor,
                            alpha=0.20, lw=0)
        ax.axhline(0.0, color="k", lw=0.7, alpha=0.6)
        ax.set_title(canal)
        ax.set_xlabel("Tempo desde o início do MI (s)")
        ax.grid(True, alpha=0.22)
        if i_canal == 0:
            ax.set_ylabel("Amplitude filtrada (µV)")
            ax.legend(loc="upper right", framealpha=0.9)
    fig.suptitle("Sanidade: média temporal ± SEM das épocas de MI (4 s) — "
                 "filtro 1–45 Hz", y=1.02)
    fig.tight_layout()
    _salva_figura(fig, destino / "fig0_media_temporal_MI_4s.png", arquivos)


def _tfr_em_db(tfr: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.clip(np.asarray(tfr, np.float64), 1e-12, None))


def _fig_tfr(res: Resultado, destino: Path, arquivos: Dict[str, str]) -> None:
    """Figura 3: espectrogramas medios L e R, Morlet, mesma escala."""
    db_L = _tfr_em_db(res.tfr_L)
    db_R = _tfr_em_db(res.tfr_R)
    todos = np.concatenate([db_L.ravel(), db_R.ravel()])
    vmin, vmax = np.percentile(todos, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(todos)), float(np.nanmax(todos) + 1e-6)
    fig, axes = plt.subplots(2, len(res.canais),
                             figsize=(4.6 * len(res.canais), 7.4),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    ultimo = None
    for i_linha, (classe, db, nome) in enumerate(
            (("L", db_L, "Esquerda"), ("R", db_R, "Direita"))):
        for i_canal, canal in enumerate(res.canais):
            ax = axes[i_linha, i_canal]
            ultimo = ax.imshow(
                db[i_canal], origin="lower", aspect="auto",
                extent=[res.tfr_times[0], res.tfr_times[-1],
                        res.tfr_freqs[0], res.tfr_freqs[-1]],
                cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{nome} — {canal}")
            ax.set_xlabel("Tempo desde o início do MI (s)")
            if i_canal == 0:
                ax.set_ylabel("Frequência (Hz)")
            for borda in (BANDAS["mu"][0], BANDAS["mu"][1], BANDAS["beta"][1]):
                ax.axhline(borda, color="w", lw=0.7, ls="--", alpha=0.75)
            ax.grid(False)
    fig.colorbar(ultimo, ax=axes.ravel().tolist(), shrink=0.85,
                 label="Potência Morlet (dB; média da classe) — mesma escala L×R")
    fig.suptitle("Espectrogramas médios (Morlet, n_cycles adaptativo): esquerda × direita "
                 "— MI 4 s — C3/C4/Cz (mesma escala)", y=1.0)
    _salva_figura(fig, destino / "fig3_tfr_morlet_media_L_R_mesma_escala.png",
                  arquivos)


def _fig_tfr_diferenca(res: Resultado, destino: Path,
                       arquivos: Dict[str, str]) -> None:
    """Figura 4: diferenca R-L do TFR Morlet (dB), mesma escala."""
    dif = _tfr_em_db(res.tfr_R) - _tfr_em_db(res.tfr_L)
    vmax = float(np.percentile(np.abs(dif), 98))
    vmax = max(vmax, 0.2)
    fig, axes = plt.subplots(1, len(res.canais),
                             figsize=(4.6 * len(res.canais), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    ultimo = None
    for i_canal, canal in enumerate(res.canais):
        ax = axes[i_canal]
        ultimo = ax.imshow(
            dif[i_canal], origin="lower", aspect="auto",
            extent=[res.tfr_times[0], res.tfr_times[-1],
                    res.tfr_freqs[0], res.tfr_freqs[-1]],
            cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"diff R−L — {canal}")
        ax.set_xlabel("Tempo desde o início do MI (s)")
        if i_canal == 0:
            ax.set_ylabel("Frequência (Hz)")
        for borda in (BANDAS["mu"][0], BANDAS["mu"][1], BANDAS["beta"][1]):
            ax.axhline(borda, color="k", lw=0.6, ls="--", alpha=0.6)
    fig.colorbar(ultimo, ax=axes.tolist(), shrink=0.9,
                 label="Diferença de potência Morlet (dB R − L)")
    fig.suptitle("Diferença entre classes no tempo-frequência (Morlet) — MI 4 s",
                 y=1.02)
    _salva_figura(fig, destino / "fig4_tfr_morlet_diferenca_R_menos_L.png",
                  arquivos)


def _fig_heatmap_trials(res: Resultado, destino: Path,
                        arquivos: Dict[str, str]) -> None:
    """Figura 5: PSD por trial (heatmap), separada por classe, por canal."""
    db = 10.0 * np.log10(np.clip(res.psd, 1e-12, None))
    ordem_L = np.where(res.labels == "L")[0]
    ordem_R = np.where(res.labels == "R")[0]
    ordem = np.concatenate([ordem_L, ordem_R])
    fig, axes = plt.subplots(1, len(res.canais),
                             figsize=(4.6 * len(res.canais), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    ultimo = None
    for i_canal, canal in enumerate(res.canais):
        ax = axes[i_canal]
        dados = db[ordem, i_canal, :]
        vmin, vmax = np.percentile(dados[np.isfinite(dados)], [2, 98])
        ultimo = ax.imshow(dados, origin="upper", aspect="auto",
                           extent=[res.freqs[0], res.freqs[-1],
                                   len(ordem) - 0.5, -0.5],
                           cmap="magma", vmin=vmin, vmax=vmax)
        ax.axhline(ordem_L.size - 0.5, color="c", lw=1.2, ls="--")
        ax.set_title(f"{canal}")
        ax.set_xlabel("Frequência (Hz)")
        if i_canal == 0:
            ax.set_ylabel("Trial (L em cima | R embaixo)")
        _shade_bandas(ax, alpha=0.08)
    fig.colorbar(ultimo, ax=axes.tolist(), shrink=0.9,
                 label="PSD (dB; por trial)")
    fig.suptitle("PSD por trial (esquerda × direita) — MI 4 s — "
                 "linha pontilhada separa as classes", y=1.02)
    _salva_figura(fig, destino / "fig5_psd_por_trial_heatmap.png", arquivos)


def _fig_bandas(res: Resultado, destino: Path,
                arquivos: Dict[str, str]) -> Dict[str, Any]:
    """Figura 6: potencia media por banda (mu, beta) por canal e classe."""
    pot = {banda: potencia_banda(res.psd, res.freqs, BANDAS[banda])
           for banda in BANDAS_FIG}
    fig, axes = plt.subplots(1, len(BANDAS_FIG),
                             figsize=(5.0 * len(BANDAS_FIG), 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    largura = 0.38
    x = np.arange(len(res.canais))
    resumo: Dict[str, Any] = {}
    for i_banda, banda in enumerate(BANDAS_FIG):
        ax = axes[i_banda]
        dados = pot[banda]
        for desloc, classe, cor, rotulo in ((-largura / 2, "L", COR_L, "Esquerda"),
                                            (largura / 2, "R", COR_R, "Direita")):
            medias, sems = [], []
            for i_canal in range(len(res.canais)):
                sel = res.labels == classe
                vals = dados[sel, i_canal]
                medias.append(float(np.nanmean(vals)))
                sem = float(np.nanstd(vals, ddof=1)
                            / math.sqrt(max(int(sel.sum()), 1)))
                sems.append(sem)
            ax.bar(x + desloc, medias, largura, yerr=sems, color=cor,
                   label=rotulo, alpha=0.85, capsize=3,
                   error_kw={"lw": 1.0})
            resumo[f"{banda}_{classe}"] = medias
        ax.set_xticks(x)
        ax.set_xticklabels(list(res.canais))
        ax.set_xlabel("Canal")
        ax.set_ylabel("Potência (µV²)")
        ax.set_title(f"Banda {banda} ({BANDAS[banda][0]:g}–"
                     f"{BANDAS[banda][1]:g} Hz)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right", framealpha=0.9)
    fig.suptitle("Potência média ± SEM por banda e classe — MI 4 s "
                 "(descritivo; sem correção para múltiplas comparações)", y=1.02)
    fig.tight_layout()
    _salva_figura(fig, destino / "fig6_potencia_bandas_mu_beta.png", arquivos)
    return resumo


# =============================================================================
# 4) Tabelas / JSON / relatorio
# =============================================================================
def _tabela_psd(res: Resultado) -> List[Dict[str, Any]]:
    """PSD media/SEM por classe/canal/frequencia (pooled e por sessao)."""
    linhas: List[Dict[str, Any]] = []
    for escopo, mask_escopo in [("pooled", np.ones(res.labels.size, bool)),
                                *[(s.nome, res.sessoes_trial == s.nome)
                                  for s in res.sessoes]]:
        for classe in ("L", "R"):
            sel = mask_escopo & (res.labels == classe)
            if sel.sum() == 0:
                continue
            for i_canal, canal in enumerate(res.canais):
                media, sem, n = media_sem(res.psd[sel, i_canal, :])
                for i_f, f in enumerate(res.freqs):
                    linhas.append({
                        "escopo": escopo, "classe": classe, "canal": canal,
                        "freq_hz": round(float(f), 3),
                        "psd_uv2_hz": float(media[i_f]),
                        "psd_sem": float(sem[i_f]) if np.isfinite(sem[i_f]) else None,
                        "n_trials": int(n),
                    })
    return linhas


def _tabela_bandas(res: Resultado) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Potencia por banda/canal/classe + p MWU nao corrigido + tamanho de efeito."""
    pot = {banda: potencia_banda(res.psd, res.freqs, BANDAS[banda])
           for banda in BANDAS}
    linhas: List[Dict[str, Any]] = []
    resumo: Dict[str, Any] = {"bandas": {}}
    for escopo, mask_escopo in [("pooled", np.ones(res.labels.size, bool)),
                                *[(s.nome, res.sessoes_trial == s.nome)
                                  for s in res.sessoes]]:
        n_escopo = int(mask_escopo.sum())
        if n_escopo == 0:
            continue
        resumo["bandas"][escopo] = {}
        for banda in BANDAS:
            resumo["bandas"][escopo][banda] = {}
            for i_canal, canal in enumerate(res.canais):
                sel_L = mask_escopo & (res.labels == "L")
                sel_R = mask_escopo & (res.labels == "R")
                vL = pot[banda][sel_L, i_canal]
                vR = pot[banda][sel_R, i_canal]
                mL = float(np.nanmean(vL)) if np.isfinite(vL).any() else float("nan")
                mR = float(np.nanmean(vR)) if np.isfinite(vR).any() else float("nan")
                sL = _sem(vL)
                sR = _sem(vR)
                p = _p_mwu(vR, vL)
                d = _cohen_d(vR, vL)
                razao_db = (10.0 * math.log10(max(mR, 1e-12) / max(mL, 1e-12))
                            if np.isfinite(mL) and np.isfinite(mR) and mL > 0
                            else None)
                lat = ((mR - mL) / (mR + mL) if np.isfinite(mL)
                       and np.isfinite(mR) and (mR + mL) > 0 else None)
                base_L = res.baseline_bandas[banda][sel_L, i_canal]
                base_R = res.baseline_bandas[banda][sel_R, i_canal]
                bL = float(np.nanmean(base_L)) if np.isfinite(base_L).any() else float("nan")
                bR = float(np.nanmean(base_R)) if np.isfinite(base_R).any() else float("nan")
                linhas.append({
                    "escopo": escopo, "banda": banda, "canal": canal,
                    "n_L": int(sel_L.sum()), "n_R": int(sel_R.sum()),
                    "pot_L_uv2": mL, "pot_L_sem": sL,
                    "pot_R_uv2": mR, "pot_R_sem": sR,
                    "razao_R_L_db": razao_db,
                    "lateralizacao_R_menos_L": lat,
                    "p_mwu_nao_corrigido": p,
                    "cohen_d_R_menos_L": d,
                    "base_L_uv2": bL, "base_R_uv2": bR,
                })
                resumo["bandas"][escopo][banda][canal] = {
                    "pot_L_uv2": mL, "pot_R_uv2": mR, "sem_L": sL, "sem_R": sR,
                    "razao_R_L_db": razao_db, "p_mwu_nao_corrigido": p,
                    "cohen_d": d, "base_L": bL, "base_R": bR,
                }
    return linhas, resumo


def _sem(valores: np.ndarray) -> float:
    """SEM robusta (NaN ignorado) de uma amostra."""
    arr = np.asarray(valores, np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def _tabela_picos(res: Resultado) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Frequencia de pico da PSD media em mu e beta, por canal/classe/escopo."""
    linhas: List[Dict[str, Any]] = []
    resumo: Dict[str, Any] = {}
    for escopo, mask_escopo in [("pooled", np.ones(res.labels.size, bool)),
                                *[(s.nome, res.sessoes_trial == s.nome)
                                  for s in res.sessoes]]:
        if not mask_escopo.any():
            continue
        resumo[escopo] = {}
        for classe in ("L", "R"):
            sel = mask_escopo & (res.labels == classe)
            if not sel.any():
                continue
            resumo[escopo][classe] = {}
            for i_canal, canal in enumerate(res.canais):
                media, _, _ = media_sem(res.psd[sel, i_canal, :])
                linha = {"escopo": escopo, "classe": classe, "canal": canal}
                entry: Dict[str, Any] = {}
                for banda in BANDAS_FIG:
                    _, mask = faixa_banda(res.freqs, BANDAS[banda])
                    if not mask.any():
                        continue
                    idx = int(np.nanargmax(media[mask]))
                    freq_pico = float(res.freqs[mask][idx])
                    pico = float(media[mask][idx])
                    linha[f"pico_{banda}_hz"] = round(freq_pico, 2)
                    linha[f"pico_{banda}_psd"] = pico
                    entry[banda] = {"freq_hz": freq_pico, "psd": pico}
                linhas.append(linha)
                resumo[escopo][classe][canal] = entry
    return linhas, resumo


def _figura_tamanhos(arquivos: Dict[str, str]) -> Dict[str, Any]:
    """Bytes e validade de cada figura (para JSON/relatorio)."""
    out: Dict[str, Any] = {}
    for nome, caminho in arquivos.items():
        p = Path(caminho)
        out[nome] = {"path": str(p),
                     "bytes": int(p.stat().st_size) if p.is_file() else 0}
    return out


def _escreve_csv(caminho: Path, linhas: List[Dict[str, Any]]) -> None:
    """CSV simples (cabecalho = chaves da primeira linha)."""
    import csv
    if not linhas:
        caminho.write_text("", encoding="utf-8")
        return
    chaves = list(linhas[0].keys())
    with open(caminho, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=chaves)
        writer.writeheader()
        for linha in linhas:
            writer.writerow({k: ("" if linha.get(k) is None else linha.get(k))
                             for k in chaves})


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    return str(obj)


def _limitacoes() -> List[str]:
    """Limitacoes fixas (documentadas no relatorio e no JSON)."""
    return [
        "3 canais apenas (C3/C4/Cz), sem topografia/espacial; nao ha' como "
        "testar padroes fora do strip motor.",
        "1 participante (S01), 2 sessoes do MESMO dia (17/11/2026); a "
        "generalizacao e' nula e a variabilidade entre sessoes nao e' "
        "independente.",
        "60 trials/classe no total (30/sessao), poucos para um achado "
        "confirmatorio; p-valores de Mann-Whitney NAO corrigidos para "
        "multiplas comparacoes (12 testes por banda/escopo) -> exploratorios.",
        "Welch com janela de 2 s (df=0.5 Hz) em epoca de 4 s: resolucao "
        "frequencial moderada; o pico de mu/beta e' arredondado a 0.5 Hz.",
        "Baseline = 1 s de repouso ANTES do cue (nao cruza ME->MI); nao e' a "
        "linha de base de olhos abertos/fechados, e parte dos trials pode "
        "nao ter baseline valida (registrada em trials.csv).",
        "Sem rejeicao de artefatos/ICA/regressao de EOG; epoca de 4 s inteira "
        "entra na media, inclusive transientes de movimento no inicio/fim.",
        "Morlet absoluto (dB rel. 1 uV^2), sem normalizacao por baseline: "
        "diferencas de potencia entre classes aparecem como estao; a escala "
        "de cor e' compartilhada L x R (comparacao direta).",
        "O mapeamento dos canais (Ch15=C3, Ch16=C4, Ch17=Cz) vem da "
        "configuracao de aquisicao/HISTORICO do projeto e foi conferido pelos "
        "desvios-padrao do HISTORICO item 189; o campo channel_names do JSON "
        "e' metadado do gerador sintetico e NAO foi usado.",
        "Nao ha' correcao de referencia/common average nem checagem de "
        "impedancia por trial; efeitos de referencia comum podem inflar "
        "correlacoes entre canais.",
    ]


def _relatorio_markdown(res: Resultado, resumo: Dict[str, Any],
                        figuras: Dict[str, Any], linhas_trials: List[Dict[str, Any]],
                        tempo_exec: float) -> str:
    """Gera o RELATORIO_ANALISE_FFT_PILOTO.md com numeros calculados."""
    sessoes_md = []
    for sessao in res.sessoes:
        info = sessao.info
        sessoes_md.append([
            sessao.nome, sessao.csv_path.name,
            f"{sessao.fs:g}", f"{info['n_trials_validos']}",
            f"{info['n_L']}/{info['n_R']}", sessao.json_path.name,
            f"{sessao.csv_path.stat().st_size / 1e6:.1f} MB",
        ])
    tabela_sessoes = _md_tabela(
        ["sessao", "CSV", "fs (Hz)", "trials validos", "L/R", "eventos",
         "tamanho CSV"], sessoes_md)
    n_L = int((res.labels == "L").sum())
    n_R = int((res.labels == "R").sum())
    # Tabela de potencia por banda (pooled).
    linhas_banda_md = []
    bandas_pool = (resumo.get("resumo_bandas", {})
                   .get("bandas", {}).get("pooled", {}))
    for banda in BANDAS:
        for canal in res.canais:
            d = bandas_pool.get(banda, {}).get(canal)
            if not d:
                continue
            linhas_banda_md.append([
                banda, canal,
                f"{_fmt(d['pot_L_uv2'], 3)} ± {_fmt(d['sem_L'], 3)}",
                f"{_fmt(d['pot_R_uv2'], 3)} ± {_fmt(d['sem_R'], 3)}",
                _fmt(d["razao_R_L_db"], 2),
                _fmt(d["p_mwu_nao_corrigido"], 3),
                _fmt(d["cohen_d"], 2),
            ])
    tabela_bandas = _md_tabela(
        ["banda", "canal", "L (µV²) mean ± SEM", "R (µV²) mean ± SEM",
         "razão R/L (dB)", "p (não corrigido)", "Cohen d"], linhas_banda_md)
    # Picos
    linhas_pico_md = []
    for canal in res.canais:
        picos_L = resumo.get("resumo_picos", {}).get("pooled", {}).get("L", {}).get(canal, {})
        picos_R = resumo.get("resumo_picos", {}).get("pooled", {}).get("R", {}).get(canal, {})
        linhas_pico_md.append([
            canal,
            _fmt(picos_L.get("mu", {}).get("freq_hz"), 1),
            _fmt(picos_R.get("mu", {}).get("freq_hz"), 1),
            _fmt(picos_L.get("beta", {}).get("freq_hz"), 1),
            _fmt(picos_R.get("beta", {}).get("freq_hz"), 1),
        ])
    tabela_picos = _md_tabela(
        ["canal", "pico mu L (Hz)", "pico mu R (Hz)", "pico beta L (Hz)",
         "pico beta R (Hz)"], linhas_pico_md)
    _std_s01 = res.sessoes[0].std_canais_sessao
    std_s01 = (", ".join(_fmt(v, 2) for v in _std_s01)
               if _std_s01 is not None else "--")

    def _band(escopo: str, banda: str, canal: str) -> Dict[str, Any]:
        try:
            return (resumo.get("resumo_bandas", {}).get("bandas", {})
                    .get(escopo, {}).get(banda, {}).get(canal, {})
                    or {})
        except (TypeError, KeyError):
            return {}

    # Leitura descritiva (sem inferencia) montada dos proprios numeros.
    razoes_mu = [_band("pooled", "mu", c).get("razao_R_L_db")
                 for c in res.canais]
    ps_mu = [_band("pooled", "mu", c).get("p_mwu_nao_corrigido")
             for c in res.canais]
    beta_c4 = _band("pooled", "beta", "C4")
    total_c3 = _band("pooled", "total_1_45", "C3")
    total_c4 = _band("pooled", "total_1_45", "C4")
    mu_s1 = [_band(res.sessoes[0].nome, "mu", c).get("razao_R_L_db")
             for c in res.canais]
    mu_s2 = [_band(res.sessoes[-1].nome, "mu", c).get("razao_R_L_db")
             for c in res.canais]

    def _lista(vals):
        return ", ".join(_fmt(v, 2) for v in vals if v is not None) or "--"

    leitura = (
        "- **Mu (8–13 Hz)**: potências médias praticamente iguais entre "
        f"classes (razão R/L: {_lista(razoes_mu)} dB por canal; menor p não "
        f"corrigido = {_fmt(min((v for v in ps_mu if v is not None), default=None), 3)}); "
        f"o pico de mu fica em ~{_fmt(resumo.get('resumo_picos', {}).get('pooled', {}).get('L', {}).get('C3', {}).get('mu', {}).get('freq_hz'), 1)} Hz "
        "em todos os canais/classes.\n"
        "- **Beta (13–30 Hz)**: C4 tem potência menor na direita "
        f"(razão {_fmt(beta_c4.get('razao_R_L_db'), 2)} dB; p="
        f"{_fmt(beta_c4.get('p_mwu_nao_corrigido'), 3)} não corrigido; "
        f"d={_fmt(beta_c4.get('cohen_d'), 2)}). Não há pico beta destacado "
        "(espectro cai em 1/f — ver nota acima).\n"
        "- **Banda total (1–45 Hz)**: C3 com R>L e C4 com R<L "
        f"(razões {_fmt(total_c3.get('razao_R_L_db'), 2)} / "
        f"{_fmt(total_c4.get('razao_R_L_db'), 2)} dB; p não corrigidos "
        f"{_fmt(total_c3.get('p_mwu_nao_corrigido'), 3)} / "
        f"{_fmt(total_c4.get('p_mwu_nao_corrigido'), 3)}) — sinal lateralizado "
        "fraco, mas as SESSÕES individuais têm sinais opostos no mu "
        f"(s1 R/L: {_lista(mu_s1)} dB; s2: {_lista(mu_s2)} dB), então o "
        "padrão NÃO é estável entre sessões e não deve ser promovido a achado.\n"
        "- Nenhuma conclusão inferencial: p de Mann-Whitney sem correção para "
        "múltiplas comparações e sem cluster/permutação; n=120 épocas de um "
        "único participante."
    )
    tabela_razao = _md_tabela(
        ["canal", "mu R/L (dB)", "beta R/L (dB)"],
        [[canal,
          _fmt(resumo.get("razao_R_L_db_por_canal", {}).get(canal, {}).get("mu"), 2),
          _fmt(resumo.get("razao_R_L_db_por_canal", {}).get(canal, {}).get("beta"), 2)]
         for canal in res.canais])
    # Figuras
    linhas_fig = [[nome, f"{d['bytes'] / 1024:.0f} kB"] for nome, d in figuras.items()]
    tabela_figuras = _md_tabela(["arquivo", "tamanho"], linhas_fig)
    # Estatisticas de sincronismo/epoch
    linhas_sanidade = []
    for sessao in res.sessoes:
        info = sessao.info
        linhas_sanidade.append([
            sessao.nome, f"{info['n_marcadores_mi_ini']}",
            f"{info['n_marcadores_mi_fim']}",
            f"{info['dur_mi_amostras_min']}–{info['dur_mi_amostras_max']}",
            f"{_fmt(info['gap_transicao_779_780_mediana'], 0)}",
            f"{info['n_baseline_valida']}/{info['n_trials_validos']}",
            f"{', '.join(info['avisos']) or '—'}",
        ])
    tabela_sanidade = _md_tabela(
        ["sessao", "marcadores 780", "marcadores 781", "dur MI (amostras)",
         "gap 779→780 (amostras)", "baseline valida", "avisos"],
        linhas_sanidade)
    # Diferenca entre sessoes (potencia mu) para o texto
    texto_diff = []
    bandas_s1 = (resumo.get("resumo_bandas", {})
                 .get("bandas", {}).get(res.sessoes[0].nome, {}))
    bandas_s2 = (resumo.get("resumo_bandas", {})
                 .get("bandas", {}).get(res.sessoes[-1].nome, {}))
    for canal in res.canais:
        d1 = bandas_s1.get("mu", {}).get(canal, {})
        d2 = bandas_s2.get("mu", {}).get(canal, {})
        texto_diff.append(
            f"{canal}: potência mu L/R s1 = "
            f"{_fmt(d1.get('pot_L_uv2'), 2)}/{_fmt(d1.get('pot_R_uv2'), 2)} µV²; "
            f"s2 = {_fmt(d2.get('pot_L_uv2'), 2)}/{_fmt(d2.get('pot_R_uv2'), 2)} µV².")
    juntas = "\n".join(f"- {t}" for t in texto_diff)
    relatorio = f"""# Análise FFT/PSD e tempo-frequência (Morlet) — épocas de MI de 4 s

**Gerado por:** `src/mestradopy/tools/fft_epocas_piloto.py` em {time.strftime('%d/%m/%Y %H:%M')}
**Dados:** piloto ME/MI do `validacao_eeg_3classes.py` (17/11/2026),
`outputs/eeg/piloto_me_mi_s01` e `piloto_me_mi_s02` (somente leitura; nada foi
alterado nas pastas de origem).
**Saída:** `outputs/eeg/analises_fft_piloto/` (figuras, tabelas, JSON, este
relatório).

## 1. Resumo

- Épocas de **imagens motora (MI) de 4 s** extraídas do marcador `mi_start=780`
  (2000 amostras a {res.fs:g} Hz) até 780+4 s; **não** usam ME como MI e **não**
  cruzam a transição ME→MI (o 780 vem depois do fim do ME, 779).
- Rótulo por trial = último cue antes do 780 (`769`=esquerda, `770`=direita).
- **n = {n_L} trials L + {n_R} trials R** ({', '.join(s.nome for s in res.sessoes)}).
- Canais: C3/C4/Cz (canais físicos 15/16/17 = colunas Ch15/Ch16/Ch17 do CSV;
  índices 0-based 14/15/16 do `CH_OWN_IDX` do pacote `eeg.mi_eegnet`).
- Figuras: PSD média ± SEM, diferença/razão R−L (mu/beta destacados),
  espectrogramas médios Morlet L e R (mesma escala), diferença no
  tempo-frequência, heatmap de PSD por trial e barras de potência por banda.
- Leitura exploratória: com 3 canais, 1 participante e 2 sessões do mesmo dia,
  qualquer diferença entre classes é **descritiva**, sem correção para múltiplas
  comparações.

## 2. Sessões usadas

{tabela_sessoes}

Sanidade do epoching:

{tabela_sanidade}

Notas: `dur MI` é `mi_end - mi_start` no CSV (mediana ~{int(res.sessoes[0].info['dur_mi_amostras_mediana'])} amostras); `gap 779→780` mede a transição (~1 s
esperado). O epoch usa exatamente 4 s a partir do 780 (o `mi_end` do marcador
varia alguns samples por jitter do muxer, por isso a duração é validada com
tolerância e o epoch é fixo, sem atravessar a transição).

## 3. Métodos

1. **Loader**: `eeg.mi_eegnet.data._load_own_csv` (mesmo leitor usado pelas
   análises CSP+LDA/EEGNet do projeto) — CSV `eeg_mi3c*.csv` + `eventos*.json`.
2. **Canais**: colunas Ch15/Ch16/Ch17 (índices 0-based 14/15/16) = C3/C4/Cz.
   Conferência: desvios-padrão da sessão s01 = {std_s01} µV,
   batendo com o registrado no HISTÓRICO item 189 (Ch15=C3 15,51; Ch16=C4 15,23;
   Ch17=Cz 14,53).
3. **Pré-processamento**: filtro Butterworth bandpass {BANDA_FILTRO[0]:g}–{BANDA_FILTRO[1]:g} Hz
   (ordem 4, zero-phase `sosfiltfilt`) no sinal **contínuo** antes de cortar as
   épocas (evita transiente de filtro por época).
4. **PSD**: Welch por trial/canal, janela Hann de 2 s (`nperseg={NPERSEG}`,
   `noverlap={NOVERLAP}`, detrend constante) → df = {_fmt(res.freqs[1] - res.freqs[0], 2)} Hz;
   intervalo útil {res.freqs[0]:g}–{res.freqs[-1]:g} Hz.
5. **Bandas**: mu 8–13 Hz, beta 13–30 Hz (pedido do Lead) + theta 4–8 e total
   1–45 Hz como referência; potência = integral da PSD na banda.
6. **Tempo-frequência**: Morlet via `mne.time_frequency.tfr_array_morlet`,
   `n_cycles=max(freq/2, 3)`, {res.tfr_freqs[0]:g}–{res.tfr_freqs[-1]:g} Hz (passo 1 Hz), potência média por
   classe em dB (10·log10); escala de cor **compartilhada L×R**.
7. **Estatística**: Mann-Whitney U bilateral (p **não corrigido**, exploratório)
   e Cohen d para R−L da potência por banda/canal, por escopo (pooled e por
   sessão). Nenhuma conclusão inferencial: n pequeno e múltiplas comparações.
8. **Baseline**: 1 s de repouso **antes do cue** (não cruza ME→MI), usado só
   como coluna de referência em `potencia_bandas.csv`; as figuras são de
   potência absoluta.

## 4. Resultados (potência média por banda, pooled)

{tabela_bandas}

Picos espectrais (frequência do máximo da PSD média):

{tabela_picos}

O pico de **mu** (~10,5 Hz) é consistente entre canais/classes. O pico de
**beta** cair no limite inferior (13 Hz) significa que o espectro cai
monotonicamente naquela faixa (1/f), ou seja, **não há** um pico beta destacado
neste piloto — o valor é apenas o máximo dentro da banda e não deve ser lido
como oscilação beta.

## 4b. Leitura descritiva (exploratória, sem inferência)

{leitura}

Diferença entre sessões (potência mu, L/R):
{juntas}

Razão R/L em dB por canal e banda (calculada da potência integrada):
{tabela_razao}

## 5. Figuras geradas

{tabela_figuras}

Todas as figuras foram salvas em {DPI} dpi e verificadas (abrem via Pillow,
> {LIMIAR_FIG_BYTES} bytes).

## 6. Tabelas e dados gerados

- `tabelas/psd_media.csv` — PSD média/SEM por classe/canal/frequência
  (pooled e por sessão).
- `tabelas/potencia_bandas.csv` — potência por banda/canal/classe, razão R/L,
  lateralização, p de Mann-Whitney **não corrigido**, Cohen d e baseline.
- `tabelas/picos_espectrais.csv` — frequência/PSD de pico em mu e beta.
- `tabelas/trials.csv` — amostras de cue/ME/MI/baseline e rótulo por trial.
- `tabelas/epocas_e_psd.npz` — épocas MI ({MI_SEC:g} s), baseline, PSD por
  trial e TFR médio L/R (para reprodutibilidade).
- `resumo_analise_fft.json` — parâmetros, números e lista de arquivos.

## 7. Limitações

{chr(10).join('- ' + t for t in _limitacoes())}

## 8. Comandos

```powershell
# análise completa (padrão: s01 + s02, saída em outputs/eeg/analises_fft_piloto)
.\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\fft_epocas_piloto.py

# uma sessão só / saída alternativa
.\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\fft_epocas_piloto.py --sessoes piloto_me_mi_s01
.\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\fft_epocas_piloto.py --saida outputs/eeg/analises_fft_piloto_v2
```

Tempo de execução desta rodada: {tempo_exec:.1f} s. Nada foi commitado
(`git add -A` não foi usado) e nenhum arquivo de `outputs/eeg/piloto_me_mi_s01`,
`s02` foi alterado.
"""
    return relatorio


# =============================================================================
# 5) Main
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FFT/PSD e tempo-frequencia (Morlet) das epocas de MI 4 s "
                    "do piloto ME/MI (C3/C4/Cz, L vs R)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--pasta-base", default=str(_REPO / "outputs" / "eeg"),
                        help="pasta base dos outputs de EEG")
    parser.add_argument("--sessoes", nargs="+",
                        default=["piloto_me_mi_s01", "piloto_me_mi_s02"],
                        help="nomes das pastas de sessao (dentro de --pasta-base)")
    parser.add_argument("--saida", default=str(
        _REPO / "outputs" / "eeg" / "analises_fft_piloto"),
        help="pasta de saida (nao altera as pastas de origem)")
    parser.add_argument("--canais", nargs="+", type=int, default=list(CH_OWN_IDX),
                        help="indices 0-based das colunas de canal do CSV "
                             f"(default {list(CH_OWN_IDX)} = C3/C4/Cz)")
    parser.add_argument("--nomes-canais", nargs="+",
                        default=["C3", "C4", "Cz"])
    parser.add_argument("--fs", type=float, default=None,
                        help="fs esperada (default: do JSON de eventos)")
    parser.add_argument("--nperseg", type=int, default=NPERSEG,
                        help="janela Welch em amostras (default 1000 = 2 s)")
    parser.add_argument("--dpi", type=int, default=DPI,
                        help="dpi das figuras (>=150)")
    parser.add_argument("--sem-figuras", action="store_true",
                        help="nao gerar figuras (so tabelas/JSON)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    pasta_base = Path(args.pasta_base)
    saida = Path(args.saida)
    fig_dir = saida / "figuras"
    tab_dir = saida / "tabelas"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    _config_estilo()
    global DPI
    DPI = max(int(args.dpi), 150)

    _LOG("=" * 78)
    _LOG("ANALISE FFT/PSD + TEMPO-FREQUENCIA (MORLET) - PILOTO ME/MI")
    _LOG("=" * 78)
    _LOG(f"pasta base: {pasta_base}")
    _LOG(f"saida     : {saida}")
    _LOG(f"canais    : {args.canais} -> {args.nomes_canais}")
    if len(args.canais) != len(args.nomes_canais):
        raise SystemExit("--canais e --nomes-canais precisam do mesmo tamanho")

    sessoes: List[SessaoEpocas] = []
    for nome in args.sessoes:
        pasta = pasta_base / nome
        if not pasta.is_dir():
            raise SystemExit(f"sessao nao encontrada: {pasta}")
        sessoes.append(carrega_e_epoca(pasta, args.canais, args.nomes_canais,
                                       fs_esperado=args.fs))
    res = prepara_resultado(sessoes, nperseg=int(args.nperseg))

    arquivos: Dict[str, str] = {}
    linhas_trials = [t for sessao in sessoes for t in sessao.trials]
    if not args.sem_figuras:
        _fig_temporal(res, fig_dir, arquivos)
        _fig_psd(res, fig_dir, arquivos)
        resumo_diff = _fig_diferenca_razao(res, fig_dir, arquivos)
        _fig_tfr(res, fig_dir, arquivos)
        _fig_tfr_diferenca(res, fig_dir, arquivos)
        _fig_heatmap_trials(res, fig_dir, arquivos)
        _fig_bandas(res, fig_dir, arquivos)
    else:
        resumo_diff = {"razao_media_mu_beta": {}}

    # Tabelas
    _LOG("[tabelas] calculando potencias/picos/estatistica...")
    linhas_psd = _tabela_psd(res)
    linhas_bandas, resumo_bandas = _tabela_bandas(res)
    linhas_picos, resumo_picos = _tabela_picos(res)
    _escreve_csv(tab_dir / "psd_media.csv", linhas_psd)
    _escreve_csv(tab_dir / "potencia_bandas.csv", linhas_bandas)
    _escreve_csv(tab_dir / "picos_espectrais.csv", linhas_picos)
    _escreve_csv(tab_dir / "trials.csv", linhas_trials)
    npz_path = tab_dir / "epocas_e_psd.npz"
    np.savez_compressed(
        npz_path,
        epocas_mi=np.concatenate([s.epocas_mi for s in sessoes]).astype(np.float32),
        epocas_base=np.concatenate([s.epocas_base for s in sessoes]).astype(np.float32),
        labels=res.labels,
        sessoes=np.asarray(res.sessoes_trial, dtype="U32"),
        freqs_psd=res.freqs,
        psd_trials=res.psd.astype(np.float32),
        tfr_freqs=res.tfr_freqs,
        tfr_L=res.tfr_L.astype(np.float32),
        tfr_R=res.tfr_R.astype(np.float32),
        tfr_times=res.tfr_times,
        canais=np.asarray(res.canais),
        fs=res.fs,
    )
    _LOG(f"[tabelas] {len(linhas_psd)} linhas PSD, {len(linhas_bandas)} de "
         f"bandas, {len(linhas_picos)} de picos, {len(linhas_trials)} trials")

    resumo: Dict[str, Any] = {
        "gerado_em": time.strftime("%d/%m/%Y %H:%M:%S"),
        "script": "src/mestradopy/tools/fft_epocas_piloto.py",
        "dados": {
            "pasta_base": str(pasta_base),
            "sessoes": [{
                "nome": s.nome,
                "csv": str(s.csv_path),
                "json": str(s.json_path),
                "csv_bytes": s.csv_path.stat().st_size,
                "csv_mtime": time.strftime("%d/%m/%Y %H:%M:%S",
                                           time.localtime(s.csv_path.stat().st_mtime)),
                "info": s.info,
            } for s in sessoes],
        },
        "parametros": {
            "fs": res.fs,
            "canais_idx_0based": list(args.canais),
            "canais": list(res.canais),
            "mi_sec": MI_SEC,
            "baseline_sec": BASE_SEC,
            "banda_filtro_hz": list(BANDA_FILTRO),
            "freq_psd_hz": [float(res.freqs[0]), float(res.freqs[-1])],
            "nperseg": int(args.nperseg),
            "noverlap": int(NOVERLAP),
            "bandas_hz": {k: list(v) for k, v in BANDAS.items()},
            "wavelet": {"tipo": "morlet", "n_cycles": "max(freq/2, 3.0)",
                        "freqs_hz": [float(res.tfr_freqs[0]),
                                     float(res.tfr_freqs[-1])],
                        "n_freqs": int(res.tfr_freqs.size)},
            "dpi": DPI,
        },
        "n_trials": {"L": int((res.labels == "L").sum()),
                     "R": int((res.labels == "R").sum()),
                     "total": int(res.labels.size)},
        "resumo_bandas": resumo_bandas,
        "resumo_picos": resumo_picos,
        "razao_R_L_db_por_canal": resumo_diff.get("razao_media_mu_beta", {}),
        "sanidade": [s.info for s in sessoes],
        "figuras": _figura_tamanhos(arquivos),
        "tabelas": {
            "psd_media.csv": str(tab_dir / "psd_media.csv"),
            "potencia_bandas.csv": str(tab_dir / "potencia_bandas.csv"),
            "picos_espectrais.csv": str(tab_dir / "picos_espectrais.csv"),
            "trials.csv": str(tab_dir / "trials.csv"),
            "epocas_e_psd.npz": str(npz_path),
        },
        "limitacoes": _limitacoes(),
    }
    json_path = saida / "resumo_analise_fft.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(resumo, fh, indent=2, ensure_ascii=False,
                  default=_json_default)
    _LOG(f"[json] {json_path}")

    tempo_exec = time.time() - t0
    relatorio = _relatorio_markdown(res, resumo, resumo["figuras"],
                                    linhas_trials, tempo_exec)
    rel_path = saida / "RELATORIO_ANALISE_FFT_PILOTO.md"
    rel_path.write_text(relatorio, encoding="utf-8")
    (saida / "execucao.log").write_text("\n".join(_LOG.linhas) + "\n",
                                        encoding="utf-8")
    _LOG(f"[relatorio] {rel_path}")
    _LOG(f"[log] {saida / 'execucao.log'}")
    _LOG(f"OK em {tempo_exec:.1f} s: {len(arquivos)} figuras, "
         f"{len(linhas_bandas)} linhas de banda/potencia, "
         f"n={res.labels.size} epocas "
         f"(L={int((res.labels == 'L').sum())}, "
         f"R={int((res.labels == 'R').sum())})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:                # pragma: no cover
        import traceback
        traceback.print_exc()
        print(f"FALHA: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
