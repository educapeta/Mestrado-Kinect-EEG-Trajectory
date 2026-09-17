"""Treino do SAND de TRAJETORIA (EEG -> posicao 3D da mao) com as gravacoes
do Programa 1 (eeg_motor_paradigm.py, protocolo final da dissertacao).

Protocolo de cada trial (12 s): repouso 2 s -> cue + ME 4 s -> pausa + video
2 s -> MI 4 s. A janela de EEG comeca no marcador escolhido por --event-code
(778 = inicio da ME/execucao; 780 = inicio da MI/imaginacao) e o ALVO e' a
trajetoria KT_x/y/z no mesmo intervalo, reamostrada para 30 pontos.

KT_* = landmark 9 do MediaPipe Hands (centro da palma) triangulado para 3D,
em metros, RELATIVO a origem (mao apoiada na mesa na calibracao inicial).

Pre-processamento identico ao tempo real (sand_traj_tempo_real.py):
    filtro banda 1-30 Hz (scipy, zero-phase) -> selecao de canais ->
    CAR -> z-score por canal (window por janela ou dataset pelos stats).

Uso:
    python sand_traj_treino.py --data gravacoes/
    python sand_traj_treino.py --data gravacoes/ --event-code 780   # MI
    python sand_traj_treino.py --selftest               # dados sinteticos

O checkpoint (sand_traj_model.pt) e consumido por sand_traj_tempo_real.py.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import scipy.signal
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import sand_trajectory_model as st

TRAJ_AXIS_ORDER = {"x": 0, "y": 1, "z": 2}

try:
    from sklearn.model_selection import KFold
except ImportError:
    KFold = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", default=".",
                        help="Pasta com gravacao_MEMI_*.csv (+ *_eventos.json)")
    parser.add_argument("--model-out", default="sand_traj_model.pt")
    parser.add_argument("--results-out", default="sand_traj_results.csv")
    parser.add_argument("--fs", type=float, default=st.SAMPLING_RATE)
    parser.add_argument("--window-sec", type=float,
                        default=st.DEFAULT_WINDOW_SEC)
    parser.add_argument("--event-code", type=int, default=778,
                        help="Marcador que ancora a janela de EEG/alvo "
                             "(778 = inicio da ME; 780 = inicio da MI)")
    parser.add_argument("--output-seq-len", type=int,
                        default=st.DEFAULT_OUTPUT_SEQ_LEN)
    parser.add_argument("--window-start-sec", type=float, default=0.0,
                        help="Deslocamento do inicio da janela pos-evento (s)")
    parser.add_argument("--channel-count", type=int, default=32,
                        help="Total de canais do dispositivo (selftest/CSV)")
    parser.add_argument("--channels", default=None,
                        help="Lista 1-based de canais do dispositivo usados "
                             "como montagem (padrao: todos os EEG_Ch do CSV)")
    parser.add_argument("--no-car", action="store_true",
                        help="Desativa a referencia media comum")
    parser.add_argument("--zscore", choices=["window", "dataset"],
                        default="window")
    parser.add_argument("--f-lo", type=float, default=st.DEFAULT_F_LO)
    parser.add_argument("--f-hi", type=float, default=st.DEFAULT_F_HI)
    parser.add_argument("--ktt-valid-min", type=float, default=0.5,
                        help="Fracao minima de amostras validas (KTT_valid) "
                             "dentro da janela para aceitar o trial")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.2,
                        help="Fracao de SESSAO retirada para validacao "
                             "(janelas da mesma sessao ficam juntas)")
    parser.add_argument("--selftest", action="store_true",
                        help="Gera dados sinteticos e valida o pipeline "
                             "inteiro sem exigir gravacoes")
    parser.add_argument("--selftest-samples", type=int, default=96)
    return parser.parse_args()


def find_recordings(data_dir):
    """Lista (csv, eventos_json) disponiveis na pasta."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    results = []
    for path in csv_files:
        if path.endswith("_movimento.csv"):
            continue                     # e' o arquivo de movimento, nao a sessao
        events = re.sub(r"\.csv$", "_eventos.json", path)
        if os.path.exists(events):
            results.append((path, events))
    return results


class MotionTrack:
    """Movimento em arquivo proprio a 30 Hz (Programa 1, item #1).

    Alinhamento com o EEG pelo relogio ABSOLUTO: o arquivo traz `t_mono_s`
    (time.monotonic, absoluto) e cada evento do JSON de eventos traz
    (sample_index no EEG, monotonic_s) -- dois eventos definem a reta
    amostra -> tempo. Assim o EEG a 500 Hz e o movimento a 30 Hz ficam no mesmo
    eixo temporal sem supor taxas.
    """

    def __init__(self, path):
        self.path = str(path)
        self.t = None             # t_mono_s (s, absoluto)
        self.kt = None            # (N, 3) m, relativo a origem
        self.src = None           # KT_src (0..8)
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            index = {name: position for position, name in enumerate(header)}
            for obrigatoria in ("t_mono_s", "KT_x_m", "KT_y_m", "KT_z_m"):
                if obrigatoria not in index:
                    raise ValueError(f"{self.path}: falta a coluna "
                                     f"{obrigatoria} no arquivo de movimento.")
            coluna_src = index.get("KT_src")
            tempos, pontos, fontes = [], [], []
            for row in reader:
                tempos.append(_to_float(row[index["t_mono_s"]]))
                pontos.append([_to_float(row[index["KT_x_m"]]),
                               _to_float(row[index["KT_y_m"]]),
                               _to_float(row[index["KT_z_m"]])])
                fontes.append(_to_float(row[coluna_src])
                              if coluna_src is not None else 1.0)
        self.t = np.asarray(tempos, np.float64)
        self.kt = np.asarray(pontos, np.float64)
        self.src = np.asarray(fontes, np.float64)
        ordem = np.argsort(self.t)
        self.t, self.kt, self.src = (self.t[ordem], self.kt[ordem],
                                     self.src[ordem])

    def at(self, times):
        """KT (N,3) e validade por instante (vizinho mais proximo no tempo).

        Validade = KT_src <= 3: apenas medida REAL (triangulacao, depth do
        Kinect ou esqueleto do SDK). Estimativas (modelo/ultima valida/nominal)
        ficam invalidas para nao virar alvo de treino.
        """
        times = np.asarray(times, np.float64)
        if len(self.t) == 0:
            return np.full((len(times), 3), np.nan), np.zeros(len(times))
        posicao = np.clip(np.searchsorted(self.t, times), 1, len(self.t) - 1)
        esquerda = np.abs(times - self.t[posicao - 1])
        direita = np.abs(times - self.t[posicao])
        escolha = np.where(esquerda <= direita, posicao - 1, posicao)
        return self.kt[escolha], (self.src[escolha] <= 3).astype(np.float64)

    @staticmethod
    def sibling_path(eeg_csv):
        """Caminho do CSV de movimento correspondente ao CSV de EEG."""
        raiz, _ = os.path.splitext(str(eeg_csv))
        return raiz + "_movimento.csv"


# =============================================================================
# Carregamento das gravacoes do Programa 1
# =============================================================================
class Recording:
    """Uma gravacao (CSV + eventos): colunas, dados e marcadores na memoria."""

    def __init__(self, csv_path, events_path, fs, window_n, output_seq_len,
                 event_code, ktt_valid_min, f_lo, f_hi, start_offset=0):
        self.path = csv_path
        self.events_path = events_path
        self.fs = float(fs)
        self.window_n = int(window_n)
        self.output_seq_len = int(output_seq_len)
        self.event_code = int(event_code)
        self.ktt_valid_min = float(ktt_valid_min)
        self.start_offset = int(start_offset)
        self.f_lo, self.f_hi = float(f_lo), float(f_hi)
        self.header = None
        self.eeg_cols = []
        self.kt_cols = []
        #: Movimento no formato novo (#1): arquivo separado a 30 Hz.
        self.motion = None
        self._tempo_por_amostra = None
        self.ktt_valid_col = None
        self.samples = []
        self.markers = []          # lista de {sample_index, code} (todos)
        self.epochs = None         # X (N, C, T), Y (N, seq, 3) build em load()

    def _parse_header(self):
        with open(self.path, "r", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            self.header = next(reader)
        self.eeg_cols = [i for i, name in enumerate(self.header)
                         if re.fullmatch(r"EEG_Ch\d+", name)]
        kt_map = {"KT_x_m": "x", "KT_y_m": "y", "KT_z_m": "z"}
        for column, axis in kt_map.items():
            if column in self.header:
                self.kt_cols.append((self.header.index(column), axis))
        self.kt_cols.sort(key=lambda pair: pair[1])
        if "KTT_valid" in self.header:
            self.ktt_valid_col = self.header.index("KTT_valid")
        if not self.eeg_cols:
            raise ValueError(f"{self.path}: sem colunas EEG_Ch (cabecalho "
                             f"invalido ou gravacao de outro formato).")
        if len(self.kt_cols) != 3:
            # Formato NOVO (item #1): o movimento vive em
            # <prefixo>_movimento.csv, a 30 Hz, e o alinhamento com o EEG e'
            # feito pelo relogio absoluto (ancoras do JSON de eventos).
            caminho = MotionTrack.sibling_path(self.path)
            if not os.path.exists(caminho):
                raise ValueError(
                    f"{self.path}: preciso de KT_x_m/KT_y_m/KT_z_m no CSV ou "
                    f"do arquivo de movimento {os.path.basename(caminho)}.")
            self.motion = MotionTrack(caminho)
        return self

    def _sample_to_time(self):
        """Reta (a, b) de `t_mono = a * sample_index + b` (ancoras do JSON).

        Cada evento do JSON traz a amostra exata no EEG e o instante
        (`monotonic_s`) do mesmo evento; dois eventos definem a reta, o que
        amarra 500 Hz (EEG) e 30 Hz (movimento) sem supor taxas.
        """
        if self._tempo_por_amostra is not None:
            return self._tempo_por_amostra
        pares = []
        with open(self.events_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for evento in payload.get("eventos", []):
            amostra = evento.get("sample_index")
            instante = evento.get("monotonic_s")
            if amostra is not None and instante is not None:
                pares.append((float(amostra), float(instante)))
        if len(pares) >= 2:
            amostras = np.asarray([par[0] for par in pares], np.float64)
            tempos = np.asarray([par[1] for par in pares], np.float64)
            a, b = np.polyfit(amostras, tempos, 1)
            self._tempo_por_amostra = (float(a), float(b))
        elif len(pares) == 1:
            a = 1.0 / self.fs
            self._tempo_por_amostra = (a, pares[0][1] - a * pares[0][0])
        else:
            print(f"[aviso] {os.path.basename(self.events_path)}: sem ancoras "
                  f"de relogio; supondo fs={self.fs:g} Hz a partir de 0",
                  flush=True)
            self._tempo_por_amostra = (1.0 / self.fs, 0.0)
        return self._tempo_por_amostra

    def list_event_samples(self):
        """Indices (linha do CSV) dos marcadores pedidos, do JSON de eventos."""
        with open(self.events_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [int(event["sample_index"]) for event in payload["eventos"]
                if int(event["code"]) == self.event_code]

    def recording_origin(self):
        """Origem absoluta (m) desta gravacao (eventos.json) ou None."""
        try:
            with open(self.events_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            origin = payload.get("origem_xyz_m")
            if isinstance(origin, (list, tuple)) and len(origin) == 3:
                origin = [float(value) for value in origin]
                if np.isfinite(origin).all():
                    return np.asarray(origin, np.float64)
        except (OSError, ValueError, TypeError):
            pass
        return None

    def load_rows(self, start, n):
        """Le n linhas a partir de start (sem o cabecalho)."""
        eeg = np.zeros((n, len(self.eeg_cols)), np.float64)
        kt = np.zeros((n, 3), np.float64)
        valid = np.zeros((n,), np.float64)
        kt_positions = {axis: col for col, axis in self.kt_cols}
        lidas = 0
        with open(self.path, "r", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            next(reader)                       # pula o cabecalho
            for row_index, row in enumerate(reader):
                if row_index < start:
                    continue
                if row_index >= start + n:
                    break
                local = row_index - start
                channel = 0
                for col_index in self.eeg_cols:
                    eeg[local, channel] = _to_float(row[col_index])
                    channel += 1
                for axis, col_index in kt_positions.items():
                    kt[local, TRAJ_AXIS_ORDER[axis]] = _to_float(row[col_index])
                if self.ktt_valid_col is not None:
                    valid[local] = _to_float(row[self.ktt_valid_col])
                lidas = local + 1
        if self.motion is not None:
            # Movimento no arquivo proprio: converte a linha do EEG em instante
            # absoluto (retas das ancoras) e busca o KT mais proximo no tempo.
            a, b = self._sample_to_time()
            linha = np.arange(start, start + n, dtype=np.float64)
            kt, valid = self.motion.at(a * linha + b)
            if lidas < n:                       # fim do arquivo: resto invalido
                kt[lidas:] = np.nan
                valid[lidas:] = 0.0
        return eeg, kt, valid

    def build_epochs(self, start_samples):
        """Monta (X (N,C,T) filtrado, Y (N,seq,3)) para cada inicio pedido."""
        sos = scipy.signal.butter(4, [self.f_lo, self.f_hi], btype="bandpass",
                                  fs=self.fs, output="sos")
        X, Y = [], []
        for start in start_samples:
            eeg, kt, valid = self.load_rows(start, self.window_n)
            if valid.size and valid.mean() < self.ktt_valid_min:
                continue                                # Kinect perdeu a mao
            if not np.isfinite(eeg).all() or not np.isfinite(kt).all():
                continue
            filtered = np.asarray([scipy.signal.sosfiltfilt(sos, eeg[:, ch])
                                   for ch in range(eeg.shape[1])]).T
            target = st.trajectory_resample(kt, self.output_seq_len)
            X.append(filtered)
            Y.append(target)
        if not X:
            return np.zeros((0, len(self.eeg_cols), self.window_n), np.float32), \
                   np.zeros((0, self.output_seq_len, 3), np.float32)
        return (np.asarray(X, np.float32), np.asarray(Y, np.float32))


def _to_float(text):
    try:
        value = float(text)
        return 0.0 if np.isnan(value) else value
    except (TypeError, ValueError):
        return 0.0


# =============================================================================
# Dataset + treino
# =============================================================================
class TrajDataset(Dataset):
    """EEG (C, T) e alvo (seq, 3) ja pre-processados."""

    def __init__(self, x, y):
        self.x = torch.from_numpy(np.asarray(x, np.float32))
        self.y = torch.from_numpy(np.asarray(y, np.float32))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, seen = 0.0, 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * x_batch.shape[0]
        seen += x_batch.shape[0]
    return total / max(seen, 1)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total, seen = 0.0, 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            total += float(criterion(pred, y_batch).item()) * x_batch.shape[0]
            seen += x_batch.shape[0]
            all_pred.append(pred.float().cpu().numpy())
            all_true.append(y_batch.float().cpu().numpy())
    if not all_pred:
        return float("inf"), [0.0, 0.0, 0.0]
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    pcc = []
    for axis in range(pred.shape[2]):
        p = pred[:, :, axis].reshape(-1)
        t = true[:, :, axis].reshape(-1)
        if np.std(p) < 1e-9 or np.std(t) < 1e-9:
            pcc.append(0.0)
        else:
            pcc.append(float(np.corrcoef(p, t)[0, 1]))
    return total / max(seen, 1), pcc


def load_all_epochs(recordings, args):
    """Reune todos os trials das gravacoes em X (N,C,T), Y (N,seq,3) e o
    indice da gravacao de origem de cada trial (sessions, para hold-out).

    Devolve tambem as origens absolutas por sessao (origins, (S,3) com NaN
    quando a gravacao nao registrou origem).
    """
    X_all, Y_all, sessions, origins = [], [], [], []
    for session, (csv_path, events_path) in enumerate(recordings):
        record = Recording(csv_path, events_path, args.fs,
                           int(args.window_sec * args.fs),
                           args.output_seq_len, args.event_code,
                           args.ktt_valid_min, args.f_lo, args.f_hi,
                           start_offset=int(args.window_start_sec * args.fs))
        record._parse_header()
        starts = [sample + record.start_offset
                  for sample in record.list_event_samples()]
        x_rec, y_rec = record.build_epochs(starts)
        origin = record.recording_origin()
        origins.append(np.nan if origin is None else origin)
        if x_rec.shape[0]:
            print(f"{os.path.basename(csv_path)}: {x_rec.shape[0]} trials "
                  f"({len(starts)} eventos -> {x_rec.shape[0]} validos)")
            X_all.append(x_rec)
            Y_all.append(y_rec)
            sessions.append(np.full(x_rec.shape[0], session, dtype=int))
    if not X_all:
        raise RuntimeError(
            "Nenhum trial valido encontrado. Verifique --data (precisa dos "
            "_eventos.json) e o filtro KTT_valid.")
    return (np.concatenate(X_all), np.concatenate(Y_all),
            np.concatenate(sessions),
            np.asarray([origin if isinstance(origin, np.ndarray)
                        else np.full(3, np.nan) for origin in origins],
                       np.float64))


def synthetic_dataset(args, rng):
    """Dados sinteticos identicos em FORMATO aos de verdade (selftest).

    EEG: soma de senos por canal com fase modulada pela trajetoria alvo;
    trajetoria: suave em 3D. O selftest valida o PIPELINE inteiro (geracao de
    janelas, pre-processamento, treino, normalizacao e checkpoint), nao a
    precisao absoluta.
    """
    n = int(args.selftest_samples)
    window_n = int(args.window_sec * args.fs)
    channels = int(args.channel_count)
    seq = int(args.output_seq_len)
    time_axis = np.linspace(0.0, 1.0, window_n)
    X, Y = [], []
    for _sample in range(n):
        eeg = np.zeros((channels, window_n), np.float64)
        freqs = rng.uniform(2.0, 12.0, channels)
        for ch in range(channels):
            eeg[ch] = (np.sin(2 * np.pi * freqs[ch] * time_axis
                              + rng.uniform(0, 2 * np.pi))
                       + 0.4 * rng.standard_normal(window_n))
        t2 = np.linspace(0, 2 * np.pi, seq)
        target = np.column_stack([
            np.sin(t2 + rng.uniform(0, np.pi)) * rng.uniform(0.2, 0.6),
            np.cos(t2 * 0.8 + rng.uniform(0, np.pi)) * rng.uniform(0.2, 0.6),
            np.sin(t2 * 0.6 + rng.uniform(0, np.pi)) * rng.uniform(0.1, 0.4),
        ]).astype(np.float32)
        X.append(eeg)
        Y.append(target)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def preprocess_epochs(x_selected, args, mean=None, std=None):
    """CAR + z-score (identico ao preprocess_window do tempo real)."""
    x = np.asarray(x_selected, np.float64)
    if not args.no_car:
        x -= x.mean(axis=1, keepdims=True)
    if args.zscore == "window":
        axis_std = x.std(axis=2, keepdims=True)
        x = (x - x.mean(axis=2, keepdims=True)) / (axis_std + 1e-8)
    else:
        if mean is None or std is None:
            raise ValueError("zscore=dataset exige stats do treino.")
        x = (x - mean) / (std + 1e-8)
    return x.astype(np.float32)


def run_training(args, x_train, y_train, x_val, y_val, channel_map,
                 meta=None):
    """Treina e devolve o melhor modelo (por perda de validacao)."""
    if args.zscore == "dataset":
        eeg_mean = x_train.mean(axis=(0, 2))
        eeg_std = x_train.std(axis=(0, 2))
    else:
        eeg_mean = eeg_std = None
    x_train_pp = preprocess_epochs(x_train, args, eeg_mean, eeg_std)
    x_val_pp = preprocess_epochs(x_val, args, eeg_mean, eeg_std)

    normalizer = st.TrajectoryNormalizer()
    normalizer.fit(y_train)
    y_train_n = normalizer.to_normalized(y_train).astype(np.float32)
    y_val_n = normalizer.to_normalized(y_val).astype(np.float32)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_n = int(args.window_sec * args.fs)
    model = st.SANDTrajectory(
        n_channels=len(channel_map), n_timesteps=window_n,
        d_model=args.d_model, n_layers=args.n_layers,
        output_seq_len=args.output_seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(TrajDataset(x_train_pp, y_train_n),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TrajDataset(x_val_pp, y_val_n),
                            batch_size=args.batch_size, shuffle=False)

    parameters = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {len(channel_map)} canais x {window_n} amostras "
          f"({args.window_sec:g} s) | parametros: {parameters:,}"
          f" | device: {device}", flush=True)

    best_loss, best_pcc, best_state = float("inf"), None, None
    patience, n_without_improvement = 12, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 device)
        val_loss, pcc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_pcc = val_loss, pcc
            best_state = {key: value.detach().cpu()
                          for key, value in model.state_dict().items()}
            n_without_improvement = 0
        else:
            n_without_improvement += 1
        print(f"Epoch {epoch:02d}: train={train_loss:.4f} "
              f"val={val_loss:.4f} pcc(x,y,z)="
              f"{pcc[0]:.3f}/{pcc[1]:.3f}/{pcc[2]:.3f}", flush=True)
        if n_without_improvement >= patience:
            print(f"Early stopping na epoca {epoch}.", flush=True)
            break

    model.load_state_dict(best_state)
    config = {
        "arch": "SANDTrajectory",
        "fs": args.fs,
        "n_timesteps": window_n,
        "output_seq_len": args.output_seq_len,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "car": not args.no_car,
        "zscore": args.zscore,
        "f_lo": args.f_lo,
        "f_hi": args.f_hi,
        "channel_map": [int(ch) + 1 for ch in channel_map],
        "window_start_sec": float(args.window_start_sec),
        "window_sec": args.window_sec,
        "event_code": args.event_code,
        "meta_origin_m": ((meta or {}).get("origin_xyz_m")
                          if meta else None),
    }
    st.save_checkpoint(args.model_out, model, config, normalizer,
                       eeg_mean, eeg_std, meta=meta)
    print(f"[treino] checkpoint salvo em {args.model_out}", flush=True)
    print(f"[treino] melhor val_loss={best_loss:.4f} | PCC "
          f"x={best_pcc[0]:.3f} y={best_pcc[1]:.3f} z={best_pcc[2]:.3f}",
          flush=True)
    return {"best_val_loss": best_loss,
            "pcc_x": best_pcc[0], "pcc_y": best_pcc[1], "pcc_z": best_pcc[2],
            "epochs": epoch}


def build_channel_map(args, n_channels):
    """Montagem (indices 0-based) usada no treino e gravada no checkpoint."""
    if args.channels:
        montage = [int(ch) - 1 for ch in
                   re.split(r"[,\s]+", str(args.channels).strip())]
        if any(index < 0 for index in montage):
            raise ValueError("--channels e 1-based (>= 1).")
    else:
        montage = list(range(n_channels))
    if any(index >= n_channels for index in montage):
        raise ValueError(f"--channels fora de (1, {n_channels}) do "
                         "dispositivo/CSV.")
    return np.asarray(montage, dtype=int)


def main():
    args = parse_args()

    if args.selftest:
        rng = np.random.default_rng(args.seed)
        x_all, y_all = synthetic_dataset(args, rng)
        channel_map = build_channel_map(args, int(args.channel_count))
        x_all = x_all[:, channel_map, :]
        print(f"[selftest] {x_all.shape[0]} janelas sinteticas "
              f"({channel_map.size} canais selecionados)")
        order = rng.permutation(x_all.shape[0])
        n_test = max(1, int(x_all.shape[0] * args.test_frac))
        train_idx, val_idx = order[:-n_test], order[-n_test:]
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_val, y_val = x_all[val_idx], y_all[val_idx]
        print(f"Split sintetico: treino={len(train_idx)} val={len(val_idx)}")
        meta = {"selftest": True}
    else:
        recordings = find_recordings(args.data)
        if not recordings:
            raise RuntimeError(
                f"Nenhum par (csv + _eventos.json) em {args.data!r}. Use "
                "--selftest para validar o pipeline sem gravacoes.")
        x_all, y_all, sessions, origins = load_all_epochs(recordings, args)
        channel_map = build_channel_map(args, int(x_all.shape[1]))
        x_all = x_all[:, channel_map, :]
        # origem media das sessoes (usada pelo tempo real para projetar sobre
        # o overlay na MESMA referencia em que o KT_* foi gravado)
        finite_origins = origins[np.isfinite(origins).all(axis=1)]
        mean_origin = (finite_origins.mean(axis=0).tolist()
                       if finite_origins.size else None)
        if mean_origin:
            print(f"[treino] origem media das sessoes: "
                  f"{np.round(mean_origin, 3)} m (gravada no checkpoint)")
        # hold-out por SESSAO: escolhe `test_frac` das gravacoes para validar
        n_val = max(1, int(len(recordings) * args.test_frac))
        val_sessions = set(np.arange(len(recordings))[-n_val:])
        val_mask = np.isin(sessions, list(val_sessions))
        train_idx = np.flatnonzero(~val_mask)
        val_idx = np.flatnonzero(val_mask)
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        x_val, y_val = x_all[val_idx], y_all[val_idx]
        names = [os.path.basename(path) for path, _ev in recordings]
        print(f"Validacao: sessoes {sorted(val_sessions)} "
              f"({[names[i] for i in sorted(val_sessions)]}) | "
              f"treino={len(train_idx)} trials, val={len(val_idx)}")
        meta = {"selftest": False, "data_dir": args.data,
                "sessoes": len(recordings), "origin_xyz_m": mean_origin}

    results = run_training(args, x_train, y_train, x_val, y_val, channel_map,
                           meta=meta)
    frame = pd.DataFrame([results])
    frame.to_csv(args.results_out, index=False)
    print(f"Resultados salvos em {args.results_out}")
    print(f"Para rodar em tempo real:\n"
          f"  python sand_traj_tempo_real.py --model {args.model_out}")


if __name__ == "__main__":
    main()