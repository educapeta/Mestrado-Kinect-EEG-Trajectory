"""Modelo SAND de TRAJETORIA: EEG -> posicao 3D da mao prevista.

Arquitetura identica a SAND.py / SAND_LOSO.py (extrator convolucional espacial
+ temporal + blocos FFTTransformerBlock + TrajectoryProjector), adaptada para
inferencia continua com janela deslizante:

    filtro banda 1-30 Hz (gp.Bandpass no tempo real / scipy no treino)
    -> selecao de canais (montagem) -> CAR -> z-score por canal
    -> rede SAND (trajetoria regressiva: C x T -> seq x 3)

Alvo: trajetoria 3D do punho (KT_x/y/z da gravacao do Programa 1), em metros,
RELATIVA a origem (camera space do Kinect). A saida da rede e normalizada por
eixo (z-score com stats do treino) para caber na rede, e o wrapper de
inferencia desfaz a normalizacao -> a trajetoria prevista volta para METROS e
pode ser projetada no overlay do Kinect (CameraAlignment.project_camera_points).

Este modulo e compartilhado entre o treino (sand_traj_treino.py) e o tempo
real (sand_traj_tempo_real.py) para garantir pre-processamento identico.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

#: Taxa de amostragem padrao do g.Nautilus / gravacoes do Programa 1 (Hz).
SAMPLING_RATE = 500.0

#: Janela de EEG usada pela rede (padrao: 2 s).
DEFAULT_WINDOW_SEC = 2.0

#: Numero de pontos da trajetoria prevista por janela.
DEFAULT_OUTPUT_SEQ_LEN = 30

#: Banda do filtro (igual ao gp.Bandpass do pipeline em tempo real).
DEFAULT_F_LO = 1.0
DEFAULT_F_HI = 30.0

#: Nomes dos eixos/saidas da trajetoria (na ordem das colunas KT_*).
TRAJ_AXES = ["x", "y", "z"]

TRAJ_AXIS_ORDER = {"x": 0, "y": 1, "z": 2}


def trajectory_resample(trajectory, output_len):
    """Reamostra uma trajetoria (T, 3) para (output_len, 3) por interpolacao.

    O alvo do treino e a trajetoria do punho (KT_*) durante a janela de EEG;
    a rede sempre preve o MESMO numero de pontos (output_seq_len), entao o
    alvo e reamostrado linearmente para o grid temporal padrao.
    """
    trajectory = np.asarray(trajectory, np.float64)
    if trajectory.shape[0] == 0:
        raise ValueError("Trajetoria vazia (Janela sem dados de Kinect).")
    if trajectory.shape[0] == output_len:
        return trajectory.astype(np.float32)
    indices = np.linspace(0, trajectory.shape[0] - 1, output_len)
    reference = np.arange(trajectory.shape[0])
    return np.column_stack([
        np.interp(indices, reference, trajectory[:, axis])
        for axis in range(trajectory.shape[1])
    ]).astype(np.float32)


# =============================================================================
# Pre-processamento (identico ao de sand_bci_model.py)
# =============================================================================
def preprocess_window(block_tc, channel_map, use_car=True, mean=None,
                      std=None, zscore="window"):
    """EEG (T, C_dispositivo) filtrado -> entrada da rede (C_model, T).

    - channel_map: indices 0-based dos canais do dispositivo que formam a
      montagem de treino (ordem identica ao treino);
    - CAR: subtrai a media dos canais selecionados, amostra a amostra;
    - zscore: "window" padroniza cada janela (robusto cross-sessao) ou
      "dataset" usa mean/std do treino.
    """
    channel_map = np.asarray(channel_map, dtype=int)
    if np.any(channel_map < 0) or np.any(channel_map >= block_tc.shape[1]):
        raise ValueError(f"channel_map {channel_map.tolist()} fora de "
                         f"(0, {block_tc.shape[1] - 1}) do dispositivo.")
    x = np.asarray(block_tc[:, channel_map], np.float64)
    if use_car:
        x -= x.mean(axis=1, keepdims=True)
    if zscore == "window":
        axis_std = x.std(axis=0)
        x = (x - x.mean(axis=0)) / (axis_std + 1e-8)
    else:
        if mean is None or std is None:
            raise ValueError("zscore=dataset exige mean/std do checkpoint.")
        x = (x - np.asarray(mean, np.float64)) / \
            (np.asarray(std, np.float64) + 1e-8)
    return x.T  # (C_model, T)


# =============================================================================
# Normalizacao do alvo (trajetoria em METROS)
# =============================================================================
class TrajectoryNormalizer:
    """z-score por eixo (x/y/z) da trajetoria, com stats do treino.

    A rede preve em espaco normalizado; o wrapper desfaz com
    from_normalized -> metros relativos a origem (mesma grandeza do KT_*).
    """

    def __init__(self, mean=None, std=None):
        self.mean = np.zeros(3, np.float64) if mean is None \
            else np.asarray(mean, np.float64).reshape(3)
        self.std = np.ones(3, np.float64) if std is None \
            else np.asarray(std, np.float64).reshape(3)

    def fit(self, sequences):
        """sequences: (N, seq_len, 3) -> calcula media/std por eixo."""
        stacked = np.asarray(sequences, np.float64).reshape(-1, 3)
        if stacked.shape[0] < 3:
            raise ValueError("Normalizador exige pelo menos 3 pontos.")
        self.mean = stacked.mean(axis=0)
        self.std = stacked.std(axis=0) + 1e-8

    def to_normalized(self, trajectory):
        return (np.asarray(trajectory, np.float64) - self.mean) / self.std

    def from_normalized(self, normalized):
        return (np.asarray(normalized, np.float64) * self.std) + self.mean

    def state(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


# =============================================================================
# Checkpoint
# =============================================================================
def save_checkpoint(path, model, config, traj_normalizer, eeg_mean=None,
                    eeg_std=None, meta=None):
    """Salva modelo + config + normalizacoes num unico .pt (portatil)."""
    eeg_mean = np.zeros(0, np.float64) if eeg_mean is None \
        else np.asarray(eeg_mean, np.float64)
    eeg_std = np.zeros(0, np.float64) if eeg_std is None \
        else np.asarray(eeg_std, np.float64)
    payload = {
        "arch": "SANDTrajectory",
        "config": dict(config or {}),
        "model_state": model.state_dict(),
        "normalization": {
            "trajectory": traj_normalizer.state(),
            "eeg_mean": eeg_mean,
            "eeg_std": eeg_std,
        },
        "meta": dict(meta or {}),
    }
    torch.save(payload, Path(path))


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# =============================================================================
# Wrapper de inferencia
# =============================================================================
class SandTrajectoryBCI:
    """Modelo pronto para inferencia continua: EEG (2 s) -> trajetoria (m)."""

    def __init__(self, checkpoint, device=None):
        ckpt = load_checkpoint(checkpoint) if isinstance(
            checkpoint, (str, os.PathLike)) else checkpoint
        if ckpt.get("arch") != "SANDTrajectory":
            raise ValueError("Checkpoint nao e do SANDTrajectory "
                             f"(arch={ckpt.get('arch')!r}).")
        config = ckpt["config"]
        self.cfg = config
        if "meta_origin_m" not in config:
            meta_origin = ckpt.get("meta", {}).get("origin_xyz_m")
            if meta_origin:
                config["meta_origin_m"] = meta_origin
        self.fs = float(config.get("fs", SAMPLING_RATE))
        self.n_timesteps = int(config["n_timesteps"])
        self.output_seq_len = int(config.get("output_seq_len",
                                             DEFAULT_OUTPUT_SEQ_LEN))
        self.channel_map = np.asarray(config["channel_map"], dtype=int) - 1
        self.use_car = bool(config.get("car", True))
        self.zscore = str(config.get("zscore", "window"))
        normalization = ckpt["normalization"]
        self.traj_normalizer = TrajectoryNormalizer(
            normalization["trajectory"]["mean"],
            normalization["trajectory"]["std"])
        eeg_mean = np.asarray(normalization.get("eeg_mean", []), np.float64)
        eeg_std = np.asarray(normalization.get("eeg_std", []), np.float64)
        self.eeg_mean = eeg_mean if eeg_mean.size else None
        self.eeg_std = eeg_std if eeg_std.size else None
        self.model = SANDTrajectory(
            n_channels=len(self.channel_map),
            n_timesteps=self.n_timesteps,
            d_model=int(config.get("d_model", 32)),
            n_layers=int(config.get("n_layers", 4)),
            output_seq_len=self.output_seq_len,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    def set_channel_map(self, mapping_1based):
        mapping = np.asarray(mapping_1based, dtype=int) - 1
        if mapping.ndim != 1 or len(mapping) != len(self.channel_map):
            raise ValueError(f"O mapa deve ter {len(self.channel_map)} "
                             "entradas (1-based).")
        if mapping.min() < 0:
            raise ValueError("Os canais do dispositivo sao 1-based (>= 1).")
        self.channel_map = mapping

    def predict_window(self, block_tc):
        """Janela EEG (T, C_dispositivo) FILTRADA -> trajetoria (seq, 3) em m.

        Devolve (trajeto_previsto, movimentacao): trajeto em metros
        (relativo a origem do programa 1), e o desvio padrao espacial entre
        os pontos (heuristica de "quanto movimento foi previsto"; ~0 = parado).
        """
        if block_tc.shape[0] < self.n_timesteps:
            raise ValueError(f"Janela com {block_tc.shape[0]} amostras; o "
                             f"modelo espera {self.n_timesteps}.")
        x = preprocess_window(block_tc, self.channel_map, self.use_car,
                              self.eeg_mean, self.eeg_std, zscore=self.zscore)
        tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(
            self.device)
        with torch.no_grad():
            normalized = self.model(tensor)[0].float().cpu().numpy()
        trajectory = self.traj_normalizer.from_normalized(normalized)
        trajectory = np.nan_to_num(trajectory, nan=0.0)
        movement = float(np.std(trajectory, axis=0).mean())
        return trajectory, movement

    def describe(self):
        f_lo = float(self.cfg.get("f_lo", DEFAULT_F_LO))
        f_hi = float(self.cfg.get("f_hi", DEFAULT_F_HI))
        sec = self.n_timesteps / self.fs
        return (f"SANDTrajetoria: {len(self.channel_map)} canais x "
                f"{self.n_timesteps} amostras ({sec:.2f} s) @ "
                f"{self.fs:.0f} Hz | banda {f_lo:.0f}-{f_hi:.0f} Hz | "
                f"CAR={self.use_car} | saida {self.output_seq_len} pontos 3D "
                "(m)")


class FFTAttention(nn.Module):
    """Atencao no dominio da frequencia (sob a forma irFFT(q*conj(k)))."""

    def __init__(self, d_model, d_k=None):
        super().__init__()
        d_k = d_k or d_model
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        self.layer_norm = nn.LayerNorm(d_k)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        seq_len = x.shape[1]
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)
        # cuFFT em half requer comprimentos potencias de 2: FFT em FP32 sempre.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            q = torch.fft.rfft(query.float(), dim=1)
            k = torch.fft.rfft(key.float(), dim=1)
            attention = torch.fft.irfft(q * torch.conj(k), n=seq_len, dim=1)
        return x + self.dropout(self.layer_norm(attention.to(value.dtype)) * value)


class FFTTransformerBlock(nn.Module):
    """FFTAttention + FFN com LayerNorm residual (mesmo de SAND.py)."""

    def __init__(self, d_model, d_ff=256, dropout=0.1):
        super().__init__()
        self.attn = FFTAttention(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class SpectralFeatureExtractor(nn.Module):
    """(1, C) atraves dos canais e (32, 1) no tempo -> tokens d_model."""

    def __init__(self, n_channels, n_timesteps, d_model=32):
        del n_timesteps
        super().__init__()
        self.conv_spatial = nn.Conv2d(1, 8, kernel_size=(1, n_channels))
        self.conv_temporal = nn.Conv2d(8, 16, kernel_size=(32, 1))
        self.bn = nn.BatchNorm2d(16)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.proj = nn.Linear(16, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (B, C, T) -> (B, 1, T, C)
        x = x.unsqueeze(1).permute(0, 1, 3, 2)
        x = self.conv_spatial(x)
        x = self.conv_temporal(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        x = x.squeeze(3).permute(0, 2, 1)
        return self.dropout(self.proj(x))


class TrajectoryProjector(nn.Module):
    """Tokens d_model -> sequencia de saida (output_seq_len x n_outputs)."""

    def __init__(self, d_model=32, n_outputs=3, output_seq_len=30):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, 16, kernel_size=16)
        self.conv2 = nn.Conv1d(16, n_outputs, kernel_size=16)
        self.bn = nn.BatchNorm1d(16)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_seq_len)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn(x) if x.shape[0] > 1 else x
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        return x.permute(0, 2, 1)


class SANDTrajectory(nn.Module):
    """SAND de regressao de trajetoria (C x T -> seq x 3)."""

    def __init__(self, n_channels=32, n_timesteps=1000, d_model=32,
                 n_layers=4, output_seq_len=30, dropout=0.1):
        super().__init__()
        if n_timesteps < 64:
            raise ValueError(f"n_timesteps {n_timesteps} < 64 (extrator precisa"
                             " de comprimento util).")
        self.feature_extractor = SpectralFeatureExtractor(
            n_channels, n_timesteps, d_model)
        self.transformer_blocks = nn.ModuleList([
            FFTTransformerBlock(d_model, d_ff=d_model * 4, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.projector = TrajectoryProjector(
            d_model, len(TRAJ_AXES), output_seq_len)
        self._init_weights()

    def _init_weights(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, x):
        x = self.feature_extractor(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.projector(x)


if __name__ == "__main__":
    # Teste rapido da arquitetura: python sand_trajectory_model.py
    net = SANDTrajectory(n_channels=32, n_timesteps=1000, output_seq_len=30)
    out = net(torch.randn(2, 32, 1000))
    parameters = sum(p.numel() for p in net.parameters())
    print(f"Saida: {tuple(out.shape)} | parametros: {parameters:,}")