"""Testes do modulo sand_trajectory_model (rede, pre-processamento, alvo,
normalizacao e checkpoint -> wrapper de inferencia).

Nao usa hardware. Exercita os contratos principais do Programa 2:
1. Formato da rede: (C, T) -> (seq, 3).
2. preprocess_window: CAR + z-score por janela e validacao do channel_map.
3. trajectory_resample: reamostragem correta de trajetoria (T,3) -> (seq,3).
4. TrajectoryNormalizer: round-trip to/from (normalizado <-> metros).
5. Checkpoint: save/load e SandTrajectoryBCI.devolve (seq,3) em METROS.
6. Validacoes de contrato (arch errado, janela curta).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

import sand_trajectory_model as st

TMP = Path(__file__).with_name("_tmp_sand_traj.pt")


# ---------------------------------------------------------------------------
# 1) Rede: shape e parametros
# ---------------------------------------------------------------------------
net = st.SANDTrajectory(n_channels=32, n_timesteps=1000, output_seq_len=30)
out = net(torch.randn(2, 32, 1000))
assert tuple(out.shape) == (2, 30, 3), out.shape
parameters = sum(p.numel() for p in net.parameters())
assert 40_000 < parameters < 90_000, parameters     # na casa do artigo (59k)
# janela curta demais e rejeitada
try:
    st.SANDTrajectory(n_channels=8, n_timesteps=32)
    raise AssertionError("rede deveria rejeitar n_timesteps<64")
except ValueError:
    pass

# ---------------------------------------------------------------------------
# 2) Pre-processamento
# ---------------------------------------------------------------------------
channel_map = np.array([0, 5, 9, 17, 31])           # 5 canais (0-based)
block = np.random.randn(1000, 32).astype(np.float32) * 50.0
x = st.preprocess_window(block, channel_map, use_car=True, zscore="window")
assert x.shape == (5, 1000), x.shape
assert abs(x.mean()) < 1e-6, x.mean()              # CAR + zscore por janela
assert abs(x.std() - 1.0) < 0.05, x.std()
# sem CAR o sinal segue padronizado por canal (media ~0 por linha)
x_nocar = st.preprocess_window(block, channel_map, use_car=False,
                               zscore="window")
assert abs(x_nocar.mean()) < 1e-6
# channel_map invalido -> ValueError
try:
    st.preprocess_window(block, np.array([0, 99]), zscore="window")
    raise AssertionError("mapa fora do dispositivo deveria falhar")
except ValueError:
    pass
# zscore=dataset exige stats
try:
    st.preprocess_window(block, channel_map, zscore="dataset")
    raise AssertionError("dataset sem mean/std deveria falhar")
except ValueError:
    pass
x_ds = st.preprocess_window(block, channel_map, zscore="dataset",
                            mean=np.zeros(5), std=np.ones(5) * 50)
assert x_ds.shape == (5, 1000)

# ---------------------------------------------------------------------------
# 3) Reamostragem da trajetoria alvo
# ---------------------------------------------------------------------------
traj = np.arange(1000, dtype=np.float64)[:, None].repeat(3, axis=1)  # (i,i,i)
resampled = st.trajectory_resample(traj, 30)
assert resampled.shape == (30, 3)
# pontos de uma trajetoria linear no grid padrao continuam lineares
assert abs(resampled[0, 0] - 0.0) < 1e-6
assert abs(resampled[-1, 0] - 999.0) < 1e-6
assert abs(resampled[15, 0] - 999.0 * 15 / 29) < 1e-4
assert np.allclose(np.diff(resampled[:, 0], 2), 0.0, atol=1e-2)


# ---------------------------------------------------------------------------
# 4) Normalizador de trajetoria
# ---------------------------------------------------------------------------
rng = np.random.default_rng(3)
sequences = rng.uniform(-0.8, 0.8, (40, 30, 3))
normalizer = st.TrajectoryNormalizer()
normalizer.fit(sequences)
assert normalizer.std.shape == (3,) and np.all(normalizer.std > 0)
recovered = normalizer.from_normalized(
    normalizer.to_normalized(sequences))
assert np.allclose(recovered, sequences, atol=1e-6)
stats = normalizer.state()
assert set(stats) == {"mean", "std"} and len(stats["mean"]) == 3


# ---------------------------------------------------------------------------
# 5) Checkpoint + wrapper de inferencia
# ---------------------------------------------------------------------------
config = {
    "arch": "SANDTrajectory", "fs": 500.0, "n_timesteps": 1000,
    "output_seq_len": 30, "d_model": 32, "n_layers": 4,
    "car": True, "zscore": "window", "f_lo": 1.0, "f_hi": 30.0,
    "channel_map": list(range(1, 33)),
}
st.save_checkpoint(TMP, net, config, normalizer,
                   eeg_mean=np.zeros(32), eeg_std=np.ones(32),
                   meta={"origin_xyz_m": [0.5, 1.1, 1.6]})
sand = st.SandTrajectoryBCI(TMP)
assert sand.n_timesteps == 1000 and sand.output_seq_len == 30
assert np.allclose(sand.channel_map, np.arange(32))
assert sand.cfg["meta_origin_m"] == [0.5, 1.1, 1.6]
trajectory, movement = sand.predict_window(block)
assert trajectory.shape == (30, 3)
assert np.isfinite(trajectory).all()
assert isinstance(movement, float) and movement >= 0.0
# checkpoint de outra arquitetura e rejeitado
wrong = {"arch": "SANDClassifier", "config": config}
try:
    st.SandTrajectoryBCI(wrong)
    raise AssertionError("arch errado deveria falhar")
except ValueError:
    pass
# janela curta demais na inferencia
try:
    sand.predict_window(block[:500])
    raise AssertionError("janela curta deveria falhar")
except ValueError:
    pass

TMP.unlink(missing_ok=True)
print("SAND_TRAJ_OK: 6/6 blocos")
sys.exit(0)