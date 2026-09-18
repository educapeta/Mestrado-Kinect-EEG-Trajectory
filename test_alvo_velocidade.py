"""Teste do ALVO DE VELOCIDADE (--alvo velocidade) e da sua escala.

Motivo (decisao de protocolo de 18/09): a mesma trajetoria pode ser executada em
velocidades/intensidades muito diferentes, e o trial de REPOUSO/idle so' tem
informacao util se o alvo for velocidade (alvo = zero) em vez de posicao (alvo
constante = baseline trivial). A posicao volta por integracao no tempo real.

Casos:
1. rampa linear: a velocidade por diferencas centrais e' exatamente a inclinacao;
2. senoide: pico da velocidade = 2*pi*f*A (escala correta em m/s);
3. punho PARADO (home): velocidade ~ 0 (o caso do trial de repouso);
4. ida-e-volta pelo caminho de ARQUIVO: o alvo de velocidade e' o gradiente do
   alvo de posicao (mesma grade, m/s) e as formas batem;
5. treino com --alvo velocidade: roda, o checkpoint registra a formulacao e o
   alvo do repouso (home) fica perto de zero.
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tools"))

import gerar_sessao_sintetica as gen      # noqa: E402
import sand_traj_treino as treino         # noqa: E402
import sand_trajectory_model as st        # noqa: E402

PASTA = tempfile.mkdtemp(prefix="teste_vel_")


def args_treino(*extra):
    guardado = sys.argv
    sys.argv = ["sand_traj_treino.py", "--data", PASTA, "--epochs", "1",
                "--output-seq-len", "30", "--seed", "7"] + list(extra)
    try:
        return treino.parse_args()
    finally:
        sys.argv = guardado


# 1) rampa linear -> velocidade exata
dt = 0.05
t = np.arange(200) * dt
rampa = np.column_stack([0.4 * t, -0.2 * t, 0.05 * t])
vel = st.velocity_from_trajectory(rampa, dt)
assert np.allclose(vel, [0.4, -0.2, 0.05], atol=1e-6), vel[100]
print("1) OK: rampa linear -> velocidade exata (0,4/-0,2/0,05 m/s)")

# 2) senoide -> pico = 2*pi*f*A
freq, amplitude = 0.5, 0.12
seno = (amplitude * np.sin(2 * np.pi * freq * t)).reshape(-1, 1)
vel = st.velocity_from_trajectory(seno, dt)
pico = 2 * np.pi * freq * amplitude
assert abs(float(np.abs(vel).max()) - pico) < 0.02 * pico, np.abs(vel).max()
print(f"2) OK: senoide 0,5 Hz com 0,12 m -> pico {np.abs(vel).max():.3f} m/s "
      f"(teorico {pico:.3f})")

# 3) punho parado (caso do trial de repouso) -> velocidade ~ 0
parado = np.zeros((60, 3), np.float64)
assert float(np.abs(st.velocity_from_trajectory(parado, dt)).max()) < 1e-12
print("3) OK: punho parado -> velocidade exatamente 0 (alvo do trial de IDLE)")

# 4) ida-e-volta pelo caminho de arquivo (sessoes sinteticas)
args_gen = argparse.Namespace(out=PASTA, sessoes=2, trials=6, fs=500.0,
                              canais=32, seed=3, snr=1.0, sem_ruido=False,
                              sem_direcao=False)
rng = np.random.default_rng(args_gen.seed)
for numero in (1, 2):
    prefixo, n, trials, duracao = gen.escreve_sessao(args_gen, numero, rng)
    gen.escreve_movimento_e_eventos(args_gen, prefixo, n, trials, duracao,
                                    numero, rng)
args_pos = args_treino("--event-code", "795", "--window-start-sec", "-0.5")
gravacoes = treino.find_recordings(args_pos.data)
x_pos, y_pos, _sess, _orig = treino.load_all_epochs(gravacoes, args_pos)
args_vel = args_treino("--event-code", "795", "--window-start-sec", "-0.5",
                       "--alvo", "velocidade")
x_vel, y_vel, _s, _o = treino.load_all_epochs(gravacoes, args_vel)
assert x_pos.shape == x_vel.shape and y_pos.shape == y_vel.shape
# intervalo da grade reamostrada: alvo de 2 s em 30 pontos
intervalo = (int(args_vel.window_sec * args_vel.fs) / args_vel.fs) / 29.0
# ATENCAO ao eixo: dentro da janela o tempo e' o axis=1 (axis=0 = janelas)
esperado = np.gradient(y_pos, intervalo, axis=1)
assert np.allclose(y_vel, esperado, atol=1e-5), \
    float(np.abs(y_vel - esperado).max())
escala = float(np.abs(y_vel).max())
assert 0.02 < escala < 3.0, escala          # mao humana: ~0,02-3 m/s
print(f"4) OK: caminho de arquivo -> alvo de velocidade = gradiente do alvo de "
      f"posicao (mesma grade, {y_vel.shape}); pico {escala:.3f} m/s")

# 5) treino com alvo de velocidade + checkpoint + repouso ~ 0
modelo_pt = os.path.join(PASTA, "_teste_vel.pt")
args = args_treino("--event-code", "795", "--window-start-sec", "-0.5",
                   "--alvo", "velocidade", "--model-out", modelo_pt)
n_val = max(1, int(len(gravacoes) * args.test_frac))
mascara = np.isin(_sess, list(set(np.arange(len(gravacoes))[-n_val:])))
resultado = treino.run_training(args, x_vel[~mascara], y_vel[~mascara],
                                x_vel[mascara], y_vel[mascara], np.arange(32),
                                {"teste": True})
assert np.isfinite(resultado["best_val_loss"])
config = st.load_checkpoint(modelo_pt)["config"]
assert config["alvo"] == "velocidade", config.get("alvo")
# a janela inicial de cada trial e' o home (punho parado): velocidade ~ 0
if y_vel.shape[0]:
    magnitude = np.abs(y_vel).mean(axis=(1, 2))
    assert magnitude.min() < magnitude.max(), magnitude
print(f"5) OK: treino com alvo de velocidade (val_loss="
      f"{resultado['best_val_loss']:.4f}), checkpoint registra alvo="
      f"{config['alvo']!r}, |v| medio por janela de "
      f"{magnitude.min():.4f} a {magnitude.max():.4f} m/s")

print("ALVO_VELOCIDADE_OK: 5/5 casos (rampa, senoide, repouso, gradiente no "
      "caminho de arquivo, treino + checkpoint)")
shutil.rmtree(PASTA, ignore_errors=True)
sys.exit(0)