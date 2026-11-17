"""Teste do REGULARIZADOR ANATOMICO (adaptacao do L_DCL do MTRT ao punho).

Motivo cientifico: o alvo da rede e' a posicao do PUNHO, entao o cotovelo nao e'
identificavel ponto a ponto; o que a geometria de 2 elos determina univocamente
e' o raio r = ||punho - ombro|| e, por lei dos cossenos, o angulo do cotovelo
implicado. O MTRT (Wang et al., IEEE TNSRE 2023) usa comprimentos de elo
constantes como FUNCAO DE PERDA; aqui fazemos o mesmo para o punho, com tres
termos: envelope de alcance, angulo implicado x angulo MEDIDO e limites
articulares.

Casos:
1. lei dos cossenos: r = l1+l2 -> 180 graus; r = |l1-l2| -> 0; r = hipotenusa
   de 90 graus -> 90;
2. consistencia com a definicao do projeto (kinect_imu_groundtruth.
   interior_angle_deg) para uma configuracao de 2 elos construida a mao;
3. envelope: 0 dentro do alcance, > 0 fora, e o gradiente traz o punho de volta;
4. termo de angulo: 0 quando o angulo implicado coincide com o medido;
5. limites articulares: so' disparam fora de [min, max];
6. dados de braco ausentes (tudo NaN): termos = 0 e nenhum NaN vaza para a perda;
7. ida-e-volta com o gerador sintetico: colunas ARM_* escritas e lidas certas
   (ombro, comprimentos de elo e angulo coerentes com o raio);
8. treino de 1 epoca com --peso-anatomico: termo finito, checkpoint registra a
   configuracao anatomica e o treino continua funcionando.
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "tools"))

import gerar_sessao_sintetica as gen      # noqa: E402
import sand_traj_treino as treino         # noqa: E402
import sand_trajectory_model as st        # noqa: E402

L1, L2 = 0.30, 0.27
PASTA = tempfile.mkdtemp(prefix="teste_anatomico_")


def graus(radiano):
    return float(np.rad2deg(radiano))


# 1) lei dos cossenos
assert abs(graus(st.elbow_angle_from_radius(L1 + L2, L1, L2)) - 180.0) < 1e-4
assert abs(graus(st.elbow_angle_from_radius(abs(L1 - L2), L1, L2))) < 1e-4
meio = float(np.hypot(L1, L2))
assert abs(graus(st.elbow_angle_from_radius(meio, L1, L2)) - 90.0) < 1e-4
# monotono em r e valor exato num ponto interior (r = 0,50 m -> 122,53 graus)
raios = [0.10, 0.20, 0.31, 0.40, 0.50]
angulos = [graus(st.elbow_angle_from_radius(r, L1, L2)) for r in raios]
assert all(seguinte > atual for atual, seguinte in zip(angulos, angulos[1:])), \
    angulos
assert abs(angulos[-1] - 122.53) < 0.05, angulos[-1]
print("1) OK: lei dos cossenos (180/90/0 graus nos extremos, monotono em r, "
      f"r=0,50 -> {angulos[-1]:.2f} graus)")

# 2) mesma definicao do projeto: interior_angle_deg(ombro, cotovelo, punho)
import kinect_imu_groundtruth as kig      # noqa: E402

interno_de = kig.ArmLinkModel.interior_angle_deg    # metodo estatico do projeto
ombro = np.zeros(3)
for abertura in (30.0, 60.0, 90.0, 120.0):
    angulo = np.deg2rad(abertura)
    cotovelo = np.array([L1, 0.0, 0.0])
    punho = cotovelo + L2 * np.array([np.cos(angulo), np.sin(angulo), 0.0])
    interno = float(interno_de(ombro, cotovelo, punho))
    raio = float(np.linalg.norm(punho - ombro))
    pelo_raio = graus(st.elbow_angle_from_radius(raio, L1, L2))
    assert abs(interno - pelo_raio) < 1e-3, (abertura, interno, pelo_raio)
    assert abs(interno - (180.0 - abertura)) < 1e-3, (abertura, interno)
print("2) OK: consistente com kinect_imu_groundtruth.interior_angle_deg "
      "(4 aberturas)")


def trajetoria_constante(raio, ombro=(0.0, 0.0, 0.0), pontos=4,
                         requer_grad=False):
    """Punho parado a distancia `raio` do ombro (lote de 1 trajetoria)."""
    centro = torch.tensor(ombro, dtype=torch.float32)
    punho = centro.clone()
    punho[2] = centro[2] + float(raio)          # desloca no eixo z
    punho = punho.reshape(1, 1, 3).repeat(1, pontos, 1)
    if requer_grad:
        punho.requires_grad_(True)
    return punho, centro.reshape(1, 1, 3).repeat(1, pontos, 1)


# 3) envelope de alcance
reg_env = st.AnatomicalRegularizer(L1, L2, peso_envelope=1.0, peso_angulo=0.0,
                                   peso_limites=0.0)
alcance = L1 + L2
assert float(reg_env.componentes(*trajetoria_constante(0.40))["envelope"]) == 0.0
fora = float(reg_env.componentes(*trajetoria_constante(0.70))["envelope"])
assert fora > 0.01, fora
perto = float(reg_env.componentes(*trajetoria_constante(0.02))["envelope"])
assert perto > 0.0, perto
punho, ombro = trajetoria_constante(0.70, requer_grad=True)
otimizador = torch.optim.Adam([punho], lr=0.01)
inicial = float(reg_env.componentes(punho, ombro)["envelope"].detach())
for _ in range(600):
    otimizador.zero_grad()
    reg_env.componentes(punho, ombro)["total"].backward()
    otimizador.step()
raio_final = float(torch.linalg.norm(punho - ombro, dim=-1).mean().detach())
final = float(reg_env.componentes(punho, ombro)["envelope"])
assert raio_final < alcance, (raio_final, alcance)     # entrou no alcance fisico
assert raio_final <= alcance * (1.0 - reg_env.margem) + 0.005, raio_final
assert final < inicial * 0.02, (inicial, final)
print(f"3) OK: envelope 0 dentro, >0 fora (0,70 m -> {fora:.4f}) e o gradiente "
      f"trouxe o punho de 0,70 m para {raio_final:.3f} m "
      f"(alcance util {(alcance * (1 - reg_env.margem)):.3f} m; penalidade "
      f"{inicial:.3e} -> {final:.3e})")

# 4) termo de angulo: penaliza a diferenca entre o angulo implicado e o MEDIDO
reg_ang = st.AnatomicalRegularizer(L1, L2, peso_envelope=0.0, peso_angulo=1.0,
                                   peso_limites=0.0)
raio_alvo = 0.40
medido = graus(st.elbow_angle_from_radius(raio_alvo, L1, L2))
angulo = torch.full((1, 4), medido)
assert float(reg_ang.componentes(*trajetoria_constante(raio_alvo),
                                 angulo_real_deg=angulo)["angulo"]) < 1e-9
errado = float(reg_ang.componentes(*trajetoria_constante(0.45),
                                   angulo_real_deg=angulo)["angulo"])
assert errado > 0.01, errado
print(f"4) OK: angulo implicado == medido ({medido:.1f} graus) zera o termo; "
      f"raio errado (0,45 vs 0,40) da {errado:.3f}")

# 5) limites articulares
reg_lim = st.AnatomicalRegularizer(L1, L2, peso_envelope=0.0, peso_angulo=0.0,
                                   peso_limites=1.0)
dentro = float(reg_lim.componentes(*trajetoria_constante(0.40))["limites"])
assert dentro == 0.0, dentro                      # ~89 graus, dentro da faixa
dobrado = float(reg_lim.componentes(*trajetoria_constante(0.05))["limites"])
assert dobrado > 0.0, dobrado
print(f"5) OK: limites 15-178 graus (dobrado a 0,05 m -> {dobrado:.3e})")

# 6) braco ausente (NaN) nao gera perda nem NaN
reg_completo = st.AnatomicalRegularizer(L1, L2, peso_envelope=1.0,
                                        peso_angulo=1.0, peso_limites=1.0)
nan = torch.full((1, 4, 3), float("nan"))
angulo_nan = torch.full((1, 4), float("nan"))
vazio = reg_completo.componentes(nan, nan, angulo_nan)
for chave, valor in vazio.items():
    assert float(valor) == 0.0 and np.isfinite(float(valor)), (chave, valor)
misto = reg_completo.componentes(*trajetoria_constante(0.70), angulo_real_deg=angulo_nan)
assert np.isfinite(float(misto["total"])) and float(misto["total"]) > 0.0
assert float(misto["angulo"]) == 0.0, float(misto["angulo"])
print("6) OK: sem braco medido (NaN) os tres termos ficam em 0, sem NaN; "
      "com ombro valido e angulo NaN so' o termo de angulo zera")


def args_treino(*extra):
    guardado = sys.argv
    sys.argv = ["sand_traj_treino.py", "--data", PASTA, "--epochs", "1",
                "--output-seq-len", "30", "--seed", "3"] + list(extra)
    try:
        return treino.parse_args()
    finally:
        sys.argv = guardado


# 7) ida-e-volta com o gerador sintetico (colunas ARM_* reais no arquivo)
args_gen = argparse.Namespace(out=PASTA, sessoes=2, trials=4, fs=500.0,
                              canais=32, seed=5, snr=1.0, sem_ruido=False,
                              sem_direcao=False)
rng = np.random.default_rng(args_gen.seed)
for numero in (1, 2):
    prefixo, n_amostras, trials, duracao = gen.escreve_sessao(args_gen, numero,
                                                              rng)
    gen.escreve_movimento_e_eventos(args_gen, prefixo, n_amostras, trials,
                                    duracao, numero, rng)
args = args_treino("--peso-anatomico", "1.0")
gravacoes = treino.find_recordings(args.data)
assert len(gravacoes) == 2, gravacoes
x, y, sess, orig, braco = treino.load_all_epochs(gravacoes, args,
                                                 com_braco=True)
assert x.shape[1:] == (32, 1000), x.shape
assert y.shape == (8, 30, 3), y.shape
assert braco.shape == (8, 30, 7), braco.shape
assert np.isfinite(braco).all(), "colunas ARM_* viraram NaN no caminho de arquivo"
assert np.allclose(braco[:, :, 4], gen.L1_M, atol=1e-4)
assert np.allclose(braco[:, :, 5], gen.L2_M, atol=1e-4)
assert np.allclose(braco[:, :, 6], 1.0)
assert np.allclose(braco[:, :, 0:3], gen.OMBRO_M, atol=1e-4)
raio = np.linalg.norm(y - braco[:, :, 0:3], axis=-1)
do_raio = np.rad2deg(st.elbow_angle_from_radius(raio, gen.L1_M, gen.L2_M))
erro = float(np.abs(do_raio - braco[:, :, 3]).max())
assert erro < 2.0, erro            # tolera a reamostragem 30 Hz -> 30 pontos
print(f"7) OK: ARM_* no caminho de arquivo: A={braco.shape}, elos "
      f"({braco[0, 0, 4]:.3f}/{braco[0, 0, 5]:.3f} m), angulo medido x angulo "
      f"do raio com erro maximo {erro:.2f} graus")

# 8) treino com o regularizador: termo finito e configuracao no checkpoint
modelo_pt = os.path.join(PASTA, "_teste_anat.pt")
args = args_treino("--peso-anatomico", "1.0", "--model-out", modelo_pt)
anatomico = treino.monta_anatomico(args, braco)
assert anatomico is not None and anatomico.ativo
assert abs(anatomico.l1 - gen.L1_M) < 0.02, anatomico.l1
assert abs(anatomico.l2 - gen.L2_M) < 0.02, anatomico.l2
modelo = st.SANDTrajectory(n_channels=32, n_timesteps=1000, d_model=16,
                           n_layers=2, output_seq_len=30)
carregador = torch.utils.data.DataLoader(
    treino.TrajDataset(x, y, braco), batch_size=4, shuffle=False)
otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
perda, componentes = treino.train_epoch(modelo, carregador, otimizador,
                                        torch.nn.MSELoss(),
                                        torch.device("cpu"), anatomico=anatomico)
assert np.isfinite(perda)
for chave, valor in componentes.items():
    assert np.isfinite(float(valor)), (chave, valor)
assert set(componentes) >= {"total", "envelope", "angulo", "limites"}
n_val = max(1, int(len(gravacoes) * args.test_frac))
mascara_val = np.isin(sess, list(set(np.arange(len(gravacoes))[-n_val:])))
resultado = treino.run_training(
    args, x[~mascara_val], y[~mascara_val], x[mascara_val], y[mascara_val],
    np.arange(32), {"teste": True}, anatomico=anatomico,
    arm_train=braco[~mascara_val], arm_val=braco[mascara_val])
assert np.isfinite(resultado["best_val_loss"])
config = st.load_checkpoint(modelo_pt)["config"]
assert config["anatomico"] is not None
assert abs(config["anatomico"]["l1_m"] - gen.L1_M) < 0.02, config["anatomico"]
assert config["anatomico"]["peso_envelope"] == 1.0
assert np.allclose(config["anatomico"]["limites_cotovelo_deg"],
                   [15.0, 178.0], atol=1e-9), \
    config["anatomico"]["limites_cotovelo_deg"]
print(f"8) OK: 1 epoca com o termo anatomico (total={float(componentes['total']):.2e}, "
      f"env={float(componentes['envelope']):.2e}, "
      f"ang={float(componentes['angulo']):.3f}, "
      f"lim={float(componentes['limites']):.2e}); checkpoint registra "
      f"l1={config['anatomico']['l1_m']:.3f} m")

print("ANATOMICAL_REG_OK: 8/8 casos (lei dos cossenos, IK do projeto, envelope "
      "+ gradiente, angulo medido, limites, NaN seguro, ARM_* no arquivo, "
      "treino com o termo)")
shutil.rmtree(PASTA, ignore_errors=True)
sys.exit(0)

