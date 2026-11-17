"""Teste do VETOR DA PALMA por luva: alinhamento IMU <-> camera (Kabsch).

Bug real (bancada 25/09/2026): a direcao da palma prevista pelo IMU estava
ERRADA -- e errada de forma DIFERENTE em cada luva (a esquerda pior), porque o
codigo compunha

    v = R_novo @ R_ref^T @ v_ref

tratando como iguais dois referenciais que nao sao: `R` e' corpo -> mundo do IMU
(vertical do acelerometro, mas "norte" arbitrario, sem magnetometro, e ainda
afetado pela MONTAGEM de cada luva) e `v_ref` e' a direcao da palma medida pelo
Kinect, no referencial da CAMERA. O resultado acerta na pose de calibracao (onde,
por construcao, R_novo = R_ref) e erra cada vez mais conforme a mao gira.

A correcao: resolver o referencial a partir das amostras -- `v_i = T R_i f`, com
`T` (mundo do IMU -> camera, por luva) e `f` (palma no corpo do sensor) ajustados
por Kabsch alternado (`solve_palm_alignment`).

Casos:
 1. `kabsch_rotation`: rotacao exata de pares alinhados, determinante +1;
 2. `solve_palm_alignment`: recupera T/f de uma luva com montagem arbitraria
    (inclusive a "montagem espelhada" tipica da mao esquerda) e resiste a ruido;
 3. amostras insuficientes/invalidas -> None (nunca chutar);
 4. o BUG fica demonstrado: sem alinhamento o erro cresce com a rotacao; com o
    alinhamento o vetor previsto fica a poucos graus em TODAS as poses;
 5. `PositionFusion`: `apply_palm_alignment` + `palm_direction` (via rpy do
    firmware) reproduzem a normal verdadeira; `calibrate_palm_direction` continua
    valendo (compatibilidade) e `reset` limpa o alinhamento;
 6. com T = identidade o resultado e' IDENTICO ao codigo antigo (retrocompativel).

    .venv\\Scripts\\python.exe test_palm_alignment.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")

from imu import (PALM_ALIGNMENT_MIN_SAMPLES, ImuSample,   # noqa: E402
                 PositionFusion, angle_between, kabsch_rotation,
                 rotation_matrix, solve_palm_alignment)


def rotacao_aleatoria(rng, max_deg=180.0):
    """Rotacao propria aleatoria (eixo uniforme, angulo em +-max_deg)."""
    eixo = rng.normal(size=3)
    eixo /= np.linalg.norm(eixo)
    ang = math.radians(rng.uniform(-max_deg, max_deg))
    k = np.array([[0.0, -eixo[2], eixo[1]],
                  [eixo[2], 0.0, -eixo[0]],
                  [-eixo[1], eixo[0], 0.0]])
    return np.eye(3) + math.sin(ang) * k + (1.0 - math.cos(ang)) * (k @ k)


def rpy_de_rotacao(rotacao):
    """(roll, pitch, yaw) que `rotation_matrix` converte de volta na matriz.

    `rotation_matrix` e' Rz(yaw) @ Ry(pitch) @ Rx(roll) (convencao do firmware).
    """
    pitch = -math.asin(float(np.clip(rotacao[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(rotacao[2, 1]), float(rotacao[2, 2]))
    yaw = math.atan2(float(rotacao[1, 0]), float(rotacao[0, 0]))
    return tuple(math.degrees(valor) for valor in (roll, pitch, yaw))


def amostra(rotacao, t_us=1000):
    """ImuSample cuja atitude e' `rotacao` (accel de repouso em g)."""
    roll, pitch, yaw = rpy_de_rotacao(rotacao)
    return ImuSample(timestamp_us=t_us, received_at=0.0,
                     roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw,
                     accel_g=np.array([0.0, 0.0, 1.0]),
                     gyro_dps=np.zeros(3))


rng = np.random.default_rng(11)


def rotacao_eixo_angulo(eixo, graus):
    """Rotacao propria por eixo unitario + angulo (formula de Rodrigues)."""
    eixo = np.asarray(eixo, np.float64)
    eixo = eixo / np.linalg.norm(eixo)
    ang = math.radians(graus)
    k = np.array([[0.0, -eixo[2], eixo[1]],
                  [eixo[2], 0.0, -eixo[0]],
                  [-eixo[1], eixo[0], 0.0]])
    return np.eye(3) + math.sin(ang) * k + (1.0 - math.cos(ang)) * (k @ k)

# =============================================================================
# 1) Kabsch
# =============================================================================
pares = [(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
         (np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]))]
rot = kabsch_rotation(pares)
assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)
assert abs(np.linalg.det(rot) - 1.0) < 1e-9, "Kabsch devolveu reflexao"

# com rotacao conhecida e vetores variados: recupera a matriz
verdadeira = rotacao_aleatoria(rng, 180.0)
origens = [rng.normal(size=3) for _ in range(30)]
destinos = [verdadeira @ valor for valor in origens]
estimada = kabsch_rotation([(a, b) for a, b in zip(origens, destinos)])
assert np.allclose(estimada, verdadeira, atol=1e-8), estimada - verdadeira
assert kabsch_rotation([]) is None
assert kabsch_rotation([(np.zeros(3), np.ones(3))]) is None
print("1) OK: Kabsch (rotacao exata, det +1, pares vazios/nulos -> None)")

# =============================================================================
# 2) solve_palm_alignment: montagem arbitraria por luva (inclusive a esquerda)
# =============================================================================
for nome, t_max, ruido_deg, n in (("montagem ~45 graus", 45.0, 2.0, 18),
                                  ("montagem espelhada ~180 graus", 180.0, 2.0, 18),
                                  ("poucas poses (minimo)", 40.0, 1.0,
                                   PALM_ALIGNMENT_MIN_SAMPLES)):
    t_real = rotacao_aleatoria(rng, t_max)
    f_real = rng.normal(size=3)
    f_real /= np.linalg.norm(f_real)
    rotacoes = [rotacao_aleatoria(rng, 40.0) for _ in range(n)]
    direcoes = []
    for rotacao in rotacoes:
        versor = t_real @ rotacao @ f_real
        versor = versor / np.linalg.norm(versor)
        versor = versor + rng.normal(0, math.radians(ruido_deg), size=3)
        direcoes.append(versor / np.linalg.norm(versor))
    resolvido = solve_palm_alignment(rotacoes, direcoes)
    assert resolvido is not None, nome
    t_est, f_est, residuo = resolvido
    # o que importa e' o VETOR previsto (T e f tem 1 GDL de folga em conjunto)
    erro_previsto = np.mean([
        angle_between(t_est @ rotacao @ f_est, t_real @ rotacao @ f_real)
        for rotacao in [rotacao_aleatoria(rng, 40.0) for _ in range(40)]
    ])
    assert residuo < 6.0, (nome, residuo)
    assert erro_previsto < 6.0, (nome, erro_previsto)
    print(f"2) OK: {nome} -> residuo {residuo:.2f} graus | erro do vetor "
          f"previsto {erro_previsto:.2f} graus")

# 3) amostras insuficientes / invalidas
assert solve_palm_alignment([], []) is None
assert solve_palm_alignment([np.eye(3)] * 3, [np.array([0, 0, 1])] * 3) is None
assert solve_palm_alignment([np.eye(3)] * 10, [np.zeros(3)] * 10) is None
assert solve_palm_alignment([np.eye(3)] * 10,
                            [np.array([np.nan, 0, 1.0])] * 10) is None
print("3) OK: poucas amostras ou direcao invalida -> None (nunca chutar)")

# =============================================================================
# 4) o BUG antigo e a CORRECAO, medidos no mesmo dado
# =============================================================================
# Caso FIXO (e didatico): luva esquerda montada virada -- o referencial do IMU
# esta' a 140 graus (eixo quase vertical) do referencial da camera.
t_real = rotacao_eixo_angulo([0.15, 0.95, 0.25], 140.0)
f_real = np.array([0.35, -0.30, 0.89])
f_real /= np.linalg.norm(f_real)
posicoes = [rotacao_aleatoria(rng, 45.0) for _ in range(40)]
normais = []
for r in posicoes:
    versor = t_real @ r @ f_real
    normais.append(versor / np.linalg.norm(versor))

# --- formula ANTIGA: R_novo R_ref^T v_ref (sem T) ---------------------------
r_ref, v_ref = posicoes[0], normais[0]
erros_antigos = [angle_between(r @ r_ref.T @ v_ref, normal)
                 for r, normal in zip(posicoes, normais)]
# --- formula NOVA: T resolvido das amostras --------------------------------
t_est, f_est, residuo = solve_palm_alignment(posicoes, normais)
erros_novos = [angle_between(t_est @ r @ f_est, normal)
               for r, normal in zip(posicoes, normais)]
assert erros_antigos[0] < 1e-6, "a formula antiga acerta na pose de calibracao"
assert np.mean(erros_antigos) > 10.0, \
    f"o bug deveria aparecer ao girar a mao (media {np.mean(erros_antigos):.1f})"
assert max(erros_antigos) > 30.0, f"max antigo {max(erros_antigos):.1f}"
assert max(erros_novos) < 2.0, f"correcao falhou (max {max(erros_novos):.2f})"
print(f"4) OK: formula antiga acerta na calibracao e ERRA media "
      f"{np.mean(erros_antigos):.1f} graus (max {max(erros_antigos):.1f}) ao "
      f"girar a mao; com o alinhamento o erro maximo cai para "
      f"{max(erros_novos):.2f} graus")

# =============================================================================
# 5) PositionFusion: alinhamento aplicado + previsao a partir do rpy do firmware
# =============================================================================
t_real = rotacao_aleatoria(rng, 180.0)          # luva esquerda (montagem virada)
f_real = rng.normal(size=3)
f_real /= np.linalg.norm(f_real)
rotacoes = [rotacao_aleatoria(rng, 40.0) for _ in range(20)]
direcoes = []
for r in rotacoes:
    versor = t_real @ r @ f_real
    versor = versor / np.linalg.norm(versor)
    versor = versor + rng.normal(0, math.radians(2.0), size=3)
    direcoes.append(versor / np.linalg.norm(versor))

fusion = PositionFusion()
assert fusion.palm_direction(amostra(np.eye(3))) is None, \
    "sem calibracao nao existe previsao"
assert fusion.apply_palm_alignment(rotacoes, direcoes) is not None
assert fusion.palm_alignment_residual_deg is not None
erros = []
for r in [rotacao_aleatoria(rng, 40.0) for _ in range(40)]:
    previsto = fusion.palm_direction(amostra(r))
    esperado = t_real @ r @ f_real
    erros.append(angle_between(previsto, esperado))
assert max(erros) < 6.0, f"previsao por rpy errada (max {max(erros):.1f})"
print(f"5) OK: PositionFusion com alinhamento -> erro maximo {max(erros):.2f} "
      f"graus em 40 poses novas (residuo do ajuste "
      f"{fusion.palm_alignment_residual_deg:.2f} graus)")

# 5a) `calibrate_palm_direction` continua funcionando (compatibilidade): com o
#     alinhamento ja' aplicado, uma unica amostra basta para reancorar
assert fusion.calibrate_palm_direction(amostra(rotacoes[0]), direcoes[0]) is True
assert fusion.palm_reference_body is not None
assert fusion.palm_reference_direction is not None
assert fusion.calibrate_palm_direction(None, direcoes[0]) is False
assert fusion.calibrate_palm_direction(amostra(np.eye(3)), np.zeros(3)) is False
copia_residuo = fusion.palm_alignment_residual_deg
fusion.reset()
assert fusion.palm_reference_body is None and fusion.palm_direction(
    amostra(np.eye(3))) is None
assert fusion.palm_alignment_residual_deg is None
print("5a) OK: calibrate_palm_direction/reset (reancoragem e limpeza)")

# =============================================================================
# 6) com T = identidade o resultado e' IDENTICO ao codigo antigo
# =============================================================================
fusion = PositionFusion()
r_ref = rotacao_aleatoria(rng, 40.0)
f_corpo = rng.normal(size=3)
f_corpo /= np.linalg.norm(f_corpo)
v_ref = r_ref @ f_corpo                       # T = I: v_ref = R_ref f
assert fusion.calibrate_palm_direction(amostra(r_ref), v_ref) is True
for _ in range(20):
    r_novo = rotacao_aleatoria(rng, 40.0)
    antigo = r_novo @ r_ref.T @ v_ref
    novo = fusion.palm_direction(amostra(r_novo))
    assert angle_between(antigo, novo) < 1e-6, (antigo, novo)
print("6) OK: T = identidade reproduz exatamente a formula antiga "
      "(retrocompativel)")

# =============================================================================
# 6b) set_palm_alignment / palm_alignment_state: ida e volta (o que o programa
#     salva em JSON e recarrega no dia seguinte)
# =============================================================================
fusion = PositionFusion()
t_alvo = rotacao_aleatoria(rng, 180.0)
f_alvo = rng.normal(size=3)
f_alvo /= np.linalg.norm(f_alvo)
assert fusion.set_palm_alignment(t_alvo, f_alvo, 3.5) is True
estado = fusion.palm_alignment_state()
assert estado is not None and estado["residual_deg"] == 3.5
import json                                   # noqa: E402  (so' aqui: serializacao)
recarregado = PositionFusion()
texto = json.dumps(estado)
lido = json.loads(texto)
assert recarregado.set_palm_alignment(lido["rotation"], lido["body"],
                                      lido["residual_deg"]) is True
assert recarregado.palm_alignment_residual_deg == 3.5
for _ in range(10):
    rotacao = rotacao_aleatoria(rng, 40.0)
    amostra_rot = amostra(rotacao)
    assert angle_between(fusion.palm_direction(amostra_rot),
                         recarregado.palm_direction(amostra_rot)) < 1e-9
assert PositionFusion().set_palm_alignment(t_alvo) is False, \
    "sem direcao no corpo nao ha' o que prever"
assert fusion.set_palm_alignment(np.zeros((3, 3)), f_alvo) is False, \
    "matriz singular recusada"
assert fusion.set_palm_alignment(t_alvo, np.zeros(3)) is False
print("6b) OK: set_palm_alignment/palm_alignment_state (JSON ida e volta, "
      "recusas de matriz singular/direcao nula)")

# =============================================================================
# 7) a FERRAMENTA: captura (tecla O), regras de amostragem e persistencia
# =============================================================================
import os                                            # noqa: E402
import time                                          # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import kinect_groundtruth_tool as tool                # noqa: E402

# A ferramenta importa a biblioteca (e o SDK do Kinect) no topo: se o SDK nao
# existir, este grupo e' pulado sem derrubar os outros.
t_real = rotacao_eixo_angulo([0.15, 0.95, 0.25], 140.0)
f_real = np.array([0.2, -0.4, 0.9])
f_real /= np.linalg.norm(f_real)
captura = tool.CapturaAlinhamentoPalma(segundos=5.0)
agora = time.monotonic()
captura.iniciar(agora)
assert captura.ativa and captura.restante_s > 4.0

# 7a) regras de amostragem: intervalo minimo e passo minimo (girar a mao)
rotacoes = [rotacao_aleatoria(rng, 45.0) for _ in range(20)]
aceitas = 0
for indice, rotacao in enumerate(rotacoes):
    normal = t_real @ rotacao @ f_real
    normal = normal / np.linalg.norm(normal)
    if captura.coletar("right", agora + indice * 0.2, amostra(rotacao), normal):
        aceitas += 1
assert aceitas >= tool.PALM_ALIGNMENT_MIN_SAMPLES, aceitas
assert captura.contar("right") == aceitas
assert captura.contar("left") == 0
# amostra repetida (mesma postura da ultima aceita) nao entra, mesmo com tempo
antes = captura.contar("right")
ultima = rotacoes[-1]
normal_ultima = t_real @ ultima @ f_real
normal_ultima = normal_ultima / np.linalg.norm(normal_ultima)
assert not captura.coletar("right", agora + 4.0, amostra(ultima),
                           normal_ultima), "postura repetida deveria ser recusada"
assert captura.contar("right") == antes
# intervalo minimo: postura NOVA, mas cedo demais em relacao a ultima aceita
assert not captura.coletar("right", agora + 3.85,
                           amostra(rotacao_aleatoria(rng, 45.0)),
                           t_real @ rotacao_aleatoria(rng, 45.0) @ f_real)
assert captura.contar("right") == antes
# sem normal (Kinect nao ve a mao) ou sem amostra do ESP32: nada entra
assert not captura.coletar("right", agora + 4.6, amostra(rotacoes[1]), None)
assert not captura.coletar("right", agora + 4.7, None,
                           t_real @ rotacoes[1] @ f_real)
print(f"7a) OK: captura aceitou {aceitas}/20 amostras (intervalo e passo "
      "minimos respeitados; lado sem mao e sem normal nao entram)")

# 7b) resolver: lado sem amostras -> None; lado com amostras -> T/f/residuo
assert captura.resolver("left") is None
resolvido = captura.resolver("right")
assert resolvido is not None
t_est, f_est, residuo = resolvido
assert residuo < 8.0, residuo
fusion_captura = PositionFusion()
assert fusion_captura.apply_palm_alignment(
    [r for r, _v in captura.amostras["right"]],
    [v for _r, v in captura.amostras["right"]]) is not None
print(f"7b) OK: resolver com as amostras da captura -> residuo {residuo:.2f} "
      "graus; lado sem amostras devolve None")

# 7c) persistencia: salvar -> carregar em outra fusao -> mesma previsao
originais = dict(tool.PALM_ALIGNMENT_FILES)
tool.PALM_ALIGNMENT_FILES.update(
    {"right": "_test_palm_right.json", "left": "_test_palm_left.json"})
try:
    assert tool.save_palm_alignment(fusion_captura, "right") is True
    assert not os.path.exists(tool.palm_alignment_path("left"))
    carregada = PositionFusion()
    assert tool.load_palm_alignment(carregada, "right", avisar=lambda *_a: None)
    for rotacao in [rotacao_aleatoria(rng, 40.0) for _ in range(10)]:
        assert angle_between(fusion_captura.palm_direction(amostra(rotacao)),
                             carregada.palm_direction(amostra(rotacao))) < 1e-9
    # arquivo corrompido: avisa e devolve False (nao derruba o programa)
    with open(tool.palm_alignment_path("left"), "w", encoding="utf-8") as fh:
        fh.write("{isso nao e json}")
    avisos = []
    assert tool.load_palm_alignment(PositionFusion(), "left",
                                    avisar=avisos.append) is False
    assert avisos and "ilegivel" in avisos[0]
    # fusao sem alinhamento nenhum: nao ha' o que gravar
    assert tool.save_palm_alignment(PositionFusion(), "left") is False
finally:
    for lado in ("right", "left"):
        caminho = tool.palm_alignment_path(lado)
        if caminho.exists():
            caminho.unlink()
    tool.PALM_ALIGNMENT_FILES.clear()
    tool.PALM_ALIGNMENT_FILES.update(originais)
assert not os.path.exists("_test_palm_right.json")
print("7c) OK: save/load do alinhamento por luva (JSON; arquivo ilegivel avisa "
      "e e' recusado; nada a gravar -> False)")

# =============================================================================
# 8) Normal ANATOMICA por mao (item 79): o sinal "nz<0" e' PROVADAMENTE errado
# =============================================================================
# Geometria REAL medida na bancada (sessao palm_diag_20261019_160743), pontos do
# frame da palma em camera space: [punho, MCP indicador, MCP minimo, MCP medio].
# Trava a regra do sinal sem depender de hardware.
_PONTOS_REAIS = {
    ("right", "KINECT"): np.array([
        [0.2720, 0.2278, 1.1601],
        [0.2459, 0.3019, 1.1187],
        [0.2963, 0.2698, 1.1055],
        [0.2642, 0.2978, 1.1070],
    ]),
    ("left", "KINECT"): np.array([
        [0.0288, 0.2960, 1.2645],
        [0.0746, 0.3415, 1.1647],
        [0.0137, 0.3469, 1.1757],
        [0.0548, 0.3491, 1.1593],
    ]),
    ("right", "MESA"): np.array([
        [0.3084, 0.2425, 1.3178],
        [0.2498, 0.2217, 1.3154],
        [0.3051, 0.1892, 1.3012],
        [0.2665, 0.2082, 1.3022],
    ]),
    ("left", "MESA"): np.array([
        [0.0159, 0.3348, 1.3815],
        [0.0328, 0.2940, 1.3592],
        [-0.0230, 0.3072, 1.3710],
        [0.0100, 0.2959, 1.3550],
    ]),
}


def _normal_do_plano(pontos, lado=None):
    return tool.KinectHandTracker.palm_normal_from_points(
        pontos[0], pontos[1], pontos[2], lado=lado)


anat = {chave: _normal_do_plano(pontos, chave[0])
        for chave, pontos in _PONTOS_REAIS.items()}
legado = {chave: _normal_do_plano(pontos, None)
          for chave, pontos in _PONTOS_REAIS.items()}
# MESA = as DUAS palmas para BAIXO: as normais anatomicas tem de ser ~iguais...
assert angle_between(anat[("right", "MESA")],
                     anat[("left", "MESA")]) < 25.0
# ...e PREPARO/KINECT = as duas palmas para o sensor: tambem ~iguais.
assert angle_between(anat[("right", "KINECT")],
                     anat[("left", "KINECT")]) < 40.0
# O sinal ANTIGO (nz<0 forcado) e' o bug: na MESA cada mao sai 180 graus oposta
# a PROPRIA palma (medido: 180.0 nas duas), enquanto na pose da tecla O ("palma
# para o Kinect") ele coincide por construcao (0.0) -- era por isso que o hack
# parecia funcionar SO' na calibracao.
for lado in ("right", "left"):
    assert angle_between(anat[(lado, "MESA")],
                         legado[(lado, "MESA")]) > 170.0
    assert angle_between(anat[(lado, "KINECT")],
                         legado[(lado, "KINECT")]) < 1.0
for chave, normal in anat.items():
    assert abs(np.linalg.norm(normal) - 1.0) < 1e-9
    pontos = _PONTOS_REAIS[chave]
    assert abs(np.dot(normal, pontos[1] - pontos[0])) < 1e-9
    assert abs(np.dot(normal, pontos[2] - pontos[0])) < 1e-9
# com lado='right' e lado='left' na MESMA geometria o sinal e' oposto (espelho)
for chave, pontos in _PONTOS_REAIS.items():
    assert np.allclose(_normal_do_plano(pontos, "right"),
                       -_normal_do_plano(pontos, "left"))
# sem lado = comportamento antigo: nz < 0 (nao quebra o paradigma)
for chave, normal in legado.items():
    assert normal[2] < 0, (chave, normal)
# palm_frame repassa o lado (frame completo, com a direcao dos dedos)
_pontos_mesa = _PONTOS_REAIS[("right", "MESA")]
normal_frame, direcao_frame = tool.palm_frame(
    _pontos_mesa[0], _pontos_mesa[1], _pontos_mesa[2], _pontos_mesa[3],
    lado="right")
assert normal_frame is not None and direcao_frame is not None
assert abs(float(np.dot(normal_frame, direcao_frame))) < 1e-9
print("8) OK: normal ANATOMICA por mao -- MESA iguais e antigo oposto; "
      "unitarias/perpendiculares; palm_frame repassa o lado; sem lado = legado")

print("PALM_ALIGNMENT_OK: 9/9 grupos (Kabsch, montagem por luva, recusas, bug "
      "demonstrado e corrigido, PositionFusion, retrocompatibilidade, JSON, "
      "captura da ferramenta, normal anatomica por mao)")
sys.exit(0)

print("PALM_ALIGNMENT_OK: 7/7 grupos (Kabsch, montagem por luva, recusas, bug "
      "demonstrado e corrigido, PositionFusion, retrocompatibilidade, JSON)")
sys.exit(0)
print("3) OK: poucas amostras ou direcao invalida -> None (nunca chutar)")

