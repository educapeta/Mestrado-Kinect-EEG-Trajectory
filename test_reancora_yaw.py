"""Regressao: REANCORAGEM DO YAW DO BOOT do IMU (item 97).

Problema real (21/10/2026): o alinhamento da palma acertava a MESA e errava
~100 graus quando a palma ficava de frente para o sensor. Diagnostico: o yaw
ABSOLUTO do IMU de 6 eixos e' ARBITRARIO a cada boot (nao ha magnetometro; so'
roll/pitch vem do acelerometro). O `T` (mundo do IMU -> camera) do alinhamento
salvo absorve o yaw daquele boot, entao o arquivo deixa de valer depois de
religar/reflashar o ESP32 -- mesmo com a luva montada igual (o `f`, no corpo do
sensor, nao muda). E' por isso que "refazer a O" nao resolvia de forma duradoura.

Solucao: `reancorar_yaw` resolve o delta de yaw do boot atual usando a normal da
palma medida pelo Kinect e aplica `T_novo = T_salvo Rz(-delta)`; o
`AncoradorYawBoot` junta as amostras e aplica isso na `PositionFusion` EM
MEMORIA (o JSON salvo continua valendo para qualquer boot).

Casos:
  1. yaw do boot conhecido (varios deltas, inclusive > 90 graus): o reancoramento
     recupera o delta e a seta volta a bater com a verdade;
  2. mesmo com ruido na normal a recuperacao continua boa;
  3. pose degenerada (palma ~vertical, R f ~ z do mundo): devolve None (o yaw
     nao e' observavel) -- o ancorador tem de esperar outra pose, nao chutar;
  4. `AncoradorYawBoot`: antes de ancorar a seta erra; na amostra que ancora a
     fusion passa a prever a verdade; nao ancora duas vezes; timeout marca
     `falhou` e avisa;
  5. a integracao no tool esta' ligada (import, hook antes da seta, flag de
     desligar e o delta publicado na mao).

Harness: aritmetica pura (sem Qt, sem Kinect, sem rede).
"""
from __future__ import annotations

import math

import numpy as np

import imu

FALHAS = []


def check(nome, condicao, detalhe=""):
    if condicao:
        print(f"   ok: {nome}")
    else:
        FALHAS.append(nome)
        print(f"   FALHOU: {nome} {detalhe}")


def rpy_de(rotacao):
    """Euler ZYX (mesma convencao de `imu.rotation_matrix`) de uma rotacao."""
    n = np.asarray(rotacao, np.float64).reshape(3, 3)
    pitch = math.asin(float(np.clip(-n[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(n[2, 1]), float(n[2, 2]))
    yaw = math.atan2(float(n[1, 0]), float(n[0, 0]))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def amostra_imu(rotacao):
    roll, pitch, yaw = rpy_de(rotacao)
    return imu.ImuSample(timestamp_us=0, received_at=0.0, roll_deg=roll,
                         pitch_deg=pitch, yaw_deg=yaw,
                         accel_g=np.array([0.0, 0.0, 1.0]),
                         gyro_dps=np.zeros(3))


def angulo_rotacao(a, b):
    """Angulo (graus) entre duas rotacoes (a distancia geodesica)."""
    traco = float(np.trace(np.asarray(a).T @ np.asarray(b)))
    return math.degrees(math.acos(float(np.clip((traco - 1.0) / 2.0, -1, 1))))


def monta_sessao(rng, n=60, f=(1.0, 0.0, 0.0)):
    """(R0, f, T, v) sinteticos: atitude + normal da palma coerentes."""
    f = np.asarray(f, np.float64)
    f = f / np.linalg.norm(f)
    T = imu.rotation_matrix(*rng.uniform(-180, 180, 3))
    rotacoes, normais = [], []
    for _ in range(n):
        r0 = imu.rotation_matrix(rng.uniform(-40, 40),
                                 rng.uniform(-40, 40),
                                 rng.uniform(-180, 180))
        rotacoes.append(r0)
        normais.append(T @ r0 @ f)
    return rotacoes, f, T, normais


def aplica_boot(rotacoes, delta_deg):
    """Simula o reboot: R_boot = Rz(delta) R_anterior (so' o yaw muda)."""
    rz = imu.rotation_matrix(0.0, 0.0, delta_deg)
    return [rz @ r for r in rotacoes]


def erro_seta(fusion, rotacoes, normais):
    erros = []
    for r, v in zip(rotacoes, normais):
        previsto = fusion.palm_direction(amostra_imu(r))
        if previsto is None:
            continue
        erros.append(imu.angle_between(np.asarray(previsto), np.asarray(v)))
    return float(np.mean(erros)) if erros else float("nan")


# ============================================================================
# 1) yaw do boot conhecido: o reancoramento recupera o delta e a seta
# ============================================================================
print("1) reancoragem recupera o delta de yaw do boot")
rng = np.random.default_rng(7)
rotacoes, f_verdade, T_verdade, normais = monta_sessao(rng)
resolvido = imu.solve_palm_alignment(rotacoes, normais)
check("solve_palm_alignment resolveu a sessao base", resolvido is not None)
T_fit, f_fit, residuo_fit = resolvido
check("residuo do alinhamento base ~0", residuo_fit < 1.0, residuo_fit)

for delta_boot in (0.0, 45.0, -120.0, 175.0):
    rot_boot = aplica_boot(rotacoes, delta_boot)
    resultado = imu.reancorar_yaw(T_fit, f_fit, rot_boot, normais)
    check(f"delta {delta_boot:+.0f}: reancorou", resultado is not None)
    if resultado is None:
        continue
    t_novo, delta, erro, n = resultado
    check(f"delta {delta_boot:+.0f}: delta recuperado ({delta:+.1f})",
          abs(delta - delta_boot) <= 1.0, delta)
    check(f"delta {delta_boot:+.0f}: erro medio baixo ({erro:.2f})",
          erro < 2.0, erro)
    # O T que vale no boot novo e' T_salvo Rz(-delta) (o mundo do IMU girou):
    # ele NAO e' igual ao T verdadeiro da sessao original quando delta != 0.
    esperado = T_fit @ imu.rotation_matrix(0.0, 0.0, -delta_boot)
    check(f"delta {delta_boot:+.0f}: T novo = T_salvo Rz(-delta)",
          angulo_rotacao(t_novo, esperado) < 0.5
          and abs(float(np.linalg.det(t_novo)) - 1.0) < 1e-6)
    # fusao usa o T novo e preve a verdade
    fusao = imu.PositionFusion()
    fusao.set_palm_alignment(T_fit, f_fit, residuo_fit)
    antes = erro_seta(fusao, rot_boot, normais)
    fusao.set_palm_alignment(t_novo, f_fit, residuo_fit)
    depois = erro_seta(fusao, rot_boot, normais)
    if abs(delta_boot) > 5.0:
        check(f"delta {delta_boot:+.0f}: seta antes {antes:.1f} -> depois "
              f"{depois:.2f} graus", antes > 20.0 and depois < 2.0)
    else:
        check(f"delta {delta_boot:+.0f}: sem mudanca de yaw a seta ja' estava "
              f"boa ({antes:.1f} -> {depois:.2f})",
              antes < 2.0 and depois < 2.0)

# ============================================================================
# 2) ruido na normal: recupera o delta com folga
# ============================================================================
print("2) recupera o delta mesmo com ruido na normal do Kinect")
rng = np.random.default_rng(11)
rotacoes, _f, _T, normais = monta_sessao(rng)
resolvido = imu.solve_palm_alignment(rotacoes, normais)
T_fit, f_fit, _res = resolvido
ruidosas = []
for v in normais:
    n = np.asarray(v) + rng.normal(0.0, math.radians(2.5), 3)
    ruidosas.append(n / np.linalg.norm(n))
rot_boot = aplica_boot(rotacoes, 70.0)
resultado = imu.reancorar_yaw(T_fit, f_fit, rot_boot, ruidosas)
check("com ruido: reancorou", resultado is not None)
if resultado is not None:
    _t, delta, erro, _n = resultado
    check("com ruido: delta dentro de 3 graus", abs(delta - 70.0) <= 3.0, delta)
    check("com ruido: erro medio < 8 graus", erro < 8.0, erro)

# ============================================================================
# 3) pose degenerada (palma ~vertical): None, nunca chute
# ============================================================================
print("3) palma apoiada/vertical o tempo todo = yaw nao observavel")
f_vertical = np.array([0.0, 0.0, -1.0])
rotacoes = [imu.rotation_matrix(2.0, 1.0, y) for y in (0.0, 90.0, 180.0)]
normais = [np.array([0.0, 0.0, 1.0]) for _ in rotacoes]
resultado = imu.reancorar_yaw(np.eye(3), f_vertical, rotacoes, normais)
check("palma vertical: devolve None", resultado is None, resultado)
# e com sensibilidade baixa mas nao nula, respeita o parametro
resultado = imu.reancorar_yaw(np.eye(3), f_vertical, rotacoes, normais,
                              sensibilidade_min=0.0)
check("sensibilidade forcada a 0: ainda devolve algo coerente",
      resultado is None or np.isfinite(resultado[1]))

# ============================================================================
# 4) AncoradorYawBoot: aplica na fusao, nao repete e respeita o timeout
# ============================================================================
print("4) AncoradorYawBoot: aplica na fusao, nao ancora duas vezes, timeout")
rng = np.random.default_rng(23)
rotacoes, _f, T_verdade, normais = monta_sessao(rng)
T_fit, f_fit, residuo_fit = imu.solve_palm_alignment(rotacoes, normais)
rot_boot = aplica_boot(rotacoes, 130.0)
fusao = imu.PositionFusion()
fusao.set_palm_alignment(T_fit, f_fit, residuo_fit)
antes = erro_seta(fusao, rot_boot, normais)
avisos = []
ancora = imu.AncoradorYawBoot(fusao, lado="right", amostras_min=8,
                              timeout_s=5.0, avisar=avisos.append)
resultado = None
for i, (r, v) in enumerate(zip(rot_boot, normais)):
    resultado = ancora.amostra(r, v)
    if resultado is not None:
        break
check("ancorou depois de juntar amostras", resultado is not None, i)
check("avisou o delta", any("reancorado" in aviso for aviso in avisos), avisos)
depois = erro_seta(fusao, rot_boot, normais)
check(f"fusao corrigida: antes {antes:.1f} -> depois {depois:.2f} graus",
      antes > 20.0 and depois < 2.0)
check("nao ancora de novo (devolve None depois de pronto)",
      ancora.amostra(rot_boot[0], normais[0]) is None)
check("segue 'pronto'", ancora.pronto is True)
check("delta guardado", ancora.delta_deg is not None and abs(
    ancora.delta_deg - 130.0) <= 1.0, ancora.delta_deg)

# timeout: pose vertical para sempre -> falha avisando, sem chutar
tempo = {"agora": 0.0}
original_time = imu.time


class _TempoFake:
    @staticmethod
    def monotonic():
        return tempo["agora"]


imu.time = _TempoFake
try:
    fusao2 = imu.PositionFusion()
    fusao2.set_palm_alignment(T_fit, f_fit, residuo_fit)
    avisos2 = []
    ancora2 = imu.AncoradorYawBoot(fusao2, lado="left", amostras_min=4,
                                   timeout_s=2.0, avisar=avisos2.append)
    for _ in range(6):
        tempo["agora"] += 1.0
        ancora2.amostra(rotacoes[0], np.array([0.0, 0.0, 1.0]))
    check("timeout: marca falhou", ancora2.falhou is True)
    check("timeout: avisa uma vez", any("nao consegui reancorar" in aviso
                                        for aviso in avisos2), avisos2)
finally:
    imu.time = original_time

# ============================================================================
# 5) integracao no tool (import, hook antes da seta, flag e delta na mao)
# ============================================================================
print("5) integracao no tool esta' ligada")
fonte_tool = open("tools/kinect_groundtruth_tool.py", encoding="utf-8").read()
check("tool importa AncoradorYawBoot", "AncoradorYawBoot" in fonte_tool)
check("tool tem flag --sem-ancora-yaw", "--sem-ancora-yaw" in fonte_tool)
check("tool alimenta o ancorador ANTES da seta",
      fonte_tool.index("ancora_yaw.amostra(")
      < fonte_tool.index('mao["palm_direction"] = ('))
check("tool publica yaw_boot_delta_deg", "yaw_boot_delta_deg" in fonte_tool)
check("imu tem as duas APIs",
      hasattr(imu, "reancorar_yaw") and hasattr(imu, "AncoradorYawBoot"))

if FALHAS:
    print()
    print(f"REANCORA_YAW_FALHOU: {len(FALHAS)} verificacao(oes): "
          + " | ".join(FALHAS))
    raise SystemExit(1)
print()
print("REANCORA_YAW_OK: yaw do boot recuperado (4 deltas, inclusive >90) | ruido "
      "de normal tolerado | pose vertical = None (sem chute) | AncoradorYawBoot "
      "aplica na fusao, nao repete e respeita o timeout | integracao no tool "
      "ligada")
