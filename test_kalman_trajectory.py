"""Teste do filtro de Kalman de trajetoria (velocidade da EEG + aceleracao IMU).

Motivo (decisao de 18/09): o modelo passa a prever VELOCIDADE (`--alvo
velocidade`). Reconstruir a posicao integrando aceleracao duas vezes faz o erro
crescer com t^2 e explodir com o bias do acelerometro; com o filtro de Kalman o
bias entra no estado e a velocidade decodificada da EEG ancora a integracao, de
modo que o erro fica limitado -- e a aceleracao do MPU6050 entra como medida
direta (melhor que posicao, cujo erro de integracao e' quadratico).

Casos:
1. referencia da aceleracao: 1 g no eixo vertical -> 0; 2 g -> +9,807 m/s^2;
   atitude de pitch 90 graus gira a leitura para o eixo x; bias e' removido;
2. integracao trapezoidal de velocidade (e erros de entrada);
3. punho PARADO com medidas zero (o caso do trial IDLE) -> nao deriva;
4. velocidade constante -> a posicao segue v*t;
5. **erro quadratico**: aceleracao com bias SEM medidas de velocidade cresce com
   t^2 (2 cm em 2 s e 2,5 m em 10 s com 5 cm/s^2 de bias); o MESMO bias COM o
   filtro + velocidade da EEG fica limitado (< 3 cm) e o bias e' estimado;
6. suavizacao: a velocidade estimada e' mais proxima da verdade que a medida
   ruidosa;
7. degrau de velocidade: o filtro acompanha em menos de 1 s;
8. predicao a frente (overlay): com velocidade constante a trajetoria prevista e'
   linear e termina em v*horizonte.
"""
import sys

import numpy as np

sys.path.insert(0, ".")

from imu import (GRAVITY_MPS2, UP_AXIS, ImuSample, KalmanTrajectory,
                 accel_no_referencial, integra_velocidade)


def amostra(accel_g, roll=0.0, pitch=0.0, yaw=0.0, t_us=0):
    return ImuSample(timestamp_us=t_us, received_at=0.0, roll_deg=roll,
                     pitch_deg=pitch, yaw_deg=yaw,
                     accel_g=np.asarray(accel_g, float),
                     gyro_dps=np.zeros(3))


# 1) referencia da aceleracao
parado = accel_no_referencial(amostra([0.0, 0.0, 1.0]))
assert np.abs(parado).max() < 1e-9, parado
dobro = accel_no_referencial(amostra([0.0, 0.0, 2.0]))
assert abs(dobro[UP_AXIS] - GRAVITY_MPS2) < 1e-9, dobro
girado = accel_no_referencial(amostra([0.0, 0.0, 1.0], pitch=90.0))
assert abs(girado[0] - GRAVITY_MPS2) < 1e-6, girado
assert abs(girado[UP_AXIS] + GRAVITY_MPS2) < 1e-6, girado
com_bias = accel_no_referencial(amostra([0.0, 0.0, 1.0]),
                                bias_g=[0.0, 0.0, 0.05])
assert abs(com_bias[UP_AXIS] + 0.05 * GRAVITY_MPS2) < 1e-6, com_bias
assert accel_no_referencial(None) is None
print("1) OK: referencia da aceleracao (1 g -> 0; 2 g -> 9,807; pitch 90 graus "
      "-> eixo x; bias removido)")

# 2) integracao trapezoidal
posicoes = integra_velocidade(np.full((3, 3), 0.2), 0.5)
assert np.allclose(posicoes, [[0, 0, 0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
assert integra_velocidade(np.full((1, 3), 0.2), 0.5).shape == (1, 3)
for ruim, erro in (((np.zeros(3), 0.5), ValueError),
                   ((np.zeros((2, 2)), 0.5), ValueError),
                   ((np.zeros((2, 3)), 0.0), ValueError)):
    try:
        integra_velocidade(ruim[0], ruim[1])
        raise AssertionError(f"aceitou entrada invalida: {ruim}")
    except erro:
        pass
print("2) OK: integracao trapezoidal de velocidade (0,2 m/s -> 0,1/0,2 m) e "
      "validacao de entrada")


def simula(filtro, passos, dt=0.05, accel=None, verdade_velocidade=0.0,
           sigma_medida=None, rng=None, medida_a_cada=1):
    """Roda o filtro com aceleracao medida; `sigma_medida=None` = sem velocidade."""
    rng = np.random.default_rng(0) if rng is None else rng
    for passo in range(passos):
        medida = None
        if sigma_medida is not None and passo % medida_a_cada == 0:
            ruido = (np.zeros(3) if sigma_medida == 0
                     else rng.normal(0, sigma_medida, 3))
            medida = np.full(3, float(verdade_velocidade)) + ruido
        filtro.step(dt, accel_m_s2=accel, velocidade_m_s=medida)
    return filtro


# 3) punho parado (trial IDLE): nada deriva
filtro = simula(KalmanTrajectory(), 200, accel=np.zeros(3),
                verdade_velocidade=0.0, sigma_medida=0.01)
desvio = float(np.abs(filtro.position).max())
assert desvio < 0.03, desvio
print(f"3) OK: punho parado com medidas zero (IDLE) por 10 s -> desvio "
      f"{desvio * 1000:.1f} mm (nao deriva)")

# 4) velocidade constante
filtro = simula(KalmanTrajectory(), 40, dt=0.05,
                accel=np.zeros(3), verdade_velocidade=0.2, sigma_medida=0)
esperado = 0.2 * 40 * 0.05
assert abs(filtro.position[0] - esperado) < 0.01, (filtro.position, esperado)
# 5) o argumento do ERRO QUADRATICO: bias do acelerometro sem e com o filtro
#
# Lei do erro (e' so' a integracao): um erro CONSTANTE na ACELERACAO (bias eps)
# vira erro LINEAR na velocidade (eps*t) e QUADRATICO na POSICAO (0.5*eps*t^2).
# O erro esta' na POSICAO -- a aceleracao nao melhora nem piora; quem amplifica e'
# a dupla integracao.
bias = 0.05                       # m/s^2 (5 mg: tipico de MPU6050 sem zeragem)
dt = 0.02
horizonte = 10.0
# (a) integracao dupla NUMERICA do acelerometro com bias, sem nenhum filtro
posicao_dupla = np.zeros(3)
velocidade_dupla = np.zeros(3)
for _ in range(int(horizonte / dt)):
    velocidade_dupla += np.full(3, bias) * dt
    posicao_dupla += velocidade_dupla * dt
naive = {t: 0.5 * bias * t ** 2 for t in (2.0, 5.0, 10.0)}
# A integracao NUMERICA reproduz a lei fisica 0.5*eps*t^2 mais o termo de
# discretizacao do esquema de Euler (0.5*eps*t*dt, aqui 5 mm) -- a lei fisica e'
# o limite dt -> 0.
esperado_euler = 0.5 * bias * horizonte ** 2 + 0.5 * bias * horizonte * dt
assert abs(float(posicao_dupla[0]) - esperado_euler) < 1e-9, \
    (posicao_dupla, esperado_euler)
assert abs(float(posicao_dupla[0]) - naive[10.0]) < 0.01, posicao_dupla
# e o erro cresce com t^2: 5x mais tempo -> 25x mais erro
assert abs(naive[10.0] / naive[2.0] - 25.0) < 1e-9, naive
# (b) o MESMO bias com o filtro + velocidade decodificada a cada 0,5 s
filtro = KalmanTrajectory()
rng = np.random.default_rng(1)
for passo in range(int(horizonte / dt)):
    medida = (rng.normal(0.0, 0.03, 3)
              if passo % int(0.5 / dt) == 0 else None)         # verdade = 0 m/s
    filtro.step(dt, accel_m_s2=np.full(3, bias), velocidade_m_s=medida)
erro_filtro = float(np.abs(filtro.position).max())
assert naive[2.0] > 0.09 and naive[10.0] > 2.4, naive
assert naive[10.0] / naive[2.0] > 20, naive        # crescimento quadratico
assert erro_filtro < 0.10, erro_filtro
assert erro_filtro < naive[10.0] / 20, (erro_filtro, naive[10.0])
assert abs(float(filtro.bias.mean()) - bias) < 0.02, filtro.bias
print(f"5) OK: erro quadratico do bias {bias} m/s^2 -> integracao dupla: "
      f"{naive[2.0] * 100:.1f} cm em 2 s e {naive[10.0]:.2f} m em 10 s "
      f"(x25 do erro em 5x de tempo = t^2); com o filtro + velocidade da EEG: "
      f"{erro_filtro * 100:.1f} cm em 10 s = {naive[10.0] / erro_filtro:.0f}x "
      f"menor (bias estimado {float(filtro.bias.mean()):.3f} m/s^2)")

# 6) suavizacao da velocidade medida
filtro = KalmanTrajectory()
rng = np.random.default_rng(2)
medidas, estimativas = [], []
for passo in range(200):
    valor = 0.3 + rng.normal(0, 0.1)
    medidas.append(valor)
    filtro.step(0.05, accel_m_s2=np.zeros(3), velocidade_m_s=np.full(3, valor))
    estimativas.append(float(filtro.velocity[0]))
rmse_medida = float(np.sqrt(np.mean((np.array(medidas) - 0.3) ** 2)))
rmse_filtro = float(np.sqrt(np.mean((np.array(estimativas)[50:] - 0.3) ** 2)))
assert rmse_filtro < 0.5 * rmse_medida, (rmse_medida, rmse_filtro)
print(f"6) OK: suavizacao da velocidade: RMSE {rmse_medida:.3f} m/s (medida) "
      f"-> {rmse_filtro:.3f} m/s (filtro), {rmse_medida / rmse_filtro:.1f}x "
      "menor")

# 7) degrau de velocidade: acompanha em ~2 s
filtro = KalmanTrajectory()
for _ in range(20):
    filtro.step(0.05, accel_m_s2=np.zeros(3), velocidade_m_s=np.zeros(3))
for _ in range(40):
    filtro.step(0.05, accel_m_s2=np.zeros(3), velocidade_m_s=np.full(3, 0.3))
assert float(filtro.velocity[0]) > 0.9 * 0.3, filtro.velocity
print(f"7) OK: degrau de 0 -> 0,3 m/s: a velocidade filtrada chegou a "
      f"{float(filtro.velocity[0]):.3f} m/s em 2 s")

# 8) predicao a frente (trajetoria do overlay): reta com aceleracao estimada ~0
filtro = KalmanTrajectory()
for _ in range(40):
    filtro.step(0.05, accel_m_s2=np.zeros(3), velocidade_m_s=np.full(3, 0.3))
assert np.abs(filtro.acceleration_est).max() < 0.05, filtro.acceleration_est
previsao = filtro.predizer(1.0, passos=11)
assert previsao.shape == (11, 3), previsao.shape
assert np.allclose(previsao[0], filtro.position, atol=1e-9)
# contrato do estado: p(t) = p0 + v0*t + 0.5*a*t^2
final = (filtro.position + filtro.velocity * 1.0
         + 0.5 * filtro.acceleration_est * 1.0 ** 2)
assert np.allclose(previsao[-1], final, atol=1e-9), (previsao[-1], final)
deslocamento = float(previsao[-1][0] - filtro.position[0])
assert deslocamento > 0.25, deslocamento
print(f"8) OK: previsao 1 s a frente = p0 + v*t + 0,5*a*t^2 "
      f"(v={float(filtro.velocity[0]):.3f} m/s, a="
      f"{float(filtro.acceleration_est[0]):+.3f} m/s^2 -> "
      f"{deslocamento:.3f} m em 1 s)")

print("KALMAN_TRAJ_OK: 8/8 casos (referencia da aceleracao, integracao, IDLE sem "
      "deriva, velocidade constante, erro quadratico do bias, suavizacao, "
      "degrau, predicao)")
sys.exit(0)
