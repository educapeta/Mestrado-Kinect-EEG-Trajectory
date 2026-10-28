#!/usr/bin/env python3
"""
Testes sintéticos para palm_normal_edge_on.py
Cobre: plano com ruído, rotação rígida, edge-on, pontos colineares/ruins,
sinal right vs left, outlier de fundo.
"""
import sys
import numpy as np

from palm_normal_edge_on import (
    condicao_plano,
    normal_por_pontos,
    normal_mao_landmarks,
    normal_robusta_edge_on,
)


def assert_close(a, b, tol=1e-6, msg=""):
    if not np.allclose(a, b, atol=tol):
        raise AssertionError(f"{msg}: {a} != {b} (tol={tol})")


def assert_angle_close(v1, v2, max_deg=5.0, msg=""):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.degrees(np.arccos(abs(dot)))
    if angle > max_deg:
        raise AssertionError(f"{msg}: angulo {angle:.2f} deg > {max_deg} deg")


def test_condicao_plano():
    print("Teste: condicao_plano")
    # Plano bem condicionado
    r = condicao_plano(np.array([10.0, 1.0, 0.1]))
    assert r["qualidade"] > 0.05
    assert not r["mau_condicionado"]

    # Plano mal condicionado (qualidade baixa): razao_23/razao_12 pequeno
    # lambda1=1000, lambda2=10, lambda3=0.1 -> razao_12=100, razao_23=100, qualidade=1
    # lambda1=100, lambda2=1, lambda3=0.001 -> razao_12=100, razao_23=1000, qualidade=10 -> clip 1
    # lambda1=10000, lambda2=100, lambda3=0.1 -> razao_12=100, razao_23=1000, qualidade=10 -> clip 1
    # Para qualidade < 0.05: razao_23/razao_12 < 0.05
    # Ex: lambda1=100, lambda2=10, lambda3=0.001 -> razao_12=10, razao_23=10000, qual=1000->1
    # Ex: lambda1=1000, lambda2=100, lambda3=0.1 -> razao_12=10, razao_23=1000, qual=100->1
    # Ex: lambda1=100, lambda2=1, lambda3=0.5 -> razao_12=100, razao_23=2, qual=0.02 < 0.05
    r = condicao_plano(np.array([100.0, 1.0, 0.5]))
    assert r["qualidade"] < 0.05
    assert r["mau_condicionado"]

    # razao_12 > 100
    r = condicao_plano(np.array([200.0, 1.0, 0.5]))
    assert r["mau_condicionado"]
    print("  OK")


def test_plano_com_ruido():
    print("Teste: plano com ruído")
    # Cria plano XY com ruído pequeno
    np.random.seed(42)
    n = 20
    pontos = {}
    for i in range(n):
        x = np.random.uniform(-0.05, 0.05)
        y = np.random.uniform(-0.04, 0.04)
        z = np.random.normal(0, 0.001)  # ruído 1mm
        pontos[f"p{i}"] = np.array([x, y, z])

    res = normal_por_pontos(pontos, lado="right")
    assert res is not None
    assert res["normal"] is not None
    assert res["confiavel"] is True
    assert res["n_pontos"] == n
    # Normal deve ser aproximadamente [0, 0, 1] ou [0, 0, -1]
    assert_angle_close(res["normal"], np.array([0.0, 0.0, 1.0]), max_deg=5.0, msg="normal plano XY")
    # Sinal correto para right hand (sem referência anatômica, usa cross)
    # cross(index-pinky, middle-wrist) no plano XY aponta para +Z ou -Z
    print(f"  normal={res['normal']}, rms={res['rms_m']:.6f}, qualidade={res['qualidade']:.4f}")
    print("  OK")


def test_rotacao_rigida():
    print("Teste: rotação rígida")
    # Pontos no plano XY COM PEQUENA ESPESSURA (ruído em z)
    np.random.seed(123)
    n = 15
    base_pontos = {}
    for i in range(n):
        x = np.random.uniform(-0.05, 0.05)
        y = np.random.uniform(-0.04, 0.04)
        z = np.random.normal(0, 0.0005)  # pequena espessura
        base_pontos[f"p{i}"] = np.array([x, y, z])

    # Rotação de 45° em torno do eixo X
    theta = np.radians(45)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    pontos_rot = {k: Rx @ v for k, v in base_pontos.items()}
    res_base = normal_por_pontos(base_pontos, lado="right")
    res_rot = normal_por_pontos(pontos_rot, lado="right")

    assert res_base["normal"] is not None, f"Base falhou: {res_base['motivo']}"
    assert res_rot["normal"] is not None, f"Rot falhou: {res_rot['motivo']}"
    # Normal rotacionada deve ser igual à normal base rotacionada
    normal_esperada = Rx @ res_base["normal"]
    assert_angle_close(res_rot["normal"], normal_esperada, max_deg=2.0, msg="rotacao rigida")
    print(f"  normal_base={res_base['normal']}, normal_rot={res_rot['normal']}")
    print("  OK")


def test_edge_on_21_vs_5():
    print("Teste: edge-on (21 pontos vs 5 pontos)")
    # Cria uma mão "edge-on": pontos aproximadamente no plano YZ (fino em X)
    # 21 landmarks estilo MediaPipe
    np.random.seed(456)
    landmarks_21 = {}
    # Wrist
    landmarks_21["0"] = np.array([0.001, 0.0, 0.0])
    # Dedos - 4 pontos cada (MCP, PIP, DIP, TIP)
    # Index (1-4)
    landmarks_21["1"] = np.array([0.002, 0.02, 0.01])
    landmarks_21["2"] = np.array([0.002, 0.04, 0.02])
    landmarks_21["3"] = np.array([0.002, 0.05, 0.03])
    landmarks_21["4"] = np.array([0.002, 0.06, 0.04])
    # Middle (5-8)
    landmarks_21["5"] = np.array([0.001, 0.02, -0.01])
    landmarks_21["6"] = np.array([0.001, 0.045, -0.02])
    landmarks_21["7"] = np.array([0.001, 0.06, -0.025])
    landmarks_21["8"] = np.array([0.001, 0.07, -0.03])
    # Ring (9-12)
    landmarks_21["9"] = np.array([0.001, 0.02, -0.03])
    landmarks_21["10"] = np.array([0.001, 0.04, -0.04])
    landmarks_21["11"] = np.array([0.001, 0.05, -0.045])
    landmarks_21["12"] = np.array([0.001, 0.06, -0.05])
    # Pinky (13-16)
    landmarks_21["13"] = np.array([0.002, 0.015, -0.045])
    landmarks_21["14"] = np.array([0.002, 0.03, -0.05])
    landmarks_21["15"] = np.array([0.002, 0.04, -0.052])
    landmarks_21["16"] = np.array([0.002, 0.05, -0.055])
    # Thumb (17-20)
    landmarks_21["17"] = np.array([-0.001, -0.01, 0.02])
    landmarks_21["18"] = np.array([-0.002, -0.02, 0.03])
    landmarks_21["19"] = np.array([-0.002, -0.025, 0.035])
    landmarks_21["20"] = np.array([-0.003, -0.03, 0.04])

    # Adiciona ruído pequeno
    for k in landmarks_21:
        landmarks_21[k] += np.random.normal(0, 0.0005, 3)

    res_21 = normal_mao_landmarks(landmarks_21, lado="right")
    print(f"  21 pts: n={res_21['n_pontos']}, qualidade={res_21['qualidade']:.4f}, confiavel={res_21['confiavel']}, fonte={res_21['fonte']}")

    # Fallback 5 pontos (dorso)
    dorsom_keys = ["wrist", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"]
    dorsom_idx = ["0", "5", "9", "13", "17"]
    pontos_5 = {}
    for nome, idx in zip(dorsom_keys, dorsom_idx):
        if idx in landmarks_21:
            pontos_5[nome] = landmarks_21[idx]

    res_5 = normal_por_pontos(pontos_5, lado="right")
    print(f"  5 pts: n={res_5['n_pontos']}, qualidade={res_5['qualidade']:.4f}, confiavel={res_5['confiavel']}")

    # 21 pontos devem funcionar melhor ou igual
    assert res_21["normal"] is not None
    assert res_5["normal"] is not None
    # Qualidade com 21 pontos deve ser >= qualidade com 5 pontos (ou similar)
    assert res_21["qualidade"] >= res_5["qualidade"] * 0.8  # tolerância
    # Confiável coerente
    assert res_21["confiavel"] == (res_21["qualidade"] >= 0.05)
    assert res_5["confiavel"] == (res_5["qualidade"] >= 0.05)
    print("  OK")


def test_pontos_colineares_ruins():
    print("Teste: pontos colineares/ruins")
    # Pontos em linha reta (colineares)
    pontos_linha = {
        "p0": np.array([0.0, 0.0, 0.0]),
        "p1": np.array([0.01, 0.0, 0.0]),
        "p2": np.array([0.02, 0.0, 0.0]),
        "p3": np.array([0.03, 0.0, 0.0]),
    }
    res = normal_por_pontos(pontos_linha)
    assert res["normal"] is None
    assert res["confiavel"] is False
    print(f"  colineares: normal={res['normal']}, confiavel={res['confiavel']}, motivo={res['motivo']}")

    # Pontos com NaN
    pontos_nan = {
        "p0": np.array([0.0, 0.0, 0.0]),
        "p1": np.array([np.nan, 0.0, 0.0]),
        "p2": np.array([0.0, 0.01, 0.0]),
        "p3": np.array([0.0, 0.0, 0.01]),
    }
    res = normal_por_pontos(pontos_nan)
    assert res["normal"] is None
    assert res["confiavel"] is False
    print(f"  NaN: normal={res['normal']}, confiavel={res['confiavel']}")

    # Menos de 4 pontos
    pontos_poucos = {
        "p0": np.array([0.0, 0.0, 0.0]),
        "p1": np.array([0.01, 0.0, 0.0]),
        "p2": np.array([0.0, 0.01, 0.0]),
    }
    res = normal_por_pontos(pontos_poucos)
    assert res["normal"] is None
    assert res["confiavel"] is False
    print(f"  poucos: normal={res['normal']}, confiavel={res['confiavel']}")

    # Teste normal_mao_landmarks com entrada ruim (pontos inválidos)
    landmarks_ruins = {
        "0": np.array([np.nan, 0.0, 0.0]),
        "5": np.array([0.0, 0.0, 0.0]),
    }
    res = normal_mao_landmarks(landmarks_ruins)
    assert res["confiavel"] is False
    assert res["normal"] is None
    print(f"  landmarks ruins: confiavel={res['confiavel']}")

    # Sem exceção
    print("  OK")


def test_sinal_right_vs_left():
    print("Teste: sinal right vs left")
    # Plano XY com PEQUENA ESPESSURA em Z
    pontos = {
        "wrist": np.array([0.0, 0.0, 0.001]),
        "index_mcp": np.array([0.04, 0.0, -0.001]),
        "middle_mcp": np.array([0.0, 0.04, 0.002]),
        "pinky_mcp": np.array([-0.04, 0.0, -0.002]),
    }
    res_right = normal_por_pontos(pontos, lado="right")
    res_left = normal_por_pontos(pontos, lado="left")

    assert res_right["normal"] is not None
    assert res_left["normal"] is not None
    # Devem ser opostas
    dot = np.dot(res_right["normal"], res_left["normal"])
    assert dot < -0.99, f"Esperava opostas, dot={dot}"
    print(f"  right: {res_right['normal']}, left: {res_left['normal']}, dot={dot:.4f}")
    print("  OK")


def test_outlier_fundo():
    print("Teste: outlier de fundo (>0.12m)")
    np.random.seed(789)
    # Pontos válidos no plano XY centrados na origem
    pontos = {}
    for i in range(10):
        x = np.random.uniform(-0.03, 0.03)
        y = np.random.uniform(-0.03, 0.03)
        z = np.random.normal(0, 0.001)
        pontos[f"p{i}"] = np.array([x, y, z])

    # Adiciona outlier longe (>0.12m do centro)
    pontos["outlier"] = np.array([0.5, 0.5, 0.5])  # ~0.86m do centro

    res = normal_por_pontos(pontos)
    # Outlier deve ser rejeitado, n_pontos = 10
    assert res["n_pontos"] == 10
    assert res["normal"] is not None
    assert res["confiavel"] is True
    print(f"  n_pontos={res['n_pontos']} (esperado 10), normal={res['normal']}")
    print("  OK")


def test_normal_robusta_edge_on():
    print("Teste: normal_robusta_edge_on")
    # Cria landmarks válidos
    landmarks = {
        "0": np.array([0.0, 0.0, 0.0]),
        "5": np.array([0.04, 0.0, 0.0]),
        "9": np.array([0.0, 0.04, 0.0]),
        "13": np.array([-0.03, 0.0, -0.02]),
        "17": np.array([-0.04, 0.0, 0.02]),
    }
    # Adiciona mais pontos
    for i in range(1, 5):
        landmarks[str(i)] = np.array([0.04 * i/4, 0.0, 0.0])
    for i in range(6, 9):
        landmarks[str(i)] = np.array([0.0, 0.04 * (i-5)/3, 0.0])

    res = normal_robusta_edge_on(landmarks, lado="right")
    assert res["normal"] is not None
    assert res["confiavel"] is True
    print(f"  normal={res['normal']}, qualidade={res['qualidade']:.4f}, confiavel={res['confiavel']}")

    # Teste alinhamento com normal_anterior
    normal_ant = np.array([0.0, 0.0, 1.0])
    res2 = normal_robusta_edge_on(landmarks, lado="right", normal_anterior=normal_ant)
    # Sinal deve estar alinhado com normal_anterior
    dot = np.dot(res2["normal"], normal_ant)
    assert dot > 0, f"Normal não alinhada com anterior: dot={dot}"
    print(f"  com anterior: normal={res2['normal']}, dot={dot:.4f}")
    print("  OK")


def main():
    print("=" * 60)
    print("INICIANDO TESTES palm_normal_edge_on")
    print("=" * 60)

    all_ok = True
    tests = [
        test_condicao_plano,
        test_plano_com_ruido,
        test_rotacao_rigida,
        test_edge_on_21_vs_5,
        test_pontos_colineares_ruins,
        test_sinal_right_vs_left,
        test_outlier_fundo,
        test_normal_robusta_edge_on,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  FALHOU: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("PALM_NORMAL_EDGE_ON_OK: todos os testes passaram")
        return 0
    else:
        print("PALM_NORMAL_EDGE_ON_OK: ALGUNS TESTES FALHARAM")
        return 1


if __name__ == "__main__":
    sys.exit(main())