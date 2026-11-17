"""
Testes para palm_orientacao.py - estilo flat com helper check.
"""

import sys
import math
import numpy as np
from palm_orientacao import (
    angulo_dobrado,
    qa_efetivo,
    EstabilizadorSinal,
    corrigir_normal_para_referencia
)


def check(cond: bool, msg: str) -> bool:
    """Helper de asserção: imprime OK/FAIL e retorna bool."""
    if cond:
        print(f"  OK: {msg}")
        return True
    else:
        print(f"  FAIL: {msg}")
        return False


def test_angulo_dobrado() -> bool:
    """Testa angulo_dobrado com vários casos."""
    print("\n=== test_angulo_dobrado ===")
    all_ok = True

    all_ok &= check(angulo_dobrado(0) == 0.0, "angulo 0 -> 0")
    all_ok &= check(angulo_dobrado(90) == 90.0, "angulo 90 -> 90")
    all_ok &= check(angulo_dobrado(170) == 10.0, "angulo 170 -> 10 (dobrado)")
    all_ok &= check(angulo_dobrado(180) == 0.0, "angulo 180 -> 0")
    all_ok &= check(angulo_dobrado(None) is None, "None -> None")
    all_ok &= check(angulo_dobrado(float('nan')) is None, "NaN -> None")
    all_ok &= check(angulo_dobrado(45) == 45.0, "angulo 45 -> 45")
    all_ok &= check(angulo_dobrado(135) == 45.0, "angulo 135 -> 45 (dobrado)")
    all_ok &= check(angulo_dobrado(360) == 0.0, "angulo 360 -> 0 (normalizado)")
    all_ok &= check(angulo_dobrado(270) == 90.0, "angulo 270 -> 90 (normalizado)")

    return all_ok


def test_qa_efetivo() -> bool:
    """Testa qa_efetivo: geometrica, posicao, degenerados."""
    print("\n=== test_qa_efetivo ===")
    all_ok = True

    # Vetores de teste
    n_z = np.array([0.0, 0.0, 1.0])
    n_neg_z = np.array([0.0, 0.0, -1.0])
    n_x = np.array([1.0, 0.0, 0.0])
    n_45 = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    zero = np.array([0.0, 0.0, 0.0])

    # --- Geométrica ---
    # Mesma direção: aprovado
    r = qa_efetivo(n_z, n_z, "geometrica", 10.0, 10.0)
    all_ok &= check(r["aprovado"] is True, "geometrica: mesma dir -> aprovado")
    all_ok &= check(abs(r["bruto_deg"] - 0.0) < 1e-6, "geometrica: bruto 0")
    all_ok &= check(abs(r["efetivo_deg"] - 0.0) < 1e-6, "geometrica: efetivo 0")

    # Invertida (180°): REJEITADA no modo geometrica
    r = qa_efetivo(n_z, n_neg_z, "geometrica", 10.0, 10.0)
    all_ok &= check(r["aprovado"] is False, "geometrica: invertida -> rejeitado")
    all_ok &= check(abs(r["bruto_deg"] - 180.0) < 1e-6, "geometrica: bruto 180")
    all_ok &= check(r["motivo"] == "geometrica_acima_limite", "geometrica: motivo correto")

    # 45°: aprovado se limite > 45
    r = qa_efetivo(n_z, n_45, "geometrica", 50.0, 10.0)
    all_ok &= check(r["aprovado"] is True, "geometrica: 45° com limite 50 -> aprovado")
    r = qa_efetivo(n_z, n_45, "geometrica", 30.0, 10.0)
    all_ok &= check(r["aprovado"] is False, "geometrica: 45° com limite 30 -> rejeitado")

    # --- Posição ---
    # Mesma direção: aprovado
    r = qa_efetivo(n_z, n_z, "posicao", 10.0, 10.0)
    all_ok &= check(r["aprovado"] is True, "posicao: mesma dir -> aprovado")

    # Invertida (180° -> dobrado 0°): ACEITA no modo posicao
    r = qa_efetivo(n_z, n_neg_z, "posicao", 10.0, 10.0)
    all_ok &= check(r["aprovado"] is True, "posicao: invertida -> aprovado (dobrado 0)")
    all_ok &= check(abs(r["bruto_deg"] - 180.0) < 1e-6, "posicao: bruto 180")
    all_ok &= check(abs(r["efetivo_deg"] - 0.0) < 1e-6, "posicao: efetivo 0 (dobrado)")

    # 170° -> dobrado 10°: aprovado se limite >= 10
    n_170 = np.array([np.sin(np.radians(10)), 0.0, -np.cos(np.radians(10))])
    r = qa_efetivo(n_z, n_170, "posicao", 10.0, 15.0)
    all_ok &= check(r["aprovado"] is True, "posicao: 170° (dobrado 10) com limite 15 -> aprovado")
    r = qa_efetivo(n_z, n_170, "posicao", 10.0, 5.0)
    all_ok &= check(r["aprovado"] is False, "posicao: 170° (dobrado 10) com limite 5 -> rejeitado")

    # --- Degenerados ---
    r = qa_efetivo(None, n_z, "geometrica", 10.0, 10.0)
    all_ok &= check(r["aprovado"] is False, "degenerado: normal None -> rejeitado")
    all_ok &= check(r["motivo"] == "entrada_none", "degenerado: motivo entrada_none")

    r = qa_efetivo(n_z, None, "geometrica", 10.0, 10.0)
    all_ok &= check(r["aprovado"] is False, "degenerado: alvo None -> rejeitado")

    r = qa_efetivo(zero, n_z, "geometrica", 10.0, 10.0)
    all_ok &= check(r["aprovado"] is False, "degenerado: norma zero -> rejeitado")
    all_ok &= check(r["motivo"] == "norma_zero_ou_nan", "degenerado: motivo norma_zero_ou_nan")

    r = qa_efetivo(n_z, zero, "geometrica", 10.0, 10.0)
    all_ok &= check(r["aprovado"] is False, "degenerado: alvo norma zero -> rejeitado")

    # NaN
    nan_vec = np.array([float('nan'), 0.0, 0.0])
    r = qa_efetivo(nan_vec, n_z, "geometrica", 10.0, 10.0)
    all_ok &= check(r["aprovado"] is False, "degenerado: NaN -> rejeitado")

    # Tipo inválido
    r = qa_efetivo(n_z, n_z, "invalido", 10.0, 10.0)
    all_ok &= check(r["aprovado"] is False, "tipo invalido -> rejeitado")
    all_ok &= check(r["motivo"] == "tipo_invalido", "tipo invalido: motivo correto")

    # Shape inválido
    r = qa_efetivo(np.array([1.0, 2.0]), n_z, "geometrica", 10.0, 10.0)
    all_ok &= check(r["aprovado"] is False, "shape invalido -> rejeitado")
    all_ok &= check(r["motivo"] == "shape_invalido", "shape invalido: motivo correto")

    return all_ok


def test_estabilizador_sinal() -> bool:
    """Testa EstabilizadorSinal: ruído não troca, virada real troca, reset."""
    print("\n=== test_estabilizador_sinal ===")
    all_ok = True

    est = EstabilizadorSinal(frames_para_trocar=8, tolerancia_deg=170.0)

    # Referência inicial: +Z
    ref = np.array([0.0, 0.0, 1.0])

    # --- Ruído de 10° NÃO deve trocar ---
    # Gera vetores com ruído de ±10° ao redor de +Z
    np.random.seed(42)
    for i in range(20):
        # Ruído pequeno: desvia no máximo 10°
        theta = np.random.uniform(-10, 10) * np.pi / 180
        phi = np.random.uniform(0, 2*np.pi)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        noisy = np.array([x, y, z])
        out = est.atualizar(noisy, float(i))
        # Deve manter sinal positivo (próximo a +Z)
        all_ok &= check(out[2] > 0, f"ruido frame {i}: sinal mantido positivo (z={out[2]:.3f})")

    all_ok &= check(est.frames_opostos == 0, "ruido: frames_opostos zerado")
    all_ok &= check(np.dot(est.referencia, ref) > 0.99, "ruido: referencia nao trocou")

    # --- Virada real: 8 frames consecutivos opostos (>= 170°) DEVE trocar ---
    # Vetor oposto a +Z (ou seja, -Z) com pequeno ruído
    for i in range(8):
        # -Z com ruído de 1° (ângulo ~179° em relação a +Z)
        theta = np.pi - np.random.uniform(0, 1) * np.pi / 180
        phi = np.random.uniform(0, 2*np.pi)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        opposite = np.array([x, y, z])
        out = est.atualizar(opposite, float(20 + i))

    # Após 8 frames opostos, a referência deve ter trocado para -Z
    all_ok &= check(est.frames_opostos == 0, "virada: frames_opostos resetado apos troca")
    all_ok &= check(np.dot(est.referencia, ref) < -0.99, "virada: referencia trocou para -Z")
    # Saída deve estar alinhada com nova referência (-Z)
    all_ok &= check(out[2] < 0, "virada: saida alinhada com nova referencia (-Z)")

    # --- Reset ---
    est.reset()
    all_ok &= check(est.referencia is None, "reset: referencia None")
    all_ok &= check(est.frames_opostos == 0, "reset: frames_opostos 0")

    # Nova referência após reset
    out = est.atualizar(ref, 100.0)
    all_ok &= check(out[2] > 0, "pos-reset: nova referencia +Z")

    return all_ok


def test_corrigir_normal_para_referencia() -> bool:
    """Testa corrigir_normal_para_referencia."""
    print("\n=== test_corrigir_normal_para_referencia ===")
    all_ok = True

    ref = np.array([0.0, 0.0, 1.0])

    # Mesmo sinal: retorna igual
    n = np.array([0.0, 0.0, 1.0])
    out = corrigir_normal_para_referencia(n, ref)
    all_ok &= check(out is not None, "mesmo sinal: nao None")
    all_ok &= check(np.allclose(out, n), "mesmo sinal: vetor inalterado")

    # Sinal oposto: inverte
    n = np.array([0.0, 0.0, -1.0])
    out = corrigir_normal_para_referencia(n, ref)
    all_ok &= check(out is not None, "sinal oposto: nao None")
    all_ok &= check(np.allclose(out, -n), "sinal oposto: invertido para +Z")

    # Perpendicular: dot = 0, não inverte (já que dot >= 0)
    n = np.array([1.0, 0.0, 0.0])
    out = corrigir_normal_para_referencia(n, ref)
    all_ok &= check(out is not None, "perpendicular: nao None")
    all_ok &= check(np.allclose(out, n), "perpendicular: inalterado (dot=0)")

    # None entries
    all_ok &= check(corrigir_normal_para_referencia(None, ref) is None, "normal None -> None")
    all_ok &= check(corrigir_normal_para_referencia(n, None) is None, "referencia None -> None")

    # Norma zero
    all_ok &= check(corrigir_normal_para_referencia(np.zeros(3), ref) is None, "norma zero normal -> None")
    all_ok &= check(corrigir_normal_para_referencia(n, np.zeros(3)) is None, "norma zero ref -> None")

    # NaN
    nan_vec = np.array([float('nan'), 0.0, 0.0])
    all_ok &= check(corrigir_normal_para_referencia(nan_vec, ref) is None, "NaN normal -> None")
    all_ok &= check(corrigir_normal_para_referencia(n, nan_vec) is None, "NaN ref -> None")

    # Shape inválido
    all_ok &= check(corrigir_normal_para_referencia(np.array([1.0, 2.0]), ref) is None, "shape invalido normal -> None")
    all_ok &= check(corrigir_normal_para_referencia(n, np.array([1.0, 2.0])) is None, "shape invalido ref -> None")

    return all_ok


def main() -> int:
    """Executa todos os testes e retorna código de saída."""
    print("Iniciando testes palm_orientacao...")

    all_ok = True
    all_ok &= test_angulo_dobrado()
    all_ok &= test_qa_efetivo()
    all_ok &= test_estabilizador_sinal()
    all_ok &= test_corrigir_normal_para_referencia()

    if all_ok:
        print("\nPALM_ORIENTACAO_OK: todos os testes passaram")
        return 0
    else:
        print("\nPALM_ORIENTACAO_OK: FALHOU - alguns testes falharam")
        return 1


if __name__ == "__main__":
    sys.exit(main())