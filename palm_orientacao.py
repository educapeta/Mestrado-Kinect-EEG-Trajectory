"""
Módulo de orientação da palma (palm_orientacao.py).

Fornece utilitários para:
- Cálculo de ângulo dobrado (menor ângulo entre dois vetores, 0-90 graus).
- Controle de qualidade (QA) de normais medidos contra alvos.
- Estabilização de sinal de normal para evitar flutuação de sinal em ruído.
- Correção de normal para alinhar com uma referência.
"""

import numpy as np
from typing import Optional, Dict, Any, Union


def angulo_dobrado(ang: Optional[Union[float, int]]) -> Optional[float]:
    """
    Retorna o ângulo dobrado: min(ang, 180 - ang) para ângulos finitos.

    O ângulo dobrado representa o menor ângulo entre duas direções,
    ignorando a orientação (sentido) do vetor. Útil para comparar
    direções onde o sinal (normal vs -normal) não importa.

    Args:
        ang: Ângulo em graus (0 a 180). Pode ser None ou NaN.

    Returns:
        float: Ângulo dobrado em [0, 90] graus.
        None: Se entrada for None ou NaN.
    """
    if ang is None:
        return None
    try:
        # np.isnan falha para tipos não numéricos, então tentamos converter
        ang_float = float(ang)
        if np.isnan(ang_float):
            return None
    except (TypeError, ValueError):
        return None

    # Normaliza para [0, 180]
    ang_norm = ang_float % 360.0
    if ang_norm > 180.0:
        ang_norm = 360.0 - ang_norm

    return min(ang_norm, 180.0 - ang_norm)


def qa_efetivo(
    normal_medida: np.ndarray,
    alvo_camera: np.ndarray,
    tipo: str,
    limite_geometrica_deg: float,
    limite_posicao_deg: float
) -> Dict[str, Any]:
    """
    Avalia qualidade de uma normal medida contra um alvo da câmera.

    Dois modos de avaliação:
    - "geometrica": Usa o ângulo bruto entre vetores. Rejeita normais invertidas
      (próximas a 180°) pois a geometria da palma exige orientação correta.
    - "posicao": Usa o ângulo dobrado. Aceita normais invertidas pois apenas
      a direção da palma importa, não o sentido da normal.

    Args:
        normal_medida: Vetor normal medido (array shape (3,)).
        alvo_camera: Vetor alvo da câmera (array shape (3,)).
        tipo: "geometrica" ou "posicao".
        limite_geometrica_deg: Limite em graus para modo geometrica.
        limite_posicao_deg: Limite em graus para modo posicao.

    Returns:
        Dict com chaves:
        - bruto_deg: Ângulo bruto entre vetores (0-180), ou None se degenerado.
        - efetivo_deg: Ângulo efetivo usado na decisão (bruto ou dobrado).
        - limite_deg: Limite aplicado.
        - aprovado: True se efetivo_deg <= limite_deg.
        - motivo: String explicando falha (apenas se não aprovado ou degenerado).
    """
    # Validação de entradas degeneradas
    if normal_medida is None or alvo_camera is None:
        return {
            "bruto_deg": None,
            "efetivo_deg": None,
            "limite_deg": limite_geometrica_deg if tipo == "geometrica" else limite_posicao_deg,
            "aprovado": False,
            "motivo": "entrada_none"
        }

    try:
        n = np.asarray(normal_medida, dtype=float).flatten()
        a = np.asarray(alvo_camera, dtype=float).flatten()
    except (ValueError, TypeError):
        return {
            "bruto_deg": None,
            "efetivo_deg": None,
            "limite_deg": limite_geometrica_deg if tipo == "geometrica" else limite_posicao_deg,
            "aprovado": False,
            "motivo": "entrada_invalida"
        }

    if n.shape != (3,) or a.shape != (3,):
        return {
            "bruto_deg": None,
            "efetivo_deg": None,
            "limite_deg": limite_geometrica_deg if tipo == "geometrica" else limite_posicao_deg,
            "aprovado": False,
            "motivo": "shape_invalido"
        }

    norm_n = np.linalg.norm(n)
    norm_a = np.linalg.norm(a)

    if norm_n == 0.0 or norm_a == 0.0 or np.isnan(norm_n) or np.isnan(norm_a):
        return {
            "bruto_deg": None,
            "efetivo_deg": None,
            "limite_deg": limite_geometrica_deg if tipo == "geometrica" else limite_posicao_deg,
            "aprovado": False,
            "motivo": "norma_zero_ou_nan"
        }

    # Normaliza vetores
    n_unit = n / norm_n
    a_unit = a / norm_a

    # Produto escalar clampado para [-1, 1] para estabilidade numérica
    dot = np.clip(np.dot(n_unit, a_unit), -1.0, 1.0)
    bruto_rad = np.arccos(dot)
    bruto_deg = float(np.degrees(bruto_rad))

    if tipo == "geometrica":
        efetivo_deg = bruto_deg
        limite_deg = limite_geometrica_deg
    elif tipo == "posicao":
        efetivo_deg = angulo_dobrado(bruto_deg)
        limite_deg = limite_posicao_deg
    else:
        return {
            "bruto_deg": bruto_deg,
            "efetivo_deg": None,
            "limite_deg": 0.0,
            "aprovado": False,
            "motivo": "tipo_invalido"
        }

    aprovado = efetivo_deg is not None and efetivo_deg <= limite_deg
    motivo = None if aprovado else f"{tipo}_acima_limite"

    return {
        "bruto_deg": bruto_deg,
        "efetivo_deg": efetivo_deg,
        "limite_deg": limite_deg,
        "aprovado": aprovado,
        "motivo": motivo
    }


class EstabilizadorSinal:
    """
    Estabiliza o sinal de uma normal vetorial ao longo do tempo.

    Problema: Ruído de medição pode causar flutuação do sinal da normal
    (normal vs -normal) quando a palma está de lado (edge-on), pois
    pequenos ruídos projetam a normal para lados opostos.

    Solução: Mantém uma referência de sinal e só troca depois de
    `frames_para_trocar` amostras CONSECUTIVAS opostas com ângulo
    >= `tolerancia_deg` (próximo a 180°, ex: 170°). Isso ignora ruído
    de até ~10° mas detecta viradas reais sustentadas.
    """

    def __init__(self, frames_para_trocar: int = 8, tolerancia_deg: float = 170.0):
        """
        Inicializa o estabilizador.

        Args:
            frames_para_trocar: Número de amostras consecutivas opostas
                                necessárias para inverter o sinal da referência.
            tolerancia_deg: Ângulo mínimo (graus) para considerar uma amostra
                            "oposta" à referência. Valores típicos: 170-175°.
        """
        if frames_para_trocar < 1:
            raise ValueError("frames_para_trocar deve ser >= 1")
        if not (0 < tolerancia_deg <= 180):
            raise ValueError("tolerancia_deg deve estar em (0, 180]")

        self._frames_para_trocar = frames_para_trocar
        self._tolerancia_deg = tolerancia_deg
        self._referencia: Optional[np.ndarray] = None
        self._frames_opostos: int = 0

    @property
    def referencia(self) -> Optional[np.ndarray]:
        """Retorna a normal de referência atual (unitária) ou None."""
        return self._referencia.copy() if self._referencia is not None else None

    @property
    def frames_opostos(self) -> int:
        """Retorna o contador atual de frames consecutivos opostos."""
        return self._frames_opostos

    def reset(self) -> None:
        """Reseta o estado interno (referência e contador)."""
        self._referencia = None
        self._frames_opostos = 0

    def atualizar(self, normal: Optional[np.ndarray], agora_s: float) -> Optional[np.ndarray]:
        """
        Atualiza o estabilizador com uma nova normal e retorna a normal estabilizada.

        Args:
            normal: Vetor normal medido (shape (3,)) ou None.
            agora_s: Timestamp atual em segundos (não usado na lógica, reservado para futuro).

        Returns:
            Normal estabilizada (mesmo sinal da referência) ou None se entrada inválida.
        """
        if normal is None:
            return None

        try:
            n = np.asarray(normal, dtype=float).flatten()
        except (ValueError, TypeError):
            return None

        if n.shape != (3,):
            return None

        norm_n = np.linalg.norm(n)
        if norm_n == 0.0 or np.isnan(norm_n):
            return None

        n_unit = n / norm_n

        # Primeira amostra: define referência
        if self._referencia is None:
            self._referencia = n_unit.copy()
            self._frames_opostos = 0
            return self._referencia.copy()

        # Calcula ângulo entre normal atual e referência
        dot = np.clip(np.dot(n_unit, self._referencia), -1.0, 1.0)
        ang_deg = float(np.degrees(np.arccos(dot)))

        # Verifica se é oposta (ângulo >= tolerancia)
        if ang_deg >= self._tolerancia_deg:
            self._frames_opostos += 1
            # Troca o sinal da referência se atingiu o limiar
            if self._frames_opostos >= self._frames_para_trocar:
                self._referencia = -self._referencia
                self._frames_opostos = 0
        else:
            # Não é oposta: reseta contador
            self._frames_opostos = 0

        # Retorna normal com sinal alinhado à referência
        dot_ref = np.dot(n_unit, self._referencia)
        if dot_ref < 0:
            return -n_unit
        return n_unit


def corrigir_normal_para_referencia(
    normal: Optional[np.ndarray],
    referencia: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """
    Alinha o sinal de `normal` para ter produto escalar positivo com `referencia`.

    Se o dot product for negativo, inverte a normal. Útil para garantir
    consistência de orientação entre frames ou entre câmeras.

    Args:
        normal: Vetor normal a corrigir (shape (3,)) ou None.
        referencia: Vetor de referência (shape (3,)) ou None.

    Returns:
        Normal corrigida (mesma magnitude, sinal alinhado) ou None se
        qualquer entrada for None, norma zero, ou NaN.
    """
    if normal is None or referencia is None:
        return None

    try:
        n = np.asarray(normal, dtype=float).flatten()
        r = np.asarray(referencia, dtype=float).flatten()
    except (ValueError, TypeError):
        return None

    if n.shape != (3,) or r.shape != (3,):
        return None

    norm_n = np.linalg.norm(n)
    norm_r = np.linalg.norm(r)

    if norm_n == 0.0 or norm_r == 0.0 or np.isnan(norm_n) or np.isnan(norm_r):
        return None

    n_unit = n / norm_n
    r_unit = r / norm_r

    if np.dot(n_unit, r_unit) < 0:
        return -n_unit
    return n_unit