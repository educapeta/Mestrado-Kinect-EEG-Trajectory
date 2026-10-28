"""
Módulo para cálculo robusto de normal da palma da mão usando landmarks 3D.
Implementa PCA/SVD com controle de qualidade e rejeição de outliers.
"""
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any


def condicao_plano(autovalores: np.ndarray) -> Dict[str, Any]:
    """
    Avalia o condicionamento do plano a partir de 3 autovalores decrescentes.

    Args:
        autovalores: Array com 3 autovalores em ordem decrescente (lambda1 >= lambda2 >= lambda3).

    Returns:
        Dicionário com:
        - razao_12: lambda1 / lambda2
        - razao_23: lambda2 / lambda3
        - qualidade: razao_23 / razao_12 clipada em [0, 1]
        - mau_condicionado: True se qualidade < 0.05 ou razao_12 > 100
    """
    autovalores = np.asarray(autovalores, dtype=float)
    if autovalores.shape != (3,):
        raise ValueError("autovalores deve ter exatamente 3 elementos")

    lambda1, lambda2, lambda3 = autovalores

    if lambda2 <= 0 or lambda3 <= 0:
        return {
            "razao_12": float("inf"),
            "razao_23": float("inf"),
            "qualidade": 0.0,
            "mau_condicionado": True,
        }

    razao_12 = lambda1 / lambda2
    razao_23 = lambda2 / lambda3
    qualidade = float(np.clip(razao_23 / razao_12, 0.0, 1.0))
    mau_condicionado = qualidade < 0.05 or razao_12 > 100.0

    return {
        "razao_12": float(razao_12),
        "razao_23": float(razao_23),
        "qualidade": qualidade,
        "mau_condicionado": bool(mau_condicionado),
    }


def _filtrar_pontos_validos(
    pontos: Union[Dict[str, np.ndarray], List[np.ndarray]],
    max_dist_centro: float = 0.12,
) -> Tuple[np.ndarray, List[str]]:
    """
    Filtra pontos inválidos e outliers por distância ao centroide.

    Args:
        pontos: Dict nome->(3,) ou lista de arrays (3,).
        max_dist_centro: Distância máxima ao centroide para aceitar o ponto.

    Returns:
        Tupla (array (N,3) de pontos válidos, lista de nomes correspondentes).
    """
    if isinstance(pontos, dict):
        nomes = list(pontos.keys())
        arr = np.array([pontos[n] for n in nomes], dtype=float)
    else:
        arr = np.array(pontos, dtype=float)
        nomes = [f"p{i}" for i in range(len(arr))]

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Pontos devem ser arrays de forma (N, 3)")

    # Remove NaN/inf
    validos = np.isfinite(arr).all(axis=1)
    arr = arr[validos]
    nomes = [nomes[i] for i, v in enumerate(validos) if v]

    if len(arr) == 0:
        return np.empty((0, 3)), []

    # Centroide e filtro de distância
    centroide = arr.mean(axis=0)
    dists = np.linalg.norm(arr - centroide, axis=1)
    mask = dists <= max_dist_centro

    arr_filtrado = arr[mask]
    nomes_filtrados = [nomes[i] for i, m in enumerate(mask) if m]

    return arr_filtrado, nomes_filtrados


def _calcular_normal_pca(pontos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calcula normal via PCA/SVD dos pontos centrados.

    Args:
        pontos: Array (N, 3) com N >= 3.

    Returns:
        Tupla (normal (3,), autovalores (3,), rms_m).
    """
    if len(pontos) < 3:
        raise ValueError("Mínimo 3 pontos para PCA")

    # Centra os pontos
    centroide = pontos.mean(axis=0)
    centrados = pontos - centroide

    # SVD
    U, S, Vt = np.linalg.svd(centrados, full_matrices=False)
    # Autovalores proporcionais a S^2
    autovalores = (S ** 2) / (len(pontos) - 1) if len(pontos) > 1 else S ** 2
    autovalores = np.sort(autovalores)[::-1]  # Decrescente

    # Normal = autovetor do menor autovalor (última linha de Vt)
    normal = Vt[-1, :]
    normal = normal / np.linalg.norm(normal)

    # RMS = raiz do menor autovalor (desvio padrão na direção da normal)
    rms_m = float(np.sqrt(max(autovalores[-1], 0.0)))

    return normal, autovalores, rms_m


def _referencia_anatomica(pontos: np.ndarray, nomes: List[str]) -> Optional[np.ndarray]:
    """
    Calcula vetor de referência anatômica: cross(index-pinky, middle-wrist).

    Args:
        pontos: Array (N, 3) de pontos filtrados.
        nomes: Lista de nomes correspondentes.

    Returns:
        Vetor referência normalizado ou None se pontos necessários não existirem.
    """
    nome_to_idx = {n.lower(): i for i, n in enumerate(nomes)}

    # Tenta nomes padrão do MediaPipe
    idx_map = {
        "wrist": ["wrist", "0"],
        "index_mcp": ["index_mcp", "5"],
        "middle_mcp": ["middle_mcp", "9"],
        "pinky_mcp": ["pinky_mcp", "17"],
    }

    indices = {}
    for chave, alternativas in idx_map.items():
        for alt in alternativas:
            if alt in nome_to_idx:
                indices[chave] = nome_to_idx[alt]
                break

    if len(indices) != 4:
        return None

    wrist = pontos[indices["wrist"]]
    index_mcp = pontos[indices["index_mcp"]]
    middle_mcp = pontos[indices["middle_mcp"]]
    pinky_mcp = pontos[indices["pinky_mcp"]]

    v1 = index_mcp - pinky_mcp
    v2 = middle_mcp - wrist

    ref = np.cross(v1, v2)
    norm = np.linalg.norm(ref)
    if norm < 1e-8:
        return None

    return ref / norm


def normal_por_pontos(
    pontos: Union[Dict[str, np.ndarray], List[np.ndarray]],
    lado: Optional[str] = None,
    referencia: Optional[np.ndarray] = None,
    rms_max_m: float = 0.030,
    min_pontos: int = 4,
) -> Optional[Dict[str, Any]]:
    """
    Calcula normal do plano a partir de pontos 3D com controle de qualidade.

    Args:
        pontos: Dict nome->(3,) ou lista de arrays (3,).
        lado: "left" ou "right" para espelhar a normal se necessário.
        referencia: Vetor (3,) para orientar o sinal da normal (produto escalar > 0).
        rms_max_m: RMS máximo aceitável em metros.
        min_pontos: Número mínimo de pontos válidos após filtro.

    Returns:
        Dicionário com:
        - normal: vetor unitário (3,)
        - rms_m: float
        - n_pontos: int
        - autovalores: array (3,) decrescente
        - condicao: dict de condicao_plano
        - qualidade: float [0,1]
        - confiavel: bool
        - motivo: str (se None, indica falha)
        Ou None se falhar (com motivo no dict retornado).
    """
    # Filtra pontos válidos
    pts_filtrados, nomes_filtrados = _filtrar_pontos_validos(pontos)

    if len(pts_filtrados) < min_pontos:
        return {
            "normal": None,
            "rms_m": None,
            "n_pontos": len(pts_filtrados),
            "autovalores": None,
            "condicao": None,
            "qualidade": 0.0,
            "confiavel": False,
            "motivo": f"Pontos insuficientes após filtro: {len(pts_filtrados)} < {min_pontos}",
        }

    # PCA/SVD
    try:
        normal, autovalores, rms_m = _calcular_normal_pca(pts_filtrados)
    except Exception as e:
        return {
            "normal": None,
            "rms_m": None,
            "n_pontos": len(pts_filtrados),
            "autovalores": None,
            "condicao": None,
            "qualidade": 0.0,
            "confiavel": False,
            "motivo": f"Erro no PCA: {e}",
        }

    # Condicionamento
    condicao = condicao_plano(autovalores)
    qualidade = condicao["qualidade"]

    # Rejeita se mal condicionado
    if condicao["mau_condicionado"]:
        return {
            "normal": None,
            "rms_m": rms_m,
            "n_pontos": len(pts_filtrados),
            "autovalores": autovalores.tolist(),
            "condicao": condicao,
            "qualidade": qualidade,
            "confiavel": False,
            "motivo": "Plano mal condicionado (qualidade < 0.05 ou razao_12 > 100)",
        }

    # Rejeita se RMS alto
    if rms_m > rms_max_m:
        return {
            "normal": None,
            "rms_m": rms_m,
            "n_pontos": len(pts_filtrados),
            "autovalores": autovalores.tolist(),
            "condicao": condicao,
            "qualidade": qualidade,
            "confiavel": False,
            "motivo": f"RMS {rms_m:.4f}m > máximo {rms_max_m:.4f}m",
        }

    # Orientação do sinal
    if referencia is not None:
        # A referência fornecida JÁ está na convenção do projeto (por lado):
        # alinhar a ela e NÃO espelhar de novo para a esquerda.
        ref = np.asarray(referencia, dtype=float)
        if ref.shape == (3,) and np.linalg.norm(ref) > 1e-8:
            if np.dot(normal, ref) < 0:
                normal = -normal
    else:
        # Tenta referência anatômica; aí sim espelha para a esquerda.
        ref_anat = _referencia_anatomica(pts_filtrados, nomes_filtrados)
        if ref_anat is not None:
            if np.dot(normal, ref_anat) < 0:
                normal = -normal
        if lado and lado.lower() == "left":
            normal = -normal

    return {
        "normal": normal.astype(float),
        "rms_m": float(rms_m),
        "n_pontos": len(pts_filtrados),
        "autovalores": autovalores.tolist(),
        "condicao": condicao,
        "qualidade": qualidade,
        "confiavel": True,
        "motivo": "sucesso",
    }


def normal_mao_landmarks(
    landmarks_3d: Dict[str, np.ndarray],
    lado: Optional[str] = None,
    referencia: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """
    Calcula normal da palma a partir de landmarks 3D da mão (até 21 pontos).

    Args:
        landmarks_3d: Dict com chaves "0".."20" ou nomes anatômicos
                     (wrist, index_mcp, middle_mcp, pinky_mcp, etc.).
        lado: "left" ou "right".
        referencia: Vetor (3,) para orientar sinal.

    Returns:
        Dicionário com:
        - normal: vetor unitário (3,) ou None
        - rms_m: float ou None
        - n_pontos: int
        - qualidade: float [0,1]
        - fonte: "todos_pontos" ou "dorso_5_pontos"
        - confiavel: bool
        - motivo: str
    """
    # Normaliza chaves para lowercase
    landmarks = {k.lower(): np.asarray(v, dtype=float) for k, v in landmarks_3d.items()}

    # Tenta com TODOS os pontos válidos
    resultado = normal_por_pontos(landmarks, lado=lado, referencia=referencia)

    if resultado is not None and resultado["normal"] is not None:
        resultado["fonte"] = "todos_pontos"
        resultado["confiavel"] = resultado["qualidade"] >= 0.05
        return resultado

    # Fallback: subconjunto de 5 pontos do dorso da mão
    dorsom_keys = ["wrist", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"]
    dorsom_idx = ["0", "5", "9", "13", "17"]

    pontos_dorso = {}
    for nome, idx in zip(dorsom_keys, dorsom_idx):
        if idx in landmarks:
            pontos_dorso[nome] = landmarks[idx]
        elif nome in landmarks:
            pontos_dorso[nome] = landmarks[nome]

    if len(pontos_dorso) >= 4:
        resultado_dorso = normal_por_pontos(pontos_dorso, lado=lado, referencia=referencia)
        if resultado_dorso is not None and resultado_dorso["normal"] is not None:
            resultado_dorso["fonte"] = "dorso_5_pontos"
            resultado_dorso["confiavel"] = resultado_dorso["qualidade"] >= 0.05
            return resultado_dorso

    # Falha total
    motivo = resultado["motivo"] if resultado else "Pontos insuficientes ou inválidos"
    return {
        "normal": None,
        "rms_m": None,
        "n_pontos": 0,
        "qualidade": 0.0,
        "fonte": "falha",
        "confiavel": False,
        "motivo": motivo,
    }


def normal_robusta_edge_on(
    landmarks_3d: Dict[str, np.ndarray],
    lado: Optional[str] = None,
    normal_anterior: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Pipeline robusto para normal da palma em configuração edge-on.

    Args:
        landmarks_3d: Dict com landmarks 3D da mão.
        lado: "left" ou "right".
        normal_anterior: Normal do frame anterior para estabilização temporal.

    Returns:
        Dicionário final com normal, rms_m, n_pontos, qualidade, fonte, confiavel, motivo.
    """
    resultado = normal_mao_landmarks(landmarks_3d, lado=lado, referencia=normal_anterior)

    if resultado is None:
        return {
            "normal": None,
            "rms_m": None,
            "n_pontos": 0,
            "qualidade": 0.0,
            "fonte": "falha",
            "confiavel": False,
            "motivo": "normal_mao_landmarks retornou None",
        }

    # Se qualidade muito baixa, marca como não confiável
    if resultado["qualidade"] < 0.05:
        resultado["confiavel"] = False
        resultado["motivo"] = "Qualidade < 0.05 (plano mal condicionado)"

    # Alinha sinal à normal anterior se fornecida
    if normal_anterior is not None and resultado["normal"] is not None:
        normal_anterior = np.asarray(normal_anterior, dtype=float)
        if normal_anterior.shape == (3,) and np.linalg.norm(normal_anterior) > 1e-8:
            if np.dot(resultado["normal"], normal_anterior) < 0:
                resultado["normal"] = -resultado["normal"]

    return resultado