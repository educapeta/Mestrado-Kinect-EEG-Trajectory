"""Estabilizacao temporal e qualidade da normal da palma.

Camada entre o estimador geometrico do Kinect (`palm_plane_camera`) e o filtro
de fusao. Ela NAO recalcula a normal: recebe a normal crua + RMS do plano e
devolve uma normal unitaria com:

- continuidade de sinal (evita flip de 180 graus dentro do filtro);
- suavizacao angular adaptativa por confianca (EMA com alpha variavel);
- confianca/`sigma`/`edge_on` explicitos;
- hold curto quando a visao perde a mao (dropout de poucos frames);
- reset em perda longa/troca de lado.

O alpha e' alto quando a medida e' boa (segue o movimento) e baixo quando o
plano esta' ruidoso ou edge-on (estabiliza sem inventar direcao).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class StabilizedNormal:
    """Saida estabilizada para consumo pelo adaptador/fusao."""
    timestamp: float
    hand: str
    normal_cam: np.ndarray
    confidence: float = 0.0
    sigma_deg: float = 180.0
    rms_m: float = 0.0
    edge_on: bool = False
    valid: bool = False
    held: bool = False


@dataclass
class StabilizerConfig:
    alpha_min: float = 0.25
    alpha_max: float = 0.75
    #: Gap acima do qual o estado e' resetado (nao misturar maos/quadros velhos).
    max_gap_s: float = 0.25
    #: Tempo maximo para repetir a ultima normal em um dropout curto.
    hold_s: float = 0.12
    #: |n_z| abaixo disso: plano edge-on, normal mal condicionada.
    edge_on_abs_z: float = 0.25
    #: RMS do plano em metros.
    rms_good_m: float = 0.004
    rms_bad_m: float = 0.025
    #: Sigma angular base assumindo profundidade/aproximacao de ~0.10 m.
    sigma_scale_m: float = 0.10
    max_sigma_deg: float = 60.0


class PalmNormalStabilizer:
    """Estado temporal por lado (right/left)."""

    def __init__(self, config: Optional[StabilizerConfig] = None):
        self.config = config if config is not None else StabilizerConfig()
        self._normal: Dict[str, np.ndarray] = {}
        self._time: Dict[str, float] = {}
        self._rms: Dict[str, float] = {}
        self._edge: Dict[str, bool] = {}

    def reset(self, hand: str) -> None:
        self._normal.pop(hand, None)
        self._time.pop(hand, None)
        self._rms.pop(hand, None)
        self._edge.pop(hand, None)

    @staticmethod
    def _unit(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float64).reshape(3)
        if not np.isfinite(arr).all():
            return None
        n = float(np.linalg.norm(arr))
        if n < 1e-9:
            return None
        return arr / n

    def _quality(self, rms_m: float, edge_on: bool) -> float:
        c = self.config
        rms = max(0.0, float(rms_m))
        if rms <= c.rms_good_m:
            q_rms = 1.0
        elif rms >= c.rms_bad_m:
            q_rms = 0.0
        else:
            q_rms = 1.0 - (rms - c.rms_good_m) / (c.rms_bad_m - c.rms_good_m)
        q = max(0.0, min(1.0, q_rms))
        if edge_on:
            q *= 0.35
        return q

    def _sigma(self, rms_m: float, edge_on: bool, quality: float) -> float:
        c = self.config
        # Erro angular aproximado: atan(rms / escala de profundidade).
        sigma = float(np.degrees(np.arctan2(max(0.0, rms_m),
                                            c.sigma_scale_m)))
        sigma = max(0.5, sigma)
        if edge_on:
            sigma *= 3.0
        sigma *= (1.0 + 0.5 * (1.0 - max(0.0, min(1.0, quality))))
        return float(np.clip(sigma, 0.5, c.max_sigma_deg))

    def update(self, hand: str, normal_raw: Optional[np.ndarray],
               rms_m: float, timestamp: float) -> StabilizedNormal:
        """Atualiza um lado; devolve normal estabilizada ou invalida."""
        c = self.config
        raw = self._unit(normal_raw)
        anterior = self._normal.get(hand)
        t_anterior = self._time.get(hand)

        if raw is None:
            # Dropout curto: repete a ultima normal por hold_s, com confianca
            # decrescente. Dropout longo: invalida e limpa o estado.
            if anterior is not None and t_anterior is not None:
                idade = timestamp - t_anterior
                if 0.0 <= idade <= c.hold_s:
                    factor = max(0.0, 1.0 - idade / max(c.hold_s, 1e-6))
                    return StabilizedNormal(
                        timestamp=timestamp, hand=hand,
                        normal_cam=anterior.copy(),
                        confidence=0.25 * factor,
                        sigma_deg=self._sigma(self._rms.get(hand, 0.0),
                                              self._edge.get(hand, False),
                                              0.25),
                        rms_m=float(self._rms.get(hand, 0.0)),
                        edge_on=bool(self._edge.get(hand, False)),
                        valid=True, held=True)
            self.reset(hand)
            return StabilizedNormal(
                timestamp=timestamp, hand=hand,
                normal_cam=np.zeros(3, dtype=np.float64),
                confidence=0.0, sigma_deg=180.0, valid=False)

        edge_on = abs(float(raw[2])) < c.edge_on_abs_z
        quality = self._quality(rms_m, edge_on)
        sigma = self._sigma(rms_m, edge_on, quality)

        alpha = c.alpha_min + (c.alpha_max - c.alpha_min) * quality
        gap = (timestamp - t_anterior) if t_anterior is not None else 1e9
        if anterior is None or gap > c.max_gap_s:
            normal = raw
            alpha = 1.0
        else:
            # Continuidade: uma normal oposta e' medida do MESMO plano; alinhar
            # pelo anterior evita um salto de 180 graus no filtro.
            if float(np.dot(raw, anterior)) < 0.0:
                raw = -raw
            normal = alpha * raw + (1.0 - alpha) * anterior
            nn = float(np.linalg.norm(normal))
            normal = (normal / nn) if nn > 1e-9 else raw

        self._normal[hand] = normal.copy()
        self._time[hand] = float(timestamp)
        self._rms[hand] = float(rms_m)
        self._edge[hand] = bool(edge_on)

        confidence = float(np.clip(quality * (0.7 if edge_on else 1.0),
                                   0.0, 1.0))
        return StabilizedNormal(
            timestamp=timestamp, hand=hand,
            normal_cam=normal, confidence=confidence, sigma_deg=sigma,
            rms_m=float(rms_m), edge_on=edge_on, valid=True, held=False)
