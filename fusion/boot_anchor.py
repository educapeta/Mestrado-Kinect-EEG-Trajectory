"""Reancoragem do yaw do boot no FusionEngine.

O T/f salvo no JSON nao contem o yaw ABSOLUTO do IMU (filtro de 6 eixos, sem
magnetometro): ele vale para o boot em que foi medido. O tool do groundtruth
resolve isso com `imu.AncoradorYawBoot`, que estima o delta de yaw usando
amostras `(R_imu, normal_do_Kinect)` e corrige o T EM MEMORIA.

Aqui fazemos o mesmo para a `Calibration` do `FusionEngine`, sem tocar no JSON.
Depois de ancorar, o filtro e' reinicializado com a atitude completa nova e a
primeira visao alinha a normal; a seta amarela `n_imu` fica correta neste boot.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from imu import AncoradorYawBoot, angle_between, rotation_matrix
from .contracts import Calibration, ImuSample


class _CalibrationAdapter:
    """Adapta `Calibration` a interface que o AncoradorYawBoot espera."""

    def __init__(self, cal: Calibration):
        self.cal = cal
        self.yaw_boot_delta_deg = 0.0

    @property
    def imu_to_camera(self) -> np.ndarray:
        return self.cal.T_cam_world

    @property
    def palm_reference_body(self) -> np.ndarray:
        return self.cal.f_body

    @property
    def palm_alignment_residual_deg(self):
        return self.cal.residual_deg

    def set_palm_alignment(self, rotation, body, residual=None, base=False):
        try:
            self.cal.T_cam_world = np.asarray(rotation, np.float64).reshape(3, 3)
            self.cal.f_body = np.asarray(body, np.float64).reshape(3)
            if residual is not None:
                self.cal.residual_deg = float(residual)
            return True
        except (TypeError, ValueError):
            return False


class CalibrationBootAnchor:
    """Um ancorador por mao, ligado a uma Calibration mutavel."""

    def __init__(self, calibration: Calibration, lado: str,
                 amostras_min: int = 8, sensibilidade_min: float = 0.35,
                 timeout_s: float = 90.0, avisar=print):
        self.calibration = calibration
        self.lado = str(lado)
        self._adapter = _CalibrationAdapter(calibration)
        self._ancora = AncoradorYawBoot(
            self._adapter, lado=lado, amostras_min=amostras_min,
            sensibilidade_min=sensibilidade_min, timeout_s=timeout_s,
            avisar=avisar)

    @property
    def done(self) -> bool:
        return bool(self._ancora.pronto or self._ancora.falhou)

    @property
    def delta_deg(self) -> Optional[float]:
        return self._ancora.delta_deg

    def sample(self, imu: ImuSample, normal_cam: np.ndarray):
        """Alimenta uma amostra; devolve (delta, erro, n) quando ancorou."""
        if self.done or imu is None or not imu.valid:
            return None
        normal = np.asarray(normal_cam, np.float64).reshape(3)
        if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
            return None
        R = rotation_matrix(float(imu.roll_deg), float(imu.pitch_deg),
                            float(imu.yaw_deg))
        try:
            return self._ancora.amostra(R, normal)
        except Exception:
            return None
