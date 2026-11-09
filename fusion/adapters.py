"""
Sensor adapters for live fusion pipeline.

Provides:
- KinectMediaPipeSource: Kinect + MediaPipe hand tracking adapter
- ImuBankSource: IMU bank adapter for dual-hand UDP receivers
- SyntheticSources: Fake sources for testing/demo without hardware
- load_calibrations: Load hand-eye calibration JSON files
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from .contracts import (
    KinectObservation,
    ImuSample,
    Calibration,
)
from .calibration import load_calibration
from .palm_normal import PalmNormalStabilizer

# Mapping from hand side string to ImuBank side code (1=right, 2=left)
SIDE_TO_BANK = {'right': 1, 'left': 2}


class KinectMediaPipeSource:
    """
    Adapter for Kinect depth camera + MediaPipe hand tracking.

    Provides synchronized color/depth frames and per-hand observations
    with palm orientation in camera frame.
    """

    def __init__(self, tracker: Optional[Any] = None):
        """
        Initialize source.

        Args:
            tracker: Optional pre-created KinectHandTracker instance.
                     If None, created lazily in start().
        """
        self._tracker = tracker
        self._started = False
        self._runtime: Optional[Any] = None
        self._normal_stab = PalmNormalStabilizer()
        self._last_raw_normal: Dict[str, np.ndarray] = {}
        #: Ultima observacao valida por lado, para hold entre frames do Kinect.
        self._last_obs: Dict[str, KinectObservation] = {}
        self._last_obs_time: Dict[str, float] = {}
        self.max_hold_s = 0.15

    def _create_tracker(self) -> Any:
        """Lazy factory for KinectHandTracker."""
        import kinect_compat
        kinect_compat.aplicar()
        import kinect_imu_groundtruth as kinect_gt
        return kinect_gt.KinectHandTracker()

    def start(self) -> None:
        """Start the Kinect runtime and tracker."""
        if self._started:
            return
        if self._tracker is None:
            self._tracker = self._create_tracker()
        # KinectHandTracker abre o runtime no __init__; nao tem start().
        start = getattr(self._tracker, "start", None)
        if callable(start):
            start()
        self._started = True

    def close(self) -> None:
        """Stop the Kinect runtime and release resources."""
        if not self._started:
            return
        if self._tracker is not None:
            close = getattr(self._tracker, "close", None)
            if callable(close):
                close()
            else:
                kinect = getattr(self._tracker, "kinect", None)
                if kinect is not None:
                    try:
                        kinect.close()
                    except Exception:
                        pass
        self._started = False

    def _hold(self, now: float) -> Dict[str, KinectObservation]:
        """Reusa a ultima observacao por ate max_hold_s (curto)."""
        saida: Dict[str, KinectObservation] = {}
        for side, obs in self._last_obs.items():
            idade = now - self._last_obs_time.get(side, -1e9)
            if idade < 0.0 or idade > self.max_hold_s:
                continue
            # Decaimento linear; nao deixa um frame velho com confianca alta.
            fator = max(0.0, 1.0 - idade / max(self.max_hold_s, 1e-6))
            from dataclasses import replace
            novo = replace(
                obs,
                timestamp=now,
                confidence=float(obs.confidence) * 0.5 * fator,
                normal_held=True,
                source="kinect_depth_hold",
            )
            if novo.confidence > 0.15:
                saida[side] = novo
        return saida

    def poll(self) -> tuple[float, Optional[np.ndarray], Dict[str, KinectObservation]]:
        """
        Poll for next frame and hand observations.

        Returns:
            Tuple of (timestamp, frame_bgr, observations_dict)
            - timestamp: monotonic time in seconds
            - frame_bgr: BGR color frame with overlay drawn (or None)
            - observations_dict: mapping hand side -> KinectObservation
        """
        now = time.monotonic()
        frame_bgr = None
        observations: Dict[str, KinectObservation] = {}

        if not self._started or self._tracker is None:
            return now, frame_bgr, observations

        try:
            # Get frames from tracker
            frames_result = self._tracker.frames()
            if frames_result is None:
                return now, frame_bgr, observations

            color_bgra, depth = frames_result
            if color_bgra is None:
                return now, frame_bgr, observations

            # Convert BGRA to BGR for display
            if color_bgra.ndim == 3 and color_bgra.shape[2] == 4:
                frame_bgr = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)
            else:
                frame_bgr = color_bgra.copy()

            # detect_hands_all devolve (frame_desenhado, lista_de_maos)
            detect_result = self._tracker.detect_hands_all(color_bgra, depth)
            if detect_result is None:
                return now, frame_bgr, observations
            if isinstance(detect_result, tuple) and len(detect_result) == 2:
                frame_detectado, hands = detect_result
                if isinstance(frame_detectado, np.ndarray):
                    frame_bgr = (cv2.cvtColor(frame_detectado, cv2.COLOR_BGRA2BGR)
                                 if frame_detectado.ndim == 3
                                 and frame_detectado.shape[2] == 4
                                 else frame_detectado.copy())
            else:
                hands = detect_result
            if not hands:
                return now, frame_bgr, observations

            seen_sides = set()

            for hand_data in hands:
                side = hand_data.get('side')
                if side not in ('right', 'left'):
                    continue
                if side in seen_sides:
                    continue  # skip duplicates

                landmarks = hand_data.get('landmarks')
                position = hand_data.get('position')

                if landmarks is None or position is None:
                    continue

                # A calibracao Y usa o PLANO MEDIO do dorso (5 pontos).
                # Usar primeiro essa mesma verdade evita comparar macas com
                # laranjas (normal de 3 pontos e' bem mais ruidosa).
                normal = None
                direction = None
                rms_normal = 0.0
                raw_normal = None
                try:
                    # Passar a normal anterior ajuda o plano e a continuidade
                    # do sinal; o estabilizador faz a suavizacao angular.
                    raw_normal, rms_normal = self._tracker.palm_plane_camera(
                        landmarks, side,
                        normal_anterior=self._last_raw_normal.get(side))
                except Exception:
                    raw_normal, rms_normal = None, 0.0
                # Fallback: normal de 3 pontos + direcao dos dedos.
                try:
                    if raw_normal is None:
                        raw_normal, direction = self._tracker.palm_orientation_camera(
                            landmarks, side)
                        # Fallback tem mais ruido: sinaliza RMS pior.
                        rms_normal = 0.015
                    else:
                        _n3, direction = self._tracker.palm_orientation_camera(
                            landmarks, side)
                except Exception:
                    if raw_normal is None:
                        raw_normal = None
                        direction = None
                if raw_normal is not None:
                    try:
                        self._last_raw_normal[side] = np.asarray(
                            raw_normal, dtype=np.float64).reshape(3).copy()
                    except (TypeError, ValueError):
                        pass
                stabil = self._normal_stab.update(
                    side, raw_normal, float(rms_normal or 0.0), now)
                normal = stabil.normal_cam if stabil.valid else None

                # Posicao 2D da palma (landmark 9) para ancorar o overlay.
                palm_px = None
                try:
                    lm = landmarks[9]
                    h_img, w_img = color_bgra.shape[:2]
                    palm_px = np.array([float(lm.x) * w_img,
                                        float(lm.y) * h_img],
                                       dtype=np.float64)
                except (IndexError, AttributeError, TypeError, ValueError):
                    palm_px = None

                # Create observation
                valid = bool(stabil.valid and normal is not None
                             and np.all(np.isfinite(normal)))
                confidence = float(stabil.confidence if valid else 0.0)

                obs = KinectObservation(
                    timestamp=now,
                    hand=side,
                    wrist_cam=np.asarray(position, dtype=np.float64).reshape(3) if position is not None else np.zeros(3),
                    palm_normal_cam=np.asarray(normal, dtype=np.float64).reshape(3) if normal is not None else np.zeros(3),
                    palm_dir_cam=np.asarray(direction, dtype=np.float64).reshape(3) if direction is not None else np.zeros(3),
                    confidence=confidence,
                    valid=bool(valid),
                    source='kinect_depth',
                    palm_px=palm_px,
                    normal_rms_m=float(stabil.rms_m),
                    normal_sigma_deg=float(stabil.sigma_deg),
                    edge_on=bool(stabil.edge_on),
                    normal_held=bool(stabil.held),
                )
                observations[side] = obs
                self._last_obs[side] = obs
                self._last_obs_time[side] = now
                seen_sides.add(side)

        except Exception:
            # Never throw on bad frame - return what we have
            pass

        # Entre frames do Kinect/MediaPipe, reusa por poucos ms a ultima
        # observacao. Sem isso o FusionEngine fica majoritariamente imu_only
        # porque o loop roda mais rapido que a visao (~30 Hz).
        if self._started:
            for side, extra in self._hold(now).items():
                observations.setdefault(side, extra)

        return now, frame_bgr, observations


class ImuBankSource:
    """
    Adapter for ImuBank (dual-hand IMU receiver bank).

    Polls latest IMU samples for each hand and converts to ImuSample contracts.
    """

    def __init__(self, bank: Optional[Any] = None, portas: str = '4210,4211'):
        """
        Initialize IMU bank source.

        Args:
            bank: Optional pre-created ImuBank instance. If None, created lazily.
            portas: Comma-separated port string for ImuBank (default: '4210,4211')
        """
        self._bank = bank
        self._portas = portas
        self._started = False

    def _create_bank(self) -> Any:
        """Lazy factory for ImuBank."""
        import imu
        return imu.ImuBank(self._portas)

    def start(self) -> None:
        """Start the IMU bank receivers."""
        if self._started:
            return
        if self._bank is None:
            self._bank = self._create_bank()
        self._bank.start()
        self._started = True

    def stop(self) -> None:
        """Stop the IMU bank receivers."""
        if not self._started:
            return
        if self._bank is not None:
            self._bank.stop()
        self._started = False

    def poll(self, now: Optional[float] = None) -> Dict[str, ImuSample]:
        """
        Poll latest IMU samples for both hands.

        Args:
            now: Current monotonic timestamp (defaults to time.monotonic())

        Returns:
            Dict mapping hand side ('right', 'left') -> ImuSample
        """
        if now is None:
            now = time.monotonic()

        samples: Dict[str, ImuSample] = {}

        if not self._started or self._bank is None:
            return samples

        for hand, bank_side in SIDE_TO_BANK.items():
            try:
                sample = self._bank.amostra(bank_side)
                if sample is None:
                    continue

                # Check validity: exists, finite values, age < 0.5s
                age = self._bank.idade_s(bank_side)
                valid = (
                    sample is not None
                    and np.all(np.isfinite(sample.accel_g))
                    and np.all(np.isfinite(sample.gyro_dps))
                    and np.isfinite(sample.roll_deg)
                    and np.isfinite(sample.pitch_deg)
                    and np.isfinite(sample.yaw_deg)
                    and age is not None
                    and age < 0.5
                )

                imu_sample = ImuSample(
                    timestamp=sample.received_at,
                    timestamp_us=sample.timestamp_us,
                    hand=hand,
                    roll_deg=float(sample.roll_deg),
                    pitch_deg=float(sample.pitch_deg),
                    yaw_deg=float(sample.yaw_deg),
                    accel_g=np.asarray(sample.accel_g, dtype=np.float64).reshape(3),
                    gyro_dps=np.asarray(sample.gyro_dps, dtype=np.float64).reshape(3),
                    valid=bool(valid),
                    source='imu_udp'
                )
                samples[hand] = imu_sample

            except Exception:
                # Skip invalid samples silently
                continue

        return samples


class SyntheticSources:
    """
    Synthetic data sources for testing/demo without hardware.

    Generates zero frames and synthetic observations for both hands.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int] = (1080, 1920),
        include_depth: bool = True
    ):
        """
        Initialize synthetic sources.

        Args:
            frame_shape: (height, width) for generated color frame
            include_depth: Whether to generate depth frame
        """
        self.frame_shape = frame_shape
        self.include_depth = include_depth
        self._frame_count = 0
        self._start_time = time.monotonic()

    def start(self) -> None:
        """Start synthetic source (no-op)."""
        self._start_time = time.monotonic()
        self._frame_count = 0

    def close(self) -> None:
        """Close synthetic source (no-op)."""
        pass

    def poll(self) -> tuple[float, Optional[np.ndarray], Dict[str, KinectObservation]]:
        """
        Generate synthetic frame and observations.

        Returns:
            Tuple of (timestamp, frame_bgr, observations_dict)
        """
        now = time.monotonic()
        self._frame_count += 1

        # Create zero BGR frame
        h, w = self.frame_shape
        frame_bgr = np.zeros((h, w, 3), dtype=np.uint8)

        # Create synthetic observations for both hands
        observations: Dict[str, KinectObservation] = {}

        for side in ('right', 'left'):
            # Simple synthetic palm normal (pointing toward camera)
            normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            # Palm direction (fingers pointing up)
            direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            # Wrist position (offset left/right)
            wrist_x = -0.2 if side == 'right' else 0.2
            wrist = np.array([wrist_x, 0.0, 1.0], dtype=np.float64)

            obs = KinectObservation(
                timestamp=now,
                hand=side,
                wrist_cam=wrist,
                palm_normal_cam=normal,
                palm_dir_cam=direction,
                confidence=1.0,
                valid=True,
                source='synthetic'
            )
            observations[side] = obs

        return now, frame_bgr, observations

    def poll_imu(self, now: Optional[float] = None) -> Dict[str, ImuSample]:
        """
        Generate synthetic IMU samples.

        Args:
            now: Current timestamp (defaults to time.monotonic())

        Returns:
            Dict mapping hand side -> ImuSample
        """
        if now is None:
            now = time.monotonic()

        samples: Dict[str, ImuSample] = {}

        for side in ('right', 'left'):
            sample = ImuSample(
                timestamp=now,
                timestamp_us=int(now * 1_000_000),
                hand=side,
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0,
                accel_g=np.array([0.0, 0.0, 1.0], dtype=np.float64),  # 1g up
                gyro_dps=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                valid=True,
                source='synthetic'
            )
            samples[side] = sample

        return samples


def load_calibrations(
    directory: str | Path = '.',
    lado_arquivos: Optional[Dict[str, str]] = None
) -> Dict[str, Calibration]:
    """
    Load hand-eye calibration files for both hands.

    Tries to load imu_palm_alignment_right.json and imu_palm_alignment_left.json
    from the given directory using fusion.calibration.load_calibration.

    Args:
        directory: Directory containing calibration JSON files
        lado_arquivos: Optional mapping of hand -> custom filename

    Returns:
        Dict mapping hand ('right', 'left') -> valid Calibration
    """
    calibrations: Dict[str, Calibration] = {}

    default_files = {
        'right': 'imu_palm_alignment_right.json',
        'left': 'imu_palm_alignment_left.json',
    }

    files_to_try = lado_arquivos if lado_arquivos is not None else default_files

    for hand, filename in files_to_try.items():
        try:
            filepath = Path(directory) / filename
            cal = load_calibration(filepath, hand)
            if cal is not None and cal.validate():
                calibrations[hand] = cal
        except Exception:
            # Skip invalid/missing calibrations silently
            continue

    return calibrations


# Import cv2 at module level for KinectMediaPipeSource
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore