"""
Live fusion engine and application for real-time sensor fusion.

Provides:
- LiveConfig: Configuration dataclass for live fusion
- FusionEngine: Core fusion logic combining IMU + vision
- LiveFusionApp: Complete application with sources, engine, and visualization
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import math
import time
import numpy as np

from .contracts import (
    KinectObservation,
    ImuSample,
    Calibration,
    FusedState,
    SourceStatus,
)
from .complementary import (
    QuaternionComplementaryFilter,
    ComplementaryConfig,
)
from .sensor_manager import SensorManager
from .visualization import draw_overlay


@dataclass
class LiveConfig:
    """Configuration for live fusion application."""
    duration_s: float = 30.0          # Total run duration in seconds
    show: bool = True                 # Whether to show OpenCV window
    print_every_s: float = 1.0        # Print status every N seconds
    max_dt: float = 0.1               # Maximum dt for filter steps


class FusionEngine:
    """
    Core fusion engine combining IMU and vision observations.

    Maintains one QuaternionComplementaryFilter per hand and produces
    FusedState estimates using SensorManager for source selection.
    """

    def __init__(
        self,
        calibrations: Dict[str, Calibration],
        configs: Optional[Dict[str, ComplementaryConfig]] = None
    ):
        """
        Initialize fusion engine.

        Args:
            calibrations: Dict mapping hand -> Calibration (must be valid)
            configs: Optional per-hand ComplementaryConfig overrides
        """
        self._filters: Dict[str, QuaternionComplementaryFilter] = {}
        self._sensor_managers: Dict[str, SensorManager] = {}
        self._last_update: Dict[str, float] = {}
        #: Inicializacao unica de q com a atitude do IMU + calibracao.
        self._inicializados: Dict[str, bool] = {}
        #: Primeira visao valida usada para alinhar a normal (yaw do boot).
        self._visao_alinhada: Dict[str, bool] = {}

        for hand, cal in calibrations.items():
            if not cal.validate():
                continue

            # Default config with gravity vector from calibration
            # gravity_cam = cal.T_cam_world @ [0, 0, -1] (world down -> camera)
            gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            gravity_cam = cal.T_cam_world @ gravity_world

            if configs and hand in configs:
                config = configs[hand]
                config.gravity_cam = gravity_cam
            else:
                config = ComplementaryConfig(gravity_cam=gravity_cam)

            # Create filter with calibration
            filter_obj = QuaternionComplementaryFilter(
                config=config,
                calibration=cal
            )
            self._filters[hand] = filter_obj

            # Sensor manager for this hand
            self._sensor_managers[hand] = SensorManager()

            self._last_update[hand] = 0.0
            self._inicializados[hand] = False
            self._visao_alinhada[hand] = False

    def calibration(self, hand: str) -> Optional[Calibration]:
        """Calibracao ativa do lado (para diagnostico)."""
        filtro = self._filters.get(hand)
        return None if filtro is None else filtro.calibration

    def update(
        self,
        now: float,
        kinect_obs: Dict[str, KinectObservation],
        imu_samples: Dict[str, ImuSample]
    ) -> Dict[str, FusedState]:
        """
        Update fusion engine with latest observations.

        Args:
            now: Current monotonic timestamp
            kinect_obs: Dict mapping hand -> KinectObservation
            imu_samples: Dict mapping hand -> ImuSample

        Returns:
            Dict mapping hand -> FusedState (only for hands with valid calibration)
        """
        fused_states: Dict[str, FusedState] = {}

        for hand, filter_obj in self._filters.items():
            # Get observations for this hand
            obs = kinect_obs.get(hand)
            imu = imu_samples.get(hand)

            # Compute dt
            last = self._last_update.get(hand, 0.0)
            if last == 0.0:
                dt = 1.0 / 30.0  # First frame default
            else:
                dt = now - last
            dt = float(np.clip(dt, 1e-6, filter_obj.config.max_dt))
            self._last_update[hand] = now

            # Determine vision input
            vision_normal = None
            vision_confidence = 0.0
            vision_valid = False
            dir_cam = None

            if obs is not None and obs.valid and obs.palm_normal_cam is not None:
                normal = obs.palm_normal_cam
                if np.all(np.isfinite(normal)) and np.linalg.norm(normal) > 1e-6:
                    vision_normal = normal / np.linalg.norm(normal)
                    vision_confidence = float(obs.confidence)
                    # Edge-on/held/sigma alto: reduzir o ganho de yaw em vez
                    # de deixar um frame ruim "puxar" o filtro.
                    if getattr(obs, "edge_on", False):
                        vision_confidence *= 0.25
                    sigma_vis = float(getattr(obs, "normal_sigma_deg", 0.0))
                    if sigma_vis > 15.0:
                        vision_confidence *= max(0.1, 15.0 / sigma_vis)
                    if getattr(obs, "normal_held", False):
                        vision_confidence *= 0.5
                    vision_valid = True
                    dir_cam = obs.palm_dir_cam

            # 1) Inicializacao de ATITUDE COMPLETA com IMU + calibracao:
            # q = T_cam_world * R_imu(body->world). Nao basta alinhar so' a
            # normal (fica rolando em torno dela e o gyro deriva o vetor).
            if (not self._inicializados.get(hand, False)
                    and imu is not None and imu.valid):
                try:
                    q0 = filter_obj.calibration.q_body_cam_from_imu(imu)
                    filter_obj.reset(q0)
                    self._inicializados[hand] = True
                    # Estado completo novo: alinhar a normal ainda e' util
                    # para corrigir o yaw do boot.
                    self._visao_alinhada[hand] = False
                except Exception:
                    pass

            # 2) Primeira normal valida da visao: corrige yaw do boot/residuo
            # de T/f SEM perder a atitude completa do IMU.
            if (vision_normal is not None
                    and not self._visao_alinhada.get(hand, False)):
                try:
                    if filter_obj.align_to_normal(vision_normal):
                        self._visao_alinhada[hand] = True
                        self._inicializados[hand] = True
                except Exception:
                    pass

            # Run filter step
            q_updated = filter_obj.step(
                imu=imu if (imu is not None and imu.valid) else None,
                normal_cam=vision_normal,
                dt=dt,
                vision_confidence=vision_confidence
            )

            # Get sensor decision
            sensor_mgr = self._sensor_managers[hand]
            decision = sensor_mgr.decide(
                kinect=obs,
                vision=obs,
                imu=imu if (imu is not None and imu.valid) else None,
                now=now
            )

            # Get fused state
            fused_state = filter_obj.get_fused_state(
                timestamp=now,
                hand=hand,
                confidence=decision.confidence,
                source=decision.source,
                latency_ms=((now - obs.timestamp) * 1000.0
                            if obs is not None else 0.0),
                dir_cam=dir_cam if dir_cam is not None and np.linalg.norm(dir_cam) > 1e-6 else None,
                palm_px=(obs.palm_px if obs is not None else None),
            )

            # Only return states for hands with valid calibration
            if fused_state.valid:
                fused_states[hand] = fused_state

        return fused_states


class LiveFusionApp:
    """
    Complete live fusion application.

    Coordinates Kinect source, IMU source, fusion engine, and visualization.
    """

    def __init__(
        self,
        engine: FusionEngine,
        kinect_source: Optional[Any] = None,
        imu_source: Optional[Any] = None,
        config: Optional[LiveConfig] = None,
        visualizer: Optional[Any] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize live fusion app.

        Args:
            engine: FusionEngine instance
            kinect_source: KinectMediaPipeSource or compatible (poll() -> (ts, frame, obs))
            imu_source: ImuBankSource or compatible (poll(now) -> dict[hand, ImuSample])
            config: LiveConfig instance
            visualizer: Optional visualizer with draw_overlay(frame, state, statuses) method
        """
        self.engine = engine
        self.kinect_source = kinect_source
        self.imu_source = imu_source
        self.config = config if config is not None else LiveConfig()
        self.visualizer = visualizer
        self.logger = logger

        # Statistics
        self._frame_count = 0
        self._state_count = 0
        self._start_time = 0.0
        self._last_print = 0.0

    def run(self) -> Dict[str, Any]:
        """
        Run the live fusion loop.

        Returns:
            Dict with summary statistics (frames, states, fps, duration)
        """
        # Start sources FIRST: abrir Kinect/MediaPipe leva alguns segundos e
        # esse warmup NAO pode consumir a duracao pedida pelo usuario.
        if self.kinect_source is not None:
            try:
                self.kinect_source.start()
            except Exception as exc:
                print(f"[AVISO] Kinect nao iniciou: {exc}", flush=True)

        if self.imu_source is not None:
            try:
                self.imu_source.start()
            except Exception as exc:
                print(f"[AVISO] IMU nao iniciou: {exc}", flush=True)

        self._start_time = time.monotonic()
        self._last_print = self._start_time
        self._frame_count = 0
        self._state_count = 0
        if self.logger is not None:
            try:
                self.logger.open()
            except Exception as exc:
                print(f"[AVISO] log CSV nao abriu: {exc}", flush=True)
                self.logger = None

        try:
            while True:
                now = time.monotonic()

                # Check duration
                if now - self._start_time >= self.config.duration_s:
                    break

                # Poll sources
                kinect_obs: Dict[str, KinectObservation] = {}
                frame_bgr = None

                if self.kinect_source is not None:
                    try:
                        ts, frame, obs = self.kinect_source.poll()
                        frame_bgr = frame
                        kinect_obs = obs
                    except Exception:
                        pass

                imu_samples: Dict[str, ImuSample] = {}
                if self.imu_source is not None:
                    try:
                        imu_samples = self.imu_source.poll(now)
                    except Exception:
                        pass

                # O poll do Kinect pode levar ~40 ms; usar o relogio DEPOIS
                # do poll evita amostra IMU "no futuro" em relacao ao frame.
                now = time.monotonic()
                fused_states = self.engine.update(now, kinect_obs, imu_samples)
                self._state_count += len(fused_states)

                if self.logger is not None:
                    for hand, state in fused_states.items():
                        try:
                            self.logger.write_state(
                                now, hand, state,
                                kinect_obs.get(hand),
                                imu_samples.get(hand),
                                self.engine.calibration(hand),
                            )
                        except Exception:
                            pass

                # Visualize if frame available (visualizer always called, window only if show=True)
                if frame_bgr is not None and frame_bgr.size > 0:
                    # Draw overlay for each fused state
                    for indice_mao, (hand, state) in enumerate(
                            fused_states.items()):
                        # Build source statuses for visualization
                        statuses: List[SourceStatus] = []

                        # Kinect status
                        obs = kinect_obs.get(hand)
                        if obs is not None:
                            age_ms = max(0.0, (now - obs.timestamp) * 1000.0)
                            statuses.append(SourceStatus(
                                source=obs.source,
                                age_ms=age_ms,
                                confidence=obs.confidence,
                                valid=obs.valid,
                                reason="" if obs.valid else "invalid observation"
                            ))

                        # IMU status
                        imu = imu_samples.get(hand)
                        if imu is not None:
                            age_ms = max(0.0, (now - imu.timestamp) * 1000.0)
                            statuses.append(SourceStatus(
                                source=imu.source,
                                age_ms=age_ms,
                                confidence=1.0 if imu.valid else 0.0,
                                valid=imu.valid,
                                reason="" if imu.valid else "IMU invalid or stale"
                            ))

                        # Draw using provided visualizer or default
                        if self.visualizer is not None and hasattr(self.visualizer, 'draw_overlay'):
                            self.visualizer.draw_overlay(frame_bgr, state, statuses)
                        else:
                            draw_overlay(
                                frame_bgr, state, statuses,
                                vision_normal_cam=(obs.palm_normal_cam
                                                   if obs is not None else None),
                                panel_origin=(10, 30 + indice_mao * 235),
                            )

                    # Show window only if show=True
                    if self.config.show:
                        try:
                            cv2.imshow('Fusion V1', frame_bgr)
                            key = cv2.waitKey(1) & 0xFF
                            if key == 27:  # ESC
                                break
                        except Exception:
                            pass  # Headless environment

                self._frame_count += 1
                # Evita busy-loop com fonte sintetica; fontes reais ja' gastam
                # o tempo do frame.
                time.sleep(0.002)

                # Periodic status print
                if now - self._last_print >= self.config.print_every_s:
                    elapsed = now - self._start_time
                    fps = self._frame_count / elapsed if elapsed > 0 else 0
                    fontes = []
                    for hand, state in fused_states.items():
                        amostra = imu_samples.get(hand)
                        idade = ("sem IMU" if amostra is None
                                 else f"imu {max(0.0, (now - amostra.timestamp) * 1000.0):.0f}ms")
                        obs = kinect_obs.get(hand)
                        extra = ""
                        if obs is not None and obs.valid:
                            n_vis = np.asarray(obs.palm_normal_cam, float)
                            n_fus = np.asarray(state.normal_cam, float)
                            nv = np.linalg.norm(n_vis); nf = np.linalg.norm(n_fus)
                            if nv > 1e-9 and nf > 1e-9:
                                dot = float(np.clip(np.dot(n_vis / nv, n_fus / nf), -1, 1))
                                extra = f" erro_vis={math.degrees(math.acos(dot)):.1f}deg"
                            cal_diag = self.engine.calibration(hand)
                            if (cal_diag is not None and amostra is not None
                                    and amostra.valid):
                                try:
                                    n_cal = np.asarray(
                                        cal_diag.normal_cam_from_imu(amostra), float)
                                    nc = np.linalg.norm(n_cal)
                                    if nc > 1e-9:
                                        dot2 = float(np.clip(
                                            np.dot(n_vis / nv, n_cal / nc), -1, 1))
                                        extra += (f" erro_cal="
                                                  f"{math.degrees(math.acos(dot2)):.1f}deg")
                                except Exception:
                                    pass
                        fontes.append(f"{hand}:{state.source}({idade}{extra})")
                    print(f"[Fusion] t={elapsed:.1f}s frames={self._frame_count} "
                          f"states={self._state_count} fps={fps:.1f} "
                          f"fontes={fontes}")
                    self._last_print = now

        finally:
            # Cleanup
            if self.kinect_source is not None:
                try:
                    self.kinect_source.close()
                except Exception:
                    pass

            if self.imu_source is not None:
                try:
                    self.imu_source.stop()
                except Exception:
                    pass

            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

            if self.logger is not None:
                try:
                    self.logger.flush()
                    self.logger.close()
                except Exception:
                    pass

        elapsed = time.monotonic() - self._start_time
        return {
            'frames': self._frame_count,
            'states': self._state_count,
            'duration_s': elapsed,
            'fps': self._frame_count / elapsed if elapsed > 0 else 0.0,
        }


# Import cv2 at module level for LiveFusionApp
try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore