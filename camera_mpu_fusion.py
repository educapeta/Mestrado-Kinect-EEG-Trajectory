"""Camera + MPU6050 hand-position prototype.

The camera supplies a metric-corrected relative arm/hand position and joint
angles. The MPU supplies orientation and acceleration. A complementary filter
uses the camera to correct the integrated IMU position drift.

Install once with Python 3.13:
    python -m pip install opencv-python mediapipe

Controls:
    C: capture the current camera pose as the position origin
    R: reset the fused position
    ESC: quit

The camera is monocular: absolute metric depth is not observable by itself.
Set UPPER_ARM_LENGTH_M and FOREARM_LENGTH_M from physical measurements.
"""

from __future__ import annotations

import math
import socket
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

UDP_PORT = 4210
CAMERA_INDEX = 0
UPPER_ARM_LENGTH_M = 0.30
FOREARM_LENGTH_M = 0.25
CAMERA_CORRECTION_GAIN = 0.08
IMU_ACCEL_GAIN = 0.92
DISPLAY_SCALE = 520.0
# ---- zeragem do IMU com as cameras (mesmo comportamento de
# kinect_imu_groundtruth.PositionFusion) -------------------------------
IMU_ZERO_WINDOW = 15
IMU_ZERO_STD_M = 0.008
IMU_ZERO_BIAS_EMA = 0.05
IMU_ZERO_MAX_BIAS_G = 0.5
UP_AXIS = 2
POSE_MODEL_PATH = Path(__file__).with_name("pose_landmarker_lite.task")
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
RIGHT_SHOULDER_INDEX = 12
RIGHT_ELBOW_INDEX = 14
RIGHT_WRIST_INDEX = 16


@dataclass
class ImuSample:
    timestamp_us: int
    received_at: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    accel_g: np.ndarray
    gyro_dps: np.ndarray


class ImuReceiver:
    def __init__(self, port: int):
        self.port = port
        self.running = False
        self.latest: ImuSample | None = None
        self.lock = threading.Lock()
        self.socket: socket.socket | None = None

    def start(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("0.0.0.0", self.port))
        self.socket.settimeout(0.2)
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        assert self.socket is not None
        while self.running:
            try:
                payload, _ = self.socket.recvfrom(512)
                values = np.fromstring(payload.decode("ascii").strip(), sep=",")
                received_at = time.monotonic()
                if values.size < 4 or not np.isfinite(values[:4]).all():
                    continue
                accel = values[4:7] if values.size >= 7 else np.zeros(3)
                gyro = values[7:10] if values.size >= 10 else np.zeros(3)
                if not np.isfinite(accel).all() or not np.isfinite(gyro).all():
                    continue
                sample = ImuSample(
                    timestamp_us=int(values[0]),
                    received_at=received_at,
                    roll_deg=float(values[1]),
                    pitch_deg=float(values[2]),
                    yaw_deg=float(values[3]),
                    accel_g=accel.astype(np.float64),
                    gyro_dps=gyro.astype(np.float64),
                )
                with self.lock:
                    self.latest = sample
            except (OSError, UnicodeDecodeError, ValueError):
                continue

    def get_latest(self) -> ImuSample | None:
        with self.lock:
            return self.latest

    def stop(self) -> None:
        self.running = False
        if self.socket is not None:
            self.socket.close()
        self.socket = None


def rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def landmark_vector(landmark, image_width: int, image_height: int) -> np.ndarray:
    # MediaPipe image y points down; convert to a camera frame with y up.
    return np.array([
        landmark.x * image_width,
        (1.0 - landmark.y) * image_height,
        -landmark.z * image_width,
    ], dtype=np.float64)


def metric_arm_points(landmarks, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shoulder = landmark_vector(landmarks[RIGHT_SHOULDER_INDEX], width, height)
    elbow = landmark_vector(landmarks[RIGHT_ELBOW_INDEX], width, height)
    wrist = landmark_vector(landmarks[RIGHT_WRIST_INDEX], width, height)

    upper = elbow - shoulder
    forearm = wrist - elbow
    upper_norm = np.linalg.norm(upper)
    forearm_norm = np.linalg.norm(forearm)
    if upper_norm < 1e-6 or forearm_norm < 1e-6:
        raise ValueError("Pose arm too small or not visible")

    # Keep camera direction and correct each segment independently to known lengths.
    elbow_metric = shoulder + upper / upper_norm * UPPER_ARM_LENGTH_M
    wrist_metric = elbow_metric + forearm / forearm_norm * FOREARM_LENGTH_M
    return shoulder, elbow_metric, wrist_metric


def angle_between(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-9:
        return float("nan")
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class PositionFusion:
    """Fusao IMU + camera com ZERAGEM (espelho de kinect_imu_groundtruth)."""

    def __init__(self):
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.last_imu_timestamp_us: int | None = None
        self.camera_origin: np.ndarray | None = None
        self.accel_bias = np.zeros(3, dtype=np.float64)
        self.zero_lock = False
        self.bias_estimated = False
        self._camera_window = []
        self._camera_lost_frames = 0

    def reset(self) -> None:
        self.position[:] = 0.0
        self.velocity[:] = 0.0
        self.last_imu_timestamp_us = None
        self.camera_origin = None
        self.zero_lock = False
        self._camera_window = []
        self._camera_lost_frames = 0

    @property
    def absolute_position(self) -> np.ndarray | None:
        if self.camera_origin is None:
            return None
        return self.camera_origin + self.position

    def _camera_is_stable(self, camera_position) -> bool:
        self._camera_window.append(camera_position.copy())
        if len(self._camera_window) > IMU_ZERO_WINDOW:
            self._camera_window.pop(0)
        if len(self._camera_window) < IMU_ZERO_WINDOW:
            return False
        window = np.asarray(self._camera_window, np.float64)
        return bool(np.all(window.std(axis=0) < IMU_ZERO_STD_M))

    def _update_accel_bias(self, sample) -> None:
        rot = rotation_matrix(sample.roll_deg, sample.pitch_deg,
                              sample.yaw_deg)
        residual_world = rot @ (sample.accel_g * 9.80665)
        residual_world[UP_AXIS] -= 9.80665
        hint = (rot.T @ residual_world) / 9.80665
        hint = np.clip(hint, -IMU_ZERO_MAX_BIAS_G, IMU_ZERO_MAX_BIAS_G)
        if np.isfinite(hint).all():
            self.accel_bias += IMU_ZERO_BIAS_EMA * (hint - self.accel_bias)
            self.bias_estimated = True

    def update(self, sample: ImuSample | None,
               camera_position: np.ndarray | None) -> np.ndarray:
        if sample is not None and self.last_imu_timestamp_us is not None:
            dt = (sample.timestamp_us - self.last_imu_timestamp_us) * 1e-6
            if 0.0 < dt < 0.1:
                acceleration_world = rotation_matrix(
                    sample.roll_deg, sample.pitch_deg, sample.yaw_deg
                ) @ ((sample.accel_g - self.accel_bias) * 9.80665)
                acceleration_world[UP_AXIS] -= 9.80665
                self.velocity += acceleration_world * dt
                self.position += (self.velocity * dt
                                  + 0.5 * acceleration_world * dt * dt)
        if sample is not None:
            self.last_imu_timestamp_us = sample.timestamp_us

        self.zero_lock = False
        if camera_position is not None:
            if self.camera_origin is None:
                self.camera_origin = camera_position.copy()
            camera_relative = camera_position - self.camera_origin
            if self._camera_is_stable(camera_position):
                # ---- zeragem: mao parada e vista pela camera ----
                self.zero_lock = True
                self._camera_lost_frames = 0
                self.position[:] = camera_relative
                self.velocity[:] = 0.0
                if sample is not None:
                    self._update_accel_bias(sample)
            else:
                self._camera_lost_frames = 0
                self.position = (
                    IMU_ACCEL_GAIN * self.position
                    + CAMERA_CORRECTION_GAIN * camera_relative
                )
                self.velocity *= 0.98
        else:
            self._camera_lost_frames += 1
            if self._camera_lost_frames > 3:
                self._camera_window = []
        return self.position.copy()


def ensure_pose_model() -> Path:
    if not POSE_MODEL_PATH.exists():
        print("Baixando modelo MediaPipe de pose...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    return POSE_MODEL_PATH


def main() -> None:
    receiver = ImuReceiver(UDP_PORT)
    receiver.start()
    fusion = PositionFusion()
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError("Nao foi possivel abrir a camera do computador")

    base_options = mp_python.BaseOptions(model_asset_path=str(ensure_pose_model()))
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    with mp_vision.PoseLandmarker.create_from_options(pose_options) as pose:
        try:
            frame_timestamp_ms = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                frame_timestamp_ms += 33
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = pose.detect_for_video(image, frame_timestamp_ms)
                camera_position = None
                elbow_angle = float("nan")

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    try:
                        shoulder, elbow, wrist = metric_arm_points(landmarks, width, height)
                        camera_position = wrist
                        elbow_angle = angle_between(shoulder - elbow, wrist - elbow)
                        for first, second in ((12, 14), (14, 16)):
                            first_point = (int(landmarks[first].x * width), int(landmarks[first].y * height))
                            second_point = (int(landmarks[second].x * width), int(landmarks[second].y * height))
                            cv2.line(frame, first_point, second_point, (0, 255, 0), 3)
                        wrist_point = (int(landmarks[RIGHT_WRIST_INDEX].x * width), int(landmarks[RIGHT_WRIST_INDEX].y * height))
                        cv2.circle(frame, wrist_point, 8, (0, 255, 255), -1)
                    except ValueError:
                        pass

                sample = receiver.get_latest()
                fused = fusion.update(sample, camera_position)
                if sample is None:
                    imu_text = "IMU: aguardando UDP"
                else:
                    imu_text = f"IMU R/P/Y: {sample.roll_deg:.1f}/{sample.pitch_deg:.1f}/{sample.yaw_deg:.1f}"
                lines = [
                    imu_text,
                    f"Camera wrist XYZ: {camera_position if camera_position is not None else '---'}",
                    f"Fusao P XYZ [m]: {fused[0]:.3f}, {fused[1]:.3f}, {fused[2]:.3f}",
                    f"Angulo cotovelo: {elbow_angle:.1f} deg",
                    "C: origem | R: reset | ESC: sair",
                ]
                for line_index, text in enumerate(lines):
                    cv2.putText(frame, text, (20, 30 + line_index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
                cv2.imshow("Camera + MPU6050 - fusao espacial", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c") and camera_position is not None:
                    fusion.camera_origin = camera_position.copy()
                    fusion.position[:] = 0.0
                    fusion.velocity[:] = 0.0
                elif key == ord("r"):
                    fusion.reset()
                elif key == 27:
                    break
        finally:
            receiver.stop()
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
