#!/usr/bin/env python
"""
Synthetic demo for quaternion complementary filter.

Simulates a rotating hand with IMU and vision measurements,
runs the filter, and reports angular error before/after fusion.
"""

from __future__ import annotations
import argparse
import sys
import numpy as np

from fusion.complementary import QuaternionComplementaryFilter, ComplementaryConfig
from fusion.contracts import ImuSample, Calibration, FusedState
from fusion.quaternion import (
    normalize, multiply, rotate_vector, quat_from_axis_angle, angle_between_deg
)


def simulate_true_orientation(t: float, freq_yaw: float = 0.2, freq_tilt: float = 0.15) -> np.ndarray:
    """
    Generate true orientation quaternion (body to camera) at time t.

    Simulates yaw rotation + tilt oscillation.
    """
    # Yaw rotation around Z (camera frame)
    yaw_angle = 2.0 * np.pi * freq_yaw * t
    q_yaw = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw_angle)

    # Tilt oscillation around X (camera frame)
    tilt_angle = 0.3 * np.sin(2.0 * np.pi * freq_tilt * t)
    q_tilt = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), tilt_angle)

    # Combined: q = q_yaw * q_tilt (apply tilt first, then yaw)
    q_true = multiply(q_yaw, q_tilt)
    return normalize(q_true)


def simulate_gyro(q_true: np.ndarray, q_prev: np.ndarray, dt: float, noise_dps: float = 0.5) -> np.ndarray:
    """
    Simulate gyroscope reading from true orientation change.

    Args:
        q_true: Current true orientation (body to camera)
        q_prev: Previous true orientation
        dt: Time step
        noise_dps: Gyro noise standard deviation (deg/s)

    Returns:
        Gyro reading in body frame (deg/s)
    """
    # Relative rotation: q_rel = q_prev^-1 * q_true
    q_prev_conj = np.array([q_prev[0], -q_prev[1], -q_prev[2], -q_prev[3]])
    q_rel = multiply(q_prev_conj, q_true)

    # Convert to rotation vector (axis * angle)
    # q = [cos(theta/2), sin(theta/2) * axis]
    theta = 2.0 * np.arccos(np.clip(q_rel[0], -1.0, 1.0))
    if theta < 1e-10:
        omega_cam = np.zeros(3)
    else:
        axis = q_rel[1:] / np.sin(theta / 2.0)
        omega_cam = axis * theta / dt  # rad/s in camera frame

    # Transform to body frame: omega_body = R(q_true)^T * omega_cam
    # R(q) rotates body->cam, so R^T rotates cam->body
    q_conj = np.array([q_true[0], -q_true[1], -q_true[2], -q_true[3]])
    omega_body = rotate_vector(q_conj, omega_cam)

    # Add noise
    omega_body += np.random.normal(0, np.deg2rad(noise_dps), 3)

    # Return in deg/s
    return np.rad2deg(omega_body)


def simulate_accel(q_true: np.ndarray, gravity_cam: np.ndarray, noise_g: float = 0.02) -> np.ndarray:
    """
    Simulate accelerometer reading (specific force in body frame).

    Args:
        q_true: True orientation (body to camera)
        gravity_cam: Gravity vector in camera frame
        noise_g: Accelerometer noise standard deviation (g)

    Returns:
        Accelerometer reading in body frame (g) - points UP (opposite to gravity)
    """
    # Gravity in body frame: g_body = R(q)^T * gravity_cam (points DOWN)
    q_conj = np.array([q_true[0], -q_true[1], -q_true[2], -q_true[3]])
    g_body = rotate_vector(q_conj, gravity_cam)

    # Accelerometer measures specific force = -gravity (points UP)
    accel_body = -g_body

    # Add noise
    accel_body += np.random.normal(0, noise_g, 3)

    return accel_body


def simulate_vision_normal(
    q_true: np.ndarray,
    calibration: Calibration,
    noise_rad: float = 0.05,
    outlier_prob: float = 0.05,
    outlier_mag: float = 1.0
) -> np.ndarray:
    """
    Simulate vision measurement of palm normal in camera frame.

    Args:
        q_true: True orientation (body to camera)
        calibration: Hand-eye calibration
        noise_rad: Angular noise standard deviation (rad)
        outlier_prob: Probability of outlier
        outlier_mag: Outlier magnitude (rad)

    Returns:
        Measured normal in camera frame
    """
    # True normal in camera frame
    n_true = calibration.normal_cam_from_q(q_true)

    # Add angular noise
    if np.random.random() < outlier_prob:
        # Outlier: large random rotation
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(-outlier_mag, outlier_mag)
        q_noise = quat_from_axis_angle(axis, angle)
        n_meas = rotate_vector(q_noise, n_true)
    else:
        # Normal noise: small random rotation
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.normal(0, noise_rad)
        q_noise = quat_from_axis_angle(axis, angle)
        n_meas = rotate_vector(q_noise, n_true)

    return n_meas / np.linalg.norm(n_meas)


def run_demo(
    duration: float = 10.0,
    dt: float = 0.01,
    show: bool = False,
    seed: int = 42
) -> int:
    """
    Run synthetic demo.

    Args:
        duration: Simulation duration (seconds)
        dt: Time step (seconds)
        show: Show OpenCV visualization
        seed: Random seed

    Returns:
        Exit code (0 for success)
    """
    np.random.seed(seed)

    # Configuration
    config = ComplementaryConfig(
        kp_grav=2.0,
        kp_yaw=0.5,
        max_dt=0.1,
        accel_norm_min=0.5,
        accel_norm_max=1.5,
        max_grav_error=0.5,
        max_yaw_error=0.3,
        gravity_cam=np.array([0.0, 0.0, -1.0], dtype=np.float64)
    )

    # Calibration: identity for simplicity (body frame aligned with camera at zero)
    calibration = Calibration(
        hand="right",
        T_cam_world=np.eye(3, dtype=np.float64),
        f_body=np.array([0.0, 0.0, 1.0], dtype=np.float64),  # Palm normal along +Z in body
        yaw_boot_delta_deg=0.0,
        residual_deg=1.0,
        source="synthetic"
    )

    # Initialize filter
    filter = QuaternionComplementaryFilter(config=config, calibration=calibration)
    filter.reset(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))

    # Simulation loop
    n_steps = int(duration / dt)
    errors_predict_only = []
    errors_fused = []

    q_prev = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # For visualization
    if show:
        import cv2
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(n_steps):
        t = i * dt

        # True orientation
        q_true = simulate_true_orientation(t)

        # Simulate IMU
        gyro_dps = simulate_gyro(q_true, q_prev, dt, noise_dps=0.5)
        accel_g = simulate_accel(q_true, config.gravity_cam, noise_g=0.02)

        imu = ImuSample(
            timestamp=t,
            timestamp_us=int(t * 1e6),
            hand="right",
            roll_deg=0.0,  # Not used by filter
            pitch_deg=0.0,
            yaw_deg=0.0,
            accel_g=accel_g,
            gyro_dps=gyro_dps,
            valid=True,
            source="imu_sim"
        )

        # Simulate vision (at lower rate, e.g., 30 Hz)
        vision_dt = 1.0 / 30.0
        if i % int(vision_dt / dt) == 0:
            normal_cam = simulate_vision_normal(q_true, calibration, noise_rad=0.05)
            vision_confidence = 0.9
        else:
            normal_cam = None
            vision_confidence = 0.0

        # Run filter step
        q_est = filter.step(imu, normal_cam, dt, vision_confidence)

        # Compute errors
        # Predict-only: integrate gyro without corrections
        # We'll simulate this by running a second filter with kp_grav=kp_yaw=0
        if i == 0:
            # Initialize predict-only filter
            config_predict = ComplementaryConfig(
                kp_grav=0.0,
                kp_yaw=0.0,
                max_dt=config.max_dt,
                gravity_cam=config.gravity_cam
            )
            filter_predict = QuaternionComplementaryFilter(config=config_predict)
            filter_predict.reset(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))

        q_predict = filter_predict.step(imu, None, dt, 0.0)

        err_predict = angle_between_deg(q_predict, q_true)
        err_fused = angle_between_deg(q_est, q_true)

        errors_predict_only.append(err_predict)
        errors_fused.append(err_fused)

        q_prev = q_true

        # Visualization
        if show and i % 5 == 0:  # Update display every 5 steps
            import cv2
            frame.fill(0)

            # Create dummy statuses
            from fusion.contracts import SourceStatus
            statuses = [
                SourceStatus("imu_sim", dt*1000, 0.9, True, ""),
                SourceStatus("vision_sim", dt*1000, 0.9 if normal_cam is not None else 0.0,
                           normal_cam is not None, "")
            ]

            state = filter.get_fused_state(t, "right", confidence=0.9, source="demo", latency_ms=1.0)
            frame = draw_overlay_demo(frame, state, statuses)

            cv2.imshow("Fusion Demo", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

    if show:
        import cv2
        cv2.destroyAllWindows()

    # Statistics
    errors_predict = np.array(errors_predict_only)
    errors_fused = np.array(errors_fused)

    print(f"\n=== FUSION DEMO RESULTS ===")
    print(f"Duration: {duration:.1f}s, dt: {dt*1000:.1f}ms, Steps: {n_steps}")
    print(f"\nPredict-only (gyro integration):")
    print(f"  Mean error: {np.mean(errors_predict):.3f} deg")
    print(f"  RMS error:  {np.sqrt(np.mean(errors_predict**2)):.3f} deg")
    print(f"  Max error:  {np.max(errors_predict):.3f} deg")
    print(f"\nFused (complementary filter):")
    print(f"  Mean error: {np.mean(errors_fused):.3f} deg")
    print(f"  RMS error:  {np.sqrt(np.mean(errors_fused**2)):.3f} deg")
    print(f"  Max error:  {np.max(errors_fused):.3f} deg")
    print(f"\nImprovement:")
    print(f"  Mean error reduction: {np.mean(errors_predict) - np.mean(errors_fused):.3f} deg "
          f"({(1 - np.mean(errors_fused)/np.mean(errors_predict))*100:.1f}%)")
    print(f"  RMS error reduction:  {np.sqrt(np.mean(errors_predict**2)) - np.sqrt(np.mean(errors_fused**2)):.3f} deg "
          f"({(1 - np.sqrt(np.mean(errors_fused**2))/np.sqrt(np.mean(errors_predict**2)))*100:.1f}%)")

    # Success criteria: fused error should be significantly lower
    mean_improvement = (np.mean(errors_predict) - np.mean(errors_fused)) / np.mean(errors_predict)
    if mean_improvement > 0.3:  # At least 30% improvement
        print("\nRESULT: PASS - Fusion significantly reduces orientation error")
        return 0
    else:
        print("\nRESULT: FAIL - Fusion did not improve sufficiently")
        return 1


def draw_overlay_demo(
    frame_bgr: np.ndarray,
    state: FusedState,
    statuses: list
) -> np.ndarray:
    """Simple overlay for demo (avoids importing fusion.visualization to keep demo standalone)."""
    h, w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    # Text panel
    y = 30
    cv2.putText(frame_bgr, f"Fusion Demo - {state.source}", (10, y), font, font_scale, (255, 0, 255), thickness, cv2.LINE_AA)
    y += 25
    cv2.putText(frame_bgr, f"Confidence: {state.confidence:.2f}", (10, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    y += 25
    cv2.putText(frame_bgr, f"Valid: {'YES' if state.valid else 'NO'}", (10, y), font, font_scale,
                (0, 255, 0) if state.valid else (0, 0, 255), thickness, cv2.LINE_AA)
    y += 25
    q = state.q_body_cam
    cv2.putText(frame_bgr, f"q: [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]", (10, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Center arrow for fused normal
    center = (w // 2, h // 2)
    arrow_len = min(w, h) // 4
    if np.linalg.norm(state.normal_cam) > 1e-6:
        n = state.normal_cam
        end_pt = (int(center[0] + n[0] * arrow_len), int(center[1] - n[1] * arrow_len))
        cv2.arrowedLine(frame_bgr, center, end_pt, (255, 0, 255), 3, tipLength=0.3)

    # Source indicators
    for i, status in enumerate(statuses):
        color = (0, 255, 0) if status.valid else (0, 0, 255)
        if "imu" in status.source.lower():
            color = (0, 255, 255)
        elif "vision" in status.source.lower() or "kinect" in status.source.lower():
            color = (0, 255, 0)
        cv2.putText(frame_bgr, f"{status.source}: {'OK' if status.valid else 'FAIL'}",
                    (10, h - 60 + i * 25), font, 0.5, color, 1, cv2.LINE_AA)

    return frame_bgr


def main():
    parser = argparse.ArgumentParser(description="Synthetic fusion demo")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (s)")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step (s)")
    parser.add_argument("--show", action="store_true", help="Show OpenCV visualization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    return run_demo(args.duration, args.dt, args.show, args.seed)


if __name__ == "__main__":
    sys.exit(main())