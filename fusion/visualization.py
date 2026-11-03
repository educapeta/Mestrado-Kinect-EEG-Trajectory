"""
Visualization utilities for sensor fusion overlay.

Draws fusion state and source status on OpenCV frames.
"""

from __future__ import annotations
from typing import Optional, List
import numpy as np
import cv2

from .contracts import FusedState, SourceStatus, Calibration


# Color constants (BGR format for OpenCV)
COLOR_FUSED = (255, 0, 255)    # Magenta
COLOR_VISION = (0, 255, 0)     # Green
COLOR_IMU = (0, 255, 255)      # Yellow
COLOR_TEXT = (255, 255, 255)   # White
COLOR_BG = (0, 0, 0)           # Black
COLOR_VALID = (0, 255, 0)      # Green
COLOR_INVALID = (0, 0, 255)    # Red


def draw_overlay(
    frame_bgr: np.ndarray,
    state: FusedState,
    statuses: List[SourceStatus],
    calibration: Optional[Calibration] = None,
    source_text: Optional[str] = None
) -> np.ndarray:
    """
    Draw fusion overlay on frame.

    Args:
        frame_bgr: Input BGR frame (modified in place and returned)
        state: Fused state estimate
        statuses: List of source statuses
        calibration: Optional calibration for additional info
        source_text: Optional custom source text

    Returns:
        Frame with overlay drawn (same array as input)
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]

    # Draw text panel in top-left corner
    panel_x = 10
    panel_y = 30
    line_height = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    # Background rectangle for text panel
    panel_lines = 5 + len(statuses)
    panel_height = panel_lines * line_height + 10
    panel_width = 320
    cv2.rectangle(
        frame_bgr,
        (panel_x - 5, panel_y - 25),
        (panel_x + panel_width, panel_y + panel_height),
        COLOR_BG,
        -1
    )
    cv2.rectangle(
        frame_bgr,
        (panel_x - 5, panel_y - 25),
        (panel_x + panel_width, panel_y + panel_height),
        COLOR_FUSED,
        1
    )

    # Title
    title = f"Fusion: {state.source if state.source else 'unknown'}"
    if source_text:
        title += f" ({source_text})"
    cv2.putText(
        frame_bgr, title,
        (panel_x, panel_y),
        font, font_scale, COLOR_FUSED, thickness, cv2.LINE_AA
    )
    panel_y += line_height

    # Confidence
    conf_color = COLOR_VALID if state.confidence > 0.5 else COLOR_INVALID
    cv2.putText(
        frame_bgr,
        f"Confidence: {state.confidence:.2f}",
        (panel_x, panel_y),
        font, font_scale, conf_color, thickness, cv2.LINE_AA
    )
    panel_y += line_height

    # Validity
    valid_color = COLOR_VALID if state.valid else COLOR_INVALID
    cv2.putText(
        frame_bgr,
        f"Valid: {'YES' if state.valid else 'NO'}",
        (panel_x, panel_y),
        font, font_scale, valid_color, thickness, cv2.LINE_AA
    )
    panel_y += line_height

    # Latency
    cv2.putText(
        frame_bgr,
        f"Latency: {state.latency_ms:.1f} ms",
        (panel_x, panel_y),
        font, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA
    )
    panel_y += line_height

    # Quaternion (compact)
    q = state.q_body_cam
    cv2.putText(
        frame_bgr,
        f"q: [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]",
        (panel_x, panel_y),
        font, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA
    )
    panel_y += line_height

    # Source statuses
    for status in statuses:
        age_str = f"{status.age_ms:.0f}ms" if status.age_ms < 1000 else f"{status.age_ms/1000:.1f}s"
        conf_str = f"{status.confidence:.2f}"
        valid_str = "OK" if status.valid else "FAIL"
        color = COLOR_VALID if status.valid else COLOR_INVALID
        text = f"{status.source}: age={age_str} conf={conf_str} {valid_str}"
        if status.reason:
            text += f" ({status.reason})"
        cv2.putText(
            frame_bgr, text,
            (panel_x, panel_y),
            font, font_scale, color, thickness, cv2.LINE_AA
        )
        panel_y += line_height

    # Draw 2D arrow of projected normal at center
    center = (w // 2, h // 2)
    arrow_length = min(w, h) // 4

    # Fused normal (magenta)
    if state.valid and np.linalg.norm(state.normal_cam) > 1e-6:
        n = state.normal_cam
        # Project to 2D (ignore Z, flip Y for image coordinates)
        end_pt = (
            int(center[0] + n[0] * arrow_length),
            int(center[1] - n[1] * arrow_length)  # Flip Y
        )
        cv2.arrowedLine(
            frame_bgr, center, end_pt,
            COLOR_FUSED, 3, tipLength=0.3
        )
        cv2.putText(
            frame_bgr, "FUSED",
            (end_pt[0] + 5, end_pt[1]),
            font, 0.45, COLOR_FUSED, 1, cv2.LINE_AA
        )

    # Vision normal (green) - if available in statuses
    for status in statuses:
        if status.source.lower() in ("kinect", "mediapipe", "vision") and status.valid:
            # We don't have the actual normal here, but we can indicate vision is active
            # Draw a small indicator
            cv2.circle(frame_bgr, (center[0] + arrow_length + 20, center[1]), 8, COLOR_VISION, -1)
            cv2.putText(
                frame_bgr, "VIS",
                (center[0] + arrow_length + 30, center[1] + 4),
                font, 0.4, COLOR_VISION, 1, cv2.LINE_AA
            )
            break

    # IMU indicator (yellow)
    for status in statuses:
        if status.source.lower() in ("imu", "imu_udp") and status.valid:
            cv2.circle(frame_bgr, (center[0] - arrow_length - 20, center[1]), 8, COLOR_IMU, -1)
            cv2.putText(
                frame_bgr, "IMU",
                (center[0] - arrow_length - 50, center[1] + 4),
                font, 0.4, COLOR_IMU, 1, cv2.LINE_AA
            )
            break

    return frame_bgr


def draw_calibration_overlay(
    frame_bgr: np.ndarray,
    calibration: Calibration,
    position: tuple = (10, 10)
) -> np.ndarray:
    """
    Draw calibration info overlay.

    Args:
        frame_bgr: Input BGR frame
        calibration: Calibration data
        position: Top-left position for text

    Returns:
        Frame with overlay
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20

    lines = [
        f"Calibration: {calibration.source}",
        f"Hand: {calibration.hand}",
        f"Residual: {calibration.residual_deg:.2f} deg" if calibration.residual_deg else "Residual: N/A",
        f"Yaw boot delta: {calibration.yaw_boot_delta_deg:.2f} deg",
        f"f_body: [{calibration.f_body[0]:.3f}, {calibration.f_body[1]:.3f}, {calibration.f_body[2]:.3f}]"
    ]

    for i, line in enumerate(lines):
        cv2.putText(
            frame_bgr, line,
            (x, y + i * line_height),
            font, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA
        )

    return frame_bgr