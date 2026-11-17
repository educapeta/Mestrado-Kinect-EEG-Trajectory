"""
Pure core vision backends for palm normal estimation.

Implements controlled bifurcation between KinectDepth and MultiView
for the palm NORMAL (not full pose). No hardware dependencies; numpy only.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class VisionNormal:
    """
    Palm normal estimate from a vision backend.

    Attributes:
        timestamp: Monotonic timestamp in seconds
        hand: Hand identifier ("left" or "right")
        normal_cam: Palm normal in camera frame (3,) unit vector
        position_cam: Palm position in camera frame (3,) meters, or None
        confidence: Detection confidence [0, 1]
        sigma_deg: Angular uncertainty estimate in degrees
        rms_m: RMS distance of points to fitted plane in meters
        edge_on: True if normal is nearly perpendicular to optical axis (|n_z| small)
        source: Source identifier string (e.g., "kinect_depth", "multiview")
        valid: Whether the estimate is valid
    """
    timestamp: float
    hand: str
    normal_cam: np.ndarray
    position_cam: Optional[np.ndarray] = None
    confidence: float = 0.0
    sigma_deg: float = 180.0
    rms_m: float = 0.0
    edge_on: bool = False
    source: str = ""
    valid: bool = False

    def __post_init__(self):
        """Ensure arrays are float64 and correct shape."""
        if self.normal_cam is not None:
            self.normal_cam = np.asarray(self.normal_cam, dtype=np.float64).reshape(3)
        if self.position_cam is not None:
            self.position_cam = np.asarray(self.position_cam, dtype=np.float64).reshape(3)


def robust_plane_normal(
    points_3d: np.ndarray,
    weights: Optional[np.ndarray] = None,
    side: str = "right",
    normal_anterior: Optional[np.ndarray] = None,
    max_iter: int = 3,
    outlier_threshold_sigma: float = 2.5,
    min_points: int = 4,
    edge_on_cos_threshold: float = 0.15,
    min_singular_ratio: float = 0.3,
) -> Optional[tuple[np.ndarray, float, bool, float]]:
    """
    Robust plane normal estimation via weighted SVD with iterative outlier rejection.

    Args:
        points_3d: Array of shape (N, 3) with 3D points in camera frame (meters)
        weights: Optional weights of shape (N,) for weighted PCA. Higher = more trust.
        side: Hand side ("left" or "right") for anatomical sign convention
        normal_anterior: Previous frame normal for temporal continuity (3,)
        max_iter: Maximum iterations for outlier rejection
        outlier_threshold_sigma: Points with distance > threshold * std are outliers
        min_points: Minimum finite points required after outlier rejection
        edge_on_cos_threshold: |n_z| below this -> edge_on flag (cos ~81 deg)
        min_singular_ratio: Minimum ratio of smallest to largest singular value
                           to consider the fit a valid plane (not line/point)

    Returns:
        Tuple of (normal, rms_m, edge_on, sigma_deg) or None if failed

    Sign resolution:
        SVD gives a normal with arbitrary sign (a plane has two sides).
        This routine does NOT infer an anatomical sign from the hand side --
        palm/back direction depends on the pose, not on being left/right.
        Priority:
        1. temporal continuity: if normal_anterior is given and dot < 0,
           flip to match the previous normal;
        2. otherwise use a deterministic canonical sign (largest absolute
           component positive). The caller (fusion/IMU/geometry) is responsible
           for assigning anatomical palm/back sign.

    Edge-on detection:
        When |n_z| is small (normal nearly perpendicular to optical axis),
        depth uncertainty projects strongly to normal uncertainty.
        We flag edge_on = True and inflate sigma_deg accordingly.

    Sigma_deg derivation:
        Base sigma from RMS: sigma_rms_deg ≈ atan(rms / depth) * 180/pi
        Inflated by: 1/sqrt(N_eff) for point count, 1/|n_z| for edge-on,
        and condition number of the point covariance.
    """
    # Filter finite points
    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None

    finite_mask = np.isfinite(pts).all(axis=1)
    if not np.any(finite_mask):
        return None

    pts = pts[finite_mask]
    n_pts = len(pts)

    if n_pts < min_points:
        return None

    # Handle weights
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        w = w[finite_mask]
        if len(w) != n_pts:
            w = None
        else:
            # Normalize weights
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum * n_pts
            else:
                w = None
    else:
        w = None

    # Iterative outlier rejection
    inlier_mask = np.ones(n_pts, dtype=bool)

    # Depth pre-filter: remove points far from median depth (likely background)
    depths = pts[:, 2]
    median_depth = float(np.median(depths))
    depth_mad = float(np.median(np.abs(depths - median_depth)))
    depth_scale = depth_mad * 1.4826 if depth_mad > 1e-6 else float(np.std(depths))
    if depth_scale > 0:
        depth_threshold = 3.0 * depth_scale
        depth_inlier = np.abs(depths - median_depth) <= depth_threshold
        if depth_inlier.sum() >= min_points:
            inlier_mask &= depth_inlier

    for _ in range(max_iter):
        if inlier_mask.sum() < min_points:
            return None

        current_pts = pts[inlier_mask]
        current_n = len(current_pts)
        current_w = w[inlier_mask] if w is not None else None

        # Weighted centroid
        if current_w is not None:
            centroid = np.average(current_pts, axis=0, weights=current_w)
        else:
            centroid = current_pts.mean(axis=0)

        # Center points
        centered = current_pts - centroid

        # Weighted covariance / SVD
        if current_w is not None:
            # Weighted SVD via covariance
            W_sqrt = np.sqrt(current_w)
            weighted_centered = centered * W_sqrt[:, np.newaxis]
            try:
                _, s_iter, vt = np.linalg.svd(weighted_centered, full_matrices=False)
            except np.linalg.LinAlgError:
                return None
        else:
            try:
                _, s_iter, vt = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                return None

        # Check singular values for planarity
        # For a plane: s[0] >= s[1] >> s[2] (two large, one small)
        # For a line: s[0] >> s[1] >= s[2] (one large, two small)
        # For isotropic: s[0] ~ s[1] ~ s[2]
        if len(s_iter) >= 3 and s_iter[0] > 1e-12:
            # Reject if not plane-like: s[2] should be much smaller than s[1]
            if s_iter[2] / max(s_iter[1], 1e-12) > min_singular_ratio:
                return None
            # Also reject colinear: s[1] should not be too small relative to s[0]
            if s_iter[1] / s_iter[0] < 0.01:
                return None

        normal = vt[-1].copy()
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            return None
        normal = normal / norm

        # Distances to plane
        distances = centered @ normal
        rms = float(np.sqrt(np.mean(distances**2)))

        # Outlier rejection based on distance MAD (robust)
        if current_n > min_points + 1:
            # Use MAD for robust threshold
            median_dist = float(np.median(distances))
            mad = float(np.median(np.abs(distances - median_dist)))
            # MAD * 1.4826 ≈ std for normal distribution
            robust_std = mad * 1.4826 if mad > 1e-9 else float(np.std(distances))
            if robust_std > 1e-9:
                threshold = outlier_threshold_sigma * robust_std
                new_inlier = np.abs(distances - median_dist) <= threshold
                if new_inlier.sum() >= min_points and new_inlier.sum() < current_n:
                    # Map back to original indices
                    inlier_indices = np.where(inlier_mask)[0]
                    inlier_mask[inlier_indices] = new_inlier
                    continue

        # Converged
        break

    # Final fit on inliers
    final_pts = pts[inlier_mask]
    final_n = len(final_pts)
    final_w = w[inlier_mask] if w is not None else None

    if final_w is not None:
        centroid = np.average(final_pts, axis=0, weights=final_w)
    else:
        centroid = final_pts.mean(axis=0)

    centered = final_pts - centroid

    if final_w is not None:
        W_sqrt = np.sqrt(final_w)
        weighted_centered = centered * W_sqrt[:, np.newaxis]
        try:
            _, s_vals, vt = np.linalg.svd(weighted_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
    else:
        try:
            _, s_vals, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

    # Reject colinear/degenerate: smallest singular value must be small relative to middle
    # and middle must not be too small relative to largest (not a line)
    if len(s_vals) >= 3 and s_vals[0] > 1e-12:
        if s_vals[2] / max(s_vals[1], 1e-12) > min_singular_ratio:
            return None
        if s_vals[1] / s_vals[0] < 0.01:
            return None

    normal = vt[-1].copy()
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal = normal / norm

    # RMS on final inliers
    distances = centered @ normal
    rms = float(np.sqrt(np.mean(distances**2)))

    # ---- SIGN RESOLUTION ----
    # O lado da mao nao define o sinal: palma/dorso depende da POSE. Usamos
    # continuidade e, sem anterior, um sinal canonico deterministico.
    if normal_anterior is not None:
        anterior = np.asarray(normal_anterior, dtype=np.float64).reshape(3)
        anterior_norm = np.linalg.norm(anterior)
        if anterior_norm > 1e-9:
            anterior = anterior / anterior_norm
            if float(np.dot(normal, anterior)) < 0.0:
                normal = -normal
    else:
        eixo = int(np.argmax(np.abs(normal)))
        if normal[eixo] < 0.0:
            normal = -normal

    normal = normal / np.linalg.norm(normal)

    # ---- EDGE-ON DETECTION ----
    # Camera optical axis is +Z (into scene). Edge-on when normal ~perp to Z.
    nz_abs = abs(normal[2])
    edge_on = nz_abs < edge_on_cos_threshold

    # ---- SIGMA_DEG ESTIMATION ----
    # Base from RMS: angle ≈ atan(rms / depth)
    depth = float(np.median(final_pts[:, 2])) if final_n > 0 else 1.0
    depth = max(depth, 0.1)  # Clamp

    sigma_rms_deg = float(np.degrees(np.arctan2(rms, depth)))

    # Inflation factors
    # 1. Point count: 1/sqrt(N_eff)
    n_eff = final_n
    if final_w is not None:
        # Effective sample size for weighted data
        n_eff = float(final_w.sum()**2 / (final_w**2).sum())
    count_factor = 1.0 / np.sqrt(max(n_eff, 1.0))

    # 2. Geometry: condition number of covariance (spread of points)
    if final_w is not None:
        W_sqrt = np.sqrt(final_w)
        cov = (centered * W_sqrt[:, np.newaxis]).T @ (centered * W_sqrt[:, np.newaxis])
    else:
        cov = centered.T @ centered
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-12]
    if len(eigvals) >= 2:
        cond = eigvals[-1] / eigvals[0] if eigvals[0] > 0 else 10.0
        geom_factor = np.sqrt(min(cond, 10.0))  # Cap at sqrt(10)
    else:
        geom_factor = 2.0

    # 3. Edge-on inflation: 1/|n_z| when small
    edge_factor = 1.0
    if edge_on:
        edge_factor = 1.0 / max(nz_abs, 0.05)  # Cap at 20x

    # Combined sigma
    sigma_deg = sigma_rms_deg * count_factor * geom_factor * edge_factor
    # Floor at 0.5 deg, cap at 180
    sigma_deg = float(np.clip(sigma_deg, 0.5, 180.0))

    return normal, rms, edge_on, sigma_deg


class KinectDepthBackend:
    """
    Kinect depth-based palm normal backend.

    Uses depth points from the hand region (e.g., dorsal landmarks projected
    to depth) to fit a plane via robust_plane_normal.
    """

    name = "kinect_depth"

    def compute(
        self,
        points_3d: np.ndarray,
        hand: str,
        timestamp: float,
        normal_anterior: Optional[np.ndarray] = None,
    ) -> VisionNormal:
        """
        Compute palm normal from Kinect depth points.

        Args:
            points_3d: (N, 3) array of 3D points in camera frame (meters)
            hand: "left" or "right"
            timestamp: Monotonic timestamp in seconds
            normal_anterior: Previous frame normal for continuity

        Returns:
            VisionNormal with the estimate
        """
        result = robust_plane_normal(
            points_3d=points_3d,
            weights=None,  # Kinect depth: uniform weighting
            side=hand,
            normal_anterior=normal_anterior,
        )

        if result is None:
            return VisionNormal(
                timestamp=timestamp,
                hand=hand,
                normal_cam=np.zeros(3, dtype=np.float64),
                position_cam=None,
                confidence=0.0,
                sigma_deg=180.0,
                rms_m=0.0,
                edge_on=False,
                source=self.name,
                valid=False,
            )

        normal, rms, edge_on, sigma_deg = result

        # Position = median of points (robust to outliers)
        finite_pts = points_3d[np.isfinite(points_3d).all(axis=1)]
        if len(finite_pts) > 0:
            position = np.median(finite_pts, axis=0)
        else:
            position = None

        # Confidence: inverse of sigma, scaled
        confidence = float(np.clip(1.0 - sigma_deg / 90.0, 0.0, 1.0))
        if edge_on:
            confidence *= 0.5  # Penalize edge-on

        return VisionNormal(
            timestamp=timestamp,
            hand=hand,
            normal_cam=normal,
            position_cam=position,
            confidence=confidence,
            sigma_deg=sigma_deg,
            rms_m=rms,
            edge_on=edge_on,
            source=self.name,
            valid=True,
        )


class MultiViewBackend:
    """
    Multi-view (stereo/triangulation) palm normal backend.

    Uses 3D points from multi-view triangulation with reprojection errors
    as weights for the plane fit.
    """

    name = "multiview"

    def compute(
        self,
        points_3d: np.ndarray,
        hand: str,
        timestamp: float,
        reproj_errors: Optional[np.ndarray] = None,
        normal_anterior: Optional[np.ndarray] = None,
    ) -> VisionNormal:
        """
        Compute palm normal from multi-view triangulated points.

        Args:
            points_3d: (N, 3) array of 3D points in camera frame (meters)
            hand: "left" or "right"
            timestamp: Monotonic timestamp in seconds
            reproj_errors: (N,) reprojection errors in pixels. Lower = better.
            normal_anterior: Previous frame normal for continuity

        Returns:
            VisionNormal with the estimate
        """
        # Convert reprojection errors to weights: weight = 1 / (error^2 + eps)
        weights = None
        if reproj_errors is not None:
            errors = np.asarray(reproj_errors, dtype=np.float64).reshape(-1)
            if len(errors) == len(points_3d):
                # Filter finite
                finite_mask = np.isfinite(points_3d).all(axis=1) & np.isfinite(errors)
                if np.any(finite_mask):
                    errors = errors[finite_mask]
                    # Weight = exp(-error / scale), scale ~ 2 pixels
                    weights = np.exp(-errors / 2.0)

        result = robust_plane_normal(
            points_3d=points_3d,
            weights=weights,
            side=hand,
            normal_anterior=normal_anterior,
        )

        if result is None:
            return VisionNormal(
                timestamp=timestamp,
                hand=hand,
                normal_cam=np.zeros(3, dtype=np.float64),
                position_cam=None,
                confidence=0.0,
                sigma_deg=180.0,
                rms_m=0.0,
                edge_on=False,
                source=self.name,
                valid=False,
            )

        normal, rms, edge_on, sigma_deg = result

        # Position = median of points
        finite_pts = points_3d[np.isfinite(points_3d).all(axis=1)]
        if len(finite_pts) > 0:
            position = np.median(finite_pts, axis=0)
        else:
            position = None

        # Confidence from reprojection errors
        if reproj_errors is not None and len(reproj_errors) > 0:
            finite_errors = reproj_errors[np.isfinite(reproj_errors)]
            if len(finite_errors) > 0:
                median_error = float(np.median(finite_errors))
                # Confidence drops with median reprojection error
                confidence = float(np.clip(1.0 - median_error / 10.0, 0.0, 1.0))
            else:
                confidence = 0.0
        else:
            # No reproj errors -> base on sigma
            confidence = float(np.clip(1.0 - sigma_deg / 90.0, 0.0, 1.0))

        if edge_on:
            confidence *= 0.5

        return VisionNormal(
            timestamp=timestamp,
            hand=hand,
            normal_cam=normal,
            position_cam=position,
            confidence=confidence,
            sigma_deg=sigma_deg,
            rms_m=rms,
            edge_on=edge_on,
            source=self.name,
            valid=True,
        )


def compare_backends(
    points_3d: np.ndarray,
    hand: str,
    timestamp: float,
    reproj_errors: Optional[np.ndarray] = None,
    normal_anterior: Optional[np.ndarray] = None,
) -> dict:
    """
    Compare both backends on the same input points.

    Args:
        points_3d: (N, 3) array of 3D points in camera frame
        hand: "left" or "right"
        timestamp: Monotonic timestamp
        reproj_errors: Optional reprojection errors for MultiView
        normal_anterior: Previous normal for continuity

    Returns:
        Dict with:
            - "kinect_depth": VisionNormal from KinectDepthBackend
            - "multiview": VisionNormal from MultiViewBackend
            - "angle_deg": Angle between the two normals (degrees)
            - "agree": True if angle < 15 degrees
    """
    kinect_backend = KinectDepthBackend()
    mv_backend = MultiViewBackend()

    vn_kinect = kinect_backend.compute(points_3d, hand, timestamp, normal_anterior)
    vn_mv = mv_backend.compute(points_3d, hand, timestamp, reproj_errors, normal_anterior)

    # Angle entre os EIXOS das normais (sinal anatomico e' responsabilidade
    # de quem consome; aqui comparamos a direcao do plano).
    angle_deg = 180.0
    sign_agree = False
    if vn_kinect.valid and vn_mv.valid:
        n1 = vn_kinect.normal_cam
        n2 = vn_mv.normal_cam
        dot_raw = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        sign_agree = dot_raw >= 0.0
        angle_deg = float(np.degrees(np.arccos(abs(dot_raw))))

    return {
        "kinect_depth": vn_kinect,
        "multiview": vn_mv,
        "angle_deg": angle_deg,
        "sign_agree": sign_agree,
        "agree": angle_deg < 15.0,
    }