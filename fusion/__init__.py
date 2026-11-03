"""
Fusion package for multi-sensor hand tracking.

Provides shared contracts and quaternion math for sensor fusion pipeline.
"""

from .contracts import (
    KinectObservation,
    MediaPipeObservation,
    ImuSample,
    Calibration,
    FusedState,
    SourceStatus,
)

from .sync import SampleBuffer, SyncBuffer, SyncSnapshot
from .calibration import CalibrationManager, load_calibration, save_calibration
from .sensor_manager import SensorDecision, SensorManager
from .complementary import ComplementaryConfig, QuaternionComplementaryFilter
from .visualization import draw_overlay

from .quaternion import (
    normalize,
    multiply,
    conjugate,
    rotate_vector,
    quat_from_axis_angle,
    exp_so3,
    slerp,
    from_euler_zyx_deg,
    to_matrix,
    from_matrix,
    angle_between_deg,
    quat_from_two_vectors,
)

__all__ = [
    "SampleBuffer", "SyncBuffer", "SyncSnapshot",
    "CalibrationManager", "load_calibration", "save_calibration",
    "SensorDecision", "SensorManager",
    "ComplementaryConfig", "QuaternionComplementaryFilter",
    "draw_overlay",
    # Contracts
    "KinectObservation",
    "MediaPipeObservation",
    "ImuSample",
    "Calibration",
    "FusedState",
    "SourceStatus",
    # Quaternion functions
    "normalize",
    "multiply",
    "conjugate",
    "rotate_vector",
    "quat_from_axis_angle",
    "exp_so3",
    "slerp",
    "from_euler_zyx_deg",
    "to_matrix",
    "from_matrix",
    "angle_between_deg",
    "quat_from_two_vectors",
]

# Version for compatibility checking
__version__ = "0.1.0"