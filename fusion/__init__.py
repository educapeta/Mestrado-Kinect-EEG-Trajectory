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

from .adapters import (
    KinectMediaPipeSource, ImuBankSource, SyntheticSources, load_calibrations,
)
from .logging import FusionCsvLogger
from .metrics import summarize_fusion_log, print_summary as print_fusion_summary
from .eeg_export import (FUSED_COLUMNS, FUSED_SCHEMA_VERSION,
                         fused_columns, fused_row, fused_row_dict,
                         append_fused_columns, validate_fused_row)
from .live import FusionEngine, LiveFusionApp, LiveConfig
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
    "FusionCsvLogger", "summarize_fusion_log", "print_fusion_summary",
    "FUSED_COLUMNS", "FUSED_SCHEMA_VERSION", "fused_columns", "fused_row",
    "fused_row_dict", "append_fused_columns", "validate_fused_row",
    "KinectMediaPipeSource", "ImuBankSource", "SyntheticSources",
    "load_calibrations", "FusionEngine", "LiveFusionApp", "LiveConfig",
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