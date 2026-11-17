"""
Tests for fusion/live.py

Tests FusionEngine and LiveFusionApp without requiring hardware.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fusion.live import (
    LiveConfig,
    FusionEngine,
    LiveFusionApp,
)
from fusion.contracts import (
    KinectObservation,
    ImuSample,
    Calibration,
    FusedState,
    SourceStatus,
)
from fusion.complementary import ComplementaryConfig
from fusion.adapters import SyntheticSources


# =============================================================================
# Fake Sources for LiveFusionApp tests
# =============================================================================

class FakeKinectSource:
    """Fake Kinect source for testing."""

    def __init__(self, hands=('right', 'left'), valid=True):
        self.hands = hands
        self.valid = valid
        self._started = False
        self._frame_count = 0

    def start(self):
        self._started = True

    def close(self):
        self._started = False

    def poll(self):
        if not self._started:
            return time.monotonic(), None, {}

        self._frame_count += 1
        now = time.monotonic()

        # Create simple BGR frame
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        observations = {}
        for side in self.hands:
            normal = np.array([0.0, 0.0, -1.0], dtype=np.float64) if self.valid else None
            direction = np.array([0.0, 1.0, 0.0], dtype=np.float64) if self.valid else None
            wrist = np.array([0.1 if side == 'right' else -0.1, 0.0, 1.0], dtype=np.float64)

            obs = KinectObservation(
                timestamp=now,
                hand=side,
                wrist_cam=wrist,
                palm_normal_cam=normal if normal is not None else np.zeros(3),
                palm_dir_cam=direction if direction is not None else np.zeros(3),
                confidence=0.9 if self.valid else 0.1,
                valid=self.valid,
                source='kinect_depth'
            )
            observations[side] = obs

        return now, frame_bgr, observations


class FakeImuSource:
    """Fake IMU source for testing."""

    def __init__(self, hands=('right', 'left'), valid=True):
        self.hands = hands
        self.valid = valid
        self._started = False

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def poll(self, now):
        if not self._started:
            return {}

        samples = {}
        for side in self.hands:
            sample = ImuSample(
                timestamp=now,
                timestamp_us=int(now * 1_000_000),
                hand=side,
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0,
                accel_g=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                gyro_dps=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                valid=self.valid,
                source='imu_udp'
            )
            samples[side] = sample
        return samples


class FakeVisualizer:
    """Fake visualizer that records calls instead of drawing."""

    def __init__(self):
        self.calls = []

    def draw_overlay(self, frame, state, statuses):
        self.calls.append({
            'frame_shape': frame.shape if frame is not None else None,
            'state_hand': state.hand if state else None,
            'state_source': state.source if state else None,
            'status_count': len(statuses),
        })


# =============================================================================
# Test Helpers
# =============================================================================

def create_identity_calibration(hand: str) -> Calibration:
    """Create a valid identity calibration for testing."""
    return Calibration(
        hand=hand,
        T_cam_world=np.eye(3, dtype=np.float64),
        f_body=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        yaw_boot_delta_deg=0.0,
        residual_deg=1.0,
        source='test_identity'
    )


# =============================================================================
# Tests
# =============================================================================

def test_fusion_engine_creation():
    """Test FusionEngine creation with valid calibrations."""
    calibrations = {
        'right': create_identity_calibration('right'),
        'left': create_identity_calibration('left'),
    }

    engine = FusionEngine(calibrations)

    assert 'right' in engine._filters
    assert 'left' in engine._filters
    assert 'right' in engine._sensor_managers
    assert 'left' in engine._sensor_managers

    # Check gravity_cam is set from calibration
    for hand in ('right', 'left'):
        filter_obj = engine._filters[hand]
        expected_gravity = calibrations[hand].T_cam_world @ np.array([0.0, 0.0, -1.0])
        assert np.allclose(filter_obj.config.gravity_cam, expected_gravity)

    print("[OK] test_fusion_engine_creation passed")


def test_fusion_engine_skips_invalid_calibration():
    """Test FusionEngine skips invalid calibrations."""
    calibrations = {
        'right': create_identity_calibration('right'),
        'left': Calibration(hand='left', T_cam_world=np.zeros((3,3))),  # Invalid
    }

    engine = FusionEngine(calibrations)

    assert 'right' in engine._filters
    assert 'left' not in engine._filters  # Skipped

    print("[OK] test_fusion_engine_skips_invalid_calibration passed")


def test_fusion_engine_update_with_imu_and_vision():
    """Test FusionEngine.update with both IMU and vision."""
    calibrations = {
        'right': create_identity_calibration('right'),
        'left': create_identity_calibration('left'),
    }
    engine = FusionEngine(calibrations)

    now = time.monotonic()

    # Valid IMU samples
    imu_samples = {
        'right': ImuSample(
            timestamp=now, timestamp_us=int(now*1e6), hand='right',
            roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
            accel_g=np.array([0.0, 0.0, 1.0]), gyro_dps=np.array([0.0, 0.0, 0.0]),
            valid=True, source='imu_udp'
        ),
        'left': ImuSample(
            timestamp=now, timestamp_us=int(now*1e6), hand='left',
            roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
            accel_g=np.array([0.0, 0.0, 1.0]), gyro_dps=np.array([0.0, 0.0, 0.0]),
            valid=True, source='imu_udp'
        ),
    }

    # Valid Kinect observations
    kinect_obs = {
        'right': KinectObservation(
            timestamp=now, hand='right',
            wrist_cam=np.array([0.1, 0.0, 1.0]),
            palm_normal_cam=np.array([0.0, 0.0, -1.0]),
            palm_dir_cam=np.array([0.0, 1.0, 0.0]),
            confidence=0.9, valid=True, source='kinect_depth'
        ),
        'left': KinectObservation(
            timestamp=now, hand='left',
            wrist_cam=np.array([-0.1, 0.0, 1.0]),
            palm_normal_cam=np.array([0.0, 0.0, -1.0]),
            palm_dir_cam=np.array([0.0, 1.0, 0.0]),
            confidence=0.9, valid=True, source='kinect_depth'
        ),
    }

    fused = engine.update(now, kinect_obs, imu_samples)

    assert 'right' in fused
    assert 'left' in fused

    for hand in ('right', 'left'):
        state = fused[hand]
        assert isinstance(state, FusedState)
        assert state.hand == hand
        assert state.valid is True
        assert state.source == 'fused'
        assert state.q_body_cam.shape == (4,)
        assert state.normal_cam.shape == (3,)
        assert state.dir_cam.shape == (3,)
        # Quaternion should be unit
        assert np.isclose(np.linalg.norm(state.q_body_cam), 1.0, atol=1e-6)

    print("[OK] test_fusion_engine_update_with_imu_and_vision passed")


def test_fusion_engine_update_imu_only():
    """Test FusionEngine.update with only IMU (no vision)."""
    calibrations = {'right': create_identity_calibration('right')}
    engine = FusionEngine(calibrations)

    now = time.monotonic()

    imu_samples = {
        'right': ImuSample(
            timestamp=now, timestamp_us=int(now*1e6), hand='right',
            roll_deg=10.0, pitch_deg=5.0, yaw_deg=0.0,
            accel_g=np.array([0.0, 0.0, 1.0]), gyro_dps=np.array([0.0, 0.0, 0.0]),
            valid=True, source='imu_udp'
        ),
    }
    kinect_obs = {}  # No vision

    fused = engine.update(now, kinect_obs, imu_samples)

    assert 'right' in fused
    state = fused['right']
    assert state.valid is True
    assert state.source == 'imu_only'

    print("[OK] test_fusion_engine_update_imu_only passed")


def test_fusion_engine_update_vision_only():
    """Test FusionEngine.update with only vision (no IMU)."""
    calibrations = {'right': create_identity_calibration('right')}
    engine = FusionEngine(calibrations)

    now = time.monotonic()

    imu_samples = {}  # No IMU
    kinect_obs = {
        'right': KinectObservation(
            timestamp=now, hand='right',
            wrist_cam=np.array([0.1, 0.0, 1.0]),
            palm_normal_cam=np.array([0.0, 0.0, -1.0]),
            palm_dir_cam=np.array([0.0, 1.0, 0.0]),
            confidence=0.9, valid=True, source='kinect_depth'
        ),
    }

    fused = engine.update(now, kinect_obs, imu_samples)

    assert 'right' in fused
    state = fused['right']
    assert state.valid is True
    assert state.source == 'vision_only:kinect_depth'

    print("[OK] test_fusion_engine_update_vision_only passed")


def test_fusion_engine_update_no_calibration_returns_empty():
    """Test FusionEngine.update returns states only for calibrated hands."""
    # Engine with only 'right' calibration
    calibrations = {'right': create_identity_calibration('right')}
    engine = FusionEngine(calibrations)

    now = time.monotonic()

    # Provide data for 'left' which has no calibration
    imu_samples = {
        'left': ImuSample(
            timestamp=now, timestamp_us=int(now*1e6), hand='left',
            roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
            accel_g=np.array([0.0, 0.0, 1.0]), gyro_dps=np.array([0.0, 0.0, 0.0]),
            valid=True, source='imu_udp'
        ),
    }
    kinect_obs = {
        'left': KinectObservation(
            timestamp=now, hand='left',
            wrist_cam=np.array([-0.1, 0.0, 1.0]),
            palm_normal_cam=np.array([0.0, 0.0, -1.0]),
            palm_dir_cam=np.array([0.0, 1.0, 0.0]),
            confidence=0.9, valid=True, source='kinect_depth'
        ),
    }

    fused = engine.update(now, kinect_obs, imu_samples)

    # Should return state for 'right' (calibrated) but not 'left' (no calibration)
    assert 'right' in fused
    assert 'left' not in fused
    # 'right' state has no sensor data -> source='none', confidence=0.0
    assert fused['right'].source == 'none'
    assert fused['right'].confidence == 0.0

    print("[OK] test_fusion_engine_update_no_calibration_returns_empty passed")


def test_fusion_engine_dt_handling():
    """Test FusionEngine handles dt correctly (first frame, clamp)."""
    calibrations = {'right': create_identity_calibration('right')}
    engine = FusionEngine(calibrations)

    now = time.monotonic()

    imu_samples = {
        'right': ImuSample(
            timestamp=now, timestamp_us=int(now*1e6), hand='right',
            roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
            accel_g=np.array([0.0, 0.0, 1.0]), gyro_dps=np.array([0.0, 0.0, 0.0]),
            valid=True, source='imu_udp'
        ),
    }
    kinect_obs = {}

    # First update - dt should be 1/30
    fused1 = engine.update(now, kinect_obs, imu_samples)
    assert 'right' in fused1

    # Second update with small dt
    time.sleep(0.01)
    now2 = time.monotonic()
    imu_samples['right'] = ImuSample(
        timestamp=now2, timestamp_us=int(now2*1e6), hand='right',
        roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
        accel_g=np.array([0.0, 0.0, 1.0]), gyro_dps=np.array([0.0, 0.0, 0.0]),
        valid=True, source='imu_udp'
    )
    fused2 = engine.update(now2, kinect_obs, imu_samples)
    assert 'right' in fused2

    # Test with huge dt (should be clamped)
    now3 = now + 10.0  # 10 seconds later
    imu_samples['right'] = ImuSample(
        timestamp=now3, timestamp_us=int(now3*1e6), hand='right',
        roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
        accel_g=np.array([0.0, 0.0, 1.0]), gyro_dps=np.array([0.0, 0.0, 0.0]),
        valid=True, source='imu_udp'
    )
    fused3 = engine.update(now3, kinect_obs, imu_samples)
    assert 'right' in fused3
    # Should not explode

    print("[OK] test_fusion_engine_dt_handling passed")


def test_fusion_engine_invalid_vision_normal():
    """Test FusionEngine handles invalid vision normal gracefully."""
    calibrations = {'right': create_identity_calibration('right')}
    engine = FusionEngine(calibrations)

    now = time.monotonic()

    imu_samples = {
        'right': ImuSample(
            timestamp=now, timestamp_us=int(now*1e6), hand='right',
            roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
            accel_g=np.array([0.0, 0.0, 1.0]), gyro_dps=np.array([0.0, 0.0, 0.0]),
            valid=True, source='imu_udp'
        ),
    }

    # Vision with NaN normal
    kinect_obs = {
        'right': KinectObservation(
            timestamp=now, hand='right',
            wrist_cam=np.array([0.1, 0.0, 1.0]),
            palm_normal_cam=np.array([float('nan'), 0.0, 0.0]),
            palm_dir_cam=np.array([0.0, 1.0, 0.0]),
            confidence=0.9, valid=True, source='kinect_depth'
        ),
    }

    fused = engine.update(now, kinect_obs, imu_samples)

    # Should fall back to imu_only since vision is invalid
    assert 'right' in fused
    state = fused['right']
    assert state.valid is True
    assert state.source in ('imu_only', 'fused')  # SensorManager decides

    print("[OK] test_fusion_engine_invalid_vision_normal passed")


def test_live_fusion_app_run_short():
    """Test LiveFusionApp.run completes with fake sources and show=False."""
    calibrations = {
        'right': create_identity_calibration('right'),
        'left': create_identity_calibration('left'),
    }
    engine = FusionEngine(calibrations)

    kinect_source = FakeKinectSource()
    imu_source = FakeImuSource()
    visualizer = FakeVisualizer()

    config = LiveConfig(
        duration_s=0.1,  # Very short
        show=False,      # No window
        print_every_s=10.0,  # Don't print
        max_dt=0.1
    )

    app = LiveFusionApp(
        engine=engine,
        kinect_source=kinect_source,
        imu_source=imu_source,
        config=config,
        visualizer=visualizer
    )

    result = app.run()

    # Check result structure
    assert 'frames' in result
    assert 'states' in result
    assert 'duration_s' in result
    assert 'fps' in result
    assert result['frames'] > 0
    assert result['states'] > 0
    assert result['duration_s'] > 0
    assert result['fps'] > 0

    # Check visualizer was called
    assert len(visualizer.calls) > 0
    for call in visualizer.calls:
        assert call['frame_shape'] == (480, 640, 3)
        assert call['state_hand'] in ('right', 'left')
        assert call['state_source'] in ('fused', 'imu_only', 'vision_only:kinect_depth')

    print("[OK] test_live_fusion_app_run_short passed")


def test_live_fusion_app_no_sources():
    """Test LiveFusionApp.run with no sources (just engine)."""
    calibrations = {'right': create_identity_calibration('right')}
    engine = FusionEngine(calibrations)

    config = LiveConfig(duration_s=0.05, show=False)

    app = LiveFusionApp(engine=engine, config=config)
    result = app.run()

    assert result['frames'] >= 0
    # Engine returns states for calibrated hands even without sources (source='none')
    assert result['states'] > 0
    assert result['duration_s'] > 0

    print("[OK] test_live_fusion_app_no_sources passed")


def test_live_fusion_app_kinect_only():
    """Test LiveFusionApp with only Kinect source."""
    calibrations = {'right': create_identity_calibration('right')}
    engine = FusionEngine(calibrations)

    kinect_source = FakeKinectSource(hands=('right',), valid=True)
    config = LiveConfig(duration_s=0.05, show=False)

    app = LiveFusionApp(engine=engine, kinect_source=kinect_source, config=config)
    result = app.run()

    assert result['frames'] > 0
    assert result['states'] > 0

    print("[OK] test_live_fusion_app_kinect_only passed")


def test_live_fusion_app_imu_only():
    """Test LiveFusionApp with only IMU source."""
    calibrations = {'right': create_identity_calibration('right')}
    engine = FusionEngine(calibrations)

    imu_source = FakeImuSource(hands=('right',), valid=True)
    config = LiveConfig(duration_s=0.05, show=False)

    app = LiveFusionApp(engine=engine, imu_source=imu_source, config=config)
    result = app.run()

    assert result['frames'] > 0
    assert result['states'] > 0

    print("[OK] test_live_fusion_app_imu_only passed")


def test_live_fusion_app_invalid_sources():
    """Test LiveFusionApp handles invalid source data gracefully."""
    calibrations = {'right': create_identity_calibration('right')}
    engine = FusionEngine(calibrations)

    # Sources that return invalid data
    kinect_source = FakeKinectSource(hands=('right',), valid=False)
    imu_source = FakeImuSource(hands=('right',), valid=False)

    config = LiveConfig(duration_s=0.05, show=False)

    app = LiveFusionApp(engine=engine, kinect_source=kinect_source, imu_source=imu_source, config=config)
    result = app.run()

    # Should complete without crashing
    assert result['frames'] > 0
    # States may be 0 if all data invalid

    print("[OK] test_live_fusion_app_invalid_sources passed")


def test_live_config_defaults():
    """Test LiveConfig default values."""
    config = LiveConfig()
    assert config.duration_s == 30.0
    assert config.show is True
    assert config.print_every_s == 1.0
    assert config.max_dt == 0.1

    config2 = LiveConfig(duration_s=10.0, show=False)
    assert config2.duration_s == 10.0
    assert config2.show is False

    print("[OK] test_live_config_defaults passed")


def test_fusion_engine_custom_configs():
    """Test FusionEngine with custom ComplementaryConfig per hand."""
    calibrations = {'right': create_identity_calibration('right')}

    custom_config = ComplementaryConfig(
        kp_grav=1.0,
        kp_yaw=0.1,
        max_dt=0.05,
        gravity_cam=np.array([0.0, 0.0, -1.0])  # Will be overwritten
    )

    engine = FusionEngine(calibrations, configs={'right': custom_config})

    filter_obj = engine._filters['right']
    assert filter_obj.config.kp_grav == 1.0
    assert filter_obj.config.kp_yaw == 0.1
    assert filter_obj.config.max_dt == 0.05
    # gravity_cam should come from calibration
    assert np.allclose(filter_obj.config.gravity_cam, [0.0, 0.0, -1.0])

    print("[OK] test_fusion_engine_custom_configs passed")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    test_fusion_engine_creation()
    test_fusion_engine_skips_invalid_calibration()
    test_fusion_engine_update_with_imu_and_vision()
    test_fusion_engine_update_imu_only()
    test_fusion_engine_update_vision_only()
    test_fusion_engine_update_no_calibration_returns_empty()
    test_fusion_engine_dt_handling()
    test_fusion_engine_invalid_vision_normal()
    test_live_fusion_app_run_short()
    test_live_fusion_app_no_sources()
    test_live_fusion_app_kinect_only()
    test_live_fusion_app_imu_only()
    test_live_fusion_app_invalid_sources()
    test_live_config_defaults()
    test_fusion_engine_custom_configs()

    print("\n[OK] All test_fusion_live.py tests passed!")
    sys.exit(0)