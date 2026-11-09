"""
Tests for fusion/adapters.py

Tests KinectMediaPipeSource, ImuBankSource, SyntheticSources, and load_calibrations
without requiring hardware.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fusion.adapters import (
    SIDE_TO_BANK,
    KinectMediaPipeSource,
    ImuBankSource,
    SyntheticSources,
    load_calibrations,
)
from fusion.contracts import (
    KinectObservation,
    ImuSample,
    Calibration,
)
from fusion.calibration import save_calibration


# =============================================================================
# Fake Tracker for KinectMediaPipeSource tests
# =============================================================================

class FakeKinectHandTracker:
    """Fake KinectHandTracker for testing without hardware."""

    def __init__(self):
        self._started = False
        self._frame_count = 0

    def start(self):
        self._started = True

    def close(self):
        self._started = False

    def frames(self):
        """Return fake (color_bgra, depth) frames."""
        if not self._started:
            return None
        self._frame_count += 1
        # 1080x1920 BGRA frame
        color_bgra = np.zeros((1080, 1920, 4), dtype=np.uint8)
        depth = np.zeros((1080, 1920), dtype=np.uint16)
        return color_bgra, depth

    def detect_hands_all(self, color_frame, depth_frame):
        """Return fake hand detections."""
        if not self._started:
            return []

        # Return both hands with valid data
        return [
            {
                'side': 'right',
                'landmarks': np.zeros((21, 3), dtype=np.float32),
                'position': np.array([0.1, 0.0, 1.0], dtype=np.float32),
            },
            {
                'side': 'left',
                'landmarks': np.zeros((21, 3), dtype=np.float32),
                'position': np.array([-0.1, 0.0, 1.0], dtype=np.float32),
            },
        ]

    def palm_orientation_camera(self, landmarks, side):
        """Return fake palm orientation (normal, direction)."""
        # Return valid normal for both hands
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return normal, direction

    def palm_plane_camera(self, landmarks, side, normal_anterior=None):
        """Plano medio sintetico (normal, rms_m)."""
        normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return normal, 0.001


# =============================================================================
# Fake ImuBank for ImuBankSource tests
# =============================================================================

class FakeImuBank:
    """Fake ImuBank for testing without hardware."""

    def __init__(self, portas='4210,4211'):
        self.portas = portas
        self._started = False
        self._sample_count = 0

    def start(self):
        self._started = True

    def stop(self):
        self._started = False

    def amostra(self, lado):
        """Return fake IMU sample for given side (1=right, 2=left)."""
        if not self._started:
            return None
        self._sample_count += 1

        # Create a fake ImuSample-like object
        class FakeImuSample:
            def __init__(self, lado):
                self.timestamp_us = int(time.monotonic() * 1_000_000)
                self.received_at = time.monotonic()
                self.roll_deg = 0.0
                self.pitch_deg = 0.0
                self.yaw_deg = 0.0
                self.accel_g = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                self.gyro_dps = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        return FakeImuSample(lado)

    def idade_s(self, lado):
        """Return age of latest sample in seconds."""
        if not self._started:
            return None
        return 0.01  # Fresh sample


# =============================================================================
# Tests
# =============================================================================

def test_side_to_bank_mapping():
    """Test SIDE_TO_BANK constant mapping."""
    assert SIDE_TO_BANK == {'right': 1, 'left': 2}
    print("[OK] test_side_to_bank_mapping passed")


def test_kinect_source_poll_with_fake_tracker():
    """Test KinectMediaPipeSource.poll with fake tracker."""
    tracker = FakeKinectHandTracker()
    source = KinectMediaPipeSource(tracker=tracker)
    source.start()

    try:
        timestamp, frame_bgr, observations = source.poll()

        # Check timestamp
        assert isinstance(timestamp, float)
        assert timestamp > 0

        # Check frame
        assert frame_bgr is not None
        assert frame_bgr.shape == (1080, 1920, 3)  # BGR
        assert frame_bgr.dtype == np.uint8

        # Check observations
        assert isinstance(observations, dict)
        assert 'right' in observations
        assert 'left' in observations

        for side in ('right', 'left'):
            obs = observations[side]
            assert isinstance(obs, KinectObservation)
            assert obs.hand == side
            assert obs.valid is True
            assert 0.0 <= obs.confidence <= 1.0
            assert obs.normal_sigma_deg >= 0.0
            assert obs.source == 'kinect_depth'
            assert obs.palm_normal_cam.shape == (3,)
            assert obs.palm_dir_cam.shape == (3,)
            assert obs.wrist_cam.shape == (3,)

        print("[OK] test_kinect_source_poll_with_fake_tracker passed")
    finally:
        source.close()


def test_kinect_source_poll_without_start():
    """Test KinectMediaPipeSource.poll without starting returns empty."""
    source = KinectMediaPipeSource(tracker=None)  # No tracker, not started

    timestamp, frame_bgr, observations = source.poll()

    assert isinstance(timestamp, float)
    assert frame_bgr is None
    assert observations == {}

    print("[OK] test_kinect_source_poll_without_start passed")


def test_kinect_source_invalid_normal_fallback():
    """Test KinectMediaPipeSource falls back to palm_plane_camera when normal is None."""

    class TrackerWithNoneNormal(FakeKinectHandTracker):
        def palm_orientation_camera(self, landmarks, side):
            return None, None  # Force fallback

    tracker = TrackerWithNoneNormal()
    source = KinectMediaPipeSource(tracker=tracker)
    source.start()

    try:
        timestamp, frame_bgr, observations = source.poll()

        # Should still get observations via fallback
        assert 'right' in observations
        assert 'left' in observations
        for obs in observations.values():
            # Fallback gives valid normal
            assert obs.valid is True
            assert 0.0 <= obs.confidence <= 1.0
            assert obs.normal_sigma_deg >= 0.0

        print("[OK] test_kinect_source_invalid_normal_fallback passed")
    finally:
        source.close()


def test_kinect_source_skips_duplicates():
    """Test KinectMediaPipeSource skips duplicate hand sides."""

    class TrackerWithDuplicates(FakeKinectHandTracker):
        def detect_hands_all(self, color_frame, depth_frame):
            return [
                {'side': 'right', 'landmarks': np.zeros((21, 3)), 'position': np.array([0.1, 0.0, 1.0])},
                {'side': 'right', 'landmarks': np.zeros((21, 3)), 'position': np.array([0.2, 0.0, 1.0])},  # Duplicate
                {'side': 'left', 'landmarks': np.zeros((21, 3)), 'position': np.array([-0.1, 0.0, 1.0])},
            ]

    tracker = TrackerWithDuplicates()
    source = KinectMediaPipeSource(tracker=tracker)
    source.start()

    try:
        timestamp, frame_bgr, observations = source.poll()

        # Should only have one 'right' observation
        assert len(observations) == 2
        assert 'right' in observations
        assert 'left' in observations

        print("[OK] test_kinect_source_skips_duplicates passed")
    finally:
        source.close()


def test_imu_bank_source_poll_with_fake_bank():
    """Test ImuBankSource.poll with fake bank."""
    bank = FakeImuBank()
    source = ImuBankSource(bank=bank)
    source.start()

    try:
        now = time.monotonic()
        samples = source.poll(now)

        assert isinstance(samples, dict)
        assert 'right' in samples
        assert 'left' in samples

        for side in ('right', 'left'):
            sample = samples[side]
            assert isinstance(sample, ImuSample)
            assert sample.hand == side
            assert sample.valid is True
            assert sample.source == 'imu_udp'
            assert sample.accel_g.shape == (3,)
            assert sample.gyro_dps.shape == (3,)
            assert np.isfinite(sample.accel_g).all()
            assert np.isfinite(sample.gyro_dps).all()

        print("[OK] test_imu_bank_source_poll_with_fake_bank passed")
    finally:
        source.stop()


def test_imu_bank_source_invalid_sample():
    """Test ImuBankSource marks invalid samples correctly."""

    class FakeBankWithBadData(FakeImuBank):
        def amostra(self, lado):
            class BadSample:
                timestamp_us = int(time.monotonic() * 1_000_000)
                received_at = time.monotonic()
                roll_deg = float('nan')
                pitch_deg = 0.0
                yaw_deg = 0.0
                accel_g = np.array([float('nan'), 0.0, 1.0])
                gyro_dps = np.array([0.0, 0.0, 0.0])
            return BadSample()

        def idade_s(self, lado):
            return 0.01

    bank = FakeBankWithBadData()
    source = ImuBankSource(bank=bank)
    source.start()

    try:
        samples = source.poll(time.monotonic())

        # Both hands should have invalid samples
        for side in ('right', 'left'):
            assert side in samples
            assert samples[side].valid is False

        print("[OK] test_imu_bank_source_invalid_sample passed")
    finally:
        source.stop()


def test_imu_bank_source_stale_sample():
    """Test ImuBankSource marks stale samples (>0.5s) as invalid."""

    class FakeBankStale(FakeImuBank):
        def idade_s(self, lado):
            return 1.0  # Stale

    bank = FakeBankStale()
    source = ImuBankSource(bank=bank)
    source.start()

    try:
        samples = source.poll(time.monotonic())

        for side in ('right', 'left'):
            assert side in samples
            assert samples[side].valid is False

        print("[OK] test_imu_bank_source_stale_sample passed")
    finally:
        source.stop()


def test_imu_bank_source_without_start():
    """Test ImuBankSource.poll without starting returns empty."""
    source = ImuBankSource(bank=None)
    samples = source.poll(time.monotonic())
    assert samples == {}
    print("[OK] test_imu_bank_source_without_start passed")


def test_synthetic_sources_kinect_poll():
    """Test SyntheticSources poll for Kinect data."""
    synth = SyntheticSources()
    synth.start()

    try:
        timestamp, frame_bgr, observations = synth.poll()

        assert isinstance(timestamp, float)
        assert frame_bgr is not None
        assert frame_bgr.shape == (1080, 1920, 3)
        assert frame_bgr.dtype == np.uint8

        assert 'right' in observations
        assert 'left' in observations

        for side in ('right', 'left'):
            obs = observations[side]
            assert isinstance(obs, KinectObservation)
            assert obs.valid is True
            assert obs.confidence == 1.0
            assert obs.source == 'synthetic'

        print("[OK] test_synthetic_sources_kinect_poll passed")
    finally:
        synth.close()


def test_synthetic_sources_imu_poll():
    """Test SyntheticSources poll_imu for IMU data."""
    synth = SyntheticSources()
    synth.start()

    try:
        samples = synth.poll_imu()

        assert 'right' in samples
        assert 'left' in samples

        for side in ('right', 'left'):
            sample = samples[side]
            assert isinstance(sample, ImuSample)
            assert sample.valid is True
            assert sample.source == 'synthetic'
            # Accel should be [0, 0, 1] (1g up)
            assert np.allclose(sample.accel_g, [0.0, 0.0, 1.0])
            assert np.allclose(sample.gyro_dps, [0.0, 0.0, 0.0])

        print("[OK] test_synthetic_sources_imu_poll passed")
    finally:
        synth.close()


def test_load_calibrations_creates_valid_calibrations(tmp_path):
    """Test load_calibrations loads valid calibration files."""
    # Create valid calibration files
    cal_right = Calibration(
        hand='right',
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
        yaw_boot_delta_deg=0.0,
        residual_deg=2.5,
        source='test'
    )
    cal_left = Calibration(
        hand='left',
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
        yaw_boot_delta_deg=0.0,
        residual_deg=2.5,
        source='test'
    )

    right_file = tmp_path / 'imu_palm_alignment_right.json'
    left_file = tmp_path / 'imu_palm_alignment_left.json'

    save_calibration(cal_right, right_file)
    save_calibration(cal_left, left_file)

    # Load calibrations
    calibrations = load_calibrations(str(tmp_path))

    assert 'right' in calibrations
    assert 'left' in calibrations
    assert calibrations['right'].validate()
    assert calibrations['left'].validate()
    assert np.allclose(calibrations['right'].T_cam_world, np.eye(3))
    assert np.allclose(calibrations['left'].T_cam_world, np.eye(3))

    print("[OK] test_load_calibrations_creates_valid_calibrations passed")


def test_load_calibrations_skips_invalid():
    """Test load_calibrations skips missing/invalid files."""
    # No files exist
    calibrations = load_calibrations('/nonexistent/path')
    assert calibrations == {}

    print("[OK] test_load_calibrations_skips_invalid passed")


def test_load_calibrations_custom_filenames(tmp_path):
    """Test load_calibrations with custom filename mapping."""
    cal_right = Calibration(
        hand='right',
        T_cam_world=np.eye(3),
        f_body=np.array([0.0, 0.0, 1.0]),
    )
    custom_file = tmp_path / 'custom_right.json'
    save_calibration(cal_right, custom_file)

    calibrations = load_calibrations(
        str(tmp_path),
        lado_arquivos={'right': 'custom_right.json'}
    )

    assert 'right' in calibrations
    assert 'left' not in calibrations

    print("[OK] test_load_calibrations_custom_filenames passed")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import tempfile

    # Run all tests
    test_side_to_bank_mapping()
    test_kinect_source_poll_with_fake_tracker()
    test_kinect_source_poll_without_start()
    test_kinect_source_invalid_normal_fallback()
    test_kinect_source_skips_duplicates()
    test_imu_bank_source_poll_with_fake_bank()
    test_imu_bank_source_invalid_sample()
    test_imu_bank_source_stale_sample()
    test_imu_bank_source_without_start()
    test_synthetic_sources_kinect_poll()
    test_synthetic_sources_imu_poll()

    # Tests requiring temp directory
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_load_calibrations_creates_valid_calibrations(tmp_path)
        test_load_calibrations_skips_invalid()
        test_load_calibrations_custom_filenames(tmp_path)

    print("\n[OK] All test_fusion_adapters.py tests passed!")
    sys.exit(0)