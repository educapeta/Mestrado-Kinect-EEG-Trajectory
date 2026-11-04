#!/usr/bin/env python
"""
Testes para fusion.aux_vision e tools.vision_ab (modo sintético).

Executa:
- py_compile nos arquivos novos
- Testes unitários com fake tracker/alignment
- Roda synthetic_compare_run sem hardware
"""

from __future__ import annotations
import sys
import subprocess
import tempfile
import os
from pathlib import Path

# Garante que o diretório do projeto está no path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import time

# ============================================================
# FAKE TRACKER / ALIGNMENT PARA TESTES
# ============================================================

class FakeTracker:
    """Fake KinectHandTracker para testes sem hardware."""
    def __init__(self, aux_indices=None):
        self.aux_indices = tuple(int(i) for i in (aux_indices or (0,)))
        self.color_width = 1920
        self.color_height = 1080
        self._frame_count = 0
        self._hands_data = {
            "right": {"landmarks": self._fake_landmarks(0.1), "side": "right", "position": np.array([0.0, 0.0, 0.8])},
            "left": {"landmarks": self._fake_landmarks(-0.1), "side": "left", "position": np.array([0.0, 0.0, 0.8])},
        }

    def _fake_landmarks(self, x_offset: float):
        """Gera 21 landmarks MediaPipe normalizados."""
        class FakeLM:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        lms = []
        for i in range(21):
            # Palma no centro, dedos espalhados
            u = (i % 5) * 0.02 - 0.04
            v = (i // 5) * 0.02 - 0.02
            lms.append(FakeLM(0.5 + x_offset + u, 0.5 + v))
        return lms

    def frames(self):
        self._frame_count += 1
        # Retorna (color_bgra, depth) - color_bgra fake
        color = np.zeros((self.color_height, self.color_width, 4), dtype=np.uint8)
        depth = np.full((512, 512), 800, dtype=np.uint16)  # 800mm
        return color, depth

    def detect_hands_all(self, color_bgra, depth):
        """Retorna maos detectadas no Kinect."""
        hands = []
        for side in ("right", "left"):
            h = self._hands_data[side].copy()
            hands.append(h)
        return None, hands  # (frame_desenhado, lista_maos)

    def detect_auxiliary_hands_sides(self, auxiliary_frame, aux_index=None):
        """Retorna maos detectadas na auxiliar."""
        hands = []
        for side in ("right", "left"):
            h = self._hands_data[side].copy()
            h["side_bruto"] = side
            hands.append(h)
        return hands

    def close(self):
        pass


class FakeAlignment:
    """Fake CameraAlignment com triangulacao controlada."""
    def __init__(self, aux_indices=None, usable=True):
        self.aux_indices = tuple(int(i) for i in (aux_indices or (0,)))
        self._usable = usable
        self.active_aux = 0

    def active_view(self):
        return self.aux_view(self.active_aux)

    def aux_view(self, aux_index):
        class FakeView:
            def __init__(self, usable):
                self.usable = usable
                self.stereo_usable = usable
                self.kinect_matrix = np.eye(3) * 1000
                self.kinect_dist = np.zeros(5)
                self.matrix = np.eye(3) * 1000
                self.dist = np.zeros(5)
                self.rotation = np.eye(3)
                self.translation = np.array([0.1, 0.0, 0.0])  # 10cm baseline
                self.rms = 1.0
                self.calibrated_size = (640, 480)
                self.runtime_size = None
                self.runtime_matrix = None
            @property
            def size(self):
                return self.runtime_size or self.calibrated_size
            @property
            def effective_matrix(self):
                return self.runtime_matrix if self.runtime_matrix is not None else self.matrix
        return FakeView(self._usable)

    def triangulate_hand_points(self, aux_pixels, kinect_pixels):
        """Triangulação fake: devolve pontos 3D no frame do Kinect."""
        if not self._usable:
            return None, None
        n = min(len(aux_pixels), len(kinect_pixels))
        if n < 5:
            return None, None
        # Gera pontos 3D plausíveis (plano frontal a ~0.8m)
        points_3d = []
        for i in range(n):
            # Triangulação simples: média ponderada + ruído
            x = (aux_pixels[i, 0] / 640.0 - 0.5) * 0.2
            y = (0.5 - aux_pixels[i, 1] / 480.0) * 0.2
            z = 0.8 + np.random.normal(0, 0.005)
            points_3d.append([x, y, z])
        points_3d = np.array(points_3d, dtype=np.float64)
        # Erros de reprojeção fake (0.5 a 2.0 pixels)
        reproj = np.random.uniform(0.5, 2.0, n).astype(np.float64)
        return points_3d, reproj

    def aux_usable(self, aux_index):
        return self._usable

    def stereo_status(self, aux_index=None):
        return ("aceita", "fake") if self._usable else ("rejeitada", "fake")


# ============================================================
# TESTES
# ============================================================

def test_aux_stereo_source_basic():
    """Testa AuxStereoSource.poll() com fakes."""
    print("Test 1: AuxStereoSource básico...")
    from fusion.aux_vision import AuxStereoSource, AuxStereoFrame

    tracker = FakeTracker()
    alignment = FakeAlignment(usable=True)
    source = AuxStereoSource(tracker=tracker, alignment=alignment, aux_index=0)
    source.start()

    result = source.poll()

    assert isinstance(result, dict), "poll() deve retornar dict"
    assert "right" in result, "Deve detectar mao direita"
    assert "left" in result, "Deve detectar mao esquerda"

    for hand, frame in result.items():
        assert isinstance(frame, AuxStereoFrame), f"{hand}: deve ser AuxStereoFrame"
        assert frame.hand == hand
        assert frame.valid, f"{hand}: frame deve ser válido"
        assert frame.points_3d.shape[1] == 3, f"{hand}: points_3d deve ser (N, 3)"
        assert frame.reproj_errors.shape[0] == frame.points_3d.shape[0]
        assert frame.timestamp > 0

    source.close()
    print("  OK")


def test_aux_stereo_source_unusable():
    """Testa AuxStereoSource quando stereo nao usable."""
    print("Test 2: AuxStereoSource stereo unusable...")
    from fusion.aux_vision import AuxStereoSource
    import warnings

    tracker = FakeTracker()
    alignment = FakeAlignment(usable=False)
    source = AuxStereoSource(tracker=tracker, alignment=alignment, aux_index=0)
    source.start()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = source.poll()
        assert len(w) >= 1, "Deve avisar quando stereo nao usable"
        assert "stereo calibration" in str(w[0].message).lower()
        assert result == {}, "Deve retornar {} quando stereo unusable"

    source.close()
    print("  OK")


def test_compare_frame():
    """Testa compare_frame com pontos conhecidos."""
    print("Test 3: compare_frame...")
    from fusion.aux_vision import compare_frame
    from fusion.vision_backends import VisionNormal

    # Pontos no plano XY a z=0.8, normal = -Z (right hand)
    n_pts = 21
    points = []
    for i in range(n_pts):
        u = (i % 5) * 0.02 - 0.04
        v = (i // 5) * 0.02 - 0.02
        points.append([u, v, 0.8])
    points = np.array(points, dtype=np.float64)

    # Adiciona pequeno ruído
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.001, points.shape)

    reproj = np.full(n_pts, 1.0, dtype=np.float64)

    result = compare_frame(
        kinect_points=points.copy(),
        multiview_points=points.copy(),
        hand="right",
        timestamp=time.monotonic(),
        reproj_errors=reproj,
    )

    assert "kinect_depth" in result
    assert "multiview" in result
    assert "angle_deg" in result
    assert "agree" in result

    vn_kd = result["kinect_depth"]
    vn_mv = result["multiview"]

    assert isinstance(vn_kd, VisionNormal)
    assert isinstance(vn_mv, VisionNormal)
    assert vn_kd.valid, "KinectDepth deve ser válido"
    assert vn_mv.valid, "MultiView deve ser válido"
    assert vn_kd.hand == "right"
    assert vn_mv.hand == "right"
    # Normais devem ser ~ -Z (palma virada para camera)
    assert abs(abs(vn_kd.normal_cam[2]) - 1.0) < 0.2, f"Normal KD Z = {vn_kd.normal_cam[2]}"
    assert abs(abs(vn_mv.normal_cam[2]) - 1.0) < 0.2, f"Normal MV Z = {vn_mv.normal_cam[2]}"
    # Angulo entre backends deve ser pequeno (mesmos pontos)
    assert result["angle_deg"] < 5.0, f"Angulo = {result['angle_deg']} deg"
    assert result["agree"], "Backends devem concordar"

    print(f"  angle_deg = {result['angle_deg']:.3f}, agree = {result['agree']}")
    print("  OK")


def test_compare_frame_different_points():
    """Testa compare_frame com pontos diferentes (simula ruído diferente)."""
    print("Test 4: compare_frame com pontos diferentes...")
    from fusion.aux_vision import compare_frame

    rng = np.random.default_rng(123)
    # Base points
    base = []
    for i in range(21):
        u = (i % 5) * 0.02 - 0.04
        v = (i // 5) * 0.02 - 0.02
        base.append([u, v, 0.8])
    base = np.array(base, dtype=np.float64)

    # Kinect: mais ruído
    k_pts = base + rng.normal(0, 0.005, base.shape)
    # MultiView: menos ruído
    mv_pts = base + rng.normal(0, 0.001, base.shape)
    reproj = rng.uniform(0.5, 2.0, 21)

    result = compare_frame(
        kinect_points=k_pts,
        multiview_points=mv_pts,
        hand="right",
        timestamp=time.monotonic(),
        reproj_errors=reproj,
    )

    assert result["kinect_depth"].valid
    assert result["multiview"].valid
    # Angulo pode ser maior mas ainda concordam (< 15 deg)
    print(f"  angle_deg = {result['angle_deg']:.3f}, agree = {result['agree']}")
    print("  OK")


def test_synthetic_compare_run():
    """Testa synthetic_compare_run (integração completa)."""
    print("Test 5: synthetic_compare_run...")
    from fusion.aux_vision import synthetic_compare_run

    results = synthetic_compare_run(
        duration=0.5,  # Curto para teste
        hand="right",
        n_points=21,
        noise_mm=2.0,
        outlier_frac=0.05,
        seed=999,
    )

    assert len(results) > 0, "Deve gerar pelo menos 1 frame"
    for r in results:
        assert "kinect_depth" in r
        assert "multiview" in r
        assert "angle_deg" in r
        assert "agree" in r
        assert r["kinect_depth"].valid or r["multiview"].valid

    angles = [r["angle_deg"] for r in results if r["kinect_depth"].valid and r["multiview"].valid]
    print(f"  frames = {len(results)}, validos = {len(angles)}, angulo medio = {np.mean(angles):.3f} deg")
    print("  OK")


def test_vision_ab_cli_help():
    """Testa CLI --help sem abrir hardware."""
    print("Test 6: vision_ab.py --help...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "vision_ab.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"Help falhou: {result.stderr}"
    assert "synthetic" in result.stdout
    assert "duracao" in result.stdout
    print("  OK")


def test_vision_ab_synthetic():
    """Testa vision_ab.py --synthetic --duracao 0.2 --log-csv tmp.csv"""
    print("Test 7: vision_ab.py --synthetic...")
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        csv_path = tmp.name
    try:
        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "tools" / "vision_ab.py"),
                "--synthetic", "--duracao", "0.2", "--log-csv", csv_path
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Synthetic falhou: {result.stderr}"
        assert "Resumo" in result.stdout
        # Verifica CSV
        assert os.path.exists(csv_path), "CSV não criado"
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        assert len(lines) >= 2, f"CSV deve ter header + dados, tem {len(lines)} linhas"
        header = lines[0].strip()
        assert "timestamp" in header and "angle_deg" in header
        print(f"  CSV linhas = {len(lines)}")
        print("  OK")
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)


def test_py_compile():
    """Testa py_compile nos arquivos novos."""
    print("Test 8: py_compile...")
    files = [
        PROJECT_ROOT / "fusion" / "aux_vision.py",
        PROJECT_ROOT / "tools" / "vision_ab.py",
        PROJECT_ROOT / "test_fusion_vision_ab.py",
    ]
    for f in files:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(f)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"py_compile falhou em {f}: {result.stderr}"
    print("  OK")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TESTES fusion/aux_vision.py + tools/vision_ab.py")
    print("=" * 60)

    tests = [
        test_py_compile,
        test_aux_stereo_source_basic,
        test_aux_stereo_source_unusable,
        test_compare_frame,
        test_compare_frame_different_points,
        test_synthetic_compare_run,
        test_vision_ab_cli_help,
        test_vision_ab_synthetic,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FALHOU: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"RESULTADO: {passed} passaram, {failed} falharam")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()