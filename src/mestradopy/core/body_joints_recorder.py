"""Body Joints Recorder - Logging 3D skeleton joints from Kinect in camera space.

Este modulo grava os nós 3D do esqueleto do Kinect (corpo inteiro) em coordenadas
BRUTAS do camera space, para permitir transformar depois os labels para um frame
corporal (ombro/esterno/cabeça) em pós-processamento.

Arquivo de saida: corpo_<sessao>_<timestamp>.csv a ~30 Hz.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

try:
    from pykinect2 import PyKinectV2
except Exception:
    # Em alguns ambientes o import do pykinect2 falha antes de ImportError
    # (ex.: assert sizeof(tagSTATSTG) por mistura 32/64 bits). O modulo nao
    # pode derrubar o paradigma principal por isso: os indices das juntas sao
    # os mesmos do enum Kinect v2 e o fallback abaixo cobre o logging.
    PyKinectV2 = None


# Juntas minimas requeridas (conforme especificação)
REQUIRED_JOINTS = [
    "Head",
    "Neck",
    "SpineShoulder",
    "SpineMid",
    "SpineBase",
    "ShoulderLeft",
    "ShoulderRight",
    "ElbowLeft",
    "ElbowRight",
    "WristLeft",
    "WristRight",
    "HandLeft",
    "HandRight",
]

# Mapeamento de nomes para PyKinectV2.JointType_*
# Valores baseados no enum PyKinectV2.JointType (0-24 para 25 juntas)
JOINT_TYPE_MAP = {
    "Head": 3 if PyKinectV2 is None else PyKinectV2.JointType_Head,
    "Neck": 2 if PyKinectV2 is None else PyKinectV2.JointType_Neck,
    "SpineShoulder": 20 if PyKinectV2 is None else PyKinectV2.JointType_SpineShoulder,
    "SpineMid": 1 if PyKinectV2 is None else PyKinectV2.JointType_SpineMid,
    "SpineBase": 0 if PyKinectV2 is None else PyKinectV2.JointType_SpineBase,
    "ShoulderLeft": 4 if PyKinectV2 is None else PyKinectV2.JointType_ShoulderLeft,
    "ShoulderRight": 8 if PyKinectV2 is None else PyKinectV2.JointType_ShoulderRight,
    "ElbowLeft": 5 if PyKinectV2 is None else PyKinectV2.JointType_ElbowLeft,
    "ElbowRight": 9 if PyKinectV2 is None else PyKinectV2.JointType_ElbowRight,
    "WristLeft": 6 if PyKinectV2 is None else PyKinectV2.JointType_WristLeft,
    "WristRight": 10 if PyKinectV2 is None else PyKinectV2.JointType_WristRight,
    "HandLeft": 7 if PyKinectV2 is None else PyKinectV2.JointType_HandLeft,
    "HandRight": 11 if PyKinectV2 is None else PyKinectV2.JointType_HandRight,
}

# Tracking state codes (PyKinectV2.TrackingState enum)
TRACKING_STATE_NOT_TRACKED = 0
TRACKING_STATE_INFERRED = 1
TRACKING_STATE_TRACKED = 2


@dataclass
class JointData:
    """Dados de uma junta 3D."""
    x_m: float
    y_m: float
    z_m: float
    tracked: int  # 0=NotTracked, 1=Inferred, 2=Tracked


def extract_body_joints(last_bodies, joint_names: List[str] = None) -> Dict[str, JointData]:
    """Extrai as juntas 3D do corpo a partir de last_bodies do KinectHandTracker.
    
    Args:
        last_bodies: Objeto last_bodies do KinectHandTracker (PyKinectV2 body frame)
        joint_names: Lista de nomes das juntas a extrair (default: REQUIRED_JOINTS)
    
    Returns:
        Dict mapeando nome da junta -> JointData (coordenadas em metros, camera space)
    """
    if joint_names is None:
        joint_names = REQUIRED_JOINTS
    
    result = {}
    
    if last_bodies is None:
        # Retorna NaN para todas as juntas
        for name in joint_names:
            result[name] = JointData(np.nan, np.nan, np.nan, TRACKING_STATE_NOT_TRACKED)
        return result
    
    # Encontra o primeiro corpo rastreado
    tracked_body = None
    for body in last_bodies.bodies:
        if body.is_tracked:
            tracked_body = body
            break
    
    if tracked_body is None:
        # Nenhum corpo rastreado
        for name in joint_names:
            result[name] = JointData(np.nan, np.nan, np.nan, TRACKING_STATE_NOT_TRACKED)
        return result
    
    joints = getattr(tracked_body, "joints", None)
    if joints is None:
        for name in joint_names:
            result[name] = JointData(np.nan, np.nan, np.nan, TRACKING_STATE_NOT_TRACKED)
        return result
    
    # Extrai cada junta requerida
    for name in joint_names:
        joint_type = JOINT_TYPE_MAP.get(name)
        if joint_type is None:
            result[name] = JointData(np.nan, np.nan, np.nan, TRACKING_STATE_NOT_TRACKED)
            continue
        
        try:
            joint = joints[joint_type]
        except (IndexError, KeyError, TypeError):
            result[name] = JointData(np.nan, np.nan, np.nan, TRACKING_STATE_NOT_TRACKED)
            continue
        
        tracking_state = joint.TrackingState
        pos = joint.Position
        
        if (tracking_state == TRACKING_STATE_NOT_TRACKED or
            not np.isfinite([pos.x, pos.y, pos.z]).all()):
            result[name] = JointData(np.nan, np.nan, np.nan, int(tracking_state))
        else:
            result[name] = JointData(
                float(pos.x), float(pos.y), float(pos.z),
                int(tracking_state)
            )
    
    return result


def build_body_joints_header(joint_names: List[str] = None) -> List[str]:
    """Constrói o cabeçalho do CSV de juntas do corpo."""
    if joint_names is None:
        joint_names = REQUIRED_JOINTS
    
    header = ["t_mono_s", "t_epoch_s", "frame"]
    for name in joint_names:
        header.extend([
            f"{name}_x_m",
            f"{name}_y_m",
            f"{name}_z_m",
            f"{name}_tracked"
        ])
    return header


def build_body_joints_row(t_mono: float, t_epoch: float, frame: int,
                           joints_data: Dict[str, JointData],
                           joint_names: List[str] = None) -> List[str]:
    """Constrói uma linha do CSV de juntas do corpo."""
    if joint_names is None:
        joint_names = REQUIRED_JOINTS
    
    row = [f"{t_mono:.6f}", f"{t_epoch:.6f}", str(frame)]
    for name in joint_names:
        jd = joints_data.get(name)
        if jd is None:
            row.extend(["nan", "nan", "nan", "0"])
        else:
            row.extend([
                f"{jd.x_m:.6f}" if np.isfinite(jd.x_m) else "nan",
                f"{jd.y_m:.6f}" if np.isfinite(jd.y_m) else "nan",
                f"{jd.z_m:.6f}" if np.isfinite(jd.z_m) else "nan",
                str(jd.tracked)
            ])
    return row


class BodyJointsRecorder(threading.Thread):
    """Grava as juntas 3D do esqueleto do Kinect em CSV separado a ~30 Hz.
    
    Coordenadas RAW no camera space do Kinect (não relativas à origem).
    Registra também tracking state/qualidade e metadados JSON.
    """
    
    def __init__(self, state, path: str, hz: float = 30.0,
                 joint_names: List[str] = None,
                 tracker_ref=None):
        super().__init__(daemon=True)
        self._state = state
        self._path = str(path)
        self._dt = 1.0 / max(1e-3, float(hz))
        self._joint_names = joint_names or REQUIRED_JOINTS
        self._tracker_ref = tracker_ref  # Referência ao KinectHandTracker para acessar last_bodies
        
        self._stop_event = threading.Event()
        self._rows = 0
        self._error = None
        self._file_handle = None
        self._writer = None
        self._frame_counter = 0
        
        # Metadados para o JSON
        self._metadata = {
            "joints": self._joint_names,
            "joint_count": len(self._joint_names),
            "coordinate_frame": "kinect_camera_space",
            "units": "meters",
            "tracking_state_codes": {
                "0": "NotTracked",
                "1": "Inferred",
                "2": "Tracked"
            },
            "version": "1.0",
            "description": "3D body joints from Kinect v2 in raw camera space coordinates"
        }
    
    @property
    def rows(self) -> int:
        return self._rows
    
    @property
    def path(self) -> str:
        return self._path
    
    @property
    def error(self):
        return self._error
    
    def run(self):
        try:
            # Abre arquivo e escreve cabeçalho
            self._file_handle = open(self._path, "w", encoding="utf-8", newline="")
            self._writer = csv.writer(self._file_handle)
            
            header = build_body_joints_header(self._joint_names)
            self._writer.writerow(header)
            self._file_handle.flush()
            
            print(f"[corpo] gravando {os.path.basename(self._path)} a {1.0/self._dt:.0f} Hz ({len(self._joint_names)} juntas)", flush=True)
            
            next_t = time.monotonic()
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now < next_t:
                    self._flush_quiet()
                    time.sleep(min(0.005, next_t - now))
                    continue
                next_t += self._dt
                if next_t < now - 1.0:
                    next_t = now + self._dt
                
                try:
                    # Obtém last_bodies do tracker
                    last_bodies = None
                    if self._tracker_ref is not None:
                        ref = self._tracker_ref
                        # A thread de tracking guarda o tracker em `.tracker`;
                        # aceitar tanto o tracker direto quanto a thread evita
                        # gravar NaN silenciosamente.
                        if getattr(ref, "last_bodies", None) is None:
                            ref = getattr(ref, "tracker", None)
                        last_bodies = getattr(ref, "last_bodies", None)
                    
                    # Extrai juntas
                    joints_data = extract_body_joints(last_bodies, self._joint_names)
                    
                    # Constrói linha
                    t_epoch = time.time()
                    row = build_body_joints_row(now, t_epoch, self._frame_counter,
                                                joints_data, self._joint_names)
                    self._writer.writerow(row)
                    self._frame_counter += 1
                    self._rows += 1
                    
                    # Flush periódico
                    if self._rows % 10 == 0:
                        self._flush_quiet()
                        
                except Exception as exc:
                    self._error = exc
                    print(f"[corpo] ERRO na linha {self._rows + 1}: {type(exc).__name__}: {exc}", flush=True)
        
        except Exception as exc:
            self._error = exc
            print(f"[corpo] FALHA ao gravar {self._path}: {type(exc).__name__}: {exc}", flush=True)
        
        finally:
            self._flush_quiet()
            # Escreve metadados JSON ao lado do CSV
            self._write_metadata()
    
    def _flush_quiet(self):
        if self._file_handle is not None:
            try:
                self._file_handle.flush()
            except Exception:
                pass
    
    def _write_metadata(self):
        """Escreve arquivo JSON de metadados ao lado do CSV."""
        meta_path = Path(self._path).with_suffix(".json")
        try:
            self._metadata.update({
                "csv_file": os.path.basename(self._path),
                "rows_written": self._rows,
                "duration_s": self._rows * self._dt if self._rows > 0 else 0,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2)
            print(f"[corpo] metadados salvos em {meta_path}", flush=True)
        except Exception as exc:
            print(f"[corpo] aviso: falha ao salvar metadados: {exc}", flush=True)
    
    def stop(self):
        self._stop_event.set()
        self.join(timeout=2.0)


# =============================================================================
# Helper de teste com mock
# =============================================================================

class MockJoint:
    """Mock de uma junta do PyKinectV2 para testes."""
    def __init__(self, x: float, y: float, z: float, tracking_state: int = 2):
        self.Position = type('Pos', (), {'x': x, 'y': y, 'z': z})()
        self.TrackingState = tracking_state


class MockBody:
    """Mock de um corpo do PyKinectV2 para testes."""
    def __init__(self, is_tracked: bool = True, joints: Dict = None):
        self.is_tracked = is_tracked
        self.joints = joints or {}


class MockBodyFrame:
    """Mock do body frame do PyKinectV2 para testes."""
    def __init__(self, bodies: List[MockBody] = None):
        self.bodies = bodies or []


def create_mock_last_bodies(joint_positions: Dict[str, tuple] = None) -> MockBodyFrame:
    """Cria um last_bodies mock para testes.
    
    Args:
        joint_positions: Dict {nome_junta: (x, y, z)} com posições em metros.
                        Se None, usa posições padrão anatômicas.
    
    Returns:
        MockBodyFrame que pode ser passado para extract_body_joints().
    """
    if joint_positions is None:
        # Posições padrão aproximadas para um adulto em pé (camera space)
        # Origem no Kinect: X direita, Y cima, Z frente
        joint_positions = {
            "Head": (0.0, 1.6, 2.0),
            "Neck": (0.0, 1.45, 2.0),
            "SpineShoulder": (0.0, 1.35, 2.0),
            "SpineMid": (0.0, 1.1, 2.0),
            "SpineBase": (0.0, 0.9, 2.0),
            "ShoulderLeft": (-0.2, 1.35, 2.0),
            "ShoulderRight": (0.2, 1.35, 2.0),
            "ElbowLeft": (-0.35, 1.1, 2.0),
            "ElbowRight": (0.35, 1.1, 2.0),
            "WristLeft": (-0.5, 0.9, 2.0),
            "WristRight": (0.5, 0.9, 2.0),
            "HandLeft": (-0.55, 0.85, 2.0),
            "HandRight": (0.55, 0.85, 2.0),
        }
    
    joints = {}
    for name, (x, y, z) in joint_positions.items():
        joint_type = JOINT_TYPE_MAP.get(name)
        if joint_type is not None:
            joints[joint_type] = MockJoint(x, y, z, TRACKING_STATE_TRACKED)
    
    body = MockBody(is_tracked=True, joints=joints)
    return MockBodyFrame(bodies=[body])


def test_extract_body_joints_with_mock():
    """Testa extract_body_joints com mock de last_bodies.
    
    Returns:
        Lista de linhas CSV esperadas (strings).
    """
    # Cria mock
    mock_bodies = create_mock_last_bodies()
    
    # Extrai juntas
    joints_data = extract_body_joints(mock_bodies)
    
    # Verifica se todas as juntas foram extraídas
    assert len(joints_data) == len(REQUIRED_JOINTS), f"Expected {len(REQUIRED_JOINTS)} joints, got {len(joints_data)}"
    
    # Constrói linha CSV
    row = build_body_joints_row(123.456, 1700000000.123, 42, joints_data)
    
    # Verifica formato
    assert len(row) == 3 + len(REQUIRED_JOINTS) * 4, f"Row length mismatch: {len(row)}"
    assert row[0] == "123.456000"
    assert row[1] == "1700000000.123000"
    assert row[2] == "42"
    
    # Verifica Head
    head_idx = 3 + REQUIRED_JOINTS.index("Head") * 4
    assert row[head_idx] == "0.000000"  # x
    assert row[head_idx + 1] == "1.600000"  # y
    assert row[head_idx + 2] == "2.000000"  # z
    assert row[head_idx + 3] == "2"  # tracked
    
    # Verifica ShoulderLeft
    sl_idx = 3 + REQUIRED_JOINTS.index("ShoulderLeft") * 4
    assert row[sl_idx] == "-0.200000"
    assert row[sl_idx + 1] == "1.350000"
    assert row[sl_idx + 2] == "2.000000"
    assert row[sl_idx + 3] == "2"
    
    print("[OK] Teste extract_body_joints com mock PASSOU")
    print(f"  Linha CSV: {','.join(row)}")
    return row


if __name__ == "__main__":
    # Executa teste
    test_extract_body_joints_with_mock()
    print("\n[OK] Todos os testes do helper mock passaram!")