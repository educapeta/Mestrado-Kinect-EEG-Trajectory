"""Diagnostico de body tracking/JointOrientation do Kinect v2 (color+depth+body)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import kinect_compat
    kinect_compat.aplicar()
    from pykinect2 import PyKinectV2
    from pykinect2.PyKinectRuntime import PyKinectRuntime

    fonte = (PyKinectV2.FrameSourceTypes_Color
             | PyKinectV2.FrameSourceTypes_Depth
             | PyKinectV2.FrameSourceTypes_Body)
    print("abrindo runtime (color+depth+body)...", flush=True)
    k = PyKinectRuntime(fonte)
    print(f"color {k.color_frame_desc.Width}x{k.color_frame_desc.Height} | "
          f"depth {k.depth_frame_desc.Width}x{k.depth_frame_desc.Height} | "
          f"max bodies={k.max_body_count}", flush=True)
    color_salvo = depth_salvo = False
    cor_frames = prof_frames = body_frames = 0
    corpos = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < 8.0:
        if k.has_new_color_frame() and not color_salvo:
            cor = k.get_last_color_frame()
            if cor is not None:
                cor = cor.reshape((k.color_frame_desc.Height,
                                   k.color_frame_desc.Width, 4))
                try:
                    import cv2
                    cv2.imwrite("_kinect_color.png", cor[:, :, :3])
                except Exception as exc:
                    print("color save:", exc)
                print("COLOR frame OK", flush=True)
                cor_frames += 1
                color_salvo = True
        if k.has_new_depth_frame() and not depth_salvo:
            prof = k.get_last_depth_frame()
            if prof is not None:
                prof = prof.reshape((k.depth_frame_desc.Height,
                                     k.depth_frame_desc.Width))
                try:
                    import cv2
                    vis = np.clip(prof.astype(np.float32) * 0.25, 0, 255).astype(np.uint8)
                    cv2.imwrite("_kinect_depth.png", vis)
                except Exception as exc:
                    print("depth save:", exc)
                print("DEPTH frame OK", flush=True)
                prof_frames += 1
                depth_salvo = True
        if k.has_new_body_frame():
            body_frames += 1
            bodies = k.get_last_body_frame()
            if bodies is not None:
                for i, body in enumerate(bodies.bodies):
                    if body.is_tracked:
                        corpos += 1
                        if corpos <= 4:
                            print(f"BODY {i} TRACKED | "
                                  f"handR={body.joints[PyKinectV2.JointType_HandRight].Position.x:.3f},"
                                  f"{body.joints[PyKinectV2.JointType_HandRight].Position.y:.3f},"
                                  f"{body.joints[PyKinectV2.JointType_HandRight].Position.z:.3f}",
                                  flush=True)
                            for nome, jid in (("HandRight", PyKinectV2.JointType_HandRight),
                                              ("WristRight", PyKinectV2.JointType_WristRight),
                                              ("HandLeft", PyKinectV2.JointType_HandLeft)):
                                ori = body.joint_orientations[jid]
                                q = np.array([ori.x, ori.y, ori.z, ori.w], float)
                                print(f"  {nome}: q={np.round(q,4)} |q|={np.linalg.norm(q):.4f}",
                                      flush=True)
            else:
                print("body frame veio None", flush=True)
        time.sleep(0.02)
    k.close()
    print(f"\nCOR={cor_frames} DEPTH={prof_frames} BODY_FRAMES={body_frames} "
          f"BODIES_TRACKED={corpos}", flush=True)
    print("imagens: _kinect_color.png / _kinect_depth.png", flush=True)
    return 0 if cor_frames and prof_frames else 3


if __name__ == "__main__":
    raise SystemExit(main())
