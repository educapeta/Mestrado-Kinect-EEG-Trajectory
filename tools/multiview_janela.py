#!/usr/bin/env python
"""Janela de diagnostico do multi-view: Kinect x auxiliar.

Mostra lado a lado:
- Kinect RGB com os landmarks 2D da mao;
- auxiliar (ja espelhada) com:
    verde  = landmarks detectados pela MediaPipe na auxiliar;
    amarelo= projecao de volta dos pontos TRIANGULADOS;
    ciano  = projecao de volta dos pontos de DEPTH do Kinect.
Se verde e amarelo nao coincidem, a triangulacao/correspondencia esta errada.
Se verde e ciano nao coincidem, o problema esta na calibracao/mapeamento.

A janela espera ESPACO antes de comecar. ESC encerra.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnostico visual Kinect x multi-view")
    p.add_argument("--aux-index", type=int, default=0)
    p.add_argument("--duracao", type=float, default=20.0)
    p.add_argument("--sem-janela", action="store_true")
    return p


def _beep(freq=880, ms=120):
    try:
        import winsound
        winsound.Beep(freq, ms)
    except Exception:
        pass


def _draw_points(img, pts, cor, raio=5):
    import cv2
    for p in np.asarray(pts, float).reshape(-1, 2):
        if np.isfinite(p).all():
            cv2.circle(img, (int(round(p[0])), int(round(p[1]))), raio, cor, -1)
    return img


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import cv2
    import kinect_compat
    kinect_compat.aplicar()
    from fusion.aux_vision import AuxStereoSource, _project_kinect_to_aux

    source = AuxStereoSource(aux_index=args.aux_index)
    source.start()
    win = "MultiView debug - Kinect | Auxiliar"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 700)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    frame_k = None
    frame_a = None
    while True:
        _ = source.poll()
        if source.last_kinect_bgra is not None:
            frame_k = cv2.cvtColor(source.last_kinect_bgra, cv2.COLOR_BGRA2BGR)
        if source.last_aux_frame is not None:
            frame_a = source.last_aux_frame.copy()
        base = np.zeros((720, 1280, 3), np.uint8)
        esq = cv2.resize(frame_k if frame_k is not None else base, (800, 450))
        dir_ = cv2.resize(frame_a if frame_a is not None else base, (800, 450))
        cv2.putText(esq, "KINECT", (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
        cv2.putText(dir_, "AUXILIAR (espelhada)", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        vis = np.hstack([esq, dir_])
        cv2.putText(vis, "PRESSIONE ESPACO para iniciar | ESC sai",
                    (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 220, 255), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == 32:
            _beep()
            break
        if key == 27:
            source.close()
            cv2.destroyAllWindows()
            return 0

    t0 = time.monotonic()
    ultimo_print = 0.0
    total_debug = 0
    try:
        while time.monotonic() - t0 < args.duracao:
            frames = source.poll()
            total_debug += len(source.last_debug or {})
            agora = time.monotonic()
            if agora - ultimo_print >= 0.5:
                ultimo_print = agora
                if source.last_debug:
                    partes = []
                    for hand, dbg in source.last_debug.items():
                        err = dbg.get("aux_errors")
                        kerr = dbg.get("reproj_kinect")
                        kderr = dbg.get("kd_errors")
                        partes.append(
                            f"{hand}: aux_reproj="
                            f"{np.nanmedian(err):.1f}px kd_reproj_aux="
                            f"{'--' if kderr is None else f'{np.nanmedian(kderr):.1f}'}px "
                            f"kin_reproj={np.nanmedian(kerr):.1f}px "
                            f"n={len(dbg['points_3d'])}")
                    print(f"[debug t={agora-t0:.1f}s] " + " | ".join(partes),
                          flush=True)
                else:
                    print(f"[debug t={agora-t0:.1f}s] sem maos/triangulacao",
                          flush=True)
            if source.last_kinect_bgra is not None:
                frame_k = cv2.cvtColor(source.last_kinect_bgra, cv2.COLOR_BGRA2BGR)
            if source.last_aux_frame is not None:
                frame_a = source.last_aux_frame.copy()
            esq = cv2.resize(frame_k if frame_k is not None else base, (800, 450))
            dir_ = cv2.resize(frame_a if frame_a is not None else base, (800, 450))
            cv2.putText(esq, "KINECT", (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            cv2.putText(dir_, "AUXILIAR", (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

            # Escala os pontos do frame original para o painel 800x450.
            if frame_a is not None:
                ah, aw = frame_a.shape[:2]
                sx_a, sy_a = 800.0 / max(aw, 1), 450.0 / max(ah, 1)
            else:
                sx_a = sy_a = 1.0
            if frame_k is not None:
                kh, kw = frame_k.shape[:2]
                sx_k, sy_k = 800.0 / max(kw, 1), 450.0 / max(kh, 1)
            else:
                sx_k = sy_k = 1.0
            for hand, dbg in (source.last_debug or {}).items():
                try:
                    kp = np.asarray(dbg["kinect_pixels"], float).copy()
                    kp[:, 0] *= sx_k
                    kp[:, 1] *= sy_k
                    _draw_points(vis, kp, (0, 255, 0), 4)
                    ap = np.asarray(dbg["aux_pixels"], float).copy()
                    ap[:, 0] = ap[:, 0] * sx_a + 800.0
                    ap[:, 1] *= sy_a
                    _draw_points(vis, ap, (0, 255, 0), 4)
                    if dbg.get("aux_projected") is not None:
                        pp = np.asarray(dbg["aux_projected"], float).copy()
                        pp[:, 0] = pp[:, 0] * sx_a + 800.0
                        pp[:, 1] *= sy_a
                        _draw_points(vis, pp, (0, 255, 255), 4)
                    if dbg.get("kd_projected") is not None:
                        kp2 = np.asarray(dbg["kd_projected"], float).copy()
                        kp2[:, 0] = kp2[:, 0] * sx_a + 800.0
                        kp2[:, 1] *= sy_a
                        _draw_points(vis, kp2, (255, 255, 0), 5)
                except Exception:
                    pass
                err = dbg.get("aux_errors")
                text = (f"{hand}: reproj_aux="
                        f"{np.nanmedian(err):.1f}px" if err is not None
                        else f"{hand}: reproj_aux=--")
                cv2.putText(vis, text, (20, 480 if hand == "right" else 510),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if not source.last_debug:
                cv2.putText(vis, "sem triangulacao neste frame", (20, 480),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(win, vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    finally:
        source.close()
        cv2.destroyAllWindows()
    print(f"[debug] frames com debug: {total_debug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
