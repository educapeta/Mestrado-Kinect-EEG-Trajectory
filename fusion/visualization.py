"""
Visualizacao OpenCV leve da fusao.

Convencao visual:
- a imagem tem x para a direita e y para BAIXO;
- o frame da camera do Kinect tem X para a direita, Y para CIMA e Z para
  DENTRO da cena (eixo optico);
- a normal da palma e' ancorada no landmark 9 (palma) de cada mao;
- magenta = normal fundida; verde = normal medida pelo Kinect (visao);
- no canto superior direito fica a triade da camera (X vermelho, Y verde,
  Z azul: circulo cheio = +Z/fundo, cruz = -Z/frente).
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import cv2
import numpy as np

from .contracts import FusedState, SourceStatus, Calibration

COLOR_FUSED = (255, 0, 255)    # magenta
COLOR_VISION = (0, 255, 0)     # verde
COLOR_IMU = (0, 255, 255)      # amarelo
COLOR_TEXT = (255, 255, 255)   # branco
COLOR_BG = (0, 0, 0)
COLOR_VALID = (0, 255, 0)
COLOR_INVALID = (0, 0, 255)
COLOR_X = (0, 0, 255)          # X vermelho (BGR)
COLOR_Y = (0, 255, 0)          # Y verde
COLOR_Z = (255, 128, 0)        # Z azul
COLOR_TANGENT = (255, 255, 0)  # ciano
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _unit(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-9:
        return None
    return v / n


def _origem_palma(state: FusedState, width: int, height: int) -> Tuple[int, int]:
    """Ponto 2D da palma, ou centro (fallback)."""
    px = None if state.palm_px is None else np.asarray(state.palm_px, float).reshape(-1)
    if px is not None and px.size >= 2 and np.isfinite(px[:2]).all():
        x = int(np.clip(px[0], 0, width - 1))
        y = int(np.clip(px[1], 0, height - 1))
        return x, y
    return width // 2, height // 2


def _seta_normal(frame, origem, normal, cor, rotulo, comprimento=110):
    """Desenha a normal no plano da imagem.

    Se a normal estiver quase no eixo optico (|z| dominante), nao da' para
    desenhar uma seta 2D: desenha circulo cheio (-Z/frente) ou cruz (+Z/fundo).
    """
    n = _unit(normal)
    if n is None:
        return
    x, y = int(origem[0]), int(origem[1])
    sx, sy = float(n[0]), float(-n[1])  # camera Y para cima -> imagem Y para baixo
    paralelo = float(np.hypot(sx, sy))

    if paralelo < 0.22:
        raio = int(max(9, min(28, 14 + 18 * abs(float(n[2])))))
        if n[2] < 0:
            # -Z: apontando para a lente/frente do Kinect
            cv2.circle(frame, (x, y), raio, cor, -1)
            cv2.circle(frame, (x, y), max(3, raio // 3), COLOR_BG, -1)
            txt = f"{rotulo} -Z (frente)"
        else:
            # +Z: apontando para o fundo da cena
            cv2.circle(frame, (x, y), raio, cor, 2)
            cv2.line(frame, (x - raio // 2, y - raio // 2),
                     (x + raio // 2, y + raio // 2), cor, 2)
            cv2.line(frame, (x + raio // 2, y - raio // 2),
                     (x - raio // 2, y + raio // 2), cor, 2)
            txt = f"{rotulo} +Z (fundo)"
        cv2.putText(frame, txt, (x + raio + 6, y + 5),
                    FONT, 0.42, cor, 1, cv2.LINE_AA)
    else:
        fim = (int(x + sx * comprimento), int(y + sy * comprimento))
        # contorno escuro para a seta aparecer sobre a luva/pele
        cv2.arrowedLine(frame, (x, y), fim, (0, 0, 0), 6, tipLength=0.25)
        cv2.arrowedLine(frame, (x, y), fim, cor, 3, tipLength=0.25)
        cv2.putText(frame, rotulo, (fim[0] + 5, fim[1] + 4),
                    FONT, 0.45, cor, 1, cv2.LINE_AA)


def draw_camera_axes(frame_bgr: np.ndarray,
                     origem: Optional[Tuple[int, int]] = None,
                     tamanho: int = 70) -> np.ndarray:
    """Triade da camera: X direita, Y para cima na cena, Z para dentro."""
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if origem is None:
        origem = (w - 330, 95)
    x0, y0 = int(origem[0]), int(origem[1])
    fim_x = (x0 + tamanho, y0)
    fim_y = (x0, y0 - tamanho)
    cv2.arrowedLine(frame_bgr, (x0, y0), fim_x, COLOR_X, 2, tipLength=0.25)
    cv2.arrowedLine(frame_bgr, (x0, y0), fim_y, COLOR_Y, 2, tipLength=0.25)
    cv2.putText(frame_bgr, "X", (fim_x[0] + 5, fim_x[1] + 4), FONT, 0.5,
                COLOR_X, 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, "Y", (fim_y[0] + 5, fim_y[1] + 4), FONT, 0.5,
                COLOR_Y, 1, cv2.LINE_AA)
    zx, zy = x0 + tamanho, y0 + 26
    cv2.circle(frame_bgr, (zx, zy), 9, COLOR_Z, 2)
    cv2.circle(frame_bgr, (zx, zy), 3, COLOR_Z, -1)
    cv2.putText(frame_bgr, "Z+ (fundo)", (zx + 14, zy + 5), FONT, 0.42,
                COLOR_Z, 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, "camera", (x0 - 10, y0 + tamanho // 2 + 30),
                FONT, 0.42, COLOR_TEXT, 1, cv2.LINE_AA)
    return frame_bgr


def draw_overlay(
    frame_bgr: np.ndarray,
    state: FusedState,
    statuses: List[SourceStatus],
    calibration: Optional[Calibration] = None,
    source_text: Optional[str] = None,
    vision_normal_cam: Optional[np.ndarray] = None,
    panel_origin: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Desenha a fusao ancorada na palma + painel + triade da camera."""
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if panel_origin is None:
        panel_origin = (10, 30)
    painel_x, painel_y = int(panel_origin[0]), int(panel_origin[1])

    normal = _unit(state.normal_cam)
    origem_palma = _origem_palma(state, w, h)

    # Painel de status (uma mao por chamada; o live passa panel_origin).
    linhas = 10 + len(statuses)
    altura_painel = linhas * 21 + 10
    largura_painel = 560
    cv2.rectangle(frame_bgr,
                  (painel_x - 5, painel_y - 25),
                  (painel_x + largura_painel, painel_y + altura_painel),
                  COLOR_BG, -1)
    cv2.rectangle(frame_bgr,
                  (painel_x - 5, painel_y - 25),
                  (painel_x + largura_painel, painel_y + altura_painel),
                  COLOR_FUSED, 1)
    cv2.putText(frame_bgr, f"Fusion: {state.hand} | {state.source or '?'}",
                (painel_x, painel_y), FONT, 0.55, COLOR_FUSED, 1, cv2.LINE_AA)
    painel_y += 21
    conf_color = COLOR_VALID if state.confidence >= 0.5 else COLOR_INVALID
    cv2.putText(frame_bgr, f"conf={state.confidence:.2f} valid={int(state.valid)} "
                f"lat={state.latency_ms:.1f}ms",
                (painel_x, painel_y), FONT, 0.5, conf_color, 1, cv2.LINE_AA)
    painel_y += 21
    if normal is not None:
        cv2.putText(frame_bgr,
                    f"n=({normal[0]:+.2f},{normal[1]:+.2f},{normal[2]:+.2f}) "
                    f"z={normal[2]:+.2f}",
                    (painel_x, painel_y), FONT, 0.5, COLOR_FUSED, 1, cv2.LINE_AA)
        painel_y += 21
        aviso_z = "z~0: palma PARALELA ao eixo optico" if abs(normal[2]) < 0.15 \
            else "z!=0: palma inclinada em relacao ao eixo optico"
        cv2.putText(frame_bgr, aviso_z, (painel_x, painel_y), FONT, 0.45,
                    COLOR_TEXT, 1, cv2.LINE_AA)
        painel_y += 21
    cv2.putText(frame_bgr, "magenta=FUSED  verde=KINECT  ciano=dir",
                (painel_x, painel_y), FONT, 0.42, COLOR_TEXT, 1, cv2.LINE_AA)
    painel_y += 21
    if calibration is not None:
        cv2.putText(frame_bgr,
                    f"cal={calibration.source or '?'} resid="
                    f"{calibration.residual_deg if calibration.residual_deg is not None else '?'}",
                    (painel_x, painel_y), FONT, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        painel_y += 21
    for status in statuses:
        age = f"{status.age_ms:.0f}ms" if status.age_ms < 1000 else f"{status.age_ms/1000:.1f}s"
        cor = COLOR_VALID if status.valid else COLOR_INVALID
        txt = f"{status.source}: age={age} conf={status.confidence:.2f} " \
              f"{'OK' if status.valid else 'FAIL'}"
        if status.reason:
            txt += f" ({status.reason})"
        cv2.putText(frame_bgr, txt, (painel_x, painel_y), FONT, 0.42, cor, 1,
                    cv2.LINE_AA)
        painel_y += 21

    # Cruz no ponto da palma.
    cv2.drawMarker(frame_bgr, origem_palma, (255, 255, 255),
                   cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"palma {state.hand}", (origem_palma[0] + 10,
                origem_palma[1] + 18), FONT, 0.42, COLOR_TEXT, 1, cv2.LINE_AA)
    if normal is not None:
        texto_n = (f"n=({normal[0]:+.2f},{normal[1]:+.2f},{normal[2]:+.2f}) "
                   f"z={normal[2]:+.2f}")
        cv2.putText(frame_bgr, texto_n, (origem_palma[0] + 10,
                    origem_palma[1] - 12), FONT, 0.42, COLOR_FUSED, 1,
                    cv2.LINE_AA)

    # Normal fundida (magenta) e verdade do Kinect (verde) no MESMO ponto.
    _seta_normal(frame_bgr, origem_palma, normal, COLOR_FUSED, "FUSED")
    if vision_normal_cam is not None:
        _seta_normal(frame_bgr, origem_palma, vision_normal_cam, COLOR_VISION, "KINECT")

    # Direcao dos dedos, se a visao forneceu.
    dirv = _unit(state.dir_cam)
    if dirv is not None:
        x, y = origem_palma
        sx, sy = float(dirv[0]), float(-dirv[1])
        fim = (int(x + sx * 70), int(y + sy * 70))
        cv2.arrowedLine(frame_bgr, origem_palma, fim, COLOR_TANGENT, 2,
                        tipLength=0.25)
        cv2.putText(frame_bgr, "dir", (fim[0] + 5, fim[1] + 4), FONT, 0.42,
                    COLOR_TANGENT, 1, cv2.LINE_AA)

    # Triade da camera no canto.
    draw_camera_axes(frame_bgr, (w - 120, 80), 60)
    return frame_bgr


def draw_calibration_overlay(frame_bgr: np.ndarray, calibration: Calibration,
                             position: tuple = (10, 10)) -> np.ndarray:
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    x, y = position
    linhas = [
        f"Calibration: {calibration.source}",
        f"Hand: {calibration.hand}",
        ("Residual: %.2f deg" % calibration.residual_deg
         if calibration.residual_deg is not None else "Residual: N/A"),
        f"Yaw boot delta: {calibration.yaw_boot_delta_deg:.2f} deg",
        "f_body: [%.3f, %.3f, %.3f]" % tuple(calibration.f_body),
    ]
    for i, linha in enumerate(linhas):
        cv2.putText(frame_bgr, linha, (x, y + i * 20), FONT, 0.5,
                    COLOR_TEXT, 1, cv2.LINE_AA)
    return frame_bgr
