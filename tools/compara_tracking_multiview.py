#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ferramenta de demonstração: Triangulação 3 vistas (Kinect RGB + Aux 0 + Aux 1).

Abre as 3 câmeras, detecta landmarks com MediaPipe, triangula em 3D
usando a nova função triangulate_three_views, e mostra:
- Janela do Kinect RGB com esqueletos projetados (verde=Kinect, vermelho=aux0, magenta=aux1)
- Métricas em tempo real: erro de reprojeção, método usado, pontos válidos
- Log CSV com métricas por frame

Uso:
    .venv\Scripts\python.exe tools/compara_tracking_multiview.py

Teclas:
    ESPAÇO - Pausa/continua
    Q/ESC - Sai
    S - Salva frame atual
"""

import sys
import time
import csv
from pathlib import Path
from collections import deque

import cv2
import numpy as np

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinect_imu_groundtruth import (
    KinectHandTracker,
    CameraAlignment,
    triangulate_three_views,
    AuxCameraView,
    desenhar_esqueleto_projetado,
    mascara_dentro_do_frame,
    transform_auxiliary_frame,
    AUXILIARY_MIRROR_HORIZONTAL,
    AUX_CAMERA_INDICES,
    indices_auxiliares_automaticos,
    nome_auxiliar,
)


# Configurações
WINDOW_NAME = "Triangulacao 3 Vistas - Kinect + Aux0 + Aux1"
CSV_LOG_PATH = Path(__file__).parent / "triangulacao_3view_log.csv"
MAX_FPS_HISTORY = 60
TARGET_AUX_INDICES = (0, 1)  # Índices das auxiliares a usar


class MetricsLogger:
    """Logger de métricas para CSV."""
    
    def __init__(self, path):
        self.path = path
        self.file = None
        self.writer = None
        self.frame_count = 0
    
    def open(self):
        self.file = open(self.path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'frame', 'timestamp', 'method', 'n_valid', 'views_used',
            'reproj_median_kin', 'reproj_median_aux0', 'reproj_median_aux1',
            'max_reproj_px', 'fps'
        ])
        self.file.flush()
    
    def log(self, frame_idx, timestamp, info, fps):
        if self.writer is None:
            return
        
        method = info.get('method', 'failed')
        n_valid = info.get('n_valid', 0)
        views_used = str(info.get('views_used', []))
        reproj = info.get('reproj_errors_median_px', [])
        max_reproj = info.get('max_reproj_error_px', 0.0)
        
        r_kin = reproj[0] if len(reproj) > 0 and not np.isnan(reproj[0]) else ''
        r_aux0 = reproj[1] if len(reproj) > 1 and not np.isnan(reproj[1]) else ''
        r_aux1 = reproj[2] if len(reproj) > 2 and not np.isnan(reproj[2]) else ''
        
        self.writer.writerow([
            frame_idx, f'{timestamp:.3f}', method, n_valid, views_used,
            r_kin, r_aux0, r_aux1,
            f'{max_reproj:.2f}' if max_reproj else '',
            f'{fps:.1f}'
        ])
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.file.flush()
    
    def close(self):
        if self.file:
            self.file.flush()
            self.file.close()


def open_aux_cameras(indices):
    """Abre as câmeras auxiliares pelo índice."""
    caps = {}
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Configura resolução 1280x720 @ 30fps
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Aux {idx} ({nome_auxiliar(idx)}): {actual_w}x{actual_h} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")
            caps[idx] = cap
        else:
            print(f"  AVISO: Não conseguiu abrir câmera auxiliar {idx}")
    return caps


def read_aux_frames(caps):
    """Lê frames de todas as câmeras auxiliares."""
    frames = {}
    for idx, cap in caps.items():
        ret, frame = cap.read()
        if ret and frame is not None:
            if AUXILIARY_MIRROR_HORIZONTAL:
                frame = cv2.flip(frame, 1)
            frames[idx] = frame
        else:
            frames[idx] = None
    return frames




def _landmarks_de_entrada(entrada):
    """Aceita landmark list do MediaPipe OU dict {'landmarks': [...]}.

    `detect_hands_all` (Kinect) devolve dict com 'side'/'landmarks'; 
    `detect_auxiliary_hands_all` devolve diretamente listas de landmarks.
    O demo antigo assumia dict para as auxiliares e quebrava com 
    'list indices must be integers'.
    """
    if entrada is None:
        return None
    if isinstance(entrada, dict):
        return entrada.get('landmarks')
    return entrada


def get_landmarks_pixels(tracker, color_frame, depth_frame, aux_frames, alignment, want_side=None):
    """Detecta landmarks em todas as câmeras e retorna pixels (21,2) por câmera."""
    
    # Kinect
    kinect_color, kinect_hands = tracker.detect_hands_all(color_frame, depth_frame)
    
    # Escolhe a mão desejada (se want_side especificado)
    kinect_landmarks = None
    if kinect_hands:
        if want_side:
            for hand in kinect_hands:
                if hand['side'] == want_side:
                    kinect_landmarks = _landmarks_de_entrada(hand)
                    break
        if kinect_landmarks is None and kinect_hands:
            kinect_landmarks = _landmarks_de_entrada(kinect_hands[0])
    
    kinect_pixels = None
    if kinect_landmarks:
        kinect_pixels = np.array([
            (lm.x * tracker.color_width, lm.y * tracker.color_height)
            for lm in kinect_landmarks
        ], dtype=np.float64)
    
    # Auxiliares
    aux_pixels = {}
    aux_hands_all = {}
    
    for idx, frame in aux_frames.items():
        if frame is None:
            continue
        hands = tracker.detect_auxiliary_hands_all({idx: frame})
        aux_hands_all[idx] = hands.get(idx, [])
        
        if hands.get(idx):
            # Usa a primeira mão detectada (ou filtra por lado se necessário)
            landmarks = _landmarks_de_entrada(hands[idx][0])
            view = alignment.aux_view(idx)
            if view:
                aux_pixels[idx] = view.pixels(landmarks)
    
    return {
        'kinect': kinect_pixels,
        'aux': aux_pixels,
        'kinect_hands': kinect_hands,
        'aux_hands': aux_hands_all,
        'kinect_color': kinect_color,
    }


def draw_overlay(kinect_color, detections, triang_info, alignment, tracker, fps):
    """Desenha overlays na imagem do Kinect."""
    frame = kinect_color.copy()
    
    # Esqueleto do Kinect (verde)
    if detections['kinect_hands']:
        for hand in detections['kinect_hands']:
            landmarks = hand['landmarks']
            pixels = np.array([
                (lm.x * tracker.color_width, lm.y * tracker.color_height)
                for lm in landmarks
            ], dtype=np.float64)
            desenhar_esqueleto_projetado(frame, pixels, cor_nos=(0, 255, 0), cor_linhas=(0, 200, 0),
                                         rotulo="Kinect", raio=3, espessura=2)
    
    # Esqueletos das auxiliares projetados
    colors = {0: (0, 0, 255), 1: (255, 0, 255)}  # aux0=vermelho, aux1=magenta
    labels = {0: "Aux0", 1: "Aux1"}
    
    for idx, hands in detections['aux_hands'].items():
        if not hands:
            continue
        view = alignment.aux_view(idx)
        if view is None:
            continue
        
            landmarks = _landmarks_de_entrada(hands[0])
        aux_px = view.pixels(landmarks)
        
        # Projeta para Kinect usando a triangulação 3D se disponível
        if 'triang_3d' in triang_info and triang_info['triang_3d'] is not None:
            pts3d = triang_info['triang_3d']
            valid = triang_info['valid_mask']
            if valid.any():
                # Projeta pontos 3D para Kinect
                projected, _ = cv2.projectPoints(
                    np.nan_to_num(pts3d).reshape(-1, 1, 3),
                    np.zeros(3), np.zeros(3),
                    alignment.kinect_matrix, alignment.kinect_dist
                )
                kinect_proj = projected.reshape(-1, 2)
                kinect_proj[~valid] = np.nan
                desenhar_esqueleto_projetado(frame, kinect_proj, 
                                             cor_nos=colors.get(idx, (255, 255, 0)),
                                             cor_linhas=tuple(c//2 for c in colors.get(idx, (255, 255, 0))),
                                             rotulo=f"{labels.get(idx, f'Aux{idx}')} 3D", raio=3, espessura=2)
        else:
            # Fallback: projeção stereo simples com profundidade nominal
            depth_guess = 1.0
            if hasattr(tracker, 'last_hand_depth_m') and tracker.last_hand_depth_m:
                depth_guess = tracker.last_hand_depth_m
            kinect_proj = alignment.auxiliary_to_kinect_stereo(aux_px.reshape(-1, 2), [depth_guess])
            if kinect_proj is not None:
                desenhar_esqueleto_projetado(frame, kinect_proj,
                                             cor_nos=colors.get(idx, (255, 255, 0)),
                                             cor_linhas=tuple(c//2 for c in colors.get(idx, (255, 255, 0))),
                                             rotulo=f"{labels.get(idx, f'Aux{idx}')} 2D", raio=3, espessura=2)
    
    # Info de triangulação
    method = triang_info.get('method', 'failed')
    n_valid = triang_info.get('n_valid', 0)
    views = triang_info.get('views_used', [])
    max_reproj = triang_info.get('max_reproj_error_px', 0)
    
    y0 = 30
    cv2.putText(frame, f"Método: {method}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Pontos válidos: {n_valid}/21", (10, y0+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Vistas: {views}", (10, y0+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(frame, f"Max reproj: {max_reproj:.1f}px", (10, y0+75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y0+100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Erros de reprojeção por vista
    reproj = triang_info.get('reproj_errors_median_px', [])
    if len(reproj) > 0:
        names = ['Kinect', 'Aux0', 'Aux1']
        for i, err in enumerate(reproj):
            if not np.isnan(err):
                cv2.putText(frame, f"  {names[i]}: {err:.1f}px", (10, y0+125+i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    return frame


def main():
    print("=" * 60)
    print("DEMONSTRAÇÃO: Triangulação 3 Vistas (Kinect + Aux0 + Aux1)")
    print("=" * 60)
    
    # 1. Inicializa Kinect e alinhamento
    print("\n[1/4] Inicializando Kinect...")
    aux_indices = indices_auxiliares_automaticos()
    if not aux_indices:
        print("  Usando índices padrão:", AUX_CAMERA_INDICES)
        aux_indices = AUX_CAMERA_INDICES
    else:
        print(f"  Câmeras auxiliares detectadas: {aux_indices}")
    
    # Usa apenas as 2 primeiras
    aux_indices = aux_indices[:2]
    print(f"  Usando auxiliares: {aux_indices} ({[nome_auxiliar(i) for i in aux_indices]})")
    
    tracker = KinectHandTracker(aux_indices=aux_indices)
    print("  Kinect OK")
    
    # 2. Carrega alinhamento (calibração estéreo)
    print("\n[2/4] Carregando calibração estéreo...")
    alignment = CameraAlignment(aux_indices=aux_indices)
    
    # Verifica calibrações
    for idx in aux_indices:
        view = alignment.aux_view(idx)
        if view and view.usable:
            print(f"  Aux {idx} ({nome_auxiliar(idx)}): RMS={view.rms:.2f}px - OK")
        else:
            print(f"  Aux {idx}: SEM CALIBRAÇÃO ESTÉREO USÁVEL")
    
    if alignment.kinect_matrix is None:
        print("  ERRO: Kinect sem matriz intrínseca!")
        return 1
    
    # 3. Abre câmeras auxiliares
    print("\n[3/4] Abrindo câmeras auxiliares...")
    aux_caps = open_aux_cameras(aux_indices)
    if len(aux_caps) < 2:
        print("  AVISO: Menos de 2 auxiliares abertas - funcionalidade limitada")
    
    # 4. Prepara logger
    print("\n[4/4] Preparando log CSV...")
    logger = MetricsLogger(CSV_LOG_PATH)
    logger.open()
    print(f"  Log: {CSV_LOG_PATH}")
    
    print("\n" + "=" * 60)
    print("INICIANDO LOOP PRINCIPAL - Pressione Q/ESC para sair")
    print("=" * 60)
    
    # Loop principal
    paused = False
    frame_idx = 0
    fps_history = deque(maxlen=MAX_FPS_HISTORY)
    last_time = time.time()
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    
    try:
        while True:
            loop_start = time.time()
            
            if not paused:
                # Captura frames
                color_frame, depth_frame = tracker.frames()
                if color_frame is None:
                    time.sleep(0.01)
                    continue
                
                aux_frames = read_aux_frames(aux_caps)
                
                # Detecta landmarks
                want_side = None  # None = primeira mão detectada
                detections = get_landmarks_pixels(
                    tracker, color_frame, depth_frame, aux_frames, alignment, want_side
                )
                
                # Prepara pixels para triangulação
                kinect_px = detections['kinect']
                aux0_px = detections['aux'].get(aux_indices[0]) if len(aux_indices) > 0 else None
                aux1_px = detections['aux'].get(aux_indices[1]) if len(aux_indices) > 1 else None
                
                # Triangulação 3 vistas
                triang_info = {'method': 'no_detection', 'n_valid': 0, 'views_used': []}
                
                if kinect_px is not None or aux0_px is not None or aux1_px is not None:
                    pts3d, mask, info = triangulate_three_views(
                        alignment, kinect_px, aux0_px, aux1_px,
                        aux_indices=aux_indices,
                        min_valid_points=5,
                        max_reproj_error_px=15.0
                    )
                    triang_info = info
                    if pts3d is not None:
                        triang_info['triang_3d'] = pts3d
                        triang_info['valid_mask'] = mask
                
                # Calcula FPS
                now = time.time()
                dt = now - last_time
                last_time = now
                fps = 1.0 / dt if dt > 0 else 0
                fps_history.append(fps)
                avg_fps = np.mean(fps_history) if fps_history else fps
                
                # Desenha overlay
                display_frame = draw_overlay(
                    detections['kinect_color'], detections, triang_info, alignment, tracker, avg_fps
                )
                
                # Log
                logger.log(frame_idx, time.time(), triang_info, avg_fps)
                frame_idx += 1
            
            # Mostra frame
            cv2.imshow(WINDOW_NAME, display_frame)
            
            # Teclas
            key = cv2.waitKey(1 if not paused else 100) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q ou ESC
                break
            elif key == ord(' '):  # Espaço - pausa
                paused = not paused
                print(f"{'PAUSADO' if paused else 'CONTINUANDO'}")
            elif key == ord('s'):  # Salva frame
                fname = f"frame_{frame_idx:06d}.png"
                cv2.imwrite(fname, display_frame)
                print(f"Frame salvo: {fname}")
    
    finally:
        # Cleanup
        print("\nFinalizando...")
        tracker.close()
        for cap in aux_caps.values():
            cap.release()
        cv2.destroyAllWindows()
        logger.close()
        print(f"Log salvo em: {CSV_LOG_PATH}")
        print(f"Frames processados: {frame_idx}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
