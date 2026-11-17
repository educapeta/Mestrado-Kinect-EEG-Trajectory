#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste sintético da triangulação de 3 vistas.

Gera pontos 3D conhecidos no frame do Kinect, projeta nas 3 câmeras
usando as calibrações reais, adiciona ruído opcional, e mede o erro
de reconstrução da triangulação.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinect_imu_groundtruth import (
    CameraAlignment,
    build_projection_matrices,
    triangulate_three_views,
    _undistort_points_batch,
    _triangulate_dlt_multi,
    _reprojection_errors,
)


def create_synthetic_test():
    """Cria um cenário de teste sintético com calibrações reais."""
    
    # Carrega a calibração real
    print("Carregando calibrações...")
    alignment = CameraAlignment(aux_indices=(0, 1))
    
    # Verifica se as calibrações estão disponíveis
    if alignment.kinect_matrix is None:
        print("ERRO: Calibração do Kinect não encontrada")
        return False
    
    view0 = alignment.aux_view(0)
    view1 = alignment.aux_view(1)
    
    if view0 is None or not view0.usable:
        print("ERRO: Aux 0 não utilizável")
        return False
    
    if view1 is None or not view1.usable:
        print("ERRO: Aux 1 não utilizável")
        return False
    
    print(f"Kinect: {alignment.kinect_matrix.shape}")
    print(f"Aux 0 (idx={view0.index}): RMS={view0.rms:.2f}px, size={view0.size}")
    print(f"Aux 1 (idx={view1.index}): RMS={view1.rms:.2f}px, size={view1.size}")
    
    return alignment


def generate_test_points_3d(n_points=21, seed=42):
    """Gera pontos 3D sintéticos no frame do Kinect simulando landmarks da mão."""
    rng = np.random.default_rng(seed)
    
    # Centro da mão a ~1m do Kinect
    center = rng.uniform([-0.1, -0.1, 0.8], [0.1, 0.1, 1.2])
    
    # Gera pontos em torno do centro (escala ~0.08m = tamanho da mão)
    points = center + rng.normal(0, 0.04, size=(n_points, 3))
    
    # Garante Z positivo (frente da câmera)
    points[:, 2] = np.clip(points[:, 2], 0.5, 2.0)
    
    return points


def project_points_to_cameras(points_3d, alignment):
    """Projeta pontos 3D nas 3 câmeras usando as calibrações reais."""
    
    # Kinect
    K_kin = alignment.kinect_matrix
    dist_kin = alignment.kinect_dist
    kin_pixels, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3).astype(np.float32),
        np.zeros(3, np.float32), np.zeros(3, np.float32),
        K_kin.astype(np.float32), dist_kin.astype(np.float32)
    )
    kin_pixels = kin_pixels.reshape(-1, 2)
    
    # Aux 0
    view0 = alignment.aux_view(0)
    K0 = view0.effective_matrix
    dist0 = view0.dist
    R0 = view0.rotation
    T0 = view0.translation
    
    # Transforma para frame da aux0: X_aux = R X_kin + T
    points_aux0 = (R0 @ points_3d.T + T0.reshape(3, 1)).T
    aux0_pixels, _ = cv2.projectPoints(
        points_aux0.reshape(-1, 1, 3).astype(np.float32),
        np.zeros(3, np.float32), np.zeros(3, np.float32),
        K0.astype(np.float32), dist0.astype(np.float32)
    )
    aux0_pixels = aux0_pixels.reshape(-1, 2)
    
    # Aux 1
    view1 = alignment.aux_view(1)
    K1 = view1.effective_matrix
    dist1 = view1.dist
    R1 = view1.rotation
    T1 = view1.translation
    
    points_aux1 = (R1 @ points_3d.T + T1.reshape(3, 1)).T
    aux1_pixels, _ = cv2.projectPoints(
        points_aux1.reshape(-1, 1, 3).astype(np.float32),
        np.zeros(3, np.float32), np.zeros(3, np.float32),
        K1.astype(np.float32), dist1.astype(np.float32)
    )
    aux1_pixels = aux1_pixels.reshape(-1, 2)
    
    return kin_pixels, aux0_pixels, aux1_pixels


def add_noise(pixels, noise_px=0.5, seed=123):
    """Adiciona ruído gaussiano aos pixels."""
    rng = np.random.default_rng(seed)
    noisy = pixels.copy()
    noisy += rng.normal(0, noise_px, size=pixels.shape)
    return noisy


def compute_errors(gt_points, est_points, valid_mask):
    """Calcula erros de reconstrução."""
    errors = np.linalg.norm(gt_points - est_points, axis=1)
    valid_errors = errors[valid_mask]
    
    if len(valid_errors) == 0:
        return None
    
    return {
        'mean_mm': np.mean(valid_errors) * 1000,
        'median_mm': np.median(valid_errors) * 1000,
        'std_mm': np.std(valid_errors) * 1000,
        'max_mm': np.max(valid_errors) * 1000,
        'p90_mm': np.percentile(valid_errors, 90) * 1000,
        'n_valid': np.sum(valid_mask),
        'n_total': len(valid_mask),
    }


def run_synthetic_test():
    """Executa o teste sintético completo."""
    
    print("=" * 60)
    print("TESTE SINTÉTICO - TRIANGULAÇÃO 3 VISTAS")
    print("=" * 60)
    
    # 1. Carrega calibrações
    alignment = create_synthetic_test()
    if not alignment:
        return False
    
    # 2. Gera pontos 3D ground truth
    print("\nGerando pontos 3D sintéticos...")
    gt_points = generate_test_points_3d(21, seed=42)
    print(f"Pontos GT: Z range = [{gt_points[:,2].min():.3f}, {gt_points[:,2].max():.3f}] m")
    
    # 3. Projeta nas câmeras (sem ruído - caso ideal)
    print("\nProjetando nas 3 câmeras (sem ruído)...")
    kin_px, aux0_px, aux1_px = project_points_to_cameras(gt_points, alignment)
    
    # 4. Triangulação 3 vistas (sem ruído)
    print("\n--- Triangulação 3 vistas (sem ruído) ---")
    pts3d, mask, info = triangulate_three_views(
        alignment, kin_px, aux0_px, aux1_px,
        aux_indices=(0, 1),
        min_valid_points=5,
        max_reproj_error_px=10.0
    )
    
    if pts3d is not None:
        errors = compute_errors(gt_points, pts3d, mask)
        print(f"Método: {info['method']}")
        print(f"Views usadas: {info['views_used']}")
        print(f"Pontos válidos: {errors['n_valid']}/{errors['n_total']}")
        print(f"Erro médio: {errors['mean_mm']:.2f} mm")
        print(f"Erro mediano: {errors['median_mm']:.2f} mm")
        print(f"Desvio padrão: {errors['std_mm']:.2f} mm")
        print(f"Erro máx: {errors['max_mm']:.2f} mm")
        print(f"P90: {errors['p90_mm']:.2f} mm")
        if 'reproj_errors_median_px' in info:
            print(f"Reproj. medianas (px): {info['reproj_errors_median_px']}")
    else:
        print(f"FALHOU: {info}")
        return False
    
    # 5. Teste com ruído realista (0.5 px)
    print("\n--- Triangulação 3 vistas (ruído 0.5 px) ---")
    kin_noisy = add_noise(kin_px, 0.5, seed=1)
    aux0_noisy = add_noise(aux0_px, 0.5, seed=2)
    aux1_noisy = add_noise(aux1_px, 0.5, seed=3)
    
    pts3d_n, mask_n, info_n = triangulate_three_views(
        alignment, kin_noisy, aux0_noisy, aux1_noisy,
        aux_indices=(0, 1),
        min_valid_points=5,
        max_reproj_error_px=10.0
    )
    
    if pts3d_n is not None:
        errors_n = compute_errors(gt_points, pts3d_n, mask_n)
        print(f"Método: {info_n['method']}")
        print(f"Pontos válidos: {errors_n['n_valid']}/{errors_n['n_total']}")
        print(f"Erro médio: {errors_n['mean_mm']:.2f} mm")
        print(f"Erro mediano: {errors_n['median_mm']:.2f} mm")
        print(f"Desvio padrão: {errors_n['std_mm']:.2f} mm")
        print(f"Erro máx: {errors_n['max_mm']:.2f} mm")
        print(f"P90: {errors_n['p90_mm']:.2f} mm")
    else:
        print(f"FALHOU: {info_n}")
    
    # 6. Teste de fallback: simula falha de uma câmera
    print("\n--- Teste de FALLBACK (Aux 1 falha) ---")
    pts3d_fb, mask_fb, info_fb = triangulate_three_views(
        alignment, kin_noisy, aux0_noisy, None,  # Aux 1 = None
        aux_indices=(0, 1),
        min_valid_points=5,
        max_reproj_error_px=10.0
    )
    
    if pts3d_fb is not None:
        errors_fb = compute_errors(gt_points, pts3d_fb, mask_fb)
        print(f"Método: {info_fb['method']}")
        print(f"Views usadas: {info_fb['views_used']}")
        print(f"Pontos válidos: {errors_fb['n_valid']}/{errors_fb['n_total']}")
        print(f"Erro médio: {errors_fb['mean_mm']:.2f} mm")
        print(f"Erro mediano: {errors_fb['median_mm']:.2f} mm")
    else:
        print(f"FALHOU: {info_fb}")
    
    # 7. Teste de fallback: Kinect falha (apenas aux-aux)
    print("\n--- Teste de FALLBACK (Kinect falha, apenas Aux-Aux) ---")
    pts3d_fb2, mask_fb2, info_fb2 = triangulate_three_views(
        alignment, None, aux0_noisy, aux1_noisy,  # Kinect = None
        aux_indices=(0, 1),
        min_valid_points=5,
        max_reproj_error_px=10.0
    )
    
    if pts3d_fb2 is not None:
        errors_fb2 = compute_errors(gt_points, pts3d_fb2, mask_fb2)
        print(f"Método: {info_fb2['method']}")
        print(f"Views usadas: {info_fb2['views_used']}")
        print(f"Pontos válidos: {errors_fb2['n_valid']}/{errors_fb2['n_total']}")
        print(f"Erro médio: {errors_fb2['mean_mm']:.2f} mm")
        print(f"Erro mediano: {errors_fb2['median_mm']:.2f} mm")
    else:
        print(f"FALHOU: {info_fb2}")
    
    # 8. Teste com ruído maior (2.0 px)
    print("\n--- Triangulação 3 vistas (ruído 2.0 px) ---")
    kin_noisy2 = add_noise(kin_px, 2.0, seed=10)
    aux0_noisy2 = add_noise(aux0_px, 2.0, seed=11)
    aux1_noisy2 = add_noise(aux1_px, 2.0, seed=12)
    
    pts3d_n2, mask_n2, info_n2 = triangulate_three_views(
        alignment, kin_noisy2, aux0_noisy2, aux1_noisy2,
        aux_indices=(0, 1),
        min_valid_points=5,
        max_reproj_error_px=15.0  # Gate mais tolerante
    )
    
    if pts3d_n2 is not None:
        errors_n2 = compute_errors(gt_points, pts3d_n2, mask_n2)
        print(f"Método: {info_n2['method']}")
        print(f"Pontos válidos: {errors_n2['n_valid']}/{errors_n2['n_total']}")
        print(f"Erro médio: {errors_n2['mean_mm']:.2f} mm")
        print(f"Erro mediano: {errors_n2['median_mm']:.2f} mm")
        print(f"Desvio padrão: {errors_n2['std_mm']:.2f} mm")
        print(f"Erro máx: {errors_n2['max_mm']:.2f} mm")
    else:
        print(f"FALHOU: {info_n2}")
    
    print("\n" + "=" * 60)
    print("TESTE SINTÉTICO CONCLUÍDO")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = run_synthetic_test()
    sys.exit(0 if success else 1)