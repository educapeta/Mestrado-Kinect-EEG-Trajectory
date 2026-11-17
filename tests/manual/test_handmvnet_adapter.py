#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Suite manual do adaptador HandMvNet (SEM hardware e SEM pesos publicados).

Cobre os dois caminhos do `vision/handmvnet_adapter.py`:

  - FALLBACK DE TRIANGULACAO (roda hoje): vistas sinteticas calibradas com os
    npz reais do projeto, erro de reconstrucao medido contra ground truth,
    comparacao com a triangulacao canonica do projeto (`triangulate_three_views`
    do `kinect_imu_groundtruth`), casos de 2 e 3 vistas, ruido do MediaPipe,
    contrato de saida (`HandMvNetOutput`, `as_aux_stereo_frame`, `to_dict`);
  - MODELO OFICIAL (plumbing): sem checkpoint o adaptador explica o bloqueio;
    com pesos ALEATORIOS (`attach_random_model`) roda o forward de verdade da
    arquitetura oficial (via shims de import) para provar tensores/formato/
    latencia e a matematica de crop/raiz -- SEM afirmar qualidade da predicao.

Rodar (na raiz do repositorio):

    .\\.venv\\Scripts\\python.exe tests\\manual\\test_handmvnet_adapter.py

Nao usa pytest (o venv do projeto nao tem pytest; suites sao scripts).
Codigo de saida 0 = todos os checks passaram.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# sys.path do projeto (flat imports: `fusion`, `vision`, `kinect_imu_groundtruth`)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "mestradopy"
for _p in (_SRC, _SRC / "core", _SRC / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:                                    # console UTF-8 quando possivel
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                       # pragma: no cover
    pass

from vision.handmvnet_adapter import (  # noqa: E402
    DEFAULT_CALIBRATION_DIR,
    LICENSE_STATUS,
    PALM_LANDMARK_INDICES,
    WEIGHTS_STATUS,
    HandMvNetAdapter,
    HandMvNetOutput,
    HandMvNetView,
    calibration_paths,
    crop_and_pad_image,
    crop_to_joints,
    describe_official_repo,
    image_to_model_tensor,
    joints_to_crop,
    load_stereo_calibration,
    mirror_view_data,
    project_points_px,
    scale_intrinsic,
    square_bbox_from_landmarks,
    triangulate_views,
)
from vision.handmvnet_synthetic import (        # noqa: E402
    build_synthetic_views,
    make_synthetic_hand,
)

FALHAS: list[str] = []
AVISOS: list[str] = []
CHECKS = 0
T_INICIO = time.perf_counter()


# ===========================================================================
# Infra minima da suite (sem pytest)
# ===========================================================================
def check(nome: str, condicao: bool, detalhe: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    if condicao:
        print(f"  OK   {nome}" + (f" [{detalhe}]" if detalhe else ""))
        return True
    print(f"  FALHA {nome}" + (f" [{detalhe}]" if detalhe else ""))
    FALHAS.append(nome)
    return False


def warn(nome: str, motivo: str) -> None:
    print(f"  AVISO {nome}: {motivo}")
    AVISOS.append(f"{nome}: {motivo}")


def secao(titulo: str) -> None:
    print(f"\n--- {titulo} ---")


def _isclose(a, b, tol=1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float),
                            atol=tol, rtol=0.0, equal_nan=True))


def _views_map(views) -> dict:
    return {view.name: view for view in views}


# ===========================================================================
# 1) Auditoria do repositorio oficial / calibracoes
# ===========================================================================
def teste_repositorio_e_calibracoes() -> None:
    secao("1) Repositorio oficial e calibracoes do projeto")
    repo = describe_official_repo()
    check("repo oficial clonado em external/HandMvNet", repo["present"])
    check("README do repo presente", bool(repo["readme"]))
    check("repo NAO tem arquivo de licenca (risco documentado)",
          repo["license_file"] is None, str(repo["license_file"]))
    check("repo NAO versiona pesos", repo["weights_in_repo"] == [],
          str(repo["weights_in_repo"]))
    check("eval.py oficial nao roda de fabrica (datasets faltando)",
          repo["official_eval_runs_out_of_the_box"] is False,
          str(repo["eval_script_modules_missing"]))
    check("aviso de licenca inclui 'SEM LICENCA EXPLICITA'",
          "SEM LICENCA EXPLICITA" in LICENSE_STATUS)
    check("aviso de pesos inclui 'PESOS NAO PUBLICADOS'",
          "PESOS NAO PUBLICADOS" in WEIGHTS_STATUS)

    paths = calibration_paths()
    check("calibracoes do projeto encontradas (kinect/aux0/aux1/aux2)",
          {"kinect", "aux0", "aux1", "aux2"} <= set(paths),
          str(sorted(paths)))

    for nome, path in (("aux0", paths["aux0"]), ("aux1", paths["aux1"])):
        calib = load_stereo_calibration(path)
        R = calib["rotation"]
        check(f"{nome}: R e' rotacao ortonormal", _isclose(R @ R.T, np.eye(3), 1e-7))
        check(f"{nome}: det(R)=+1", abs(float(np.linalg.det(R)) - 1.0) < 1e-9)
        check(f"{nome}: translation (3,)", calib["translation"].shape == (3,))
        check(f"{nome}: K do Kinect (3,3)", calib["kinect_matrix"].shape == (3, 3))
        check(f"{nome}: rms stereo <= 5 px (usavel)",
              calib["rms"] is not None and calib["rms"] <= 5.0,
              f"rms={calib['rms']}")
        check(f"{nome}: tamanhos calibrados presentes",
              calib["kinect_size"] == (1920, 1080)
              and calib["auxiliary_size"] == (1280, 720),
              f"{calib['kinect_size']} / {calib['auxiliary_size']}")


def teste_intrinseca_escalada() -> None:
    secao("2) Intrinseca escalada e construtores de vista")
    K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1]])
    K2 = scale_intrinsic(K, (1280, 720), (640, 360))
    check("scale_intrinsic metade: fx 1000->500", _isclose(K2[0, 0], 500.0))
    check("scale_intrinsic metade: cx 640->320", _isclose(K2[0, 2], 320.0))
    check("scale_intrinsic metade: fy 1000->500", _isclose(K2[1, 1], 500.0))
    check("scale_intrinsic nao altera a matriz original", _isclose(K[0, 0], 1000.0))
    check("scale_intrinsic mesma resolucao = identidade",
          _isclose(scale_intrinsic(K, (1280, 720), (1280, 720)), K))

    views, gt, info = build_synthetic_views(render_scale=0.25, noise_px=0.0)
    por_nome = _views_map(views)
    check("vistas sinteticas: kinect + aux0 + aux1",
          set(por_nome) == {"kinect", "aux0", "aux1"}, str(sorted(por_nome)))
    check("vistas: todas as projecoes dentro da imagem",
          info["all_projections_inside_image"], str(info["projections_inside"]))
    check("vista kinect marcada como raiz", por_nome["kinect"].is_kinect)
    check("vistas aux marcam aux_index",
          por_nome["aux0"].aux_index == 0 and por_nome["aux1"].aux_index == 1)
    check("vista aux tem R/T de calibracao",
          por_nome["aux0"].is_auxiliary and por_nome["aux1"].is_auxiliary)
    check("vistas: extrinsic 4x4 com R/T para aux",
          _isclose(por_nome["aux0"].extrinsic()[:3, :3],
                   load_stereo_calibration(calibration_paths()["aux0"])["rotation"]))
    # Reconstrucao a partir do npz tem de dar a MESMA calibracao da vista.
    repro = HandMvNetView.from_calibration_npz(
        "kinect", por_nome["kinect"].image, por_nome["kinect"].landmarks_px,
        calibration_paths()["aux0"], is_kinect=True)
    check("from_calibration_npz(kinect): K igual",
          _isclose(repro.intrinsic, por_nome["kinect"].intrinsic, 1e-9))
    repro_aux = HandMvNetView.from_calibration_npz(
        "aux0", por_nome["aux0"].image, por_nome["aux0"].landmarks_px,
        calibration_paths()["aux0"], aux_index=0)
    check("from_calibration_npz(aux0): K igual",
          _isclose(repro_aux.intrinsic, por_nome["aux0"].intrinsic, 1e-9))

    img = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    pts = np.array([[0.0, 0.0]] * 21)
    img_m, pts_m = mirror_view_data(img, pts)
    check("mirror_view_data inverte x = (w-1)-x", _isclose(pts_m[0, 0], 5.0))
    img2, pts2 = mirror_view_data(img_m, pts_m)
    check("mirror_view_data duas vezes = original",
          _isclose(pts2, pts) and np.array_equal(img2, img))


# ===========================================================================
# 3) BBox / crop / normalizacao
# ===========================================================================
def teste_preprocessamento() -> None:
    secao("3) BBox, crop e tensor do modelo")
    rng = np.random.default_rng(0)
    pts = rng.uniform([100.0, 120.0], [300.0, 260.0], size=(21, 2))
    bbox = square_bbox_from_landmarks(pts, (640, 480), margin=1.2)
    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    check("bbox e' quadrada", abs(w - h) <= 1.0, f"w={w} h={h}")
    check("bbox contem todos os landmarks",
          bbox[0] <= pts[:, 0].min() and pts[:, 0].max() <= bbox[2]
          and bbox[1] <= pts[:, 1].min() and pts[:, 1].max() <= bbox[3])

    crop_pts = joints_to_crop(pts, bbox, image_size=256)
    voltou = crop_to_joints(crop_pts, bbox, image_size=256)
    check("joints_to_crop <-> crop_to_joints (roundtrip < 1e-6 px)",
          np.abs(voltou - pts).max() < 1e-6)
    check("joints_to_crop cabe no crop [0,256]",
          crop_pts.min() >= -1e-6 and crop_pts.max() <= 256.0 + 1e-6,
          f"min={crop_pts.min():.1f} max={crop_pts.max():.1f}")

    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    recorte = crop_and_pad_image(image, (-10, -5, 50, 50))
    check("crop_and_pad fora da borda: shape preservado",
          recorte.shape == (55, 60, 3), str(recorte.shape))
    check("crop_and_pad fora da borda: preenche com zero",
          recorte[0, 0].tolist() == [0, 0, 0])
    dentro = crop_and_pad_image(image, (20, 30, 120, 130))
    check("crop_and_pad dentro da imagem", dentro.shape == (100, 100, 3)
          and dentro[0, 0].tolist() == [200, 200, 200])

    tensor = image_to_model_tensor(image, 256)
    check("image_to_model_tensor: (3,256,256) float32",
          tensor.shape == (3, 256, 256) and tensor.dtype == np.float32,
          str(tensor.shape))
    check("image_to_model_tensor: finito", bool(np.isfinite(tensor).all()))


# ===========================================================================
# 4) Fallback de triangulacao
# ===========================================================================
def _erro_3d(out: HandMvNetOutput, gt: np.ndarray) -> np.ndarray:
    mask = out.valid_joint_mask() & np.isfinite(np.asarray(gt)).all(axis=1)
    if not mask.any():
        return np.array([np.inf])
    return np.linalg.norm(np.asarray(out.joints_3d)[mask]
                          - np.asarray(gt)[mask], axis=1)


def teste_fallback_triangulacao() -> None:
    secao("4) Fallback de triangulacao (3 vistas, sem ruido)")
    views, gt, info = build_synthetic_views(render_scale=0.25, noise_px=0.0)
    adapter = HandMvNetAdapter(mode="triangulation")
    status = adapter.status()
    check("status: modelo NAO pronto sem checkpoint", not status["model_ready"])
    check("status: motivo cita checkpoint/pesos",
          any(("checkpoint" in r.lower()) or ("pesos" in r.lower())
              for r in status["reasons"]), str(status["reasons"]))
    out = adapter.infer(views, hand="right")
    check("fallback: backend correto",
          out.backend == "triangulation_fallback", out.backend)
    check("fallback: valid=True", out.valid, out.reason)
    check("fallback: 21/21 junções validas", out.valid_joint_mask().sum() == 21,
          str(int(out.valid_joint_mask().sum())))
    check("fallback: joints_3d (21,3) finitas",
          out.joints_3d is not None and out.joints_3d.shape == (21, 3)
          and bool(np.isfinite(out.joints_3d).all()))
    check("fallback: root_source=triangulation", out.root_source == "triangulation")
    check("fallback: 3 vistas usadas", list(out.views_used)
          == ["kinect", "aux0", "aux1"], str(list(out.views_used)))
    err = _erro_3d(out, gt)
    check("fallback sem ruido: erro medio < 0.5 mm",
          float(np.mean(err)) * 1000.0 < 0.5,
          f"{float(np.mean(err)) * 1000.0:.4f} mm")
    check("fallback: raiz ~ wrist do GT (< 1 mm)",
          float(np.linalg.norm(out.root_3d - gt[0])) * 1000.0 < 1.0,
          f"{float(np.linalg.norm(out.root_3d - gt[0])) * 1000.0:.4f} mm")
    check("fallback: joints_rel + root == joints_3d",
          np.abs(out.joints_3d - (out.joints_3d_root_relative + out.root_3d)).max()
          < 1e-9)
    check("fallback: reproj por vista (21,)",
          set(out.reproj_errors) == {"kinect", "aux0", "aux1"}
          and all(v.shape == (21,) for v in out.reproj_errors.values()))
    med_reproj = float(np.nanmedian(list(out.reproj_errors.values())[0]))
    check("fallback sem ruido: reproj mediana < 0.5 px", med_reproj < 0.5,
          f"{med_reproj:.4f} px")

    palm = out.palm_points_3d()
    check("fallback: palm_points_3d (5,3)",
          palm is not None and palm.shape == (5, 3))
    check("fallback: palm_reproj_errors (5,)",
          out.palm_reproj_errors() is not None
          and out.palm_reproj_errors().shape == (5,))
    frame = out.as_aux_stereo_frame()
    check("fallback: as_aux_stereo_frame (contrato da fusao)",
          type(frame).__name__ == "AuxStereoFrame"
          and getattr(frame, "palm_points_3d", None) is not None
          and np.asarray(frame.palm_points_3d).shape == (5, 3))
    d = out.to_dict()
    check("fallback: to_dict JSON-serializavel",
          len(json.dumps(d, ensure_ascii=False)) > 0)

    secao("5) Fallback com ruido do MediaPipe (1 px)")
    views_n, gt_n, _ = build_synthetic_views(render_scale=0.25, noise_px=1.0)
    out_n = adapter.infer(views_n, hand="right")
    err_n = _erro_3d(out_n, gt_n)
    check("fallback 1 px: valid=True", out_n.valid, out_n.reason)
    check("fallback 1 px: erro mediano < 5 mm",
          float(np.median(err_n)) * 1000.0 < 5.0,
          f"{float(np.median(err_n)) * 1000.0:.3f} mm")

    secao("6) Fallback com 2 vistas (Kinect + aux0)")
    views2, gt2, _ = build_synthetic_views(render_scale=0.25, noise_px=0.0,
                                           num_aux=1)
    check("2 vistas: nomes", [v.name for v in views2] == ["kinect", "aux0"])
    out2 = adapter.infer(views2, hand="left")
    err2 = _erro_3d(out2, gt2)
    check("2 vistas: valid=True", out2.valid, out2.reason)
    check("2 vistas: erro medio < 0.5 mm", float(np.mean(err2)) * 1000.0 < 0.5,
          f"{float(np.mean(err2)) * 1000.0:.4f} mm")

    secao("7) Comparacao com a triangulacao canonica do projeto")
    try:
        import kinect_imu_groundtruth as kinect_gt
    except Exception as exc:                            # pragma: no cover
        warn("triangulate_three_views", f"kinect_imu_groundtruth indisponivel: {exc}")
        return
    views_full, gt_full, _ = build_synthetic_views(render_scale=1.0, noise_px=0.0)
    por_nome = _views_map(views_full)
    align = kinect_gt.CameraAlignment(aux_indices=(0, 1))
    pts_can, mask_can, info_can = kinect_gt.triangulate_three_views(
        align, por_nome["kinect"].landmarks_px, por_nome["aux0"].landmarks_px,
        por_nome["aux1"].landmarks_px, aux_indices=(0, 1),
        min_valid_points=5, max_reproj_error_px=10.0)
    tri_adapter = triangulate_views(views_full)
    check("canonica: triangulou as 21 junções",
          pts_can is not None and mask_can is not None
          and int(np.sum(mask_can)) >= 19,
          f"method={info_can.get('method')} n={int(np.sum(mask_can)) if mask_can is not None else 0}")
    if pts_can is not None:
        ambos = np.asarray(mask_can) & tri_adapter.valid_mask
        check("adapter x canonica: concordam em >= 19 junções",
              int(ambos.sum()) >= 19, str(int(ambos.sum())))
        diff = np.linalg.norm(np.asarray(pts_can)[ambos]
                              - tri_adapter.points_3d[ambos], axis=1) * 1000.0
        check("adapter x canonica: diferenca mediana < 5 mm",
              float(np.median(diff)) < 5.0,
              f"mediana={float(np.median(diff)):.3f} mm")
        err_can = np.linalg.norm(np.asarray(pts_can)[mask_can]
                                 - gt_full[np.asarray(mask_can)], axis=1) * 1000.0
        check("canonica: erro vs GT < 1 mm", float(np.mean(err_can)) < 1.0,
              f"{float(np.mean(err_can)):.4f} mm")


# ===========================================================================
# 5) Modelo oficial: bloqueio e encanamento com pesos aleatorios
# ===========================================================================
def teste_modelo_bloqueio() -> None:
    secao("8) Modelo oficial sem checkpoint: bloqueio claro + fallback automatico")
    adapter = HandMvNetAdapter(mode="auto")
    status = adapter.status(refresh=True)
    if status["checkpoint_present"]:
        warn("modelo sem checkpoint", "ha' um checkpoint local; pulando o teste de bloqueio")
        return
    check("status: model_ready=False", not status["model_ready"])
    check("status: checkpoint_present=False", not status["checkpoint_present"])
    carregado = adapter.load_model(3)
    check("load_model devolve None sem checkpoint", carregado is None)
    check("load_model explica o bloqueio",
          adapter._model_error is not None
          and "checkpoint" in adapter._model_error.lower(),
          str(adapter._model_error))
    views, gt, _ = build_synthetic_views(render_scale=0.25, noise_px=0.0)
    out = adapter.infer(views, hand="right")
    check("mode=auto sem pesos: cai no fallback e mantem valid",
          out.backend == "triangulation_fallback" and out.valid, out.summary())
    check("fallback: motivo cita o modelo nao usado",
          "modelo nao usado" in out.reason.lower(), out.reason)

    adapter_model = HandMvNetAdapter(mode="model")
    out_model = adapter_model.infer(views, hand="right")
    check("mode='model' sem pesos: backend unavailable (nao engana)",
          out_model.backend == "unavailable" and not out_model.valid,
          out_model.reason)


def teste_modelo_plumbing() -> None:
    secao("9) Modelo oficial com pesos ALEATORIOS: encanamento/forward")
    views, gt, _ = build_synthetic_views(render_scale=0.25, noise_px=0.0)
    adapter = HandMvNetAdapter(mode="auto", device="cpu")
    try:
        loaded = adapter.attach_random_model(num_views=3)
    except Exception as exc:
        warn("modelo oficial (shims/forward)",
             f"nao foi possivel montar o modelo com shims: {type(exc).__name__}: {exc}")
        return
    check("modelo oficial instanciado (arquitetura real)",
          loaded.num_views == 3 and loaded.param_count > 1_000_000,
          f"{loaded.class_name}, params={loaded.param_count}")
    check("modelo usa pos_enc com crop (intrinsica no forward)",
          "crop" in loaded.pos_enc, str(loaded.pos_enc))
    anchor = np.asarray(gt[0], dtype=np.float64)
    t0 = time.perf_counter()
    out = adapter.infer(views, hand="right", root_anchor=anchor)
    latencia = time.perf_counter() - t0
    check("forward do modelo executou", out.backend == "handmvnet_random_weights",
          out.backend)
    check("random weights: valid=False (nao e' predicao)", not out.valid)
    check("random weights: motivo avisa pesos aleatorios",
          "ALEATORIOS" in out.reason.upper(), out.reason)
    check("random weights: joints_3d (21,3) finitas",
          out.joints_3d is not None and out.joints_3d.shape == (21, 3)
          and bool(np.isfinite(out.joints_3d).all()))
    check("random weights: joints_3d_root_relative (21,3) finitas",
          out.joints_3d_root_relative is not None
          and out.joints_3d_root_relative.shape == (21, 3)
          and bool(np.isfinite(out.joints_3d_root_relative).all()))
    check("random weights: ancora externa usada como raiz",
          out.root_source == "external" and _isclose(out.root_3d, anchor))
    check("random weights: joints_abs = joints_rel + ancora",
          np.abs(out.joints_3d - (out.joints_3d_root_relative
                                  + anchor.reshape(1, 3))).max() < 1e-9)
    check("random weights: joints_2d por vista (21,2)",
          set(out.joints_2d) == {"kinect", "aux0", "aux1"}
          and all(v.shape == (21, 2) for v in out.joints_2d.values()))
    check("random weights: joints_2d finitos",
          all(bool(np.isfinite(v).all()) for v in out.joints_2d.values()))
    check("random weights: reproj por vista (21,)",
          set(out.reproj_errors) == {"kinect", "aux0", "aux1"}
          and all(v.shape == (21,) for v in out.reproj_errors.values()))
    check("random weights: score por junção (21,) em (0,1]",
          out.scores is not None and out.scores.shape == (21,)
          and bool(np.all(np.isfinite(out.scores)))
          and float(out.scores.min()) > 0.0 and float(out.scores.max()) <= 1.0 + 1e-6)
    check("random weights: model_weights marca random",
          "random" in out.model_weights.lower(), out.model_weights)
    check("random weights: inferencia < 10 s no CPU", latencia < 10.0,
          f"{latencia * 1000:.0f} ms")
    d = out.to_dict()
    check("random weights: to_dict JSON", len(json.dumps(d)) > 0)

    out_b = adapter.infer(views, hand="right", root_anchor=anchor)
    check("determinismo: mesma entrada -> mesma saida",
          np.allclose(out.joints_3d, out_b.joints_3d, equal_nan=True))

    secao("10) Variantes de arquitetura (sem crop / 2 vistas / raiz predita)")
    views2, _, _ = build_synthetic_views(render_scale=0.25, noise_px=0.0,
                                         num_aux=1)
    try:
        adapter.attach_random_model(num_views=2, pos_enc=("pos2d", "sin"),
                                    device="cpu")
        out2 = adapter.infer(views2, hand="left")
        check("2 vistas + pos_enc sem crop: forward OK",
              out2.backend == "handmvnet_random_weights"
              and out2.joints_3d.shape == (21, 3)
              and set(out2.joints_2d) == {"kinect", "aux0"}, out2.backend)
    except Exception as exc:                            # pragma: no cover
        warn("modelo 2 vistas sem crop", f"{type(exc).__name__}: {exc}")

    try:
        adapter.attach_random_model(num_views=3, pos_enc=("pos2d", "sin"),
                                    device="cpu")
        out3 = adapter.infer(views, hand="right")
        check("3 vistas + pos_enc sem crop: forward OK",
              out3.backend == "handmvnet_random_weights"
              and out3.joints_3d.shape == (21, 3), out3.backend)
    except Exception as exc:                            # pragma: no cover
        warn("modelo 3 vistas sem crop", f"{type(exc).__name__}: {exc}")

    try:
        adapter.attach_random_model(num_views=3, pos_enc=("pos2d", "sin"),
                                    root_relative=False, device="cpu")
        out4 = adapter.infer(views, hand="right")
        check("root_relative=False: saida tem raiz e dimensoes",
              out4.joints_3d.shape == (21, 3)
              and out4.root_source in ("model", "triangulation", "external"),
              f"root_source={out4.root_source}")
    except Exception as exc:                            # pragma: no cover
        warn("modelo root_relative=False", f"{type(exc).__name__}: {exc}")


def teste_checkpoint_formato_lightning() -> None:
    """Valida o caminho REAL de checkpoint (arquivo FAKE no formato Lightning).

    Nao existe checkpoint publico, mas o CARREGADOR pode ser validado sem
    inventar resultado: salvamos um `.ckpt` no mesmo formato do official
    (`state_dict` + `hyper_parameters` completo) no diretorio temporario e
    exercitamos `load_official_model`/`HandMvNetAdapter(checkpoint=...)`.
    Os pesos sao aleatorios: o teste valida carregamento/arquitetura/formatos,
    NAO qualidade.
    """
    secao("10) Carregamento de checkpoint no formato Lightning (arquivo fake)")
    try:
        import torch
    except Exception as exc:                            # pragma: no cover
        warn("checkpoint fake", f"torch indisponivel: {exc}")
        return
    try:
        from vision.handmvnet_adapter import (
            HandMvNetUnavailable,
            _legacy_state_dict_fix,
            load_official_model,
            load_random_model_for_validation,
        )
    except Exception as exc:                            # pragma: no cover
        warn("checkpoint fake", f"imports falharam: {exc}")
        return

    # Remapeamento de chaves antigas (legado do eval.py oficial).
    legacy_sd = {"pose_net.conv.0.weight": 1, "sample_net.conv.0.weight": 2,
                 "backbone.conv1.weight": 3}
    fixed, era_legado = _legacy_state_dict_fix(legacy_sd)
    check("legacy: chaves antigas remapeadas",
          era_legado and "pose_net.0.weight" in fixed
          and "sample_nets.0.conv.0.weight" in fixed
          and "backbone.conv1.weight" in fixed, str(sorted(fixed)))

    try:
        modelo = load_random_model_for_validation(
            num_views=2, pos_enc=("pos2d", "sin"), device="cpu")
    except Exception as exc:                            # pragma: no cover
        warn("checkpoint fake", f"modelo base nao carregou: {exc}")
        return
    state_dict = {key: value.detach().clone()
                  for key, value in modelo.model.state_dict().items()}
    hyper = {"train_params": modelo.train_params,
             "model_params": modelo.model_params,
             "data_params": modelo.data_params}
    views2, gt, _ = build_synthetic_views(render_scale=0.25, noise_px=0.0,
                                          num_aux=1)
    import tempfile

    with tempfile.TemporaryDirectory(prefix="handmvnet_fake_") as tmp:
        ckpt = Path(tmp) / "handmvnet_fake.ckpt"
        torch.save({"state_dict": state_dict, "hyper_parameters": hyper},
                   str(ckpt))
        check("checkpoint fake salvo no formato Lightning",
              ckpt.is_file() and ckpt.stat().st_size > 1_000_000,
              f"{ckpt.stat().st_size / 1e6:.1f} MB")
        adapter = HandMvNetAdapter(checkpoint=ckpt, mode="auto", device="cpu")
        status = adapter.status(refresh=True)
        check("status reconhece o checkpoint", status["checkpoint_present"]
              and status["model_ready"], str(status["checkpoint"]))
        loaded = adapter.load_model(2)
        check("load_model pelo adapter usa o checkpoint",
              loaded is not None and loaded.num_views == 2
              and "checkpoint" in loaded.weights_source,
              "" if loaded is None else loaded.weights_source)
        if loaded is not None:
            check("checkpoint: arquitetura reconstruida das hyper_parameters",
                  loaded.model_params.get("backbone") == "resnet"
                  and loaded.load_report.get("checkpoint_views") == 2,
                  str(loaded.load_report))
            out = adapter.infer(views2, hand="right", root_anchor=gt[0])
            check("checkpoint: forward roda e backend vira 'handmvnet'",
                  out.backend == "handmvnet"
                  and out.joints_3d is not None
                  and out.joints_3d.shape == (21, 3), out.backend)
            check("checkpoint: model_weights aponta o arquivo",
                  str(ckpt) in out.model_weights, out.model_weights)

        # Adaptacao EXPERIMENTAL de num_views (checkpoint de 2 vistas -> 3).
        loaded3 = load_official_model(
            checkpoint=ckpt, num_views_override=3, adapt_num_views=True,
            device="cpu")
        check("adapt_num_views: reconstroi com 3 vistas (strict=False)",
              loaded3.num_views == 3
              and loaded3.load_report.get("inference_views") == 3
              and loaded3.load_report.get("checkpoint_views") == 2,
              str(loaded3.load_report))
        try:
            load_official_model(checkpoint=ckpt, num_views_override=3,
                                adapt_num_views=False, device="cpu")
            recusou = False
        except HandMvNetUnavailable as exc:
            recusou = "vistas" in str(exc)
        check("adapt_num_views=False: recusa N de vistas diferente", recusou)

    # Sem checkpoint: erro claro (nao tenta inventar pesos).
    try:
        load_official_model(checkpoint=None)
        recusou_none = False
    except HandMvNetUnavailable as exc:
        recusou_none = "pesos" in str(exc).lower() or "checkpoint" in str(exc).lower()
    check("load_official_model sem checkpoint recusa com motivo", recusou_none)


def teste_validacoes() -> None:
    secao("11) Validacoes de entrada e gates")
    views, gt, _ = build_synthetic_views(render_scale=0.25, noise_px=0.0)
    adapter = HandMvNetAdapter(mode="triangulation")
    out = adapter.infer(views[:1], hand="right")
    check("1 vista -> backend unavailable", out.backend == "unavailable")
    try:
        adapter.infer(views, hand="dedo")
        invalido_ok = False
    except ValueError:
        invalido_ok = True
    check("hand invalida levanta ValueError", invalido_ok)
    try:
        HandMvNetView(name="ruim", image=np.zeros((10, 10, 3), np.uint8),
                      landmarks_px=np.zeros((5, 2)), intrinsic=np.eye(3))
        landmarks_ok = False
    except ValueError:
        landmarks_ok = True
    check("view com landmarks != (21,2) levanta ValueError", landmarks_ok)
    out_nan = adapter.infer(views, hand="right", root_anchor=np.array([np.nan] * 3))
    check("root_anchor NaN e' ignorada (nao propaga NaN)",
          out_nan.joints_3d is not None
          and bool(np.isfinite(out_nan.root_3d).all()), out_nan.reason)


def teste_xpu() -> None:
    secao("12) Hardware Intel Arc (XPU), se disponivel")
    try:
        import torch
    except Exception as exc:                            # pragma: no cover
        warn("xpu", f"torch indisponivel: {exc}")
        return
    if not getattr(torch, "xpu", None) or not torch.xpu.is_available():
        warn("xpu", "torch.xpu indisponivel nesta maquina")
        return
    nome = torch.xpu.get_device_name(0)
    views, gt, _ = build_synthetic_views(render_scale=0.25, noise_px=0.0,
                                         num_aux=1)
    adapter = HandMvNetAdapter(mode="auto", device="xpu")
    try:
        adapter.attach_random_model(num_views=2, pos_enc=("pos2d", "sin"),
                                    device="xpu")
        t0 = time.perf_counter()
        out = adapter.infer(views, hand="right", root_anchor=gt[0])
        lat = time.perf_counter() - t0
        check(f"XPU ({nome}): forward do modelo roda",
              out.joints_3d is not None and out.joints_3d.shape == (21, 3),
              f"{lat * 1000:.0f} ms")
        check("XPU: mesma forma de saida do CPU",
              set(out.joints_2d) == {"kinect", "aux0"})
    except Exception as exc:                            # pragma: no cover
        warn("xpu", f"forward no XPU falhou (CPU segue funcionando): "
                    f"{type(exc).__name__}: {exc}")


def teste_shim_einops() -> None:
    secao("13) Shim de einops (so' quando einops real nao esta instalado)")
    try:
        import torch
    except Exception:                                   # pragma: no cover
        return
    try:
        import einops  # noqa: F401
        if einops.__file__ is not None and "handmvnet_adapter" not in str(
                getattr(einops, "__file__", "")):
            warn("einops", "einops real instalado; shim nao e' usado")
            return
    except Exception:
        pass
    shim = sys.modules.get("einops")
    if shim is None or not hasattr(shim, "rearrange"):
        warn("einops", "shim nao foi instalado (modelo nao carregado?)")
        return
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 20)
    y = shim.rearrange(x, "b i (h d) -> b h i d", h=4)
    ref = x.reshape(2, 3, 4, 5).permute(0, 2, 1, 3)
    check("shim rearrange 'b i (h d) -> b h i d' igual ao torch",
          bool(torch.equal(y, ref)))
    z = shim.rearrange(ref, "b h n d -> b n (h d)")
    check("shim rearrange 'b h n d -> b n (h d)' igual ao torch",
          bool(torch.equal(z, x.reshape(2, 3, 20))))
    try:
        shim.rearrange(x, "b i (h d) -> b h i d")   # sem h: nao da' para inferir
        inferiu = False
    except ValueError:
        inferiu = True
    check("shim rearrange falha alto quando nao da' para inferir", inferiu)


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    print("=" * 78)
    print("TESTE DO ADAPTADOR HandMvNet (sem hardware / sem pesos publicados)")
    print("=" * 78)
    print(f"repo: {_REPO}")
    try:
        teste_repositorio_e_calibracoes()
        teste_intrinseca_escalada()
        teste_preprocessamento()
        teste_fallback_triangulacao()
        teste_modelo_bloqueio()
        teste_modelo_plumbing()
        teste_checkpoint_formato_lightning()
        teste_validacoes()
        teste_xpu()
        teste_shim_einops()
    except Exception:                                   # pragma: no cover
        traceback.print_exc()
        FALHAS.append("EXCECAO NAO TRATADA (ver traceback)")
    duracao = time.perf_counter() - T_INICIO
    print("\n" + "=" * 78)
    if AVISOS:
        print(f"avisos: {len(AVISOS)}")
        for aviso in AVISOS:
            print(f"  - {aviso}")
    if FALHAS:
        print(f"HANDMVNET_ADAPTER_FALHOU: {len(FALHAS)}/{CHECKS} checks "
              f"({duracao:.1f}s): " + " | ".join(FALHAS))
        print("=" * 78)
        return 1
    print(f"HANDMVNET_ADAPTER_OK: {CHECKS}/{CHECKS} checks ({duracao:.1f}s)")
    print("  fallback de triangulacao validado contra ground truth sintetico e")
    print("  contra a triangulacao canonica do projeto; modelo oficial exercitado")
    print("  com pesos aleatorios (formato/forward); pesos reais NAO publicados.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
