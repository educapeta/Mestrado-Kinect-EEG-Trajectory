#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entry point do adaptador HandMvNet (status / smoke sintetico / frames npz).

Ferramenta de bancada SEM hardware para avaliar e integrar o HandMvNet ao
pipeline multiview do projeto. Nao abre cameras: usa as calibracoes reais de
`configs/calibracao/`, gera (ou carrega) frames sincronizados e roda o
adaptador `vision.handmvnet_adapter`.

Uso:
    # 1) Diagnostico: repositorio, dependencias, checkpoint, licenca, pesos
    .\\.venv\\Scripts\\python.exe tools\\handmvnet_infer.py status
    .\\.venv\\Scripts\\python.exe tools\\handmvnet_infer.py status --json

    # 2) Smoke sintetico (fallback de triangulacao, sem pesos, sem hardware)
    .\\.venv\\Scripts\\python.exe tools\\handmvnet_infer.py synthetic --triangulation
    .\\.venv\\Scripts\\python.exe tools\\handmvnet_infer.py synthetic --views 2 --noise-px 1.0

    # 3) Validar o ENCANAMENTO do modelo oficial com pesos ALEATORIOS
    #    (prova imports/forward/formatos; NAO e' predicao)
    .\\.venv\\Scripts\\python.exe tools\\handmvnet_infer.py synthetic --random-weights

    # 4) Com um checkpoint real (quando existir):
    .\\.venv\\Scripts\\python.exe tools\\handmvnet_infer.py synthetic --checkpoint C:\\caminho\\handmvnet.ckpt
    $env:HANDMVNET_CHECKPOINT = "C:\\caminho\\handmvnet.ckpt"

    # 5) Frames gravados em .npz (ver formato em `infer-npz --help`)
    .\\.venv\\Scripts\\python.exe tools\\handmvnet_infer.py infer-npz frames.npz --hand right

Saida: texto legivel (default) ou JSON (`--json`) para o Lead/automacao.
Codigo de saida: 0 = pipeline executou; 1 = falha de configuracao/dados.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# sys.path do projeto (mesmo esquema dos outros tools: flat imports)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[1]                    # src/mestradopy
for _p in (_SRC, _SRC / "core", _SRC / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vision.handmvnet_adapter import (      # noqa: E402
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_REPO_ROOT,
    LICENSE_STATUS,
    WEIGHTS_STATUS,
    HandMvNetAdapter,
    HandMvNetOutput,
    HandMvNetView,
    calibration_paths,
    describe_official_repo,
    load_stereo_calibration,
)
from vision.handmvnet_synthetic import (    # noqa: E402
    build_synthetic_views,
    save_synthetic_frames,
)


# ===========================================================================
# Helpers de impressao
# ===========================================================================
def _print_header(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def _print_kv(pairs: Sequence[tuple], indent: int = 2) -> None:
    width = max((len(str(k)) for k, _ in pairs), default=0)
    for key, value in pairs:
        print(" " * indent + f"{str(key):<{width}} : {value}")


def _fmt_metrics(out: HandMvNetOutput, gt: Optional[np.ndarray]) -> Dict[str, Any]:
    """Metricas do smoke: erro 3D vs ground truth e reprojecao por vista."""
    metrics: Dict[str, Any] = {
        "backend": out.backend,
        "valid": bool(out.valid),
        "n_valid": int(out.valid_joint_mask().sum()),
        "latency_ms": round(float(out.latency_ms), 2),
        "reason": out.reason,
        "reproj_median_px": {},
        "error_vs_gt_mm": None,
    }
    for name, errs in out.reproj_errors.items():
        arr = np.asarray(errs, dtype=np.float64)
        metrics["reproj_median_px"][name] = (
            None if not np.isfinite(arr).any() else round(float(np.nanmedian(arr)), 2))
    if gt is not None and out.joints_3d is not None:
        mask = out.valid_joint_mask() & np.isfinite(np.asarray(gt)).all(axis=1)
        if mask.any():
            err = np.linalg.norm(np.asarray(out.joints_3d)[mask]
                                 - np.asarray(gt)[mask], axis=1) * 1000.0
            metrics["error_vs_gt_mm"] = {
                "n": int(mask.sum()),
                "mean": round(float(np.mean(err)), 3),
                "median": round(float(np.median(err)), 3),
                "p90": round(float(np.percentile(err, 90)), 3),
                "max": round(float(np.max(err)), 3),
            }
    return metrics


def _print_output(out: HandMvNetOutput, metrics: Dict[str, Any]) -> None:
    print()
    print("RESULTADO DO ADAPTADOR")
    print("-" * 78)
    _print_kv([
        ("hand", out.hand),
        ("backend", out.backend),
        ("valid", out.valid),
        ("junções válidas", f"{metrics['n_valid']}/21"),
        ("raiz 3D (m)", None if out.root_3d is None
         else np.round(out.root_3d, 4).tolist()),
        ("origem da raiz", out.root_source or "-"),
        ("latência", f"{out.latency_ms:.1f} ms"),
        ("vistas", list(out.views_used)),
        ("pesos", out.model_weights or "-"),
        ("motivo", out.reason or "-"),
    ])
    if metrics["reproj_median_px"]:
        print("  reprojecao mediana (px) : "
              + ", ".join(f"{k}={v}" for k, v in metrics["reproj_median_px"].items()))
    if metrics["error_vs_gt_mm"]:
        e = metrics["error_vs_gt_mm"]
        print(f"  erro 3D vs GT ({e['n']} junções) : "
              f"média={e['mean']:.2f}mm mediana={e['median']:.2f}mm "
              f"p90={e['p90']:.2f}mm max={e['max']:.2f}mm")
    if metrics.get("model_only_plumbing"):
        print("  ATENCAO: pesos aleatorios -> numeros NAO sao predicao; o teste "
              "valida shapes/forward/formato.")
    if out.warnings:
        print("  avisos:")
        for warning in out.warnings:
            print(f"    - {warning}")


# ===========================================================================
# Subcomando: status
# ===========================================================================
def cmd_status(args: argparse.Namespace) -> int:
    repo = describe_official_repo(args.repo_root)
    adapter = HandMvNetAdapter(
        repo_root=args.repo_root,
        checkpoint=args.checkpoint,
        calibration_dir=args.calibration_dir,
        device=args.device,
        mode="auto",
    )
    status = adapter.status(refresh=True)
    calib = {name: str(path) for name, path in
             calibration_paths(args.calibration_dir).items()}
    payload = {
        "repo": repo,
        "status": status,
        "calibrations": calib,
        "device_requested": args.device,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return 0

    _print_header("HandMvNet -- diagnostico de integracao")
    print("\n[1] Repositorio oficial")
    _print_kv([
        ("URL", repo["url"]),
        ("clone em", repo["repo_root"]),
        ("presente", repo["present"]),
        ("commit", (repo["commit"] or "-")[:12]),
        ("arquivo de licenca", repo.get("license_file") or "AUSENTE"),
        ("pesos no repo", repo["weights_in_repo"] or "NENHUM"),
        ("eval.py roda de fabrica", repo["official_eval_runs_out_of_the_box"]),
        ("modulos que faltam p/ eval", repo["eval_script_modules_missing"]),
    ])
    print("  licenca : " + LICENSE_STATUS)
    print("  pesos   : " + WEIGHTS_STATUS)

    print("\n[2] Dependencias do modo modelo (informativo)")
    for name, ok in status["optional_deps"].items():
        mark = "OK" if ok else "AUSENTE (shim cobre o import de inferencia)"
        print(f"    {name:<28} {mark}")
    print(f"    torch                        "
          f"{'OK' if status['torch_available'] else 'AUSENTE'}")
    print(f"    device pedido/resolvido      {status['device_requested']} / "
          f"{status.get('device_resolved', 'nao resolvido (modelo nao carregado)')}")

    print("\n[3] Checkpoint (pesos)")
    _print_kv([
        ("procurado em", f"{Path(args.repo_root) / 'weights'} (e env "
                         "HANDMVNET_CHECKPOINT)"),
        ("encontrado", status["checkpoint"] or "NAO"),
        ("modelo pronto", status["model_ready"]),
    ])
    for reason in status["reasons"]:
        print(f"    motivo: {reason}")

    print("\n[4] Calibracoes do projeto")
    for name, path in calib.items():
        print(f"    {name:<8} {path}")
    if not calib:
        print("    NENHUMA calibracao encontrada em:", args.calibration_dir)

    print("\n[5] Como habilitar o modo modelo (quando houver pesos)")
    print("    $env:HANDMVNET_CHECKPOINT = 'C:\\caminho\\handmvnet.ckpt'")
    print("    .\\.venv\\Scripts\\python.exe tools\\handmvnet_infer.py status --json")
    print("    .\\.venv\\Scripts\\python.exe tools\\handmvnet_infer.py synthetic "
          "--checkpoint C:\\caminho\\handmvnet.ckpt")
    print("\n    Obs.: os checkpoints publicos do repo oficial foram treinados com")
    print("    5 (HO3D), 8 (DexYCB) ou 4 (MVHand) vistas; para o nosso rig de 3")
    print("    vistas use `adapt_num_views` (experimental) ou retreine.")
    return 0


# ===========================================================================
# Subcomando: synthetic
# ===========================================================================
def _make_adapter(args: argparse.Namespace) -> HandMvNetAdapter:
    return HandMvNetAdapter(
        repo_root=args.repo_root,
        checkpoint=args.checkpoint,
        calibration_dir=args.calibration_dir,
        device=args.device,
        mode=args.mode,
        image_size=args.image_size,
        crop_margin=args.crop_margin,
        adapt_num_views=args.adapt_num_views,
        allow_import_shims=not args.no_shims,
        verbose=False,
    )


def cmd_synthetic(args: argparse.Namespace) -> int:
    num_aux = max(int(args.views) - 1, 0)
    calib_dir = Path(args.calibration_dir)
    available = calibration_paths(calib_dir)
    if not available:
        raise SystemExit(f"sem calibracoes em {calib_dir}: nao da' para gerar vistas")
    if num_aux > len([n for n in available if n.startswith("aux")]):
        raise SystemExit(
            f"--views {args.views} pede {num_aux} auxiliares, mas so' ha' "
            f"{[n for n in available if n.startswith('aux')]}")
    views, gt, info = build_synthetic_views(
        calib_dir, num_aux=num_aux, render_scale=args.render_scale,
        noise_px=args.noise_px, seed=args.seed, draw_images=not args.sem_imagem)
    if not info["all_projections_inside_image"]:
        print("AVISO: projecao CAIU FORA da imagem em alguma vista: "
              + str(info["projections_inside"]))
    adapter = _make_adapter(args)
    if args.mode == "triangulation" or args.train_views_only:
        return _run_and_report(args, adapter, views, gt, info)
    # Checa disponibilidade real do modelo (sem mascara).
    status = adapter.status(refresh=True)
    if args.random_weights:
        adapter.attach_random_model(
            num_views=int(args.views), device=args.device)
        print("[modo] modelo OFICIAL com pesos ALEATORIOS (validacao de encanamento)")
    elif status["model_ready"]:
        print(f"[modo] modelo OFICIAL com checkpoint: {status['checkpoint']}")
    else:
        print("[modo] sem checkpoint -> fallback de triangulacao")
        for reason in status["reasons"]:
            print("       motivo: " + reason)
    return _run_and_report(args, adapter, views, gt, info)


def _run_and_report(args: argparse.Namespace, adapter: HandMvNetAdapter,
                    views: List[HandMvNetView], gt: np.ndarray,
                    info: Dict[str, Any]) -> int:
    _print_header("HandMvNet -- smoke sintetico (sem hardware)")
    _print_kv([
        ("calibracao", info["calibration_dir"]),
        ("auxiliares", info["aux_indices"]),
        ("vistas", [(v.name, v.size) for v in views]),
        ("render_scale", info["render_scale"]),
        ("ruido landmarks", f"{info['noise_px']} px"),
        ("modo adaptador", adapter.mode),
    ])
    out = adapter.infer(views, hand=args.hand)
    metrics = _fmt_metrics(out, gt)
    metrics["model_only_plumbing"] = out.backend == "handmvnet_random_weights"
    _print_output(out, metrics)
    if args.save_frames:
        written = save_synthetic_frames(views, args.save_frames, info)
        print(f"\nframes salvos em: {args.save_frames} ({len(written)} arquivos)")
    if args.json:
        payload = {"views": [v.describe() for v in views],
                   "output": out.to_dict(), "metrics": metrics,
                   "info": {k: v for k, v in info.items()
                            if k not in ("gt_joints", "projections")}}
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


# ===========================================================================
# Subcomando: infer-npz (frames gravados, sem hardware)
# ===========================================================================
_NPZ_DOC = """\
Formato do .npz (frames sincronizados + landmarks MediaPipe, ja' espelhados):

  kinect_img : (H,W,3) uint8  imagem BGR do Kinect
  kinect_px  : (21,2) float   landmarks em pixels da imagem do Kinect
  aux0_img   : (H,W,3) uint8  imagem BGR da auxiliar 0  (opcional)
  aux0_px    : (21,2) float   landmarks em pixels da auxiliar 0
  aux1_img / aux1_px : idem para a auxiliar 1 (opcional)

As calibracoes vem de configs/calibracao (stereo_calibration*.npz); os pixels
das auxiliares precisam estar na convencao da calibracao (imagem espelhada)."""


def cmd_infer_npz(args: argparse.Namespace) -> int:
    data = np.load(args.npz)
    names = [name for name in ("kinect", "aux0", "aux1")
             if f"{name}_img" in data.files and f"{name}_px" in data.files]
    if "kinect" not in names:
        raise SystemExit("npz precisa de kinect_img/kinect_px; ver --help")
    folder = Path(args.calibration_dir)
    paths = calibration_paths(folder)
    views: List[HandMvNetView] = []
    for name in names:
        # A vista do Kinect usa o arquivo de PAR (kinect_intrinsics sozinho nao
        # tem a calibracao stereo); `aux0` e' o par Kinect+aux0 canonico.
        npz_path = (paths.get("aux0") or paths.get("kinect")) \
            if name == "kinect" else paths.get(name)
        if npz_path is None:
            raise SystemExit(f"sem calibracao para a vista {name} em {folder}")
        aux_index = int(name[3:]) if name.startswith("aux") else None
        views.append(HandMvNetView.from_calibration_npz(
            name, data[f"{name}_img"], data[f"{name}_px"], npz_path,
            is_kinect=name == "kinect", aux_index=aux_index))
    adapter = _make_adapter(args)
    status = adapter.status(refresh=True)
    if args.random_weights and status["checkpoint_present"] is False:
        adapter.attach_random_model(num_views=len(views), device=args.device)
    _print_header("HandMvNet -- inferencia em frames .npz (sem hardware)")
    out = adapter.infer(views, hand=args.hand)
    metrics = _fmt_metrics(out, None)
    metrics["model_only_plumbing"] = out.backend == "handmvnet_random_weights"
    _print_output(out, metrics)
    if args.json:
        print(json.dumps({"output": out.to_dict(), "metrics": metrics},
                         indent=2, ensure_ascii=False, default=str))
    return 0


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT),
                        help="clone do repo oficial (default: external/HandMvNet)")
    common.add_argument("--calibration-dir", default=str(DEFAULT_CALIBRATION_DIR),
                        help="pasta dos npz de calibracao do projeto")
    common.add_argument("--checkpoint", default=None,
                        help="path do .ckpt do HandMvNet (ou env HANDMVNET_CHECKPOINT)")
    common.add_argument("--device", default="auto",
                        choices=("auto", "cpu", "xpu", "cuda"),
                        help="device de inferencia (auto = cuda>xpu>cpu)")
    common.add_argument("--mode", default="auto",
                        choices=("auto", "model", "triangulation"),
                        help="auto = modelo se houver checkpoint, senao fallback")
    common.add_argument("--image-size", type=int, default=None,
                        help="lado do crop (default: o do checkpoint / 256)")
    common.add_argument("--crop-margin", type=float, default=1.2,
                        help="margem do bbox quadrado em volta dos landmarks")
    common.add_argument("--adapt-num-views", action="store_true",
                        help="EXPERIMENTAL: usar checkpoint de N vistas com outro N")
    common.add_argument("--no-shims", action="store_true",
                        help="nao instalar shims de import para o codigo oficial")
    common.add_argument("--json", action="store_true", help="saida JSON")
    common.add_argument("--hand", default="right", choices=("right", "left"))

    parser = argparse.ArgumentParser(
        description="HandMvNet: status e smoke test do adaptador multiview",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", parents=[common],
                              help="diagnostico: repo, deps, pesos, licenca")
    p_status.set_defaults(func=cmd_status)

    p_syn = sub.add_parser("synthetic", parents=[common],
                           help="vistas sinteticas calibradas (sem hardware)")
    p_syn.add_argument("--views", type=int, default=3, choices=(2, 3),
                       help="2 = Kinect+aux0; 3 = Kinect+aux0+aux1")
    p_syn.add_argument("--noise-px", type=float, default=0.5,
                       help="ruido gaussiano nos landmarks (px)")
    p_syn.add_argument("--render-scale", type=float, default=1.0,
                       help="escala das imagens sinteticas (1.0 = resolucao calibrada)")
    p_syn.add_argument("--seed", type=int, default=0)
    p_syn.add_argument("--sem-imagem", action="store_true",
                       help="nao desenhar o esqueleto nas imagens sinteticas")
    p_syn.add_argument("--random-weights", action="store_true",
                       help="modelo OFICIAL com pesos ALEATORIOS (so' encanamento)")
    p_syn.add_argument("--triangulation", dest="train_views_only",
                       action="store_true",
                       help="forca o fallback de triangulacao")
    p_syn.add_argument("--save-frames", default=None,
                       help="diretorio para salvar PNGs das vistas (debug)")
    p_syn.set_defaults(func=cmd_synthetic)

    p_npz = sub.add_parser("infer-npz", parents=[common],
                           help="inferencia em frames .npz gravados")
    p_npz.epilog = _NPZ_DOC
    p_npz.add_argument("npz", help="arquivo .npz sincronizado (ver --help)")
    p_npz.add_argument("--random-weights", action="store_true",
                       help="modelo OFICIAL com pesos ALEATORIOS (so' encanamento)")
    p_npz.set_defaults(func=cmd_infer_npz)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:                                    # acentos no terminal UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                       # pragma: no cover
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        args.train_views_only = False          # compat com _make_adapter
        args.random_weights = False
    try:
        return int(args.func(args) or 0)
    except SystemExit:
        raise
    except KeyboardInterrupt:                  # pragma: no cover
        print("\ninterrompido")
        return 130
    except Exception as exc:                   # pragma: no cover
        print(f"FALHA: {type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "json", False):
            print(json.dumps({"erro": f"{type(exc).__name__}: {exc}"},
                             ensure_ascii=False), file=sys.stderr)
        else:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
