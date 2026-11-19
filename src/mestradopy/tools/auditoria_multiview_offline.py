#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auditoria OFFLINE do multi-view (Kinect depth x camera auxiliar), sem hardware.

Motivo (18/11/2026): a metrica `depth 3D na AUXILIAR` do
`tools/compara_tracking_mao.py` estava em 75-88 px nas sessoes de 12/11 e foi
lida como "calibracao errada". A analise dos CSVs gravados mostrou que metade do
erro e' de FRAME: o 3D do depth do Kinect sai do `MapDepthPointToCameraSpace`
(frame da CAMERA DE PROFUNDIDADE), mas a extrinsica das auxiliares foi calibrada
no frame da CAMERA DE COR. Este tool reprojeta o 3D gravado na auxiliar com:

  * frame do depth (como o tool antigo media) -> numero inflado;
  * frame da cor (extrinsica depth->color ajustada por PnP nos proprios dados)
    -> numero honesto.

Tambem mede: erro se a imagem da auxiliar estivesse espelhada (teste de
convencao), sensibilidade a Z (px por cm de depth), erro por landmark, frames
parados x em movimento e a reprojecao do multi-view (se estiver no CSV).

Entradas: os CSVs longos/por-frame do `compara_tracking_mao.py`
(`<prefixo>_landmarks.csv`, `<prefixo>_frames.csv`) e o `.npz` de calibracao
stereo da auxiliar.

Exemplos:
    # auditoria da sessao de 12/11 (aux0) com a calibracao de entao
    .venv\\Scripts\\python.exe src\\mestradopy\\tools\\auditoria_multiview_offline.py ^
        --sessao-dir "C:\\...\\Tracking\\COMPARACAO_20261112\\AUX0" --prefixo aux0 ^
        --calib configs\\calibracao\\stereo_calibration.npz

    # comparar duas calibracoes na mesma sessao
    ... --calib configs\\calibracao\\stereo_calibration.npz ^
        --calib-referencia antes\\stereo_calibration.npz --json auditoria.json

    # ajustar e salvar a extrinsica depth->color do Kinect
    ... --salvar-frame-cor configs\\calibracao\\kinect_depth_para_cor.npz

Saida: relatorio Markdown no console (ou --relatorio), JSON opcional.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]          # ...\C:\mestradopy
CONFIG_CALIB = RAIZ / "configs" / "calibracao"


# --------------------------------------------------------------------------
# Calibracao / geometria
# --------------------------------------------------------------------------
def carrega_calib(caminho):
    """Le o npz stereo de UMA auxiliar (schema do stereo_calibration.py)."""
    d = np.load(caminho, allow_pickle=True)
    obrig = {"kinect_matrix", "kinect_dist", "auxiliary_matrix", "auxiliary_dist",
             "rotation", "translation"}
    faltando = obrig - set(d.files)
    if faltando:
        raise SystemExit(f"{caminho}: faltam chaves {sorted(faltando)}")
    return {
        "arquivo": str(caminho),
        "K_k": np.asarray(d["kinect_matrix"], np.float64).reshape(3, 3),
        "dist_k": np.asarray(d["kinect_dist"], np.float64).reshape(-1),
        "K_a": np.asarray(d["auxiliary_matrix"], np.float64).reshape(3, 3),
        "dist_a": np.asarray(d["auxiliary_dist"], np.float64).reshape(-1),
        "R": np.asarray(d["rotation"], np.float64).reshape(3, 3),
        "T": np.asarray(d["translation"], np.float64).reshape(3),
        "rms": float(d["rms"]) if "rms" in d.files else float("nan"),
        "size_a": tuple(int(v) for v in d["auxiliary_size"])
        if "auxiliary_size" in d.files else (1280, 720),
        "rms_kin": float(d["kinect_rms"]) if "kinect_rms" in d.files else float("nan"),
        "rms_aux": float(d["auxiliary_rms"]) if "auxiliary_rms" in d.files else float("nan"),
        "dev_px": float(d["kinect_deviation_px"])
        if "kinect_deviation_px" in d.files else None,
        "per_pose_rms": np.asarray(d["per_pose_rms"], np.float64)
        if "per_pose_rms" in d.files else None,
        "camera": str(d["camera_name"]) if "camera_name" in d.files else "",
    }


def projeta(xyz_cam, K, dist):
    """Projeta (N,3) em pixels com cv2.projectPoints (frame da propria camera)."""
    import cv2
    xyz = np.asarray(xyz_cam, np.float64).reshape(-1, 3)
    px, _ = cv2.projectPoints(xyz.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                              K, dist)
    return px.reshape(-1, 2)


def projeta_aux(xyz_kin_frame_cor, cal):
    """X_aux = R X_kin + T -> pixels na auxiliar."""
    xyz = np.asarray(xyz_kin_frame_cor, np.float64).reshape(-1, 3)
    xa = xyz @ cal["R"].T + cal["T"].reshape(1, 3)
    return projeta(xa, cal["K_a"], cal["dist_a"])


def ajusta_depth_para_cor(xyz_depth, kin_px, cal):
    """Extrinsica rigida frame-depth -> frame-cor (X_cor = R X_depth + t).

    Usa os pares (3D do depth, pixel da cor do MESMO landmark) gravados pelo
    proprio tool. Devolve (R, t, residuo_px) -- residuo = reprojecao do 3D
    corrigido no RGB do Kinect, que valida o ajuste (o que importa e' ~1-2 px).
    """
    import cv2
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(xyz_depth, np.float64).reshape(-1, 1, 3),
        np.asarray(kin_px, np.float64).reshape(-1, 1, 2),
        cal["K_k"], cal["dist_k"],
        flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=3.0,
        iterationsCount=500, confidence=0.999)
    R, _ = cv2.Rodrigues(rvec)
    residuo = np.linalg.norm(
        projeta(np.asarray(xyz_depth) @ R.T + tvec.reshape(1, 3),
                cal["K_k"], cal["dist_k"]) - np.asarray(kin_px), axis=1)
    return R, tvec.reshape(3), residuo


def carrega_frame_cor(caminho):
    d = np.load(caminho, allow_pickle=True)
    return {"R": np.asarray(d["rotation"], np.float64).reshape(3, 3),
            "t": np.asarray(d["translation"], np.float64).reshape(3),
            "residuo_px": float(d["residuo_px"]) if "residuo_px" in d.files
            else float("nan"),
            "arquivo": str(caminho)}


def salva_frame_cor(R, t, residuo_px, caminho, n):
    np.savez(caminho, rotation=R, translation=t.reshape(3, 1),
             residuo_px=float(residuo_px), n_amostras=int(n),
             metodo="solvePnP (pixels de cor do Kinect x 3D do depth, "
                    "gravados por compara_tracking_mao)",
             data="18/11/2026")


# --------------------------------------------------------------------------
# Leitura dos CSVs
# --------------------------------------------------------------------------
def le_landmarks(caminho):
    amostras = []
    with open(caminho, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("k_valid") != "1":
                continue
            if not r.get("kin_px_x") or not r.get("aux_px_x"):
                continue
            try:
                amostras.append({
                    "seq": int(r["seq"]),
                    "hand": r.get("hand", ""),
                    "lm": int(r.get("landmark_id", len(amostras) % 21)),
                    "xyz_depth": (float(r["k_x_m"]), float(r["k_y_m"]),
                                  float(r["k_z_m"])),
                    "kin_px": (float(r["kin_px_x"]), float(r["kin_px_y"])),
                    "aux_px": (float(r["aux_px_x"]), float(r["aux_px_y"])),
                    "kd_gravado": (float(r["kd_aux_err_px"])
                                   if r.get("kd_aux_err_px") else None),
                    "mv_reproj_aux": (float(r["mv_reproj_aux_px"])
                                      if r.get("mv_reproj_aux_px") else None),
                    "mv_valid": int(r.get("mv_valid") or 0),
                    "mv_implausivel": int(r.get("mv_implausivel") or 0),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return amostras


def le_frames(caminho):
    frames = {}
    if not Path(caminho).exists():
        return frames
    with open(caminho, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                frames[int(r["seq"])] = {
                    "palm": (float(r["palm9_k_x"]), float(r["palm9_k_y"]),
                             float(r["palm9_k_z"])) if r.get("palm9_k_x") else None,
                    "dt": float(r["dt_s"]) if r.get("dt_s") else None,
                    "mv_triangulou": int(r.get("mv_triangulou") or 0),
                    "mv_valid": int(r.get("mv_valid") or 0),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return frames


def velocidade_por_frame(frames):
    seqs = sorted(frames)
    vel = {}
    for a, b in zip(seqs, seqs[1:]):
        pa, pb = frames[a]["palm"], frames[b]["palm"]
        dt = frames[b]["dt"]
        if pa is None or pb is None or not dt or dt <= 0:
            continue
        d = float(np.linalg.norm(np.subtract(pb, pa)))
        if d < 0.5:                      # rejeita salto de identidade/oclusao
            vel[b] = d / dt
    return vel


# --------------------------------------------------------------------------
# Estatistica
# --------------------------------------------------------------------------
def st(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "med": float("nan"), "p95": float("nan")}
    return {"n": int(v.size), "med": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "media": float(np.mean(v))}


def sensibilidade_z(cal, zs=(0.6, 0.8, 1.0, 1.2, 1.4)):
    """px de deslocamento na auxiliar por 1 cm de Z (no Kinect)."""
    out = {}
    for z in zs:
        px = projeta_aux(np.array([[0.0, 0.0, z], [0.0, 0.0, z + 0.01]]), cal)
        out[z] = float(np.linalg.norm(px[1] - px[0]))
    return out


# --------------------------------------------------------------------------
# Auditoria
# --------------------------------------------------------------------------
def audita(sessao_dir, prefixo, cal, cal_ref=None, frame_cor=None,
           salvar_frame_cor=None):
    sessao = Path(sessao_dir)
    amostras = le_landmarks(sessao / f"{prefixo}_landmarks.csv")
    frames = le_frames(sessao / f"{prefixo}_frames.csv")
    vel = velocidade_por_frame(frames)
    if not amostras:
        raise SystemExit(f"nenhuma amostra valida em {sessao}")

    Xd = np.array([a["xyz_depth"] for a in amostras], np.float64)
    aux_px = np.array([a["aux_px"] for a in amostras], np.float64)
    kin_px = np.array([a["kin_px"] for a in amostras], np.float64)
    largura = cal["size_a"][0]

    # extrinsica depth->color: usa a salva, senao ajusta nos proprios dados
    if frame_cor is not None:
        Rdc, tdc, res_pnp = frame_cor["R"], frame_cor["t"], frame_cor["residuo_px"]
        origem_cor = f"config {frame_cor['arquivo']}"
    else:
        Rdc, tdc, res_pnp = ajusta_depth_para_cor(Xd, kin_px, cal)
        origem_cor = "PnP nos dados desta sessao"
    Xc = Xd @ Rdc.T + tdc.reshape(1, 3)
    if salvar_frame_cor:
        salva_frame_cor(Rdc, tdc, float(np.median(res_pnp)),
                        Path(salvar_frame_cor), len(amostras))

    # erro depth3D -> auxiliar
    erros = {}
    for nome, XYZ in (("frame_depth", Xd), ("frame_cor", Xc)):
        px = projeta_aux(XYZ, cal)
        err = np.linalg.norm(px - aux_px, axis=1)
        bias = np.nanmedian(px - aux_px, axis=0)
        # espelhamento: o mesmo erro se o pixel da aux estivesse espelhado
        aux_m = aux_px.copy()
        aux_m[:, 0] = (largura - 1) - aux_m[:, 0]
        err_m = np.linalg.norm(px - aux_m, axis=1)
        erros[nome] = {
            "erro": st(err), "bias_dx_dy": [float(bias[0]), float(bias[1])],
            "erro_se_aux_espelhada": st(err_m),
            "por_landmark": {lm: st(err[np.array([a["lm"] for a in amostras]) == lm])
                             .get("med") for lm in range(21)},
        }
        # parado x em movimento
        parado, mov = [], []
        for a, e in zip(amostras, err):
            v = vel.get(a["seq"])
            if v is None:
                continue
            (parado if v < 0.03 else mov).append(e)
        erros[nome]["parado_lt_0.03_mps"] = st(parado)
        erros[nome]["movimento"] = st(mov)
        # por faixa de Z
        zs = Xd[:, 2]
        faixas = {}
        for lo, hi in ((0.3, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.6), (1.6, 4.0)):
            m = (zs >= lo) & (zs < hi)
            if m.sum() >= 20:
                faixas[f"{lo}-{hi}"] = st(err[m])
        erros[nome]["por_z"] = faixas

    # calibracao de referencia (opcional) no frame da cor
    ref = None
    if cal_ref is not None:
        px = projeta_aux(Xc, cal_ref)
        ref = {"arquivo": cal_ref["arquivo"],
               "erro": st(np.linalg.norm(px - aux_px, axis=1))}

    # gravado pelo tool antigo (frame depth) -- valida o replay
    gravado = [a["kd_gravado"] for a in amostras if a["kd_gravado"] is not None]
    gravado_st = st(gravado)

    # multi-view: reprojecao e cobertura (se o CSV tiver)
    mv_reproj = [a["mv_reproj_aux"] for a in amostras
                 if a["mv_reproj_aux"] is not None]
    mv_valid = [a["mv_valid"] for a in amostras]
    mv_impl = [a["mv_implausivel"] for a in amostras]
    mv = {
        "reproj_aux_px": st(mv_reproj) if mv_reproj else {"n": 0},
        "cobertura_mv_pct": 100.0 * float(np.mean(mv_valid)) if mv_valid else None,
        "implausivel_pct": 100.0 * float(np.mean(mv_impl)) if mv_impl else None,
    }
    if frames:
        tri = [f["mv_triangulou"] for f in frames.values()]
        val = [f["mv_valid"] for f in frames.values()]
        mv["cobertura_frames_pct"] = 100.0 * float(np.mean(val)) if val else None
        mv["triangulou_frames_pct"] = 100.0 * float(np.mean(tri)) if tri else None

    return {
        "sessao": str(sessao), "prefixo": prefixo,
        "calibracao": {k: cal[k] for k in ("arquivo", "rms", "rms_kin", "rms_aux",
                                           "dev_px", "camera", "size_a")},
        "frame_depth_para_cor": {
            "origem": origem_cor, "residuo_pnp_px": st(res_pnp),
            "t_m": [float(x) for x in tdc],
            "norma_t_m": float(np.linalg.norm(tdc)),
        },
        "n_amostras": len(amostras),
        "n_frames": len(frames),
        "erro_depth3d_auxiliar": erros,
        "kd_aux_gravado_pelo_tool": gravado_st,
        "multiview": mv,
        "calibracao_referencia": ref,
        "sensibilidade_z_px_por_cm": sensibilidade_z(cal),
    }


def formata_md(r):
    e = r["erro_depth3d_auxiliar"]
    fd, fc = e["frame_depth"], e["frame_cor"]
    l = []
    l.append(f"# Auditoria offline multi-view — `{r['prefixo']}`")
    l.append("")
    l.append(f"- sessão: `{r['sessao']}`")
    l.append(f"- calibração: `{r['calibracao']['arquivo']}` "
             f"(RMS {r['calibracao']['rms']:.2f} px; câmera "
             f"`{r['calibracao']['camera']}`)")
    l.append(f"- amostras válidas: {r['n_amostras']} | frames: {r['n_frames']}")
    l.append("")
    l.append("## `depth 3D na AUXILIAR` (o que a calibração e o frame fazem)")
    l.append("")
    l.append("| reprojeção do 3D do depth na auxiliar | mediana (px) | p95 (px) | viés dx,dy (px) |")
    l.append("|---|---:|---:|---:|")
    l.append(f"| frame da PROFUNDIDADE (metrica antiga) | {fd['erro']['med']:.1f} | "
             f"{fd['erro']['p95']:.1f} | ({fd['bias_dx_dy'][0]:.1f}, {fd['bias_dx_dy'][1]:.1f}) |")
    l.append(f"| frame da COR (corrigido por PnP) | {fc['erro']['med']:.1f} | "
             f"{fc['erro']['p95']:.1f} | ({fc['bias_dx_dy'][0]:.1f}, {fc['bias_dx_dy'][1]:.1f}) |")
    l.append("")
    l.append(f"- PnP depth→cor: resíduo mediano "
             f"{r['frame_depth_para_cor']['residuo_pnp_px']['med']:.2f} px; "
             f"|t| = {r['frame_depth_para_cor']['norma_t_m']:.3f} m "
             f"({r['frame_depth_para_cor']['origem']}).")
    l.append(f"- teste de espelhamento (erro se a auxiliar estivesse espelhada): "
             f"frame depth {fd['erro_se_aux_espelhada']['med']:.1f} px vs "
             f"{fd['erro']['med']:.1f} px normal -> "
             + ("**convenção de espelhamento ERRADA**"
                if fd["erro_se_aux_espelhada"]["med"] < fd["erro"]["med"]
                else "convenção de espelhamento OK"))
    l.append("")
    l.append("| condição | frame depth (mediana px) | frame cor (mediana px) |")
    l.append("|---|---:|---:|")
    l.append(f"| parado (<0,03 m/s) | {fd['parado_lt_0.03_mps']['med']:.1f} "
             f"(n={fd['parado_lt_0.03_mps']['n']}) | {fc['parado_lt_0.03_mps']['med']:.1f} "
             f"(n={fc['parado_lt_0.03_mps']['n']}) |")
    l.append(f"| movimento | {fd['movimento']['med']:.1f} (n={fd['movimento']['n']}) | "
             f"{fc['movimento']['med']:.1f} (n={fc['movimento']['n']}) |")
    l.append("")
    l.append("| faixa de Z (m) | frame depth (px) | frame cor (px) | n |")
    l.append("|---|---:|---:|---:|")
    for faixa in fd["por_z"]:
        l.append(f"| {faixa} | {fd['por_z'][faixa]['med']:.1f} | "
                 f"{fc['por_z'][faixa]['med']:.1f} | {fd['por_z'][faixa]['n']} |")
    l.append("")
    l.append("## Multi-view (reprojeção na auxiliar / cobertura)")
    l.append("")
    mv = r["multiview"]
    l.append(f"- reproj MV na auxiliar: mediana {mv['reproj_aux_px'].get('med', float('nan')):.2f} px, "
             f"p95 {mv['reproj_aux_px'].get('p95', float('nan')):.2f} px")
    l.append(f"- cobertura MV por landmark: {mv.get('cobertura_mv_pct')}% | "
             f"implausível: {mv.get('implausivel_pct')}% | "
             f"frames com MV válido: {mv.get('cobertura_frames_pct')}%")
    l.append("")
    l.append("## Sensibilidade a Z (px de deslocamento na auxiliar por 1 cm de depth)")
    l.append("")
    l.append("| Z (m) | px/cm |")
    l.append("|---|---:|")
    for z, v in r["sensibilidade_z_px_por_cm"].items():
        l.append(f"| {z} | {v:.1f} |")
    l.append("")
    if r.get("calibracao_referencia"):
        ref = r["calibracao_referencia"]
        l.append("## Calibração de referência (opcional)")
        l.append("")
        l.append(f"- `{ref['arquivo']}`: mediana {ref['erro']['med']:.1f} px, "
                 f"p95 {ref['erro']['p95']:.1f} px")
        l.append("")
    l.append("## Erro por landmark (frame da cor, mediana px)")
    l.append("")
    porm = fc["por_landmark"]
    l.append("| " + " | ".join(str(i) for i in range(21)) + " |")
    l.append("|" + "---|" * 21)
    l.append("| " + " | ".join(
        "--" if porm.get(i) is None or not np.isfinite(porm.get(i, np.nan))
        else f"{porm[i]:.1f}" for i in range(21)) + " |")
    l.append("")
    l.append(f"> `kd_aux` gravado pelo tool antigo (frame depth): mediana "
             f"{r['kd_aux_gravado_pelo_tool']['med']:.1f} px, "
             f"p95 {r['kd_aux_gravado_pelo_tool']['p95']:.1f} px — confere com a "
             "coluna frame da profundidade (replay validado).")
    return "\n".join(l)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessao-dir", required=True,
                    help="pasta com <prefixo>_landmarks.csv e <prefixo>_frames.csv")
    ap.add_argument("--prefixo", default="aux0")
    ap.add_argument("--calib", default=str(CONFIG_CALIB / "stereo_calibration.npz"))
    ap.add_argument("--calib-referencia", default=None,
                    help="outra calibracao para comparar na mesma sessao")
    ap.add_argument("--frame-cor", default=None,
                    help="npz com rotation/translation depth->color; se omitido, "
                         "ajusta por PnP nos dados")
    ap.add_argument("--salvar-frame-cor", default=None,
                    help="salva o npz depth->color ajustado neste caminho")
    ap.add_argument("--json", default=None)
    ap.add_argument("--relatorio", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cal = carrega_calib(Path(args.calib))
    cal_ref = carrega_calib(Path(args.calib_referencia)) if args.calib_referencia else None
    frame_cor = carrega_frame_cor(Path(args.frame_cor)) if args.frame_cor else None
    r = audita(args.sessao_dir, args.prefixo, cal, cal_ref, frame_cor,
               salvar_frame_cor=args.salvar_frame_cor)
    texto = formata_md(r)
    print(texto)
    if args.relatorio:
        Path(args.relatorio).write_text(texto + "\n", encoding="utf-8")
        print(f"\nRelatorio salvo em {args.relatorio}")
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"JSON salvo em {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
