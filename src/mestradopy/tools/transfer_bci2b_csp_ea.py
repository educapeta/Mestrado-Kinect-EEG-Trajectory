#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Transferência BCIC IV-2b -> piloto próprio com CSP+LDA e Euclidean Alignment.

Protocolo (sem vazamento):
1. BCIC IV-2b (9 sujeitos x 3 sessões T): CSP+LDA within-subject, deixando uma
   sessão de fora; média/desvio por sujeito.
2. Alvo (nossos dados): janelas de MI 2 s (0,5 s de passo), agrupadas por trial.
   - target_only: CSP+LDA treinado só no alvo, 5-fold agrupado por trial;
   - target_only_EA: idem, mas EA ajustado no treino de cada fold;
   - source_only: CSP+LDA treinado no 2b (EA por sujeito) e aplicado ao alvo
     (EA ajustado no alvo, sem labels);
   - source_csp_target_lda: filtros CSP do 2b + LDA re-treinado no treino alvo;
   - source_target_csp_lda: CSP+LDA treinado no 2b + treino alvo do fold.

Uso:
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\transfer_bci2b_csp_ea.py
    .\\.venv\\Scripts\\python.exe src\\mestradopy\\tools\\transfer_bci2b_csp_ea.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from mne.decoding import CSP

mne.set_log_level("ERROR")
_SRC = Path(__file__).resolve().parents[1]
for _p in (_SRC, _SRC / "eeg", _SRC / "core", _SRC / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from eeg.mi_eegnet.data import carrega_own  # noqa: E402

CH_2B = ["EEG:C3", "EEG:Cz", "EEG:C4"]
FS = 128
WIN = 2.0
STEP = 0.5
BAND = (8.0, 30.0)


def _finite_np(z):
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _pipeline(n_csp=2):
    import mne
    mne.set_log_level("ERROR")
    return Pipeline([
        ("csp", CSP(n_components=n_csp, reg="ledoit_wolf", log=True)),
        ("finite", FunctionTransformer(_finite_np)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def carrega_bci2b_sessoes(pasta: Path) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
    """BCIC 2b T: X (n,3,256), y, sujeito, sessão. Uma janela/trial."""
    xs, ys, subj, sess = [], [], [], []
    for f in sorted(Path(pasta).glob("*T.gdf")):
        nome = f.stem  # B0101T
        try:
            s = int(nome[1:3]); sessao = int(nome[3:5])
        except Exception:
            continue
        raw = mne.io.read_raw_gdf(str(f), preload=True, verbose=False)
        raw.filter(*BAND, verbose=False)
        raw.resample(FS, verbose=False)
        picks = [c for c in CH_2B if c in raw.ch_names]
        if len(picks) != 3:
            print(f"[aviso] {f.name}: canais {raw.ch_names}; ignorado")
            continue
        raw.pick(picks)
        data = raw.get_data()
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        mapa = {}
        for cod, nome_cls in ((769, 0), (770, 1)):
            if str(cod) in event_id:
                mapa[int(event_id[str(cod)])] = nome_cls
        ev = events[np.isin(events[:, 2], list(mapa))]
        n = int(WIN * FS)
        for amostra, _, cod in ev:
            start = int(amostra + 1.0 * FS)
            end = start + n
            if start < 0 or end > data.shape[1]:
                continue
            xs.append(data[:, start:end].astype(np.float32))
            ys.append(mapa[int(cod)])
            subj.append(s); sess.append(sessao)
    if not xs:
        raise RuntimeError(f"nenhuma sessao T carregada de {pasta}")
    return (np.stack(xs), np.array(ys, int), np.array(subj, int),
            np.array(sess, int))


def carrega_alvo(sessoes: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, g = [], [], []
    offset = 0
    for pasta in sessoes:
        Xi, yi, gi, _ = carrega_own(pasta, fs=FS, fase="MI")
        X.append(Xi); y.append(yi); g.append(gi + offset)
        offset += int(gi.max()) + 1
    return np.concatenate(X), np.concatenate(y), np.concatenate(g)


def ea_fit(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Euclidean Alignment: R = média das covariâncias; W = R^{-1/2}."""
    X = np.asarray(X, np.float64)
    cov = np.einsum("nct,ndt->ncd", X, X) / max(1, X.shape[-1])
    R = cov.mean(axis=0)
    R = R + eps * np.eye(R.shape[0])
    eig, V = np.linalg.eigh(R)
    eig = np.clip(eig, eps, None)
    return (V * (1.0 / np.sqrt(eig))) @ V.T


def ea_apply(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.einsum("cd,ndt->nct", W, np.asarray(X, np.float64))


def agrega_trial(prob: np.ndarray, y: np.ndarray, groups: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    yt, yp = [], []
    for g in np.unique(groups):
        m = groups == g
        yt.append(int(y[m][0]))
        yp.append(int(np.mean(prob[m], axis=0).argmax()))
    return np.array(yt), np.array(yp)


def metrica(y, pred) -> Dict[str, object]:
    ba = float(balanced_accuracy_score(y, pred)) if len(y) else float("nan")
    return {"n": int(len(y)), "ba": round(ba, 4),
            "confusao": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def fonte_validacao(Xb, yb, subj, sess) -> Dict[str, object]:
    """CSP+LDA within-subject: treina 2 sessões T, testa a terceira."""
    out = []
    for s in sorted(set(subj.tolist())):
        ss = sorted(set(sess[subj == s].tolist()))
        for teste in ss:
            tr = (subj == s) & (sess != teste)
            te = (subj == s) & (sess == teste)
            if tr.sum() < 20 or te.sum() < 10:
                continue
            pipe = _pipeline(); pipe.fit(Xb[tr], yb[tr])
            pred = pipe.predict(Xb[te])
            out.append({"sujeito": int(s), "teste": int(teste),
                        **metrica(yb[te], pred)})
    bas = [x["ba"] for x in out if np.isfinite(x["ba"])]
    return {"folds": out, "ba_media": round(float(np.mean(bas)), 4),
            "ba_dp": round(float(np.std(bas)), 4), "n_folds": len(bas)}


def fonte_ea(Xb, yb, subj, sess) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica EA por sujeito/sessão e devolve fonte pronta para CSP global."""
    Xa = np.empty_like(Xb)
    for s in sorted(set(subj.tolist())):
        for se in sorted(set(sess[subj == s].tolist())):
            m = (subj == s) & (sess == se)
            W = ea_fit(Xb[m])
            Xa[m] = ea_apply(Xb[m], W)
    return Xa, yb


def avalia_alvo_cv(X, y, g, *, com_ea: bool) -> Dict[str, object]:
    """5-fold agrupado por trial; CSP+LDA treinado só no alvo."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    yt, yp = [], []
    for tr, te in sgkf.split(X, y, groups=g):
        if com_ea:
            W = ea_fit(X[tr])
            Xtr = ea_apply(X[tr], W); Xte = ea_apply(X[te], W)
        else:
            Xtr, Xte = X[tr], X[te]
        pipe = _pipeline(); pipe.fit(Xtr, y[tr])
        prob = pipe.predict_proba(Xte)
        a, b = agrega_trial(prob, y[te], g[te])
        yt.extend(a); yp.extend(b)
    return metrica(np.array(yt), np.array(yp))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bci", type=Path,
                    default=Path("data/raw/Datasets/BCICIV_2b_gdf"))
    ap.add_argument("--alvo", type=Path, nargs="+",
                    default=[Path("outputs/eeg/piloto_me_mi_s01"),
                             Path("outputs/eeg/piloto_me_mi_s02")])
    ap.add_argument("--saida", type=Path,
                    default=Path("outputs/eeg/transfer_bci2b_ea"))
    ap.add_argument("--smoke", action="store_true",
                    help="carrega só 2 sujeitos e 1 sessão alvo (teste rápido)")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    print("Carregando BCIC IV-2b T...")
    Xb, yb, subj, sess = carrega_bci2b_sessoes(args.bci)
    if args.smoke:
        keep = np.isin(subj, [1, 2])
        Xb, yb, subj, sess = Xb[keep], yb[keep], subj[keep], sess[keep]
        args.alvo = args.alvo[:1]
    print("BCIC 2b:", Xb.shape, "classes:", dict(zip(*np.unique(yb, return_counts=True))))

    print("Carregando alvo...")
    Xt, yt, gt = carrega_alvo(args.alvo)
    print("alvo:", Xt.shape, "classes:", dict(zip(*np.unique(yt, return_counts=True))),
          "trials:", len(np.unique(gt)))

    print("Validando CSP+LDA no 2b (within-subject, 3 sessoes)...")
    val_bci = fonte_validacao(Xb, yb, subj, sess)
    print(f"  BA media={val_bci['ba_media']:.3f} dp={val_bci['ba_dp']:.3f} "
          f"folds={val_bci['n_folds']}")

    print("Alvo: CSP+LDA só no alvo (5-fold por trial)...")
    res = {}
    res["target_only"] = avalia_alvo_cv(Xt, yt, gt, com_ea=False)
    res["target_only_EA"] = avalia_alvo_cv(Xt, yt, gt, com_ea=True)

    print("Fonte EA + CSP+LDA global...")
    Xb_ea, _ = fonte_ea(Xb, yb, subj, sess)
    pipe_fonte = _pipeline(); pipe_fonte.fit(Xb_ea, yb)
    Wt = ea_fit(Xt)
    Xt_ea = ea_apply(Xt, Wt)
    prob = pipe_fonte.predict_proba(Xt_ea)
    a, b = agrega_trial(prob, yt, gt)
    res["source_only_EA"] = metrica(a, b)

    print("Fonte CSP -> LDA re-treinado no alvo (5-fold)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    yt_all, yp_all = [], []
    for tr, te in sgkf.split(Xt_ea, yt, groups=gt):
        csp = pipe_fonte.named_steps["csp"]
        Ftr = csp.transform(Xt_ea[tr]); Fte = csp.transform(Xt_ea[te])
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(Ftr, yt[tr])
        la, lb = agrega_trial(lda.predict_proba(Fte), yt[te], gt[te])
        yt_all.extend(la); yp_all.extend(lb)
    res["source_csp_target_lda"] = metrica(np.array(yt_all), np.array(yp_all))

    print("Fonte + alvo: CSP+LDA combinado (5-fold)...")
    yt_all, yp_all = [], []
    for tr, te in sgkf.split(Xt_ea, yt, groups=gt):
        Xcomb = np.concatenate([Xb_ea, Xt_ea[tr]])
        ycomb = np.concatenate([yb, yt[tr]])
        pipe = _pipeline(); pipe.fit(Xcomb, ycomb)
        la, lb = agrega_trial(pipe.predict_proba(Xt_ea[te]), yt[te], gt[te])
        yt_all.extend(la); yp_all.extend(lb)
    res["source_target_csp_lda"] = metrica(np.array(yt_all), np.array(yp_all))

    out = {"bci_validacao": val_bci, "alvo_n": {"janelas": int(len(yt)),
            "trials": int(len(np.unique(gt))),
            "classes": {str(k): int(v) for k, v in zip(*np.unique(yt, return_counts=True))}},
           "resultados": res, "aviso": (
               "O alvo disponivel aqui tem 60 trials/sessao (300 janelas), "
               "nao 40 trials/200 janelas. 2 s, passo 0,5 s, C3/C4/Cz, 128 Hz.")}
    (args.saida / "resultado.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    linhas = ["# Transferencia BCIC IV-2b -> piloto (CSP+LDA/EA)", "",
              f"- BCIC: {Xb.shape[0]} janelas, {len(set(subj.tolist()))} sujeitos",
              f"- validacao 2b: BA={val_bci['ba_media']:.3f} "
              f"(dp {val_bci['ba_dp']:.3f}, {val_bci['n_folds']} folds)",
              f"- alvo: {len(yt)} janelas, {len(np.unique(gt))} trials",
              "", "## Resultados no alvo", "",
              "| metodo | BA | n trials | confusao |",
              "|---|---:|---:|---|"]
    for nome, m in res.items():
        linhas.append(f"| {nome} | {m['ba']:.3f} | {m['n']} | {m['confusao']} |")
    linhas += ["", "Ver `resultado.json` para detalhes e custos."]
    (args.saida / "relatorio.md").write_text("\n".join(linhas) + "\n",
                                             encoding="utf-8")
    print("\n" + "\n".join(linhas))
    print("salvo em", args.saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
