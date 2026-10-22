"""Analisa/refaz o alinhamento da palma usando o PLANO MEDIO DO DORSO.

POR QUE EXISTE (item 102): a verdade do Kinect usada ate' agora era a normal do
produto vetorial de 3 pontos (punho, MCP indicador, MCP minimo). O ruido de
profundidade de UM no' gira a normal inteira, e o vetor da palma pode sair
"apontando para lugares ruins" mesmo com T/f bons. Aqui a verdade e' o plano
medio ajustado por minimos quadrados nos nos do DORSO (punho + MCPs indicador,
medio, anelar e minimo) -- a normal desse plano.

O CSV de entrada e' o do diagnostico da tecla D (`palm_diag_*.csv`), que ja'
grava os 4 pontos 3D crus do frame da palma por linha (item 78).

O que o programa faz, por lado:
  1) reconstroi a normal do PLANO do dorso em cada linha e mede o RMS do plano
     (controle de qualidade: profundidade ruidosa/borda sao descartadas);
  2) compara a normal do plano com a normal de 3 pontos que o tool usou;
  3) refaz T/f (`solve_palm_alignment`) com a verdade NOVA e mostra o residuo;
  4) compara com o alinhamento SALVO (imu_palm_alignment_<lado>.json): erro por
     fase do CSV na verdade nova -- e' o que responde "a seta esta' boa?";
  5) `--salvar` grava o refit por plano (com backup .json.bak).

Uso:
    .venv\\Scripts\\python.exe tools\\calibra_palma_plano.py
    .venv\\Scripts\\python.exe tools\\calibra_palma_plano.py palm_diag_20261019_180702.csv
    .venv\\Scripts\\python.exe tools\\calibra_palma_plano.py <csv> --salvar
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from imu import angle_between, rotation_matrix, solve_palm_alignment  # noqa: E402
from kinect_imu_groundtruth import palm_normal_from_plane  # noqa: E402

LIMITE_DEG = 15.0
SETTLE_S = 1.5
ORDEM_FASES = ("PREPARO", "MESA", "REZANDO", "KINECT")
ARQUIVOS_ALINHAMENTO = {"right": "imu_palm_alignment_right.json",
                        "left": "imu_palm_alignment_left.json"}
#: Linhas minimas de dados para o CSV valer (a tecla D de um teste abortado
#: grava poucas; o analisador oficial usa o mesmo criterio).
MIN_LINHAS = 200


def mais_recente(pasta=RAIZ):
    """palm_diag_*.csv mais novo que tenha dados de verdade (nao abortado)."""
    candidatos = sorted(Path(pasta).glob("palm_diag_*.csv"),
                        key=lambda caminho: caminho.stat().st_mtime,
                        reverse=True)
    for caminho in candidatos:
        try:
            with open(caminho, encoding="utf-8") as fh:
                total = sum(1 for _ in fh) - 1
        except OSError:
            continue
        if total >= MIN_LINHAS:
            return caminho
    return None


def _numero(texto, padrao=float("nan")):
    try:
        return float(texto)
    except (TypeError, ValueError):
        return padrao


def carrega_linhas(caminho):
    """Lista de dicts com lado/fase/tempos/rotacao/pontos crus/normal antiga."""
    linhas = []
    with open(caminho, encoding="utf-8", newline="") as fh:
        for bruta in csv.DictReader(fh):
            lado = str(bruta.get("lado", "") or "").strip()
            if lado not in ("right", "left"):
                continue
            roll = _numero(bruta.get("roll_deg"))
            pitch = _numero(bruta.get("pitch_deg"))
            yaw = _numero(bruta.get("yaw_deg"))
            if not np.isfinite([roll, pitch, yaw]).all():
                continue
            pontos = {}
            for nome in ("wrist", "index", "middle", "pinky"):
                chave = f"{nome}_x"
                if chave not in bruta:
                    continue
                x = _numero(bruta.get(f"{nome}_x"))
                y = _numero(bruta.get(f"{nome}_y"))
                z = _numero(bruta.get(f"{nome}_z"))
                if np.isfinite([x, y, z]).all():
                    pontos[nome] = np.array([x, y, z], dtype=float)
            if len(pontos) < 4:
                continue
            normal_antiga = None
            nx = _numero(bruta.get("nx"))
            ny = _numero(bruta.get("ny"))
            nz = _numero(bruta.get("nz"))
            if np.isfinite([nx, ny, nz]).all():
                vetor = np.array([nx, ny, nz], dtype=float)
                norma = float(np.linalg.norm(vetor))
                if norma > 1e-9:
                    normal_antiga = vetor / norma
            linhas.append({
                "lado": lado,
                "fase": str(bruta.get("fase_nome", "") or "").strip(),
                "t_fase_s": _numero(bruta.get("t_fase_s")),
                "rotacao": rotation_matrix(roll, pitch, yaw),
                "normal_antiga": normal_antiga,
                "pontos": pontos,
            })
    return linhas


def _erros_por_fase(rotacoes, normais, fases, tempos, T, f_body,
                    settle=SETTLE_S):
    """{fase: (n, mediana_graus)} da predicao `T R f` contra `normais`."""
    por_fase = {}
    for rotacao, normal, fase, t_fase in zip(rotacoes, normais, fases, tempos):
        if t_fase is not None and np.isfinite(t_fase) and t_fase < settle:
            continue
        previsto = T @ rotacao @ f_body
        norma = float(np.linalg.norm(previsto))
        if norma < 1e-9:
            continue
        erro = angle_between(normal, previsto / norma)
        por_fase.setdefault(fase or "?", []).append(float(erro))
    return {fase: (len(valores), float(np.median(valores)))
            for fase, valores in por_fase.items()}


def _carrega_alinhamento_salvo(lado):
    caminho = RAIZ / ARQUIVOS_ALINHAMENTO[lado]
    if not caminho.exists():
        return None, caminho
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        rotacao = np.asarray(dados["rotation"], dtype=float).reshape(3, 3)
        corpo = np.asarray(dados["body"], dtype=float).reshape(3)
        corpo = corpo / max(1e-12, float(np.linalg.norm(corpo)))
        return (rotacao, corpo, dados.get("residual_deg")), caminho
    except (OSError, KeyError, TypeError, ValueError):
        return None, caminho


def _grava_alinhamento(lado, rotacao, corpo, residuo):
    """Grava o refit com backup do arquivo anterior (.json.bak)."""
    caminho = RAIZ / ARQUIVOS_ALINHAMENTO[lado]
    backup = caminho.with_suffix(".json.bak")
    anterior = None
    if caminho.exists():
        try:
            anterior = json.loads(caminho.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            anterior = None
    dados = {"rotation": np.asarray(rotacao).tolist(),
             "body": np.asarray(corpo).tolist(),
             "residual_deg": float(residuo)}
    try:
        caminho.write_text(json.dumps(dados, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"  {lado}: FALHOU gravar {caminho.name} ({exc})")
        return False
    if anterior is not None:
        try:
            backup.write_text(json.dumps(anterior, indent=2), encoding="utf-8")
        except OSError as exc:
            print(f"  {lado}: AVISO backup falhou ({exc})")
    return True


def analisa(caminho, salvar=False, settle=SETTLE_S, limite=LIMITE_DEG):
    """Relatorio + (opcional) refit salvo. Devolve 0 (sempre analisavel)."""
    linhas = carrega_linhas(caminho)
    print("=" * 78)
    print(f"PLANO MEDIO DO DORSO -- {Path(caminho).name} ({len(linhas)} linhas "
          f"com pontos crus)")
    print("=" * 78)
    if not linhas:
        print("  nenhuma linha com os 4 pontos crus: CSV antigo (sem colunas do "
              "item 78)? Nada a analisar.")
        return 0
    por_lado = {"right": [], "left": []}
    for linha in linhas:
        normal, rms = palm_normal_from_plane(linha["pontos"], linha["lado"])
        if normal is None:
            continue
        linha["normal_plano"] = normal
        linha["rms_m"] = rms
        por_lado[linha["lado"]].append(linha)

    resultados = {}
    for lado in ("right", "left"):
        amostras = por_lado[lado]
        print(f"\n--- {lado.upper()} ({len(amostras)} linhas com plano valido) ---")
        if len(amostras) < 6:
            print("  amostras insuficientes para refit -- nada a concluir")
            continue
        rotacoes = [a["rotacao"] for a in amostras]
        normais_plano = [a["normal_plano"] for a in amostras]
        normais_antigas = [a["normal_antiga"] for a in amostras
                           if a["normal_antiga"] is not None]
        diferencas = [angle_between(a["normal_plano"], a["normal_antiga"])
                      for a in amostras if a["normal_antiga"] is not None]
        rms = [a["rms_m"] for a in amostras]
        print(f"  RMS do plano: mediana {np.median(rms)*1000:.1f} mm | "
              f"p90 {np.percentile(rms, 90)*1000:.1f} mm")
        if diferencas:
            print(f"  normal do PLANO x normal de 3 PONTOS (a que o tool usava): "
                  f"mediana {np.median(diferencas):.1f} graus | "
                  f"p90 {np.percentile(diferencas, 90):.1f} graus")
        refit_plano = solve_palm_alignment(rotacoes, normais_plano)
        refit_antigo = (solve_palm_alignment(rotacoes, normais_antigas)
                        if len(normais_antigas) >= 6 else None)
        if refit_antigo is not None:
            print(f"  residuo do refit com a verdade ANTIGA (3 pontos): "
                  f"{refit_antigo[2]:.2f} graus")
        if refit_plano is None:
            print("  refit com a verdade do PLANO falhou (amostras degeneradas)")
            continue
        t_novo, f_novo, residuo_novo = refit_plano
        print(f"  residuo do refit com a verdade do PLANO: {residuo_novo:.2f} "
              f"graus" + (" (OK)" if residuo_novo <= limite else " (ALTO)"))

        rotacoes_amostras = [a["rotacao"] for a in amostras]
        fases = [a["fase"] for a in amostras]
        tempos = [a["t_fase_s"] for a in amostras]
        salvo, caminho_salvo = _carrega_alinhamento_salvo(lado)
        erros_salvo = {}
        if salvo is not None:
            erros_salvo = _erros_por_fase(rotacoes_amostras, normais_plano,
                                          fases, tempos, salvo[0], salvo[1],
                                          settle=settle)
        erros_refit = _erros_por_fase(rotacoes_amostras, normais_plano,
                                      fases, tempos, t_novo, f_novo,
                                      settle=settle)
        print(f"  erros por fase contra a verdade do PLANO (mediana):")
        print("    fase      n    SALVO    REFIT_PLANO")
        for fase in ORDEM_FASES:
            if fase not in erros_salvo and fase not in erros_refit:
                continue
            n_salvo, erro_salvo = erros_salvo.get(fase, (0, float("nan")))
            n_refit, erro_refit = erros_refit.get(fase, (0, float("nan")))
            print(f"    {fase:8s} {max(n_salvo, n_refit):3d}   "
                  f"{erro_salvo:6.1f}    {erro_refit:6.1f}")
        mediana_salvo = (float(np.median([v for _n, v in erros_salvo.values()]))
                         if erros_salvo else float("nan"))
        if salvo is not None:
            print(f"  alinhamento SALVO: {caminho_salvo.name} -> mediana "
                  f"{mediana_salvo:.1f} graus nas fases do CSV")
            if mediana_salvo > limite:
                print("  VEREDITO: a seta esta' ERRADA com a verdade do plano -- "
                      "no tool, aperte J (calibracao por plano do dorso) ou use "
                      "--salvar aqui")
            else:
                print("  VEREDITO: seta OK (<= limite) na verdade do plano")
        else:
            print(f"  sem {caminho_salvo.name}: nada carregado pelo tool")
        resultados[lado] = {"t_novo": t_novo, "f_novo": f_novo,
                            "residuo": residuo_novo}
        if salvar and salvo is not None:
            if _grava_alinhamento(lado, t_novo, f_novo, residuo_novo):
                print(f"  gravado: {ARQUIVOS_ALINHAMENTO[lado]} (anterior em "
                      ".json.bak)")

    print()
    if salvar:
        print("para o tool carregar agora: reinicie-o (ele le o JSON no inicio) "
              "e confira a seta com a tecla J/D")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default=None,
                        help="palm_diag_*.csv (padrao: o mais recente valido)")
    parser.add_argument("--salvar", action="store_true",
                        help="grava o refit por PLANO (backup .json.bak)")
    parser.add_argument("--settle", type=float, default=SETTLE_S,
                        help="s (padrao 1.5) descartados no inicio de cada fase")
    parser.add_argument("--limite", type=float, default=LIMITE_DEG,
                        help="graus (padrao 15) para o veredito")
    args = parser.parse_args(argv)
    caminho = Path(args.csv) if args.csv else mais_recente()
    if caminho is None:
        print("nenhum palm_diag_*.csv encontrado -- rode a tecla D no tool "
              "(21 s, maos sem encostar) e tente de novo")
        return 2
    return analisa(caminho, salvar=args.salvar, settle=args.settle,
                   limite=args.limite)


if __name__ == "__main__":
    sys.exit(main())
