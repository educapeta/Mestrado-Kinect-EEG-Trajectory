"""Analise estatistica dos artefatos do projeto (nivel de acaso, PCC, overlay).

Por que existe (ver `docs/ESTADO_DA_ARTE_ESTATISTICA.md`): antes de escrever
"PCC = 0,71" ou "o desvio medio foi 22 px" num artigo e' preciso dizer se aquilo
difere do acaso/do baseline E com que incerteza. Este CLI aplica a bateria de
`estat_bci.py` aos arquivos que o projeto ja' produz -- sem inventar formato:

  --acaso 15/20            critico EXATO do binomio para um resultado medido
  --acaso-tabela 10 20 100 tabela do critico para varios numeros de trials
  --overlay-log CSV        desvio do overlay por no + BH-FDR + Kinect vendo x cego
  --pcc-tabela CSV         {sujeito,metodo,pcc} -> Friedman + Wilcoxon/Holm
  --pcc-npz NPZ            previsao/alvo (N,T,3): PCC, IC, permutacao, baseline
  --auto-teste             gera dados sinteticos e roda TUDO (valida o CLI)

Os modos de dados aceitam `--json CAMINHO` para gravar o resultado completo (o
numero que vai para a dissertacao fica rastreavel).

Exemplos:
    .venv\\Scripts\\python.exe tools\\analise_estatistica.py --acaso 15/20
    .venv\\Scripts\\python.exe tools\\analise_estatistica.py --overlay-log
    .venv\\Scripts\\python.exe tools\\analise_estatistica.py --pcc-npz previsoes.npz --json pcc.json
    .venv\\Scripts\\python.exe tools\\analise_estatistica.py --auto-teste
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import estat_bci as eb  # noqa: E402

RAIZ = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OVERLAY_PADRAO = RAIZ / "overlay_deviation_log.csv"
#: Coluna do log do overlay que diz a ORIGEM da profundidade usada naquele frame
#: ("Kinect" = o proprio Kinect viu a mao; "esqueletoSDK"/outros = o Kinect
#: estava cego e a profundidade veio de reserva -- e' o contraste que interessa).
COLUNA_FONTE = "note"
N_NOS = 21


def _flutuante(texto, padrao=float("nan")):
    """float() tolerante para CSV: vazio/ilegivel vira NaN (nunca 0)."""
    try:
        return float(texto)
    except (TypeError, ValueError):
        return padrao


def formata_ic(media, lo, hi, casas=3):
    """'0,712 [0,655; 0,769]' -- o formato exigido nos relatorios do projeto."""
    if not np.isfinite(media):
        return "nan"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{media:.{casas}f} [sem IC]"
    return (f"{media:.{casas}f} [{lo:.{casas}f}; {hi:.{casas}f}]")


# ============================================================ modo: acaso
def modo_acaso(texto, n_classes, alfa):
    if "/" in texto:
        acertos, n_trials = (int(parte) for parte in texto.split("/", 1))
        info = eb.acerto_minimo(n_trials, alfa, n_classes)
        p_valor = eb.p_valor_acerto(acertos, n_trials, n_classes)
        poder = eb.poder_binomial(n_trials, acertos / max(n_trials, 1),
                                  alfa, n_classes)
        print(f"resultado medido: {acertos}/{n_trials} = "
              f"{100.0 * acertos / max(n_trials, 1):.1f}%")
        if info["acertos_criticos"] is None:
            print(f"  ACASO: {info['nota']}")
        else:
            print(f"  critico exato ({n_classes} classes, alfa={alfa:g}): "
                  f"{info['acertos_criticos']}/{n_trials} = "
                  f"{info['acerto_critico_pct']:.1f}%")
            print(f"  aproximacao normal (conferencia): "
                  f"{100.0 * eb.acaso_aproximado(n_trials, alfa, n_classes):.1f}%")
        print(f"  p exato = {p_valor:.4f} -> "
              f"{'ACIMA do acaso' if p_valor <= alfa else 'NAO difere do acaso'}")
        print(f"  poder do teste para essa acuracia = {poder:.2f}")
        return {"acertos": acertos, "n_trials": n_trials, "p": p_valor,
                "critico": info["acertos_criticos"], "poder": poder}
    n_trials = int(texto)
    info = eb.acerto_minimo(n_trials, alfa, n_classes)
    print(f"N = {n_trials} trials, {n_classes} classes, alfa = {alfa:g}")
    print(f"  critico exato: {info['acerto_critico_pct']:.1f}% "
          f"({info['acertos_criticos']} acertos) | p no ponto "
          f"{info['p_no_critico']:.4f}")
    print(f"  aproximacao normal: "
          f"{100.0 * eb.acaso_aproximado(n_trials, alfa, n_classes):.1f}%")
    for acerto in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        print(f"    poder se o acerto verdadeiro for {acerto:.0%}: "
              f"{eb.poder_binomial(n_trials, acerto, alfa, n_classes):.2f}")
    return {k: v for k, v in info.items() if k != "nota"}


def modo_acaso_tabela(ns, n_classes, alfa):
    print(f"Critico EXATO do binomio ({n_classes} classes, alfa={alfa:g}) -- "
          "e' ESTE numero que vale, nao o acaso teorico:")
    print("   N trials | acertos | acerto minimo | p no ponto | aprox. normal")
    linhas = []
    for n_trials in ns:
        info = eb.acerto_minimo(n_trials, alfa, n_classes)
        if info["acertos_criticos"] is None:
            continue
        linhas.append({"n_trials": n_trials,
                       "acertos": info["acertos_criticos"],
                       "acerto_minimo_pct": info["acerto_critico_pct"],
                       "p_no_critico": info["p_no_critico"],
                       "aproximado_pct": 100.0 * eb.acaso_aproximado(
                           n_trials, alfa, n_classes)})
        print(f"   {n_trials:8d} | {info['acertos_criticos']:7d} | "
              f"{info['acerto_critico_pct']:12.1f}% | "
              f"{info['p_no_critico']:10.4f} | "
              f"{linhas[-1]['aproximado_pct']:12.1f}%")
    return {"n_classes": n_classes, "alfa": alfa, "tabela": linhas}


# ====================================================== modo: log do overlay
def le_overlay(caminho):
    """Le o log do overlay: lista de linhas (dict) + colunas por no presentes."""
    with open(caminho, "r", encoding="utf-8", newline="") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    if not linhas:
        raise RuntimeError(
            f"{caminho} tem SO' o cabecalho (nenhum frame gravado). Ligue o log "
            "no tools/kinect_groundtruth_tool.py com a tecla L, mova a mao com "
            "o Kinect vendo E depois com ele cego, e rode de novo.")
    return linhas


def modo_overlay(caminho, alfa, n_perm, rng):
    """Desvio do overlay por no: mediana, IC, teste vs 0 e BH-FDR na familia."""
    linhas = le_overlay(caminho)
    print(f"log do overlay: {caminho.name} | {len(linhas)} frames")
    fontes = {}
    for linha in linhas:
        fonte = (linha.get(COLUNA_FONTE) or "").strip() or "(vazio)"
        fontes[fonte] = fontes.get(fonte, 0) + 1
    print("  fonte de profundidade por frame: " + ", ".join(
        f"{chave}={valor}" for chave, valor in sorted(fontes.items())))

    resultado = {"arquivo": caminho.name, "n_frames": len(linhas),
                 "fontes": fontes, "nos": []}
    p_valores = []
    for indice in range(N_NOS):
        valores = np.array([_flutuante(linha.get(f"n{indice}_dev_px"))
                            for linha in linhas], np.float64)
        dx = np.array([_flutuante(linha.get(f"n{indice}_dx"))
                       for linha in linhas], np.float64)
        dy = np.array([_flutuante(linha.get(f"n{indice}_dy"))
                       for linha in linhas], np.float64)
        finitos = valores[np.isfinite(valores)]
        registro = {"no": indice, "n": int(finitos.size)}
        resultado["nos"].append(registro)
        if finitos.size < 5:
            p_valores.append(float("nan"))
            continue
        ic = eb.ic_media_bootstrap(finitos, n_boot=2000, rng=rng)
        teste = eb.comparacao_pareada(finitos, np.zeros(finitos.size),
                                      n_perm=2000, rng=rng)
        p_valores.append(teste["p_perm"])
        registro.update({
            "mediana_px": float(np.median(finitos)),
            "p90_px": float(np.percentile(finitos, 90)),
            "media_px": float(finitos.mean()),
            "media_ic": [float(ic["lo"][0]), float(ic["hi"][0])],
            "dx_mediano": float(np.nanmedian(dx)),
            "dy_mediano": float(np.nanmedian(dy)),
            "p_vs_zero": float(teste["p_perm"]),
            "d_z_vs_zero": float(teste["d_z"])})
    ajuste = eb.bh_fdr(p_valores, alfa=alfa)
    print(f"  desvio por no (familia de {ajuste['n_testes']} nos, BH-FDR "
          f"alfa={alfa:g} -> {ajuste['n_rejeitados']} rejeitados; * = "
          "significativo apos correcao):")
    print("    no |    n | mediana |  p90 | media [IC 95%]        | dx med | "
          "dy med |  p(BH)")
    for posicao, registro in enumerate(resultado["nos"]):
        if "mediana_px" not in registro:
            print(f"   {registro['no']:2d} | {registro['n']:4d} | (poucos "
                  "frames validos)")
            continue
        p_ajustado = float(ajuste["p_ajustado"][posicao])
        registro["p_ajustado"] = p_ajustado
        registro["significante"] = bool(ajuste["rejeita"][posicao])
        marca = "*" if registro["significante"] else " "
        print(f"   {registro['no']:2d} | {registro['n']:4d} | "
              f"{registro['mediana_px']:7.1f} | {registro['p90_px']:4.0f} | "
              f"{formata_ic(registro['media_px'], *registro['media_ic'], 1):<22}| "
              f"{registro['dx_mediano']:+6.1f} | {registro['dy_mediano']:+6.1f} "
              f"| {p_ajustado:7.4f} {marca}")

    # --- Kinect VENDO a mao x Kinect CEGO (fonte de profundidade do frame) ----
    # E' o contraste que responde "a reserva (esqueletoSDK/par de auxiliares)
    # piora o overlay?". O n de frames NAO e' n independente (frames vizinhos
    # repetem a mesma mao): o p-valor e' indicio, nao prova -- por isso o texto
    # diz isso em voz alta todas as vezes.
    por_fonte = {}
    for linha in linhas:
        fonte = (linha.get(COLUNA_FONTE) or "").strip() or "(vazio)"
        por_fonte.setdefault(fonte, []).append(_flutuante(linha.get("n9_dev_px")))
    if len(por_fonte) >= 2:
        chaves = sorted(por_fonte, key=lambda c: -len(por_fonte[c]))
        print("  palma (no 9) por fonte de profundidade:")
        resultado["palma_por_fonte"] = {}
        for chave in chaves:
            dados = np.asarray(por_fonte[chave], np.float64)
            dados = dados[np.isfinite(dados)]
            if dados.size == 0:
                continue
            ic = eb.ic_media_bootstrap(dados, n_boot=2000, rng=rng)
            resultado["palma_por_fonte"][chave] = {
                "n": int(dados.size), "mediana": float(np.median(dados)),
                "media": float(dados.mean()),
                "ic": [float(ic["lo"][0]), float(ic["hi"][0])]}
            print(f"    {chave:<16} n={dados.size:4d} | mediana "
                  f"{np.median(dados):6.1f} px | media "
                  f"{formata_ic(dados.mean(), ic['lo'][0], ic['hi'][0], 1)}")
        a = np.asarray(por_fonte[chaves[0]], np.float64)
        b = np.asarray(por_fonte[chaves[1]], np.float64)
        comparacao = eb.comparar_grupos(a, b, pareado=False, n_perm=n_perm,
                                        rng=rng)
        resultado["comparacao_fontes"] = {
            "a": chaves[0], "b": chaves[1],
            "diferenca_media": comparacao["diferenca_media"],
            "p_perm": comparacao["p_perm"], "d": comparacao["d"],
            "n_a": comparacao["n_a"], "n_b": comparacao["n_b"]}
        print(f"    {chaves[0]} x {chaves[1]}: diferenca "
              f"{comparacao['diferenca_media']:+.1f} px | p(perm)="
              f"{comparacao['p_perm']:.4f} | d={comparacao['d']:+.2f} "
              f"(n {comparacao['n_a']} vs {comparacao['n_b']})")
        print("    ATENCAO: frames vizinhos nao sao independentes; use como "
              "indicio, nao como prova.")
    else:
        print("  (uma unica fonte de profundidade no log: sem contraste "
              "Kinect vendo x cego)")
    return resultado


# ============================================== modo: previsoes de trajetoria
def modo_pcc_npz(caminho, alfa, n_perm, rng):
    """Analise honesta do decodificador de trajetoria (PCC por eixo).

    Contrato do npz (o que `sand_traj_treino.py` deve exportar na validacao):
        previsao  (N, T, 3)  trajetoria prevista, em metros, no referencial do
                             Kinect (eixo 0 = X, 1 = Y, 2 = Z)
        alvo      (N, T, 3)  trajetoria medida (ground truth)
        bloco     (N,)       OPCIONAL, id da sessao de cada trial (para a
                             permutacao dentro do bloco; sem ele a permutacao e'
                             global)
    Aceita tambem as chaves em ingles (pred/target/block).
    """
    dados = np.load(caminho)
    previsao = dados.get("previsao", dados.get("pred"))
    alvo = dados.get("alvo", dados.get("target"))
    bloco = dados.get("bloco", dados.get("block"))
    if previsao is None or alvo is None:
        raise RuntimeError(f"{caminho} precisa das chaves 'previsao' e 'alvo'")
    previsao = np.asarray(previsao, np.float64)
    alvo = np.asarray(alvo, np.float64)
    n_trials = previsao.shape[0]
    print(f"previsoes: {caminho.name} | {n_trials} trials x "
          f"{previsao.shape[1]} pontos x {previsao.shape[2]} eixos"
          + ("" if bloco is None else f" | {len(np.unique(bloco))} blocos"))
    por_trial = eb.pcc_por_trial(previsao, alvo)
    pendente = eb.teste_permutacao_pcc(previsao, alvo, n_perm=n_perm,
                                       bloco=bloco, rng=rng)
    baseline = eb.pcc_alvo_medio(alvo)
    nomes = ("x", "y", "z")
    resultado = {"arquivo": caminho.name, "n_trials": int(n_trials),
                 "n_pontos": int(previsao.shape[1]),
                 "n_perm": int(n_perm), "eixos": {}}
    print("  eixo | PCC modelo [IC 95%] (bootstrap)      | PCC baseline "
          "| p(permutacao) | p(BH)")
    p_eixos = np.asarray(pendente["p_por_eixo"], np.float64)
    ajuste = eb.holm(p_eixos, alfa=alfa)
    for eixo in range(por_trial.shape[1]):
        modelo = por_trial[:, eixo]
        base_eixo = baseline[:, eixo]
        ic = eb.ic_media_bootstrap(modelo, n_boot=5000, rng=rng)
        comparado = eb.comparacao_pareada(modelo, base_eixo, n_perm=2000,
                                          rng=rng)
        nome = nomes[eixo] if eixo < len(nomes) else str(eixo)
        resultado["eixos"][nome] = {
            "pcc": float(ic["media"][0]), "ic": [float(ic["lo"][0]),
                                                 float(ic["hi"][0])],
            "pcc_baseline_alvo_medio": float(np.nanmean(base_eixo)),
            "p_vs_baseline": float(comparado["p_wilcoxon"]),
            "d_z_vs_baseline": float(comparado["d_z"]),
            "p_permutacao": float(p_eixos[eixo]),
            "p_permutacao_holm": float(ajuste["p_ajustado"][eixo]),
            "ganho_sobre_baseline": float(ic["media"][0]
                                          - np.nanmean(base_eixo))}
        print(f"    {nome}  | {formata_ic(ic['media'][0], ic['lo'][0], ic['hi'][0]):<36}"
              f"| {np.nanmean(base_eixo):11.3f} | {p_eixos[eixo]:13.5f} | "
              f"{ajuste['p_ajustado'][eixo]:.5f}")
    print(f"  agregado: PCC medio {pendente['observado_medio']:.3f} | nulo "
          f"(pareamento embaralhado) media "
          f"{float(np.nanmean(pendente['nulo_media'])):.3f} | p = "
          f"{pendente['p_agregado']:.5f}")
    print("  LEITURA: 'p(permutacao)' testa se o modelo usa informacao DO TRIAL "
          "(sem isso, acertar a trajetoria media da condicao ja' da' PCC alto) "
          "e 'baseline' mostra quanto desse PCC e' so' a media.")
    resultado["agregado"] = {
        "pcc_medio": pendente["observado_medio"],
        "nulo_medio": float(np.nanmean(pendente["nulo_media"])),
        "p_agregado": pendente["p_agregado"]}
    return resultado


# ======================================= modo: tabela de PCC por sujeito/metodo
def modo_pcc_tabela(caminho, alfa, rng):
    """CSV longo {sujeito,metodo,pcc} -> Friedman + Wilcoxon/Holm por par."""
    por_metodo = {}
    with open(caminho, "r", encoding="utf-8", newline="") as arquivo:
        for linha in csv.DictReader(arquivo):
            metodo = (linha.get("metodo") or "").strip()
            valor = _flutuante(linha.get("pcc"))
            if metodo and np.isfinite(valor):
                por_metodo.setdefault(metodo, []).append(valor)
    if len(por_metodo) < 2:
        raise RuntimeError("a tabela precisa de >= 2 metodos com PCC valido")
    print(f"tabela de PCC: {caminho.name} | "
          f"{len(por_metodo)} metodos | "
          f"{max(len(v) for v in por_metodo.values())} unidades (max)")
    for metodo, valores in sorted(por_metodo.items()):
        dados = np.asarray(valores, np.float64)
        ic = eb.ic_media_bootstrap(dados, n_boot=5000, rng=rng)
        print(f"  {metodo:<24} n={dados.size:3d} | PCC "
              f"{formata_ic(dados.mean(), ic['lo'][0], ic['hi'][0])}")
    resultado = eb.comparar_metodos(por_metodo, alfa=alfa, rng=rng)
    if "friedman" in resultado:
        print(f"  Friedman: chi2={resultado['friedman']['estatistica']:.2f}, "
              f"gl={resultado['friedman']['gl']}, "
              f"p={resultado['friedman']['p']:.5f}")
    print("  vitorias por unidade: " + ", ".join(
        f"{nome}={total}" for nome, total in sorted(resultado["vitorias"].items())))
    print("  comparacoes pareadas (Holm):")
    for par in resultado["pares"]:
        marca = "*" if par["significante"] else " "
        print(f"    {par['a']} x {par['b']}: p={par['p']:.5f} | "
              f"p(ajustado)={par['p_ajustado']:.5f} {marca} | d_z={par['d_z']:+.2f}")
    return resultado


# =============================================================== auto-teste
def modo_auto_teste(alfa, n_perm, n_classes, rng, pasta):
    """Gera dados sinteticos, roda TODOS os modos e confere o resultado.

    Smoke test do CLI (tambem chamavel pelo `tools\\roda_testes.py`): nenhuma
    funcao abaixo depende de arquivo do projeto, e as assercoes sao o contrato de
    cada modo (diferenca plantada = detectada; nada plantado = nao detectado).
    """
    pasta.mkdir(exist_ok=True)
    campos = ["t_s", "lado", "z_usado_m", "z_bruto_m", "cx_px", "cy_px",
              "borda_0c_1b", "vel_px_s", COLUNA_FONTE]
    for indice in range(N_NOS):
        campos += [f"n{indice}_dev_px", f"n{indice}_dx", f"n{indice}_dy"]
    caminho_overlay = pasta / "_auto_overlay.csv"
    with open(caminho_overlay, "w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        for frame in range(120):
            # 80 frames com o Kinect VENDO a mao (desvio ~20 px) e 40 com o
            # Kinect CEGO (reserva, desvio ~30 px): diferenca plantada.
            cego = frame >= 80
            centro = 30.0 if cego else 20.0
            linha = {"t_s": f"{frame / 30.0:.4f}", "lado": "right",
                     "z_usado_m": "1.00", "z_bruto_m": "1.02",
                     "cx_px": "960", "cy_px": "540", "borda_0c_1b": "0",
                     "vel_px_s": "50",
                     COLUNA_FONTE: "esqueletoSDK" if cego else "Kinect"}
            for indice in range(N_NOS):
                escala = 1.0 + 0.05 * indice
                linha[f"n{indice}_dev_px"] = (
                    f"{max(0.0, rng.normal(centro * escala, 4.0)):.3f}")
                linha[f"n{indice}_dx"] = f"{rng.normal(2.0, 1.0):.3f}"
                linha[f"n{indice}_dy"] = f"{rng.normal(-1.5, 1.0):.3f}"
            escritor.writerow(linha)

    caminho_pcc = pasta / "_auto_pcc.npz"
    n_trials, n_pontos = 30, 30
    t = np.linspace(0.0, 1.0, n_pontos)
    alvo = np.array([np.stack([np.sin((3.0 + 0.6 * i) * t),
                               np.cos((2.0 + 0.5 * i) * t),
                               0.5 * np.sin((1.5 + 0.4 * i) * t)], axis=1)
                     for i in range(n_trials)])
    previsao = alvo + rng.normal(scale=0.02, size=alvo.shape)
    np.savez(caminho_pcc, previsao=previsao, alvo=alvo,
             bloco=np.repeat(np.arange(3), n_trials // 3))

    caminho_tabela = pasta / "_auto_tabela.csv"
    with open(caminho_tabela, "w", encoding="utf-8", newline="") as arquivo:
        arquivo.write("sujeito,metodo,pcc\n")
        for sujeito in range(6):
            base = 0.55 + 0.02 * sujeito
            for metodo, ganho in (("A", 0.10), ("B", 0.03), ("C", 0.00)):
                arquivo.write(f"S{sujeito + 1},{metodo},"
                              f"{base + ganho + rng.normal(scale=0.005):.4f}\n")

    print("### 1/4 nivel de acaso (tabela exata)")
    modo_acaso_tabela([10, 20, 40, 100], n_classes, alfa)
    print("\n### 2/4 overlay sintetico (Kinect vendo x cego)")
    overlay = modo_overlay(caminho_overlay, alfa, n_perm, rng)
    comparacao = overlay["comparacao_fontes"]
    # 'a' = a fonte mais FREQUENTE (o Kinect vendo a mao) e 'b' = a reserva:
    # a diferenca sai NEGATIVA porque quem esta' CEGO tem desvio MAIOR.
    assert comparacao["p_perm"] < 0.05, comparacao
    assert comparacao["diferenca_media"] < -5.0, comparacao
    assert comparacao["d"] < -1.0, comparacao
    print("\n### 3/4 previsoes sinteticas (PCC decodificado)")
    pcc = modo_pcc_npz(caminho_pcc, alfa, n_perm, rng)
    assert pcc["agregado"]["p_agregado"] < 0.01, pcc["agregado"]
    for eixo, dados in pcc["eixos"].items():
        assert dados["pcc"] > 0.9, (eixo, dados)
        assert dados["ganho_sobre_baseline"] > 0.0, (eixo, dados)
    print("\n### 4/4 tabela de metodos sintetica (A > B > C)")
    tabela = modo_pcc_tabela(caminho_tabela, alfa, rng)
    assert tabela["friedman"]["p"] < 0.01, tabela["friedman"]
    assert tabela["vitorias"]["A"] == 6, tabela["vitorias"]
    print("\nAUTO_TESTE_OK: overlay (diferenca plantada detectada), PCC "
          "(permutacao significativa e acima do baseline), metodos (A vence em "
          "6/6 com Friedman significativo)")
    for caminho in (caminho_overlay, caminho_pcc, caminho_tabela):
        try:
            caminho.unlink()
        except OSError:
            pass
    try:
        pasta.rmdir()
    except OSError:
        pass
    return {"overlay": overlay, "pcc": pcc, "metodos": tabela}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--acaso", metavar="ACERTOS/N_TRIALS",
                        help="Ex.: 15/20 (resultado medido) ou so' o N, para "
                             "ver o critico exato e o poder.")
    parser.add_argument("--acaso-tabela", type=int, nargs="+", metavar="N",
                        help="Tabela do critico EXATO para varios N.")
    parser.add_argument("--overlay-log", nargs="?", const=str(OVERLAY_PADRAO),
                        metavar="CSV",
                        help="Log do overlay (padrao: overlay_deviation_log.csv)")
    parser.add_argument("--pcc-npz", metavar="NPZ",
                        help="npz com previsao (N,T,3), alvo (N,T,3) e bloco "
                             "opcional (N,)")
    parser.add_argument("--pcc-tabela", metavar="CSV",
                        help="CSV longo com sujeito,metodo,pcc")
    parser.add_argument("--auto-teste", action="store_true",
                        help="Roda tudo com dados sinteticos (smoke test)")
    parser.add_argument("--classes", type=int, default=2,
                        help="Numero de classes no nivel de acaso")
    parser.add_argument("--alfa", type=float, default=eb.ALFA)
    parser.add_argument("--n-perm", type=int, default=5000,
                        help="Permutacoes das distribuicoes nulas")
    parser.add_argument("--semente", type=int, default=20260923,
                        help="Semente do gerador (reprodutibilidade)")
    parser.add_argument("--json", metavar="CAMINHO",
                        help="Grava o resultado completo em JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.semente)
    resultados = {}
    try:
        if args.acaso:
            resultados["acaso"] = modo_acaso(args.acaso, args.classes, args.alfa)
        if args.acaso_tabela:
            resultados["acaso_tabela"] = modo_acaso_tabela(
                args.acaso_tabela, args.classes, args.alfa)
        if args.overlay_log:
            caminho = Path(args.overlay_log)
            if not caminho.exists():
                print(f"ERRO: {caminho} nao existe (rode o tool com a tecla L "
                      "ligada para gerar o log).")
                return 2
            resultados["overlay"] = modo_overlay(caminho, args.alfa,
                                                 args.n_perm, rng)
        if args.pcc_npz:
            resultados["pcc"] = modo_pcc_npz(Path(args.pcc_npz), args.alfa,
                                             args.n_perm, rng)
        if args.pcc_tabela:
            resultados["pcc_tabela"] = modo_pcc_tabela(Path(args.pcc_tabela),
                                                       args.alfa, rng)
        if args.auto_teste:
            resultados["auto_teste"] = modo_auto_teste(
                args.alfa, args.n_perm, args.classes, rng, RAIZ / "_auto_estat")
        if not resultados:
            print("Nada para fazer: use --acaso, --acaso-tabela, --overlay-log, "
                  "--pcc-npz, --pcc-tabela ou --auto-teste (veja --help).")
            return 1
    except (RuntimeError, ValueError, OSError) as erro:
        print(f"\nERRO: {erro}")
        return 2
    if args.json:
        with open(args.json, "w", encoding="utf-8") as arquivo:
            json.dump(resultados, arquivo, indent=2, ensure_ascii=False,
                      default=float)
        print(f"\nresultado salvo em {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
