"""Inventario do dataset WAY-EEG-GAL local (so' leitura, nao altera nada).

Uso:
    .venv\\Scripts\\python.exe tools\\inventario_way_eeg_gal.py [pasta]

Mostra:
  - sujeitos/series e tamanho em disco;
  - o cache gerado pelo codigo legado (SAND_cache: *.json + shapes dos *.dat);
  - a estrutura interna de um arquivo .mat (campos e shapes);
  - as colunas de cinematica do sensor P4 usadas pelo artigo (21, 25, 29) e
    seus intervalos, para decidir a normalizacao do alvo.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

RAIZ = (sys.argv[1] if len(sys.argv) > 1
        else os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "WAY-EEG-GAL"))

print(f"raiz: {RAIZ}")
pastas = sorted(
    [d for d in glob.glob(os.path.join(RAIZ, "P*")) if os.path.isdir(d)],
    key=lambda p: int(os.path.basename(p)[1:]))
total_arquivos, total_bytes = 0, 0
for pasta in pastas:
    arquivos = sorted(glob.glob(os.path.join(pasta, "*.mat")))
    total_arquivos += len(arquivos)
    total_bytes += sum(os.path.getsize(f) for f in arquivos)
    exemplo = os.path.basename(arquivos[0]) if arquivos else "-"
    print(f"  {os.path.basename(pasta):>4}: {len(arquivos):2d} arquivos | ex.: {exemplo}")
print(f"TOTAL: {len(pastas)} sujeitos, {total_arquivos} arquivos, "
      f"{total_bytes / 1e9:.2f} GB")

cache = os.path.join(RAIZ, "SAND_cache")
print("\n=== SAND_cache (gerado pelo codigo legado) ===")
if os.path.isdir(cache):
    for meta in sorted(glob.glob(os.path.join(cache, "*.json"))):
        with open(meta, encoding="utf-8") as handle:
            dados = json.load(handle)
        print(f"  {os.path.basename(meta)}:")
        print(f"    {json.dumps(dados, ensure_ascii=False)[:500]}")
    for nome in sorted(glob.glob(os.path.join(cache, "*.dat")))[:4]:
        print(f"  {os.path.basename(nome)}: {os.path.getsize(nome) / 1e6:.1f} MB")
else:
    print("  (nao existe)")

primeiro = None
for pasta in pastas:
    achados = sorted(glob.glob(os.path.join(pasta, "*.mat")))
    if achados:
        primeiro = achados[0]
        break
if primeiro:
    import scipy.io
    print(f"\n=== estrutura de {os.path.relpath(primeiro, RAIZ)} "
          f"({os.path.getsize(primeiro) / 1e6:.1f} MB) ===")
    dados = scipy.io.loadmat(primeiro, squeeze_me=False,
                             struct_as_record=False)
    for chave, valor in dados.items():
        if chave.startswith("__"):
            continue
        print(f"  {chave}: {type(valor).__name__} shape={np.shape(valor)} "
              f"dtype={getattr(valor, 'dtype', None)}")
    # Se houver estruturas MATLAB com campos, mostra os campos da 1a janela.
    for chave, valor in dados.items():
        if chave.startswith("__"):
            continue
        if isinstance(valor, np.ndarray) and valor.dtype == object \
                and valor.size:
            elemento = valor.reshape(-1)[0]
            campos = getattr(elemento, "_fieldnames", None)
            print(f"  -> {chave}: struct MATLAB; campos: {campos}")
            if not campos:
                continue
            for nome in campos:
                try:
                    campo = getattr(elemento, nome)
                except AttributeError:
                    continue
                if nome == "ColNames":
                    nomes = [str(linha[0]) if isinstance(linha, np.ndarray)
                             else str(linha) for linha in np.asarray(campo).reshape(-1, 1)]
                    print(f"       {chave}.{nome} ({len(nomes)} colunas):")
                    for posicao, rotulo in enumerate(nomes):
                        marca = "  <-- P4 (usado pelo artigo)" if posicao in (21, 25, 29) else ""
                        print(f"         [{posicao:2d}] {rotulo}{marca}")
                    continue
                print(f"       {chave}.{nome}: shape={np.shape(campo)} "
                      f"dtype={getattr(campo, 'dtype', None)}")
                if nome.lower() == "eeg":
                    eeg = np.asarray(campo, np.float64)
                    print(f"         eeg: canais={eeg.shape[0] if eeg.ndim > 1 else '-'} "
                          f"amostras={eeg.shape[1] if eeg.ndim > 1 else eeg.size} "
                          f"| min={np.nanmin(eeg):.4g} max={np.nanmax(eeg):.4g} "
                          f"media={np.nanmean(eeg):.4g}")
                if nome.lower() == "kin":
                    kin = np.asarray(campo, np.float64)
                    print(f"         kin: forma={kin.shape} | colunas "
                          f"21/25/29 (P4): "
                          f"min={np.round(np.nanmin(kin[:, [21, 25, 29]], axis=0), 2)} "
                          f"max={np.round(np.nanmax(kin[:, [21, 25, 29]], axis=0), 2)}")
                if nome.lower() in ("marker", "markers", "trigger", "mrk"):
                    marca = np.asarray(campo).reshape(-1)
                    print(f"         {nome}: {marca.shape} valores "
                          f"{np.unique(marca)[:12]}")