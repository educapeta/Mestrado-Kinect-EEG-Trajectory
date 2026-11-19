#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checagem de higiene da raiz do projeto.

A raiz deve conter APENAS os itens estruturais do projeto. Arquivos de saida
(PNG, CSV, MP4, NPZ de diagnostico, logs) vao para `outputs/`, `archive/` ou
`docs/`, nunca para `C:\\mestradopy`.

Uso:
    .\.venv\Scripts\python.exe tests\manual\test_estrutura_raiz.py
"""
from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]

ARQUIVOS_PERMITIDOS = {
    ".gitignore",
    "HISTORICO_ITERACOES.txt",   # atalho de compatibilidade (hardlink)
    "README.md",
    "pyproject.toml",
    "requirements.txt",
}

DIRETORIOS_PERMITIDOS = {
    ".git",
    ".venv",
    ".vscode",
    ".pytest_cache",
    ".idea",
    "archive",
    "assets",
    "configs",
    "data",
    "docs",
    "external",
    "firmware",
    "outputs",
    "scripts",
    "src",
    "tests",
}


def violacoes(raiz: Path = RAIZ) -> list[str]:
    erros: list[str] = []
    for item in sorted(raiz.iterdir(), key=lambda p: p.name.lower()):
        nome = item.name
        if item.is_file():
            if nome not in ARQUIVOS_PERMITIDOS:
                erros.append(
                    f"arquivo solto na raiz: {nome!r} "
                    f"({item.stat().st_size} bytes) — mova para outputs/, "
                    "archive/ ou docs/"
                )
        elif item.is_dir():
            if nome not in DIRETORIOS_PERMITIDOS:
                erros.append(
                    f"diretório inesperado na raiz: {nome!r} — use a "
                    "estrutura src/, scripts/, configs/, outputs/ etc."
                )
        else:
            erros.append(f"item especial na raiz: {nome!r}")
    return erros


def main() -> int:
    erros = violacoes()
    if erros:
        print("RAIZ_SUJA:")
        for erro in erros:
            print(" -", erro)
        return 1
    print(f"RAIZ_OK: {RAIZ} contém apenas itens estruturais.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
