"""Roda todas as suites `test_*.py` e resume o resultado (sem hardware).

Uso:
    .venv\\Scripts\\python.exe tools\\roda_testes.py
    .venv\\Scripts\\python.exe tools\\roda_testes.py --detalhado

Saida: uma linha por suite (OK/FALHA e o resumo final de cada uma) e o total de
falhas. Codigo de saida 0 = todas passaram.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def interpretador():
    """Prefere o Python do .venv do projeto (mesmas dependencias dos testes)."""
    candidato = os.path.join(RAIZ, ".venv", "Scripts", "python.exe")
    return candidato if os.path.exists(candidato) else sys.executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detalhado", action="store_true",
                        help="Mostra as ultimas linhas das suites que falharem")
    args = parser.parse_args()
    python = interpretador()
    suites = sorted(glob.glob(os.path.join(RAIZ, "test_*.py")))
    if not suites:
        print("nenhuma suite encontrada")
        return 1
    print(f"{len(suites)} suites (interpretador: {python})")
    falhas = []
    for caminho in suites:
        nome = os.path.basename(caminho)
        try:
            processo = subprocess.run([python, nome], cwd=RAIZ,
                                      capture_output=True, text=True,
                                      timeout=900)
        except subprocess.TimeoutExpired:
            falhas.append(nome)
            print(f"  TIMEOUT {nome}")
            continue
        linhas = [linha.strip() for linha in
                  (processo.stdout + processo.stderr).splitlines() if linha.strip()]
        resumo = next((linha for linha in reversed(linhas)
                       if "_OK" in linha or "Error" in linha
                       or "assert" in linha), "")
        if processo.returncode == 0:
            print(f"  OK    {nome} | {resumo[:118]}")
        else:
            falhas.append(nome)
            print(f"  FALHA {nome} | {resumo[:118]}")
            if args.detalhado:
                for linha in linhas[-6:]:
                    print(f"         {linha[:118]}")
    print(f"\nsuites: {len(suites)} | falhas: {len(falhas)}"
          + (f" -> {', '.join(falhas)}" if falhas else " -> todas passaram"))
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())