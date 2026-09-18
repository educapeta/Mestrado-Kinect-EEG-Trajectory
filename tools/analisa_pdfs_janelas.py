"""Extrai dos PDFs as frases sobre JANELA de analise, ALVO e PRE-PROCESSAMENTO.

Uso:
    .venv\\Scripts\\python.exe tools\\analisa_pdfs_janelas.py "SAND_Spectral"
    .venv\\Scripts\\python.exe tools\\analisa_pdfs_janelas.py fnbot-13-00094 E2T MTRT ...
    .venv\\Scripts\\python.exe tools\\analisa_pdfs_janelas.py --todos          (os 7 do estudo)
    .venv\\Scripts\\python.exe tools\\analisa_pdfs_janelas.py --texto-completo SAND

Sem argumentos, usa a lista padrao dos 7 artigos do projeto. Os PDFs sao
procurados em ATACHMENTS (pasta do Citavi). Nada e' modificado nos PDFs.
"""
from __future__ import annotations

import os
import re
import sys

try:                                     # PyMuPDF >= 1.24 usa o nome novo
    import pymupdf as fitz
except ImportError:                      # versoes antigas
    import fitz

# Console do Windows (cp1252) nao aceita ligaduras/unicode dos PDFs IEEE: forca
# UTF-8 com substituicao, senao o relatorio morre no meio da impressao.
for _fluxo in (sys.stdout, sys.stderr):
    if hasattr(_fluxo, "reconfigure"):
        try:
            _fluxo.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

# Expande ligaduras (fi, fl, ff) em vez de preserva-las: sem isso o texto sai
# com caracteres da area privada (ex.: 'de\uf020nition').
FLAGS = fitz.TEXTFLAGS_TEXT & ~fitz.TEXT_PRESERVE_LIGATURES

PRIVADOS = re.compile(r"[\ue000-\uf8ff\x00-\x08\x0b\x0c\x0e-\x1f]")

ATACHMENTS = (r"C:\Users\elgut\OneDrive\Desktop\Mestrado_Projeto\Citavi"
              r"\Mestrado_BCI\Citavi Attachments")

PADRAO = [
    "SAND_Spectral-Attention",
    "fnbot-13-00094",
    "E2T_EEG-to-Trajectory",
    "Reconstruction_of_Continuous_Hand_Grasp",
    "fnins-17-1086472",
    "MTRT_Motion_Trajectory",
    "s11571-025-10403-1",
]

CATEGORIAS = {
    "JANELA": [r"\bwindow", r"\bepoch", r"\bsegment", r"\bsliding",
               r"post-?stimulus", r"pre-?stimulus", r"\blatenc",
               r"time interval", r"time range", r"\b\d+(?:[.,]\d+)?\s*"
               r"(?:ms|milliseconds|s|sec|seconds)\b"],
    "ALVO": [r"\btrajector", r"\bkinematic", r"\bground[- ]truth", r"\bposition",
             r"\bcoordinate"],
    "PRE-PROC": [r"band-?pass", r"\bfilter", r"downsampl", r"resampl",
                 r"normaliz", r"artifact", r"\bICA\b", r"\bCAR\b",
                 r"common average"],
    "MOTOR": [r"readiness", r"movement onset", r"\bMRCP\b", r"\bmu\b",
              r"\bbeta\b", r"\bERD\b", r"\bERS\b", r"Bereitschaft"],
}


def frases(texto):
    plano = re.sub(r"\s+", " ", PRIVADOS.sub("", texto))
    return [f.strip() for f in re.split(r"(?<=[.!?;])\s+", plano) if len(f) > 30]


def texto_do_pdf(caminho):
    """Texto de todas as paginas (ligaduras expandidas, unicode limpo)."""
    documento = fitz.open(caminho)
    paginas = [documento[i].get_text("text", flags=FLAGS)
               for i in range(documento.page_count)]
    documento.close()
    return [PRIVADOS.sub("", pagina) for pagina in paginas]


def relatorio(caminho, limite_por_categoria=14, max_chars=420):
    paginas = texto_do_pdf(caminho)
    texto = "\n".join(paginas)
    cabeçalho = " ".join(frases("\n".join(paginas[:1]))[:2])
    print("=" * 78)
    print(f"ARQUIVO: {os.path.basename(caminho)} ({len(paginas)} paginas)")
    print(f"INICIO : {cabeçalho[:290]}")
    for nome, padroes in CATEGORIAS.items():
        compilados = [re.compile(p, re.I) for p in padroes]
        vistas, saida = set(), []
        for frase in frases(texto):
            if any(p.search(frase) for p in compilados):
                chave = frase[:80].lower()
                if chave in vistas:
                    continue
                vistas.add(chave)
                saida.append(frase)
        print(f"\n--- {nome} ({len(saida)} frases) ---")
        for frase in saida[:limite_por_categoria]:
            print(f"  • {frase[:max_chars]}")


MODOS = {
    "JANELA": [
        r"\b\d+(?:[.,]\d+)?\s*(?:ms|milliseconds|s|sec|seconds)\b[^.]{0,90}"
        r"(?:window|epoch|segment|interval|trial|duration)",
        r"(?:window|epoch|segment|interval|duration)[^.]{0,90}"
        r"\b\d+(?:[.,]\d+)?\s*(?:ms|milliseconds|s|sec|seconds)\b",
        r"sliding (?:time )?window", r"\b\d{3,5}\s*samples\b",
        r"window (?:size|length)s?", r"post-?stimulus", r"pre-?stimulus",
    ],
    "ALVO": [
        r"(?:predict|estim|reconstruct|decod)[^.]{0,90}"
        r"(?:trajector|kinematic|velocity|position)",
        r"ground[- ]truth[^.]{0,90}(?:trajector|kinematic|velocity|position)",
        r"(?:target|label|output)[^.]{0,70}"
        r"(?:trajector|kinematic|velocity|position|3D|2D)",
    ],
}


def digest(caminho_pdf, maximo=6):
    paginas = texto_do_pdf(caminho_pdf)
    texto = "\n".join(paginas)
    print("=" * 78)
    print(f"{os.path.basename(caminho_pdf)} ({len(paginas)} p.)")
    for nome, padroes in MODOS.items():
        compilados = [re.compile(p, re.I) for p in padroes]
        vistas, saida = set(), []
        for frase in frases(texto):
            if any(p.search(frase) for p in compilados):
                chave = frase[:70].lower()
                if chave in vistas:
                    continue
                vistas.add(chave)
                saida.append(frase)
        if saida:
            print(f"  [{nome}]")
        for frase in saida[:maximo]:
            print(f"    - {frase[:260]}")


def main():
    argumentos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--texto-completo" in sys.argv:
        alvo = argumentos[0] if argumentos else PADRAO[0]
        caminho = os.path.join(ATACHMENTS, alvo)
        if not caminho.lower().endswith(".pdf"):
            candidatos = [n for n in os.listdir(ATACHMENTS)
                          if alvo.lower() in n.lower() and n.lower().endswith(".pdf")]
            caminho = os.path.join(ATACHMENTS, candidatos[0])
        texto = "\n".join(texto_do_pdf(caminho))
        destino = os.path.join("_textos_pdf",
                               os.path.basename(caminho) + ".txt")
        os.makedirs("_textos_pdf", exist_ok=True)
        with open(destino, "w", encoding="utf-8") as handle:
            handle.write(texto)
        print(f"texto completo salvo em {destino} ({len(texto)} caracteres)")
        return

    padroes = argumentos or PADRAO
    arquivos = sorted(os.listdir(ATACHMENTS))
    usar_digest = "--digest" in sys.argv
    for padrao in padroes:
        achados = [n for n in arquivos
                   if padrao.lower() in n.lower() and n.lower().endswith(".pdf")]
        if not achados:
            print(f"!! nao encontrado: {padrao}")
            continue
        caminho = os.path.join(ATACHMENTS, achados[0])
        if usar_digest:
            digest(caminho)
        else:
            relatorio(caminho)


if __name__ == "__main__":
    main()