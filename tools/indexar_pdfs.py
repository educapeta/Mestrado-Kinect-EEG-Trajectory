#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INDEXADOR DE PDFs
=================
Analisa recursivamente uma pasta com PDFs, extrai o texto e os metadados
de cada arquivo, e escreve tudo em um único TXT centralizado e estruturado.

O TXT gerado pode ser usado posteriormente como contexto para uma LLM,
que poderá identificar quais arquivos são textos científicos.

Uso:
    python indexar_pdfs.py "C:\\caminho\\para\\a\\pasta" -o saida.txt

Dependências:
    pip install pymupdf
"""

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

# Pastas que devem ser ignoradas na busca recursiva
# (bibliotecas, ambientes virtuais, caches, etc.)
PASTAS_IGNORADAS = {
    ".venv",
    "venv",
    "env",
    "site-packages",
    "node_modules",
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".eggs",
}

try:
    import pymupdf  # PyMuPDF (nova API)
except ImportError:
    print("=" * 70)
    print("ERRO: Biblioteca PyMuPDF não encontrada.")
    print("Instale com:  pip install pymupdf")
    print("=" * 70)
    sys.exit(1)


# =============================================================================
# Funções auxiliares
# =============================================================================

def format_date(date_obj):
    """Formata um objeto datetime para string legível."""
    if date_obj is None:
        return "N/A"
    try:
        return date_obj.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(date_obj)


def sha256_of_file(file_path):
    """Calcula o hash SHA-256 do arquivo (útil para deduplicação)."""
    try:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "N/A"


def extract_pdf_info(pdf_path, max_pages=0, max_chars=0):
    """
    Extrai metadados e texto de um PDF.

    Retorna um dicionário com:
      - path, folder, filename
      - file_size_bytes, file_size_mb
      - last_modified, created
      - sha256
      - metadata (título, autor, etc.)
      - page_count
      - text (texto extraído, com marcadores de página)
      - has_text_layer (False se for PDF escaneado/sem texto)
      - error (mensagem de erro, se houver)
    """
    info = {
        "path": str(pdf_path),
        "folder": str(pdf_path.parent),
        "filename": pdf_path.name,
        "file_size_bytes": pdf_path.stat().st_size,
        "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
        "last_modified": format_date(datetime.fromtimestamp(pdf_path.stat().st_mtime)),
        "created": format_date(datetime.fromtimestamp(pdf_path.stat().st_ctime)),
        "sha256": sha256_of_file(pdf_path),
        "metadata": {},
        "page_count": 0,
        "text": "",
        "has_text_layer": True,
        "error": None,
    }

    try:
        doc = pymupdf.open(str(pdf_path))

        # Se o PDF for criptografado, tenta abrir com senha vazia
        if doc.needs_pass:
            if not doc.authenticate(""):
                info["error"] = "PDF protegido por senha (não foi possível abrir)."
                doc.close()
                return info

        info["page_count"] = doc.page_count
        info["metadata"] = doc.metadata or {}

        # Define quantas páginas extrair
        pages_to_extract = doc.page_count
        if max_pages > 0:
            pages_to_extract = min(doc.page_count, max_pages)

        # Extrai texto página por página
        text_parts = []
        total_chars = 0
        for page_num in range(pages_to_extract):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            text_parts.append(f"--- Página {page_num + 1} ---\n{page_text}")
            total_chars += len(page_text)

            # Limite de caracteres por documento (se configurado)
            if max_chars > 0 and total_chars >= max_chars:
                break

        info["text"] = "\n\n".join(text_parts)

        # Detecta PDF escaneado (sem camada de texto)
        if len(info["text"].strip()) < 50:
            info["has_text_layer"] = False

        doc.close()

    except Exception as e:
        info["error"] = str(e)

    return info


# =============================================================================
# Função principal
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Indexa todos os PDFs de uma pasta em um único TXT estruturado."
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default=r"C:\Users\elgut\Downloads",
        help="Pasta contendo os PDFs a analisar (busca recursiva). "
             "Se não informada, usa o caminho padrão configurado.",
    )
    parser.add_argument(
        "-o", "--output",
        default="pdf_index.txt",
        help="Arquivo TXT de saída (padrão: pdf_index.txt).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Limite de páginas extraídas por PDF (0 = todas as páginas).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Limite de caracteres de texto extraído por PDF (0 = sem limite).",
    )
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"ERRO: Pasta não encontrada: {input_folder}")
        sys.exit(1)

    # Busca todos os PDFs recursivamente, ignorando pastas de bibliotecas
    pdf_files = sorted(
        pdf for pdf in input_folder.rglob("*.pdf")
        if not any(part.lower() in PASTAS_IGNORADAS for part in pdf.parts[:-1])
    )
    print(f"Pasta analisada: {input_folder}")
    print(f"PDFs encontrados: {len(pdf_files)}")

    if not pdf_files:
        print("Nenhum PDF encontrado. Nada a fazer.")
        sys.exit(0)

    # =========================================================================
    # Cabeçalho do arquivo de saída
    # =========================================================================
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("ÍNDICE DE PDFs — TEXTO EXTRAÍDO E INDEXADO")
    output_lines.append("=" * 80)
    output_lines.append(f"Data de geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Pasta analisada: {input_folder}")
    output_lines.append(f"Total de PDFs encontrados: {len(pdf_files)}")
    output_lines.append("=" * 80)
    output_lines.append("")

    # =========================================================================
    # Processa cada PDF
    # =========================================================================
    for idx, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{idx}/{len(pdf_files)}] Processando: {pdf_path.name} ...")
        info = extract_pdf_info(pdf_path, args.max_pages, args.max_chars)

        output_lines.append("#" * 80)
        output_lines.append(f"### DOCUMENTO {idx} ###")
        output_lines.append("#" * 80)
        output_lines.append(f"LOCAL DA PASTA: {info['folder']}")
        output_lines.append(f"TITULO DO ARQUIVO: {info['filename']}")
        output_lines.append(f"CAMINHO COMPLETO: {info['path']}")
        output_lines.append(f"TAMANHO DO ARQUIVO: {info['file_size_mb']} MB ({info['file_size_bytes']} bytes)")
        output_lines.append(f"DATA DE MODIFICAÇÃO: {info['last_modified']}")
        output_lines.append(f"DATA DE CRIAÇÃO: {info['created']}")
        output_lines.append(f"HASH SHA-256: {info['sha256']}")
        output_lines.append(f"NÚMERO DE PÁGINAS: {info['page_count']}")

        # Metadados do PDF
        output_lines.append("--- METADADOS DO PDF ---")
        meta = info["metadata"]
        output_lines.append(f"Título: {meta.get('title', 'N/A')}")
        output_lines.append(f"Autor: {meta.get('author', 'N/A')}")
        output_lines.append(f"Assunto: {meta.get('subject', 'N/A')}")
        output_lines.append(f"Palavras-chave: {meta.get('keywords', 'N/A')}")
        output_lines.append(f"Criador: {meta.get('creator', 'N/A')}")
        output_lines.append(f"Produtor: {meta.get('producer', 'N/A')}")
        output_lines.append(f"Data de criação: {meta.get('creationDate', 'N/A')}")
        output_lines.append(f"Data de modificação: {meta.get('modDate', 'N/A')}")

        # Texto extraído ou avisos
        if info["error"]:
            output_lines.append(f"ERRO AO LER PDF: {info['error']}")
        elif not info["has_text_layer"]:
            output_lines.append("AVISO: PDF possivelmente escaneado (sem camada de texto extraível).")
            output_lines.append("TEXTO EXTRAÍDO: [NENHUM — PDF ESCANEADO/IMAGEM]")
        else:
            output_lines.append("--- TEXTO EXTRAÍDO ---")
            output_lines.append(info["text"])

        output_lines.append("")
        output_lines.append("")

    # =========================================================================
    # Resumo final (visão geral para a LLM)
    # =========================================================================
    output_lines.append("=" * 80)
    output_lines.append("RESUMO GERAL DOS DOCUMENTOS INDEXADOS")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("Lista de todos os documentos encontrados:")
    output_lines.append("")

    for idx, pdf_path in enumerate(pdf_files, start=1):
        info = extract_pdf_info(pdf_path, max_pages=1, max_chars=0)
        output_lines.append(f"[{idx}] {info['filename']}")
        output_lines.append(f"    Pasta: {info['folder']}")
        output_lines.append(f"    Páginas: {info['page_count']} | Tamanho: {info['file_size_mb']} MB")
        if info["error"]:
            output_lines.append(f"    Status: ERRO ({info['error']})")
        elif not info["has_text_layer"]:
            output_lines.append("    Status: PDF escaneado (sem texto extraível)")
        else:
            output_lines.append("    Status: OK (texto extraído)")
        output_lines.append("")

    output_lines.append("=" * 80)
    output_lines.append("FIM DO ÍNDICE")
    output_lines.append("=" * 80)

    # =========================================================================
    # Escreve o arquivo de saída
    # =========================================================================
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print()
    print(f"Concluído! {len(pdf_files)} PDFs indexados.")
    print(f"Arquivo de saída: {output_path.resolve()}")


if __name__ == "__main__":
    main()