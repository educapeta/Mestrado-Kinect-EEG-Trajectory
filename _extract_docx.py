"""Extrai texto (paragrafos + tabelas) de um .docx sem python-docx.

Uso:  python _extract_docx.py <arquivo.docx> [--out saida.txt]
"""

from __future__ import annotations

import sys
import zipfile
import re
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _para_text(paragraph) -> str:
    parts = []
    for node in paragraph.iter():
        tag = node.tag.split("}")[-1]
        if tag == "t":
            parts.append(node.text or "")
        elif tag == "tab":
            parts.append("\t")
        elif tag in ("br", "cr"):
            parts.append("\n")
    return "".join(parts)


def extract(path: Path) -> str:
    out = []
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        xml_files = [n for n in names if n == "word/document.xml"]
        if not xml_files:
            raise SystemExit("word/document.xml nao encontrado")
        root = ET.fromstring(zf.read("word/document.xml"))
    body = root.find("w:body", NS)
    if body is None:
        return ""
    for child in body:
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text = _para_text(child)
            style = ""
            ppr = child.find("w:pPr", NS)
            if ppr is not None:
                ps = ppr.find("w:pStyle", NS)
                if ps is not None:
                    style = ps.get(
                        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", ""
                    )
            prefix = f"[{style}] " if style else ""
            out.append(prefix + text)
        elif tag == "tbl":
            out.append("<<TABELA>>")
            for row in child.findall("w:tr", NS):
                cells = []
                for cell in row.findall("w:tc", NS):
                    cell_text = " ".join(
                        _para_text(p) for p in cell.findall("w:p", NS)
                    ).strip()
                    cells.append(cell_text)
                out.append(" | ".join(cells))
            out.append("<</TABELA>>")
    return "\n".join(out)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("uso: python _extract_docx.py arquivo.docx [--out saida.txt]")
    src = Path(sys.argv[1])
    text = extract(src)
    out_path = None
    if "--out" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("--out") + 1])
    if out_path is None:
        print(text)
    else:
        out_path.write_text(text, encoding="utf-8")
        print(f"OK -> {out_path} ({len(text)} chars)")


if __name__ == "__main__":
    main()
