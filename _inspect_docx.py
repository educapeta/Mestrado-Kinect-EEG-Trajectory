"""Inspeciona a estrutura de um .docx (paragrafos, estilos e tabelas)."""

from __future__ import annotations

import sys
from pathlib import Path

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_block_items(parent):
    from docx.oxml.ns import qn

    body = parent.element.body if hasattr(parent, "element") else parent._element
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def main() -> None:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    document = docx.Document(str(src))
    lines = []
    pi = 0
    ti = 0
    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            style = block.style.name if block.style is not None else ""
            lines.append(f"P{pi:03d} [{style}] {block.text!r}")
            pi += 1
        else:
            lines.append(f"=== T{ti:02d} TABLE rows={len(block.rows)} ===")
            for ri, row in enumerate(block.rows):
                cells = [c.text.replace("\n", " \\n ") for c in row.cells]
                lines.append(f"  T{ti:02d}R{ri:02d}: " + " || ".join(cells))
            ti += 1
    text = "\n".join(lines)
    if out is None:
        print(text)
    else:
        out.write_text(text, encoding="utf-8")
        print(f"OK -> {out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
