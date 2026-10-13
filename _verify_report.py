# -*- coding: utf-8 -*-
"""Verifica o relatorio gerado: dump de paragrafos e tabelas em UTF-8."""

from __future__ import annotations

from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

import _fill_report as F

DST_TXT = Path(r"c:\mestradopy\docs\_relatorios\_out_struct.txt")


def iter_blocks(document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def main() -> None:
    document = docx.Document(str(F.DST))
    lines = []
    pi = 0
    ti = 0
    for block in iter_blocks(document):
        if isinstance(block, Paragraph):
            if block.text.strip():
                lines.append(f"P{pi:03d} [{block.style.name}] {block.text}")
            pi += 1
        else:
            lines.append(f"=== T{ti:02d} rows={len(block.rows)} ===")
            for ri, row in enumerate(block.rows):
                seen = []
                for cell in row.cells:
                    if all(cell._tc is not s for s in seen):
                        seen.append(cell)
                if not seen:
                    continue
                cells = [c.text.replace("\n", " \\n ") for c in seen]
                lines.append(f"  T{ti:02d}R{ri:02d}: " + " || ".join(cells))
            ti += 1
    DST_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"OK -> {DST_TXT} ({len(lines)} linhas)")


if __name__ == "__main__":
    main()
