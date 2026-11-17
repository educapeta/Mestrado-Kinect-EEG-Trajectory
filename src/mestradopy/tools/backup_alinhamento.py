"""Backup simples dos alinhamentos da palma (JSONs) para uma pasta com data/hora."""
from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DESTINO_BASE = RAIZ / "backups_alinhamento"


def principal() -> int:
    arquivos = sorted(RAIZ.glob("imu_palm_alignment_*.json*"))
    if not arquivos:
        print("Nenhum imu_palm_alignment_*.json encontrado na raiz.")
        return 1
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = DESTINO_BASE / carimbo
    destino.mkdir(parents=True, exist_ok=True)
    print(f"Backup em: {destino}")
    with (destino / "SHA256.txt").open("w", encoding="utf-8") as hash_fh:
        for arquivo in arquivos:
            alvo = destino / arquivo.name
            shutil.copy2(arquivo, alvo)
            resumo = hashlib.sha256(alvo.read_bytes()).hexdigest()
            hash_fh.write(f"{resumo}  {arquivo.name}\n")
            print(f"  {arquivo.name}  ({arquivo.stat().st_size} bytes)")
    print(f"{len(arquivos)} arquivo(s) copiado(s). Pode recalibrar com seguranca.")
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
