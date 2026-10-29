"""Testes do T base vs yaw do boot e do save (item 153).

Prova que o arquivo `imu_palm_alignment_<lado>.json` continua valendo para
qualquer boot: o `AncoradorYawBoot` altera so' o T em memoria, e
`save_palm_alignment` grava o T ORIGINAL (base), nao o reancorado.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

import imu

FALHAS = []


def check(nome, condicao, detalhe=None):
    if condicao:
        print(f"   ok: {nome}")
    else:
        FALHAS.append(nome)
        print(f"   FALHOU: {nome} | {detalhe!r}")


def rot(yaw_graus, pitch_graus=0.0, roll_graus=0.0):
    return imu.rotation_matrix(roll_graus, pitch_graus, yaw_graus)


T1 = rot(10.0, 5.0, -3.0)
T2 = rot(40.0, 5.0, -3.0)
f = np.array([0.10, -0.20, 0.97])
f = f / np.linalg.norm(f)

print("1) posicao do T base x T reancorado")
fusao = imu.PositionFusion()
check("set calibracao devolve True", fusao.set_palm_alignment(T1, f, 3.0))
estado = fusao.palm_alignment_state(original=True)
check("base guardada", estado is not None and
      np.allclose(np.asarray(estado["rotation"]), T1, atol=1e-12))
fusao.set_palm_alignment(T2, f, 3.0, base=False)
check("T atual = reancorado",
      np.allclose(fusao.palm_alignment_state()["rotation"], T2, atol=1e-12))
check("T base continua o original",
      np.allclose(fusao.palm_alignment_state(original=True)["rotation"], T1,
                  atol=1e-12))
fusao.yaw_boot_delta_deg = 17.5
check("delta de yaw registrado no estado",
      abs(fusao.palm_alignment_state(original=True)["yaw_boot_delta_deg"] - 17.5)
      < 1e-9)

print("2) AncoradorYawBoot nao contamina o T base")
fusao2 = imu.PositionFusion()
fusao2.set_palm_alignment(T1, f, 3.0)
ancora = imu.AncoradorYawBoot(fusao2, lado="right", amostras_min=1,
                              avisar=lambda *_a: None)
with mock.patch("imu.reancorar_yaw",
                return_value=(T2, 17.5, 0.5, 1)):
    ancora.amostra(np.eye(3), np.array([0.0, 0.0, -1.0]))
check("ancorador aplicou T reancorado em memoria",
      np.allclose(fusao2.palm_alignment_state()["rotation"], T2, atol=1e-12))
check("ancorador manteve T base original",
      np.allclose(fusao2.palm_alignment_state(original=True)["rotation"], T1,
                  atol=1e-12))
check("delta registrado", abs(fusao2.yaw_boot_delta_deg - 17.5) < 1e-9)
check("origem nao virou bootstrap", fusao2.palm_alignment_source != "bootstrap")

print("3) save_palm_alignment grava o T base, nao o reancorado")
raiz = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "kgt_tool_reancora", raiz / "tools" / "kinect_groundtruth_tool.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

cwd_antes = os.getcwd()
with tempfile.TemporaryDirectory(prefix="reancora_") as tmp:
    tmp = Path(tmp)
    os.chdir(tmp)
    tool.PALM_ALIGNMENT_FILES = {
        "right": str(tmp / "imu_palm_alignment_right.json"),
        "left": str(tmp / "imu_palm_alignment_left.json"),
    }
    try:
        fusao3 = imu.PositionFusion()
        fusao3.set_palm_alignment(T1, f, 3.0)
        # simula reancoragem (T2 em memoria)
        fusao3.set_palm_alignment(T2, f, 3.0, base=False)
        fusao3.yaw_boot_delta_deg = 17.5
        salvou = tool.save_palm_alignment(fusao3, "right",
                                          avisar=lambda *_a: None)
        check("save devolve True", salvou is True)
        caminho = tmp / "imu_palm_alignment_right.json"
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        check("JSON tem o T ORIGINAL",
              np.allclose(np.asarray(dados["rotation"]), T1, atol=1e-12))
        check("JSON tem delta de yaw separado",
              abs(dados.get("yaw_boot_delta_deg", 0.0) - 17.5) < 1e-9)
        # load de volta: base e atual = T1
        fusao4 = imu.PositionFusion()
        carregou = tool.load_palm_alignment(fusao4, "right",
                                            avisar=lambda *_a: None)
        check("load aplica o T salvo", carregou and
              np.allclose(fusao4.palm_alignment_state()["rotation"], T1,
                          atol=1e-12))
        check("load tambem preenche a base", fusao4.imu_to_camera_base is not None
              and np.allclose(fusao4.imu_to_camera_base, T1, atol=1e-12))
    finally:
        os.chdir(cwd_antes)

if FALHAS:
    print()
    print("REANCORAGEM_ALINHAMENTO_FALHOU: " + " | ".join(FALHAS))
    raise SystemExit(1)
print()
print("REANCORAGEM_ALINHAMENTO_OK: T base preservado no save, AncoradorYawBoot "
      "so' altera o T em memoria, load repoe base")
