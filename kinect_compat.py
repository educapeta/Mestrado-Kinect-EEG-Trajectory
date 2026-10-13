"""Compatibilidade do pykinect2 (PyPI 0.1.0) com o Python 64 BITS deste projeto.

PROBLEMA (medido na bancada, 28/09/2026)
  `from pykinect2 import PyKinectV2` morre com
      AssertionError: 80        (pykinect2/PyKinectV2.py, linha 2216)
  porque o pykinect2 foi gerado para Python 32 BITS (3.4): em 64 bits o
  `tagSTATSTG` do OLE ocupa 80 bytes (o `LPOLESTR` vira ponteiro de 8 bytes e
  forca mais 4 de padding), nao os 72 que o assert exige. E' o problema
  conhecido do upstream (issues #37 e #115 do Kinect/PyKinect2), cuja receita
  oficial e' EDITAR o site-packages -- o que se perde na proxima instalacao.
  Com o assert no caminho, NENHUM programa que importe
  `kinect_imu_groundtruth`/`stereo_calibration` roda (eram 14 suites de teste
  falhando por isso, 28/09/2026).

  A estrutura `tagSTATSTG` NAO e' usada para falar com o sensor: ela vem do
  `IStream` de gravacao do type library (o Kinect entrega color/depth/body por
  outras interfaces). O assert e' apenas a checagem de sanidade que o gerador
  de codigo do comtypes planta no fim de cada struct. Aceitar as DUAS medidas
  (72 = 32 bits, 80 = 64 bits) preserva a checagem e nao muda ABI nenhuma.

SOLUCAO (sem tocar no site-packages)
  * `aplicar()` instala um IMPORT HOOK para `pykinect2.PyKinectV2`: o fonte
    instalado e' lido, o Patch A e' aplicado e so' entao executado. Como o
    patch vive no REPOSITORIO, ele sobrevive a `pip install --force-reinstall`
    e a recriacao do `.venv` (a receita manual, nao).
  * Patch C (mesmas issues, Python 3.8+): o `PyKinectRuntime` chama
    `time.clock()`, removido no 3.8. Em vez de reescrever o fonte, o shim
    devolve o equivalente (`time.perf_counter`) quando a funcao nao existe --
    vale para qualquer modulo do site-packages, nao so' o pykinect2.
  * Patch B (mesmas issues): o fim do arquivo instalado chama
    `from comtypes import _check_version; _check_version('')` -- com o comtypes
    moderno (1.4.17 aqui) isso levanta `ImportError: Wrong version`, porque a
    chamada compara a versao do GERADOR de codigo com uma string vazia. O
    pykinect2 ja' traz o type library gerado em codigo, entao a checagem nao
    tem o que validar: o shim troca a linha por um comentario (no FONTE que
    vai ser executado, sem mexer no site-packages e sem afetar outros usuarios
    do comtypes, que continuam com a checagem ligada).
  * Patch D (mesma familia): o `PyKinectRuntime` faz
    `numpy.ndarray(..., dtype=numpy.object)` -- alias legado que o numpy 2.4
    ja' marca como condenado (FutureWarning em todo stderr de sessao, 28/09/
    2026). O shim troca por `dtype=object`, que e' o MESMO tipo.

Uso (no paradigma/ferramentas): chame `aplicar()` ANTES do import do pykinect2.
O import de `kinect_compat` e' leve (stdlib apenas) e pode ficar no topo do
modulo, fora do `try: from pykinect2 import ...`:

    import kinect_compat
    kinect_compat.aplicar()
    from pykinect2 import PyKinectV2

Depois de aplicar, `motivo()` diz o que foi corrigido (usado no diagnostico de
cameras para nao reportar um "pykinect2 indisponivel" generico).
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import time

#: Modulos cujo fonte precisa de ajuste antes de executar.
ALVOS = ("pykinect2.PyKinectV2", "pykinect2.PyKinectRuntime")

#: Codec que o arquivo instalado declara (`# -*- coding: mbcs -*-`).
CODIFICACAO = "mbcs"

#: Correcoes por modulo (Patch A/B/C/D do docstring deste modulo). A troca e'
#: por TEXTO no fonte instalado -- sem tocar no site-packages e sem afetar
#: outros usuarios do comtypes/numpy (a checagem e o alias continuam valendo
#: para eles; so' o fonte do pykinect2 e' que sai ajustado).
_CORRECOES = {
    "pykinect2.PyKinectV2": (
        # Patch A -- o assert do upstream e' o que derruba o import em 64 bits.
        # Trocamos "== 72" por "in (72, 80)": continua sendo um assert (se o
        # `tagSTATSTG` vier com outro tamanho, ainda avisa), mas aceita as
        # duas ABIs.
        ("assert sizeof(tagSTATSTG) == 72, sizeof(tagSTATSTG)",
         "assert sizeof(tagSTATSTG) in (72, 80), sizeof(tagSTATSTG)"),
        # Patch B -- a checagem de versao do comtypes no fim do arquivo (que
        # recebe string VAZIA como "versao real") sempre falha no comtypes
        # moderno: vira um comentario.
        ("from comtypes import _check_version; _check_version('')",
         "# Patch B (kinect_compat): checagem de versao do comtypes desativada"),
    ),
    "pykinect2.PyKinectRuntime": (
        # Patch D -- o alias `numpy.object` esta' condenado (FutureWarning no
        # numpy 2.4 e remocao marcada); `object` e' o mesmo tipo e cala o aviso
        # que poluia o stderr de toda sessao.
        ("dtype=numpy.object", "dtype=object"),
    ),
}

#: Estado do modulo: evita reinstalar o finder e guarda o que foi corrigido.
_aplicado = False
_motivos: list[str] = []


class _LoaderFonte(importlib.machinery.SourceFileLoader):
    """SourceFileLoader que ajusta o texto do modulo antes de compilar.

    Sobrescreve `get_code` (e nao `get_data`) de proposito: o
    `SourceFileLoader` do CPython reaproveita o `.pyc` do `__pycache__` quando
    ele existe, e o `.pyc` foi gravado com o assert ORIGINAL (o import falha ao
    EXECUTAR, depois de compilar/cachear). Reescrevendo a compilacao,
    garantimos que o patch vale mesmo com cache velho na pasta.
    """

    def __init__(self, nome, caminho, correcoes):
        super().__init__(nome, caminho)
        self._correcoes = tuple(correcoes)

    def get_code(self, fullname):
        caminho = self.get_filename(fullname)
        fonte = self.get_data(caminho)              # bytes, nao texto
        for antigo, novo in self._correcoes:
            fonte = fonte.replace(antigo.encode(CODIFICACAO),
                                  novo.encode(CODIFICACAO))
        return self.source_to_code(fonte, caminho)  # compila o fonte ajustado


class _Finder(importlib.abc.MetaPathFinder):
    """Desvia SO' o import dos `ALVOS` para o loader que aplica as correcoes."""

    def find_spec(self, fullname, path=None, target=None):
        correcoes = _CORRECOES.get(fullname)
        if correcoes is None:
            return None
        especificacao = importlib.machinery.PathFinder.find_spec(fullname, path)
        carregador = getattr(especificacao, "loader", None)
        if not isinstance(carregador, importlib.machinery.SourceFileLoader):
            return None                             # nao e' fonte .py: nao mexe
        if not isinstance(carregador, _LoaderFonte):
            especificacao.loader = _LoaderFonte(carregador.name,
                                                carregador.path, correcoes)
        return especificacao


def _patch_time_clock() -> bool:
    """Patch C: devolve `time.clock` (removido no Python 3.8) se faltar."""
    if hasattr(time, "clock"):
        return False
    setattr(time, "clock", time.perf_counter)       # shim do nome legado
    return True


def aplicar(forcar: bool = False) -> bool:
    """Garante os patches do pykinect2 64 bits (idempotente).

    Devolve True na primeira aplicacao, False se ja' estava aplicado. Chamar
    varias vezes (um modulo por programa importa o pykinect2) e' barato e sem
    efeito colateral: o finder entra UMA vez em `sys.meta_path`.
    """
    global _aplicado
    if _aplicado and not forcar:
        return False
    _motivos.clear()
    ja_importados = [nome for nome in ALVOS if nome in sys.modules]
    if not ja_importados:
        if not any(isinstance(f, _Finder) for f in sys.meta_path):
            sys.meta_path.insert(0, _Finder())
        _motivos.append("Patches A/B/D: tagSTATSTG em 64 bits, checagem de "
                        "versao do comtypes e o alias numpy.object")
    else:
        # Tarde demais para este processo: o modulo ja' foi executado. Nao
        # inventamos erro -- o chamador decide (a sonda `disponivel()` avisa).
        _motivos.append("aviso: ja' importado(s) neste processo: "
                        + ", ".join(ja_importados))
    if _patch_time_clock():
        _motivos.append("Patch C: time.clock -> time.perf_counter")
    _aplicado = True
    return True


def motivo() -> str:
    """Texto curto do que foi (ou nao) corrigido -- para logs e diagnosticos."""
    if not _aplicado:
        return "(compatibilidade do pykinect2 ainda nao aplicada)"
    return "; ".join(_motivos) if _motivos else "(nada a corrigir)"


def disponivel() -> bool:
    """True se o pykinect2 importa DEPOIS dos patches (sondagem rapida)."""
    aplicar()
    try:
        # PyKinectRuntime importa o PyKinectV2 por dentro: um import cobre os
        # dois modulos que o shim ajusta.
        importlib.import_module("pykinect2.PyKinectRuntime")
    except Exception:                                   # noqa: BLE001 - sonda
        return False
    return True


if __name__ == "__main__":                          # diagnostico manual
    aplicar()
    print(f"kinect_compat: {motivo()}")
    print(f"pykinect2 importavel: {disponivel()}")
