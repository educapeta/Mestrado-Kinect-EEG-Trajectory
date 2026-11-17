"""Suite do `tools/checklist_kinect_mudo.py` -- o que sobra depois dos descartes.

ONDE ESTA SUITE SE ENCAIXA (bancada de 07/10/2026)
  Com o `IsAvailable` FALSE e o Kinect Studio mostrando imagem, o valor do
  checklist nao e' listar tudo de novo: e' listar SO' o que sobrou depois de
  DESCARTAR por medicao (hardware enumerado, driver ligado, Kinect20.dll unico,
  servico do SDK rodando, melhorias de audio desabilitadas + reinicio). E o
  primeiro item dele -- suspensao seletiva de USB -- foi escolhido porque foi
  MEDIDO ligado nesta maquina (CA=1, CC=1 no plano Equilibrado), junto com o
  proprio sensor marcado `DeviceSelectiveSuspended = 1` no registro.

  Os casos abaixo travam:
   1. `suspensao_seletiva` le o valor e nao o rotulo: o `powercfg /query` em
      portugues escreve "Correntes Alternadas/Continuas" e o numero vem em
      `0x00000001`/`0x00000000` -- procurar "current ac" acha em ingles e
      PERDE em portugues;
   2. a decodificacao e' forcada em UTF-8 com `errors="replace"`: com `text=True`
      sozinho o `subprocess` usa o codec do console e MORRE com
      `UnicodeDecodeError` na saida em portugues do powercfg (medido: byte 0x8d
      em "Correntes Continuas") -- e a excecao na thread de leitura deixa
      `proc.stdout` None, que era o segundo erro em cascata;
   3. falha do powercfg devolve (None, None) e nao estoura;
   4. o primeiro passo do checklist e' a suspensao seletiva, com o comando de
      IDA e o de VOLTA (o operador tem de poder desfazer);
   5. `estado()` classifica o IsAvailable sem inventar: FALSE e' FALSE.
"""
from __future__ import annotations

import io
import contextlib
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
if str(RAIZ / "tools") not in sys.path:
    sys.path.insert(0, str(RAIZ / "tools"))

import checklist_kinect_mudo as ck                          # noqa: E402

casos = 0
original_run = subprocess.run


class _Proc:
    def __init__(self, saida, erro=""):
        self.stdout = saida
        self.stderr = erro
        self.returncode = 0


# --- 1) o parser acha os valores nos DOIS idiomas ---------------------------
SAIDA_PT = (
    "GUID do Esquema de Energia: 381b4222-f694-41f0-9685-ff5bb260df2e  (Equilibrado)\n"
    "  GUID de Subgrupos: 2a737441-1930-4402-8d77-b2bebba308a3  (Configuracoes USB)\n"
    "    GUID de Configuracao de Energia: 48e6b7a6-50f5-4782-a5d4-53bb8f07e226\n"
    "      Indice de Configuracoes de Correntes Alternadas Atuais: 0x00000001\n"
    "      Indice de Configuracoes de Correntes Continuas Atuais: 0x00000001\n"
)
SAIDA_EN = (
    "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)\n"
    "  Current AC Power Setting Index: 0x00000000\n"
    "  Current DC Power Setting Index: 0x00000001\n"
)
for saida, esperado in ((SAIDA_PT, (1, 1)), (SAIDA_EN, (0, 1))):
    subprocess.run = lambda *a, **k: _Proc(saida)
    obtido = ck.suspensao_seletiva()
    assert obtido == esperado, (obtido, esperado, saida[:40])
casos += 1

# --- 2) powercfg que falha devolve (None, None) ----------------------------
subprocess.run = lambda *a, **k: _Proc("")
assert ck.suspensao_seletiva() == (None, None)
subprocess.run = lambda *a, **k: (_ for _ in ()).throw(OSError("sem powercfg"))
assert ck.suspensao_seletiva() == (None, None)
subprocess.run = original_run
casos += 1

# --- 3) a decodificacao e' forcada (o bug real de 07/10/2026) --------------
#: O `subprocess.run` de verdade, com `text=True` e sem `encoding`, levantou
#: UnicodeDecodeError na thread de leitura. Aqui o duble confere que o modulo
#: PEDE `encoding="utf-8"` e `errors="replace"`: sem isso, a saida em portugues
#: do powercfg derruba o programa no meio do checklist.
pedidos = {}


def run_espiado(*args, **kwargs):
    pedidos.update(kwargs)
    return _Proc(SAIDA_PT)


subprocess.run = run_espiado
ck.suspensao_seletiva()
subprocess.run = original_run
assert pedidos.get("encoding") == "utf-8", pedidos
assert pedidos.get("errors") == "replace", pedidos
assert pedidos.get("text") is True, pedidos
casos += 1

# --- 4) o primeiro passo e' a suspensao seletiva, com IDA e VOLTA ----------
numero, titulo, comandos, explicacao = ck.PASSOS[0]
assert numero == "1", numero
assert "SUSPENSAO SELETIVA" in titulo.upper(), titulo
assert any("/setacvalueindex" in c for c in comandos), comandos
assert any("/setdcvalueindex" in c for c in comandos), comandos
assert any(c.rstrip().endswith(" 0") for c in comandos), \
    "o comando de IDA tem de desligar (0)"
assert "1 no ultimo numero" in explicacao, explicacao
#: O GUID do subgrupo/configuracao tem de ser o de suspensao seletiva de USB.
assert ck.SUBGRUPO_USB == "2a737441-1930-4402-8d77-b2bebba308a3", ck.SUBGRUPO_USB
assert ck.CONFIG_SUSPENSAO_SELETIVA == "48e6b7a6-50f5-4782-a5d4-53bb8f07e226", \
    ck.CONFIG_SUSPENSAO_SELETIVA
casos += 1

# --- 5) todos os passos tem numero unico e explicacao ----------------------
numeros = [p[0] for p in ck.PASSOS]
assert numeros == [str(i) for i in range(1, len(ck.PASSOS) + 1)], numeros
for numero, titulo, comandos, explicacao in ck.PASSOS:
    assert titulo and explicacao, (numero, titulo, explicacao)
casos += 1

# --- 6) estado() nao inventa veredito e casa com o probe ------------------
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    ca, cc = ck.estado()
saida = buffer.getvalue()
assert "IsAvailable" in saida, saida
#: O estado desta maquina pode mudar; o que se trava e' a coerencia: se a
#: suspensao seletiva esta' ligada (1), o texto tem de avisar.
if ca == 1 or cc == 1:
    assert "suspeito" in saida, saida
else:
    assert "suspeito" not in saida, saida
casos += 1

# --- 7) `--estado` imprime o estado e NAO o checklist ---------------------
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    codigo = ck.main(["--estado"])
saida = buffer.getvalue()
assert codigo == 0, codigo
assert "ESTADO MEDIDO AGORA" in saida, saida
assert "CHECKLIST" not in saida, "--estado imprimiu o checklist"
casos += 1

print("CHECKLIST_KINECT_MUDO_OK: %d casos (suspensao seletiva lida pelo VALOR "
      "nos dois idiomas; decodificacao forcada em UTF-8 com replace; powercfg "
      "que falha sem estourar; passo 1 com comando de ida e de volta; passos "
      "numerados; estado coerente com o aviso; --estado sem checklist)" % casos)
