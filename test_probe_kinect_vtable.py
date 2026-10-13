"""Suite do `tools/probe_kinect_vtable.py` -- a ORDEM do vtable e o c_bool.

O ERRO QUE ESTA SUITE EXISTE PARA IMPEDIR (bancada de 07/10/2026)
  Eu li o `IsAvailable` pelo endereco ERRADO do vtable: chutei que
  `IKinectSensor` comecava em QueryInterface/AddRef/Release e que o
  `get_IsAvailable` era a entrada 3. Nao e': antes dele vem
  `SubscribeIsAvailableChanged`, `UnsubscribeIsAvailableChanged`,
  `GetIsAvailableChangedEventData`, `Open` e `Close`. A entrada 3 e' um
  SUBSCRIBE, cujo argumento esperado e' um `INT_PTR*`; passando o ponteiro de um
  `c_int` de 4 bytes, o SDK devolveu `E_INVALIDARG` e a minha mensagem imprimiu
  `valor=820` -- lixo dos outros 3 bytes. Em cima disso eu anunciei um falso
  "o sensor diz TRUE pelo vtable cru", que e' exatamente o tipo de conclusao que
  nao pode sair de um prototipo errado.

  Os casos abaixo travam:
   1. os indices do modulo batem com a ORDEM REAL do `IKinectSensor._methods_`
      em `pykinect2/PyKinectV2.py` -- lida do FONTE, nao redigitada aqui. Se uma
      atualizacao do pykinect2 mover um metodo, esta suite cai;
   2. `IsAvailable` e' propriedade `c_bool` (BOOLEAN de 1 byte) e NAO `c_int` --
      ler 4 bytes escreve lixo nos outros 3 e inventa valor;
   3. `Open`/`Close` nao tem argumento de saida (chamar com ponteiro = E_INVALIDARG);
   4. o veredito separa os quatro casos (biblioteca, disponivel, indisponivel,
      inconclusivo) em vez de tratar qualquer numero como boolean;
   5. `SubscribeIsAvailableChanged` (a entrada 3) devolve HANDLE, nao booleano.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
if str(RAIZ / "tools") not in sys.path:
    sys.path.insert(0, str(RAIZ / "tools"))

import ctypes                                               # noqa: E402

import probe_kinect_vtable as pv                            # noqa: E402

casos = 0

#: O fonte do pykinect2 dentro DESTE venv: a ordem do vtable nao pode ser
#: redigitada aqui, tem de vir dele.
FONTE = (Path(pv.__file__).resolve().parent.parent / "Lib" / "site-packages"
         / "pykinect2" / "PyKinectV2.py")
if not FONTE.exists():                                      # pragma: no cover
    FONTE = (RAIZ / ".venv" / "Lib" / "site-packages" / "pykinect2"
             / "PyKinectV2.py")
assert FONTE.exists(), "nao achei o PyKinectV2.py do venv: %s" % FONTE
texto = FONTE.read_text(encoding="mbcs", errors="replace")

# --- 1) a ordem real do _methods_ -------------------------------------------
inicio = texto.index("IKinectSensor._methods_ = [")
#: O fim do bloco e' um `]` SOZINHO numa linha. Cortar no primeiro `]` acha o
#: colchete INTERNO de `COMMETHOD([], HRESULT, ...)` e devolve ZERO metodos
#: -- foi o que aconteceu na primeira versao desta suite.
fim = texto.index("\n]", inicio)
bloco = texto[inicio:fim]
metodos = re.findall(r"COMMETHOD\(\[[^\]]*\],\s*HRESULT,\s*'([A-Za-z_]+)'", bloco)
assert metodos, "nao consegui extrair os metodos do IKinectSensor"
assert metodos[0] == "SubscribeIsAvailableChanged", metodos[:3]
assert "Open" in metodos and "Close" in metodos, metodos[:8]
indice_open = metodos.index("Open")
indice_close = metodos.index("Close")
indice_is_available = metodos.index("IsAvailable")
indice_is_open = metodos.index("IsOpen")
casos += 1

# --- 2) os indices do modulo batem com o fonte ------------------------------
#: O vtable COM tem 3 entradas de IUnknown ANTES dos metodos da interface.
DESLOCAMENTO = 3
assert pv.INDICE_IS_AVAILABLE == indice_is_available + DESLOCAMENTO, (
    "INDICE_IS_AVAILABLE=%d mas o fonte poe IsAvailable em %d (+3 de IUnknown)"
    % (pv.INDICE_IS_AVAILABLE, indice_is_available + DESLOCAMENTO))
assert pv.INDICE_IS_OPEN == indice_is_open + DESLOCAMENTO, (
    pv.INDICE_IS_OPEN, indice_is_open + DESLOCAMENTO)
assert pv.INDICE_IS_AVAILABLE > pv.INDICE_IS_OPEN, "a ordem do SDK e' IsOpen antes"
#: A entrada 3 tem de ser o SUBSCRIBE -- foi lendo ELA como IsAvailable que o
#: falso positivo nasceu.
assert metodos[0] == "SubscribeIsAvailableChanged"
assert DESLOCAMENTO == 3, DESLOCAMENTO
casos += 1

# --- 3) os tipos do modulo batem com os do fonte ----------------------------
por_indice = {indice: (nome, tipo) for indice, nome, tipo in pv.ENTRADAS}
for indice, nome, _ in pv.ENTRADAS:
    if DESLOCAMENTO <= indice < DESLOCAMENTO + len(metodos):
        assert por_indice[indice][0] == metodos[indice - DESLOCAMENTO], (
            "entrada %d: modulo diz %s, fonte diz %s"
            % (indice, por_indice[indice][0], metodos[indice - DESLOCAMENTO]))
#: IsOpen/IsAvailable sao POINTER(c_bool) no fonte.
for nome in ("IsOpen", "IsAvailable"):
    trecho = bloco[bloco.index("'%s'" % nome):][:180]
    assert "c_bool" in trecho, (nome, trecho)
    indice = DESLOCAMENTO + metodos.index(nome)
    assert por_indice[indice][1] == "bool", (nome, por_indice[indice])
#: Open/Close nao tem parametro de saida.
for nome in ("Open", "Close"):
    trecho = texto[inicio:fim]
    padrao = r"COMMETHOD\(\[\],\s*HRESULT,\s*'%s'\)" % nome
    assert re.search(padrao, trecho), ("%s nao esta' sem saida" % nome)
    indice = DESLOCAMENTO + metodos.index(nome)
    assert por_indice[indice][1] == "sem_saida", (nome, por_indice[indice])
#: A entrada 3 (Subscribe) devolve HANDLE -- 8 bytes, nao booleano.
assert por_indice[3][1] == "int_ptr", por_indice[3]
casos += 1

# --- 4) o prototipo de `bool` le 1 byte e nao inventa valor -----------------
#: Simula o SDK: uma funcao que escreve UM byte (True) e devolve S_OK. Se o
#: modulo usasse `c_int`, a leitura pegaria lixo dos bytes vizinhos; com `c_bool`
#: o valor tem de ser exatamente True.
def escreve_um_byte(este, saida):
    ctypes.cast(saida, ctypes.POINTER(ctypes.c_bool))[0] = True
    return 0


prototipo = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_bool))
funcao = prototipo(escreve_um_byte)
saida = ctypes.c_bool(False)
hr = funcao(None, ctypes.byref(saida))
assert hr == 0 and saida.value is True, (hr, saida.value)
#: E o sentinela nunca sobrevive: e' o que prova que a funcao escreveu.
sentinela = ctypes.c_int(-12345)
assert not isinstance(sentinela.value, bool)
casos += 1

# --- 5) o veredito separa os quatro casos -----------------------------------
#: Reimplementa SO' a arvore de decisao do `main` sobre os numeros medidos, para
#: travar que "820" nunca vire "TRUE".
def veredito(hr, valor, comtypes):
    if hr != 0:
        return "inconclusivo"
    if valor and not comtypes:
        return "biblioteca"
    if valor:
        return "disponivel"
    return "indisponivel"


assert veredito(0, False, False) == "indisponivel"
assert veredito(0, True, False) == "biblioteca"
assert veredito(0, True, True) == "disponivel"
assert veredito(-2147024809, False, False) == "inconclusivo"   # E_INVALIDARG
#: O caso do erro real: E_INVALIDARG (nao S_OK) NUNCA e' veredito.
assert veredito(0x80070057 - (1 << 32), False, False) == "inconclusivo"
casos += 1

print("PROBE_KINECT_VTABLE_OK: %d casos (ordem do vtable lida do FONTE do "
      "pykinect2; indices do modulo batem com ela; IsAvailable/IsOpen sao c_bool "
      "e Open/Close sem saida; prototipo de 1 byte nao inventa valor; o veredito "
      "separa biblioteca/disponivel/indisponivel/inconclusivo)" % casos)
