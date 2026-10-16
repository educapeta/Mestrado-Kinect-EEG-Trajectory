"""Feedback do PRIMEIRO trial real (14/10/2026; item 58 do historico).

Duas reclamacoes do experimentador apos a sessao:

1. TEXTO CORTADO nos slides de instrucao: o `_paint_text` antigo usava a
   janela INTEIRA e fonte fixa (0,045 x o menor lado) SEM quebra de
   palavra -- a linha mais longa dos slides media ~1050 px em Arial Bold
   40 pt e era cortada na janela de 900 px. Agora o texto quebra em
   palavras (Qt.TextWordWrap) e a fonte encolhe ate' caber (margem de 5%).

2. PAINEL DE IMPEDANCIAS NUNCA APARECIA: no gpype 4.0.0 o `gp.GNautilus`
   e' um OChain SEM `__getattr__` -- o `_device` (gtec_gds.GNautilus, que
   tem `get_impedance(first)` e `get_scaling()`) mora no `_GNautilusCore`,
   um dos `internal_nodes`. O gate antigo `getattr(source, "_device")`
   devolvia None e `hasattr(source, "get_impedance")` tambem falhava: o
   `ImpedanceMonitor` NUNCA era criado e a tela de preparacao ficava em
   "(medindo impedancias...)" para sempre. `resolve_device` anda pelo
   chain (mesma estrutura real: Chain -> [core, Sync]) e o mesmo
   resolvedor alimenta o relatorio `--impedancia` e a escala counts->uV.
"""
import inspect
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import eeg_motor_paradigm as p

# ---------------------------------------------------------------------------
# 1) _ajusta_tamanho_fonte: encolhe ate' caber, com piso, sem loop infinito
# ---------------------------------------------------------------------------
CAIXA = (855, 855)          # 900x900 com a margem de 5% do _paint_text
INICIAL = 40                # 0.045 x 900 (o default da tela)
PISO = 16


def medida_falsa(texto, tamanho):
    """Medida no espirito do Arial Bold (~0.62 pt por char) COM wrap.

    Emula o Qt.TextWordWrap: linhas mais largas que a caixa viram varias
    (empacotamento guloso por palavras) -- a largura final nao excede a
    caixa, so' a altura cresce.
    """
    char = 0.62 * tamanho
    linhas_totais, maior = 0, 0.0
    for linha in texto.split("\n"):
        atual = 0.0
        for palavra in linha.split(" "):
            largura = len(palavra) * char
            espaco = char if atual > 0 else 0.0
            if atual > 0 and atual + espaco + largura > CAIXA[0]:
                linhas_totais += 1
                maior = max(maior, atual)
                atual = largura
            else:
                atual += espaco + largura
        linhas_totais += 1
        maior = max(maior, atual)
    return min(maior, CAIXA[0]), linhas_totais * 1.2 * tamanho


# texto curto: nem mexe no tamanho
assert p._ajusta_tamanho_fonte("ESPACO", CAIXA[0], CAIXA[1],
                               INICIAL, PISO, medida_falsa) == INICIAL
# texto absurdo: vai ao piso e PARA la' (sem loop infinito)
gigante = ("palavrapalavrapalavra " * 60 + "\n") * 30
assert p._ajusta_tamanho_fonte(gigante, CAIXA[0], CAIXA[1],
                               INICIAL, PISO, medida_falsa) == PISO
# os4 slides REAIS cabem na janela 900x900 acima do piso
for indice, slide in enumerate(p.slides_instrucao()):
    tamanho = p._ajusta_tamanho_fonte(slide, CAIXA[0], CAIXA[1],
                                      INICIAL, PISO, medida_falsa)
    larg, alt = medida_falsa(slide, tamanho)
    assert larg <= CAIXA[0] and alt <= CAIXA[1], \
        f"slide {indice} nao coube com {tamanho} pt ({larg:.0f}x{alt:.0f})"
    assert tamanho >= PISO
# e a mensagem LONGA da preparacao (veredito + tabela de 32 canais)
tabela = p._impedance_table(np.array([12.3] * 32), 50.0)
preparacao = ("PREPARACAO\n\n1) Eletrodos / impedancia (abaixo)\n\n"
              "ELETRODOS: 32/32 BONS (TODOS OK, <= 50 kOhm)\n\n"
              f"{tabela}\n\nESPACO inicia")
tam_prep = p._ajusta_tamanho_fonte(preparacao, CAIXA[0], CAIXA[1],
                                   INICIAL, PISO, medida_falsa)
larg, alt = medida_falsa(preparacao, tam_prep)
assert larg <= CAIXA[0] and alt <= CAIXA[1], (tam_prep, larg, alt)

# ---------------------------------------------------------------------------
# 2) _paint_text: o contrato da tela (wrap + encolhimento) e' fonte
# ---------------------------------------------------------------------------
src = inspect.getsource(p._StimulusCanvas._paint_text)
assert "Qt.TextWordWrap" in src, "_paint_text sem quebra de palavra"
assert "_ajusta_tamanho_fonte" in src, "_paint_text sem encolhimento"
# o countdown mantem a fonte fixa (numero curto; nao precisa de ajuste)
assert "420," in src


# ---------------------------------------------------------------------------
# 3) resolve_device: acha o aparelho ATRAVES do chain do gpype 4.x
# ---------------------------------------------------------------------------
class FakeDevice:
    """O que importa do gtec_gds.GNautilus para este resolvedor."""

    def get_impedance(self, first=True):
        return np.array([1.0, 2.0])

    def get_scaling(self):
        return ([1.0, 1.0], [0.0, 0.0])


class FakeCore:
    """_GNautilusCore: o aparelho mora DENTRO do no, nao no chain."""

    def __init__(self, device=None):
        self._device = device if device is not None else FakeDevice()


class FakeSync:
    pass                                       # Sync: no sem device


class FakeChain:
    """Mesma forma do OChain do ioiocore: internal_nodes, sem __getattr__."""

    def __init__(self, nodes):
        self.internal_nodes = list(nodes)


# a cadeia real: chain -> [core, Sync]; o chain NAO tem _device proprio
chain = FakeChain([FakeCore(), FakeSync()])
assert not hasattr(chain, "_device"), "o chain falso deve imitar o real"
assert p.resolve_device(chain) is chain.internal_nodes[0]._device
# device direto no no (caso antigo / mais simples) tambem funciona
direto = FakeCore()
assert p.resolve_device(direto) is direto._device
# sem aparelho nenhum: None (generator e' exatamente isso)
assert p.resolve_device(None) is None
assert p.resolve_device(FakeChain([FakeSync()])) is None
# device SEM get_impedance/get_scaling nao engana o resolvedor
class Pif:
    pass
assert p.resolve_device(FakeChain([FakeCore(Pif()), FakeSync()])) is None
# profundidade limitada: chain dentro de chain continua resolvendo
aninhado = FakeChain([FakeChain([FakeSync()]), FakeCore()])
assert p.resolve_device(aninhado) is not None
# generator REAL do gpype: sem _device em lugar nenhum -> None
import gpype as gp
gerador = gp.Generator(sampling_rate=500, channel_count=8, frame_size=1)
assert p.resolve_device(gerador) is None

# ---------------------------------------------------------------------------
# 4) Integracao: com o aparelho resolvido o painel MEDDE e PUBLICA
# ---------------------------------------------------------------------------
class AparelhoContador:
    def __init__(self):
        self.leituras = 0

    def get_impedance(self, _primeira):
        self.leituras += 1
        return np.array([1.5, 2.5, 99.0])


aparelho = AparelhoContador()
estado = p.SharedState()
monitor = p.ImpedanceMonitor(p.resolve_device(
    FakeChain([FakeCore(aparelho)])), estado, limite_kohm=50.0, period=0.01)
monitor.start()
limite = 300
while aparelho.leituras < 2 and limite:
    time.sleep(0.01)
    limite -= 1
monitor.stop()
assert not monitor.is_alive()
assert aparelho.leituras >= 2, aparelho.leituras
assert estado.impedance_values is not None, "a medida nao foi publicada"
bons = sum(1 for v in estado.impedance_values if 0 <= v <= 50.0)
assert bons == 2, bons                     # veredito da tela: 2/3 BONS

print("TRIAL_FEEDBACK_OK",
      f"slides={len(p.slides_instrucao())} cabem em 900x900",
      "| resolve_device achou o aparelho no core do chain",
      f"| {aparelho.leituras} leituras publicadas")
