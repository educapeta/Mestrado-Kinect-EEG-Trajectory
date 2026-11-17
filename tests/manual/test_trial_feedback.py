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

3. (item 59) ERRO REAL ENGOLIDO: mesmo com o aparelho resolvido, o
   `get_impedance` podia falhar (ex.: CRC transitrio do link sem fio) e
   o `_medir` antigo ENGOLIA a excecao e re-caia no raise "sem aparelho
   direto nem API do g.Pype" -- que o `run` trata como PERMANENTE: uma
   unica falha transitria matava o painel com "(indisponivel: sem
   aparelho direto nem API do g.Pype)". Agora os DOIS caminhos (direto e
   g.Pype) capturam o erro com tipo+mensagem, o `_medir` PROPAGA o erro
   real quando os caminhos existem mas falharam, e o `run` RE-TENTA
   mostrando "(falha na medicao: <erro real>)"; so' a ausencia dos dois
   caminhos e' permanente (a thread sai sozinha).
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

# ---------------------------------------------------------------------------
# 5) Falha TRANSITORIA (CRC do link): o painel NAO morre e mostra o erro
#    REAL. O _medir antigo ENGOLIA a excecao do aparelho e re-caia no raise
#    "sem aparelho direto nem API do g.Pype", que o run trata como
#    PERMANENTE -- uma unica falha de CRC matava o painel (item 59).
# ---------------------------------------------------------------------------
class AparelhoQueFalha:
    """GetImpedanceEx levanta CRC as 1a e 2a vez; na 3a funciona."""

    def __init__(self):
        self.tentativas = 0

    def get_impedance(self, _primeira):
        self.tentativas += 1
        if self.tentativas <= 2:
            raise RuntimeError("Invalid CRC checksum")
        return np.array([3.0, 4.0])


falho = AparelhoQueFalha()
estado2 = p.SharedState()
mon2 = p.ImpedanceMonitor(falho, estado2, limite_kohm=50.0, period=0.01)
mon2.start()
limite = 400
while falho.tentativas < 3 and limite:
    time.sleep(0.01)
    limite -= 1
mon2.stop()
assert not mon2.is_alive()
assert falho.tentativas >= 3, \
    f"o painel morreu na 1a falha ({falho.tentativas} tentativas)"
assert estado2.impedance_values is not None, \
    "depois de re-tentar, a medida tem de publicar"
# a mensagem de falha INTERMEDIARIA (se apareceu) carrega o erro REAL
texto2 = estado2.impedance_text or ""
assert "CRC" in texto2 or estado2.impedance_values is not None, texto2

# e a ausencia TOTAL de caminhos continua PERMANENTE (sai da thread)
mon3 = p.ImpedanceMonitor(None, p.SharedState(), period=0.01)
mon3.start()
limite = 300
while mon3.is_alive() and limite:
    time.sleep(0.01)
    limite -= 1
assert not mon3.is_alive(), "sem aparelho a thread tem de sair sozinho"

# o _medir propaga o erro REAL (tipo + mensagem), nao o generico
mon4 = p.ImpedanceMonitor(AparelhoQueFalha(), p.SharedState())
mon4._primeiro = True
try:
    mon4._medir()
    raise AssertionError("devia ter falhado na 1a leitura")
except RuntimeError as exc:
    assert "Invalid CRC checksum" in str(exc), str(exc)
    assert "aparelho direto" in str(exc), str(exc)

print("TRIAL_FEEDBACK_OK",
      f"slides={len(p.slides_instrucao())} cabem em 900x900",
      "| resolve_device achou o aparelho no core do chain",
      f"| {aparelho.leituras} leituras publicadas",
      f"| transitorio re-tentou {falho.tentativas}x e publicou")
