"""Teste das JANELAS de video: uma por camera, sem uma tapar a outra.

Contexto (bancada, 25/09/2026): na ferramenta do ground truth "nao aparecia o
video da terceira camera". Duas causas, as duas cobertas aqui:

  1. a janela da webcam NAO-ATIVA nao existia -- so' a camera ativa tinha janela
     (as outras apareciam apenas como esqueleto sobre o RGB do Kinect);
  2. o Windows abre todas as janelas na MESMA posicao, empilhadas -- mesmo as
     que existem ficam escondidas atras das outras. Pior: o WINDOW_SCALE de 0.5
     nao faz as 4 janelas caberem quando o Windows entrega uma area util menor
     que a resolucao do monitor (DPI > 100%; nesta maquina, 1536x960 num painel
     1920x1080 a 125%).

Casos:
 1. grade_de_janelas ....... grade sem sobreposicao, dentro da tela; None quando
                            nao cabe ou nao ha' tela;
 2. escala_para_grade ...... maior escala que faz TODAS caberem (0.5 em 1080p;
                            ~0.4 em 1536x960; nunca abaixo do minimo);
 3. posicoes_das_janelas ... modo "grade" / "cascata" (degraus, sem duas
                            janelas na mesma posicao) / "sem-tela";
 4. GradeDeJanelas ......... cria com WINDOW_NORMAL, redimensiona, posiciona,
                            reposiciona quando entra janela nova e destroi;
 5. ferramenta ............. uma janela de tracking POR camera auxiliar (o texto
                            da fonte cobre o caso -- ver docs/HISTORICO).

Sem hardware e sem GUI: o cv2 e' substituido por um fake que so' registra o que
foi pedido.

    .venv\\Scripts\\python.exe test_janelas_grade.py
"""
import os
import sys

import numpy as np

import kinect_imu_groundtruth as gt

RAIZ = os.path.dirname(os.path.abspath(__file__))
# A ferramenta vive em tools/ (o proprio modulo ajusta o sys.path; aqui e' so'
# para o IMPORT achar o arquivo). Importar NAO abre Kinect nem camera: o
# `main()` so' roda com `__main__` -- o teste usa so' o gerenciador de janelas.
sys.path.insert(0, os.path.join(RAIZ, "tools"))
import kinect_groundtruth_tool as tool  # noqa: E402

#: Tamanhos REAIS dos quadros (o que a ferramenta passa para a escala): RGB e
#: depth do Kinect v2 + duas webcams 1280x720.
CAIXAS_BASE = [(1920, 1080), (512, 424), (1280, 720), (1280, 720)]


def caixas_em(escala):
    return [(max(1, int(round(largura * escala))),
             max(1, int(round(altura * escala))))
            for largura, altura in CAIXAS_BASE]


def sem_sobreposicao(posicoes, caixas):
    """Confere que nenhum par de janelas se sobrepoe (nem na borda)."""
    for i, (x1, y1) in enumerate(posicoes):
        for j in range(i + 1, len(posicoes)):
            x2, y2 = posicoes[j]
            separadas = (x1 + caixas[i][0] <= x2 or x2 + caixas[j][0] <= x1
                         or y1 + caixas[i][1] <= y2
                         or y2 + caixas[j][1] <= y1)
            assert separadas, (i, j, posicoes[i], posicoes[j], caixas)


def dentro_da_tela(posicoes, caixas, largura, altura):
    for (x, y), (w, h) in zip(posicoes, caixas):
        assert x >= 0 and y >= 0, (x, y)
        assert x + w <= largura and y + h <= altura, (x, y, w, h, largura, altura)


# =============================================================================
# 1) grade_de_janelas
# =============================================================================
assert gt.grade_de_janelas([], 1920, 1080) is None
assert gt.grade_de_janelas([(640, 360)], 0, 0) is None, "sem tela: nao posiciona"
assert gt.grade_de_janelas([(640, 360)], 1920, 0) is None

# 1a) uma janela so': vai para o canto, com a margem documentada
simples = gt.grade_de_janelas([(640, 360)], 1920, 1080)
assert simples == [(0, gt.WINDOW_GRID_MARGIN)], simples

# 1b) janela mais larga que a tela -> None (nao da' para acomodar)
assert gt.grade_de_janelas([(2000, 300)], 1920, 1080) is None

# 1c) as QUATRO janelas reais em 1080p (escala 0.5): 2 colunas, sem sobreposicao
caixas = caixas_em(gt.WINDOW_SCALE)
posicoes = gt.grade_de_janelas(caixas, 1920, 1080)
assert posicoes is not None, (caixas, 1920, 1080)
sem_sobreposicao(posicoes, caixas)
dentro_da_tela(posicoes, caixas, 1920, 1080)
assert len(set(posicoes)) == 4, posicoes
assert posicoes[0] == (0, gt.WINDOW_GRID_MARGIN), posicoes

# 1d) uma tela pequenininha (nem a maior janela cabe em 2 colunas) -> None
assert gt.grade_de_janelas(caixas, 640, 480) is None

# 1e) empilhamento: com tela estreita e alta, tudo em UMA coluna
posicoes = gt.grade_de_janelas([(400, 200), (400, 200)], 500, 1000)
assert posicoes == [(0, 4), (0, 208)], posicoes
sem_sobreposicao(posicoes, [(400, 200), (400, 200)])

# 1f) o teto de colunas e' respeitado (colunas_max=1 -> so' empilha)
assert gt.grade_de_janelas(caixas, 1920, 1080, colunas_max=1) is None


# =============================================================================
# 2) escala_para_grade: a maior escala em que TODAS as janelas cabem
# =============================================================================
# 2a) 1080p: o padrao 0.5 ja' cabe -> nao mexe
assert gt.escala_para_grade(CAIXAS_BASE, 1920, 1080) == gt.WINDOW_SCALE

# 2b) a tela REAL desta maquina (DPI 125%): 0.5 nao cabe, cai para 0.4
escala = gt.escala_para_grade(CAIXAS_BASE, 1536, 960)
assert escala < gt.WINDOW_SCALE, escala
assert abs(escala - 0.40) < 1e-9, escala
caixas = caixas_em(escala)
posicoes = gt.grade_de_janelas(caixas, 1536, 960)
assert posicoes is not None
sem_sobreposicao(posicoes, caixas)
dentro_da_tela(posicoes, caixas, 1536, 960)

# 2c) sem tela conhecida -> devolve o teto (nao reduzir sem motivo)
assert gt.escala_para_grade(CAIXAS_BASE, 0, 0) == gt.WINDOW_SCALE

# 2d) tela minuscula -> nunca abaixo do minimo
assert gt.escala_para_grade(CAIXAS_BASE, 400, 300) == gt.WINDOW_SCALE_MIN
assert gt.escala_para_grade(CAIXAS_BASE, 512, 320) >= gt.WINDOW_SCALE_MIN

# 2e) a escala encontrada respeita os limites em varias telas
for tela in ((1536, 960), (1366, 768), (1920, 1080), (2560, 1440)):
    escala = gt.escala_para_grade(CAIXAS_BASE, *tela)
    assert gt.WINDOW_SCALE_MIN <= escala <= gt.WINDOW_SCALE, (tela, escala)
    caixas = caixas_em(escala)
    posicoes = gt.grade_de_janelas(caixas, *tela)
    if posicoes is None:                 # tela pequena: a ferramenta faz cascata
        assert tela in ((1366, 768), (1536, 960)), (tela, escala)
        continue
    sem_sobreposicao(posicoes, caixas)
    dentro_da_tela(posicoes, caixas, *tela)

# =============================================================================
# 3) posicoes_das_janelas: grade / cascata / sem-tela
# =============================================================================
caixas = caixas_em(gt.WINDOW_SCALE)
posicoes, modo = gt.posicoes_das_janelas(caixas, 1920, 1080)
assert modo == "grade", modo
sem_sobreposicao(posicoes, caixas)

# 3a) nao cabe -> CASCATA: posicoes distintas (nenhuma janela exatamente atras)
caixas_grandes = caixas_em(1.0)
posicoes, modo = gt.posicoes_das_janelas(caixas_grandes, 800, 600)
assert modo == "cascata", modo
assert len(set(posicoes)) == len(posicoes), posicoes
assert all(0 <= x < 800 and 0 <= y < 600 for x, y in posicoes), posicoes

# 3b) sem tela -> None e modo "sem-tela" (o Windows posiciona)
posicoes, modo = gt.posicoes_das_janelas(caixas, 0, 0)
assert posicoes is None and modo == "sem-tela", (posicoes, modo)
assert gt.posicoes_das_janelas([], 1920, 1080)[1] == "sem-tela"

# 3c) tamanho real da tela desta maquina (nunca (0, 0) no teste: seria Windows
#     sem user32, o que nao e' o caso)
tela = gt.tamanho_da_tela()
assert isinstance(tela, tuple) and len(tela) == 2

# =============================================================================
# 4) GradeDeJanelas: cria, posiciona, reposiciona e destroi (com cv2 falso)
# =============================================================================
class _Cv2Fake:
    """cv2 minimo: registra janelas/tamanhos/posicoes (sem GUI, sem camera)."""

    WINDOW_NORMAL = 0

    def __init__(self):
        self.flags = {}          # titulo -> flags do namedWindow
        self.tamanhos = {}       # titulo -> (w, h) pedidos
        self.posicoes = {}       # titulo -> (x, y)
        self.imagens = {}        # titulo -> (w, h) da ultima imagem mostrada
        self.destruidas = []

    def namedWindow(self, titulo, flags=0):
        self.flags[titulo] = flags

    def resizeWindow(self, titulo, largura, altura):
        self.tamanhos[titulo] = (int(largura), int(altura))

    def moveWindow(self, titulo, x, y):
        self.posicoes[titulo] = (int(x), int(y))

    def imshow(self, titulo, imagem):
        self.imagens[titulo] = (int(imagem.shape[1]), int(imagem.shape[0]))

    def destroyWindow(self, titulo):
        self.destruidas.append(titulo)

    def resize(self, imagem, _dst, fx=1.0, fy=1.0):
        return np.zeros((max(1, int(round(imagem.shape[0] * fy))),
                         max(1, int(round(imagem.shape[1] * fx))), 3),
                        dtype=np.uint8)


cv2_fake = _Cv2Fake()
avisos = []
janelas = tool.GradeDeJanelas(cv2_fake, escala=0.5, tela=(1920, 1080),
                              avisar=avisos.append)
assert janelas.escala == 0.5 and janelas.tela == (1920, 1080)

# 4a) as quatro janelas do dia a dia entram em grade, sem sobreposicao
quadros = {
    "Kinect RGB + esqueleto + IMU": (1920, 1080),
    "Kinect depth": (512, 424),
    "Camera auxiliar 2 - tracking": (1280, 720),
    "Camera auxiliar 0 - tracking": (1280, 720),
}
for titulo, (largura, altura) in quadros.items():
    janelas.mostrar(titulo, janelas.redimensionar(
        np.zeros((altura, largura, 3), dtype=np.uint8)))
assert set(cv2_fake.flags) == set(quadros), cv2_fake.flags
assert all(flag == cv2_fake.WINDOW_NORMAL for flag in cv2_fake.flags.values())
assert janelas.modo == "grade", janelas.modo
assert len(janelas.posicoes) == 4 and len(set(janelas.posicoes.values())) == 4

# 4b) a imagem mostrada e' a REDUZIDA (escala 0.5)
assert cv2_fake.imagens["Kinect RGB + esqueleto + IMU"] == (960, 540)
assert cv2_fake.imagens["Camera auxiliar 2 - tracking"] == (640, 360)
assert cv2_fake.tamanhos["Camera auxiliar 0 - tracking"] == (640, 360)
assert any("grade" in aviso for aviso in avisos), avisos

# 4c) janela nova (ex.: calibracao, tecla T) reposiciona TODAS (nao empilha)
antes = dict(janelas.posicoes)
janelas.mostrar("Calibracao 2 - Kinect",
                janelas.redimensionar(np.zeros((1080, 1920, 3), np.uint8)))
assert set(janelas.posicoes) == set(quadros) | {"Calibracao 2 - Kinect"}
assert janelas.posicoes != antes, "a grade tinha de ser refeita"
assert len(set(janelas.posicoes.values())) == len(janelas.posicoes)

# 4d) quadro None nao estoura o redimensionamento
assert janelas.redimensionar(None) is None

# 4e) destruir fecha exatamente o que este objeto criou
janelas.destruir()
assert set(cv2_fake.destruidas) == set(quadros) | {"Calibracao 2 - Kinect"}
assert not janelas.tamanhos and not janelas.posicoes

# 4f) sem tela conhecida: avisa UMA vez e nao move nada
avisos.clear()
cv2_fake2 = _Cv2Fake()
sem_tela = tool.GradeDeJanelas(cv2_fake2, escala=0.5, tela=(0, 0),
                               avisar=avisos.append)
sem_tela.mostrar("Kinect depth",
                 sem_tela.redimensionar(np.zeros((424, 512, 3), np.uint8)))
sem_tela.mostrar("Camera auxiliar 2 - tracking",
                 sem_tela.redimensionar(np.zeros((720, 1280, 3), np.uint8)))
assert sem_tela.modo == "sem-tela" and not cv2_fake2.posicoes
assert sum("tamanho da tela" in aviso for aviso in avisos) == 1, avisos

# 4g) grade que NAO cabe: cascata + aviso apontando --escala-janelas
avisos.clear()
cv2_fake3 = _Cv2Fake()
apertado = tool.GradeDeJanelas(cv2_fake3, escala=1.0, tela=(800, 600),
                               avisar=avisos.append)
for indice in range(3):
    apertado.mostrar(f"Camera auxiliar {indice} - tracking",
                     apertado.redimensionar(np.zeros((1080, 1920, 3), np.uint8)))
assert apertado.modo == "cascata", apertado.modo
assert len(set(apertado.posicoes.values())) == 3
assert any("--escala-janelas" in aviso for aviso in avisos), avisos

# =============================================================================
# 5) A FERRAMENTA abre uma janela de video POR camera auxiliar
# =============================================================================
fonte = open(os.path.join(RAIZ, "tools", "kinect_groundtruth_tool.py"),
             encoding="utf-8").read()
assert 'f"Camera auxiliar {indice_camera} - tracking"' in fonte, \
    "cada auxiliar precisa da SUA janela de video"
assert "escala_para_grade" in fonte and "--escala-janelas" in fonte
assert "janelas.redimensionar" in fonte
assert "cv2.imshow(" not in fonte.replace("self.cv2.imshow", ""), \
    "todo imshow da ferramenta passa pela GradeDeJanelas"

print("JANELAS_GRADE_OK: 5/5 grupos (grade sem sobreposicao, escala "
      "automatica, cascata, GradeDeJanelas com cv2 falso, uma janela por "
      "camera na ferramenta)")
sys.exit(0)

assert all(isinstance(valor, int) for valor in tela)
