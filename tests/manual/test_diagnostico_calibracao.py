"""Teste da auditoria OFFLINE da calibracao do overlay (sem hardware, sem Kinect).

Contexto (bancada, 30/09/2026): com o RMS stereo em 0,44-0,80 px -- DENTRO do
limite de 2 px -- o esqueleto da auxiliar aparecia dezenas de px fora da mao no
video do Kinect. `tools\\diagnostico_calibracao.py` existe para achar isso SEM
hardware e SEM acreditar no RMS. Este teste TRAVA o que faz o diagnostico valer:

  1. A GRADE espelhada: `stereo_calibration.py` (KINECT_EVAL_*) e o tool
     (JANELA_X/JANELA_Y/PASSOS_X/PASSOS_Y) tem de dar o MESMO ponto a ponto, e
     `compara_modelos_kinect` tem de dar o mesmo px que `espalhamento_px` -- os
     dois numeros sao comparados ainda contra o cv2, ponto por ponto. Os limites
     compartilhados (o px de suspeita, que vem da ferramenta da tecla T, e o RMS
     maximo do par) tambem sao conferidos contra a fonte.
  2. A MATEMATICA e' a do RUNTIME: `projeta_pela_profundidade` tem de devolver o
     MESMO pixel que `AuxCameraView.to_kinect_stereo` (o codigo que desenha o
     overlay), senao o diagnostico mente sobre o overlay que o usuario ve.
  3. As RECUSAS: arquivo velho (sem modelo), npz do modelo congelado (SO' o
     Kinect, sem rotacao/translacao: tem de ser COMPARADO, e nao ignorado -- era
     o bug que deixava o check 3 mudo), divergencia entre arquivos, o custo de
     1 cm de Z e a ordem dos arquivos (legado primeiro, e o npz do modelo
     congelado NAO pode entrar na lista de arquivos stereo).
  4. DOC x FERRAMENTA: o README tem de trazer o tool, os dois passos da bancada
     (`--salvar-intrinsecas-kinect` e a rotina) e as duas suites; o HISTORICO
     tem de registrar a sessao. (Convencao do projeto: ver
     `test_diagnostico_hotspot.py`.)

    .venv\\Scripts\\python.exe test_diagnostico_calibracao.py
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

import stereo_calibration as sc
import kinect_imu_groundtruth as gt

RAIZ = os.path.dirname(os.path.abspath(__file__))
# O tool mora em tools/ e importa-lo NAO abre camera nenhuma (o `main()` so' roda
# em `__main__`). `kinect_groundtruth_tool` entra so' para conferir o limite de px
# que a tecla T usa -- a duplicata existe para o tool ficar offline.
sys.path.insert(0, os.path.join(RAIZ, "tools"))

import diagnostico_calibracao as dc    # noqa: E402
import kinect_groundtruth_tool as tool  # noqa: E402

LARGURA_KINECT, ALTURA_KINECT = 1920, 1080
LARGURA_AUX, ALTURA_AUX = 1280, 720
KINECT_SIZE = (LARGURA_KINECT, ALTURA_KINECT)
AUX_SIZE = (LARGURA_AUX, ALTURA_AUX)

#: Modelo do Kinect do teste: fx/fy curtos de proposito (1000 px), para a grade
#: inteira da regiao de trabalho cair DENTRO do quadro -- e' o que permite
#: comparar `espalhamento_px` (que filtra os pontos fora) com
#: `compara_modelos_kinect` (que nao filtra) e esperar o MESMO numero.
K_KINECT = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
D_KINECT = np.array([0.05, -0.02, 0.001, -0.001, 0.0])
K_AUX = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
D_AUX = np.array([-0.10, 0.03, 0.0, 0.0, 0.0])

#: O bias medido na bancada pela tecla L: o modelo de uma sessao saiu com o
#: principal point (+32, +26) px fora do de outra. Virou esta constante porque o
#: desvio de px entre dois modelos do Kinect e' um DESLOCAMENTO CONSTANTE quando
#: so' o principal point muda -- da' para afirmar o numero exato.
BIAS_PRINCIPAL_PX = (32.0, 26.0)

#: Posicao FISICA da auxiliar no referencial do Kinect (m), do lado direito.
CZ_AUX = np.array([0.5, 0.0, 0.1])


def rot_y(graus):
    """Rotacao de `graus` em torno de Y (graus)."""
    c, s = np.cos(np.radians(graus)), np.sin(np.radians(graus))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# ARMADILHA da convencao do projeto (X_aux = R X_kin + T): o eixo otico da
# auxiliar no referencial do Kinect e' a TERCEIRA LINHA de R (R^T @ e3), nao a
# terceira coluna. Com rot_y(+30) a auxiliar aponta para a ESQUERDA-frente, que
# e' o que faz uma auxiliar da DIREITA enxergar o volume de trabalho.
R_AUX = rot_y(30.0)
T_AUX = -R_AUX @ CZ_AUX


def escreve_npz(caminho, matriz_k=K_KINECT, dist_k=D_KINECT, matriz_a=K_AUX,
                dist_a=D_AUX, rms=0.40, nome="", rotacao=R_AUX,
                translacao=T_AUX, completo=True):
    """npz stereo no formato que `carrega` le (ou no formato VELHO, sem modelo)."""
    if not completo:
        np.savez(caminho, rms=float(rms), camera_name=str(nome))
        return Path(caminho)
    np.savez(
        caminho,
        kinect_matrix=np.asarray(matriz_k, np.float64),
        kinect_dist=np.asarray(dist_k, np.float64),
        auxiliary_matrix=np.asarray(matriz_a, np.float64),
        auxiliary_dist=np.asarray(dist_a, np.float64),
        rotation=np.asarray(rotacao, np.float64),
        translation=np.asarray(translacao, np.float64).reshape(3),
        kinect_size=np.asarray(KINECT_SIZE), auxiliary_size=np.asarray(AUX_SIZE),
        rms=float(rms), kinect_rms=0.35, auxiliary_rms=0.45,
        refined_intrinsics=0.0,
        used_poses=np.ones(16), per_pose_rms=np.full(16, 0.30),
        camera_name=str(nome), checkerboard_size=np.asarray(sc.CHECKERBOARD_SIZE),
        square_size_m=float(sc.SQUARE_SIZE_M))
    return Path(caminho)


def escreve_modelo_congelado(caminho, matriz_k=K_KINECT, dist_k=D_KINECT,
                             rms=0.31, poses=12, span=0.42):
    """npz do MODELO CONGELADO no formato REAL que `--salvar-intrinsecas-kinect`
    grava: SO' o Kinect (sem rotacao/translacao, sem auxiliar). Copiado da
    `salva_intrinsecas_kinect` de proposito: se os campos mudarem la', o teste
    tem de acusar aqui."""
    np.savez(
        caminho, version=sc.KINECT_INTRINSICS_VERSION,
        kinect_matrix=np.asarray(matriz_k, np.float64),
        kinect_dist=np.asarray(dist_k, np.float64),
        kinect_size=np.asarray(KINECT_SIZE), rms=float(rms), poses=int(poses),
        span=float(span), checkerboard_size=np.asarray(sc.CHECKERBOARD_SIZE),
        square_size_m=float(sc.SQUARE_SIZE_M), frozen_at="2026-09-30 10:00:00")
    return Path(caminho)


def projeta_cv2(pontos, matriz, dist):
    """(N,2) px de pontos 3D do referencial DA camera (conta direta do cv2)."""
    px, _ = cv2.projectPoints(np.asarray(pontos, np.float64).reshape(-1, 1, 3),
                              np.zeros(3), np.zeros(3),
                              np.asarray(matriz, np.float64),
                              np.asarray(dist, np.float64).reshape(-1, 1))
    return px.reshape(-1, 2)


def vista_do_modelo(model):
    """`AuxCameraView` (o codigo que desenha o overlay) com os numeros do npz."""
    return gt.AuxCameraView(
        index=2, kinect_matrix=np.asarray(model["kinect_matrix"], np.float64),
        kinect_dist=np.asarray(model["kinect_dist"], np.float64),
        matrix=np.asarray(model["auxiliary_matrix"], np.float64),
        dist=np.asarray(model["auxiliary_dist"], np.float64),
        rotation=np.asarray(model["rotation"], np.float64),
        translation=np.asarray(model["translation"], np.float64).reshape(3, 1),
        rms=0.4, stereo_usable=True, runtime_size=AUX_SIZE,
        calibrated_size=AUX_SIZE)


def captura(funcao, *args, **kwargs):
    """Roda e devolve (resultado, texto impresso) -- o relatorio e' texto."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        resultado = funcao(*args, **kwargs)
    return resultado, buffer.getvalue()


def pontos_visiveis(model, profundidade):
    """(pontos 3D, px da auxiliar) da grade que a auxiliar enxerga com margem."""
    rotacao = np.asarray(model["rotation"], np.float64).reshape(3, 3)
    translacao = np.asarray(model["translation"], np.float64).reshape(3)
    pontos = dc.grade_da_regiao(profundidade)
    pontos_aux = (rotacao @ pontos.T).T + translacao
    px = projeta_cv2(pontos_aux, model["auxiliary_matrix"],
                     model["auxiliary_dist"])
    dentro = dc.pixels_dentro(px, model["auxiliary_size"])
    return pontos[dentro], px[dentro]


def bloco_grade_espelhada():
    """1) As constantes/grade dos DOIS arquivos tem de ser o mesmo numero."""
    assert dc.JANELA_X == sc.KINECT_EVAL_JANELA_X, (dc.JANELA_X,
                                                   sc.KINECT_EVAL_JANELA_X)
    assert dc.JANELA_Y == sc.KINECT_EVAL_JANELA_Y
    assert dc.PASSOS_X == sc.KINECT_EVAL_PASSOS_X
    assert dc.PASSOS_Y == sc.KINECT_EVAL_PASSOS_Y
    assert sc.KINECT_EVAL_PROFUNDIDADE_M in dc.PROFUNDIDADES_PADRAO
    for profundidade in dc.PROFUNDIDADES_PADRAO:
        assert np.array_equal(sc.grade_de_avaliacao(profundidade),
                              dc.grade_da_regiao(profundidade)), profundidade
    assert dc.grade_da_regiao(0.8).shape == (dc.PASSOS_Y * dc.PASSOS_X, 3)

    # Os limites DUPLICADOS: o do tool tem de ser o MESMO da ferramenta da tecla
    # T (quem grita na bancada) e o RMS, o mesmo do calibrador.
    assert dc.LIMITE_PX_CALIBRACAO_SUSPEITA == tool.LIMITE_PX_CALIBRACAO_SUSPEITA
    assert dc.MAX_STEREO_RMS_PX == sc.MAX_STEREO_RMS_PX
    assert dc.REFERENCIA_PADRAO == sc.KINECT_INTRINSICS_FILE.name
    assert "test_diagnostico_calibracao.py" in dc.__doc__

    # O MESMO px nos dois lados (e contra o cv2), com a grade toda no quadro.
    modelo_a = {"kinect_matrix": K_KINECT, "kinect_dist": D_KINECT,
                "kinect_size": KINECT_SIZE}
    deslocado = K_KINECT.copy()
    deslocado[0, 2] += BIAS_PRINCIPAL_PX[0]
    deslocado[1, 2] += BIAS_PRINCIPAL_PX[1]
    modelo_b = {"kinect_matrix": deslocado, "kinect_dist": D_KINECT,
                "kinect_size": KINECT_SIZE}
    pontos = dc.grade_da_regiao(0.8)
    px_a = projeta_cv2(pontos, K_KINECT, D_KINECT)
    px_b = projeta_cv2(pontos, deslocado, D_KINECT)
    assert dc.pixels_dentro(px_a, KINECT_SIZE).all(), "grade saindo do quadro"
    assert dc.pixels_dentro(px_b, KINECT_SIZE).all()
    esperado = float(np.median(np.linalg.norm(px_a - px_b, axis=1)))
    exato = float(np.hypot(*BIAS_PRINCIPAL_PX))
    assert abs(esperado - exato) < 1e-6, esperado
    mediana, pior = sc.compara_modelos_kinect(K_KINECT, D_KINECT, deslocado,
                                             D_KINECT)
    assert abs(mediana - esperado) < 1e-6, (mediana, esperado)
    assert abs(pior - exato) < 1e-6, pior
    resumo = dc.espalhamento_px(modelo_a, modelo_b, 0.8)
    assert resumo is not None and abs(resumo[0] - esperado) < 1e-6, resumo
    assert abs(resumo[2] - exato) < 1e-6, resumo
    assert dc.espalhamento_px(modelo_a, modelo_a, 0.8)[2] == 0.0
    print(f"1) GRADE/CONSTANTES: {len(pontos)} pontos identicos ponto a ponto; "
          f"bias de {esperado:.2f} px igual no calibrador, no tool e no cv2 | "
          f"limite {dc.LIMITE_PX_CALIBRACAO_SUSPEITA:g} px = o da tecla T")

def bloco_sensibilidade(pasta):
    """3) 1 cm de erro em Z: o tool concorda com o runtime e cai com a distancia."""
    modelo = dc.carrega(os.path.join(pasta, "stereo_calibration.npz"))
    vista = vista_do_modelo(modelo)
    perto = dc.sensibilidade_profundidade(modelo, 0.6)
    longe = dc.sensibilidade_profundidade(modelo, 1.3)
    assert perto is not None and longe is not None, (perto, longe)
    assert perto[0] > 5.0, perto          # 1 cm de Z custa px nesta geometria
    assert longe[0] < perto[0], (longe, perto)   # e cai com o QUADRADO de Z
    assert perto[2] >= perto[1] >= perto[0], perto   # mediana <= p90 <= max
    # O numero do tool contra o ORACULO (o proprio overlay): empurra o Z do
    # ponto da grade em 1 cm e mede o px entre as duas projecoes do runtime.
    pontos, pixels = pontos_visiveis(modelo, 0.6)
    por_cm = []
    for ponto, pixel in zip(pontos, pixels):
        alvo = np.asarray([pixel], np.float64)
        verdadeiro = vista.to_kinect_stereo(alvo, [float(ponto[2])])[0]
        empurrado = vista.to_kinect_stereo(
            alvo, [float(ponto[2]) + dc.DELTA_PROFUNDIDADE_M])[0]
        por_cm.append(float(np.linalg.norm(empurrado - verdadeiro))
                      / (dc.DELTA_PROFUNDIDADE_M * 100.0))
    oraculo = float(np.median(por_cm))
    assert abs(oraculo - perto[0]) < 0.05 * oraculo, (oraculo, perto)
    # O custo POR CM tem de ser o mesmo com um probe de 2 cm: o alerta em
    # "px por cm" so' e' honesto se a normalizacao valer. A curvatura do mapa
    # px(Z) aparece como alguns % (delta/z = 2 cm / 0,6 m = 3%).
    dobro = dc.sensibilidade_profundidade(modelo, 0.6, 2 * dc.DELTA_PROFUNDIDADE_M)
    assert abs(dobro[0] - perto[0]) < 0.05 * perto[0], (dobro[0], perto[0])
    assert abs(dobro[1] - perto[1]) < 0.05 * perto[1], (dobro[1], perto[1])
    # delta invalido nao inventa numero
    assert dc.sensibilidade_profundidade(modelo, 0.6, 0.0) is None
    print(f"3) PROFUNDIDADE: 1 cm de Z custa {perto[0]:.1f} px/cm a 0,6 m "
          f"(oraculo do overlay: {oraculo:.1f}) e {longe[0]:.1f} px/cm a 1,3 m | "
          f"{dc.LIMITE_PX_CALIBRACAO_SUSPEITA:g} px/cm e' o limite de alerta")


def bloco_matematica_do_runtime(pasta):
    """2) `projeta_pela_profundidade` tem de dar o pixel do `to_kinect_stereo`."""
    modelo = dc.carrega(os.path.join(pasta, "stereo_calibration.npz"))
    assert dc.pronto(modelo) and dc.pronto_kinect(modelo)
    vista = vista_do_modelo(modelo)
    pixels = np.array([[16.0, 16.0], [1270.0, 700.0], [640.0, 360.0],
                       [200.0, 650.0], [1100.0, 90.0]])
    pior = 0.0
    amostras = 0
    for profundidade in (0.6, 0.8, 1.3):
        esperado = vista.to_kinect_stereo(pixels, [profundidade])
        for pixel, alvo in zip(pixels, esperado):
            obtido = dc.projeta_pela_profundidade(modelo, pixel, profundidade)
            assert obtido is not None, (pixel, profundidade)
            assert np.isfinite(obtido).all(), obtido
            erro = float(np.linalg.norm(obtido - alvo))
            pior = max(pior, erro)
            amostras += 1
            # 1e-3 px e' folga para o float32 do runtime; o que importa e' que
            # nao exista diferenca de FORMULA (uma diferenca de formula seria de
            #zenas de px, nao de 1e-7)
            assert erro < 1e-3, (pixel, profundidade, obtido, alvo)
    # target_z == o Z do proprio ponto: o raio da auxiliar cai EM CIMA do ponto,
    # entao o deslocamento medido tem de ser ~0 e o px "verdadeiro" que o tool
    # devolve tem de ser o do cv2 (a ida-e-volta do diagnostico fecha).
    pontos, _ = pontos_visiveis(modelo, 0.8)
    assert len(pontos) > 0.5 * dc.PASSOS_X * dc.PASSOS_Y, len(pontos)
    pior_identidade = 0.0
    for ponto in pontos:
        erro, verdadeiro = dc.profundidade_para_px(modelo, ponto, 0.0)
        assert erro is not None, ponto
        pior_identidade = max(pior_identidade, erro)
        assert erro < 1e-4, (ponto, erro)
        assert np.allclose(
            verdadeiro,
            projeta_cv2([ponto], modelo["kinect_matrix"],
                        modelo["kinect_dist"])[0], atol=1e-6), ponto
    print(f"2) MATEMATICA DO RUNTIME: {amostras} pixels x 3 profundidades com "
          f"erro maximo {pior:.2e} px contra `to_kinect_stereo` | com delta_z=0 o "
          f"deslocamento e' {pior_identidade:.2e} px em {len(pontos)} pontos "
          f"({len(pontos)}/{dc.PASSOS_X * dc.PASSOS_Y} da grade vistos pela "
          "auxiliar)")


def bloco_arquivos_e_alertas(pasta):
    """4) O que o relatorio diz (e recusa) em cima de npz com formato de verdade."""
    limpo = escreve_npz(os.path.join(pasta, "stereo_calibration.npz"),
                        nome="externa-bancada")
    escreve_npz(os.path.join(pasta, "stereo_calibration_aux2.npz"),
                nome="externa-ampla")
    resultado, _ = captura(dc.auditoria, pasta)
    assert len(resultado["arquivos"]) == 2
    assert resultado["alertas"] == [], resultado["alertas"]
    assert resultado["referencia"] is None
    codigo, texto = captura(dc.main, ["--pasta", pasta])
    assert codigo == 0, codigo
    assert "Nada acima dos limites" in texto and "COMO CONSERTAR" not in texto
    assert "modelo congelado do Kinect: nao existe" in texto

    # O npz do MODELO CONGELADO no formato REAL (`salva_intrinsecas_kinect`): so'
    # o Kinect, SEM rotacao/translacao. Tem de ser COMPARADO -- exigir o par
    # inteiro deixava este check mudo, que era o bug.
    escreve_modelo_congelado(os.path.join(pasta, "kinect_intrinsics.npz"))
    resultado, _ = captura(dc.auditoria, pasta)
    congelado = resultado["referencia"]
    assert congelado is not None and os.path.basename(
        resultado["caminho_referencia"]) == dc.REFERENCIA_PADRAO
    assert dc.pronto(congelado) is False, "o npz congelado nao tem o PAR"
    assert dc.pronto_kinect(congelado) is True
    comparacoes = resultado["comparacoes_referencia"]
    assert len(comparacoes) == 2
    for nome, px in comparacoes.items():
        assert px is not None, nome          # <-- o bug: aqui dava None
        assert px[0] < 1e-6, (nome, px)
    assert resultado["alertas"] == [], resultado["alertas"]

    # Agora o modelo do Kinect de uma sessao com o bias medido na bancada
    # (+32,+26) px no principal point.
    deslocado = K_KINECT.copy()
    deslocado[0, 2] += BIAS_PRINCIPAL_PX[0]
    deslocado[1, 2] += BIAS_PRINCIPAL_PX[1]
    escreve_npz(os.path.join(pasta, "stereo_calibration_aux2.npz"),
                matriz_k=deslocado, nome="externa-ampla")
    resultado, _ = captura(dc.auditoria, pasta)
    esperado = float(np.hypot(*BIAS_PRINCIPAL_PX))
    assert set(par["profundidade"] for par in resultado["pares"]) == \
        set(dc.PROFUNDIDADES_PADRAO)
    for par in resultado["pares"]:
        assert abs(par["px"][0] - esperado) < 1e-6, par
    trocado = resultado["comparacoes_referencia"]["stereo_calibration_aux2.npz"]
    assert abs(trocado[0] - esperado) < 1e-6, trocado
    intacto = resultado["comparacoes_referencia"]["stereo_calibration.npz"]
    assert intacto[0] < 1e-6, intacto
    pares = [a for a in resultado["alertas"] if "divergem" in a]
    congelados = [a for a in resultado["alertas"] if "CONGELADO" in a]
    assert len(pares) == 1 and "aux2" in pares[0], pares
    assert len(congelados) == 1 and "aux2" in congelados[0], congelados
    assert "%.1f px" % esperado in pares[0], pares[0]
    codigo, texto = captura(dc.main, ["--pasta", pasta])
    assert codigo == 1, codigo
    assert "COMO CONSERTAR" in texto and "--salvar-intrinsecas-kinect" in texto
    assert dc.REFERENCIA_PADRAO in texto and "tecla T" in texto
    # e o relatorio identifica o modelo congelado (sem isso o operador nao sabe
    # se congelou o modelo certo)
    _, texto = captura(dc.relatorio, resultado)
    assert "modelo congelado do Kinect: SIM (kinect_intrinsics.npz)" in texto
    assert "RMS 0.31 px" in texto and "12 poses" in texto
    assert "42% da diagonal" in texto and "2026-09-30 10:00:00" in texto

    # --- ACEITACAO: "o arquivo existe" NAO e' "a calibracao vai usar o arquivo" -
    # MEDIDO em 01/10/2026: o diagnostico anunciava "modelo congelado do Kinect:
    # SIM" para um paliativo que `carrega_intrinsecas_kinect` RECUSAVA (gravado
    # com outra placa). O operador lia SIM e, por dentro, CADA sessao voltava a
    # re-estimar o Kinect -- dois npz da mesma bancada saíram 68,8 px um do outro.
    # O veredito agora vem da PROPRIA decisao da calibracao
    # (`stereo_calibration.aceita_modelo_kinect`), nunca de uma regra paralela.
    assert "ACEITA" in texto and "RECUSA" not in texto, texto
    assert not any("RECUSADO pela calibracao" in a for a in resultado["alertas"])
    assert dc.veredito_congelado(resultado["referencia"]) == (True, None)
    caminho_congelado = os.path.join(pasta, dc.REFERENCIA_PADRAO)
    # (a) span MEDIO abaixo do portao: recusado, citando o numero medido
    escreve_modelo_congelado(caminho_congelado, span=0.12)
    resultado_fraco, _ = captura(dc.auditoria, pasta)
    _, texto_fraco = captura(dc.relatorio, resultado_fraco)
    assert "RECUSA este arquivo" in texto_fraco, texto_fraco
    assert "12% da diagonal" in texto_fraco, texto_fraco
    assert any("RECUSADO pela calibracao" in a
               for a in resultado_fraco["alertas"]), resultado_fraco["alertas"]
    # (b) SEM poses/span: e' a cara do paliativo de 30/09/2026 (o arquivo que
    #     deixou a bancada sem modelo congelado nenhum)
    np.savez(caminho_congelado, kinect_matrix=np.asarray(K_KINECT, np.float64),
             kinect_dist=np.asarray(D_KINECT, np.float64),
             kinect_size=np.asarray(KINECT_SIZE), rms=0.19,
             frozen_at="2026-09-30 15:48 PALIATIVO")
    resultado_paliativo, _ = captura(dc.auditoria, pasta)
    _, texto_paliativo = captura(dc.relatorio, resultado_paliativo)
    assert "RECUSA este arquivo" in texto_paliativo, texto_paliativo
    assert "cobertura do quadro" in texto_paliativo, texto_paliativo
    assert dc.veredito_congelado(resultado_paliativo["referencia"])[0] is False
    # e o arquivo BOM volta ao lugar para o resto do teste
    escreve_modelo_congelado(caminho_congelado)

    # Arquivo VELHO (npz de antes do modelo do Kinect): auditavel, nunca um
    # traceback -- e nunca contaminando a comparacao dos arquivos bons.
    escreve_npz(os.path.join(pasta, "stereo_calibration_aux3.npz"),
                nome="velha", completo=False)
    # Arquivo completo, mas com o RMS do par acima do limite: o alerta tem de
    # mostrar o numero medido, nao "deu ruim".
    escreve_npz(os.path.join(pasta, "stereo_calibration_aux4.npz"), rms=3.0)
    resultado, _ = captura(dc.auditoria, pasta)
    velho = next(m for m in resultado["arquivos"] if "aux3" in m["nome"])
    assert velho["camera_name"] == "velha"
    assert abs(velho["rms"] - 0.40) < 1e-9 and velho["kinect_matrix"] is None
    assert dc.pronto(velho) is False and dc.pronto_kinect(velho) is False
    assert all(v is None for v in resultado["sensibilidade"][velho["nome"]].values())
    assert any("aux3" in a and "recalibre" in a for a in resultado["alertas"])
    assert any("aux4" in a and "RMS 3.00 px" in a for a in resultado["alertas"])
    assert resultado["comparacoes_referencia"][velho["nome"]] is None
    assert all(velho["nome"] not in (par["a"], par["b"])
               for par in resultado["pares"])
    assert (("stereo_calibration.npz", "stereo_calibration_aux4.npz")
            in {(par["a"], par["b"]) for par in resultado["pares"]})
    print(f"4) ARQUIVOS/ALERTAS: 2 no modelo congelado com 0 alertas -> bias de "
          f"{esperado:.1f} px vira 'divergem' + 'fora do CONGELADO' com exit 1 | "
          "npz velho e RMS 3,00 px acusados com o numero | o npz SO' Kinect foi "
          "comparado (era o bug)")


def bloco_ordem_dos_arquivos():
    """5) Ordem dos arquivos = a do calibrador; o modelo congelado fica FORA."""
    with tempfile.TemporaryDirectory(prefix="diag_cal_ordem_") as pasta:
        esperados = [sc.output_file(0).name, sc.output_file(2).name,
                     sc.output_file(5).name]
        for nome in esperados:
            escreve_npz(os.path.join(pasta, nome))
        escreve_modelo_congelado(os.path.join(pasta, dc.REFERENCIA_PADRAO))
        # Nomes que NAO sao stereo: nao podem entrar na auditoria (o glob)
        np.savez(os.path.join(pasta, "stereo_calibracao.npz"), rms=0.0)
        np.savez(os.path.join(pasta, "outro_stereo_calibration.npz"), rms=0.0)
        nomes = [os.path.basename(c) for c in dc.lista_arquivos(pasta)]
        assert nomes == esperados, nomes
        assert dc.lista_arquivos(os.path.join(pasta, "nao_existe")) == []
    print(f"5) ORDEM: {', '.join(esperados)} (legado primeiro, como o "
          "calibrador) | o npz do modelo congelado e os nomes parecidos ficam "
          "de fora")


def bloco_documentacao():
    """6) Doc x ferramenta: o que o README/historico prometem tem de existir."""
    def ler(nome):
        with open(os.path.join(RAIZ, nome), encoding="utf-8") as arquivo:
            return arquivo.read()

    readme = ler("README.md")
    historico = ler("HISTORICO_ITERACOES.txt")
    # O tool, as duas suites e os DOIS passos da bancada (congelar UMA vez +
    # rotina) tem de estar no README: sem isso o operador nao descobre o fluxo.
    assert "tools\\diagnostico_calibracao.py" in readme
    assert "--salvar-intrinsecas-kinect" in readme
    assert "test_diagnostico_calibracao.py" in readme
    assert "test_intrinsecas_kinect.py" in readme
    assert dc.REFERENCIA_PADRAO in readme
    # Os numeros que o tool usa tem de estar na prosa (o limite da tecla T).
    assert "%g px" % dc.LIMITE_PX_CALIBRACAO_SUSPEITA in readme, \
        dc.LIMITE_PX_CALIBRACAO_SUSPEITA
    assert "tecla `T`" in readme
    # e o historico registra a sessao (o projeto atualiza a cada sessao).
    assert "31. SESSAO 2026-09-30" in historico
    assert "kinect_intrinsics.npz" in historico
    print("6) DOC x FERRAMENTA: README (tool, os 2 passos da bancada, os limites "
          "e as duas suites) e o item 31 do historico em sincronia")


def main():
    with tempfile.TemporaryDirectory(prefix="diag_cal_") as pasta:
        escreve_npz(os.path.join(pasta, "stereo_calibration.npz"),
                    nome="externa-bancada")
        bloco_grade_espelhada()
        bloco_matematica_do_runtime(pasta)
        bloco_sensibilidade(pasta)
        bloco_arquivos_e_alertas(pasta)
        bloco_ordem_dos_arquivos()
        bloco_documentacao()
    print("DIAGNOSTICO_CALIBRACAO_OK: 6/6 blocos (grade espelhada travada ponto "
          "a ponto contra o cv2, projecao identica ao to_kinect_stereo, custo do "
          "Z, recusas/alertas, ordem dos arquivos e doc x ferramenta)")


if __name__ == "__main__":
    main()

