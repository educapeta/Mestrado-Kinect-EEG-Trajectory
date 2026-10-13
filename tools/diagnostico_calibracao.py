"""Auditoria OFFLINE da calibracao do overlay (auxiliar -> RGB do Kinect).

Motivo (bancada de 30/09/2026): com o RMS stereo em 0,44/0,80 px -- bem dentro do
limite de 2 px -- o esqueleto da auxiliar continuava dezenas de px fora da mao no
video do Kinect. O RMS NAO mede isso: ele so' mede a coerencia INTERNA do npz (o
mesmo erro sistematico aparece nos dois lados da conta e cancela). Este script
separa, SEM HARDWARE, as tres causas que sobram, com numero em px:

1. INTRINSECAS DO KINECT RE-ESTIMADAS. O Kinect RGB e' UMA camera: um modelo
   (fx/fy/cx/cy + distorcao) vale para todas as sessoes. Quando cada npz traz um
   modelo diferente, a projecao do MESMO ponto 3D muda de arquivo para arquivo --
   e essa diferenca e' erro sistematico do overlay (cresce com a distancia ao
   centro da imagem). Foi isso que a tecla L mediu em 28/09/2026: bias constante
   (+32,+26 px), de sinal estavel, crescendo com a excentricidade e explodindo
   fora do quadro.
2. PROFUNDIDADE. O caminho de projecao reconstroi o ponto 3D no raio da auxiliar
   cortado pelo plano Z medido (Kinect). Cada cm de erro em Z desloca o ponto na
   imagem do Kinect; quanto, depende da geometria (baseline e distancia) -- este
   script mede em px/cm na regiao onde a mao realmente anda.
3. CALIBRACAO GRAVADA x MODELO CONGELADO. Quando existe `kinect_intrinsics.npz`
   (o modelo do Kinect validado; ver `stereo_calibration.py
   --salvar-intrinsecas-kinect`), compara CADA npz com ele: e' o numero que diz
   se aquele arquivo vai projetar certo nesta bancada. O npz do modelo congelado
   tem SO' o modelo do Kinect (uma camera, sem rotacao/translacao) -- a
   comparacao usa o modelo do Kinect dos dois lados, nunca o par inteiro.

Uso (da raiz do projeto):
    .venv\\Scripts\\python.exe tools\\diagnostico_calibracao.py
    .venv\\Scripts\\python.exe tools\\diagnostico_calibracao.py --profundidades 0.7 1.0
    .venv\\Scripts\\python.exe tools\\diagnostico_calibracao.py --referencia outro.npz

Codigo de saida 0 = nada acima dos limites; 1 = ha' achado para consertar antes de
confiar no overlay. Regressao: test_diagnostico_calibracao.py.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Espelha LIMITE_PX_CALIBRACAO_SUSPEITA de tools/kinect_groundtruth_tool.py (o
#: numero que a tecla T usa para gritar "RECALIBRE nesta maquina"). Duplicado de
#: proposito: este script e' OFFLINE e nao pode importar a ferramenta (que abre
#: Kinect). Se um dos dois mudar, mude o outro.
LIMITE_PX_CALIBRACAO_SUSPEITA = 25.0
#: Espelha MAX_STEREO_RMS_PX de stereo_calibration.py.
MAX_STEREO_RMS_PX = 2.0
#: Profundidades (m) avaliadas na regiao de trabalho (punho na mesa -> mao
#: levantada). 0.6 m e' onde o erro de Z mais pesa: a sensibilidade cai com o
#: quadrado da distancia.
PROFUNDIDADES_PADRAO = (0.6, 0.8, 1.0, 1.3)
#: Regiao de trabalho da mao no referencial do Kinect (m): X para a direita, Y
#: para cima, Z para frente.
JANELA_X = (-0.35, 0.35)
JANELA_Y = (-0.30, 0.30)
PASSOS_X = 9
PASSOS_Y = 7
#: Margem (px) para considerar um ponto OBSERVAVEL por uma camera: dentro dela a
#: matematica da projecao diz algo; na borda/fora do quadro, nada.
MARGEM_PX = 8.0
#: Modelo congelado do Kinect (ver stereo_calibration.py).
REFERENCIA_PADRAO = "kinect_intrinsics.npz"
#: Perturbacao de Z usada para medir a sensibilidade (1 cm).
DELTA_PROFUNDIDADE_M = 0.01


def lista_arquivos(pasta=RAIZ):
    """npz stereo da pasta, com o LEGADO (nome sem sufixo) primeiro.

    Mesma ordem de `output_file`/`aux_stereo_file`: o indice 0 usa o nome legado
    (`stereo_calibration.npz`), os outros `stereo_calibration_aux{N}.npz`.
    """
    def chave(caminho):
        sufixo = (os.path.basename(caminho)
                  .replace("stereo_calibration", "")
                  .replace(".npz", ""))
        return (0 if not sufixo else 1, sufixo)

    return sorted(glob.glob(os.path.join(pasta, "stereo_calibration*.npz")),
                  key=chave)


def carrega(caminho):
    """Campos do npz que o overlay usa (None quando o arquivo nao tem o campo).

    Nunca levanta por campo faltando: npz de versao antiga continua auditavel --
    o que falta vira `None` e o relatorio diz o que nao deu para conferir.
    """
    dados = np.load(caminho, allow_pickle=True)

    def campo(nome, formato=None):
        if nome not in dados.files:
            return None
        valor = np.asarray(dados[nome], np.float64)
        return valor.reshape(formato) if formato is not None else valor

    def escalar(nome):
        if nome not in dados.files:
            return None
        valor = np.asarray(dados[nome]).reshape(-1)
        return float(valor[0]) if valor.size else None

    tamanho = campo("kinect_size")
    tamanho_aux = campo("auxiliary_size")
    poses = campo("used_poses")
    per_pose = campo("per_pose_rms")
    nome_camera = (str(np.asarray(dados["camera_name"]).reshape(-1)[0]).strip()
                   if "camera_name" in dados.files else "")
    return {
        "caminho": caminho,
        "nome": os.path.basename(caminho),
        "camera_name": nome_camera,
        "rms": escalar("rms"),
        "kinect_rms": escalar("kinect_rms"),
        "auxiliary_rms": escalar("auxiliary_rms"),
        "refined_intrinsics": campo("refined_intrinsics"),
        "kinect_matrix": campo("kinect_matrix", (3, 3)),
        "kinect_dist": campo("kinect_dist", (-1,)),
        "auxiliary_matrix": campo("auxiliary_matrix", (3, 3)),
        "auxiliary_dist": campo("auxiliary_dist", (-1,)),
        "rotation": campo("rotation", (3, 3)),
        "translation": campo("translation", (3,)),
        "kinect_size": (tuple(int(v) for v in tamanho.reshape(-1))
                        if tamanho is not None else (1920, 1080)),
        "auxiliary_size": (tuple(int(v) for v in tamanho_aux.reshape(-1))
                           if tamanho_aux is not None else (1280, 720)),
        "poses": (int(poses.size) if poses is not None else None),
        "per_pose_max": (float(np.max(per_pose))
                         if per_pose is not None and per_pose.size else None),
        "board_size": (tuple(int(v) for v in campo("checkerboard_size").reshape(-1))
                       if "checkerboard_size" in dados.files else None),
        "square_size_m": escalar("square_size_m"),
        # Campos do modelo CONGELADO do Kinect (`kinect_intrinsics.npz`) -- o
        # relatorio os mostra para o operador conferir que congelou o modelo
        # certo (RMS, poses e cobertura do quadro do tabuleiro). Prefixo
        # "frozen_" porque `poses`/`rms` daqui NAO sao os do par: no arquivo
        # congelado `poses` e' o numero de poses da sessao que congelou.
        "frozen_poses": escalar("poses"),
        "frozen_span": escalar("span"),
        "frozen_at": (str(np.asarray(dados["frozen_at"]).reshape(-1)[0])
                      if "frozen_at" in dados.files else ""),
    }


def pronto(model):
    """True quando o npz tem TUDO que a projecao exige (senao nao da' para auditar)."""
    return all(model.get(chave) is not None for chave in (
        "kinect_matrix", "kinect_dist", "auxiliary_matrix", "auxiliary_dist",
        "rotation", "translation"))


def pronto_kinect(model):
    """True quando o npz traz o MODELO DO KINECT (tudo o que compara-lo exige).

    `pronto()` pede o par inteiro (matriz E rotacao/translacao); o modelo
    congelado do Kinect (`kinect_intrinsics.npz`, gravado por
    `stereo_calibration.py --salvar-intrinsecas-kinect`) descreve UMA camera so'
    e NAO tem rotacao/translacao. Exigir `pronto()` para compara-lo deixava o
    check 3 mudo justamente em cima do arquivo real -- foi o bug que o
    test_diagnostico_calibracao.py pegou (regressao: npz no formato REAL do
    modelo congelado tem de ser comparado, nao ignorado).
    """
    return all(model.get(chave) is not None for chave in
               ("kinect_matrix", "kinect_dist", "kinect_size"))


def veredito_congelado(model):
    """(ok|None, motivo) da LEITURA da calibracao para este modelo congelado.

    Import TARDIO de `stereo_calibration` DE PROPOSITO: a decisao (campos
    obrigatorios + os tres portoes) mora num lugar so', e este arquivo tem de rodar
    OFFLINE. Sem ele (maquina sem o SDK do Kinect), devolve (None, motivo) e o
    relatorio diz que nao conseguiu conferir -- NUNCA "SIM" calado.

    Por que isto existe (MEDIDO em 01/10/2026): o diagnostico anunciava "modelo
    congelado do Kinect: SIM" para um arquivo que `carrega_intrinsecas_kinect`
    RECUSAVA (ele era de outra placa). O operador lia "SIM" e o que acontecia por
    dentro era cada sessao RE-ESTIMANDO o Kinect -- dois npz da mesma bancada
    saíram 68,8 px um do outro.
    """
    try:
        #: `RAIZ` no sys.path: rodado como `python tools\\diagnostico_calibracao.py`
        #: o sys.path[0] e' `tools\\`, e sem isto o import falha -- e FALHA SO' NA
        #: MAQUINA DE VERDADE, porque o teste (`test_diagnostico_calibracao.py`)
        #: roda da raiz e ja' tem a raiz no path. MEDIDO aqui, na primeira
        #: execucao: "nao consegui conferir a aceitacao".
        if RAIZ not in sys.path:
            sys.path.insert(0, RAIZ)
        import stereo_calibration as sc
    except Exception as exc:               # noqa: BLE001 - ferramenta offline
        return None, "nao consegui importar stereo_calibration (%s)" % exc
    return sc.aceita_modelo_kinect(model.get("frozen_poses"),
                                   model.get("frozen_span"), model.get("rms"))


def posicao_da_camera(model):
    """((3,) posicao no referencial do Kinect, distancia em m) ou (None, None).

    X_aux = R X_kin + T => o centro da auxiliar e' X_kin = -R^T T. Serve para
    conferir o LAYOUT fisico (duas webcams de cada lado do Kinect, 0,6-1,2 m) sem
    acreditar no npz por ele mesmo.
    """
    if model.get("rotation") is None or model.get("translation") is None:
        return None, None
    rotacao = np.asarray(model["rotation"], np.float64).reshape(3, 3)
    translacao = np.asarray(model["translation"], np.float64).reshape(3)
    posicao = -rotacao.T @ translacao
    return posicao, float(np.linalg.norm(translacao))


def grade_da_regiao(profundidade, janela_x=JANELA_X, janela_y=JANELA_Y,
                    passos_x=PASSOS_X, passos_y=PASSOS_Y):
    """(N,3) pontos 3D no referencial do Kinect (m), na regiao de trabalho."""
    xs = np.linspace(janela_x[0], janela_x[1], passos_x)
    ys = np.linspace(janela_y[0], janela_y[1], passos_y)
    z = float(profundidade)
    return np.array([[x, y, z] for y in ys for x in xs], np.float64)


def projeta(pontos, matriz, distorcao):
    """(N,2) px de `pontos` (N,3, referencial DA camera) na camera dada."""
    pontos = np.asarray(pontos, np.float64).reshape(-1, 1, 3)
    px, _ = cv2.projectPoints(
        pontos, np.zeros(3), np.zeros(3),
        np.asarray(matriz, np.float64),
        np.asarray(distorcao, np.float64).reshape(-1, 1))
    return px.reshape(-1, 2)


def pixels_dentro(px, tamanho, margem=MARGEM_PX):
    """(N,) bool: pixel finito e dentro do frame com margem."""
    largura, altura = tamanho
    px = np.asarray(px, np.float64).reshape(-1, 2)
    return ((px[:, 0] > margem) & (px[:, 0] < largura - margem)
            & (px[:, 1] > margem) & (px[:, 1] < altura - margem)
            & np.isfinite(px).all(axis=1))


def projeta_pela_profundidade(model, px_aux, target_z):
    """Pixel do Kinect do ponto reconstruido no raio da auxiliar com Z = target_z.

    Reproduz a matematica do runtime (`AuxCameraView.to_kinect_stereo`, a que
    `CameraAlignment.auxiliary_to_kinect_stereo` delega): pixel da auxiliar ->
    raio normalizado (undistortPoints) -> intersecao do raio com o plano Z usado
    -> reprojecao no RGB do Kinect. Qualquer divergencia aqui torna este
    diagnostico mentiroso em relacao ao overlay, por isso a matematica e' copiada,
    nao reinventada (e o teste confere: com target_z == Z do ponto, o erro tem de
    ser ~0).
    """
    if not pronto(model):
        return None
    matriz_aux = np.asarray(model["auxiliary_matrix"], np.float64)
    dist_aux = np.asarray(model["auxiliary_dist"], np.float64).reshape(-1, 1)
    rotacao = np.asarray(model["rotation"], np.float64).reshape(3, 3)
    translacao = np.asarray(model["translation"], np.float64).reshape(3)
    pixel = np.asarray(px_aux, np.float64).reshape(1, 1, 2).astype(np.float32)
    raio = cv2.undistortPoints(pixel, matriz_aux, dist_aux).reshape(2)
    direcao = rotacao.T @ np.array([raio[0], raio[1], 1.0])
    origem = -rotacao.T @ translacao
    if float(target_z) <= 0.0 or abs(float(direcao[2])) < 1e-6:
        return None
    escala = (float(target_z) - origem[2]) / direcao[2]
    ponto = rotacao.T @ (escala * np.array([raio[0], raio[1], 1.0]) - translacao)
    return projeta(ponto.reshape(1, 3), model["kinect_matrix"],
                   model["kinect_dist"])[0]


def profundidade_para_px(model, ponto, delta_z):
    """(px deslocados por `delta_z` de erro em Z, pixel verdadeiro) ou (None, None).

    O ponto e' dado no referencial do KINECT (m). Ele so' entra na conta se a
    auxiliar o enxerga com margem -- nos pontos fora do quadro dela nao ha'
    overlay possivel, e a matematica ali daria numero grande e mentiroso.
    """
    if not pronto(model):
        return None, None
    ponto = np.asarray(ponto, np.float64).reshape(3)
    rotacao = np.asarray(model["rotation"], np.float64).reshape(3, 3)
    translacao = np.asarray(model["translation"], np.float64).reshape(3)
    ponto_aux = rotacao @ ponto + translacao
    if ponto_aux[2] <= 0.0:
        return None, None
    px_aux = projeta(ponto_aux.reshape(1, 3), model["auxiliary_matrix"],
                     model["auxiliary_dist"])[0]
    if not pixels_dentro(px_aux, model["auxiliary_size"])[0]:
        return None, None
    verdadeiro = projeta(ponto.reshape(1, 3), model["kinect_matrix"],
                         model["kinect_dist"])[0]
    deslocado = projeta_pela_profundidade(model, px_aux,
                                          float(ponto[2]) + float(delta_z))
    if deslocado is None:
        return None, None
    return float(np.linalg.norm(deslocado - verdadeiro)), verdadeiro


def _resumo(valores):
    """(mediana, p90, max) de uma lista, ou None se ela estiver vazia."""
    valores = np.asarray([v for v in valores if np.isfinite(v)], np.float64)
    if valores.size == 0:
        return None
    return (float(np.median(valores)), float(np.percentile(valores, 90)),
            float(valores.max()))


def sensibilidade_profundidade(model, profundidade,
                               delta_m=DELTA_PROFUNDIDADE_M):
    """(mediana, p90, max) px por CM de erro em Z, na regiao de trabalho.

    O que isto responde: "quanto custa o meu Z medido estar 1 cm fora?". Numa
    bancada com baseline grande e mao perto (0,6-0,8 m) custa dezenas de px por
    cm -- mais do que qualquer residual de calibracao. None quando a auxiliar nao
    enxerga nenhum ponto da regiao nessa profundidade.
    """
    if delta_m <= 0 or not pronto(model):
        return None
    pontos = grade_da_regiao(profundidade)
    por_cm = []
    for ponto in pontos:
        px, _ = profundidade_para_px(model, ponto, delta_m)
        if px is not None:
            por_cm.append(px / (delta_m * 100.0))
    return _resumo(por_cm)


def espalhamento_px(model_a, model_b, profundidade):
    """(mediana, p90, max) px entre os modelos do KINECT de duas calibracoes.

    O Kinect RGB e' UMA camera so': dois npz gravados na MESMA bancada tem de
    projetar o mesmo ponto 3D no mesmo pixel. O que sobra e' erro sistematico do
    overlay -- e nao aparece no RMS de nenhum dos dois arquivos, porque cada um
    mede so' a coerencia interna dele.

    So' exige o modelo do KINECT de cada lado (`pronto_kinect`): assim funciona
    tambem contra `kinect_intrinsics.npz`, que nao tem rotacao/translacao.
    """
    if not (pronto_kinect(model_a) and pronto_kinect(model_b)):
        return None
    pontos = grade_da_regiao(profundidade)
    px_a = projeta(pontos, model_a["kinect_matrix"], model_a["kinect_dist"])
    px_b = projeta(pontos, model_b["kinect_matrix"], model_b["kinect_dist"])
    dentro = (pixels_dentro(px_a, model_a["kinect_size"])
              & pixels_dentro(px_b, model_b["kinect_size"]))
    if not dentro.any():
        return None
    diferencas = np.linalg.norm(px_a[dentro] - px_b[dentro], axis=1)
    return _resumo(diferencas)


def auditoria(pasta=RAIZ, profundidades=PROFUNDIDADES_PADRAO, referencia=None):
    """Roda os tres checks e devolve tudo estruturado (SEM imprimir).

    E' o `main` que formata; a funcao existe separada para o teste poder afirmar
    sobre o resultado sem parsear texto.
    """
    caminhos = lista_arquivos(pasta)
    modelos = [carrega(caminho) for caminho in caminhos]
    caminho_ref = referencia or os.path.join(pasta, REFERENCIA_PADRAO)
    ref = carrega(caminho_ref) if os.path.exists(caminho_ref) else None
    profundidades = [float(z) for z in profundidades]

    # --- 1) os modelos do KINECT sao o mesmo modelo? ---
    pares = []
    for posicao, model_a in enumerate(modelos):
        for model_b in modelos[posicao + 1:]:
            for z in profundidades:
                px = espalhamento_px(model_a, model_b, z)
                if px is not None:
                    pares.append({"a": model_a["nome"], "b": model_b["nome"],
                                  "profundidade": z, "px": px})

    # --- 2) sensibilidade a profundidade (na profundidade de trabalho) ---
    sensibilidade = {}
    for model in modelos:
        sensibilidade[model["nome"]] = {
            z: sensibilidade_profundidade(model, z) for z in profundidades}

    # --- 3) cada arquivo contra o modelo CONGELADO do Kinect ---
    comparacoes = None
    if ref is not None:
        z_perto = min(profundidades)
        comparacoes = {}
        for model in modelos:
            comparacoes[model["nome"]] = espalhamento_px(model, ref, z_perto)

    # --- layout fisico (conferencia sem acreditar no npz) ---
    layout = []
    for posicao, model_a in enumerate(modelos):
        pos_a, dist_a = posicao_da_camera(model_a)
        for model_b in modelos[posicao + 1:]:
            pos_b, _ = posicao_da_camera(model_b)
            if pos_a is None or pos_b is None:
                continue
            layout.append({"a": model_a["nome"], "b": model_b["nome"],
                           "metros": float(np.linalg.norm(pos_a - pos_b)),
                           "dist_a": dist_a})

    alertas = []
    for model in modelos:
        if not pronto(model):
            alertas.append(
                "%s: sem matriz/rotacao/translacao -> NAO da' para projetar "
                "(recalibre)" % model["nome"])
        elif model["rms"] is not None and model["rms"] > MAX_STEREO_RMS_PX:
            alertas.append(
                "%s: RMS %.2f px acima de %g px (recalibre)"
                % (model["nome"], model["rms"], MAX_STEREO_RMS_PX))
    for par in pares:
        if par["px"][0] > LIMITE_PX_CALIBRACAO_SUSPEITA:
            alertas.append(
                "%s x %s divergem %.1f px (mediana) na projecao do Kinect em "
                "Z=%.1f m: o Kinect RGB e' UMA camera, entao cada arquivo esta' "
                "usando um modelo DIFERENTE -- e' erro sistematico do overlay"
                % (par["a"], par["b"], par["px"][0], par["profundidade"]))
            break
    if comparacoes:
        for nome, px in comparacoes.items():
            if px is not None and px[0] > LIMITE_PX_CALIBRACAO_SUSPEITA:
                alertas.append(
                    "%s usa um modelo do Kinect %.1f px fora do CONGELADO "
                    "(kinect_intrinsics.npz): recalibre com o modelo congelado"
                    % (nome, px[0]))
    for nome, por_z in sensibilidade.items():
        pior = None
        for z, px in por_z.items():
            if px is not None and (pior is None or px[0] > pior[1]):
                pior = (z, px[0], px[2])
        if pior is not None and pior[1] > LIMITE_PX_CALIBRACAO_SUSPEITA:
            alertas.append(
                "%s: 1 cm de erro no Z medido custa %.0f px (mediana) em Z=%.1f m "
                "-- acima de %g px so' de profundidade; o overlay depende do Z "
                "(prefira a triangulacao ao depth do Kinect)"
                % (nome, pior[1], pior[0], LIMITE_PX_CALIBRACAO_SUSPEITA))
    #: O arquivo congelado EXISTE e e' bom? Quem responde e' a propria leitura da
    #: calibracao (`veredito_congelado`), nao a existencia do arquivo: existir e
    #: ser recusado e' justamente o estado que faz cada sessao re-estimar o Kinect.
    if ref is not None:
        aceito, motivo = veredito_congelado(ref)
        if aceito is False:
            alertas.append(
                "o modelo congelado (%s) e' RECUSADO pela calibracao: %s"
                % (os.path.basename(caminho_ref or REFERENCIA_PADRAO), motivo))
        elif aceito is None:
            alertas.append(
                "nao consegui conferir se o modelo congelado seria ACEITO: %s"
                % motivo)

    return {
        "pasta": pasta,
        "arquivos": modelos,
        "referencia": ref,
        "caminho_referencia": caminho_ref,
        "profundidades": profundidades,
        "pares": pares,
        "sensibilidade": sensibilidade,
        "comparacoes_referencia": comparacoes,
        "layout": layout,
        "alertas": alertas,
    }


def _px_txt(valores):
    """'mediana X | p90 Y | max Z px' (ou '--' quando nao deu para medir)."""
    if valores is None:
        return "--"
    return ("mediana %.1f | p90 %.1f | max %.1f px"
            % (valores[0], valores[1], valores[2]))


def relatorio(resultado, escrever=print):
    """Formata a auditoria: um bloco por check, o veredito e o que fazer."""
    escrever("=" * 72)
    escrever("AUDITORIA DA CALIBRACAO DO OVERLAY (sem hardware)")
    escrever("=" * 72)
    escrever("pasta: %s | arquivos: %d" % (resultado["pasta"],
                                           len(resultado["arquivos"])))
    if resultado["referencia"] is not None:
        congelado = resultado["referencia"]
        detalhe = []
        if congelado["rms"] is not None:
            detalhe.append("RMS %.2f px" % congelado["rms"])
        if congelado["frozen_poses"] is not None:
            detalhe.append("%d poses" % congelado["frozen_poses"])
        if congelado["frozen_span"] is not None:
            detalhe.append("tabuleiro a %.0f%% da diagonal"
                           % (100.0 * congelado["frozen_span"]))
        if congelado["frozen_at"]:
            detalhe.append("gravado em %s" % congelado["frozen_at"])
        if not pronto_kinect(congelado):
            detalhe.append("SEM kinect_matrix/dist: nao da' para comparar")
        escrever("modelo congelado do Kinect: SIM (%s)%s"
                 % (os.path.basename(resultado["caminho_referencia"]),
                    " | " + " | ".join(detalhe) if detalhe else ""))
        #: "O arquivo existe" e "a calibracao vai usar o arquivo" sao coisas
        #: DIFERENTES -- e era essa a confusao de 01/10/2026 (o paliativo existia,
        #: era anunciado como SIM e era recusado na hora de calibrar).
        aceito, motivo = veredito_congelado(congelado)
        if aceito:
            escrever("  -> a CALIBRACAO ACEITA este arquivo: e' ele que TODA "
                     "sessao de par vai reusar (o Kinect nao sera' re-estimado)")
        elif aceito is False:
            escrever("  ! a CALIBRACAO RECUSA este arquivo: %s" % motivo)
            escrever("    Enquanto ele nao for refeito, CADA sessao RE-ESTIMA o "
                     "Kinect e o overlay sai dezenas de px fora -- o RMS dos npz "
                     "NAO denuncia isso.")
        else:
            escrever("  ! nao consegui conferir a aceitacao: %s" % motivo)
    else:
        escrever("modelo congelado do Kinect: nao existe (%s) -- e' o passo 1 do "
                 "conserto" % os.path.basename(resultado["caminho_referencia"]))

    escrever("")
    escrever("--- ARQUIVOS ---")
    for model in resultado["arquivos"]:
        escrever("%s  [%s]" % (model["nome"], model["camera_name"] or "sem nome"))
        if model["rms"] is not None:
            escrever("  RMS %.2f px (Kinect %s | auxiliar %s) | poses %s | pior "
                     "pose %.2f px"
                     % (model["rms"],
                        "%.2f" % model["kinect_rms"]
                        if model["kinect_rms"] is not None else "--",
                        "%.2f" % model["auxiliary_rms"]
                        if model["auxiliary_rms"] is not None else "--",
                        model["poses"] if model["poses"] is not None else "--",
                        model["per_pose_max"]
                        if model["per_pose_max"] is not None else float("nan")))
        if model["kinect_matrix"] is not None:
            matriz_k = np.asarray(model["kinect_matrix"], np.float64)
            dist_k = np.asarray(model["kinect_dist"], np.float64).reshape(-1)
            escrever("  Kinect  : fx %7.1f fy %7.1f cx %7.1f cy %7.1f"
                     % (matriz_k[0, 0], matriz_k[1, 1], matriz_k[0, 2],
                        matriz_k[1, 2]))
            escrever("            dist [%s]"
                     % " ".join("%+.4f" % v for v in dist_k[:5]))
            if model["refined_intrinsics"] is not None:
                refinado = float(np.asarray(
                    model["refined_intrinsics"]).reshape(-1)[0]) >= 0.5
                escrever("            intrinsecas refinadas na sessao: %s"
                         % ("SIM (o stereoCalibrate RE-ESTIMOU o Kinect)"
                            if refinado else "nao (congeladas)"))
        if model["auxiliary_matrix"] is not None:
            matriz_a = np.asarray(model["auxiliary_matrix"], np.float64)
            escrever("  auxiliar: fx %7.1f fy %7.1f cx %7.1f cy %7.1f  (%dx%d)"
                     % (matriz_a[0, 0], matriz_a[1, 1], matriz_a[0, 2],
                        matriz_a[1, 2], model["auxiliary_size"][0],
                        model["auxiliary_size"][1]))
        posicao, distancia = posicao_da_camera(model)
        if posicao is not None:
            escrever("  posicao no frame do Kinect: (%.3f, %.3f, %.3f) m | "
                     "%.3f m do Kinect"
                     % (posicao[0], posicao[1], posicao[2], distancia))

    escrever("")
    escrever("--- 1) O KINECT RGB E' UMA CAMERA SO': os modelos batem? ---")
    if not resultado["pares"]:
        escrever("  (menos de dois arquivos utilizaveis: nada a comparar)")
    for par in resultado["pares"]:
        marca = ("  << ATENCAO (acima de %g px)" % LIMITE_PX_CALIBRACAO_SUSPEITA
                 if par["px"][0] > LIMITE_PX_CALIBRACAO_SUSPEITA else "")
        escrever("%s x %s | Z %.1f m: %s%s"
                 % (par["a"], par["b"], par["profundidade"], _px_txt(par["px"]),
                    marca))

    escrever("")
    escrever("--- 2) CUSTO DE 1 CM DE ERRO NA PROFUNDIDADE (px por cm de Z) ---")
    for nome, por_z in resultado["sensibilidade"].items():
        partes = []
        for z in resultado["profundidades"]:
            px = por_z.get(z)
            partes.append("Z %.1f: %s px/cm"
                          % (z, "%.0f" % px[0] if px is not None else "--"))
        escrever("%s | %s" % (nome, " | ".join(partes)))

    if resultado["comparacoes_referencia"]:
        escrever("")
        escrever("--- 3) CADA ARQUIVO x MODELO CONGELADO (Z mais perto) ---")
        for nome, px in resultado["comparacoes_referencia"].items():
            escrever("%s: %s" % (nome, _px_txt(px)))

    if resultado["layout"]:
        escrever("")
        escrever("--- LAYOUT (distancia entre cameras, pelo proprio npz) ---")
        for par in resultado["layout"]:
            escrever("%s x %s = %.3f m" % (par["a"], par["b"], par["metros"]))

    escrever("")
    escrever("--- VEREDITO ---")
    if resultado["alertas"]:
        for alerta in resultado["alertas"]:
            escrever("  ! " + alerta)
        escrever("")
        escrever("  COMO CONSERTAR, NA ORDEM (o detalhe esta' no README):")
        escrever("   1. modelo do Kinect: congele-o UMA vez com o arranjo "
                 "KINECT-ONLY,")
        escrever("      que leva o tabuleiro a 0,30-0,45 m (SO' assim o span passa")
        escrever("      do minimo exigido -- o caminho de par fica em 11-13%, e o")
        escrever("      programa diz qual foi o numero medido):")
        escrever("        python freeze_kinect_direto.py")
        escrever("      (pelo caminho do par e' o mesmo portao, dentro da sessao:)")
        escrever("        python stereo_calibration.py --salvar-intrinsecas-kinect")
        escrever("      (o RMS so' mede a coerencia interna do npz: ele NAO")
        escrever("      denuncia um modelo do Kinect errado -- por isso este check)")
        escrever("   2. recalibre CADA auxiliar contra o modelo congelado:")
        escrever("        python stereo_calibration.py")
        escrever("   3. valide com a tecla T (2-3 distancias: 0.6, 0.9 e 1.3 m) e")
        escrever("      so' depois ajuste o trim da tecla K / confira o log da L")
        escrever("   4. se o que domina for o item 2 (profundidade), o problema")
        escrever("      NAO e' calibracao: e' o Z usado no overlay. Prefira a")
        escrever("      triangulacao (Kinect+aux ou par de auxiliares) ao depth do")
        escrever("      Kinect, que erra alguns cm justamente na mao.")
    else:
        escrever("  Nada acima dos limites: o modelo do Kinect e' coerente entre")
        escrever("  os arquivos, o RMS esta' dentro do limite e a sensibilidade a")
        escrever("  Z esta' abaixo de %g px/cm." % LIMITE_PX_CALIBRACAO_SUSPEITA)
    escrever("")
    return len(resultado["alertas"])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pasta", default=RAIZ,
                        help="pasta dos npz (padrao: a raiz do projeto)")
    parser.add_argument("--profundidades", type=float, nargs="+",
                        default=list(PROFUNDIDADES_PADRAO), metavar="M",
                        help="profundidades (m) avaliadas na regiao de trabalho")
    parser.add_argument("--referencia", default=None, metavar="ARQUIVO",
                        help="modelo congelado do Kinect (padrao: "
                             "kinect_intrinsics.npz na pasta)")
    args = parser.parse_args(argv)
    resultado = auditoria(args.pasta, args.profundidades, args.referencia)
    alertas = relatorio(resultado)
    return 1 if alertas else 0


if __name__ == "__main__":
    sys.exit(main())
