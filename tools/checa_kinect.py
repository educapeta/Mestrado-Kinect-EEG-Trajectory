"""Pre-checagem do sensor Kinect v2 ANTES de gastar bancada.

POR QUE EXISTE (bancada de 07/10/2026)
  O `freeze_kinect_direto.py` abre o Kinect por `sc.abre_kinect()` e, se o sensor
  nao entrega frame, ele para com "NADA GRAVADO" -- mas esse e' o ULTIMO passo de
  uma rotina de 5-10 min: o operador ja' posicionou o tabuleiro 12+ vezes, a 0,35
  m, e so' no fim descobre que o problema era o SENSOR (o Kinect ficou mudo depois
  do uso do Kinect Studio; medido naquele dia: runtime abria em 0,0 s reportando
  1920x1080 e NENHUM frame saia em 20 s, em color, depth, body e nos tres juntos).

  Este script responde, em segundos e SEM tabuleiro:
    * o pykinect2 importa DEPOIS do `kinect_compat` (Patch A/B/D: o
      `tagSTATSTG` de 64 bits, a checagem de versao do comtypes e o alias
      `numpy.object`)? Sem isso NENHUM programa do projeto roda;
    * o runtime ABRE? (= drivers de pe', sensor enumerado)
    * o sensor ENTREGA frame? e em quanto tempo, com reabertura se preciso
      (`sc.abre_kinect`);
    * quais webcams existem por indice (o mapa indice->NOME, porque o indice do
      OpenCV e' a POSICAO no USB e troca sozinho);
    * o que o disco diz do modelo congelado do Kinect (`kinect_intrinsics.npz`)
      e das calibracoes de par -- via `sc.aceita_modelo_kinect`, a MESMA regra da
      calibracao, para nao repetir o veredito paralelo de 01/10/2026.

Uso:
    .venv\\Scripts\\python.exe tools\\checa_kinect.py
    .venv\\Scripts\\python.exe tools\\checa_kinect.py --sem-webcams
    .venv\\Scripts\\python.exe tools\\checa_kinect.py --espera-s 30 --tentativas 4

Saida: um bloco por verificacao e um resumo final. Codigo de saida 0 = o sensor
ENTREGA frame (pode ir para a bancada); 1 = nao entrega (nao adianta montar o
tabuleiro); 2 = falta o SDK/pykinect2 (nem abre).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np                                          # noqa: E402

import kinect_compat                                        # noqa: E402

#: A compatibilidade do pykinect2 tem de entrar ANTES do import dele (ver o
#: docstring de `kinect_compat.py`), e o import do pacote pode falhar: aqui a
#: falha e' RESULTADO da checagem, nao traceback.
kinect_compat.aplicar()

import stereo_calibration as sc                             # noqa: E402

try:
    from pykinect2 import PyKinectV2
    from pykinect2.PyKinectRuntime import PyKinectRuntime
    PYKINECT2_ERRO = None
except Exception as exc:                                    # noqa: BLE001 - sonda
    PyKinectV2 = None
    PyKinectRuntime = None
    PYKINECT2_ERRO = "%s: %s" % (type(exc).__name__, exc)

#: Fonte dos streams: as 3 juntas no teste de "algum stream passa", porque o
#: sensor mudo e' mudo em todas (medido em 07/10/2026) e um stream que funcione
#: ja' e' noticia (muda o diagnostico).
STREAMS = (("color", "FrameSourceTypes_Color"),
           ("depth", "FrameSourceTypes_Depth"),
           ("body", "FrameSourceTypes_Body"))


def _texto_arquivo(caminho):
    """(existe, data local, tamanho) de um arquivo -- sem levantar."""
    caminho = Path(caminho)
    if not caminho.exists():
        return False, "-", 0
    return (True,
            time.strftime("%d/%m/%Y %H:%M",
                          time.localtime(caminho.stat().st_mtime)),
            caminho.stat().st_size)


def checa_pykinect2():
    """(ok, linhas) do import do pykinect2 depois dos patches."""
    linhas = ["pykinect2 (com kinect_compat):"]
    linhas.append("  compatibilidade : %s" % kinect_compat.motivo())
    if PYKINECT2_ERRO is not None:
        linhas.append("  IMPORT          : FALHOU -- %s" % PYKINECT2_ERRO)
        linhas.append("  -> sem isto NENHUM programa do projeto abre o Kinect "
                      "(eram 14 suites falhando por isso em 28/09/2026).")
        return False, linhas
    linhas.append("  import          : OK (PyKinectV2 + PyKinectRuntime)")
    return True, linhas


def checa_streams(runtime_factory, espera_s=4.0):
    """Tenta CADA stream separadamente e devolve o que respondeu.

    `runtime_factory` recebe as flags de fonte e devolve um runtime; existe para
    o teste injetar um sensor falso. Nunca levanta: um stream que explode e'
    resultado ("nao abre"), nao erro do programa.
    """
    resultados = []
    for nome, atributo in STREAMS:
        flag = getattr(PyKinectV2, atributo, None) if PyKinectV2 is not None else None
        if flag is None:
            resultados.append((nome, "sem flag no pykinect2", None))
            continue
        try:
            runtime = runtime_factory(flag)
        except Exception as exc:                            # noqa: BLE001 - sonda
            resultados.append((nome, "nao abriu: %s" % exc, None))
            continue
        inicio = time.time()
        achou = None
        try:
            while time.time() - inicio < espera_s:
                if nome == "color" and runtime.has_new_color_frame():
                    achou = "color"
                elif nome == "depth" and runtime.has_new_depth_frame():
                    achou = "depth"
                elif nome == "body" and runtime.has_new_body_frame():
                    achou = "body"
                if achou:
                    break
                time.sleep(0.02)
        except Exception as exc:                            # noqa: BLE001 - sonda
            resultados.append((nome, "erro ao consultar: %s" % exc, None))
            continue
        finally:
            try:
                runtime.close()
            except Exception:                               # noqa: BLE001
                pass
        resultados.append((nome, "ENTREGA frame" if achou else "NENHUM frame",
                           time.time() - inicio if achou else None))
    return resultados


def checa_webcams(maximo=4):
    """(ok, linhas) do mapa indice->NOME das webcams.

    O nome vem do Windows (DirectShow), na MESMA ordem do `CAP_DSHOW`: e' o que
    denuncia "a camera auxiliar" ser a webcam do laptop, e o que casa com o
    `camera_name` gravado em cada `stereo_calibration_aux{N}.npz`.
    """
    linhas = ["webcams (indice OpenCV -> nome do Windows):"]
    try:
        import cv2
        import camera_names
    except ImportError as exc:
        linhas.append("  indisponivel: %s" % exc)
        return False, linhas
    encontradas = 0
    for indice in range(maximo):
        try:
            nome = camera_names.nome_da_camera(indice)
        except Exception as exc:                            # noqa: BLE001 - sonda
            nome = None
            linhas.append("  indice %d: (erro ao consultar: %s)" % (indice, exc))
            continue
        if nome is None:
            linhas.append("  indice %d: (nada)" % indice)
            continue
        captura = cv2.VideoCapture(indice)
        leu = False
        if captura.isOpened():
            leu = bool(captura.read()[0])
        captura.release()
        encontradas += 1
        linhas.append("  indice %d: %-24s lê frame: %s"
                      % (indice, nome, "sim" if leu else "NAO"))
    if encontradas:
        linhas.append("  -> use ESTES indices em `--aux-index`/`--aux-cameras`; "
                      "o indice e' a posicao no USB e muda sozinho.")
    else:
        linhas.append("  -> NENHUMA webcam respondeu. A calibracao de par nao "
                      "roda sem a auxiliar.")
    return encontradas > 0, linhas


def checa_calibracoes(raiz=RAIZ):
    """(linhas) do estado em disco das calibracoes, pelo MESMO veredito da leitura.

    O modelo congelado e' julgado por `sc.aceita_modelo_kinect` (poses + span +
    RMS) -- a regra que a calibracao usa. Reimplementar aqui foi o que deixou o
    diagnostico dizer "congelado: SIM" enquanto a calibracao re-estimava o Kinect
    em silencio (01/10/2026).
    """
    linhas = ["calibracoes em disco:"]
    congelado = Path(raiz) / "kinect_intrinsics.npz"
    existe, data, tamanho = _texto_arquivo(congelado)
    if not existe:
        linhas.append("  kinect_intrinsics.npz: NAO EXISTE -- a calibracao de par "
                      "re-estima o Kinect a cada sessao (o bug que o congelamento "
                      "existe para matar). Rode: freeze_kinect_direto.py")
    else:
        matriz, distorcao, info = sc.carrega_intrinsecas_kinect(congelado)
        estado = "ACEITO" if matriz is not None else "RECUSADO"
        linhas.append("  kinect_intrinsics.npz: %s (%s, %d B)"
                      % (estado, data, tamanho))
        if matriz is None:
            linhas.append("    motivo: %s" % info.get("motivo"))
            linhas.append("    -> enquanto isso o Kinect e' re-estimado por "
                          "sessao; dois npz da MESMA bancada divergem "
                          "(medido 07/10/2026: dfx 12,5 px / dcy -23,5 px).")
        else:
            linhas.append("    poses=%s span=%s rms=%s | fonte: %s"
                          % (info.get("poses"),
                             "-" if info.get("span") is None
                             else "%.1f%%" % (info["span"] * 100.0),
                             info.get("rms"), info.get("congelado_em")))
            if info.get("aviso_placa"):
                linhas.append("    aviso: %s" % info["aviso_placa"])
    for indice in range(3):
        caminho = Path(raiz) / ("stereo_calibration%s.npz"
                                % ("" if indice == 0 else "_aux%d" % indice))
        existe, data, _ = _texto_arquivo(caminho)
        if not existe:
            continue
        try:
            dados = np.load(caminho, allow_pickle=True)
            fonte = (str(np.ravel(dados["kinect_intrinsics_source"])[0])
                     if "kinect_intrinsics_source" in dados.files else "(antigo)")
            nome = (str(np.ravel(dados["camera_name"])[0])
                    if "camera_name" in dados.files else "(sem camera_name)")
            rms = (float(np.ravel(dados["rms"])[0]) if "rms" in dados.files
                   else float("nan"))
        except Exception as exc:                            # noqa: BLE001 - sonda
            linhas.append("  %s: nao consegui ler (%s)" % (caminho.name, exc))
            continue
        linhas.append("  %s: %s | %s | RMS %.3f px | fonte do Kinect: %s"
                      % (caminho.name, nome, data, rms, fonte))
        if fonte == "sessao":
            linhas.append("    -> 'sessao' = o congelado foi RECUSADO nesta "
                          "gravacao; refaca com o modelo congelado valendo.")
    return linhas


def _imprime(titulo, linhas):
    print("\n" + "=" * 72)
    print(titulo)
    print("=" * 72)
    for linha in linhas:
        print(linha)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--espera-s", type=float, default=sc.ESPERA_KINECT_S,
                        help="espera do primeiro frame em cada tentativa (s)")
    parser.add_argument("--tentativas", type=int, default=None,
                        help="quantas vezes reabrir o runtime (padrao: %d)"
                             % sc.TENTATIVAS_KINECT)
    parser.add_argument("--espera-stream-s", type=float, default=4.0,
                        help="espera por stream no teste individual (s)")
    parser.add_argument("--sem-webcams", action="store_true",
                        help="nao abre as webcams (teste so' do Kinect)")
    parser.add_argument("--sem-calibracoes", action="store_true",
                        help="nao le os npz de calibracao")
    args = parser.parse_args(argv)

    print("checa_kinect: o sensor ENTREGA frame? (nao precisa de tabuleiro)")
    print("Banco de prova: 'abriu' NAO e' 'entrega'. O runtime abrir em 0,0 s "
          "reportando 1920x1080 com zero frames foi o sintoma de 07/10/2026.")

    ok_import, linhas_import = checa_pykinect2()
    _imprime("1) pykinect2 + compatibilidade", linhas_import)
    if not ok_import:
        print("\nRESUMO: NAO DA' PARA ABRIR -- falta o pykinect2/SDK (codigo 2).")
        return 2

    _imprime("2) streams individuais (color / depth / body)",
             ["cada stream e' testado sozinho: um que responda muda o "
              "diagnostico."])
    resultados = checa_streams(lambda flag: PyKinectRuntime(flag),
                               espera_s=args.espera_stream_s)
    for nome, estado, tempo in resultados:
        print("  %-6s: %s%s" % (nome, estado,
                                "" if tempo is None else " (%.1f s)" % tempo))

    tentativas = args.tentativas
    _imprime("3) abertura OFICIAL do projeto (sc.abre_kinect)",
             ["com reabertura: e' este o caminho que o freeze/stereo usam."])
    entrega = False
    try:
        runtime = sc.abre_kinect(espera_s=args.espera_s, tentativas=tentativas)
        entrega = True
        try:
            runtime.close()
        except Exception:                                   # noqa: BLE001
            pass
        print("  RESULTADO: o Kinect ENTREGA frame -- pode ir para a bancada.")
    except RuntimeError as exc:
        print("  RESULTADO: o Kinect NAO entrega frame.")
        print("")
        print("  %s" % exc)

    if not args.sem_webcams:
        _, linhas_web = checa_webcams()
        _imprime("4) webcams", linhas_web)
    if not args.sem_calibracoes:
        _imprime("5) calibracoes em disco", checa_calibracoes())

    print("\n" + "=" * 72)
    if entrega:
        print("RESUMO: Kinect OK -- Fase 1 (freeze_kinect_direto.py) pode rodar.")
        return 0
    print("RESUMO: Kinect NAO entrega frame. NAO monte o tabuleiro: o "
          "freeze_kinect_direto.py vai parar em `sc.abre_kinect()` e sair com "
          "'NADA GRAVADO'. Resolva a causa 1/2/3 da mensagem acima e rode este "
          "script de novo.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
