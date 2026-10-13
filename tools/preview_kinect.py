"""Preview ao vivo do Kinect RGB: abre o sensor e mostra o video numa janela.

POR QUE EXISTE (07/10/2026)
  A resposta curta de "o Kinect funciona?" nao e' `sc.abre_kinect()` bem-sucedido:
  o runtime ABRE em 0,0 s reportando 1920x1080 e pode entregar ZERO frames (foi
  o caso da bancada de 07/10: `IsAvailable = FALSE` e nenhum frame em 20 s, nos
  tres streams). O `tools/checa_kinect.py` responde isso em texto, mas a pergunta
  do operador e' outra: QUERO VER. Este script e' a resposta visual -- abre o
  sensor pelo MESMO caminho oficial (`sc.abre_kinect`, com reabertura), joga o
  color no OpenCV e devolve uma janela com resolucao, FPS e contagem de frames.

  Se o sensor estiver mudo, o programa falha ALTO (exit 1) com a mensagem do
  `sc.abre_kinect` -- nunca uma janela preta infinita: a janela so' nasce depois
  do primeiro frame entregue.

Uso:
    .venv\\Scripts\\python.exe tools\\preview_kinect.py                # ate' o ESC
    .venv\\Scripts\\python.exe tools\\preview_kinect.py --duracao-s 30  # fecha sozinho
    .venv\\Scripts\\python.exe tools\\preview_kinect.py --sem-janela    # so' conta frames

`--sem-janela` e' a sonda headless (CI/servidor): sem GUI, com teto padrao de
15 s, imprimindo o resumo -- e' o modo que a suite usa (sem hardware).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import cv2                                          # noqa: E402
import numpy as np                                  # noqa: E402

import kinect_compat                                # noqa: E402

#: A compatibilidade do pykinect2 tem de entrar ANTES do import dele (ver o
#: docstring de `kinect_compat.py`).
kinect_compat.aplicar()

import stereo_calibration as sc                     # noqa: E402

JANELA = "Kinect RGB - preview (ESC fecha)"

#: Teto do `--sem-janela` sem `--duracao-s` (mesma idea' do LIMITE_SEM_JANELA_S
#: do freeze): sem GUI nao ha' ESC -- o programa nao pode ficar para sempre.
LIMITE_SEM_JANELA_S = 15.0


def quadro_de_color(dados, largura, altura):
    """Buffer do pykinect2 (h*w*4, uint8) -> frame BGR pronto para o OpenCV.

    E' o MESMO caminho do `coleta_poses` do freeze: reshape na ordem do sensor
    (BGRA) + `COLOR_BGRA2BGR`. Aqui virou funcao para a suite travar o contrato
    sem hardware.
    """
    quadro = np.asarray(dados).reshape((altura, largura, 4))
    return cv2.cvtColor(quadro, cv2.COLOR_BGRA2BGR)


def desenha_status(exibicao, fps, n_frames, decorrido):
    """Resolucao/FPS/tempo no canto -- texto com contorno p/ qualquer fundo."""
    texto = "%dx%d | %.1f fps | %d frames | %.0f s | ESC fecha" % (
        exibicao.shape[1], exibicao.shape[0], fps, n_frames, decorrido)
    origem = (12, 28)
    cv2.putText(exibicao, texto, origem, cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (0, 0, 0), 4, cv2.LINE_AA)           # contorno preto
    cv2.putText(exibicao, texto, origem, cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)     # miolo branco
    return exibicao


def _gancho_do_fechamento(anterior=None, aviso=None):
    """`threading.excepthook` que transforma o wart do pykinect2 em UMA linha.

    Mesmo filtro do `freeze_kinect_direto.gancho_do_pykinect2`: `runtime.close()`
    derruba os handles enquanto a thread de frames espera e o Python imprime um
    TRACEBACK inteiro DEPOIS do resumo -- com cara de erro do preview. So' esse
    `AttributeError: ... GetFrameArrivedEventData` e' filtrado.
    """
    anterior = anterior or threading.__excepthook__
    aviso = aviso or (lambda texto: print(texto))

    def gancho(args):
        if isinstance(getattr(args, "exc_value", None), AttributeError) \
                and "GetFrameArrivedEventData" in str(args.exc_value):
            aviso("  (aviso: o runtime do Kinect reclamou ao fechar -- wart "
                  "conhecido do pykinect2; nao afeta o preview)")
            return
        anterior(args)

    return gancho


def preview(kinect, duracao_s=0.0, sem_janela=False, _imprime=print):
    """Loop de exibicao do color. Devolve (n_frames, fps_medio).

    * `duracao_s > 0` fecha sozinho; 0 = ate' o ESC (janela) ou o teto
      `LIMITE_SEM_JANELA_S` (sem janela -- sem GUI nao ha' tecla);
    * `fps_medio` e' a media da sessao inteira (uma media movida esconderia o
      periodo mudo, que e' justamente o sintoma que interessa).
    """
    largura = kinect.color_frame_desc.Width
    altura = kinect.color_frame_desc.Height
    if duracao_s <= 0.0 and sem_janela:
        duracao_s = LIMITE_SEM_JANELA_S
        _imprime("  --sem-janela sem --duracao-s: vale o teto de %.0f s."
                 % LIMITE_SEM_JANELA_S)
    inicio = time.time()
    n_frames = 0
    avisou_sem_frame = False
    try:
        while True:
            decorrido = time.time() - inicio
            if duracao_s > 0.0 and decorrido >= duracao_s:
                break
            quadro = None
            if kinect.has_new_color_frame():
                quadro = quadro_de_color(kinect.get_last_color_frame(),
                                         largura, altura)
                n_frames += 1
            if quadro is None:
                if not avisou_sem_frame:
                    _imprime("  aguardando o primeiro frame do sensor...")
                    avisou_sem_frame = True
                if sem_janela:
                    time.sleep(0.02)
                else:
                    #: So' manter o GUI vivo: sem frame a espera e' do SENSOR
                    #: (a janela nasce no1o frame, em main()).
                    cv2.waitKey(1)
                continue
            if sem_janela:
                time.sleep(0.02)
                continue
            exibicao = sc.reduzir_janela(quadro)
            fps = n_frames / max(time.time() - inicio, 1e-9)
            exibicao = desenha_status(exibicao, fps, n_frames, time.time() - inicio)
            cv2.imshow(JANELA, exibicao)
            if cv2.waitKey(1) & 0xFF == 27:              # ESC
                break
    except KeyboardInterrupt:
        _imprime("  interrompido (Ctrl+C).")
    decorrido = time.time() - inicio
    fps = (n_frames / decorrido) if decorrido > 0.0 else 0.0
    return n_frames, fps



def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duracao-s", type=float, default=0.0,
                        help="fecha sozinho apos N s (0 = ate' ESC; "
                             "sem janela, teto de %.0f s)" % LIMITE_SEM_JANELA_S)
    parser.add_argument("--sem-janela", action="store_true",
                        help="sem GUI: conta frames e imprime o resumo (headless)")
    args = parser.parse_args(argv)

    threading.excepthook = _gancho_do_fechamento()
    print("preview_kinect: abrindo o RGB pelo caminho oficial "
          "(sc.abre_kinect: %d tentativas)..." % sc.TENTATIVAS_KINECT)
    try:
        kinect = sc.abre_kinect()
    except RuntimeError as exc:
        print("ERRO: %s" % exc)
        print("Sensor nao entregou frame -- sem video para mostrar. "
              "Rode tools\\checklist_kinect_mudo.py para o passo a passo.")
        return 2
    try:
        largura = kinect.color_frame_desc.Width
        altura = kinect.color_frame_desc.Height
        print("Resolucao do Kinect RGB: %dx%d | %s"
              % (largura, altura,
                 "sem janela (contagem)" if args.sem_janela else
                 "janela: %s" % JANELA))
        n_frames, fps = preview(kinect, duracao_s=args.duracao_s,
                                sem_janela=args.sem_janela)
    finally:
        kinect.close()
        if not args.sem_janela:
            cv2.destroyAllWindows()
    print("Preview encerrado: %d frames, %.1f fps de media." % (n_frames, fps))
    if n_frames == 0:
        print("NENHUM frame no periodo: o sensor ABRIU mas esta MUDO. "
              "Nao e' a janela -- e' o sensor (veja checklist_kinect_mudo.py).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

