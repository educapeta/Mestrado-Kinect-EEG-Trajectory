"""Probe da LATERALIDADE das maos por camera (Kinect + webcams auxiliares).

Responde AO VIVO, e SEM JANELA, a pergunta que originou o relato da bancada: os
rotulos de lateralidade do MediaPipe estao na MESMA ordem em todas as cameras?

Por que isso importa: o groundtruth casa as maos PELO LADO (right -> IMU 4210,
left -> IMU 4211). Se o rotulo de UMA auxiliar sai trocado (o driver dela pode ja'
entregar o fluxo espelhado -- nesta maquina ha' dois MODELOS de webcam: SIGMA-W780M
e Trust USB Camera), aquela camera passa a associar a mao errada ao IMU errado, ao
overlay errado e a triangulacao errada. E isso nao aparece sozinho: o verificador
visual precisa de um numero.

O que ele faz, por frame:
  1. le o Kinect RGB + cada webcam auxiliar (com o MESMO espelho do tool);
  2. detecta as maos em todas as cameras (um detector MediaPipe por camera);
  3. roda o `ArbitroDeLateralidade`, que decide o lado pela GEOMETRIA (punho 3D do
     Kinect + homografia no-a-no da mao de cada camera), e imprime, por camera, o
     rotulo do MediaPipe CONTRA o lado medido;
  4. ao final resume por camera e mostra o que foi aprendido.

Como usar (bancada; o Kinect precisa ver as maos):
    .venv\\Scripts\\python.exe tools\\probe_lateralidade.py
    .venv\\Scripts\\python.exe tools\\probe_lateralidade.py --segundos 40 --gravar
    .venv\\Scripts\\python.exe tools\\probe_lateralidade.py --aux-cameras "2,0"

Leitura do veredito, por camera:
   CERTO     = o rotulo do MediaPipe concorda com a geometria do Kinect;
   INVERTIDO = o rotulo esta' TROCADO (e' o bug): com `--gravar` o programa grava
               a inversao e TODOS os programas (tool, protocolo, sand, demo)
               passam a corrigir sozinhos;
   INDECISO  = as maos quase se sobrepoem no quadro (sem margem para separar) ou a
               camera nao tem a homografia da mao -- NENHUMA troca e' feita.

Sem `--gravar` o probe NAO escreve nada: e' para olhar antes de decidir. Ele nao
abre janelas de video de proposito (roda ao lado do tool, ou sozinho no terminal).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kinect_imu_groundtruth as gt                     # noqa: E402
import kinect_groundtruth_tool as tool                  # noqa: E402


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Probe da lateralidade das maos por camera (sem janelas).")
    parser.add_argument(
        "--segundos", type=float, default=25.0, metavar="S",
        help="tempo de coleta (padrao 25 s; ~5-10 frames/s com 2 auxiliares).")
    parser.add_argument(
        "--aux-cameras", default="", metavar='"N,M"',
        help="indices das webcams auxiliares (vazio = 'auto': todas as externas "
             "MENOS a do laptop, pelo NOME do Windows).")
    parser.add_argument(
        "--aux-index", type=int, default=None, metavar="N",
        help="indice da webcam auxiliar ATIVA (padrao: automatico).")
    parser.add_argument(
        "--gravar", action="store_true",
        help="GRAVA a inversao aprendida em aux_handedness_flip.json (sem esta "
             "opcao o probe so' mostra: nada e' escrito no disco).")
    parser.add_argument(
        "--min-votos", type=int, default=gt.ARBITRO_MIN_VOTOS, metavar="N",
        help=f"frames com consenso para gravar (padrao {gt.ARBITRO_MIN_VOTOS}).")
    return parser.parse_args()


def rotulos_de(entradas):
    """(rotulos do MediaPipe, lados finais) de UMA camera, para a linha ao vivo."""
    return ("[%s]" % " ".join((entrada.get("side_rotulo") or "?")
                              for entrada in entradas),
            "[%s]" % " ".join((entrada.get("side") or "?")
                              for entrada in entradas))


def veredito(relatorio):
    """Uma palavra por camera: CERTO / INVERTIDO / INDECISO / SEM MAO."""
    fonte = relatorio.get("fonte", "")
    if fonte == "sem mao":
        return "SEM MAO"
    if fonte == "geometria":
        return "INVERTIDO" if relatorio.get("inverteu") else "CERTO"
    return "INDECISO"


def linha_de(indice, relatorio, corrigido):
    """Linha curta de UMA camera: veredito, rotulos e (se indeciso) o motivo.

    O motivo existe porque "INDECISO" sozinho nao diz o que fazer: `margem N px`
    e' "separe mais as maos no quadro"; `projecao a N px` e' "a projecao da mao
    desta camera caiu longe" -- e o proprio motivo diz a VIA usada (stereo 3D ou
    a homografia 2D do esqueleto, que e' imprecisa por natureza e pede o stereo);
    "sem calibracao da mao desta camera" e' "esta camera nunca foi calibrada
    (`stereo_calibration.py` com ela ativa, ou tecla H no tool)".
    """
    info = relatorio.get(indice) or {}
    rotulo, lado = rotulos_de(corrigido.get(indice) or [])
    texto = f"aux {indice}: {veredito(info)} {rotulo}->{lado}"
    if veredito(info) == "INDECISO":
        motivo = info.get("detalhe") or info.get("fonte")
        if motivo:
            texto += f" ({motivo})"
    return texto


def geometria_disponivel(alignment, indice):
    """True se a camera tem COMO projetar a mao no RGB do Kinect.

    Duas vias servem ao arbitro: o STEREO 3D daquela camera (a via do overlay --
    `aux_view(indice).usable`) e, na falta dele, a homografia no-a-no da mao.
    Sem nenhuma das duas nao ha' como medir e o lado fica no rotulo.
    """
    modelo = (alignment.aux_models or {}).get(int(indice)) or {}
    if modelo.get("hand_homography") is not None:
        return True
    try:
        vista = alignment.aux_view(indice)
    except Exception:
        return False
    return bool(vista is not None and getattr(vista, "usable", False))


def main():
    args = parse_cli()
    indices = tool._indices_auxiliares(argparse.Namespace(
        aux_cameras=args.aux_cameras, aux_index=args.aux_index))
    print("Probe de LATERALIDADE das maos (Kinect + webcams auxiliares)")
    print(f"cameras auxiliares: {indices} | {args.segundos:.0f} s | "
          + ("GRAVANDO o que for aprendido" if args.gravar
             else "sem gravar (use --gravar para fixar a correcao)"))
    for indice in indices:
        print(f"  aux {indice} ({gt.nome_auxiliar(indice)}): "
              + ("rotulo INVERTIDO (ja' corrigindo)" if gt.aux_handedness_flip(indice)
                 else "rotulo normal"))
    tracker = gt.KinectHandTracker(aux_indices=tuple(indices))
    alignment = gt.CameraAlignment(aux_indices=indices)
    arbitro = gt.ArbitroDeLateralidade(verbose=args.gravar,
                                       persistir=args.gravar,
                                       min_votos=args.min_votos)
    # Estado de partida: e' contra ele que se diz, no fim, o que foi APRENDIDO
    # nesta coleta (a inversao tambem fica na memoria quando nao se grava).
    inicial = {int(indice): gt.aux_handedness_flip(indice)
               for indice in indices}
    for indice in indices:
        if geometria_disponivel(alignment, indice):
            continue
        arquivo = (f"hand_landmark_calibration_aux{int(indice)}.npz"
                   if int(indice) else "hand_landmark_calibration.npz")
        print(f"  ATENCAO: aux {indice} ({gt.nome_auxiliar(indice)}) NAO tem "
              f"projecao utilizavel da mao ({arquivo} e nenhum stereo aceito): "
              "nesta camera a lateralidade fica no rotulo do MediaPipe. Para "
              "medir, rode `python stereo_calibration.py` com ela ATIVA (ou "
              "aperte H no tool, com ela ativa, para a calibracao do esqueleto).")
    tamanho = alignment.calibrated_auxiliary_size or (1280, 720)
    capturas = {}
    for indice in indices:
        try:
            capturas[int(indice)] = tool.abre_camera_auxiliar(indice, tamanho)
        except RuntimeError as exc:
            print(f"AVISO: webcam auxiliar {indice} nao abriu ({exc}); seguindo "
                  "com as que abriram")
    contagem = {int(indice): {"CERTO": 0, "INVERTIDO": 0, "INDECISO": 0,
                              "SEM MAO": 0} for indice in indices}
    ultimo = {}
    print("\nSegure as DUAS maos VISIVEIS e SEPARADAS (o Kinect precisa ver as "
          "duas):\n")
    try:
        fim = time.monotonic() + max(1.0, args.segundos)
        proximo_print = 0.0
        while time.monotonic() < fim:
            color, depth = tracker.frames()
            aux_por_camera = {}
            for indice, captura in capturas.items():
                ok, quadro = captura.read()
                if not ok or quadro is None:
                    aux_por_camera[indice] = []
                    continue
                quadro = gt.transform_auxiliary_frame(quadro)
                alignment.set_active_aux(indice)
                alignment.set_auxiliary_size(quadro)
                aux_por_camera[indice] = tracker.detect_auxiliary_hands_sides(
                    quadro, aux_index=indice)
            _, maos_kinect = tracker.detect_hands_all(color, depth)
            corrigido, relatorio = arbitro.arbitrar(
                alignment, maos_kinect, aux_por_camera,
                (int(tracker.color_width), int(tracker.color_height)))
            for indice, info in relatorio.items():
                contagem[indice][veredito(info)] += 1
                ultimo[indice] = info
            agora = time.monotonic()
            if agora >= proximo_print:
                proximo_print = agora + 1.0
                print(f"[{max(0.0, fim - agora):5.1f}s] Kinect: "
                      f"{len(maos_kinect)} mao(s) | " + " | ".join(
                          linha_de(indice, relatorio, corrigido)
                          for indice in sorted(relatorio)))
    except KeyboardInterrupt:
        print("\n(interrompido pelo usuario -- resumo do que foi coletado)")
    finally:
        for captura in capturas.values():
            captura.release()
        tracker.close()
    print("\n=== RESUMO POR CAMERA ===")
    for indice in sorted(contagem):
        numeros = contagem[indice]
        votos = arbitro.votos.get(int(indice), {"oposto": 0, "igual": 0})
        print(f"aux {indice} ({gt.nome_auxiliar(indice)}): "
              f"CERTO {numeros['CERTO']} | INVERTIDO {numeros['INVERTIDO']} | "
              f"INDECISO {numeros['INDECISO']} | SEM MAO {numeros['SEM MAO']}"
              f"   (votos: rotulo trocado {votos['oposto']} x confere "
              f"{votos['igual']})")
        detalhe = (ultimo.get(int(indice)) or {}).get("detalhe")
        if detalhe and detalhe != "ok":
            print(f"        ultimo frame: {detalhe}")
    print("\n=== LATERALIDADE VIGENTE ===")
    for indice in sorted(contagem):
        print(f"aux {indice} ({gt.nome_auxiliar(indice)}): "
              + ("rotulo INVERTIDO (corrigido em todos os programas)"
                 if gt.aux_handedness_flip(indice) else "rotulo normal"))
    invertidas = [indice for indice in sorted(contagem)
                  if contagem[indice]["INVERTIDO"] > contagem[indice]["CERTO"]]
    aprendidas_agora = [indice for indice in sorted(contagem)
                        if gt.aux_handedness_flip(indice) != inicial[indice]]
    # SEM MEDICAO NAO E' "esta' tudo certo": com 0 frames decididos em TODAS as
    # cameras o probe nao mediu nada (ninguem na frente das cameras, ou as maos
    # nunca visiveis nas DUAS ao mesmo tempo). Medido ao vivo em 05/10/2026: a
    # coleta sem ninguem no quadro terminava com "Nenhuma camera precisou de
    # correcao: o rotulo do MediaPipe concorda com a geometria do Kinect" -- uma
    # conclusao que os dados nao sustentam.
    decisivos = sum(numeros["CERTO"] + numeros["INVERTIDO"]
                    for numeros in contagem.values())
    if not decisivos:
        print("\nSEM MEDICAO: nenhuma camera teve um frame em que o Kinect e a "
              "auxiliar vissem mao no MESMO instante (CERTO + INVERTIDO = 0 em "
              "todas). Nada foi aprendido e nada foi gravado -- segure as DUAS "
              "maos VISIVEIS e SEPARADAS (~40 cm), com as duas cameras "
              "enxergando, e repita.")
    elif aprendidas_agora:
        resumo = ", ".join(f"aux {indice} ({gt.nome_auxiliar(indice)})"
                           for indice in aprendidas_agora)
        if args.gravar:
            print(f"\nA inversao de lateralidade foi APRENDIDA e GRAVADA ({resumo})"
                  f" em {gt.AUX_HANDEDNESS_FLIP_FILE.name}: todos os programas "
                  "(tool, protocolo, sand, demo) passam a corrigir sozinhos.")
        else:
            print(f"\nO arbitro APRENDEU nesta coleta que o rotulo do MediaPipe "
                  f"esta' trocado em {resumo} -- mas SEM GRAVAR (rode com "
                  "--gravar para fixar, senao a proxima sessao comeca do zero).")
    elif invertidas:
        print("\nAs cameras " + ", ".join(str(i) for i in invertidas)
              + " apareceram com o rotulo TROCADO em mais frames do que certos. "
              "Rode de novo com --gravar para fixar a correcao (ou abra o tool "
              "com --aux-flip-lateralidade "
              f"\"{','.join(str(i) for i in invertidas)}\").")
    else:
        print("\nNenhuma camera precisou de correcao: o rotulo do MediaPipe "
              "concorda com a geometria do Kinect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
