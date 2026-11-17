"""Aposenta o `kinect_intrinsics.npz` FORA DO AR, preservando-o como referencia.

POR QUE EXISTE (bancada de 07/10/2026)
  O `kinect_intrinsics.npz` em disco e' o PALIATIVO de 30/09/2026: ele foi
  gravado com a placa ANTIGA (5x3 cantos / 57 mm), tem span mediano de 11-13%
  (a meta e' 35%) e -- decisivo -- NAO registra os campos `poses` e `span`. A
  leitura (`stereo_calibration.carrega_intrinsecas_kinect`) recusa exatamente por
  isso:

      "kinect_intrinsics.npz: nao registra poses nem span: gravado por um
       caminho que nao mede a cobertura do quadro..."

  O efeito nao e' so' um aviso: como o arquivo e' RECUSADO, o Kinect volta a ser
  RE-ESTIMADO a cada sessao de par -- o bug que o congelamento existe para matar.
  MEDIDO (07/10/2026) nos npz de par ainda em disco, os dois com
  `kinect_intrinsics_source='sessao'`:

      dfx = +12,5 px | dcx = +11,7 px | dcy = -23,5 px   (entre as DUAS sessoes)
      contra o paliativo: ate' +19,1 px de fx e +17,0 px de cy

  E erro SISTEMATICO: o RMS nao denuncia (ele mede a coerencia interna de cada
  arquivo, nao qual modelo e' o certo) -- por isso a calibracao "parece boa" e o
  esqueleto sai longe.

O QUE ESTE PROGRAMA FAZ -- E O QUE NAO FAZ
  Ele RENOMEIA o arquivo para `<nome><SUFIXO_ARQUIVADO>` (padrao
  `kinect_intrinsics.npz.arquivado`), que a leitura NAO carrega (o loader le' o
  nome EXATO). Assim:

    * o paliativo continua no disco, inteiro, para a auditoria comparar (era a
      regra "nao apagar o paliativo antes de tudo" do plano);
    * para de competir com o proximo `freeze_kinect_direto.py`, que passa a
      escrever o arquivo BOM no lugar limpo;
    * para de ser recusado a cada abertura -- a recusa em si nao some, mas o
      programa deixa de reclamar de um arquivo que ninguem mandou apagar.

  O que ele NAO faz: **nao inventa `poses` e `span`** para o arquivo passar no
  portao. Forjar os campos que decidem se o modelo presta seria mentir para o
  proprio portao e cimentar o vies de ~46 px que o arquivo denuncia no proprio
  `frozen_at` -- o oposto do que o congelamento existe para fazer.

Uso (DRY-RUN por padrao -- nao mexe em nada ate' voce pedir):
    .venv\\Scripts\\python.exe tools\\conserta_intrinsecas_kinect.py
    .venv\\Scripts\\python.exe tools\\conserta_intrinsecas_kinect.py --sim
    .venv\\Scripts\\python.exe tools\\conserta_intrinsecas_kinect.py --sim --forcar
    .venv\\Scripts\\python.exe tools\\conserta_intrinsecas_kinect.py --sim --restaurar

Depois de arquivar: rode `freeze_kinect_direto.py` (bancada) e confira com
`tools\\checa_kinect.py` e `tools\\diagnostico_calibracao.py`.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np                                          # noqa: E402

import stereo_calibration as sc                             # noqa: E402

#: Nome do arquivo do modelo congelado (a fonte da verdade e' o modulo).
ALVO_PADRAO = sc.KINECT_INTRINSICS_FILE
#: Sufixo do arquivo aposentado. NAO e' `.npz`: assim ele nao entra em nenhum
#: glob de `*.npz`, nao e' carregado por engano e continua obviamente lido como
#: "fora do ar" por quem abrir a pasta.
SUFIXO = sc.SUFIXO_ARQUIVADO


def _resumo(caminho):
    """(ok, motivo, dados) do veredito da LEITURA sobre `caminho`.

    Usa `sc.carrega_intrinsecas_kinect` -- a MESMA regra da calibracao -- para o
    veredito daqui nunca divergir do que o programa vai fazer.
    """
    if not Path(caminho).exists():
        return False, "nao existe", None
    matriz, distorcao, info = sc.carrega_intrinsecas_kinect(caminho)
    return (matriz is not None), info.get("motivo"), info


def estado_atual(alvo=ALVO_PADRAO):
    """(linhas) do que ha' em disco AGORA: alvo, arquivado e o veredito dos dois."""
    alvo = Path(alvo)
    arquivado = Path(str(alvo) + SUFIXO)
    linhas = []
    for rotulo, caminho in (("alvo      ", alvo), ("arquivado ", arquivado)):
        aceito, motivo, info = _resumo(caminho)
        if not caminho.exists():
            linhas.append("  %s %s: (nao existe)" % (rotulo, caminho.name))
            continue
        data = time.strftime("%d/%m/%Y %H:%M",
                             time.localtime(caminho.stat().st_mtime))
        veredito = "ACEITO pela leitura" if aceito else "RECUSADO pela leitura"
        linhas.append("  %s %s: %d B, %s -- %s"
                      % (rotulo, caminho.name, caminho.stat().st_size, data,
                         veredito))
        if not aceito:
            linhas.append("      motivo: %s" % motivo)
        elif info is not None:
            linhas.append("      poses=%s span=%s rms=%s | %s"
                          % (info.get("poses"),
                             "-" if info.get("span") is None
                             else "%.1f%%" % (info["span"] * 100.0),
                             info.get("rms"), info.get("congelado_em")))
    return linhas


def por_que_o_alvo_esta_fora_do_ar(alvo=ALVO_PADRAO):
    """(linhas) do diagnostico campo a campo: por que este arquivo e' recusado."""
    alvo = Path(alvo)
    linhas = ["por que o arquivo atual esta' fora do ar:"]
    if not alvo.exists():
        linhas.append("  (o arquivo nem existe -- nada a diagnosticar)")
        return linhas
    try:
        dados = np.load(alvo, allow_pickle=True)
    except Exception as exc:                                # noqa: BLE001 - sonda
        linhas.append("  nao consegui ler: %s" % exc)
        return linhas
    faltando = [campo for campo in ("poses", "span") if campo not in dados.files]
    if faltando:
        linhas.append("  campos exigidos AUSENTES: %s -- e' esta a recusa"
                      % ", ".join(faltando))
        linhas.append("     (a leitura recusa arquivo SEM cobertura medida: nao ha' "
                      "como saber se o modelo presta)")
    if "checkerboard_size" in dados.files:
        placa = tuple(int(v) for v in np.ravel(dados["checkerboard_size"]))
        if placa != tuple(sc.CHECKERBOARD_SIZE):
            linhas.append("  placa gravada %dx%d != placa ATUAL %dx%d (aviso, nao "
                          "recusa: as intrinsecas sao da CAMERA)"
                          % (placa + tuple(sc.CHECKERBOARD_SIZE)))
    if "square_size_m" in dados.files:
        quadrado = float(np.ravel(dados["square_size_m"])[0]) * 1000.0
        if abs(quadrado - sc.SQUARE_SIZE_M * 1000.0) > 1e-6:
            linhas.append("  quadrado gravado %.0f mm != %.0f mm atual (idem: aviso)"
                          % (quadrado, sc.SQUARE_SIZE_M * 1000.0))
    if "frozen_at" in dados.files:
        congelado_em = str(np.ravel(dados["frozen_at"])[0])
        linhas.append("  frozen_at: %s" % congelado_em)
        if "PALIATIVO" in congelado_em.upper():
            linhas.append("     -> o proprio arquivo se declara paliativo: vies "
                          "desconhecido (o texto cita ate' ~46 px).")
    if "rms" in dados.files:
        linhas.append("  RMS gravado: %.3f px (NAO denuncia o problema: mede so' a "
                      "coerencia INTERNA do arquivo)"
                      % float(np.ravel(dados["rms"])[0]))
    return linhas


def arquiva(alvo=ALVO_PADRAO, forcar=False, mover=shutil.move):
    """(ok, mensagem) de aposentar o alvo em `<alvo><SUFIXO>`.

    Nunca sobrescreve um arquivado existente (perder a referencia seria trocar um
    problema por outro): exige `forcar`, que guarda o anterior com carimbo de
    tempo. `mover` e' injetavel para o teste nao tocar em disco.
    """
    alvo = Path(alvo)
    arquivado = Path(str(alvo) + SUFIXO)
    if not alvo.exists():
        return False, ("nao ha' o que arquivar: %s nao existe" % alvo.name)
    if arquivado.exists() and not forcar:
        return False, (
            "ja' existe %s -- nao sobrescrevo a referencia. Se este arquivo "
            "atual e' outro paliativo, rode com --forcar (o anterior vira "
            "%s.<carimbo>)." % (arquivado.name, arquivado.name))
    if arquivado.exists() and forcar:
        carimbo = time.strftime("%Y%m%d_%H%M%S")
        destino_antigo = Path("%s.%s" % (arquivado, carimbo))
        mover(str(arquivado), str(destino_antigo))
    mover(str(alvo), str(arquivado))
    return True, ("arquivado: %s -> %s (nada foi apagado; a leitura nao carrega "
                  "este nome)" % (alvo.name, arquivado.name))


def restaura(alvo=ALVO_PADRAO, mover=shutil.move):
    """(ok, mensagem) de desfazer: traz o arquivado de volta ao nome carregado."""
    alvo = Path(alvo)
    arquivado = Path(str(alvo) + SUFIXO)
    if not arquivado.exists():
        return False, "nao existe %s para restaurar" % arquivado.name
    if alvo.exists():
        return False, ("%s ja' existe -- nao sobrescrevo. Arquive o atual antes "
                       "(ou renomeie na mao)." % alvo.name)
    mover(str(arquivado), str(alvo))
    return True, "restaurado: %s -> %s" % (arquivado.name, alvo.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arquivo", default=str(ALVO_PADRAO),
                        help="npz alvo (padrao: %s)" % ALVO_PADRAO.name)
    parser.add_argument("--sim", action="store_true",
                        help="EXECUTA (sem isto e' dry-run: so' mostra)")
    parser.add_argument("--forcar", action="store_true",
                        help="permite arquivar por cima (o anterior ganha carimbo)")
    parser.add_argument("--restaurar", action="store_true",
                        help="desfaz: traz o arquivado de volta ao nome carregado")
    args = parser.parse_args(argv)
    alvo = Path(args.arquivo)

    print("conserta_intrinsecas_kinect: aposentar o modelo do Kinect FORA DO AR")
    print("O que este programa NAO faz: inventar `poses`/`span` para o arquivo "
          "passar no portao (isso cimentaria o vies que o proprio arquivo "
          "denuncia).\n")

    for linha in estado_atual(alvo):
        print(linha)
    print("")
    for linha in por_que_o_alvo_esta_fora_do_ar(alvo):
        print(linha)

    print("")
    if args.restaurar:
        if not args.sim:
            print("DRY-RUN: nada foi mexido (rode com --sim para executar).")
        ok, mensagem = restaura(alvo) if args.sim else (None, "(dry-run)")
        print("restaurar: %s" % mensagem)
        return 0 if (not args.sim or ok) else 1

    aceito, _, _ = _resumo(alvo)
    if aceito:
        print("Nada a fazer: o arquivo atual PASSA nos portoes da leitura -- "
              "arquiva'-lo tiraria da bancada um modelo que vale. Se voce quer "
              "mesmo refazer o congelamento, rode `freeze_kinect_direto.py` "
              "(ele substitui so' quando passa nos tres portoes).")
        return 0

    if not args.sim:
        print("DRY-RUN: nada foi mexido. Para arquivar de verdade:")
        print("    .venv\\Scripts\\python.exe tools\\conserta_intrinsecas_kinect.py "
              "--sim")
        print("Depois: rode `freeze_kinect_direto.py` (bancada) para gravar o "
              "modelo BOM no lugar.")
        return 0

    ok, mensagem = arquiva(alvo, forcar=args.forcar)
    print(mensagem)
    if not ok:
        return 1
    print("\nProximo passo (bancada): .venv\\Scripts\\python.exe "
          "freeze_kinect_direto.py")
    print("Confira antes que o sensor entrega frame: .venv\\Scripts\\python.exe "
          "tools\\checa_kinect.py")
    print("Para voltar atras: --sim --restaurar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
