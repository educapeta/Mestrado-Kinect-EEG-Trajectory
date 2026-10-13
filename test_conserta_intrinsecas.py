"""Suite do `tools/conserta_intrinsecas_kinect.py` -- aposentar o paliativo.

O QUE ESTA SUITE TRAVA (bancada de 07/10/2026)
  O `kinect_intrinsics.npz` em disco e' o PALIATIVO de 30/09/2026 (placa antiga
  5x3/57 mm, span 11-13%) e NAO registra `poses` nem `span`. A leitura recusa por
  isso -- e a consequencia nao e' um aviso, e' o Kinect voltando a ser
  RE-ESTIMADO a cada sessao (os dois npz de par em disco tem
  `kinect_intrinsics_source='sessao'`; medido: dfx 12,5 px / dcy -23,5 px entre
  duas sessoes da MESMA bancada).

  O programa arquiva (renomeia) o paliativo, sem apagar nada e sem forjar os
  campos que decidem se o modelo presta. Os casos abaixo travam:

   1. o sufixo de arquivo esta' FORA do padrao `.npz` e a LEITURA nao carrega o
      arquivado -- se carregasse, arquivar nao mudaria nada;
   2. DRY-RUN e' o padrao: sem `--sim` NADA e' tocado (a regra do plano era "nao
      apagar o paliativo antes de tudo", e um comando que mexe por engano nisso
      custa uma bancada);
   3. arquivar preserva o arquivo INTEIRO (bytes iguais) e o veredito da leitura
      sobre ele continua o MESMO -- o paliativo segue servindo de referencia
      para a auditoria comparar;
   4. com o alvo ja' arquivado o programa NAO sobrescreve a referencia: exige
      `--forcar`, que guarda o anterior com carimbo de tempo;
   5. `--restaurar` desfaz, e nao sobrescreve um alvo que exista;
   6. um arquivo que PASSA nos portoes nao e' arquivado -- tirar da bancada um
      modelo que vale seria o erro oposto;
   7. o diagnostico campo a campo nomeia a recusa REAL (`poses`/`span` ausentes),
      avisa da placa/quadrado e cita o `PALIATIVO` do `frozen_at` -- e' o que o
      operador le' em vez de "rejeitou a calibracao";
   8. `_resumo` usa `sc.carrega_intrinsecas_kinect`: o veredito da ferramenta e' o
      MESMO que a calibracao vai dar (regra paralela foi o bug de 01/10/2026).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
if str(RAIZ / "tools") not in sys.path:
    sys.path.insert(0, str(RAIZ / "tools"))

import numpy as np                                          # noqa: E402

import stereo_calibration as sc                             # noqa: E402
import conserta_intrinsecas_kinect as ci                    # noqa: E402

casos = 0

#: A pasta de trabalho fica DENTRO da raiz: o `sc.KINECT_INTRINSICS_FILE` e' um
#: Path da raiz e os testes usam `--arquivo`, entao nada aqui toca no npz real.
PASTA_PAI = RAIZ

#: O paliativo de 30/09/2026 e' RECUSADO pela leitura -- mas em 09/10/2026 o
#: freeze da Fase 1 passou e REESCREVEU o `kinect_intrinsics.npz` real com um
#: modelo VALIDO (13 poses, rms 0.674, aceito com o gate em 0.75 px do item 46).
#: O assert antigo ("tem de ser RECUSADO") comecou a FALHAR por isso. A suite NAO
#: depende do arquivo real: ela testa o conserto com paliativos SINTETICOS
#: (`_paliativo` em pasta temporaria), entao aqui so' REPORTAMOS o estado real --
#: a premissa que importa (aposentar um paliativo sem forjar poses) e' coberta
#: la embaixo, com arquivos sinteticos.
_veredito_real = sc.carrega_intrinsecas_kinect(RAIZ / "kinect_intrinsics.npz")[0]
print(f"  kinect_intrinsics.npz real em disco: "
      f"{'ACEITO (freeze da Fase 1 ok)' if _veredito_real is not None else 'recusado (paliativo)'}")


def _sha1(caminho):
    return hashlib.sha1(Path(caminho).read_bytes()).hexdigest()


def _paliativo(caminho):
    """Grava um npz no formato do paliativo de 30/09/2026 (SEM poses/span).

    O arquivo e' aberto por handle DE PROPOSITO: `np.savez("x.arquivado")` grava
    `x.arquivado.npz` (o numpy acrescenta a extensao quando o nome nao termina
    nela), e o teste do arquivado perdia o arquivo por isso.
    """
    with open(caminho, "wb") as arquivo:
        np.savez(arquivo,
                 version=1,
                 kinect_matrix=np.asarray([[1053.8927, 0.0, 957.813],
                                           [0.0, 1048.1415, 571.1753],
                                           [0.0, 0.0, 1.0]]),
                 kinect_dist=np.asarray([-0.081, 0.815, -0.0016, 0.0048, -1.787]),
                 kinect_size=np.asarray([1920, 1080]),
                 checkerboard_size=np.asarray([5, 3]),
                 square_size_m=0.057,
                 rms=0.194729,
                 frozen_at=("2026-09-30 15:48 PALIATIVO: medoide de 4 modelos "
                            "'sessao' (span 11-13%, sem gate de 35%)"))


def _layout():
    """Cria a pasta de teste (dentro da raiz) e o paliativo dentro dela."""
    return tempfile.TemporaryDirectory(dir=str(PASTA_PAI))


# --- 1) o sufixo esta' FORA do padrao .npz e a leitura nao carrega o arquivado -
assert not ci.SUFIXO.endswith(".npz"), ci.SUFIXO
assert ci.SUFIXO == sc.SUFIXO_ARQUIVADO, (ci.SUFIXO, sc.SUFIXO_ARQUIVADO)
assert ci.ALVO_PADRAO == sc.KINECT_INTRINSICS_FILE, (ci.ALVO_PADRAO,
                                                     sc.KINECT_INTRINSICS_FILE)
casos += 1

# --- 2) DRY-RUN e' o padrao: NADA e' tocado ----------------------------------
with _layout() as pasta:
    alvo = Path(pasta) / "kinect_intrinsics.npz"
    _paliativo(alvo)
    antes = _sha1(alvo)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        codigo = ci.main(["--arquivo", str(alvo)])
    saida = buffer.getvalue()
    assert codigo == 0, codigo
    assert alvo.exists(), "DRY-RUN mexeu no arquivo!"
    assert _sha1(alvo) == antes, "DRY-RUN mudou os bytes do arquivo!"
    assert not Path(str(alvo) + ci.SUFIXO).exists(), "DRY-RUN arquivou!"
    assert "DRY-RUN" in saida, saida
    assert "RECUSADO pela leitura" in saida, saida
    casos += 1

    # --- 3) `--sim` arquiva, preserva os bytes e o veredito da leitura ------
    aceito_antes, motivo_antes, _ = ci._resumo(alvo)
    assert aceito_antes is False, "o paliativo tem de ser RECUSADO pela leitura"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        codigo = ci.main(["--arquivo", str(alvo), "--sim"])
    saida = buffer.getvalue()
    assert codigo == 0, (codigo, saida)
    arquivado = Path(str(alvo) + ci.SUFIXO)
    assert arquivado.exists(), "nao arquivou"
    assert not alvo.exists(), "o nome carregado tem de ficar livre"
    assert _sha1(arquivado) == antes, "o arquivado NAO e' o mesmo arquivo"
    _, motivo_depois, _ = ci._resumo(arquivado)
    #: O motivo cita o NOME do arquivo no comeco; arquivar troca so' o nome. A
    #: recusa (o que decide) tem de continuar a MESMA -- senao a referencia
    #: guardada deixaria de servir para a auditoria comparar.
    assert motivo_antes.split(":", 1)[1] == motivo_depois.split(":", 1)[1], \
        (motivo_depois, motivo_antes)
    assert "arquivado" in motivo_depois, motivo_depois
    #: E o decisivo: com o alvo fora, a leitura do caminho PADRAO nao acha mais
    #: o paliativo (ela le' o nome exato) -- e' isso que deixa o proximo freeze
    #: gravar o arquivo bom sem competir com ele.
    _, _, info = sc.carrega_intrinsecas_kinect(alvo)
    assert info.get("motivo") and "nao existe" in info["motivo"], info
    casos += 1

    # --- 4) nao sobrescreve a referencia sem --forcar -----------------------
    _paliativo(alvo)                                        # outro paliativo
    sha_novo = _sha1(alvo)
    ok, mensagem = ci.arquiva(alvo)
    assert ok is False, "arquivou por cima da referencia!"
    assert "ja' existe" in mensagem, mensagem
    assert _sha1(arquivado) == antes, "a referencia antiga mudou"
    assert alvo.exists() and _sha1(alvo) == sha_novo, "o alvo foi tocado"
    casos += 1

    # --- 4b) com --forcar o anterior ganha carimbo de tempo -----------------
    ok, mensagem = ci.arquiva(alvo, forcar=True)
    assert ok is True, mensagem
    assert _sha1(arquivado) == sha_novo, "o novo arquivado nao e' o atual"
    carimbados = sorted(p.name for p in Path(pasta).iterdir()
                        if p.name.startswith(arquivado.name + "."))
    assert len(carimbados) == 1, carimbados
    assert _sha1(Path(pasta) / carimbados[0]) == antes, \
        "o paliativo original nao foi preservado no carimbo"
    casos += 1

    # --- 5) --restaurar desfaz, e nao sobrescreve um alvo existente ---------
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        codigo = ci.main(["--arquivo", str(alvo), "--sim", "--restaurar"])
    assert codigo == 0, (codigo, buffer.getvalue())
    assert alvo.exists(), "nao restaurou"
    assert _sha1(alvo) == sha_novo, "restaurou o arquivo errado"
    assert not arquivado.exists(), "sobrou o arquivado depois de restaurar"
    #: Com o alvo ocupado, restaurar RECUSA (nao sobrescreve nada).
    _paliativo(arquivado)
    ok, mensagem = ci.restaura(alvo)
    assert ok is False and "ja' existe" in mensagem, mensagem
    casos += 1

# --- 6) arquivo que PASSA nos portoes nao e' arquivado ----------------------
with _layout() as pasta:
    alvo = Path(pasta) / "kinect_intrinsics.npz"
    sc.salva_intrinsecas_kinect(np.eye(3), np.zeros(5), (1920, 1080), 0.25,
                                sc.KINECT_INTRINSICS_MIN_POSES,
                                sc.KINECT_INTRINSICS_MIN_SPAN + 0.05,
                                caminho=alvo)
    sha_bom = _sha1(alvo)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        codigo = ci.main(["--arquivo", str(alvo), "--sim"])
    saida = buffer.getvalue()
    assert codigo == 0, (codigo, saida)
    assert alvo.exists(), "arquivou um modelo que PASSA nos portoes!"
    assert _sha1(alvo) == sha_bom, "mexeu num modelo bom"
    assert "Nada a fazer" in saida, saida
    assert "ACEITO pela leitura" in saida, saida
    casos += 1

# --- 7) diagnostico campo a campo nomeia a recusa real ---------------------
with _layout() as pasta:
    alvo = Path(pasta) / "kinect_intrinsics.npz"
    _paliativo(alvo)
    texto = "\n".join(ci.por_que_o_alvo_esta_fora_do_ar(alvo))
    assert "poses, span" in texto, texto
    assert "5x3" in texto and "7x5" in texto, texto
    assert "57 mm" in texto and "45 mm" in texto, texto
    assert "PALIATIVO" in texto, texto
    assert "~46 px" in texto, texto
    assert "coerencia INTERNA" in texto, texto
    casos += 1

# --- 8) o veredito e' o da LEITURA, nao uma regra paralela -----------------
with _layout() as pasta:
    alvo = Path(pasta) / "kinect_intrinsics.npz"
    #: Um npz com poses/span mas com o RMS acima do portao: a LEITURA recusa; a
    #: ferramenta tem de dizer o mesmo (nada de "placa diferente -> recusa" nem
    #: de aprovacao por existencia).
    np.savez(alvo, kinect_matrix=np.eye(3), kinect_dist=np.zeros(5),
             kinect_size=np.asarray([1920, 1080]),
             poses=sc.KINECT_INTRINSICS_MIN_POSES,
             span=sc.KINECT_INTRINSICS_MIN_SPAN + 0.02,
             rms=sc.KINECT_INTRINSICS_MAX_RMS * 3)
    aceito, motivo, _ = ci._resumo(alvo)
    assert aceito is False, "RMS acima do portao tem de ser recusado"
    _, _, info_leitura = sc.carrega_intrinsecas_kinect(alvo)
    assert info_leitura["motivo"] == motivo, (motivo, info_leitura)
    assert "RMS" in motivo, motivo
    casos += 1

# --- 9) pasta vazia: diz o que fazer em vez de estourar --------------------
with _layout() as pasta:
    alvo = Path(pasta) / "kinect_intrinsics.npz"
    texto = "\n".join(ci.estado_atual(alvo))
    assert "(nao existe)" in texto, texto
    ok, mensagem = ci.arquiva(alvo)
    assert ok is False and "nao existe" in mensagem, mensagem
    ok, mensagem = ci.restaura(alvo)
    assert ok is False and "nao existe" in mensagem, mensagem
    casos += 1

print("CONSERTA_INTRINSECAS_OK: %d casos (sufixo fora do padrao .npz e leitura "
      "nao carrega o arquivado; dry-run e' o padrao e nao toca em nada; --sim "
      "preserva os bytes e o veredito; nao sobrescreve referencia sem --forcar "
      "(com carimbo); --restaurar desfaz e nao sobrescreve; modelo que passa nos "
      "portoes NAO e' arquivado; diagnostico nomeia poses/span, placa, quadrado, "
      "PALIATIVO e o RMS que nao denuncia)" % casos)
