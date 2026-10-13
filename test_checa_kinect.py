"""Suite do `tools/checa_kinect.py` -- a pre-checagem do sensor, sem hardware.

O QUE ESTA SUITE TRAVA (bancada de 07/10/2026)
  O custo do bug que ela cobre nao foi o erro: foi o MOMENTO. O operador montou o
  tabuleiro 12+ vezes, a 0,35 m, e o `freeze_kinect_direto.py` disse "NADA
  GRAVADO" porque o SENSOR estava mudo (o runtime abria em 0,0 s reportando
  1920x1080, e nenhum frame saia em 20 s -- em color, depth, body e nos tres
  juntos). A pre-checagem existe para gastar 10 s em vez de 10 min:

   1. o veredito do pykinect2 vem do `kinect_compat` (Patch A/B/D) -- quando o
      import falha, o codigo de saida e' 2 (nem abre), nao 1 (nao entrega);
   2. `checa_streams` testa CADA stream e reporta qual respondeu: um stream vivo
      muda o diagnostico (e' o que separa "link USB voltou" de "sensor morto");
   3. um stream que EXPLODE vira resultado ("nao abriu: ..."), nunca traceback;
   4. `runtime.close()` e' chamado mesmo quando a consulta levanta -- sem isso o
      teste de streams deixa o sensor preso e a Fase 1 herda o problema;
   5. o veredito do `kinect_intrinsics.npz` vem de
      `stereo_calibration.carrega_intrinsecas_kinect` -- a MESMA regra da
      calibracao. Foi uma regra PARALELA (a recusa por placa) que deixou o
      diagnostico dizer "congelado: SIM" enquanto a calibracao re-estimava o
      Kinect em silencio (01/10/2026);
   6. as webcams vem do mapa indice->NOME do Windows, na MESMA ordem do
      `CAP_DSHOW`: e' o nome, nao o indice, que denuncia "a auxiliar" ser a
      webcam do laptop (o indice do OpenCV e' a posicao no USB e troca sozinho);
   7. `--sem-webcams`/`--sem-calibracoes` de fato desligam os blocos -- o
      operador que so' quer o sensor nao paga a abertura das cameras.
"""
from __future__ import annotations

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
import checa_kinect as ck                                   # noqa: E402

casos = 0

# --- 1) o veredito do pykinect2 separa "nem abre" de "nao entrega" -----------
ok, linhas = ck.checa_pykinect2()
texto = "\n".join(linhas)
assert "compatibilidade" in texto, texto
if ck.PYKINECT2_ERRO is None:
    #: Nesta maquina o pykinect2 importa (com os patches): o veredito e' OK e a
    #: suite NAO pode exigir falha -- o que se trava e' que o compat entrou.
    assert ok is True, linhas
    assert "OK (PyKinectV2 + PyKinectRuntime)" in texto, texto
else:                                                       # pragma: no cover
    assert ok is False, linhas
    assert "FALHOU" in texto, texto
casos += 1

# --- 2) streams: cada um testado sozinho, e o que responde e' reportado ------
class SensorFalso:
    """Runtime falso com controle por stream e contagem de `close()`."""

    def __init__(self, respondem=(), explode_em=()):
        self.respondem = set(respondem)
        self.explode_em = set(explode_em)
        self.fechou = 0

    def _responde(self, tipo):
        if tipo in self.explode_em:
            raise OSError("stream %s morreu" % tipo)
        return tipo in self.respondem

    def has_new_color_frame(self):
        return self._responde("color")

    def has_new_depth_frame(self):
        return self._responde("depth")

    def has_new_body_frame(self):
        return self._responde("body")

    def close(self):
        self.fechou += 1


def fabrica(fonte_por_flag, criados):
    def criar(flag):
        tipo = fonte_por_flag[flag]
        runtime = SensorFalso(respondem=criados.get("respondem", ()),
                              explode_em=criados.get("explode_em", ()))
        criados.setdefault("instancias", []).append(runtime)
        return runtime
    return criar


flags = {getattr(ck.PyKinectV2, nome): nome.replace("FrameSourceTypes_", "").lower()
         for _, nome in ck.STREAMS}
assert sorted(flags.values()) == ["body", "color", "depth"], flags

criados = {"respondem": ()}
resultados = ck.checa_streams(fabrica(flags, criados), espera_s=0.05)
mapa = {nome: estado for nome, estado, _ in resultados}
assert set(mapa) == {"color", "depth", "body"}, mapa
assert all(estado == "NENHUM frame" for estado in mapa.values()), mapa
assert all(instancia.fechou == 1 for instancia in criados["instancias"]), \
    "sensor mudo: cada runtime tem de ser FECHADO"
casos += 1

# --- 3) um stream que responde muda o diagnostico, e so' ele ----------------
criados = {"respondem": ("depth",)}
resultados = ck.checa_streams(fabrica(flags, criados), espera_s=0.05)
mapa = {nome: (estado, tempo) for nome, estado, tempo in resultados}
assert mapa["depth"][0] == "ENTREGA frame", mapa
assert mapa["depth"][1] is not None, "o tempo do frame tem de ser reportado"
assert mapa["color"][0] == "NENHUM frame", mapa
assert mapa["body"][0] == "NENHUM frame", mapa
assert all(i.fechou == 1 for i in criados["instancias"]), criados
casos += 1

# --- 4) stream que EXPLODE vira resultado, e o runtime ainda e' fechado ------
criados = {"respondem": (), "explode_em": ("body",)}
resultados = ck.checa_streams(fabrica(flags, criados), espera_s=0.05)
mapa = {nome: estado for nome, estado, _ in resultados}
assert "erro ao consultar" in mapa["body"], mapa
assert mapa["color"] == "NENHUM frame", mapa
assert all(i.fechou == 1 for i in criados["instancias"]), \
    "consulta que levanta NAO pode deixar o sensor aberto"
casos += 1

# --- 5) runtime que nao abre (OSError do pykinect2) e' resultado, nao erro ---
def fabrica_que_nao_abre(flag):
    raise OSError("Kinect not found")


resultados = ck.checa_streams(fabrica_que_nao_abre, espera_s=0.05)
assert all("nao abriu" in estado for _, estado, _ in resultados), resultados
casos += 1

# --- 6) o veredito do npz congelado e' o da LEITURA (nenhuma regra paralela) --
with tempfile.TemporaryDirectory(dir=str(RAIZ)) as pasta:
    pasta = Path(pasta)
    np.savez(pasta / "kinect_intrinsics.npz",
             kinect_matrix=np.eye(3), kinect_dist=np.zeros(5),
             kinect_size=np.asarray([1920, 1080]), rms=0.3)
    linhas = ck.checa_calibracoes(pasta)
    texto = "\n".join(linhas)
    assert "RECUSADO" in texto, texto
    _, _, info = sc.carrega_intrinsecas_kinect(pasta / "kinect_intrinsics.npz")
    assert info["motivo"] in texto, "o motivo tem de ser o MESMO da leitura"
    casos += 1

# --- 7) npz congelado que PASSA nos portoes: mostra poses/span/rms ----------
with tempfile.TemporaryDirectory(dir=str(RAIZ)) as pasta:
    pasta = Path(pasta)
    sc.salva_intrinsecas_kinect(np.eye(3), np.zeros(5), (1920, 1080), 0.25,
                                sc.KINECT_INTRINSICS_MIN_POSES,
                                sc.KINECT_INTRINSICS_MIN_SPAN + 0.05,
                                caminho=pasta / "kinect_intrinsics.npz")
    texto = "\n".join(ck.checa_calibracoes(pasta))
    assert "ACEITO" in texto, texto
    assert "poses=%d" % sc.KINECT_INTRINSICS_MIN_POSES in texto, texto
    assert "span=" in texto, texto
    casos += 1

# --- 8) sem arquivo: a ferramenta diz o que fazer, nao fica muda ------------
with tempfile.TemporaryDirectory(dir=str(RAIZ)) as pasta:
    texto = "\n".join(ck.checa_calibracoes(Path(pasta)))
    assert "NAO EXISTE" in texto, texto
    assert "freeze_kinect_direto.py" in texto, texto
    casos += 1

# --- 9) `--sem-webcams`/`--sem-calibracoes` desligam os blocos de verdade ----
import contextlib                                           # noqa: E402
import io                                                   # noqa: E402

buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    codigo = ck.main(["--sem-webcams", "--sem-calibracoes", "--espera-s", "0.05",
                      "--espera-stream-s", "0.05", "--tentativas", "1"])
saida = buffer.getvalue()
assert "4) webcams" not in saida, "`--sem-webcams` nao desligou o bloco"
assert "5) calibracoes" not in saida, "`--sem-calibracoes` nao desligou o bloco"
assert "1) pykinect2 + compatibilidade" in saida, saida
assert "3) abertura OFICIAL do projeto" in saida, saida
#: Esta maquina HOJE nao entrega frame: o codigo tem de ser 1 (ou 2 sem import).
assert codigo in (1, 2), codigo
if codigo == 1:
    assert "NAO entregou frame" in saida, saida
    assert "NAO monte o tabuleiro" in saida, saida
casos += 1

# --- 10) o fake de `abre_kinect` conta as REABERTURAS ------------------------
#: Nao basta a mensagem: `sc.abre_kinect` tem de TENTAR de novo. Um runtime que
#: fica mudo e' reaberto; a suite garante que a reabertura acontece (e que o
#: runtime morto e' fechado antes da proxima tentativa).
class ContadorDeReaberturas:
    criados = 0

    def __init__(self):
        ContadorDeReaberturas.criados += 1

    def has_new_color_frame(self):
        return False

    def close(self):
        pass


original = sc.PyKinectRuntime
sc.PyKinectRuntime = lambda *a, **k: ContadorDeReaberturas()
try:
    try:
        sc.abre_kinect(espera_s=3.0, tentativas=3)
    except RuntimeError as exc:
        assert "NAO entregou frame" in str(exc), exc
        assert "tentei reabrir 3 vezes" in str(exc), exc
    else:
        raise AssertionError("abre_kinect aceitou um Kinect que nunca entrega")
finally:
    sc.PyKinectRuntime = original
assert ContadorDeReaberturas.criados == 3, ContadorDeReaberturas.criados
casos += 1

# --- 11) espera curta NAO reabre (o teste nao pode virar espera de minutos) --
ContadorDeReaberturas.criados = 0
sc.PyKinectRuntime = lambda *a, **k: ContadorDeReaberturas()
try:
    try:
        sc.abre_kinect(espera_s=0.1)
    except RuntimeError:
        pass
finally:
    sc.PyKinectRuntime = original
assert ContadorDeReaberturas.criados == 1, (
    "espera curta (<= 2 s) tem de valer UMA tentativa: %d"
    % ContadorDeReaberturas.criados)
casos += 1

print("CHECA_KINECT_OK: %d casos (veredito do compat separa 'nem abre' de 'nao "
      "entrega'; cada stream testado sozinho; stream que explode vira resultado; "
      "runtime sempre fechado; npz julgado pela MESMA regra da leitura; blocos "
      "--sem-* desligam; abre_kinect reabre e espera curta nao reabre)" % casos)
