"""Teste da descoberta do RUNTIME do GDS (g.tec Suite) pelo paradigma.

Contexto medido nesta bancada em 29/09/2026, depois de instalar o g.tec Suite
2024:

* o instalador registra APENAS `C:\\Program Files\\gtec\\gNEEDaccess` no PATH de
  MAQUINA, e e' dessa pasta que saem `GDSClientAPI.dll`/`GDSServer.dll`;
* o `gtec_gds` 1.7.1 procura essas DLLs PELO NOME, entao um terminal/IDE aberto
  ANTES da instalacao herda o PATH antigo e falha com `GDS library not found` --
  a MESMA mensagem do Suite ausente;
* `GNEEDACCESS_DIR` (que a mensagem ANTIGA mandava definir) faz o wrapper exigir
  os headers C e falhar com `GDS header not found ... unset GNEEDACCESS_DIR`;
* o wrapper compilado CACHEIA a falha (`_initialize_failed`): medido com
  `_gtec_order.py` que definir `GDS_LIBRARY`/PATH depois de uma chamada que
  falhou NAO conserta (a chamada seguinte falha igual), enquanto definir depois
  do `import` e ANTES da primeira chamada funciona -- por isso o reparo do
  paradigma e' preemptivo.

Blocos, todos sem hardware e sem tocar o GDS de verdade (duble no wrapper):
1. `_gds_dll_instalada` prefere o que o usuario apontou (`GDS_LIBRARY`);
2. ... e cai para `GNEEDACCESS_DIR`/pastas padrao, devolvendo `(None, None)`
   quando nao ha' nada no disco (e' o que separa "nao instalado" de "instalado e
   fora do PATH desta sessao");
3. `_gds_apontar_para` define `GDS_LIBRARY` com o caminho completo e prepende a
   pasta no PATH -- uma vez so' (idempotente) e sem sobrescrever o que o usuario
   definiu;
4. `_gds_no_path` responde o que o loader do Windows responde (PATH da sessao);
5. `_check_gtec_environment` com PATH "velho" e Suite no disco: repara, AVISA e
   termina OK (antes disto a falha era fatal e a mensagem mandava instalar o que
   ja' estava instalado);
6. sem Suite no disco: mensagem "instale o g.tec Suite" (e sem a dica ruim de
   GNEEDACCESS_DIR);
7. `GNEEDACCESS_DIR` sem os headers C: removida da sessao, com aviso, e a
   mensagem explica;
8. caminho feliz: com o GDS respondendo, imprime os dispositivos;
9. `build_source` sem aparelho: a construcao do `gp.GNautilus` morre dentro do
   ioiocore/gpype com "No g.Nautilus devices found" -- vira mensagem acionavel,
   e qualquer outro erro continua passando intacto.
"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import eeg_motor_paradigm as p

try:
    # O import do wrapper compilado faz uma TENTATIVA de Initialize: com o PATH
    # desta sessao sem a pasta do g.tec ele imprime duas linhas de erro (e cacheia
    # a falha). Aqui isso e' ruido -- os blocos 5-8 dublam `ConnectedDevices` de
    # proposito --, entao a importacao roda silenciada.
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        from gtec_gds.lib import gtec_gds_wrapper
except ImportError:                        # maquina sem o pacote g.tec
    gtec_gds_wrapper = None

_ORIGINAIS_ENV = {nome: os.environ.get(nome)
                  for nome in ("PATH", "GDS_LIBRARY", "GNEEDACCESS_DIR")}
_PADRAO_GDS = p.GDS_DLL_DIRS
_NO_PATH_ORIGINAL = p._gds_no_path
_CONECTADOS_ORIGINAL = (None if gtec_gds_wrapper is None
                        else gtec_gds_wrapper.ConnectedDevices)


def _restaurar_ambiente():
    for nome, valor in _ORIGINAIS_ENV.items():
        if valor is None:
            os.environ.pop(nome, None)
        else:
            os.environ[nome] = valor


def _fala(callable_, *args, **kwargs):
    """(resultado, texto impresso) -- sem sujar o console da suite."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        resultado = callable_(*args, **kwargs)
    return resultado, buffer.getvalue()


print("=" * 62)
print("GTEC_RUNTIME -- descoberta do runtime do GDS")
print("=" * 62)

pasta_a = tempfile.mkdtemp(prefix="gds_a_")
pasta_b = tempfile.mkdtemp(prefix="gds_b_")
try:
    with open(os.path.join(pasta_a, "GDSClientAPI.dll"), "wb") as handle:
        handle.write(b"fake")
    with open(os.path.join(pasta_b, "GDSServer.dll"), "wb") as handle:
        handle.write(b"fake")

    # -----------------------------------------------------------------------
    # 1) GDS_LIBRARY tem prioridade na busca no disco
    # -----------------------------------------------------------------------
    os.environ["GDS_LIBRARY"] = os.path.join(pasta_a, "GDSClientAPI.dll")
    os.environ.pop("GNEEDACCESS_DIR", None)
    pasta, dll = p._gds_dll_instalada()
    assert pasta == pasta_a, (pasta, pasta_a)
    assert dll == os.path.join(pasta_a, "GDSClientAPI.dll"), dll
    print("  (1) GDS_LIBRARY tem prioridade na busca no disco")

    # -----------------------------------------------------------------------
    # 2) GNEEDACCESS_DIR e as pastas padrao do Suite; e o "nada no disco"
    # -----------------------------------------------------------------------
    os.environ.pop("GDS_LIBRARY", None)
    os.environ["GNEEDACCESS_DIR"] = pasta_b
    pasta, dll = p._gds_dll_instalada()
    assert pasta == pasta_b, pasta
    assert dll.endswith("GDSServer.dll"), dll
    os.environ.pop("GNEEDACCESS_DIR", None)
    p.GDS_DLL_DIRS = (pasta_a,)
    pasta, dll = p._gds_dll_instalada()
    assert pasta == pasta_a, pasta
    p.GDS_DLL_DIRS = ()
    vazio = p._gds_dll_instalada()
    assert vazio == (None, None), vazio
    p.GDS_DLL_DIRS = _PADRAO_GDS
    print("  (2) GNEEDACCESS_DIR/pastas padrao ok; disco sem GDS -> (None, None)")

    # -----------------------------------------------------------------------
    # 3) _gds_apontar_para: GDS_LIBRARY + PATH, idempotente
    # -----------------------------------------------------------------------
    os.environ.pop("GDS_LIBRARY", None)
    os.environ["PATH"] = r"C:\Windows"
    esperado = os.path.join(pasta_a, "GDSClientAPI.dll")
    p._gds_apontar_para(pasta_a, esperado)
    assert os.environ["GDS_LIBRARY"] == esperado, os.environ["GDS_LIBRARY"]
    assert os.environ["PATH"].startswith(pasta_a + os.pathsep), os.environ["PATH"]
    p._gds_apontar_para(pasta_a, esperado)
    entradas = [parte for parte in os.environ["PATH"].split(os.pathsep)
                if parte.strip().lower() == pasta_a.lower()]
    assert len(entradas) == 1, os.environ["PATH"]
    print("  (3) _gds_apontar_para: GDS_LIBRARY + PATH, sem duplicar pasta")

    # -----------------------------------------------------------------------
    # 4) _gds_no_path = "o PATH desta sessao resolve a DLL?"
    # -----------------------------------------------------------------------
    os.environ["PATH"] = r"C:\Windows"                  # sem a pasta do g.tec
    assert p._gds_no_path() is False, "PATH sem g.tec nao pode dar True"
    os.environ["PATH"] = pasta_a + os.pathsep + r"C:\Windows"
    assert p._gds_no_path() is True, "PATH com a pasta do gtec tem de dar True"
    os.environ["PATH"] = r"C:\Windows"
    print("  (4) _gds_no_path segue o PATH da sessao")

    # -----------------------------------------------------------------------
    # 5-8) A pre-checagem, com o wrapper dublado (o GDS de verdade nao e' tocado)
    # -----------------------------------------------------------------------
    if gtec_gds_wrapper is None:
        print("  (5-8) gtec_gds ausente nesta Python: checagem do GDS nao testada")
    else:
        def _sem_dispositivos():
            return []

        def _sem_runtime():
            raise RuntimeError(
                "GDS library not found; tried GDSClientAPI.dll, GDSServer.dll. "
                "Install GDS, or set GDS_LIBRARY to its name or full path.")

        def _sem_headers():
            raise RuntimeError(
                "GDS header not found: C:\\Program Files\\gtec\\gNEEDaccess\\"
                "GDSClientAPI.h. This build carries the GDS declarations "
                "itself - unset GNEEDACCESS_DIR to use them.")

        try:
            # -----------------------------------------------------------------
            # 5) PATH velho + Suite no disco: repara, avisa e segue
            # -----------------------------------------------------------------
            p.GDS_DLL_DIRS = (pasta_a,)
            p._gds_no_path = lambda: False
            os.environ.pop("GDS_LIBRARY", None)
            gtec_gds_wrapper.ConnectedDevices = _sem_dispositivos
            _, saida = _fala(p._check_gtec_environment)
            assert "nao no PATH desta sessao" in saida, saida
            assert "GDS_LIBRARY=" in saida, saida
            assert "nenhum dispositivo conectado" in saida, saida
            print("  (5) PATH velho com o Suite instalado: repara, avisa e segue")

            # -----------------------------------------------------------------
            # 6) Sem Suite no disco: a mensagem manda INSTALAR (e nao define
            #    GNEEDACCESS_DIR, que e' o que quebra este gtec_gds)
            # -----------------------------------------------------------------
            p.GDS_DLL_DIRS = ()
            os.environ.pop("GDS_LIBRARY", None)
            gtec_gds_wrapper.ConnectedDevices = _sem_runtime
            texto = ""
            try:
                p._check_gtec_environment()
            except RuntimeError as erro:
                texto = str(erro)
            else:
                raise AssertionError("sem runtime o GDS tinha de falhar")
            for token in ("instale o g.tec Suite", "GDS_LIBRARY",
                          "trocar de Python NAO resolve", "--source generator"):
                assert token in texto, (token, texto)
            assert "GNEEDACCESS_DIR" not in texto, texto
            print("  (6) sem Suite: manda instalar o g.tec Suite (nada de "
                  "GNEEDACCESS_DIR)")

            # -----------------------------------------------------------------
            # 7) GNEEDACCESS_DIR sem headers C: removida da sessao + mensagem
            # -----------------------------------------------------------------
            sem_header = tempfile.mkdtemp(prefix="gds_sem_header_")
            try:
                os.environ["GNEEDACCESS_DIR"] = sem_header
                os.environ["GDS_LIBRARY"] = os.path.join(pasta_a,
                                                         "GDSClientAPI.dll")
                p.GDS_DLL_DIRS = (pasta_a,)
                gtec_gds_wrapper.ConnectedDevices = _sem_headers
                texto = ""
                try:
                    p._check_gtec_environment()
                except RuntimeError as erro:
                    texto = str(erro)
                else:
                    raise AssertionError("sem headers o GDS tinha de falhar")
                assert "GNEEDACCESS_DIR" not in os.environ, \
                    "a variavel tinha de sair do ambiente"
                assert "GNEEDACCESS_DIR" in texto, texto
                assert "headers C" in texto, texto

                # ... e quando o reparo 1 e' o que DESTRAVA o GDS, ele so' avisa
                os.environ["GNEEDACCESS_DIR"] = sem_header
                gtec_gds_wrapper.ConnectedDevices = _sem_dispositivos
                _, saida = _fala(p._check_gtec_environment)
                assert "GNEEDACCESS_DIR" in saida, saida
                assert "nenhum dispositivo conectado" in saida, saida
                assert "GNEEDACCESS_DIR" not in os.environ
            finally:
                shutil.rmtree(sem_header, ignore_errors=True)
            print("  (7) GNEEDACCESS_DIR sem headers: ignorada, com aviso")

            # -----------------------------------------------------------------
            # 8) Caminho feliz: com o GDS respondendo, lista os dispositivos
            # -----------------------------------------------------------------
            os.environ["GDS_LIBRARY"] = os.path.join(pasta_a, "GDSClientAPI.dll")
            p._gds_no_path = lambda: True
            gtec_gds_wrapper.ConnectedDevices = lambda: [("Nautilus-1", "USB", 0)]
            _, saida = _fala(p._check_gtec_environment)
            assert "dispositivos: [('Nautilus-1', 'USB', 0)]" in saida, saida
            print("  (8) com dispositivo: lista o que o GDS devolveu")
        finally:
            if _CONECTADOS_ORIGINAL is not None:
                gtec_gds_wrapper.ConnectedDevices = _CONECTADOS_ORIGINAL
            _restaurar_ambiente()

    # -----------------------------------------------------------------------
    # 9) build_source sem aparelho: mensagem acionavel (e outros erros passam)
    # -----------------------------------------------------------------------
    _gn_original = p.gp.GNautilus
    _check_original = p._check_gtec_environment
    p._check_gtec_environment = lambda: None

    class _Args:
        source = "gnautilus"
        fs = 500.0
        channel_count = 32
        frame_size = 1

    try:
        def _sem_aparelho(**_kwargs):
            raise Exception("No g.Nautilus devices found.")

        p.gp.GNautilus = _sem_aparelho
        texto = ""
        try:
            p.build_source(_Args())
        except RuntimeError as erro:
            texto = str(erro)
        else:
            raise AssertionError("sem aparelho o build_source tinha de falhar")
        for token in ("NENHUM amplificador", "g.Recorder", "gtec_gds",
                      "--source generator"):
            assert token in texto, (token, texto)

        def _outro_erro(**_kwargs):
            raise ValueError("cabo solto")

        p.gp.GNautilus = _outro_erro
        try:
            p.build_source(_Args())
        except ValueError as erro:
            assert "cabo solto" in str(erro), erro
        else:
            raise AssertionError("erro alheio tinha de passar intacto")
    finally:
        p.gp.GNautilus = _gn_original
        p._check_gtec_environment = _check_original
    print("  (9) build_source sem aparelho: mensagem acionavel (e o mais passa)")
finally:
    _restaurar_ambiente()
    p.GDS_DLL_DIRS = _PADRAO_GDS
    p._gds_no_path = _NO_PATH_ORIGINAL
    if gtec_gds_wrapper is not None and _CONECTADOS_ORIGINAL is not None:
        gtec_gds_wrapper.ConnectedDevices = _CONECTADOS_ORIGINAL
    shutil.rmtree(pasta_a, ignore_errors=True)
    shutil.rmtree(pasta_b, ignore_errors=True)

print("GTEC_RUNTIME_OK: 9/9 blocos (busca no disco do runtime do GDS, reparo "
      "preemptivo do PATH com aviso, GNEEDACCESS_DIR ignorada, mensagem que "
      "manda instalar o Suite e a do aparelho ausente no build_source)")
sys.exit(0)
