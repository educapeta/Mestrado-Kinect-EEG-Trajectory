"""Probe ISOLADO da impedancia do g.Nautilus (sem o paradigma, sem gpype).

Mede o erro REAL que o `get_impedance` levanta neste aparelho, nos DOIS
estados que importam:

  A) APARELHO ABERTO E PARADO (logo apos SetConfiguration, sem stream):
     e' o estado do `_GNautilusCore` ANTES do pipeline.start();
  B) APARELHO TRANSMITINDO (a thread de dados rodando): e' o estado em
     que o `ImpedanceMonitor` do paradigma realmente mede (a tela de
     preparacao so' aparece DEPOIS do pipeline.start()).

O paradigma engolia a excecao e mostrava "sem aparelho direto nem API do
g.Pype" (item 59); este probe mostra o tipo e a mensagem EXATOS, para
distinguir CRC transitrio do link sem fio de erro de estado.

FEQUE O PARADIGMA ANTES (o GDS abre o aparelho em modo EXCLUSIVO):
    python tools/probe_impedancia.py

v2 (19/10/2026): o 1o uso falhou com "No g.Nautilus devices found."
MESMO com o amplificador ligado, sendo que a propria lista
ConnectedDevices() -- uma linha antes! -- o enxergava. Chamadas
identicas seguidas deram resultados diferentes, entao o probe agora
enumera 5x (a flutuacao aparece), chama GDS._check_key() e enumera de
novo (ordem identica ao __init__), re-tenta abrir 4x re-enumerando apos
CADA falha, e -- se nada abrir -- lista os processos python/gds/
recorder do sistema (o paradigma aberto segura o aparelho e pode estar
disputando o discovery).
"""
import sys
import threading
import time
import traceback

import numpy as np


def _titulo(texto):
    print(f"\n=== {texto} " + "=" * max(0, 60 - len(texto)), flush=True)


def _enumera(vezes=5):
    """Lista ConnectedDevices() varias vezes -- o discovery flutua.

    No 1o uso do probe (19/10/2026) a 1a chamada (nao filtrada) achou
    'NR-2018.10.53 g.Nautilus em_uso=False' e a chamada SEGUINTE, dentro
    do proprio GNautilus.__init__ (apos GDS._check_key()), nao achou
    nenhum -- 'No g.Nautilus devices found.' com o aparelho ligado.
    Repetir aqui deixa a flutuacao visivel: se o aparelho piscar na
    lista, e' discovery/link; se sumir EXATO apos o _check_key, e' o
    _check_key; se ficar firme e a abertura falhar, o problema e' outro.
    """
    from gtec_gds.lib.gtec_gds_wrapper import ConnectedDevices
    from gtec_gds.lib.constants import Constants as C
    print(f"  DEVICE_TYPE_GNAUTILUS = {C.DEVICE_TYPE_GNAUTILUS!r}")
    for i in range(vezes):
        try:
            lista = list(ConnectedDevices())
            naut = sum(1 for _, t, _ in lista
                       if t == C.DEVICE_TYPE_GNAUTILUS)
            print(f"  [{i}] {lista!r}   g.Nautilus={naut}", flush=True)
        except Exception as exc:
            print(f"  [{i}] enumeracao FALHOU: {type(exc).__name__}: {exc}",
                  flush=True)
        # 1a repeticao quase imediata (reproduz o par de chamadas do
        # __init__); as demais com 500 ms para pegar piscar de discovery.
        time.sleep(0.05 if i == 0 else 0.5)


def _processos_suspeitos():
    """Lista processos que podem estar segurando/disputando o GDS."""
    import subprocess
    cmd = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
           "Get-CimInstance Win32_Process | "
           "Where-Object { $_.Name -match 'python|gds|recorder|gtec|gnautilus' } | "
           "ForEach-Object { \"$($_.ProcessId) | $($_.Name) | $($_.CommandLine)\" }")
    try:
        saida = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=20)
        linhas = [l.strip() for l in (saida.stdout or "").splitlines()
                  if l.strip()]
        for linha in linhas:
            print(f"  {linha}")
        if not linhas:
            print("  (nenhum -- paradigma/g.Recorder fechados)")
    except Exception as exc:
        print(f"  inspecao de processos falhou: {type(exc).__name__}: {exc}")


def main():
    import gtec_gds as gds
    from gtec_gds.lib.gtec_gds_wrapper import ConnectedDevices, GDS
    from gtec_gds.lib.constants import Constants as C

    # CHAVE DE USO LOCAL: o `gds.GNautilus(...)` chama `GDS._check_key()` no
    # __init__ e levanta `KeyError: 'Invalid usage.'` se a chave de uso nao
    # estiver registrada -- ANTES de qualquer get_impedance. O gpype registra
    # antes de abrir (`gpype/backend/sources/g_nautilus.py`: driver_usage.
    # register()); abrir direto, como este probe faz, pulava esse passo e o
    # probe morria no _check_key em vez de medir a impedancia. Registra aqui
    # pelo MESMO caminho do gpype; se o pacote nao expor o registro, segue e
    # deixa o erro real aparecer (o probe e' diagnostico).
    try:
        from gpype.common._private import driver_usage
        driver_usage.register()
        print("[chave] driver_usage.register() chamado (chave de uso local "
              "registrada antes de abrir)", flush=True)
    except Exception as exc:
        print(f"[chave] NAO foi possivel registrar a chave de uso local "
              f"({type(exc).__name__}: {exc}); seguindo mesmo assim -- se a "
              f"abertura falhar com 'Invalid usage.', e' esta a causa.",
              flush=True)

    _titulo("Aparelhos conectados (ConnectedDevices)")
    conectados = [(n, t, u) for n, t, u in ConnectedDevices()]
    for nome, tipo, uso in conectados:
        marca = "g.Nautilus" if tipo == C.DEVICE_TYPE_GNAUTILUS else f"tipo={tipo}"
        print(f"  {nome}  {marca}  em_uso={uso}")
    if not any(t == C.DEVICE_TYPE_GNAUTILUS for _, t, _ in conectados):
        # NAO return mais: no 1o uso (19/10) esta leitura viu o aparelho e a
        # PROXIMA (dentro do __init__) nao viu -- o valor aqui esta' em
        # comparar com as repeticoes abaixo, nao em decidir sozinha.
        print("  (aviso: nenhum g.Nautilus NESTA leitura; ver E1 -- o "
              "dispostivo pode piscar no discovery)")

    _titulo("E1) Enumeracao repetida (5x) -- caça flutuacao")
    _enumera(5)

    _titulo("E2) GDS._check_key() + enumeracao imediata (ordem do __init__)")
    try:
        GDS._check_key()
        print("  _check_key: OK", flush=True)
    except Exception as exc:
        print(f"  _check_key FALHOU: {type(exc).__name__}: {exc}", flush=True)
    try:
        print(f"  {list(ConnectedDevices())!r}", flush=True)
    except Exception as exc:
        print(f"  enumeracao FALHOU: {type(exc).__name__}: {exc}", flush=True)

    _titulo("A) Abrir o aparelho (4 tentativas, re-enumera apos falha)")
    aparelho = None
    ultimo_erro = None
    for tent in range(1, 5):
        try:
            aparelho = gds.GNautilus(sampling_rate=500, channel_count=32,
                                     frame_size=1)
            print(f"aberto na tentativa {tent}: "
                  f"serial={aparelho.serial_number} "
                  f"canais={aparelho.channel_count} "
                  f"fs={aparelho._sampling_rate}", flush=True)
            break
        except Exception as exc:
            ultimo_erro = exc
            print(f"tentativa {tent} FALHOU: {type(exc).__name__}: {exc}",
                  flush=True)
            try:
                print(f"  enum logo apos a falha: {list(ConnectedDevices())!r}",
                      flush=True)
            except Exception as exc2:
                print(f"  enum apos falha FALHOU: "
                      f"{type(exc2).__name__}: {exc2}", flush=True)
            if tent < 4:
                time.sleep(1.5)
    if aparelho is None:
        print(f"ABERTURA IMPOSSIVEL apos 4 tentativas: "
              f"{type(ultimo_erro).__name__}: {ultimo_erro}")
        if ultimo_erro is not None:
            traceback.print_exception(ultimo_erro)
        _titulo("Processos que podem estar segurando o GDS")
        _processos_suspeitos()
        return 1

    _titulo("A1) get_impedance(first=True)  [parado]")
    try:
        z = aparelho.get_impedance(True)
        z = np.asarray(z, dtype=float)
        print(f"OK: {len(z)} canais | min={np.nanmin(z):.1f} "
              f"max={np.nanmax(z):.1f} | primeiros: {z[:6]}")
    except Exception as exc:
        print(f"FALHOU: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    _titulo("A2) get_impedance(first=False)  [parado, 2a leitura]")
    try:
        z = aparelho.get_impedance(False)
        z = np.asarray(z, dtype=float)
        print(f"OK: min={np.nanmin(z):.1f} max={np.nanmax(z):.1f}")
    except Exception as exc:
        print(f"FALHOU: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    _titulo("B) Ligar o stream (como o paradigma faz) e medir de novo")
    frames = []
    erro_stream = []

    def _cb(dado):
        frames.append(np.asarray(dado).copy())

    try:
        aparelho.set_data_callback(_cb)
        aparelho.start()
        time.sleep(1.0)
        print(f"stream: {len(frames)} frames em 1 s")
        for primeira in (True, False):
            try:
                z = aparelho.get_impedance(primeira)
                z = np.asarray(z, dtype=float)
                print(f"get_impedance(first={primeira}) TRANSMITINDO: OK "
                      f"min={np.nanmin(z):.1f} max={np.nanmax(z):.1f}")
            except Exception as exc:
                print(f"get_impedance(first={primeira}) TRANSMITINDO: "
                      f"FALHOU: {type(exc).__name__}: {exc}")
                traceback.print_exc()
    except Exception as exc:
        print(f"stream FALHOU: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        try:
            aparelho.stop()
        except Exception:
            pass
        if getattr(aparelho, "stream_error", None) is not None:
            print(f"stream_error: {aparelho.stream_error}")

    _titulo("Fechar")
    try:
        aparelho.close()
        print("fechado.")
    except Exception as exc:
        print(f"fecho falhou: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
