"""Teste do CONTRATO das luvas: o firmware (ESP32) x o lado do PC (imu.py).

Contexto (28/09/2026): o firmware das luvas entrou no repositorio
(`firmware/esp32_luvas`), depois de ser recuperado da maquina antiga. Agora ha'
DOIS lugares que declaram a mesma coisa -- a porta de cada mao -- e uma troca de
lado em um deles poe os blocos `IMU_L_*`/`IMU_R_*` do CSV TROCADOS, sem nenhum
erro aparecer. Este teste le' o firmware e trava o contrato:

  1. `platformio.ini` (envs `luva_direita`/`luva_esquerda` + `GLOVE_SIDE`) x a
     formula da porta em `src/main.cpp` x `imu.IMU_PORTAS_PADRAO`
     (1 = direita = 4210, 2 = esquerda = 4211);
  2. o pacote tem de continuar com **10 campos** e SEM o lado dentro (o lado vem
     da PORTA) -- e' o que o `ImuReceiver`/`ImuBank` assumem; o receptor de
     bancada do proprio firmware (`receber_luvas.py`) confere os 10 campos;
  3. `credenciais.h.example`: unicast ligado e `UDP_DESTINO` igual ao IP que o
     `tools/diagnostico_hotspot.py` documenta (ScopeAddress do ICS);
  4. o `credenciais.h` de verdade (se existir nesta maquina, ele NAO e'
     versionado) tem de apontar para o hotspot ATUAL -- SSID, senha e IP da
     fonte unica do `diagnostico_hotspot` -- e continuar fora do git;
  5. a nota de bancada `unicast` e o README do firmware seguem a mesma rede.

    .venv\\Scripts\\python.exe test_firmware_luvas.py
"""
import os
import re
import subprocess
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "tools"))

import diagnostico_hotspot as dh  # noqa: E402
from imu import IMU_PORTAS_PADRAO  # noqa: E402

FIRMWARE = os.path.join(RAIZ, "firmware", "esp32_luvas")


def texto(*partes):
    with open(os.path.join(FIRMWARE, *partes), encoding="utf-8") as handle:
        return handle.read()


def campo(padrao, conteudo, o_que):
    achado = re.search(padrao, conteudo)
    assert achado, f"nao achei {o_que} (o firmware mudou de forma?)"
    return achado.group(1)


ini = texto("platformio.ini")
cpp = texto("src", "main.cpp")
exemplo = texto("src", "credenciais.h.example")
receber = texto("receber_luvas.py")

# =============================================================================
# 1) PORTA POR LADO: env do PlatformIO x formula do main.cpp x IMU_PORTAS_PADRAO
# =============================================================================
envs = dict(re.findall(r"\[env:(\w+)\]\s*build_flags\s*=\s*-D\s*GLOVE_SIDE=(\d+)",
                       ini))
assert set(envs) >= {"luva_direita", "luva_esquerda"}, envs
lado_direita = int(campo(r"const uint16_t UDP_PORT = \(GLOVE_SIDE == (\d+)\) \?",
                         cpp, "a formula da porta"))
porta_direita = int(campo(r"const uint16_t UDP_PORT = \(GLOVE_SIDE == \d+\) \? "
                          r"(\d+) :", cpp, "a porta da direita"))
porta_esquerda = int(campo(r"const uint16_t UDP_PORT = \(GLOVE_SIDE == \d+\) \? "
                           r"\d+ : (\d+);", cpp, "a porta da esquerda"))
assert int(envs["luva_direita"]) == lado_direita, (
    f"o env luva_direita usa GLOVE_SIDE={envs['luva_direita']} mas a formula da "
    f"porta testa == {lado_direita}: a luva DIREITA cairia na porta da esquerda")
assert int(envs["luva_esquerda"]) != lado_direita, (
    "os dois envs usam o MESMO GLOVE_SIDE: as duas luvas mandariam na mesma "
    "porta e um lado sumiria do CSV")
assert {1: porta_direita, 2: porta_esquerda} == IMU_PORTAS_PADRAO, (
    f"firmware diz direita={porta_direita}/esquerda={porta_esquerda}, projeto "
    f"diz {IMU_PORTAS_PADRAO}: os blocos IMU_L_*/IMU_R_* entrariam trocados")

# =============================================================================
# 2) PACOTE: 10 campos, SEM o lado dentro (o lado vem da PORTA)
# =============================================================================
formato = campo(r'snprintf\(packet, sizeof\(packet\), "([^"]+)"', cpp,
                "o formato do pacote UDP")
campos = formato.replace("\\n", "").split(",")
assert len(campos) == 10, (campos, "o ImuReceiver legado espera 10 campos")
assert campos[0] == "%lu", campos
assert not any("lado" in nome.lower() or "side" in nome.lower()
               for nome in campos), (
    "o lado NAO pode ir no pacote: com 11 campos o ImuReceiver mudaria de "
    "formato e o lado passaria a vir do dado, nao da porta")
assert "%d" not in formato, campos

cabecalho = campo(r'Serial\.println\("(esp_timestamp_us[^"]+)"\)', cpp,
                  "o cabecalho da serial")
nomes = cabecalho.split(",")
assert len(nomes) == 10 and nomes[1] == "roll_deg" and nomes[3] == "yaw_deg", nomes
assert nomes[4:7] == ["ax_g", "ay_g", "az_g"], nomes
assert nomes[7:] == ["gx_dps", "gy_dps", "gz_dps"], nomes

assert "len(partes) != 10" in receber, (
    "o receber_luvas.py do firmware tem de exigir os MESMOS 10 campos")
assert int(campo(r"PORTA_DIREITA = (\d+)", receber, "a porta da direita")) == \
    porta_direita
assert int(campo(r"PORTA_ESQUERDA = (\d+)", receber, "a porta da esquerda")) == \
    porta_esquerda

# =============================================================================
# 3) credenciais.h.example: unicast e o MESMO IP do ICS do projeto
# =============================================================================
assert re.search(r"^#define UDP_USAR_UNICAST 1\s*$", exemplo, re.M), (
    "sem UDP_USAR_UNICAST 1 o firmware cai em broadcast 255.255.255.255, que "
    "nao atravessa o hotspot")
ip_exemplo = campo(r'UDP_DESTINO "([^"]+)"', exemplo, "o IP de destino")
assert ip_exemplo == dh.REDE_ESP32["ip_host"], (
    f"o firmware manda para {ip_exemplo} e o ICS desta bancada e' "
    f"{dh.REDE_ESP32['ip_host']} (fonte unica: tools/diagnostico_hotspot.py)")

# =============================================================================
# 4) credenciais.h REAL (nao versionado): a rede ATUAL e fora do git
# =============================================================================
caminho_real = os.path.join(FIRMWARE, "src", "credenciais.h")
assert "credenciais.h" in texto(".gitignore") or \
    "credenciais.h" in open(os.path.join(RAIZ, ".gitignore"),
                            encoding="utf-8").read(), (
    "src/credenciais.h tem SENHA: precisa estar ignorado por algum .gitignore")
if os.path.exists(caminho_real):
    with open(caminho_real, encoding="utf-8") as handle:
        real = handle.read()
    ssid = campo(r'#define WIFI_SSID_PSK "([^"]*)"', real, "o SSID do hotspot")
    senha = campo(r'#define WIFI_SENHA_PSK "([^"]*)"', real, "a senha do hotspot")
    ip_real = campo(r'UDP_DESTINO "([^"]+)"', real, "o IP de destino")
    assert ssid == dh.REDE_ESP32["ssid"], (
        f"o credenciais.h desta maquina aponta para '{ssid}': nao e' o hotspot "
        f"'{dh.REDE_ESP32['ssid']}' desta bancada -- a luva nao conecta")
    assert senha == dh.REDE_ESP32["senha"], "a senha do hotspot mudou"
    assert ip_real == dh.REDE_ESP32["ip_host"], ip_real
    assert re.search(r"^#define UDP_USAR_UNICAST 1\s*$", real, re.M), real[:200]
    try:
        rastreio = subprocess.run(
            ["git", "ls-files", "--error-unmatch",
             os.path.join("firmware", "esp32_luvas", "src", "credenciais.h")],
            cwd=RAIZ, capture_output=True, text=True, timeout=30)
        assert rastreio.returncode != 0, (
            "src/credenciais.h esta' RASTREADO pelo git: a senha do hotspot "
            "vazaria no repositorio publico")
    except (OSError, subprocess.SubprocessError):
        pass                    # sem git: o teste do .gitignore ja' cobriu

# =============================================================================
# 5) NOTA e README do firmware seguem a rede atual (o SSID deriva do hostname)
# =============================================================================
nota = texto("unicast")
assert dh.REDE_ESP32["ssid"] in nota, (
    f"a nota 'unicast' cita outro hotspot: {nota.strip()!r}")
assert dh.REDE_ESP32["ip_host"] in nota, nota
readme = texto("README.md")
for esperado in ("4210", "4211", "GLOVE_SIDE", "UDP_DESTINO", "unicast"):
    assert esperado in readme, (esperado, "o README do firmware ficou para tras")

# 5b) os DOIS .bat de gravacao: mesmo hotspot e o `pio` do .venv como fallback
#     (a extensao do VS Code pode estar instalada SEM o core provisionado: foi o
#     que aconteceu nesta maquina em 28/09/2026)
for arquivo, porta, env in (("gravar_direita.bat", 4210, "luva_direita"),
                            ("gravar_esquerda.bat", 4211, "luva_esquerda")):
    conteudo = texto(arquivo)
    assert dh.REDE_ESP32["ssid"] in conteudo, (arquivo, "SSID antigo no .bat")
    assert dh.REDE_ESP32["ip_host"] in conteudo, arquivo
    assert str(porta) in conteudo and env in conteudo, (arquivo, porta, env)
    assert "%~dp0..\\..\\.venv\\Scripts\\pio.exe" in conteudo, (
        arquivo, "sem o fallback do pio do .venv (.venv\\Scripts\\pio.exe)")

print("FIRMWARE_LUVAS_OK: 5/5 blocos (porta por lado: env x main.cpp x "
      f"IMU_PORTAS_PADRAO -> direita={porta_direita}, esquerda={porta_esquerda}; "
      "pacote de 10 campos sem o lado; unicast para o IP do ICS; credenciais.h "
      "real na rede atual e fora do git; nota/README em sincronia)")
sys.exit(0)

