"""Teste do diagnostico do HOTSPOT/ESP32 (sem hardware).

Contexto (bancada, 28/09/2026): a rede mudou para o hotspot "EDUARDO-PC 9049" e
o firmware dos ESP32 tem de mandar UDP para 192.168.137.1 nas portas 4210/4211.
Este teste TRAVA os pontos que decidem se o datagrama chega:

  1. o parser das regras do firewall (registro do Windows: campos 'chave=valor'
     separados por '|'), inclusive portas em intervalo e o campo VAZIO
     ('Qualquer'/'Any' localizado) = qualquer porta;
  2. o filtro de PROGRAMA -- o que dava falso "tem regra": uma regra com `App`
     vale so' para aquele executavel, entao a regra UDP do python da Loja NAO
     cobre o python do .venv (caso real desta maquina);
  3. o veredito (`avalia`): hotspot desligado, politica BlockInbound sem regra,
     porta ocupada, ICS diferente do documentado e SSID que nao casa com o
     hostname; mais o snippet do firmware e a escuta ao vivo (emissor falso);
  4. que o DOCUMENTO da rede (docs/IMU_DUPLO_ESP32_DESENHO.md) tem o mesmo
     SSID/senha/IP da ferramenta -- se alguem trocar o hotspot e esquecer um
     dos dois lados, a suite falha.

    .venv\\Scripts\\python.exe test_diagnostico_hotspot.py
"""
import os
import socket
import sys
import threading
import time

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, "tools"))

import diagnostico_hotspot as dh  # noqa: E402

PORTAS = {1: 4210, 2: 4211}
EXECUTAVEL_VENV = r"C:\mestradopy\.venv\Scripts\python.exe"

#: Regras como elas estao no registro do firewall desta maquina (medido em
#: 28/09/2026): a "Python" e' do python DA LOJA -- e' a armadilha.
REGRA_PYTHON_LOJA = (
    "v2.10|Action=Allow|Active=TRUE|Dir=In|Protocol=17|Profile=Public|"
    "App=C:\\program files\\windowsapps\\pythonsoftwarefoundation.python."
    "3.11_3.11.2544.0_x64__qz5n2kfra8p0\\python3.11.exe|Name=Python|"
    "Desc=@FirewallAPI.dll,-32752")
REGRA_TCP_PYTHON = (
    "v2.10|Action=Allow|Active=TRUE|Dir=In|Protocol=6|Profile=Public|"
    "App=C:\\program files\\windowsapps\\pythonsoftwarefoundation.python."
    "3.11_3.11.2544.0_x64__qz5n2kfra8p0\\python3.11.exe|Name=Python")
REGRA_NOSSA_GLOBAL = (
    "v2.30|Action=Allow|Active=TRUE|Dir=In|Protocol=17|LPort=4210,4211|"
    "Name=" + dh.NOME_REGRA_FIREWALL)
REGRA_NOSSA_APP = (
    "v2.30|Action=Allow|Active=TRUE|Dir=In|Protocol=17|"
    "App=" + EXECUTAVEL_VENV + "|Name=Python (venv)")
REGRA_INATIVA = (
    "v2.30|Action=Allow|Active=FALSE|Dir=In|Protocol=17|LPort=4210,4211|"
    "Name=desligada")
REGRA_SAIDA = (
    "v2.30|Action=Allow|Active=TRUE|Dir=Out|Protocol=17|LPort=4210,4211|"
    "Name=saida")


def estado_fake(regras=(), ics="192.168.137.1", ips=None, hostname="Eduardo-PC",
                politica={"block": 3, "allow": 0}, portas_livres=None):
    """Estado medido de mentira, com a MESMA forma do `estado_da_rede()`."""
    if ips is None:
        ips = ("192.168.137.1", "200.135.92.58")
    return {
        "hostname": hostname,
        "ics": ics,
        "ips": [(ip, f"iface{indice}") for indice, ip in enumerate(ips)],
        "ps_ok": True,
        "regras": list(regras),
        "politica": politica,
        "portas_livres": (portas_livres if portas_livres is not None
                          else {4210: True, 4211: True}),
    }


# =============================================================================
# 1) PARSER: campos do registro, portas (lista, intervalo, vazio/localizado)
# =============================================================================
regra = dh.parseia_regra(REGRA_PYTHON_LOJA)
assert regra is not None, "a regra do registro nao foi interpretada"
assert regra["nome"] == "Python", regra
assert regra["dir"] == "In" and regra["acao"] == "Allow", regra
assert regra["ativo"] is True and regra["protocolo"] == "17", regra
assert regra["lport"] == "", "a regra 'Python' nao declara LPort (qualquer)"
assert "windowsapps" in regra["app"].lower(), regra
assert dh.parseia_regra("lixo sem campos") is None

assert dh.portas_do_campo("") == (True, []), "vazio = qualquer porta"
assert dh.portas_do_campo("Qualquer") == (True, []), "localizado = qualquer"
assert dh.portas_do_campo("4210,4211") == (False, [(4210, 4210), (4211, 4211)])
assert dh.portas_do_campo("4200-4300") == (False, [(4200, 4300)])
assert dh.porta_coberta("", 4210) and dh.porta_coberta("Qualquer", 4211)
assert dh.porta_coberta("4200-4300", 4210), "intervalo cobre 4210"
assert dh.porta_coberta("4210,4211", 4211)
assert not dh.porta_coberta("4210", 4211), "4211 nao esta' na regra 4210"
assert not dh.porta_coberta("5000-5100", 4210)

# =============================================================================
# 2) FILTRO DE PROGRAMA: a regra do python da Loja NAO cobre o .venv
# =============================================================================
assert not dh.regra_serve(regra, 4210, EXECUTAVEL_VENV), (
    "a regra UDP do python da Loja NAO pode valer para o python do .venv")
assert dh.regra_serve(regra, 4210, regra["app"]), "para o MESMO app tem de valer"
assert dh.regra_serve(dh.parseia_regra(REGRA_NOSSA_GLOBAL), 4210,
                      EXECUTAVEL_VENV), "regra global (sem App) vale para todos"
assert dh.regra_serve(dh.parseia_regra(REGRA_NOSSA_APP), 4211, EXECUTAVEL_VENV), (
    "regra de app do proprio .venv tem de valer")
assert not dh.regra_serve(dh.parseia_regra(REGRA_NOSSA_APP), 4211,
                          r"C:\outro\python.exe"), "app diferente nao vale"
assert not dh.regra_serve(dh.parseia_regra(REGRA_TCP_PYTHON), 4210,
                          EXECUTAVEL_VENV), "regra TCP nao libera UDP"
assert not dh.regra_serve(dh.parseia_regra(REGRA_NOSSA_GLOBAL), 4212,
                          EXECUTAVEL_VENV), "porta fora da regra nao vale"
assert dh.regra_serve(dh.parseia_regra(REGRA_INATIVA), 4210, EXECUTAVEL_VENV), (
    "a regra em si serve; quem filtra inativa/saida e' a lista de entrada")

# =============================================================================
# 3) COBERTURA: regras de entrada ativas e o caso "quase" (engana)
# =============================================================================
so_loja = estado_fake([REGRA_PYTHON_LOJA, REGRA_TCP_PYTHON, REGRA_INATIVA,
                       REGRA_SAIDA, "lixo"])
assert dh.regras_que_cobrem(so_loja, PORTAS, EXECUTAVEL_VENV) == [], (
    "sem regra para o .venv nao ha' cobertura -- o falso 'tem regra' era o bug")
quase = dh.regras_quase(so_loja, PORTAS, EXECUTAVEL_VENV)
assert quase and quase[0][0] == "Python", quase
assert "windowsapps" in quase[0][1].lower(), quase

com_nossa = estado_fake([REGRA_PYTHON_LOJA, REGRA_NOSSA_GLOBAL, REGRA_INATIVA])
assert dh.regras_que_cobrem(com_nossa, PORTAS, EXECUTAVEL_VENV) == [
    dh.NOME_REGRA_FIREWALL], dh.regras_que_cobrem(com_nossa, PORTAS,
                                                  EXECUTAVEL_VENV)
assert dh.regras_que_cobrem(estado_fake([REGRA_NOSSA_APP]), PORTAS,
                            EXECUTAVEL_VENV) == ["Python (venv)"]
assert dh.regras_de_entrada_ativas(estado_fake([REGRA_INATIVA, REGRA_SAIDA,
                                                "lixo"])) == [], (
    "regra desativada, de saida ou sem campos nao entra na conta")
assert dh.regras_quase(com_nossa, PORTAS, EXECUTAVEL_VENV), (
    "a regra do python da Loja continua sendo 'quase' (outro programa)")
assert dh.regras_parciais(com_nossa, PORTAS) == [], (
    "regra que libera as DUAS portas nao e' parcial")

# 3b) COBERTURA PARCIAL -- o caso REAL da bancada (28/09/2026): a regra criada
#     a mao, chamada "ESP32", liberou UDP de entrada SO' na 4211; a luva
#     DIREITA (4210) continuava bloqueada e o veredito nao dizia o que faltava.
REGRA_PARCIAL_4211 = ("v2.30|Action=Allow|Active=TRUE|Dir=In|Protocol=17|"
                      "LPort=4211|Name=ESP32")
parcial = estado_fake([REGRA_PARCIAL_4211])
assert dh.regras_que_cobrem(parcial, PORTAS, EXECUTAVEL_VENV) == [], (
    "uma regra que libera so' a 4211 NAO pode contar como cobertura")
assert dh.regras_parciais(parcial, PORTAS) == [("ESP32", [4211], [4210], "")], \
    dh.regras_parciais(parcial, PORTAS)
ok, problemas, _ = dh.avalia(parcial, dh.REDE_ESP32, PORTAS)
assert not ok and len(problemas) == 1, problemas
texto = " ".join(problemas)
assert "ESP32" in texto and "FALTA 4210" in texto, texto
assert "4211" in texto, texto

# 3c) o MESMO caso do outro lado (regra so' na 4210): a esquerda ficaria muda
parcial_4210 = estado_fake([REGRA_PARCIAL_4211.replace("4211", "4210")])
assert dh.regras_parciais(parcial_4210, PORTAS) == [("ESP32", [4210], [4211], "")], (
    dh.regras_parciais(parcial_4210, PORTAS))

# =============================================================================
# 4) VEREDITO: hotspot, ICS, SSID x hostname, firewall e porta ocupada
# =============================================================================
# 4a) a situacao desta maquina: hotspot desligado e nenhuma regra para o .venv
desligado = estado_fake([REGRA_PYTHON_LOJA], ips=("200.135.92.58",))
ok, problemas, _ = dh.avalia(desligado, dh.REDE_ESP32, PORTAS)
assert not ok and len(problemas) == 2, problemas
texto = " ".join(problemas)
assert "DESLIGADO" in texto and "192.168.137.1" in texto, texto
assert "2,4 GHz" in texto, "a armadilha do radio 2,4 GHz tem de aparecer"
assert "BlockInbound" in texto, texto
assert "windowsapps" in texto, (
    "o veredito tem de dizer que a regra existente e' de OUTRO python")

# 4b) tudo pronto: hotspot com o IP do ICS, regra global e portas livres
ok, problemas, avisos = dh.avalia(estado_fake([REGRA_NOSSA_GLOBAL]),
                                  dh.REDE_ESP32, PORTAS)
assert ok and problemas == [], problemas
assert any("firewall OK" in aviso for aviso in avisos), avisos

# 4c) porta ocupada por outro processo
ocupada = estado_fake([REGRA_NOSSA_GLOBAL],
                      portas_livres={4210: False, 4211: True})
ok, problemas, _ = dh.avalia(ocupada, dh.REDE_ESP32, PORTAS)
assert not ok and any("4210" in p and "ocupada" in p for p in problemas), problemas

# 4d) ICS diferente do documentado: aviso com o valor REAL desta maquina
outro = estado_fake([REGRA_NOSSA_GLOBAL], ics="192.168.50.1",
                    ips=("192.168.50.1",))
ok, problemas, avisos = dh.avalia(outro, dh.REDE_ESP32, PORTAS)
assert ok, problemas
assert any("192.168.50.1" in aviso for aviso in avisos), avisos

# 4e) SSID que nao casa com o hostname (o Windows gera <PC> <4 digitos>)
ok, _, avisos = dh.avalia(estado_fake([REGRA_NOSSA_GLOBAL],
                                      hostname="DUDU-LAPTOP"),
                          dh.REDE_ESP32, PORTAS)
assert ok and any("SSID" in aviso for aviso in avisos), avisos

# =============================================================================
# 5) SNIPPET do firmware: SSID, senha, IP e as duas portas
# =============================================================================
snippet = dh.snippet_firmware(dh.REDE_ESP32, PORTAS)
for esperado in (dh.REDE_ESP32["ssid"], dh.REDE_ESP32["senha"],
                 "IP_DO_PC(192, 168, 137, 1)", "4210 = mao direita",
                 "4211 = mao esquerda", "WiFi.begin(WIFI_SSID, WIFI_SENHA)",
                 "WiFi.setSleep(false)",
                 "t_us,roll,pitch,yaw,ax,ay,az,gx,gy,gz"):
    assert esperado in snippet, (esperado, snippet)
assert "IP_DO_PC(10, 0, 0, 7)" in dh.snippet_firmware(
    dh.REDE_ESP32, PORTAS, endereco_host="10.0.0.7"), (
    "com outro ICS o snippet tem de sair com o endereco REAL")

# =============================================================================
# 6) ESCUTA: emissor falso numa porta de teste (sem hardware nenhum)
# =============================================================================
PORTA_TESTE, PORTA_VAZIA = 4299, 4298


def emite(porta, quantos, intervalo=0.01):
    """Emissor falso: `quantos` datagramas por loopback (papel do ESP32)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _ in range(quantos):
            sock.sendto(b"1,2,3,4", ("127.0.0.1", porta))
            time.sleep(intervalo)
    finally:
        sock.close()


emissor = threading.Thread(target=emite, args=(PORTA_TESTE, 60), daemon=True)
emissor.start()
dados = dh.escuta({1: PORTA_TESTE}, segundos=1.0)
assert dados["total"] >= 10, dados["contagem"]
assert dados["contagem"][PORTA_TESTE] == dados["total"], dados
remetentes = list(dados["remetentes"])
assert remetentes and remetentes[0][0] == "127.0.0.1", remetentes
assert remetentes[0][1] == PORTA_TESTE, remetentes
info = dados["remetentes"][remetentes[0]]
assert info["amostras"] >= 10 and info["hz"] > 0.0, info
assert info["idade_s"] >= 0.0, info
emissor.join(timeout=2.0)

vazio = dh.escuta({1: PORTA_VAZIA}, segundos=0.3)
assert vazio["total"] == 0 and vazio["remetentes"] == {}, vazio
assert vazio["contagem"] == {PORTA_VAZIA: 0}, vazio

# =============================================================================
# 7) DOCUMENTO x FERRAMENTA: a rede nao pode ficar escrita so' de um lado
# =============================================================================
from imu import IMU_PORTAS_PADRAO  # noqa: E402

assert dh.portas() == IMU_PORTAS_PADRAO, (dh.portas(), IMU_PORTAS_PADRAO)
assert sorted(IMU_PORTAS_PADRAO.values()) == [4210, 4211], IMU_PORTAS_PADRAO

caminho_doc = os.path.join(RAIZ, "docs", "IMU_DUPLO_ESP32_DESENHO.md")
with open(caminho_doc, encoding="utf-8") as handle:
    documento = handle.read()
for esperado in (dh.REDE_ESP32["ssid"], dh.REDE_ESP32["senha"],
                 dh.REDE_ESP32["ip_host"], "diagnostico_hotspot.py",
                 "4210", "4211", "2,4 GHz"):
    assert esperado in documento, (
        f"{esperado!r} sumiu de docs/IMU_DUPLO_ESP32_DESENHO.md: se o hotspot "
        "mudou, atualize documento E ferramenta")

# =============================================================================
# 8) LEITURA REAL da maquina (sem admin) e parser do PowerShell
# =============================================================================
assert dh.parseia_saida_ps("") == []
assert dh.parseia_saida_ps("###IPS\n10.0.0.5;Wi-Fi\n\n###OUTRA\nx;y\n") == [
    ("10.0.0.5", "Wi-Fi")]
assert dh.parseia_saida_ps("lixo\n###IPS\n192.168.137.1;IF\n") == [
    ("192.168.137.1", "IF")]

endereco_ics = dh.le_scope_address()
assert endereco_ics is None or endereco_ics.count(".") == 3, endereco_ics
regras_cruas = dh.le_regras_registro()
assert regras_cruas is None or (isinstance(regras_cruas, list)
                                and len(regras_cruas) > 0), regras_cruas
if regras_cruas:
    interpretadas = [dh.parseia_regra(texto) for texto in regras_cruas]
    assert any(interpretadas), "nenhuma regra do registro foi interpretada"
    quantidade = len(dh.regras_de_entrada_ativas({"regras": regras_cruas}))
    assert quantidade > 0, "nenhuma regra de entrada ativa lida do registro"
politica = dh.politica_de_entrada()
assert politica is None or set(politica) == {"block", "allow"}, politica

print("HOTSPOT_OK: 8/8 blocos (parser das regras do registro, filtro de "
      "PROGRAMA (python da Loja x .venv), cobertura x caso 'quase' x cobertura "
      "PARCIAL (a regra 'ESP32' que libera so' uma das portas), veredito "
      "(hotspot/ICS/SSID/firewall/porta ocupada), snippet do firmware, escuta "
      "ao vivo em porta de teste, documento em sincronia e leitura real do "
      "registro sem administrador)")
sys.exit(0)



