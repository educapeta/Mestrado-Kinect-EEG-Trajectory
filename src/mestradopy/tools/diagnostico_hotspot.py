"""Diagnostico do HOTSPOT do PC para os ESP32 (luvas IMU) -- sem hardware.

Motivo (28/09/2026): a rede da bancada mudou. O firmware dos ESP32 NAO esta'
neste repositorio (vive no proprio ESP32), mas para o datagrama chegar aqui ele
precisa de TRES coisas que sao deste lado da bancada, e cada uma ja' falhou uma
vez em silencio:

  1. SSID/senha do hotspot do PC -- agora "EDUARDO-PC 9049" / "0011223344";
     o nome padrao do Windows e' "<nome do computador> <4 digitos>", entao
     renomear o PC muda o SSID (este teste confere se o SSID documentado casa
     com o hostname desta maquina);
  2. o IP do PC NA rede do hotspot -- 192.168.137.1 (ScopeAddress do ICS, lido
     do registro). O firmware manda UNICAST para esse IP fixo; com o hotspot
     DESLIGADO nao existe 192.168.137.x nenhum e o teste fica mudo;
  3. o firewall do Windows deixando ENTRAR UDP em 4210/4211 -- medido em
     28/09/2026: a politica de entrada e' BlockInbound e a UNICA regra UDP
     existente (nome "Python", perfil Publico) vale para o python DA LOJA
     (`...\\WindowsApps\\...\\python3.11.exe`), NAO para o
     `.venv\\Scripts\\python.exe` que recebe os datagramas: regra de programa
     vale SO' para aquele executavel. Resultado: o datagrama do ESP32 era
     descartado sem erro em lugar nenhum (nem no ESP32, nem aqui). As regras
     sao lidas do REGISTRO (sem admin); `--criar-regra-firewall` cria a regra
     global (essa precisa de admin -- `Get-NetFirewallRule` tambem precisa,
     por isso o registro).

Armadilha de hardware que nao da' para checar por software: o ESP32 e' RADIO
2,4 GHz. Se o hotspot estiver servindo em 5 GHz ("Faixa de rede: 5 GHz"), o
ESP32 simplesmente NAO VE a rede. Deixe a faixa em 2,4 GHz em Configuracoes >
Rede e Internet > Ponto de acesso movel (e desligue a economia de energia).

Uso:
    .venv\\Scripts\\python.exe tools\\diagnostico_hotspot.py
    .venv\\Scripts\\python.exe tools\\diagnostico_hotspot.py --escutar 20
    .venv\\Scripts\\python.exe tools\\diagnostico_hotspot.py --criar-regra-firewall  (admin)

Saida: o estado de cada item, o snippet para colar no firmware do ESP32 e a
lista do que falta. Codigo de saida 0 = pronto para os ESP32 conectarem.
"""
from __future__ import annotations

import argparse
import os
import select
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu import IMU_PORTAS_PADRAO, parse_portas_imu  # noqa: E402

#: Credenciais do hotspot desta bancada. O firmware do ESP32 tem de ter
#: EXATAMENTE estes valores (SSID com espaco e maiusculas; senha sem aspas).
REDE_ESP32 = {
    "ssid": "EDUARDO-PC 9049",
    "senha": "0011223344",
    # ScopeAddress do ICS nesta maquina (registro, medido em 28/09/2026).
    "ip_host": "192.168.137.1",
}

#: Onde o ICS guarda o endereco do PC na rede compartilhada.
CHAVE_ICS = r"SYSTEM\CurrentControlSet\Services\SharedAccess\Parameters"
VALOR_ICS = "ScopeAddress"

#: Nome da regra de firewall criada por `--criar-regra-firewall`.
NOME_REGRA_FIREWALL = "mestradopy ESP32 IMU UDP 4210/4211"

#: Uma passada so' do PowerShell para os IPs das interfaces.
PS_PROBE = (
    "$ErrorActionPreference='SilentlyContinue';"
    "'###IPS';"
    "Get-NetIPAddress -AddressFamily IPv4 | ForEach-Object "
    "{ \"$($_.IPAddress);$($_.InterfaceAlias)\" }"
)

#: Onde o Windows guarda as regras do firewall (formato 'campo=valor' por '|').
#: Ler daqui funciona SEM administrador -- `Get-NetFirewallRule` (e o
#: `Get-NetFirewallPortFilter`) respondem "Acesso negado" sem elevacao, o que
#: dava um veredito errado sobre o firewall (medido em 28/09/2026).
CHAVE_FIREWALL = (r"SYSTEM\CurrentControlSet\Services\SharedAccess\Parameters"
                  r"\FirewallPolicy\FirewallRules")

#: Protocolos do registro que interessam: 17 = UDP, 256 = qualquer.
PROTOCOLOS_UDP = ("17", "256")

#: `netsh` (tambem sem administrador) para a politica de entrada dos perfis.
#: O valor 'BlockInbound' aparece em ingles mesmo em Windows localizado.
NETSH_POLITICA = ["netsh", "advfirewall", "show", "allprofiles"]

PS_CRIA_REGRA = (
    "New-NetFirewallRule -DisplayName '{nome}' -Direction Inbound "
    "-Action Allow -Protocol UDP -LocalPort {portas} -Profile Any "
    "-Description 'Datagramas das luvas IMU (ESP32): uma porta por lado.'"
)

PS_REMOVE_REGRA = "Remove-NetFirewallRule -DisplayName '{nome}'"


def portas():
    """{lado: porta} do projeto (fonte unica: `imu.IMU_PORTAS_PADRAO`)."""
    return dict(IMU_PORTAS_PADRAO)


def le_scope_address():
    """ScopeAddress do ICS (ex.: '192.168.137.1') ou None se nao der para ler."""
    try:
        import winreg
    except ImportError:                                   # pragma: no cover
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, CHAVE_ICS) as chave:
            valor, _ = winreg.QueryValueEx(chave, VALOR_ICS)
    except OSError:
        return None
    return str(valor)


def roda_powershell(comando, timeout=60.0):
    """Roda o comando e devolve (ok, texto) -- sem excecao para fora."""
    try:
        processo = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             comando],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    texto = (processo.stdout or "") + (processo.stderr or "")
    return processo.returncode == 0, texto


def parseia_saida_ps(texto):
    """IPs das interfaces a partir da secao ###IPS (linhas 'ip;alias')."""
    ips, na_secao = [], False
    for linha in texto.splitlines():
        linha = linha.strip()
        if linha.startswith("###"):
            na_secao = linha[3:].upper().startswith("IPS")
            continue
        if na_secao and ";" in linha:
            ip, alias = linha.split(";", 1)
            if ip.strip():
                ips.append((ip.strip(), alias.strip()))
    return ips


def porta_livre(porta):
    """True se ninguem esta' com a porta UDP (bind de teste, sem reuso)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", int(porta)))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def estado_da_rede(portas_=None):
    """Mede o estado real da maquina: ips, ICS, firewall e portas."""
    portas_ = portas_ if portas_ is not None else portas()
    ok, texto = roda_powershell(PS_PROBE)
    return {
        "hostname": socket.gethostname(),
        "ics": le_scope_address(),
        "ips": parseia_saida_ps(texto) if ok else [],
        "ps_ok": ok,
        "regras": le_regras_registro(),
        "politica": politica_de_entrada(),
        "portas_livres": {int(porta): porta_livre(porta)
                          for porta in portas_.values()},
    }


def portas_do_campo(valor):
    """(qualquer, intervalos) de um campo de porta: '', '4210', 'a-b', 'a,b'.

    Campo vazio (ou com palavra localizada tipo 'Qualquer'/'Any') = qualquer
    porta -- assim o parser nao depende do idioma do Windows.
    """
    valor = (valor or "").strip()
    if not any(caractere.isdigit() for caractere in valor):
        return True, []
    intervalos = []
    for parte in valor.replace(",", " ").split():
        inicio, _, fim = parte.partition("-")
        if inicio.strip().isdigit() and fim.strip().isdigit():
            intervalos.append((int(inicio.strip()), int(fim.strip())))
        elif parte.strip().isdigit():
            numero = int(parte.strip())
            intervalos.append((numero, numero))
    return False, intervalos


def porta_coberta(valor, porta):
    """True se o campo de porta (LPort) deixa passar `porta`."""
    qualquer, intervalos = portas_do_campo(valor)
    if qualquer:
        return True
    return any(inicio <= int(porta) <= fim for inicio, fim in intervalos)


def parseia_regra(texto):
    """Valor do registro -> dict com os campos que decidem (None se vazio).

    O valor e' 'campo=valor' separado por '|' (o primeiro pedaco traz a
    versao/GUID). Guardamos so' o que decide -- inclusive `App`, que e' o
    ponto fino: regra de programa vale SO' para aquele executavel.
    """
    campos = {}
    for pedaco in str(texto).split("|"):
        if "=" in pedaco:
            chave, _, valor = pedaco.partition("=")
            campos[chave.strip()] = valor.strip()
    if "Dir" not in campos and "Protocol" not in campos:
        return None
    return {
        "nome": campos.get("Name", ""),
        "dir": campos.get("Dir", ""),
        "acao": campos.get("Action", ""),
        "ativo": campos.get("Active", "").upper() == "TRUE",
        "protocolo": campos.get("Protocol", ""),
        "lport": campos.get("LPort", ""),
        "app": campos.get("App", ""),
        "perfil": campos.get("Profile", ""),
    }


def le_regras_registro():
    """Valores crus das regras do firewall no registro (None = ilegivel)."""
    try:
        import winreg
    except ImportError:                                   # pragma: no cover
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, CHAVE_FIREWALL) as chave:
            quantos = winreg.QueryInfoKey(chave)[1]
            textos = [str(winreg.EnumValue(chave, indice)[1])
                      for indice in range(quantos)]
    except OSError:
        return None
    return textos


def regras_de_entrada_ativas(estado):
    """Regras de ENTRADA ativas (ja' interpretadas) do estado medido."""
    return [regra for regra in
            (parseia_regra(texto) for texto in (estado.get("regras") or []))
            if regra and regra["dir"] == "In" and regra["ativo"]]


def regra_serve(regra, porta, executavel=None):
    """True se a regra LIBERA `porta` para esta aplicacao.

    Tres filtros: protocolo (UDP/qualquer), porta e -- o que quase ninguem
    confere -- o PROGRAMA: uma regra com `App` vale so' para aquele
    executavel, entao a regra UDP do python da Loja NAO cobre o python do
    .venv (era o falso "tem regra" medido em 28/09/2026).
    """
    if regra["acao"] != "Allow" or regra["protocolo"] not in PROTOCOLOS_UDP:
        return False
    if not porta_coberta(regra["lport"], porta):
        return False
    app = (regra["app"] or "").strip()
    if not app:
        return True                     # regra global (qualquer programa)
    executavel = executavel or sys.executable
    return (os.path.normcase(os.path.normpath(app))
            == os.path.normcase(os.path.normpath(executavel)))


def regras_que_cobrem(estado, portas_, executavel=None):
    """Nomes das regras que liberam TODAS as portas para este executavel."""
    return [regra["nome"] or "(sem nome)"
            for regra in regras_de_entrada_ativas(estado)
            if all(regra_serve(regra, porta, executavel)
                   for porta in portas_.values())]


def regras_quase(estado, portas_, executavel=None):
    """Regras UDP que passam nas portas mas NAO no executavel: (nome, app).

    E' o caso mais enganoso da bancada -- existe regra UDP, mas o `App` dela
    e' outro python/outro programa, e o datagrama continua bloqueado.
    """
    quase = []
    for regra in regras_de_entrada_ativas(estado):
        if regra["acao"] != "Allow" or regra["protocolo"] not in PROTOCOLOS_UDP:
            continue
        if not all(porta_coberta(regra["lport"], porta)
                   for porta in portas_.values()):
            continue
        if regra_serve(regra, sorted(portas_.values())[0], executavel):
            continue
        quase.append((regra["nome"] or "(sem nome)", regra["app"]))
    # Programa que PARECE ser o nosso vem primeiro: e' o caso que engana
    # (regra do python da Loja x python do .venv), nao as regras de servico.
    return sorted(quase, key=lambda item: ("python" not in item[1].lower(),
                                           item[0]))


def regras_parciais(estado, portas_):
    """(nome, cobertas, faltando, app) das regras que cobrem SO' PARTE.

    Caso REAL da bancada (28/09/2026): a regra que o usuario criou, chamada
    "ESP32", liberou UDP de entrada **apenas na 4211** -- a luva DIREITA (4210)
    continuava bloqueada, e o veredito antigo so' sabia dizer "nenhuma regra
    libera as DUAS portas" (nem apontava o que faltava). Agora aponta.
    """
    parciais = []
    for regra in regras_de_entrada_ativas(estado):
        if regra["acao"] != "Allow" or regra["protocolo"] not in PROTOCOLOS_UDP:
            continue
        cobertas = [porta for porta in sorted(portas_.values())
                    if porta_coberta(regra["lport"], porta)]
        if 0 < len(cobertas) < len(set(portas_.values())):
            parciais.append((
                regra["nome"] or "(sem nome)", cobertas,
                [porta for porta in sorted(portas_.values())
                 if porta not in cobertas], regra["app"]))
    return parciais


def politica_de_entrada():
    """{'block': n, 'allow': n} de perfis com BlockInbound (None = nao deu).

    Vem do `netsh` porque `Get-NetFirewallProfile` (PowerShell) tambem exige
    administrador nesta maquina; o token 'BlockInbound' aparece em ingles
    mesmo em Windows localizado.
    """
    try:
        processo = subprocess.run(NETSH_POLITICA, capture_output=True,
                                  text=True, timeout=30.0,
                                  encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    texto = processo.stdout or ""
    if "BlockInbound" not in texto and "AllowInbound" not in texto:
        return None
    return {"block": texto.count("BlockInbound"),
            "allow": texto.count("AllowInbound")}


def avalia(estado, rede=None, portas_=None):
    """Veredito do estado medido: (ok, problemas, avisos).

    Separado do `estado_da_rede()` de proposito: assim a regra de decisao
    pode ser testada sem tocar no PowerShell nem no firewall da maquina.
    """
    rede = rede if rede is not None else REDE_ESP32
    portas_ = portas_ if portas_ is not None else portas()
    problemas, avisos = [], []

    # (1) hotspot LIGADO: alguma interface precisa ter o IP do ICS.
    endereco_host = estado.get("ics") or rede["ip_host"]
    ips = {ip for ip, _ in estado.get("ips", [])}
    if endereco_host not in ips:
        problemas.append(
            f"o hotspot esta' DESLIGADO (nenhuma interface com "
            f"{endereco_host}): o ESP32 nao tem para onde mandar o UDP. Ligue "
            "em Configuracoes > Rede e Internet > Ponto de acesso movel, com "
            "Faixa de rede = 2,4 GHz (o ESP32 nao ve 5 GHz)")
    if estado.get("ics") and estado["ics"] != rede["ip_host"]:
        avisos.append(
            f"o ICS desta maquina usa {estado['ics']} (e nao "
            f"{rede['ip_host']}): use {estado['ics']} como destino no "
            "firmware -- o snippet abaixo ja' sai com o valor certo")

    # (2) o SSID e' derivado do hostname: renomear o PC muda o SSID.
    hostname = str(estado.get("hostname") or "")
    if hostname and not rede["ssid"].lower().startswith(hostname.lower()):
        avisos.append(
            f"o SSID de referencia e' '{rede['ssid']}' mas o computador se "
            f"chama '{hostname}': confirme o nome REAL na tela do Ponto de "
            "acesso movel antes de gravar o firmware")

    # (3) firewall: com entrada BlockInbound, sem regra PARA ESTE executavel
    # o datagrama morre calado (regra de programa vale so' para o programa).
    politica = estado.get("politica") or {}
    cobrem = regras_que_cobrem(estado, portas_)
    quase = regras_quase(estado, portas_)
    parciais = regras_parciais(estado, portas_)
    if cobrem:
        avisos.append(
            f"firewall OK: {', '.join(cobrem)} libera(m) UDP "
            f"{'/'.join(str(p) for p in sorted(portas_.values()))} para "
            f"{sys.executable}")
    elif politica.get("block"):
        detalhe = ""
        if parciais:
            nome, cobertas, faltando, _ = parciais[0]
            detalhe += (
                f" -- ATENCAO: a regra '{nome}' libera UDP de entrada para "
                f"{'/'.join(str(p) for p in cobertas)} mas FALTA "
                f"{'/'.join(str(p) for p in faltando)} (e' uma porta por luva: "
                "as duas tem de estar na MESMA regra)")
        if quase:
            detalhe += (
                " -- e existe regra UDP que vale para outro programa ("
                + "; ".join(f"'{nome}' -> {app or 'sem App'}"
                            for nome, app in quase[:2])
                + f"), enquanto o receptor aqui e' {sys.executable}")
        problemas.append(
            f"o firewall esta' com entrada BlockInbound ({politica['block']} "
            "perfil/is) e nenhuma regra libera UDP "
            f"{'/'.join(str(p) for p in sorted(portas_.values()))} para "
            f"{sys.executable}: o datagrama do ESP32 e' descartado em "
            "silencio. Rode `--criar-regra-firewall` num terminal de "
            "ADMINISTRADOR" + detalhe)
    elif estado.get("regras") is None:
        avisos.append("nao deu para ler as regras do firewall no registro: "
                      "confira a mao que UDP 4210/4211 entra")
    else:
        avisos.append("firewall: nenhuma regra especifica para as portas das "
                      "luvas e a politica de entrada nao foi lida (pode ser "
                      "AllowInbound -- confirme com `--escutar` ou responda "
                      "'Permitir' ao aviso do Windows)")

    # (4) porta ocupada por outro processo: o receptor nao consegue abrir.
    for porta, livre in sorted(estado.get("portas_livres", {}).items()):
        if not livre:
            problemas.append(
                f"a porta UDP {porta} esta' ocupada por outro processo "
                "(feche-o, ou o ESP32 fala com quem nao escuta)")
    return (not problemas), problemas, avisos


def snippet_firmware(rede=None, portas_=None, endereco_host=None):
    """Bloco pronto para colar no firmware do ESP32 (SSID, senha, IP, porta).

    `endereco_host` permite usar o ScopeAddress REAL desta maquina quando ele
    nao for o de referencia (ver `avalia()`), em vez de um destino fixo que
    ninguem escuta.
    """
    rede = rede if rede is not None else REDE_ESP32
    portas_ = portas_ if portas_ is not None else portas()
    endereco = endereco_host or rede["ip_host"]
    octetos = endereco.split(".")
    portas_texto = ", ".join(
        f"{porta} = mao {'direita' if lado == 1 else 'esquerda'}"
        for lado, porta in sorted(portas_.items(), key=lambda item: item[1]))
    return "\n".join([
        "// ---- luvas IMU (mestradopy): rede da bancada -- cole no sketch ----",
        f'const char *WIFI_SSID  = "{rede["ssid"]}";',
        f'const char *WIFI_SENHA = "{rede["senha"]}";',
        f"IPAddress IP_DO_PC({', '.join(octetos)});  // Ponto de acesso movel "
        "(ICS)",
        f"const uint16_t PORTA_UDP = {sorted(portas_.values())[0]};"
        f"  // {portas_texto}",
        "",
        "WiFi.mode(WIFI_STA);",
        "WiFi.setSleep(false);   // sem modem sleep: menos jitter no UDP",
        "WiFi.begin(WIFI_SSID, WIFI_SENHA);",
        "while (WiFi.status() != WL_CONNECTED) { delay(250); }",
        "// ... e no envio, uma linha ASCII por datagrama:",
        "// t_us,roll,pitch,yaw,ax,ay,az,gx,gy,gz",
        "udp.beginPacket(IP_DO_PC, PORTA_UDP);",
        "udp.print(linha);",
        "udp.endPacket();",
    ])


def escuta(portas_=None, segundos=15.0):
    """Escuta as portas das luvas e devolve quem mandou (por remetente).

    Devolve {'contagem': {porta: n}, 'remetentes': {(ip, porta): {...}},
    'total': n}. Serve tanto para a bancada (ESP32 de verdade) quanto para o
    teste sem hardware (um emissor falso numa porta de teste).
    """
    portas_ = portas_ if portas_ is not None else portas()
    soquetes = {}
    for porta in sorted(set(int(p) for p in portas_.values())):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", porta))
        sock.setblocking(False)
        soquetes[porta] = sock
    contagem = {porta: 0 for porta in soquetes}
    remetentes = {}
    limite = time.monotonic() + max(float(segundos), 0.0)
    try:
        while time.monotonic() < limite:
            prontos, _, _ = select.select(list(soquetes.values()), [], [], 0.2)
            for sock in prontos:
                try:
                    _, origem = sock.recvfrom(512)
                except OSError:                            # pragma: no cover
                    continue
                porta = sock.getsockname()[1]
                agora = time.monotonic()
                contagem[porta] = contagem.get(porta, 0) + 1
                dados = remetentes.setdefault(
                    (origem[0], porta),
                    {"amostras": 0, "primeiro": agora, "ultimo": agora})
                dados["amostras"] += 1
                dados["ultimo"] = agora
    finally:
        for sock in soquetes.values():
            sock.close()
    for dados in remetentes.values():
        janela = max(dados["ultimo"] - dados["primeiro"], 1e-6)
        dados["hz"] = dados["amostras"] / janela
        dados["idade_s"] = time.monotonic() - dados["ultimo"]
    return {"contagem": contagem, "remetentes": remetentes,
            "total": sum(contagem.values())}


def cria_regra_firewall(portas_=None):
    """Cria a regra de entrada UDP das luvas (precisa de ADMINISTRADOR)."""
    portas_ = portas_ if portas_ is not None else portas()
    comando = PS_CRIA_REGRA.format(
        nome=NOME_REGRA_FIREWALL,
        portas=",".join(str(porta) for porta in sorted(portas_.values())))
    ok, texto = roda_powershell(comando)
    if not ok and ("denied" in texto.lower() or "negado" in texto.lower()):
        return False, ("sem privilegio de administrador. Abra o PowerShell "
                       "como Administrador e rode:\n    " + comando)
    return ok, texto.strip() or comando


def remove_regra_firewall():
    """Remove a regra criada por `cria_regra_firewall()` (idempotente)."""
    ok, texto = roda_powershell(
        PS_REMOVE_REGRA.format(nome=NOME_REGRA_FIREWALL))
    return ok or not texto.strip(), texto.strip() or NOME_REGRA_FIREWALL


def relatorio(estado, rede, portas_, problemas, avisos):
    """Imprime o estado medido, item por item (o veredito vem depois)."""
    hostname = estado.get("hostname") or "?"
    endereco = estado.get("ics") or rede["ip_host"]
    ips = {ip for ip, _ in estado.get("ips", [])}
    ligado = endereco in ips
    cobrem = regras_que_cobrem(estado, portas_)
    print("=== HOTSPOT DO PC PARA OS ESP32 (luvas IMU) ===")
    print(f"  computador .....: {hostname}")
    print(f"  SSID do hotspot : {rede['ssid']}")
    print(f"  senha ..........: {rede['senha']}")
    print(f"  IP do PC (ICS) .: {endereco}"
          + ("" if estado.get("ics") else "  (registro ilegivel: valor de "
             "referencia)"))
    print("  portas UDP .....: " + ", ".join(
        f"{porta} (lado {lado})" for lado, porta in sorted(portas_.items())))
    print(f"  [{'OK' if ligado else '--'}] hotspot LIGADO (interface com "
          f"{endereco})")
    for ip, alias in estado.get("ips", []):
        print(f"        {ip:<16} {alias}")
    politica = estado.get("politica") or {}
    quase = regras_quase(estado, portas_)
    print(f"  [{'OK' if cobrem else '--'}] firewall de ENTRADA UDP "
          + (f"liberado por: {', '.join(cobrem)}" if cobrem
             else "nenhuma regra libera as portas para ESTE python"))
    print(f"        politica de entrada: {politica or 'nao lida'} | "
          f"{len(regras_de_entrada_ativas(estado))} regras de entrada ativas "
          "no registro")
    print(f"        receptor (tem de estar em App= de alguma regra): "
          f"{sys.executable}")
    for nome, app in quase[:3]:
        print(f"        quase: a regra '{nome}' libera UDP, mas so' para "
              f"{app or '(sem App)'}")
    for nome, cobertas, faltando, app in regras_parciais(estado, portas_)[:3]:
        print(f"        PARCIAL: a regra '{nome}' libera so' "
              f"{'/'.join(str(p) for p in cobertas)} (falta "
              f"{'/'.join(str(p) for p in faltando)})"
              + (f" -- app={app}" if app else " -- qualquer programa"))
    for porta, livre in sorted(estado.get("portas_livres", {}).items()):
        print(f"  [{'OK' if livre else '--'}] porta UDP {porta} livre para o "
              "receptor")
    if not estado.get("ps_ok"):
        print("  [--] nao deu para consultar o firewall/ips pelo PowerShell")
    for texto in problemas:
        print(f"  FALTA: {texto}")
    for texto in avisos:
        print(f"  nota : {texto}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--portas", default="4210,4211",
                        help="Portas das luvas, na ordem direita,esquerda "
                             "(aceita 'lado:porta')")
    parser.add_argument("--escutar", type=float, default=0.0,
                        help="Escuta as portas por N segundos (0 = nao "
                             "escuta)")
    parser.add_argument("--criar-regra-firewall", action="store_true",
                        help="Cria a regra de ENTRADA UDP (terminal de "
                             "Administrador)")
    parser.add_argument("--remover-regra-firewall", action="store_true",
                        help="Remove a regra criada por este utilitario")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        portas_ = dict(parse_portas_imu(args.portas))
    except ValueError as exc:
        print(f"portas invalidas ({exc})")
        return 1

    if args.remover_regra_firewall:
        ok, texto = remove_regra_firewall()
        print(f"{'regra removida' if ok else 'NAO removida'}: {texto}")
    if args.criar_regra_firewall:
        ok, texto = cria_regra_firewall(portas_)
        print(f"{'regra criada' if ok else 'NAO criada'}: {texto}\n")

    estado = estado_da_rede(portas_)
    ok, problemas, avisos = avalia(estado, REDE_ESP32, portas_)
    relatorio(estado, REDE_ESP32, portas_, problemas, avisos)

    endereco = estado.get("ics") or REDE_ESP32["ip_host"]
    print("=== SNIPPET PARA O FIRMWARE DO ESP32 ===")
    print(snippet_firmware(REDE_ESP32, portas_, endereco_host=endereco))

    recebeu = None
    if args.escutar > 0:
        print(f"\n=== ESCUTA ({args.escutar:g} s): procure os ESP32 ===")
        dados = escuta(portas_, args.escutar)
        recebeu = dados["total"] > 0
        for (ip, porta), info in sorted(dados["remetentes"].items()):
            print(f"  {ip}:{porta} -> {info['amostras']} datagramas, "
                  f"{info['hz']:.1f} Hz, ultimo ha {info['idade_s']:.2f} s")
        for porta, quantos in sorted(dados["contagem"].items()):
            if not quantos:
                print(f"  porta {porta}: nada")
        if not recebeu:
            problemas.append(
                "nenhum datagrama na escuta. Confira nesta ordem: (1) o "
                "hotspot esta' ligado e em 2,4 GHz; (2) o ESP32 CONECTOU "
                "(imprima `WiFi.localIP()` na Serial); (3) o destino no "
                f"firmware e' {endereco} na porta certa; (4) a regra de "
                "firewall; (5) `arp -a | Select-String 192.168.137`")

    print("\n=== VEREDITO ===")
    if problemas:
        print(f"  FALTA ARRUMAR ({len(problemas)}):")
        for texto in problemas:
            print(f"    - {texto}")
    else:
        print("  PRONTO: os ESP32 podem conectar no hotspot e mandar UDP para "
              f"{endereco}"
              + (f" (escuta recebeu {dados['total']} datagramas)"
                 if recebeu else ""))
    return 0 if not problemas else 1


if __name__ == "__main__":
    sys.exit(main())

