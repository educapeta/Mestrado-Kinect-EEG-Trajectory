# Luvas IMU — firmware (ESP32 + MPU6050)

Firmware das **duas luvas** do experimento: um ESP32 + um MPU6050 por mão. Cada
luva lê o IMU, filtra `roll`/`pitch` com Kalman, integra o `yaw` e publica **uma
amostra por datagrama UDP** (e a mesma linha em CSV na serial).

Esta pasta é o **firmware**, dentro do repositório do Mestrado:
<https://github.com/educapeta/Mestrado-Kinect-EEG-Trajectory>. Aqui ficam só os
dois ESP32; quem consome os dados (receptor UDP, `ImuBank`, fusão de posição,
gravação no CSV de movimento, EEG + Kinect) está na **raiz** do repositório —
`imu.py`, `tools/teste_imu_dois_esp32.py` e `docs/IMU_DUPLO_ESP32_DESENHO.md`.

## 1. Hardware

| Item | Detalhe |
| --- | --- |
| Placa | ESP32-WROOM-32 (env `board = upesy_wroom`), framework Arduino |
| IMU | MPU6050, I2C a 400 kHz, endereço **0x68** (AD0 no GND) |
| SDA | GPIO **22** |
| SCL | GPIO **23** |
| Alimentação | 3,3 V (atenção: o MPU6050 não é tolerante a 5 V no I2C) |
| Sensor configurado | giroscópio ±250 °/s, acelerômetro ±2 g, DLPF desligado |

## 2. Uma porta UDP por lado (contrato com o projeto do Mestrado)

| Lado | env do PlatformIO | `GLOVE_SIDE` | Porta UDP |
| --- | --- | --- | --- |
| direita | `luva_direita` | 1 | **4210** |
| esquerda | `luva_esquerda` | 0 | **4211** |

Convenção do projeto: `lado 1 = direita`, `lado 2 = esquerda` (`ARM_SIDE_CODE`,
igual a `KT_hand`/`ARM_side`) e `imu.py: IMU_PORTAS_PADRAO = {1: 4210, 2: 4211}`.
Se a porta trocar de lado, os blocos `IMU_L_*`/`IMU_R_*` do CSV de movimento
entram **trocados**.

## 3. Formato do pacote

```
t_us,roll,pitch,yaw,ax,ay,az,gx,gy,gz
```

- `t_us`: `micros()` do ESP32, capturado logo após a leitura do sensor;
- `roll`, `pitch`, `yaw`: graus — `roll`/`pitch` combinam acelerômetro (referência
  da gravidade) e giroscópio por filtro de Kalman; `yaw` usa só o giroscópio e
  por isso **acumula drift**;
- `ax,ay,az`: g · `gx,gy,gz`: graus/s (com o bias do giroscópio já removido);
- 10 campos separados por vírgula, `\n` no fim; pacote inválido é descartado
  em silêncio pelo receptor;
- no boot o giroscópio é calibrado com o sensor parado (500 amostras) e a
  serial (115200 baud) recebe o cabeçalho
  `esp_timestamp_us,roll_deg,pitch_deg,yaw_deg,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps`
  seguido das mesmas linhas enviadas por UDP;
- taxa: o `loop()` tem `delay(5)` ⇒ aprox. 100–200 Hz (o limitador é o Wi-Fi,
  não o MPU6050).

## 4. Rede

### 4.1 Bancada (modo usado hoje): hotspot do notebook

O notebook vira **Ponto de acesso móvel** com IP fixo `192.168.137.1` e o
firmware envia **unicast** para esse IP. Motivo: rede institucional (eduroam)
usa *client isolation* e derruba o UDP entre clientes.

```powershell
Get-Service icssvc                       # Windows Mobile Hotspot: Running
Get-NetIPAddress -AddressFamily IPv4 | Where-Object IPAddress -like '192.168.137.*'
arp -a | Select-String '192.168.137'     # as duas luvas, depois de conectarem
```

### 4.2 eduroam (WPA2-Enterprise, EAP-TTLS + PAP)

O ESP32 **não** conecta na eduroam com `WiFi.begin(ssid, senha)`: precisa da
sobrecarga com identidade/usuário (`WPA2_AUTH_TTLS`), implementada em
`tentarEnterprise()`. Cada senha errada **conta no RADIUS da UFSC** e várias
podem bloquear a conta ⇒ mantenha `WIFI_MAX_TENTATIVAS` em 1.

### 4.3 broadcast × unicast

`UDP_USAR_UNICAST 1` → envia para `UDP_DESTINO` (ex.: `192.168.137.1`);
`UDP_USAR_UNICAST 0` → broadcast `255.255.255.255` (redes domésticas, sem
isolation). Se a macro não existir no `credenciais.h`, o `#if` a trata como 0 e
o firmware vai de broadcast.

## 5. Credenciais — não versionadas

`src/credenciais.h` contém **senha** e está no `.gitignore`. Ao clonar (ou
trocar de rede), copie o template e preencha:

```powershell
copy src\credenciais.h.example src\credenciais.h
```

| Macro | Para que serve |
| --- | --- |
| `WIFI_ENTERPRISE` | 1 = tenta a eduroam (WPA2-Enterprise) antes do fallback |
| `WIFI_SSID_ENTERPRISE` | SSID da rede enterprise (`eduroam`) |
| `WIFI_IDENTIDADE` / `WIFI_USUARIO` | identidade externa e usuário interno (o mesmo e-mail) |
| `WIFI_TIMEOUT_MS` | tempo máximo esperando a autenticação 802.1X de uma senha |
| `WIFI_MAX_TENTATIVAS` | quantas senhas de `WIFI_SENHAS[]` são testadas |
| `WIFI_SENHAS[]` | candidatas testadas na ordem (evite várias: RADIUS) |
| `WIFI_PSK_FALLBACK` | 1 = cai para uma rede WPA2-PSK comum |
| `WIFI_SSID_PSK` / `WIFI_SENHA_PSK` | rede PSK (hotspot do notebook, casa, celular) |
| `UDP_USAR_UNICAST` | 1 = unicast; 0 = broadcast |
| `UDP_DESTINO` | IP do notebook (fixo `192.168.137.1` no hotspot) |

## 6. Compilar e gravar

Pré-requisito: **PlatformIO Core** (instalado junto com a extensão PlatformIO
IDE do VS Code, ou via `pip install platformio`).

```powershell
pio run                                              # compila os dois envs
pio run -e luva_direita  -t upload --upload-port COM6
pio run -e luva_esquerda -t upload --upload-port COM7
```

Ou use os atalhos, que perguntam a porta e mostram o procedimento:
`gravar_direita.bat` (UDP 4210) e `gravar_esquerda.bat` (UDP 4211).

Se a placa não tiver auto-reset, na hora do upload: acompanhe o terminal e
**segure o BOOT** quando aparecer `Connecting....`, soltando só depois de
aparecer `Writing at 0x000...`. Para descobrir a porta: `pio device list`.

## 7. Conferir na bancada

```powershell
python receber_luvas.py        # só stdlib: mostra as duas luvas (4210/4211)
```

A inspeção completa (taxa, jitter, idade da amostra e gravação) é feita pelo
projeto do Mestrado:

```powershell
.venv\Scripts\python.exe tools\teste_imu_dois_esp32.py --segundos 20 --gravar _imu.csv
```

Se nada chega: (a) hotspot ligado; (b) firewall do Windows liberando **UDP de
entrada** em 4210/4211; (c) `UDP_DESTINO` igual ao IP do PC no `credenciais.h`;
(d) o ESP32 só usa Wi-Fi **2,4 GHz**; (e) `idade` subindo e `Hz` caindo costuma
ser Wi-Fi/bateria.

## 8. O que não vai para o git (e por quê)

| Caminho | Motivo |
| --- | --- |
| `.pio/` | artefatos de build (recompilam sozinhos) |
| `.vscode/c_cpp_properties.json`, `.vscode/launch.json`, `.vscode/.browse.*` | gerados com caminhos absolutos da máquina |
| `src/credenciais.h` | contém senha de rede |

Numa máquina nova basta clonar o repositório do Mestrado e abrir **esta pasta**
como projeto no PlatformIO (VS Code → *Open Folder* → `firmware\esp32_luvas`),
criar o `credenciais.h` (seção 5) e rodar `pio run` — não é preciso copiar a
pasta `.pio/` (ela se refaz).

## 9. Estrutura desta pasta

```
platformio.ini              dois envs (GLOVE_SIDE = 1 ou 0)
src/main.cpp                firmware: I2C + Kalman + Wi-Fi + UDP
src/credenciais.h.example   template das credenciais (copiar -> credenciais.h)
receber_luvas.py            receptor de bancada das duas luvas (UDP 4210/4211)
gravar_direita.bat          gravação em um comando (pergunta a porta)
gravar_esquerda.bat         idem, para a luva esquerda
unicast                     nota de bancada: SSID do hotspot + IP do notebook
include/ lib/ test/         pastas padrão do PlatformIO (vazias)
```
