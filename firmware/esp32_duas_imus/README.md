# ESP32 com 2 MPU6050 — firmware (as duas mãos num ESP32 só)

Firmware de **um único ESP32** que lê **dois MPU6050** (um por mão) no mesmo
barramento I2C e publica **dois datagramas UDP por ciclo** — porta **4210**
(mão direita, MPU @ 0x68) e porta **4211** (mão esquerda, MPU @ 0x69).
Substitui os dois ESP32 independentes de `../esp32_luvas/`.

## 1. Hardware

| Item | Detalhe |
| --- | --- |
| Placa | ESP32-WROOM-32 (`board = upesy_wroom`), framework Arduino |
| IMUs | 2× MPU6050, I2C 400 kHz, endereços **0x68 (direita)** e **0x69 (esquerda)** |
| SDA / SCL | **varridos no boot**: o sondeo testa TODOS os pares de GPIO seguros (4,5,13..33) nos endereços 0x68/0x69 e imprime o par achado (`[i2c] barramento vivo em SDA=.. SCL=.. (endereco 0x..)`) |
| Alimentação | 3,3 V (MPU6050 não tolera 5 V no I2C) |
| Sensor | giroscópio ±250 °/s, acelerômetro ±2 g |

## 2. Uma porta UDP por lado (contrato com `imu.py`)

| MPU (endereço I2C) | Lado | Porta UDP | Receptor no PC |
| --- | --- | --- | --- |
| **0x68** | direita | **4210** | `ImuReceiver(4210)` → lado 1 |
| **0x69** | esquerda | **4211** | `ImuReceiver(4211)` → lado 2 |

Convenção: `lado 1 = direita`, `lado 2 = esquerda`
(`imu.IMU_PORTAS_PADRAO = {1: 4210, 2: 4211}`). O firmware **não** define
`GLOVE_SIDE` — publica os dois lados ao mesmo tempo.

**Trocar qual MPU é direita/esquerda não exige trocar cabos**: edite os
`build_flags` do `platformio.ini` e recompile:

```ini
build_flags = -D ADDR_DIREITO=0x69 -D ADDR_ESQUERDO=0x68   # troca os lados
```

O lado é o **endereço I2C** (AD0 do MPU6050: GND→0x68, VCC→0x69), não a porta
serial — trocar fisicamente as luvas vira uma recompilada.

## 3. Formato do pacote (UDP) e da linha serial

```
t_us,roll,pitch,yaw,ax,ay,az,gx,gy,gz
```

- `t_us`: `micros()` logo após a leitura do sensor;
- `roll`/`pitch`: graus, fusão acelerômetro+giroscópio (Kalman, um filtro por
  MPU); `yaw`: só giroscópio, **acumula drift**;
- `ax,ay,az` em g; `gx,gy,gz` em °/s com bias removido;
- **UDP**: 10 campos, `\n` no fim; cada MPU gera um pacote próprio — quem
  separa é a **porta UDP** (o receptor 4210 só vê o MPU 0x68), sem tag de
  lado (`imu.py` não mudou);
- **Serial** (115200): a MESMA linha com o **prefixo do lado** — `1,<10
  campos>` (direita) ou `2,<10 campos>` (esquerda). É o contrato do
  `tools/imu_serial_bridge.py` (o bridge descarta qualquer linha sem o
  prefixo);
- no boot os pinos I2C são **sondeados** (22/23, 21/22, 22/21) e o giroscópio
  de **cada** MPU é calibrado (500 amostras, sensor parado); a serial (115200)
  recebe o cabeçalho `esp_timestamp_us,roll_deg,...,gz_dps` seguido das
  linhas **com o prefixo do lado** (`1,...` / `2,...`);
- taxa: `delay(5)` no loop → ~100 Hz por MPU (o limitador é o Wi-Fi).

## 4. Rede (idêntica ao firmware antigo)

Bancada: hotspot do notebook (SSID `EDUARDO-PC 9049`, 2,4 GHz), IP fixo
`192.168.137.1`, **unicast** (`UDP_USAR_UNICAST 1`). eduroam fica desligada
(`WIFI_ENTERPRISE 0` — cada senha errada conta no RADIUS; client isolation
derruba o UDP). Credenciais: `src/credenciais.h` (gitignored; copiar de
`credenciais.h.example`).

**Sem hotspot** (bancada só-USB) o firmware re-tenta o Wi-Fi com **backoff**:
6 s de tentativa a cada 15 s no loop (a primeira, no boot, tem 20 s) — antes
CADA volta do loop bloqueava 20 s e os dados só saíam em rajadas.


## 5. Compilar e gravar

**ATENÇÃO nesta máquina (07/10/2026):** o python do `.venv` é o da **Store/MSIX**
e **não consegue escrever** em `C:\Users\Usuario\.platformio\` (PermissionError
no `platforms.lock`) — use o **pio do penv** (1ª opção dos `.bat`):

```powershell
C:\Users\Usuario\.platformio\penv\Scripts\pio.exe run -e esp32_duas_imus
```

Ou use o atalho `gravar.bat [COMx]` (padrão COM4): ele escolhe o pio certo e
**repete o upload automaticamente** (10×, 4 s de intervalo; `set
GRAVAR_TENTATIVAS=1` muda para teste rápido) — basta segurar o BOOT e esperar:

```
gravar.bat COM4
```

**Gravação (auto-reset NÃO funciona nesta placa):** ao aparecer
`Connecting........`, **SEGURE o BOOT**; solte só em `Writing at 0x000...`.
Cada tentativa individual do esptool dura ~12 s; sem o BOOT ela falha com
`Wrong boot mode detected (0x13)` (= modo app, não download) — por isso o
auto-retry embutido no .bat (fluxo completo testado em 09/10/2026).

Estado medido em 08/10/2026: **compila** — RAM 13,9 % (45652 B), Flash 57,8 %
(757897 B). Diferença sobre 07/10: sondeo de pinos I2C, prefixo do lado na
serial, fim do mascaramento D→E e backoff de Wi-Fi.

## 6. Conferir na bancada

```powershell
# hotspot ligado em 2,4 GHz e firewall UDP 4210+4211 liberados:
.venv\Scripts\python.exe tools\diagnostico_hotspot.py --escutar 15
# deve mostrar remetente 192.168.137.x ~100 Hz NAS DUAS portas (4210 e 4211)
```

Alternativa (sem Wi-Fi): a serial USB sai no mesmo formato com o prefixo do
lado — **`tools/imu_serial_bridge.py`** faz a ponte serial→UDP:

```powershell
.venv\Scripts\python.exe tools\imu_serial_bridge.py --port COM4 --escutar 10
```

Diagnóstico de bancada (serial crua classificada: sondeo dos pinos I2C, bias
por MPU, erros D/E, dados por lado; `60` para o veredito completo, `10` para
só o sondeo):

```powershell
.venv\Scripts\python.exe tools\diagnostico_imu_serial.py COM4 115200 60
```

Depois, o paradigma roda igual (o `imu.py` não mudou — ele recebe via UDP):

```powershell
.venv\Scripts\python.exe eeg_motor_paradigm.py --source gnautilus
```

## 7. Estrutura desta pasta

```
platformio.ini              env esp32_duas_imus (ADDR_DIREITO/ADDR_ESQUERDO)
src/main.cpp                2× I2C + 2× Kalman + Wi-Fi + 2× UDP (4210/4211)
src/credenciais.h           local (gitignored, contém senha)
src/credenciais.h.example   template
gravar.bat                  gravação guiada (pergunta porta, mostra BOOT)
```

## 8. O que não vai para o git

| Caminho | Motivo |
| --- | --- |
| `.pio/` | artefatos de build |
| `src/credenciais.h` | contém senha de rede |
