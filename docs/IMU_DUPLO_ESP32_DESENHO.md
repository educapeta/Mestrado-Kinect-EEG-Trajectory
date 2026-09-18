# Dois IMUs (um por mão): desenho de rede, firmware e dados

> Decisão de hardware de 18/09: dois MPU6050, um em cada mão, possivelmente com um
> **ESP32 por mão**. Este documento fixa a especificação (rede, formato do
> pacote, colunas do CSV, armadilhas) e lista o que já está implementado.

## 1. Por que isso conserta um problema do protocolo atual

Hoje há **um** IMU e o protocolo **alterna as mãos** a cada trial. Resultado: em
metade dos trials o IMU está na mão que **não** executa, e as colunas
`IMU_roll/pitch/yaw` e `IMU_pos_*` **não dizem de qual mão são** — o dado é
ambíguo. Com dois IMUs:

1. os trials das duas mãos passam a ter IMU **da mão correta**;
2. cada IMU tem o **seu** bias e a sua zeragem (não se mistura calibração de
   sensores diferentes);
3. a mão que **não** executa vira uma **referência de repouso medida por
   hardware** — é o que dá substância à *taxa de falso movimento*: se o modelo
   prevê movimento enquanto os dois IMUs estão parados, é falso positivo, ponto.

## 2. Rede e firmware

### 2.1 Recomendado: um **ESP32 por mão**, uma **porta UDP por lado**

| Lado | Código (convenção do projeto) | Porta UDP |
| --- | --- | --- |
| direita | 1 | **4210** |
| esquerda | 2 | **4211** |

O firmware muda **uma linha** (a porta de destino). Sem ambiguidade: quem manda
na 4210 é a direita.

### 2.2 Alternativa: um ESP32 com dois MPU6050 (ou dois sensores no mesmo barramento)

Aí é preciso um **ID no pacote**, porque as duas amostras chegam pelo mesmo
socket. Formato sugerido (mantendo o ASCII atual e acrescentando o ID à frente):

```
lado,t_us,roll,pitch,yaw,ax,ay,az,gx,gy,gz      # lado = 1 (dir) ou 2 (esq)
```

O `ImuReceiver` aceitaria os dois formatos: 10 campos = legado (um IMU na porta),
11 campos = com ID, usado para separar os lados. (Se optar por essa via, eu
implemento o parser tolerante + teste.)

### 2.3 Formato do pacote hoje (por referência)

```
t_us,roll,pitch,yaw,ax,ay,az,gx,gy,gz          # ASCII, 1 datagrama por amostra
```

- `t_us` = tempo **do ESP32** (µs), `roll/pitch/yaw` em graus, `accel` em **g**,
  `gyro` em **graus/s**;
- taxa sugerida: **100 Hz por mão** (o MPU6050 aguenta mais; quem limita é o Wi-Fi
  e o jitter do UDP);
- datagramas vazios/truncados/NaN são descartados em silêncio (o receptor nunca
  morre por pacote ruim).

## 3. Armadilhas que o desenho tem de respeitar

| # | Armadilha | Regra |
| --- | --- | --- |
| 1 | **Dois relógios independentes** (`t_us` de cada ESP32) | amostras de lados **diferentes** nunca entram no mesmo filtro: **um `ImuReceiver` e um `PositionFusion` por lado** (`ImuBank`, já implementado). O `dt` vem do `t_us` do *próprio* dispositivo — misturar dois relógios geraria `dt` absurdo |
| 2 | **Bias e orientação de montagem** são por sensor | calibração de bias e zeragem de orientação (relação com a origem) **por lado**; nunca reutilizar a de um lado no outro |
| 3 | **Atraso/queda do Wi-Fi** | usar a idade da amostra (`ImuBank.idade_s`) como vigia, no espírito do `LinkWatchdog` do EEG (899/898): idade > ~0,5 s ⇒ lado sem medida |
| 4 | **Ordem dos lados** | seguir `ARM_SIDE_CODE` (`1 = direita`, `2 = esquerda`), a **mesma** convenção de `KT_hand`/`ARM_side`, para casar com o trial |
| 5 | **Pacotes fora de ordem** | o `PositionFusion` já ignora `dt` fora de `(0; 0,1]` s; com jitter isso pode descartar amostras boas — vale contar os descartes |

## 4. Colunas do CSV: **DECIDIDO — opção B** (blocos por lado)

Medido nos arquivos reais: o movimento custa ~5,7 B por coluna por linha (30 Hz).
Numa sessão de 3000 s (250 trials):

| Esquema | Colunas (arquivo) | Movimento | EEG | Total da sessão |
| --- | --- | --- | --- | --- |
| antigo (1 IMU, 7 colunas) | 45 | 21,8 MiB | ~305 MiB | 326 MiB |
| A (lado ativo + accel) | 52 | 25,2 MiB | ~305 MiB | 329 MiB (+0,9 %) |
| C (A + contralateral enxuto) | 54 | 26,2 MiB | ~305 MiB | 331 MiB (+1,5 %) |
| **B (blocos por lado) — ESCOLHIDA** | **61** | **29,6 MiB (+7,8)** | ~305 MiB | **334 MiB (+2,4 %)** |

**Tamanho não é argumento**: o EEG é ~14× maior que o movimento, e B custa +2,4 %
da sessão (~+16 MiB por sujeito com 2 sessões). Mais informação ganha.

Implementado em 18/09 (`eeg_motor_paradigm.py`):

- `IMU_COLUMNS = _imu_block("L") + _imu_block("R") + ["IMU_hand"]` — **23 colunas**
  (11 por lado: rpy, **aceleração bruta em g**, posição fusionada, `valid`,
  `ZERO_lock`), com `IMU_hand` = 1 direita / 2 esquerda / 0 desconhecido;
- `MOTION_COLUMNS` = **59** (arquivo: **61** com `t_mono_s`/`t_epoch_s`);
- `_imu_do_lado()`: lê `motion["imu"][lado]` (dois IMUs) **ou** as chaves legadas
  (um só IMU → o bloco do lado ativo; com a mão desconhecida, o lado declarado
  em `--imu-lado`, padrão direita — **nunca duplicando** o mesmo sensor nos dois
  blocos);
- a zeragem de orientação é **por lado** (cada sensor declara o seu `orpy_deg`);
- `test_arm_csv.py` → 6/6 (inclui um caso novo com os dois IMUs independentes).

Custo real de B (e o motivo de ser barato): **o treino não lê coluna de IMU**
(`MotionTrack` lê só `t_mono_s`, `KT_*`, `KT_src` e `ARM_*`) e o JSON de eventos
já publica `colunas_movimento` — então a análise deve ser escrita **contra o
JSON**, não contra nomes fixos.

## 4.1 Como testar as luvas (sem EEG, sem Kinect, sem g.Pype)

```powershell
# (1) bancada: as duas luvas ligadas -> mostra taxa, jitter, idade e assinatura
.venv\Scripts\python.exe tools\teste_imu_dois_esp32.py --segundos 20 --gravar _imu.csv
#     se as duas mandarem para a MESMA porta:
.venv\Scripts\python.exe tools\teste_imu_dois_esp32.py --porta 4210 --por-remetente
#     (fixe o mapa se a ordem sair trocada:  --ips "192.168.0.101=1,192.168.0.102=2")

# (2) sessao REAL com as duas luvas (uma porta por lado)
.venv\Scripts\python.exe eeg_motor_paradigm.py --imu-portas 4210,4211 ...

# (3) sem hardware: dois emissores falsos fazem o papel das luvas
.venv\Scripts\python.exe tools\emissores_imu_falsos.py --segundos 30
```

Assinaturas dos emissores falsos: **direita** = `roll +5°` e 1,2 g no eixo vertical;
**esquerda** = `roll −5°` e 1,0 g — assim da' para ver num relance se os lados
estao trocados.

Checagens rápidas se algo não chegar: (a) as duas luvas no mesmo Wi-Fi e com o
**IP do PC** configurado no firmware; (b) **firewall do Windows** liberando UDP
(entrada) nas portas escolhidas; (c) a porta de destino do firmware (uma por mão
ou `--por-remetente`); (d) se `idade` sobe e `Hz` cai, é Wi-Fi/bateria.

## 5. O que já está implementado (18/09)

- `imu.py`: **`ImuBank`** — um `ImuReceiver` + um `PositionFusion` **por lado**,
  com `start/stop`, `amostra(lado)`, `idade_s(lado)`,
  `accel_no_referencial(lado)`, `atualiza_fusao(lado, posicao_camera)` e
  `resumo()`; `parse_portas_imu("4210,4211")` (aceita `lado:porta`) e
  `IMU_PORTAS_PADRAO = {1: 4210, 2: 4211}`.
- Teste `test_imu_bank.py` (5/5) com **dois emissores UDP falsos**: confirma a
  atribuição por porta, a conversão de aceleração por lado (parado e inclinado
  30° → 4e-16 m/s²) e a independência das fusões.

## 6. O que falta (checklist)

| # | Item | Onde | Status |
| --- | --- | --- | --- |
| 1 | firmware: porta por lado (2.1) ou ID no pacote (2.2) | projeto do ESP32 | **com o usuário** |
| 2 | decidir as colunas (seção 4) e ajustar `IMU_COLUMNS`/`build_motion_row` | `eeg_motor_paradigm.py` | ✅ opção B implementada |
| 3 | atualizar o contrato de colunas nos testes | `test_arm_csv.py`, `test_paradigm_protocol.py` | ✅ 6/6 e 9/9 |
| 4 | ligar o `ImuBank` no `TrackingThread` (`--imu-portas`) e publicar `motion["imu"][lado]` | `eeg_motor_paradigm.py` | ✅ (rastreador com Kinect **e** modo `--sem-kinect`) |
| 5 | tempo real: um `KalmanTrajectory` por lado, com a aceleração do **lado ativo** | `sand_traj_tempo_real.py` | pendente |
| 6 | **taxa de falso movimento** usando o contralateral como referência | `tools/` | pendente |
| 7 | bancada: `tools/teste_imu_dois_esp32.py` (2 modos + gravação) e `tools/emissores_imu_falsos.py` | `tools/` | ✅ |
| 8 | teste da gravação simultânea sem hardware | `test_imu_gravacao_dual.py` (5/5) | ✅ |

