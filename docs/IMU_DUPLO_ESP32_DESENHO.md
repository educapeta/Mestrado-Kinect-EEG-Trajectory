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

## 4. Colunas do CSV: a decisão que falta

Hoje o bloco de IMU tem **7 colunas** (`IMU_roll/pitch/yaw_deg`,
`IMU_pos_x/y/z_m`, `ZERO_lock`) e o movimento tem **45** no total. Três opções:

| Opção | Colunas | Prós | Contras |
| --- | --- | --- | --- |
| **A — manter as 7 + `IMU_hand`** (o IMU "principal" passa a ser o da mão ativa) | **46** | mudança mínima; todo o resto continua válido | perde a evidência do lado contralateral |
| **B — blocos por lado** (`IMU_L_*`, `IMU_R_*`: rpy + pos + accel bruta + `ZERO_lock` + `IMU_valid`) | ~**56** | dado completo; permite **replay offline do Kalman** com a aceleração crua | quebra o contrato de colunas (2 testes) e toda leitura de `IMU_*` |
| **C — recomendada: A + bloco contralateral enxuto** (`IMU_hand`, 7 colunas do lado ativo, `IMU_ctrl_acc_x/y/z_g`, `ZERO_lock_ctrl`) | **52** | mantém compatibilidade do bloco antigo e traz a aceleração crua (o que o Kalman precisa para validar offline) + o controle contralateral | exige rotular com clareza na documentação |

Ponto que **qualquer** opção deve incluir: a **aceleração bruta** (`IMU_acc_*`).
Hoje ela é descartada — e sem ela não há como validar o filtro de Kalman offline
nas sessões reais (o teste sintético existe, mas o dado real não pode ser
reprocessado).

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

| # | Item | Onde |
| --- | --- | --- |
| 1 | firmware: porta por lado (2.1) ou ID no pacote (2.2) | projeto do ESP32 |
| 2 | decidir as colunas (seção 4) e ajustar `IMU_COLUMNS`/`build_motion_row` | `eeg_motor_paradigm.py` |
| 3 | atualizar o contrato de colunas nos testes | `test_paradigm_protocol.py`, `test_arm_csv.py` |
| 4 | ligar o `ImuBank` no `TrackingThread` (fusão por lado) e publicar o lado **ativo** no bloco principal | `eeg_motor_paradigm.py` |
| 5 | tempo real: um `KalmanTrajectory` por lado, usando a aceleração do **lado ativo** do trial | `sand_traj_tempo_real.py` |
| 6 | **taxa de falso movimento** usando o contralateral como referência | `tools/` |

