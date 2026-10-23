# Revisão do firmware do MPU6050 × Kalman 1D do vídeo (22/10/2026)

Escopo: `firmware/esp32_duas_imus/src/main.cpp` (o firmware **ativo** na bancada:
dois MPU6050 no mesmo I2C, endereços 0x68/0x69, UDP 4210/4211 — é o que aparece
no log como `[imu] calibrando giroscopio MPU 0x68/0x69`) e
`firmware/esp32_luvas/src/main.cpp` (versão antiga de uma luva por ESP32, mesmo
filtro). Pergunta: o firmware segue a lógica do vídeo "1D Kalman filter"?

## 1. O que o vídeo faz

O vídeo implementa **dois filtros de Kalman 1D independentes**, um para roll e um
para pitch:

1. **Predição pelo giroscópio**: `angle += dt * gyroRate` (o código da série
   também estima o bias do giro; no vídeo a versão mostrada é a mais simples).
2. **Incerteza da predição**: `P += dt² · σ_gyro²`, com `σ_gyro = 4 °/s`
   (estimativa do erro do giroscópio).
3. **Medição**: `angleAcc = atan2(...)` calculado por **trigonometria do
   acelerômetro** (com as constantes de calibração do acelerômetro).
4. **Ganho de Kalman**: `K = P / (P + σ_accel²)`, com `σ_accel = 3°`.
5. **Atualização**: `angle += K · (angleAcc − angle)`; `P = (1 − K) · P`.
6. Estado inicial: `angle = 0`, `P = 2² = 4` (incerteza de 2 graus), porque o
   quadricóptero parte de uma superfície aproximadamente nivelada.
7. **Não existe yaw**: o vídeo só calcula roll e pitch, os únicos ângulos
   observáveis com acelerômetro + giroscópio (sem magnetômetro).

Características: filtro linear, escalar, ganho ótimo pela razão das variâncias;
suaviza vibração no acelerômetro sem acumular erro de integração.

## 2. O que o firmware faz de verdade

O firmware **não usa esse Kalman para a atitude**. A classe `KalmanAngle`
(`esp32_duas_imus/src/main.cpp` linhas ~50-79) existe, mas o próprio código
declara (linhas ~190-192):

> "Os filtros de Kalman de angulo NAO sao mais usados para a atitude: quem
> integra agora e' o AttitudeMahony (acima), que tambem da' o yaw (o Kalman
> antigo nao conseguia...). A classe ficou como referencia/historico."

A atitude real é um **filtro complementar não linear de Mahony com quaternion**
(classe `AttitudeMahony`, linhas ~94-186), 6 eixos:

```cpp
// update()
wx, wy, wz = giroscópio em rad/s (bias de boot já subtraído)
if (|a| ~ 1 g) {                        // só corrige se não há aceleração linear
    u = R^T * (0,0,1);                  // "para cima" prevista, no corpo
    wx += KP*(ay*uz - az*uy);           // erro de inclinação
    wy += KP*(az*ux - ax*uz);
    wz += KP*(ax*uy - ay*ux);
}
q += 0.5 * dt * q ⊗ (0, wx, wy, wz);    // integra a rotação como quaternion
q /= |q|;                               // normaliza
```

- `KP = 2.0 1/s` (linha ~138): constante de tempo ~0,5 s para o tilt.
- Inicialização com o acelerômetro (same formulas do firmware antigo):
  `roll = atan2(ay, az)`, `pitch = atan2(-ax, hypot(ay,az))` (linhas ~111-118).
- Sai em Euler **ZYX**: `R = Rz(yaw) Ry(pitch) Rx(roll)`, exatamente a mesma
  convenção de `imu.rotation_matrix` no PC (verificado numericamente).
- **Yaw não tem referência absoluta**: é integrado pelo giroscópio e corrigido
  em inclinação; sem magnetômetro ele começa em 0 em cada boot e deriva com o
  bias residual. O projeto trata isso reancorando o yaw do boot com o Kinect
  (item 97 do histórico).

Outras características relevantes:

- **Bias do giroscópio**: 500 amostras a cada 4 ms (~2 s), por MPU, no boot
  (`calibrateGyroscopeAt`, linhas ~225-240). Não há estimação online.
  Se as luvas se mexem no boot, o bias sai errado (o projeto já mediu −12,5 °/s
  = 434 °/min; o ritual do tool, item 101, reseta o ESP32 e confere o bias por
  UDP).
- **Acelerômetro sem calibração**: `ax = rawAx / 16384.0f` direto (linhas
  ~410/448); **não há offsets de zero-g nem escalas** em lugar nenhum do
  firmware. O vídeo destaca explicitamente "put your own accelerometer
  calibration values".
- **DLPF e taxa de amostragem não configurados**: os registradores 0x1A
  (CONFIG/DLPF) e 0x19 (SMPLRT_DIV) não são escritos. No MPU6050 o padrão é
  DLPF desligado (banda do acelerômetro ~260 Hz, giroscópio ~256 Hz), enquanto o
  loop amostra a ~50-100 Hz — ou seja, ruído/vibração acima de Nyquist pode
  aliasar para dentro da banda.
- A classe `KalmanAngle` legada (não usada) no firmware dual **não** é o filtro
  do vídeo: `covariance` começa em zero, `measurementNoise = 0.03` e
  `processNoise = 0.001` (unidades e magnitudes diferentes de σ_gyro=4 °/s,
  σ_accel=3°, P0=4 do vídeo). Além disso o firmware dual tinha um termo de
  `processNoise` ausente na predição de P (inconsistente com `esp32_luvas`) —
  corrigido nesta revisão.

## 3. Verificações numéricas (simulação em Python do código do firmware)

Simulei o `AttitudeMahony` exatamente como está escrito, com a matriz de
verdade `R = Rz(yaw) Ry(pitch) Rx(roll)` e o acelerômetro medindo `R^T(0,0,1)`:

| teste | resultado |
|---|---|
| Estática ideal (roll=15°, pitch=−30°, yaw=40°) | roll/pitch exatos; yaw estimado 0° (ver nota de yaw) |
| Sinal da correção `KP·(a × u)` **correto** | partindo de roll/pitch errados (45°/10°), erro de tilt 32,7° → 6,6° → 0,86° → 0° em 0,2/1/2/5 s |
| Sinal **invertido** | divergiria para 180° (portanto o sinal do firmware está certo) |
| Rastreamento dinâmico (roll 0→30°, pitch 0→−20°, yaw 0→20° em 2 s) | erro de tilt máx **0,36°**, médio 0,29° |
| Bias de zero-g do acelerômetro (típico de clone): 0,03 g / 0,06 g / 0,06 g e −0,04 g e 0,02 g / 0,10 g e −0,10 g e 0,05 g | erro de tilt **1,5° / 2,9° / 3,6° / 7,1°** e **dependente da orientação** |
| Bias residual do giroscópio: 0,2 / 0,5 / 1,0 °/s (x,y,z) | erro de tilt 0,1–0,5° (corrigido pelo acelerômetro), mas yaw deriva 0,7 / 1,6 / 3,0° em 5 s |
| `KalmanAngle` legado (como escrito) | ganho do acelerômetro ~0 no início e só sobe para ~1 depois de ~20 s — não é o filtro do vídeo e não deveria ser usado |

Conclusões das simulações:

1. O **Mahony está implementado corretamente** (sinal, convergência,
   rastreamento dinâmico e convenção ZYX com o PC).
2. O que faz os ângulos parecerem "ruins" não é o filtro em si, e sim:
   - **acelerômetro sem calibração** → tilt sistematicamente errado em 1–7° e
     **dependente da orientação** (não é offset constante; não pode ser
     absorvido por um único `T/f` da calibração da palma);
   - **yaw não é absoluto** (sem magnetômetro): começa em 0 a cada boot e
     deriva; comparar o yaw do firmware com uma direção física/Norte não faz
     sentido;
   - **bias do giroscópio depende do boot** (precisa das luvas paradas; por
     isso o ritual do tool existe);
   - **DLPF/ODR não configurados**: ruído/vibração pode entrar na banda;
   - a classe Kalman legada não implementa o vídeo e tem problemas de
     covariância (não usar).
3. A lógica do vídeo **não está presente**; a arquitetura escolhida (Mahony)
   é a extensão natural do Kalman 1D para 3D e foi validada numericamente.
   O vídeo não tem yaw; o firmware tem yaw (integrado, não absoluto).

## 4. Evidência nos dados reais da bancada (deriva estática)

Analisando as gravações `deriva_estatica_20261021.csv` (um boot em que a luva se
moveu durante a calibração de bias) e
`deriva_estatica_20261021_pos_reboot.csv` (depois do ritual de reboot parado):

| arquivo / porta | \|a\| em repouso | giro médio (dps) | deriva de yaw |
|---|---|---|---|
| 20261021 / 4210 (direita) | **1,0676 g** | (+0,64, **−12,50**, +6,94) | −1,51°/min |
| 20261021 / 4211 (esquerda) | **1,0684 g** | (+0,33, −0,17, +0,12) | **−34,40°/min** |
| pos_reboot / 4210 | **1,0649 g** | (+0,00, −0,01, −0,00) | −0,20°/min |
| pos_reboot / 4211 | **1,0683 g** | (+0,00, −0,01, −0,00) | −0,04°/min |

Interpretação:

- O acelerômetro em repouso deveria medir **\|a\| = 1,000 g**; os valores de
  **1,065–1,068 g** mostram que **não há calibração de escala/offset** no
  firmware (as componentes em repouso também não são só gravidade pura:
  ~(-0,05, -0,03, +1,07) g na direita). Isso desloca o roll/pitch em alguns
  graus, de forma **dependente da orientação** — exatamente o que um único
  `T/f` de calibração não corrige.
- O giro da direita no primeiro arquivo tinha **−12,5 °/s** de bias: bias de
  boot ruim (luva mexida). O Mahony corrige roll/pitch com o acelerômetro, mas
  **yaw deriva** com esse bias; depois do reboot o giro fica ~0,00 °/s e o yaw
  deriva só −0,04 a −0,20 °/min.
- A esquerda no primeiro arquivo tinha yaw derivando **−34°/min** (bias
  residual), o que é catastrófico para o modelo da palma: o yaw passa a ser
  desconhecido em segundos. É exatamente o cenário em que a seta "não acompanha"
  mesmo com a calibração feita.

## 5. Comparação com o projeto ThatProject (Esp32_MPU6050_MahonyARHS)

O projeto do vídeo (`0015/ThatProject`, `MPU6050_Calibration_ESP32.ino` e
`MahonyAHRS_ESP32.ino`) usa **o mesmo filtro de Mahony com quaternion** que
nós, mas obtém orientação visivelmente mais estável. A diferença está na
preparação dos dados:

| item | ThatProject | nosso firmware (antes do item 124) |
|---|---|---|
| offsets do acelerômetro | **calibrados** (AX=-208, AY=417, AZ=93; subtraídos antes de normalizar) | **nenhum** (raw/16384; medimos \|a\|=1,065-1,068 g em repouso) |
| bias do giroscópio | **calibrado** e subtraído | só no boot, 500 amostras (~2 s) |
| faixa do giroscópio | **±2000 °/s** (FS_SEL=3) | ±250 °/s (nossos logs: picos de 230-247 °/s = saturando) |
| faixa do acelerômetro | ±8 g (ganho 4096) | ±2 g (16384; melhor resolução, satura em >2 g) |
| DLPF/ODR | DLPF 0 (260/256 Hz), loop ~4 ms | DLPF 0 (não configurado), loop ~10-15 ms |
| filtro | Mahony `twoKp=1.0`, `twoKi=0` | Mahony `KP=2.0`, sem integral |
| yaw | integrado (sem referência absoluta) | integrado (sem referência absoluta) |

O que foi **adotado no item 124** (compilado; ainda não gravado no ESP32):
1. **Calibração automática dos offsets do acelerômetro no boot**, junto com o
   bias do giroscópio, generalizando a ideia do ThatProject: com a luva parada
   e deitada (uma face do módulo paralela à mesa), o eixo que aponta para cima
   mede ±1 g e os outros ~0; o firmware detecta o eixo dominante sozinho,
   calcula os offsets de zero-g e os subtrai no loop. Se a luva não estiver
   deitada, avisa e aplica só o bias do giroscópio.
2. **Giroscópio em ±1000 °/s** (FS_SEL=2, macro `GYRO_FS_SEL`; ±500/±2000 sem
   reescrever a lógica), eliminando a saturação observada.
3. **DLPF ~42 Hz e ODR ~100 Hz** (`0x1A=0x03`, `0x19=0x09`).
4. Clock PLL do giroscópio Z (como o ThatProject), em vez do oscilador interno.

O que **não** foi adotado:
- acelerômetro ±8 g: nosso uso tem aceleração menor e ±2 g dá mais resolução
  (o gate `|a| ~ 1 g` do Mahony também filtra aceleração linear); se aparecer
  saturação >2 g, trocar é uma macro.
- loop de 4 ms: nosso I2C está a 50 kHz por causa do flat de 2 m (decisão de
  bancada documentada); subir para 400 kHz como no vídeo é risco de
  comunicação. O Mahony com KP=2 roda a ~100 Hz.
- `twoKi=0` do ThatProject é igual ao nosso; um Ki pequeno é opcional.

### 5.1 Créditos e referências

Este estudo comparativo (e a prática de calibração adotada no item 124) tem
como referência o trabalho público de **Eric – ThatProject**:

- Vídeo "MPU-6050 with Mahony algorithm": https://www.youtube.com/watch?v=UgROd5juFtM
- Repositório: https://github.com/0015/ThatProject (projeto `Esp32_MPU6050_MahonyARHS`)
- `MPU6050_Calibration_ESP32.ino`: https://github.com/0015/ThatProject/blob/master/Esp32_MPU6050_MahonyARHS/MPU6050_Calibration_ESP32/MPU6050_Calibration_ESP32.ino
- `MahonyAHRS_ESP32.ino`: https://github.com/0015/ThatProject/blob/master/Esp32_MPU6050_MahonyARHS/MahonyAHRS_ESP32/MahonyAHRS_ESP32.ino

O código dele, por sua vez, credita a origem/original: "Source:
https://blog.naver.com/ysahn2k/221410891895" (adaptação para ESP32) e as
referências MadgwickAHRS (x-io.co.uk), Jeff Rowberg (i2cdevlib), Kris Winer e
Davide Gironi (AVR MPU6050).

Como foi usado aqui: **referência pública de estudo e comparação**, não cópia
literal de código. O `AttitudeMahony` já era nosso (item 83) e a calibração de
acelerômetro foi generalizada (detecta o eixo dominante em vez de assumir Z
para cima); as constantes de offset do exemplo dele (-208, 417, 93) são do
hardware dele e não foram reutilizadas. Se algum trecho for reutilizado
literalmente no futuro, conferir a licença do repositório/origem antes.

Conclusão: **a diferença não é o algoritmo** (o Mahony é o mesmo e o nosso foi
validado por simulação) e sim **a calibração dos offsets do acelerômetro e a
faixa do giroscópio**. O item 124 fecha essas duas lacunas.

## 6. Recomendações (em ordem de impacto)

1. **Calibrar o acelerômetro** (6 posições: ±X, ±Y, ±Z contra a gravidade),
   obtendo offsets de zero-g (e, se quiser, fatores de escala), e aplicar antes
   de normalizar. É o item que o vídeo destaca e o que mais deve reduzir o erro
   de roll/pitch (de ~3-7° para <1°). Guardar as constantes em
   `credenciais.h` (não versionado) ou em header próprio não versionado.
2. **Configurar o DLPF e a taxa de amostragem** do MPU6050: escrever
   `0x1A = 0x03` (DLPF ~44 Hz no acelerômetro / 42 Hz no giroscópio) e
   `0x19 = 9` (~100 Hz) nos dois MPUs. Reduz aliasing/vibração.
3. **Reforçar o bias do giroscópio**: mais amostras, verificação de variância e
   aviso/retry se o desvio for alto (hoje o firmware não detecta "mexeu no
   boot"; o tool detecta via UDP e reseta, item 101).
4. Opcional: adicionar termo integral (Ki) no Mahony para estimar bias de
   x/y em tempo real; lembrando que yaw continua sem referência absoluta (o
   projeto reancora com o Kinect).
5. Se quiser comparar com o vídeo, implementar o Kalman 1D dele **apenas como
   referência/validação** de roll/pitch — não substitui o Mahony para 3D/yaw.

## 7. O que foi corrigido neste commit

- `calibrateGyroscopeAt` (dual) e `calibrateGyroscope` (luva única) agora
  dividem o bias pelo número de leituras **válidas** (antes dividiam por 500
  mesmo com falhas de I2C no meio, puxando o bias para zero) e avisam se menos
  de 90% das leituras foram válidas.
- Firmware dual: adicionado o termo `+ processNoise` que faltava na predição da
  covariância do `KalmanAngle` legado (consistente com o firmware da luva
  única). A classe continua **não usada** para atitude.
- Compilação verificada com PlatformIO: `esp32_duas_imus` (SUCCESS),
  `luva_direita` (SUCCESS), `luva_esquerda` (SUCCESS).
