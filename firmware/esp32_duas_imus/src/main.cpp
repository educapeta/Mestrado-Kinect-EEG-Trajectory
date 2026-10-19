#include <Wire.h>
#include <Arduino.h>
#include <math.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// Credenciais de rede (eduroam / WPA2-PSK). Arquivo esta' no .gitignore.
#include "credenciais.h"

// ==================== CONFIGURACOES DOS SENSORES ====================
// Dois MPU6050 no MESMO barramento I2C (Wire), enderecos configuraveis via
// build_flags do platformio.ini. O lado (direita/esquerda) e' determinado pelo
// endereco I2C, nao pela porta serial -- trocar fisicamente as luvas exige
// apenas recompilar com os build_flags corretos (veja README sec. 2).
#ifndef ADDR_DIREITO
#define ADDR_DIREITO 0x68   // MPU conectado a 0x68 publica na porta 4210
#endif
#ifndef ADDR_ESQUERDO
#define ADDR_ESQUERDO 0x69  // MPU conectado a 0x69 publica na porta 4211
#endif

// Pinos I2C. O firmware antigo e o documentado usavam 22/23, mas a bancada
// foi re-fiada com 21/22 (padrao dos tutoriais de ESP32). Em vez de
// adivinhar, setupI2c() SONDEIA os tres mapas possiveis no boot e imprime
// em qual deles algum MPU6050 responde (o probe pode reatribui-los).
int SDA_PIN = 22;
int SCL_PIN = 23;

// As credenciais de rede vivem em credenciais.h (nao versionado).
// ==================== IDENTIFICACAO DAS PORTAS UDP ====================
// Um ESP32, dois UDP fixos -- casa com imu.py: IMU_PORTAS_PADRAO = {1: 4210, 2: 4211}.
const uint16_t UDP_PORT_DIREITA = 4210;
const uint16_t UDP_PORT_ESQUERDA = 4211;
const char *GLOVE_NOME_DIREITA = "direita";
const char *GLOVE_NOME_ESQUERDO = "esquerda";
#if UDP_USAR_UNICAST
const char *UDP_DESTINO_STR_DIREITA = UDP_DESTINO;
const char *UDP_DESTINO_STR_ESQUERDO = UDP_DESTINO;
#define ENVIAR_UNICAST_DIREITA 1
#define ENVIAR_UNICAST_ESQUERDO 1
#else
const IPAddress UDP_BROADCAST(255, 255, 255, 255);
#define ENVIAR_UNICAST_DIREITA 0
#define ENVIAR_UNICAST_ESQUERDO 0
#endif
WiFiUDP udp_direita;
WiFiUDP udp_esquerda;

// ==================== FILTRO DE KALMAN (um por MPU) ==================
class KalmanAngle {
 public:
  float angle = 0.0f;
  float bias = 0.0f;
  float update(float measuredAngle, float gyroRate, float dt) {
    angle += dt * (gyroRate - bias);
    covariance[0][0] += dt * (dt * covariance[1][1] - covariance[0][1] - covariance[1][0] + covariance[1][0]);
    covariance[0][1] -= dt * covariance[1][1];
    covariance[1][0] -= dt * covariance[1][1];
    covariance[1][1] += biasNoise * dt;
    float innovation = measuredAngle - angle;
    float innovationCovariance = covariance[0][0] + measurementNoise;
    float gainAngle = covariance[0][0] / innovationCovariance;
    float gainBias = covariance[1][0] / innovationCovariance;
    angle += gainAngle * innovation;
    bias += gainBias * innovation;
    float p00 = covariance[0][0];
    float p01 = covariance[0][1];
    covariance[0][0] -= gainAngle * p00;
    covariance[0][1] -= gainAngle * p01;
    covariance[1][0] -= gainBias * p00;
    covariance[1][1] -= gainBias * p01;
    return angle;
  }
 private:
  float covariance[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  const float processNoise = 0.001f;
  const float biasNoise = 0.003f;
  const float measurementNoise = 0.03f;
};

// ==================== ATITUDE: quaternion + Mahony (item 83) ==========
// POR QUE ESTA CLASSE: a versao antiga fazia `yaw += gz*dt`, integrando o
// giroscopio no eixo Z do CORPO -- so' vale com o sensor NIVELADO. Com a mao
// inclinada (a bancada chega a pitch ~ -80 graus) a taxa de yaw verdadeira e'
//   yaw_dot = (gy*sin(roll) + gz*cos(roll)) / cos(pitch)
// (usa os 3 eixos; fica singular perto de pitch +-90). O erro acumulado
// depende do CAMINHO: medido em 19/10/2026, a MESMA pose fisica no comeco e no
// fim da sessao saia com yaw diferente por 61 graus, o modelo `v = T R f` nao
// fechava (residuo 38 graus) e a seta da palma errava ate' 160 graus.
// Aqui a rotacao e' integrada como QUATERNION no referencial do CORPO (sem
// singularidade) e a inclinacao e' corrigida pelo acelerometro (Mahony 6
// eixos). roll/pitch/yaw saem do quaternion na convencao ZYX que o PC ja' usa
// (imu.rotation_matrix = Rz(yaw) Ry(pitch) Rx(roll)).
class AttitudeMahony {
 public:
  bool iniciada = false;

  /** Arranca com a inclinacao do acelerometro (sem transiente de ~1 s). */
  void iniciar(float ax, float ay, float az) {
    float norma = sqrtf(ax * ax + ay * ay + az * az);
    if (norma > 0.1f) {
      ax /= norma;
      ay /= norma;
      az /= norma;
    } else {
      ax = 0.0f;
      ay = 0.0f;
      az = 1.0f;
    }
    // MESMAS formulas do firmware antigo para o roll/pitch iniciais:
    float roll = atan2f(ay, az);
    float pitch = atan2f(-ax, sqrtf(ay * ay + az * az));
    float cr = cosf(roll * 0.5f), sr = sinf(roll * 0.5f);
    float cp = cosf(pitch * 0.5f), sp = sinf(pitch * 0.5f);
    q[0] = cr * cp;    // w
    q[1] = sr * cp;    // x
    q[2] = cr * sp;    // y
    q[3] = -sr * sp;   // z
    iniciada = true;
  }

  void update(float gx_dps, float gy_dps, float gz_dps, float ax, float ay,
              float az, float dt) {
    if (!(dt > 0.0f) || dt > 0.1f) return;
    if (!iniciada) iniciar(ax, ay, az);
    // taxa angular do CORPO em rad/s
    float wx = gx_dps * (PI / 180.0f);
    float wy = gy_dps * (PI / 180.0f);
    float wz = gz_dps * (PI / 180.0f);
    // correcao de inclinacao: so' quando |a| ~ 1 g (sem aceleracao linear)
    float norma = sqrtf(ax * ax + ay * ay + az * az);
    if (norma > 0.5f && norma < 1.5f) {
      ax /= norma;
      ay /= norma;
      az /= norma;
      float ux, uy, uz;
      cima_prevista(&ux, &uy, &uz);
      const float KP = 2.0f;  // 1/s: fecha o tilt em ~1 s
      wx += KP * (ay * uz - az * uy);
      wy += KP * (az * ux - ax * uz);
      wz += KP * (ax * uy - ay * ux);
    }
    // q <- q + 0.5 * q (x) (0, omega) * dt   (omega no referencial do corpo)
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    float metade = 0.5f * dt;
    q[0] += metade * (-qx * wx - qy * wy - qz * wz);
    q[1] += metade * (qw * wx + qy * wz - qz * wy);
    q[2] += metade * (qw * wy + qz * wx - qx * wz);
    q[3] += metade * (qw * wz + qx * wy - qy * wx);
    float norma_q =
        sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (norma_q > 0.0f) {
      q[0] /= norma_q;
      q[1] /= norma_q;
      q[2] /= norma_q;
      q[3] /= norma_q;
    }
  }

  /** roll/pitch/yaw (graus) na convencao ZYX: R = Rz(yaw) Ry(pitch) Rx(roll). */
  void to_euler_zyx_deg(float &roll, float &pitch, float &yaw) const {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    float r00 = 1.0f - 2.0f * (y * y + z * z);
    float r10 = 2.0f * (x * y + w * z);
    float r20 = 2.0f * (x * z - w * y);
    float r21 = 2.0f * (y * z + w * x);
    float r22 = 1.0f - 2.0f * (x * x + y * y);
    float seno_pitch = -r20;
    if (seno_pitch > 1.0f) seno_pitch = 1.0f;
    if (seno_pitch < -1.0f) seno_pitch = -1.0f;
    pitch = asinf(seno_pitch) * (180.0f / PI);
    roll = atan2f(r21, r22) * (180.0f / PI);
    yaw = atan2f(r10, r00) * (180.0f / PI);
  }

 private:
  float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};

  /** R^T * (0,0,1): direcao "para cima" do mundo expressa NO CORPO. */
  void cima_prevista(float *ux, float *uy, float *uz) const {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    *ux = 2.0f * (x * z - w * y);
    *uy = 2.0f * (y * z + w * x);
    *uz = 1.0f - 2.0f * (x * x + y * y);
  }
};
// ==================== VARIAVEIS GLOBAIS ==============================
// Bias e filtro SEPARADOS para cada MPU (cada sensor tem o seu relogio e o seu
// viés; misturar contaminaria o outro lado).
// Os filtros de Kalman de angulo NAO sao mais usados para a atitude: quem
// integra agora e' o AttitudeMahony (acima), que tambem da' o yaw (o Kalman
// antigo nao conseguia: ver item 83). A classe ficou como referencia/historico.
float gyroBiasDx = 0.0f, gyroBiasDy = 0.0f, gyroBiasDz = 0.0f;
float gyroBiasEx = 0.0f, gyroBiasEy = 0.0f, gyroBiasEz = 0.0f;
uint32_t previousMicrosD = 0;
uint32_t previousMicrosE = 0;
// Quando o loop tentou reconectar o Wi-Fi pela ultima vez (backoff de 15 s:
// sem hotspot a reconexao de 20 s em CADA volta travava o loop inteiro).
uint32_t ultimaTentativaWifi = 0;

// ==================== FUNCOES AUXILIARES =============================
int16_t readInt16() { return (int16_t)((Wire.read() << 8) | Wire.read()); }

// Le o MPU no endereco 'addr' (0x68 ou 0x69). O resto e' identico ao firmware
// original de uma luva so.
bool readMpuAt(uint8_t addr, int16_t &accelX, int16_t &accelY, int16_t &accelZ,
               int16_t &gyroX, int16_t &gyroY, int16_t &gyroZ) {
  Wire.beginTransmission(addr);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom((uint8_t)addr, (size_t)14, true) != 14) {
    return false;
  }
  accelX = readInt16();
  accelY = readInt16();
  accelZ = readInt16();
  readInt16(); // Temperatura
  gyroX = readInt16();
  gyroY = readInt16();
  gyroZ = readInt16();
  return true;
}

// Calibra o giroscopio do MPU em 'addr' (500 amostras). Chamada DUAS vezes no
// setup (uma por MPU), com 4 ms de delay entre cada amostra.
void calibrateGyroscopeAt(uint8_t addr, float &biasX, float &biasY, float &biasZ) {
  const int samples = 500;
  long sumX = 0, sumY = 0, sumZ = 0;
  int16_t ax, ay, az, gx, gy, gz;
  Serial.printf("[imu] calibrando giroscopio MPU 0x%02X... mantenha parado\n", addr);
  for (int i = 0; i < samples; i++) {
    if (readMpuAt(addr, ax, ay, az, gx, gy, gz)) {
      sumX += gx; sumY += gy; sumZ += gz;
    }
    delay(4);
  }
  biasX = (float)sumX / samples / 131.0f;
  biasY = (float)sumY / samples / 131.0f;
  biasZ = (float)sumZ / samples / 131.0f;
  Serial.printf("[imu] bias MPU 0x%02X: (%.4f, %.4f, %.4f)\n", addr, biasX, biasY, biasZ);
}

// ==================== FUNCOES DE REDE ================================
// Reutilizadas do firmware esp32_luvas (1 luva = 1 ESP32). Aqui o Wi-Fi serve
// para publicar DOIS sockets UDP (4210 e 4211), nao um.
// ---------------------------------------------------------------------
bool conectarPsk(uint32_t timeoutMs) {
  WiFi.disconnect(true);
  delay(100);
  WiFi.mode(WIFI_STA);
  delay(100);
  WiFi.begin(WIFI_SSID_PSK, WIFI_SENHA_PSK);
  Serial.printf("Conectando ao Wi-Fi '%s'", WIFI_SSID_PSK);
  uint32_t inicio = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - inicio < timeoutMs) {
    delay(500);
    Serial.print(".");
  }
  return WiFi.status() == WL_CONNECTED;
}

void anunciarConexao(const char *rede, WiFiUDP &udp, uint16_t porta,
                     const char *nome) {
  udp.begin(porta);
  Serial.printf("\n[wifi] conectado em '%s' | IP: %s\n", rede,
                WiFi.localIP().toString().c_str());
  Serial.printf("%s -> enviando UDP unicast porta %u para %s\n", nome, porta,
                UDP_DESTINO);
}

void connectWiFi(uint32_t timeoutMs) {
  if (WiFi.status() == WL_CONNECTED) return;
#if WIFI_PSK_FALLBACK
  if (conectarPsk(timeoutMs)) {
    anunciarConexao(WIFI_SSID_PSK, udp_direita, UDP_PORT_DIREITA, GLOVE_NOME_DIREITA);
    anunciarConexao(WIFI_SSID_PSK, udp_esquerda, UDP_PORT_ESQUERDA, GLOVE_NOME_ESQUERDO);
    return;
  }
#endif
  Serial.printf("\n[wifi] FALHA ao conectar (status %d).\n", (int)WiFi.status());
}

void writeRegisterAt(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission(true);
}

// ==================== SETUP ====================
// --- Sondeo dos pinos I2C --------------------------------------------------
// Varre PARES de GPIO candidatos ate' encontrar um em que ALGUM dos dois
// enderecos (0x68/0x69) responda (ACK de verdade: nesta versao do Wire o
// endTransmission(false) e' um no-op; a transacao inteira acontece no
// requestFrom, e o Error -1 do IDF = NACK = endereco NAO respondeu).
// Cobre a fiação "de memoria" da bancada (22/23 do firmware antigo, 21/22 do
// tutorial, 22/21 trocada) e QUALQUER outra combinatoria entre GPIOs seguros
// (nada de 0-2 de boot, 6-11 da flash, 1/3 da UART): o log imprime o par achado.
static const int PinosI2cCandidatos[] = {4,  5,  13, 14, 15, 16, 17, 18, 19,
                                         21, 22, 23, 25, 26, 27, 32, 33};

bool algumMpuResponde(uint8_t &endereco) {
  Wire.beginTransmission(ADDR_DIREITO);
  if (Wire.endTransmission() == 0) { endereco = ADDR_DIREITO; return true; }
  Wire.beginTransmission(ADDR_ESQUERDO);
  if (Wire.endTransmission() == 0) { endereco = ADDR_ESQUERDO; return true; }
  return false;
}

void setupI2c() {
  const size_t n = sizeof(PinosI2cCandidatos) / sizeof(PinosI2cCandidatos[0]);
  uint8_t endereco = 0;
  bool jaBegin = false;
  for (size_t a = 0; a < n && endereco == 0; a++) {
    for (size_t b = a + 1; b < n && endereco == 0; b++) {
      // Testa as DUAS orientacoes de cada par. A bancada fixa SCL em D21 e SDA
      // em D22 (INVERTIDO do padrao do ESP32, onde SDA=21 e SCL=22); um sondeo
      // que so' testasse (sda,scl) com b>a nunca enxergava esse barramento e
      // imprimia "NENHUM MPU6050 respondeu em NENHUM par" mesmo com modulo bom
      // e cabo curto -- medido na bancada em 13/10/2026.
      const int orientacoes[2][2] = {{PinosI2cCandidatos[a], PinosI2cCandidatos[b]},
                                     {PinosI2cCandidatos[b], PinosI2cCandidatos[a]}};
      for (int o = 0; o < 2 && endereco == 0; o++) {
        const int sda = orientacoes[o][0];
        const int scl = orientacoes[o][1];
        if (jaBegin) Wire.end();
        Wire.begin(sda, scl);
        // Timeout CURTO: sem isto cada transacao de um barramento mudo espera o
        // padrao de 1000 ms do driver, e 306 combinacoes x 2 transacoes viram
        // ~10 MINUTOS de silencio antes da primeira linha do log (medido na
        // bancada em 13/10/2026 com o flat de 2 m).
        Wire.setTimeOut(15);
        // 50 kHz: o barramento da bancada usa flat de 2 m (~300-400 pF com os
        // dois modulos). Com pull-ups de 2,2 k o tempo de subida fica ~0,7 us;
        // a 100 kHz ainda era intermitente (bias zerado, pinos trocando entre
        // boots -- medido em 13/10/2026), entao baixamos para 50 kHz, que dobra
        // a margem de tempo. A taxa necessaria e' minuscula (2 IMUs x 100 Hz).
        Wire.setClock(50000);
        jaBegin = true;
        if (algumMpuResponde(endereco)) {
          SDA_PIN = sda;
          SCL_PIN = scl;
          Serial.printf("[i2c] barramento vivo em SDA=%d SCL=%d (endereco 0x%02X)%s\n",
                        sda, scl, endereco,
                        (sda == 22 && scl == 21) ? "  [bancada: SCL=D21, SDA=D22]" : "");
          return;
        }
      }
    }
  }
  Serial.println("[i2c] NENHUM MPU6050 respondeu em NENHUM par de GPIO candidato "
                 "(4,5,13..33): confira se os fios SDA/SCL chegam aos pinos "
                 "CERTOS do ESP32, se o fio esta' no pino SDA do modulo (e nao "
                 "no XDA) e se os DOIS LEDs do modulo estao' acesos. "
                 "Seguindo com SDA=22 SCL=23.");
  Wire.end();
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setTimeOut(15);
  Wire.setClock(50000);
}

void setup() {
  Serial.begin(115200);
  setupI2c();

  // Configura os DOIS MPUs (acorda, giro +/-250, accel +/-2g).
  writeRegisterAt(ADDR_DIREITO, 0x6B, 0x00);
  writeRegisterAt(ADDR_DIREITO, 0x1B, 0x00);
  writeRegisterAt(ADDR_DIREITO, 0x1C, 0x00);
  writeRegisterAt(ADDR_ESQUERDO, 0x6B, 0x00);
  writeRegisterAt(ADDR_ESQUERDO, 0x1B, 0x00);
  writeRegisterAt(ADDR_ESQUERDO, 0x1C, 0x00);

  // Primeira tentativa com tempo cheio (o hotspot da bancada pode demorar a
  // subir); depois o loop cuida da reconexao com backoff.
  connectWiFi(20000);
  ultimaTentativaWifi = millis();
  calibrateGyroscopeAt(ADDR_DIREITO, gyroBiasDx, gyroBiasDy, gyroBiasDz);
  calibrateGyroscopeAt(ADDR_ESQUERDO, gyroBiasEx, gyroBiasEy, gyroBiasEz);
  previousMicrosD = micros();
  previousMicrosE = micros();

  Serial.println("esp_timestamp_us,roll_deg,pitch_deg,yaw_deg,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
}

// ==================== LOOP PRINCIPAL ====================
void loop() {
  // Reconexao com BACKOFF: antes CADA volta bloqueava 20 s (sem hotspot a
  // bancada so' recebia dados em rajadas de ~1 s a cada 20 s). Agora: 6 s de
  // tentativa a cada 15 s -- o resto do tempo o loop le os IMUs tranquilo.
  if (WiFi.status() != WL_CONNECTED &&
      millis() - ultimaTentativaWifi >= 15000) {
    ultimaTentativaWifi = millis();
    connectWiFi(6000);
  }

  // --- DIREITO: MPU 0x68 -> UDP 4210 ---
  // Sem 'return' no erro: o 'return' antigo abortava a volta do loop e o MPU
  // ESQUERDO nem era lido -- um lado morto ESCONDIA o outro (medido ao vivo
  // em 08/10/2026: mascaramento de 100%). Cada lado pula so' o SEU corpo.
  {
    int16_t rawAx, rawAy, rawAz, rawGx, rawGy, rawGz;
    if (!readMpuAt(ADDR_DIREITO, rawAx, rawAy, rawAz, rawGx, rawGy, rawGz)) {
      Serial.println("Erro de leitura I2C D");
      delay(100);
    } else {
      uint32_t now = micros();
      float dt = (now - previousMicrosD) * 1.0e-6f;
      previousMicrosD = now;
      if (dt > 0.0f && dt <= 0.1f) {
        float ax = rawAx / 16384.0f, ay = rawAy / 16384.0f, az = rawAz / 16384.0f;
        float gx = rawGx / 131.0f - gyroBiasDx, gy = rawGy / 131.0f - gyroBiasDy, gz = rawGz / 131.0f - gyroBiasDz;
        // ATITUDE propria (quaternion + Mahony 6 eixos): substitui
        // `yaw += gz*dt` e o Kalman de roll/pitch -- ver item 83 e a classe.
        static AttitudeMahony atitudeD;
        atitudeD.update(gx, gy, gz, ax, ay, az, dt);
        float roll, pitch, yawD;
        atitudeD.to_euler_zyx_deg(roll, pitch, yawD);
        uint32_t sampleTimestampUs = micros();
        char packet[180];
        snprintf(packet, sizeof(packet), "%lu,%.2f,%.2f,%.2f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n",
                 (unsigned long)sampleTimestampUs, roll, pitch, yawD, ax, ay, az, gx, gy, gz);
        // Prefixo do lado SO' na serial (contrato do tools/imu_serial_bridge):
        // `1,<10 campos>`. O UDP continua com 10 campos (imu.py nao muda).
        Serial.print("1,");
        Serial.print(packet);
#if ENVIAR_UNICAST
        udp_direita.beginPacket(UDP_DESTINO_STR_DIREITA, UDP_PORT_DIREITA);
#else
        udp_direita.beginPacket(UDP_DESTINO_STR_DIREITA, UDP_PORT_DIREITA);
#endif
        udp_direita.write((const uint8_t *)packet, strlen(packet));
        udp_direita.endPacket();
      }
    }
  }

  // --- ESQUERDO: MPU 0x69 -> UDP 4211 ---
  {
    int16_t rawAx, rawAy, rawAz, rawGx, rawGy, rawGz;
    if (!readMpuAt(ADDR_ESQUERDO, rawAx, rawAy, rawAz, rawGx, rawGy, rawGz)) {
      Serial.println("Erro de leitura I2C E");
      delay(100);
    } else {
      uint32_t now = micros();
      float dt = (now - previousMicrosE) * 1.0e-6f;
      previousMicrosE = now;
      if (dt > 0.0f && dt <= 0.1f) {
        float ax = rawAx / 16384.0f, ay = rawAy / 16384.0f, az = rawAz / 16384.0f;
        float gx = rawGx / 131.0f - gyroBiasEx, gy = rawGy / 131.0f - gyroBiasEy, gz = rawGz / 131.0f - gyroBiasEz;
        // ATITUDE propria (quaternion + Mahony 6 eixos) -- mesmo filtro do
        // lado direito, com o SEU proprio estado (item 83).
        static AttitudeMahony atitudeE;
        atitudeE.update(gx, gy, gz, ax, ay, az, dt);
        float roll, pitch, yawE;
        atitudeE.to_euler_zyx_deg(roll, pitch, yawE);
        uint32_t sampleTimestampUs = micros();
        char packet[180];
        snprintf(packet, sizeof(packet), "%lu,%.2f,%.2f,%.2f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n",
                 (unsigned long)sampleTimestampUs, roll, pitch, yawE, ax, ay, az, gx, gy, gz);
        // Prefixo do lado SO' na serial (contrato do tools/imu_serial_bridge).
        Serial.print("2,");
        Serial.print(packet);
#if ENVIAR_UNICAST
        udp_esquerda.beginPacket(UDP_DESTINO_STR_ESQUERDO, UDP_PORT_ESQUERDA);
#else
        udp_esquerda.beginPacket(UDP_DESTINO_STR_ESQUERDO, UDP_PORT_ESQUERDA);
#endif
        udp_esquerda.write((const uint8_t *)packet, strlen(packet));
        udp_esquerda.endPacket();
      }
    }
  }

  delay(5);
}

