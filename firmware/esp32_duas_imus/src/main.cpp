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

// ==================== VARIAVEIS GLOBAIS ==============================
// Bias e filtro SEPARADOS para cada MPU (cada sensor tem o seu relogio e o seu
// viés; misturar contaminaria o outro lado).
KalmanAngle rollFilterD, pitchFilterD;   // MPU direito (0x68)
KalmanAngle rollFilterE, pitchFilterE;   // MPU esquerdo (0x69)
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
        float accelRoll = atan2f(ay, az) * 180.0f / PI;
        float accelPitch = atan2f(-ax, sqrtf(ay * ay + az * az)) * 180.0f / PI;
        float roll = rollFilterD.update(accelRoll, gx, dt);
        float pitch = pitchFilterD.update(accelPitch, gy, dt);
        static float yawD = 0.0f;
        yawD += gz * dt;
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
        float accelRoll = atan2f(ay, az) * 180.0f / PI;
        float accelPitch = atan2f(-ax, sqrtf(ay * ay + az * az)) * 180.0f / PI;
        float roll = rollFilterE.update(accelRoll, gx, dt);
        float pitch = pitchFilterE.update(accelPitch, gy, dt);
        static float yawE = 0.0f;
        yawE += gz * dt;
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

