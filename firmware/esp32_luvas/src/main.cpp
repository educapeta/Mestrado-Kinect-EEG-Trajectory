#include <Wire.h>
#include <Arduino.h>
#include <math.h>
#include <WiFi.h>
#include <WiFiUdp.h>

// Credenciais de rede (eduroam / WPA2-PSK). Arquivo esta' no .gitignore.
#include "credenciais.h"

// ==================== CONFIGURAÇÕES DO SENSOR ====================
const uint8_t MPU_ADDR = 0x68;
const int SDA_PIN = 22;
const int SCL_PIN = 23;
// As credenciais de rede vivem em credenciais.h (nao versionado).
// ==================== IDENTIFICACAO DA LUVA ====================
// Definido pelos ambientes do PlatformIO (platformio.ini):
//   luva_direita  -> GLOVE_SIDE = 1 -> porta UDP 4210
//   luva_esquerda -> GLOVE_SIDE = 0 -> porta UDP 4211
// IMPORTANTE: a convencao tem de casar com o projeto do Mestrado
// (imu.py: IMU_PORTAS_PADRAO = {1: 4210, 2: 4211}, com lado 1 = DIREITA),
// senao os lados entram TROCADOS nos blocos IMU_L_*/IMU_R_* do CSV de movimento.
#ifndef GLOVE_SIDE
#define GLOVE_SIDE 1
#endif

const uint16_t UDP_PORT = (GLOVE_SIDE == 1) ? 4210 : 4211;
const char *GLOVE_NAME = (GLOVE_SIDE == 1) ? "direita" : "esquerda";
#if UDP_USAR_UNICAST
// Unicast direto ao notebook (redes com client isolation, ex.: eduroam).
const char *UDP_DESTINO_STR = UDP_DESTINO;
#define ENVIAR_UNICAST 1
#else
const IPAddress UDP_BROADCAST(255, 255, 255, 255);
#define ENVIAR_UNICAST 0
#endif
WiFiUDP udp;

// ==================== FILTRO DE KALMAN PARA ÂNGULOS ====================
class KalmanAngle {
public:
  float angle = 0.0f;
  float bias = 0.0f;

  float update(float measuredAngle, float gyroRate, float dt) {
    // Predição
    angle += dt * (gyroRate - bias);
    covariance[0][0] += dt * (dt * covariance[1][1] - covariance[0][1] - covariance[1][0] + processNoise);
    covariance[0][1] -= dt * covariance[1][1];
    covariance[1][0] -= dt * covariance[1][1];
    covariance[1][1] += biasNoise * dt;

    // Atualização
    float innovation = measuredAngle - angle;
    float innovationCovariance = covariance[0][0] + measurementNoise;
    float gainAngle = covariance[0][0] / innovationCovariance;
    float gainBias = covariance[1][0] / innovationCovariance;
    angle += gainAngle * innovation;
    bias += gainBias * innovation;

    // Atualização da covariância
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

KalmanAngle rollFilter;
KalmanAngle pitchFilter;

// ==================== VARIÁVEIS GLOBAIS ====================
float gyroBiasX = 0.0f;
float gyroBiasY = 0.0f;
float gyroBiasZ = 0.0f;

uint32_t previousMicros = 0;

// ==================== FUNÇÕES AUXILIARES ====================
int16_t readInt16() {
  return (int16_t)((Wire.read() << 8) | Wire.read());
}

bool readMpu(int16_t &accelX, int16_t &accelY, int16_t &accelZ,
             int16_t &gyroX, int16_t &gyroY, int16_t &gyroZ) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom((uint8_t)MPU_ADDR, (size_t)14, true) != 14) {
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

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission(true);
}

void calibrateGyroscope() {
  const int samples = 500;
  long sumX = 0, sumY = 0, sumZ = 0;
  int16_t accelX, accelY, accelZ, gyroX, gyroY, gyroZ;
  Serial.println("Calibrando giroscopio. Mantenha o sensor parado...");
  for (int i = 0; i < samples; i++) {
    if (readMpu(accelX, accelY, accelZ, gyroX, gyroY, gyroZ)) {
      sumX += gyroX;
      sumY += gyroY;
      sumZ += gyroZ;
    }
    delay(4);
  }
  gyroBiasX = (float)sumX / samples / 131.0f;
  gyroBiasY = (float)sumY / samples / 131.0f;
  gyroBiasZ = (float)sumZ / samples / 131.0f;
}

// ==================== CONEXAO Wi-Fi ====================
//: Indice da senha da eduroam que JA' autenticou (-1 = ainda nao sabemos qual).
//: Guardar isso evita repetir tentativas erradas no RADIUS a cada reconexao.
int senhaOk = -1;
//: Todas as candidatas falharam neste boot: nao insiste mais na eduroam.
bool enterpriseEsgotada = false;

#if WIFI_ENTERPRISE
//: Autentica na eduroam (WPA2-Enterprise, EAP-TTLS + PAP).
//: O ESP32 NAO conecta em eduroam com `WiFi.begin(ssid, senha)`: precisa desta
//: sobrecarga de Enterprise com identidade/usuario (WiFiSTA.h -> WPA2_AUTH_TTLS).
bool tentarEnterprise(const char *senha) {
  WiFi.disconnect(true);
  delay(100);
  WiFi.mode(WIFI_STA);
  delay(100);
  WiFi.begin(WIFI_SSID_ENTERPRISE, WPA2_AUTH_TTLS,
             WIFI_IDENTIDADE, WIFI_USUARIO, senha);

  uint32_t inicio = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - inicio < WIFI_TIMEOUT_MS) {
    delay(500);
    Serial.print(".");
  }
  return WiFi.status() == WL_CONNECTED;
}
#endif

//: Conecta numa rede WPA2-PSK comum (fallback: casa, hotspot do PC/celular).
bool conectarPsk() {
  WiFi.disconnect(true);
  delay(100);
  WiFi.mode(WIFI_STA);
  delay(100);
  WiFi.begin(WIFI_SSID_PSK, WIFI_SENHA_PSK);
  Serial.printf("Conectando ao Wi-Fi '%s'", WIFI_SSID_PSK);

  uint32_t inicio = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - inicio < 20000) {
    delay(500);
    Serial.print(".");
  }
  return WiFi.status() == WL_CONNECTED;
}

//: Publica a identidade da luva e a porta UDP depois de conectar.
void anunciarConexao(const char *rede) {
  udp.begin(UDP_PORT);
  Serial.printf("\n[wifi] conectado em '%s' | IP: %s\n", rede,
                WiFi.localIP().toString().c_str());
  Serial.printf("Luva %s -> enviando UDP broadcast na porta %u\n", GLOVE_NAME,
                UDP_PORT);
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

#if WIFI_ENTERPRISE
  if (!enterpriseEsgotada) {
    if (senhaOk >= 0) {
      // Ja' sabemos qual senha funciona: reconecta direto com ela.
      Serial.printf("\n[wifi] reconectando na eduroam (senha #%d)", senhaOk + 1);
      if (tentarEnterprise(WIFI_SENHAS[senhaOk])) {
        anunciarConexao(WIFI_SSID_ENTERPRISE);
        return;
      }
    } else {
      const int disponiveis = (int)(sizeof(WIFI_SENHAS) / sizeof(WIFI_SENHAS[0]));
      const int total = (WIFI_MAX_TENTATIVAS < disponiveis) ? WIFI_MAX_TENTATIVAS
                                                            : disponiveis;
      for (int i = 0; i < total; i++) {
        Serial.printf("\n[wifi] %s | eduroam EAP-TTLS/PAP | usuario=%s | "
                      "tentativa %d/%d",
                      GLOVE_NAME, WIFI_USUARIO, i + 1, total);
        if (tentarEnterprise(WIFI_SENHAS[i])) {
          senhaOk = i;
          Serial.printf("\n[wifi] AUTENTICOU com a senha #%d", i + 1);
          anunciarConexao(WIFI_SSID_ENTERPRISE);
          return;
        }
        Serial.printf("\n[wifi] senha #%d NAO autenticou (status %d)", i + 1,
                      (int)WiFi.status());
        delay(1500);
      }
      enterpriseEsgotada = true;
      Serial.println("\n[wifi] nenhuma senha da eduroam autenticou.");
    }
  }
#endif

#if WIFI_PSK_FALLBACK
  Serial.println("\n[wifi] tentando o fallback WPA2-PSK...");
  if (conectarPsk()) {
    anunciarConexao(WIFI_SSID_PSK);
    return;
  }
#endif

  Serial.printf("\n[wifi] FALHA ao conectar (status %d).\n", (int)WiFi.status());
  Serial.println("[wifi] confira SSID/senha; o ESP32 so' usa Wi-Fi 2.4 GHz.");
}

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  writeRegister(0x6B, 0x00); // Wake up
  writeRegister(0x1B, 0x00); // Gyro: +/-250 deg/s
  writeRegister(0x1C, 0x00); // Accel: +/-2g

  connectWiFi();
  calibrateGyroscope();
  previousMicros = micros();

  Serial.println("esp_timestamp_us,roll_deg,pitch_deg,yaw_deg,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
}

// ==================== LOOP PRINCIPAL ====================
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  int16_t rawAx, rawAy, rawAz, rawGx, rawGy, rawGz;
  if (!readMpu(rawAx, rawAy, rawAz, rawGx, rawGy, rawGz)) {
    Serial.println("Erro de leitura I2C");
    delay(100);
    return;
  }

  uint32_t now = micros();
  float dt = (now - previousMicros) * 1.0e-6f;
  previousMicros = now;
  if (dt <= 0.0f || dt > 0.1f) return;

  // --- CONVERSÃO PARA UNIDADES FÍSICAS ---
  float ax = rawAx / 16384.0f;
  float ay = rawAy / 16384.0f;
  float az = rawAz / 16384.0f;
  float gx = rawGx / 131.0f - gyroBiasX;
  float gy = rawGy / 131.0f - gyroBiasY;
  float gz = rawGz / 131.0f - gyroBiasZ;

  // Roll e pitch combinam acelerometro (referencia da gravidade) e giroscopio.
  float accelRoll = atan2f(ay, az) * 180.0f / PI;
  float accelPitch = atan2f(-ax, sqrtf(ay * ay + az * az)) * 180.0f / PI;
  float roll = rollFilter.update(accelRoll, gx, dt);
  float pitch = pitchFilter.update(accelPitch, gy, dt);

  // Yaw usa somente o giroscopio e, por isso, acumula drift com o tempo.
  static float yaw = 0.0f;
  yaw += gz * dt;

  // Timestamp captured immediately after the sensor read and before UDP send.
  uint32_t sampleTimestampUs = micros();
  char packet[180];
  snprintf(packet, sizeof(packet), "%lu,%.2f,%.2f,%.2f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n",
           (unsigned long)sampleTimestampUs, roll, pitch, yaw,
           ax, ay, az, gx, gy, gz);
  Serial.print(packet);

  if (WiFi.status() == WL_CONNECTED) {
#if ENVIAR_UNICAST
    // Unicast direto ao notebook (redes com client isolation, ex.: eduroam).
    udp.beginPacket(UDP_DESTINO_STR, UDP_PORT);
#else
    udp.beginPacket(UDP_BROADCAST, UDP_PORT);
#endif
    udp.write((const uint8_t *)packet, strlen(packet));
    udp.endPacket();
  }

  delay(5);
}