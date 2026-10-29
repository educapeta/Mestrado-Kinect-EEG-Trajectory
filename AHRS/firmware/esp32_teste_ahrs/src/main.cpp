// ============================================================================
// TESTE AHRS DOS DOIS MPU6050 -- VETORES 3D EM TEMPO REAL (item 142)
//
// Base: 0015/ThatProject - Esp32_MPU6050_MahonyARHS/MahonyAHRS_ESP32
//       (MahonyAHRS, leitura MPU6050 e calibracao de bias adaptados).
// Diferenca: em vez do "dado" virtual, publica os QUATERNIONS corpo->mundo
// dos DOIS MPUs (0x68 direita / 0x69 esquerda); o tool
// tools/visualiza_vetores_ahrs.py desenha UM VETOR 3D POR SENSOR.
//
// Cada vetor e' fixo no corpo do sensor e e' "zerado" no primeiro quadro: com
// a face paralela a mesa ele comeca apontado em -Z (chao) e gira junto com o
// sensor. Formato serial (115200 baud):
//   # AHRS_VETOR v1 ...
//   A,<t_us>,<qwL>,<qxL>,<qyL>,<qzL>,<qwR>,<qxR>,<qyR>,<qzR>,<axL>,...
// Comandos: 'c' recalibra (IMUs PARADAS e deitadas), 'r' reinicia a atitude.
// ============================================================================
#include <Arduino.h>
#include <Wire.h>
#include <math.h>

#ifndef ADDR_DIREITO
#define ADDR_DIREITO 0x68
#endif
#ifndef ADDR_ESQUERDO
#define ADDR_ESQUERDO 0x69
#endif
#define GYRO_FS_SEL 0x10
#define GYRO_LSB_POR_DPS 32.8f
#define MPU_DLPF_CFG 0x03
#define MPU_SMPLRT_DIV 0x09
#define ACCEL_LSB_POR_G 16384.0f

//: Pinos usados se o sondeo nao achar (bancada: SCL=D21, SDA=D22).
int SDA_PIN = 22;
int SCL_PIN = 23;

// ============================ I2C / MPU ====================================
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

// Calibra giroscopio E acelerometro do MPU em 'addr' (item 124).
//
// O giroscopio e' a media das amostras PARADAS (bias). O acelerometro usa o
// metodo do ThatProject generalizado: com a luva DEITADA na mesa (qualquer
// face reta para cima), o eixo que estiver mais alinhado com a gravidade mede
// +-1 g e os outros medem ~0; assim o offset de zero-g de cada eixo sai da
// propria media, SEM precisar saber qual eixo e' o "z" do sensor:
//   - eixo dominante (|media| > 0.95*|vetor|): offset = media - signo*1g
//   - eixos horizontais: offset = media (verdade ~0)
// Se a luva nao estiver deitada/reta (vetor longe de um eixo), avisa e NAO
// aplica offsets de acelerometro (so' o bias do giroscopio).
void calibrateImuAt(uint8_t addr, float &biasX, float &biasY, float &biasZ,
                    float &accelOffX, float &accelOffY, float &accelOffZ) {
  const int samples = 1000;
  long sumGx = 0, sumGy = 0, sumGz = 0;
  long sumAx = 0, sumAy = 0, sumAz = 0;
  int validas = 0;
  int16_t ax, ay, az, gx, gy, gz;
  Serial.printf("[imu] calibrando giroscopio+acelerometro MPU 0x%02X"
                "... mantenha a luva PARADA e DEITADA na mesa\n", addr);
  for (int i = 0; i < samples; i++) {
    if (readMpuAt(addr, ax, ay, az, gx, gy, gz)) {
      sumGx += gx; sumGy += gy; sumGz += gz;
      sumAx += ax; sumAy += ay; sumAz += az;
      validas++;
    }
    delay(2);
  }
  const float divisor = (validas > 0) ? (float)validas : 1.0f;
  biasX = (float)sumGx / divisor / GYRO_LSB_POR_DPS;
  biasY = (float)sumGy / divisor / GYRO_LSB_POR_DPS;
  biasZ = (float)sumGz / divisor / GYRO_LSB_POR_DPS;
  if (validas < samples * 9 / 10) {
    Serial.printf("[imu] AVISO: so' %d/%d leituras validas no MPU 0x%02X\n",
                  validas, samples, addr);
  }
  //: offsets de acelerometro em LSB (unidades RAW do sensor)
  accelOffX = (float)sumAx / divisor;
  accelOffY = (float)sumAy / divisor;
  accelOffZ = (float)sumAz / divisor;
  float vetor[3] = {accelOffX, accelOffY, accelOffZ};
  float norma = sqrtf(vetor[0]*vetor[0] + vetor[1]*vetor[1] + vetor[2]*vetor[2]);
  int eixo = 0;
  for (int i = 1; i < 3; i++) {
    if (fabsf(vetor[i]) > fabsf(vetor[eixo])) eixo = i;
  }
  bool deitada = (norma > 0.6f * ACCEL_LSB_POR_G)
                 && (fabsf(vetor[eixo]) > 0.95f * norma);
  if (deitada) {
    float signo = (vetor[eixo] >= 0.0f) ? 1.0f : -1.0f;
    vetor[eixo] -= signo * ACCEL_LSB_POR_G;
    accelOffX = vetor[0]; accelOffY = vetor[1]; accelOffZ = vetor[2];
    Serial.printf("[imu] accel offsets MPU 0x%02X: (%.2f, %.2f, %.2f) g"
                  " (eixo dominante %d, %s)\n",
                  addr, accelOffX / ACCEL_LSB_POR_G,
                  accelOffY / ACCEL_LSB_POR_G, accelOffZ / ACCEL_LSB_POR_G,
                  eixo, (signo > 0.0f) ? "para cima" : "para baixo");
  } else {
    accelOffX = accelOffY = accelOffZ = 0.0f;
    Serial.printf("[imu] AVISO MPU 0x%02X: acelerometro nao esta' deitado "
                  "(%0.2f g fora de eixo) -- offsets de accel NAO aplicados; "
                  "deixe a luva deitada e reinicie\n",
                  addr, norma / ACCEL_LSB_POR_G);
  }
  Serial.printf("[imu] bias MPU 0x%02X: (%.4f, %.4f, %.4f)\n",
                addr, biasX, biasY, biasZ);
}


void writeRegisterAt(uint8_t addr, uint8_t reg, uint8_t value) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission(true);
}


// ============================ ATITUDE (Mahony 6 eixos) =====================
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


  /** Copia o quaternion corpo->mundo (w, x, y, z) para o serial. */
  void obter_quaternion(float &w, float &x, float &y, float &z) const {
    w = q[0];
    x = q[1];
    y = q[2];
    z = q[3];
  }

  /** Reinicia a atitude; o proximo update reinicializa pelo acelerometro. */
  void reiniciar() {
    q[0] = 1.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 0.0f;
    iniciada = false;
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

// ============================ ESTADO GLOBAL ================================
AttitudeMahony atitudeD;   // direita (0x68)
AttitudeMahony atitudeE;   // esquerda (0x69)
float gyroBiasDx = 0.0f, gyroBiasDy = 0.0f, gyroBiasDz = 0.0f;
float gyroBiasEx = 0.0f, gyroBiasEy = 0.0f, gyroBiasEz = 0.0f;
float accelOffDx = 0.0f, accelOffDy = 0.0f, accelOffDz = 0.0f;
float accelOffEx = 0.0f, accelOffEy = 0.0f, accelOffEz = 0.0f;
uint32_t previousMicrosD = 0, previousMicrosE = 0;
uint32_t ultimaLinhaMs = 0;
uint32_t ultimoErroDMs = 0, ultimoErroEMs = 0;
float ultimoAxD = 0, ultimoAyD = 0, ultimoAzD = 0;
float ultimoAxE = 0, ultimoAyE = 0, ultimoAzE = 0;

void calibrarTudo() {
  Serial.println("[ahrs] CALIBRANDO: deixe as DUAS IMUs PARADAS e com a face "
                 "PARALELA A MESA (4 s)...");
  calibrateImuAt(ADDR_DIREITO, gyroBiasDx, gyroBiasDy, gyroBiasDz,
                 accelOffDx, accelOffDy, accelOffDz);
  calibrateImuAt(ADDR_ESQUERDO, gyroBiasEx, gyroBiasEy, gyroBiasEz,
                 accelOffEx, accelOffEy, accelOffEz);
  atitudeD.reiniciar();
  atitudeE.reiniciar();
  previousMicrosD = micros();
  previousMicrosE = micros();
  Serial.println("[ahrs] calibracao concluida; vetores serao zerados no "
                 "proximo quadro (Z para baixo).");
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println();
  Serial.println("[ahrs] TESTE AHRS 2x MPU6050 -- vetores 3D (item 142)");
  setupI2c();

  for (uint8_t addr : {ADDR_DIREITO, ADDR_ESQUERDO}) {
    writeRegisterAt(addr, 0x6B, 0x00);
    writeRegisterAt(addr, 0x6B, 0x03);
    writeRegisterAt(addr, 0x1B, GYRO_FS_SEL);
    writeRegisterAt(addr, 0x1C, 0x00);
    writeRegisterAt(addr, 0x1A, MPU_DLPF_CFG);
    writeRegisterAt(addr, 0x19, MPU_SMPLRT_DIV);
  }
  delay(200);
  calibrarTudo();

  uint32_t inicio = millis();
  int16_t ax, ay, az, gx, gy, gz;
  float dt = 0.005f;
  while (millis() - inicio < 1500) {
    if (readMpuAt(ADDR_DIREITO, ax, ay, az, gx, gy, gz)) {
      float a[3] = {(ax - accelOffDx) / ACCEL_LSB_POR_G,
                    (ay - accelOffDy) / ACCEL_LSB_POR_G,
                    (az - accelOffDz) / ACCEL_LSB_POR_G};
      float g[3] = {gx / GYRO_LSB_POR_DPS - gyroBiasDx,
                    gy / GYRO_LSB_POR_DPS - gyroBiasDy,
                    gz / GYRO_LSB_POR_DPS - gyroBiasDz};
      atitudeD.update(g[0], g[1], g[2], a[0], a[1], a[2], dt);
    }
    if (readMpuAt(ADDR_ESQUERDO, ax, ay, az, gx, gy, gz)) {
      float a[3] = {(ax - accelOffEx) / ACCEL_LSB_POR_G,
                    (ay - accelOffEy) / ACCEL_LSB_POR_G,
                    (az - accelOffEz) / ACCEL_LSB_POR_G};
      float g[3] = {gx / GYRO_LSB_POR_DPS - gyroBiasEx,
                    gy / GYRO_LSB_POR_DPS - gyroBiasEy,
                    gz / GYRO_LSB_POR_DPS - gyroBiasEz};
      atitudeE.update(g[0], g[1], g[2], a[0], a[1], a[2], dt);
    }
    delay(5);
  }
  previousMicrosD = micros();
  previousMicrosE = micros();
  Serial.println("# AHRS_VETOR v1 | q=corpo->mundo (w,x,y,z) | "
                 "vetores zerados no 1o quadro: -Z = chao");
  Serial.println("# A,t_us,qwL,qxL,qyL,qzL,qwR,qxR,qyR,qzR,"
                 "axL,ayL,azL,axR,ayR,azR");
  Serial.println("[ahrs] pronto. Gire os MPUs; rode o visualizador no PC.");
}

void loop() {
  int16_t rawAx, rawAy, rawAz, rawGx, rawGy, rawGz;

  if (!readMpuAt(ADDR_DIREITO, rawAx, rawAy, rawAz, rawGx, rawGy, rawGz)) {
    if (millis() - ultimoErroDMs > 1000) {
      ultimoErroDMs = millis();
      Serial.println("E,0,D");
    }
  } else {
    uint32_t now = micros();
    float dt = (now - previousMicrosD) * 1.0e-6f;
    previousMicrosD = now;
    ultimoAxD = (rawAx - accelOffDx) / ACCEL_LSB_POR_G;
    ultimoAyD = (rawAy - accelOffDy) / ACCEL_LSB_POR_G;
    ultimoAzD = (rawAz - accelOffDz) / ACCEL_LSB_POR_G;
    if (dt > 0.0f && dt <= 0.1f) {
      float gx = rawGx / GYRO_LSB_POR_DPS - gyroBiasDx;
      float gy = rawGy / GYRO_LSB_POR_DPS - gyroBiasDy;
      float gz = rawGz / GYRO_LSB_POR_DPS - gyroBiasDz;
      atitudeD.update(gx, gy, gz, ultimoAxD, ultimoAyD, ultimoAzD, dt);
    }
  }

  if (!readMpuAt(ADDR_ESQUERDO, rawAx, rawAy, rawAz, rawGx, rawGy, rawGz)) {
    if (millis() - ultimoErroEMs > 1000) {
      ultimoErroEMs = millis();
      Serial.println("E,0,E");
    }
  } else {
    uint32_t now = micros();
    float dt = (now - previousMicrosE) * 1.0e-6f;
    previousMicrosE = now;
    ultimoAxE = (rawAx - accelOffEx) / ACCEL_LSB_POR_G;
    ultimoAyE = (rawAy - accelOffEy) / ACCEL_LSB_POR_G;
    ultimoAzE = (rawAz - accelOffEz) / ACCEL_LSB_POR_G;
    if (dt > 0.0f && dt <= 0.1f) {
      float gx = rawGx / GYRO_LSB_POR_DPS - gyroBiasEx;
      float gy = rawGy / GYRO_LSB_POR_DPS - gyroBiasEy;
      float gz = rawGz / GYRO_LSB_POR_DPS - gyroBiasEz;
      atitudeE.update(gx, gy, gz, ultimoAxE, ultimoAyE, ultimoAzE, dt);
    }
  }

  if (millis() - ultimaLinhaMs >= 20) {
    ultimaLinhaMs = millis();
    float wl, xl, yl, zl, wr, xr, yr, zr;
    atitudeD.obter_quaternion(wr, xr, yr, zr);
    atitudeE.obter_quaternion(wl, xl, yl, zl);
    char linha[280];
    snprintf(linha, sizeof(linha),
             "A,%lu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
             "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
             (unsigned long)micros(),
             wl, xl, yl, zl, wr, xr, yr, zr,
             ultimoAxE, ultimoAyE, ultimoAzE,
             ultimoAxD, ultimoAyD, ultimoAzD);
    Serial.print(linha);
  }

  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == 'c' || c == 'C') {
      calibrarTudo();
    } else if (c == 'r' || c == 'R') {
      atitudeD.reiniciar();
      atitudeE.reiniciar();
      Serial.println("[ahrs] atitude reiniciada; vetor sera' zerado no "
                     "proximo quadro.");
    }
  }
  delay(2);
}
