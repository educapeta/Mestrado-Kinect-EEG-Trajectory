"""Diagnostico dos 2 MPU6050 via serial CRUA do ESP32 (esp32_duas_imus).

Uso: .venv\Scripts\python.exe tools\diagnostico_imu_serial.py COM4 115200 60

Faz reset no ESP32 (DTR/RTS), captura a linha do tempo inteira do boot e
classifica:
  - tentativas de Wi-Fi (hotspot desligado = 20 s de pontos por tentativa);
  - `[imu] calibrando/bias MPU 0x68/0x69` (bias EXATAMENTE 0 = chip mudo);
  - `Erro de leitura I2C D` (0x68) / `Erro de leitura I2C E` (0x69);
  - cabecalho CSV (firmware novo) e linhas de dados de 10 campos.
"""
import re
import sys
import time
from collections import Counter

import serial

port, baud, segs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])

_F = r"-?\d+(?:\.\d+)?"
RE10 = re.compile(r"^" + _F + r"(?:," + _F + r"){9}$")

s = serial.Serial(port, baud, timeout=0.3)
# Reset explicito (circuito auto-reset do CP210x): IO0 alto = boot normal.
s.dtr = False
s.rts = True
time.sleep(0.2)
s.rts = False
time.sleep(0.1)
s.reset_input_buffer()

t0 = time.monotonic()
linhas = []  # (t_segundos, texto)
while time.monotonic() - t0 < segs:
    raw = s.readline()
    if raw:
        linhas.append((time.monotonic() - t0,
                       raw.decode("utf-8", errors="replace").rstrip()))
s.close()


def classe(l: str) -> str:
    # Linha com prefixo de lado (`1,...` / `2,...`) = firmware novo na serial.
    m = re.match(r"^([12]),(.+)$", l)
    corpo = m.group(2) if m else l
    if RE10.match(corpo):
        return f"dados_{m.group(1)}" if m else "dados"
    if l.startswith("esp_timestamp_us"):
        return "cabecalho"
    if "Erro de leitura I2C D" in l:
        return "erro_D"
    if "Erro de leitura I2C E" in l:
        return "erro_E"
    if l.startswith("Conectando"):
        return "wifi_tentando"
    if l == ".":
        return "wifi_ponto"
    if "[wifi]" in l:
        return "wifi_status"
    if "[imu] calibrando" in l:
        return "calibrando"
    if "[imu] bias" in l:
        return "bias"
    if "rst:0x" in l or "Build:" in l or "features:" in l or "boot:" in l:
        return "boot"
    if l == "":
        return "vazio"
    return "outro"


# Dump CRU com timestamps para analise posterior.
with open("_diagnostico_imu_serial.raw.log", "w", encoding="utf-8") as f:
    for t, l in linhas:
        f.write(f"{t:7.2f} {l}\n")

conta = Counter()
por_classe = {}
for t, l in linhas:
    c = classe(l)
    conta[c] += 1
    por_classe.setdefault(c, []).append((t, l))

ordem = ["boot", "wifi_tentando", "wifi_ponto", "wifi_status", "calibrando",
         "bias", "cabecalho", "erro_D", "erro_E", "dados", "dados_1",
         "dados_2", "outro", "vazio"]
print(f"=== {len(linhas)} linhas em {segs:.0f}s (porta {port}) ===")
for c in ordem:
    if conta[c]:
        print(f"{c:14s}: {conta[c]}")

for c in ["boot", "wifi_status", "calibrando", "bias", "outro", "cabecalho"]:
    for t, l in por_classe.get(c, [])[:25]:
        print(f"  [{c} {t:5.1f}s] {l}")
for c in ["erro_D", "erro_E"]:
    amostras = por_classe.get(c, [])
    for t, l in amostras[:3]:
        print(f"  [{c} {t:5.1f}s] {l}")
    if len(amostras) > 3:
        t, l = amostras[-1]
        print(f"  [{c} {t:5.1f}s] ... (ultima) {l}")
linhas_dados = (por_classe.get("dados", []) + por_classe.get("dados_1", [])
                + por_classe.get("dados_2", []))
for t, l in linhas_dados[:4] + linhas_dados[-2:]:
    print(f"  [dados {t:5.1f}s] {l}")

# Veredito automatico dos IMUs.
print("\n=== VEREDITO ===")
bias_lines = por_classe.get("bias", [])
for addr in ("68", "69"):
    linha = next((l for l in (x[1] for x in bias_lines) if f"0x{addr}" in l), None)
    if linha is None:
        print(f"MPU 0x{addr}: sem linha de bias (init nao chegou la)")
        continue
    nums = re.findall(r"-?\d+\.\d+", linha.split(":", 1)[1])
    zerado = nums and all(abs(float(n)) < 1e-9 for n in nums)
    print(f"MPU 0x{addr}: {linha.split(':',1)[1].strip()}"
          + ("  <- EXATAMENTE ZERO (leituras todas falharam = chip mudo?)"
             if zerado else "  (bias plausivel = chip respondendo)"))
print(f"linhas 'Erro de leitura I2C D' (0x68): {conta['erro_D']}")
print(f"linhas 'Erro de leitura I2C E' (0x69): {conta['erro_E']}")
print(f"linhas de dados lado 1 / lado 2     : {conta['dados_1']} / {conta['dados_2']}"
      + (f"  (sem prefixo de lado: {conta['dados']})" if conta["dados"] else ""))

# --- Analise temporal dos erros de barramento (janelas de calibracao) ------
# requestFrom com Error -1: NA versao do Wire usada aqui o endTransmission(false)
# e' um no-op (retorna 0 sem tocar o barramento); a transacao inteira acontece
# no requestFrom, e o Error -1 = ESP_FAIL do driver IDF = NACK = o endereco NAO
# respondeu. Logo, erros concentrados na janela de um chip = aquele endereco
# nao teve resposta (chip ausente, sem energia ou com fio no lugar errado).
wire = [(t, l) for c, ps in por_classe.items() if c == "outro"
        for t, l in ps if "Wire.cpp" in l]
outros_sem_wire = [(t, l) for c, ps in por_classe.items() if c == "outro"
                   for t, l in ps if "Wire.cpp" not in l]
janelas = {}
for rotulo in ("calibrando", "bias"):
    for t, l in por_classe.get(rotulo, []):
        m = re.search(r"0x([0-9A-F]{2})", l)
        if m and m.group(1).lower() not in janelas:
            janelas[m.group(1).lower()] = [t, None]
        elif m:
            janelas[m.group(1).lower()][1] = t
print("\n=== ERROS Wire.cpp POR JANELA DE CALIBRACAO ===")
print("janelas:", janelas)
for addr, (ini, fim) in janelas.items():
    if fim is None:
        fim = segs
    n = sum(1 for t, _ in wire if ini <= t <= fim)
    print(f"MPU 0x{addr} [{ini:.1f}s..{fim:.1f}s]: "
          f"{n} erros requestFrom = NACK (endereco nao respondeu)")
n_antes = sum(1 for t, _ in wire if t < min(v[0] for v in janelas.values()))
print(f"erros ANTES da primeira calibracao: {n_antes}")
print(f"Wire.cpp total: {len(wire)} | 'outro' sem Wire.cpp: {len(outros_sem_wire)}")
for t, l in outros_sem_wire[:30]:
    print(f"  [outro {t:6.2f}s] {l}")

# --- Histograma de erros Wire a cada 2 s -----------------------------------
print("\n=== HISTOGRAMA Wire.cpp (2 s) ===")
if wire:
    largura = 2.0
    hist = Counter(int(t // largura) * largura for t, _ in wire)
    for bucket in sorted(hist):
        print(f"  {bucket:5.1f}-{bucket + largura:4.1f}s: {hist[bucket]}")

