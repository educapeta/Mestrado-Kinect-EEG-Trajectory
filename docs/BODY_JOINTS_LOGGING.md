# Body Joints Logging — Especificação do CSV de Juntas 3D do Kinect

> **Objetivo**: Gravar os nós 3D do esqueleto do Kinect (corpo inteiro) em coordenadas **BRUTAS do camera space**, para permitir transformar depois os labels para um frame corporal (ombro/esterno/cabeça) em pós-processamento.

---

## 1. Decisões de Design

| Item | Decisão | Justificativa |
|------|---------|---------------|
| **Frame de coordenadas** | Camera space do Kinect (raw) | Não depende de calibração de origem; permite `T_base_cam` posterior |
| **Taxa de amostragem** | ~30 Hz (configurável via `--mov-hz`) | Banda do movimento ~15 Hz; Nyquist satisfeito; não infla CSV de EEG (500 Hz) |
| **Arquivo separado** | `corpo_<participante>_s<sessao>_<timestamp>.csv` | Independente do CSV de EEG; mesmo timestamp da sessão |
| **Juntas mínimas** | 13 juntas (Head, Neck, SpineShoulder, SpineMid, SpineBase, ShoulderLeft, ShoulderRight, ElbowLeft, ElbowRight, WristLeft, WristRight, HandLeft, HandRight) | Cobertura tronco + braços; suficiente para frame corporal |
| **Tracking state** | 0=NotTracked, 1=Inferred, 2=Tracked | Qualidade por junta; permite filtrar pós-hoc |
| **Metadados JSON** | Arquivo `.json` lado a lado | Lista de juntas, versão, frame de coordenadas, tracking codes |

---

## 2. Formato do CSV

### 2.1 Cabeçalho

```
t_mono_s,t_epoch_s,frame,
Head_x_m,Head_y_m,Head_z_m,Head_tracked,
Neck_x_m,Neck_y_m,Neck_z_m,Neck_tracked,
SpineShoulder_x_m,SpineShoulder_y_m,SpineShoulder_z_m,SpineShoulder_tracked,
SpineMid_x_m,SpineMid_y_m,SpineMid_z_m,SpineMid_tracked,
SpineBase_x_m,SpineBase_y_m,SpineBase_z_m,SpineBase_tracked,
ShoulderLeft_x_m,ShoulderLeft_y_m,ShoulderLeft_z_m,ShoulderLeft_tracked,
ShoulderRight_x_m,ShoulderRight_y_m,ShoulderRight_z_m,ShoulderRight_tracked,
ElbowLeft_x_m,ElbowLeft_y_m,ElbowLeft_z_m,ElbowLeft_tracked,
ElbowRight_x_m,ElbowRight_y_m,ElbowRight_z_m,ElbowRight_tracked,
WristLeft_x_m,WristLeft_y_m,WristLeft_z_m,WristLeft_tracked,
WristRight_x_m,WristRight_y_m,WristRight_z_m,WristRight_tracked,
HandLeft_x_m,HandLeft_y_m,HandLeft_z_m,HandLeft_tracked,
HandRight_x_m,HandRight_y_m,HandRight_z_m,HandRight_tracked
```

### 2.2 Colunas

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `t_mono_s` | float64 | `time.monotonic()` — relógio monótono do PC (mesmo relógio do JSON de eventos e do CSV de movimento) |
| `t_epoch_s` | float64 | `time.time()` — Unix epoch (wall clock) para referência absoluta |
| `frame` | int | Contador de frames do gravador (0-based) |
| `<Junta>_x_m` | float64 | Coordenada X em metros (camera space: **+X = direita da câmera**) |
| `<Junta>_y_m` | float64 | Coordenada Y em metros (camera space: **+Y = cima**) |
| `<Junta>_z_m` | float64 | Coordenada Z em metros (camera space: **+Z = para frente/para dentro da cena**) |
| `<Junta>_tracked` | int | Estado de tracking: `0=NotTracked`, `1=Inferred`, `2=Tracked` |

### 2.3 Valores especiais

- **NaN** (`nan` no CSV): junta não disponível (tracking lost, fora do FOV, etc.)
- **Tracking state 0**: `NotTracked` — junta não detectada
- **Tracking state 1**: `Inferred` — junta inferida pelo SDK (ocluída, baixa confiança)
- **Tracking state 2**: `Tracked` — junta rastreada com alta confiança

---

## 3. Frame de Coordenadas: Kinect Camera Space

```
         +Y (cima)
          |
          |
          +------ +X (direita da câmera)
         /
        /
       +Z (para frente / para dentro da cena)
```

- **Origem**: Centro óptico do sensor de profundidade do Kinect v2
- **Unidades**: Metros (float64)
- **Mão direita**: Sistema dextrógiro (X × Y = Z)
- **Não é relativo à origem da calibração** — são coordenadas absolutas do camera space

---

## 4. Transformação Posterior para Frame Corporal (T_base_cem)

Como as coordenadas são brutas no camera space, a transformação para um frame corporal (ex.: origem no esterno ou ponto médio dos ombros) pode ser feita em pós-processamento:

```python
# Exemplo de transformação para frame corporal
# base_origin = "sternum" | "shoulder_mid" | "head"

# 1. Carrega CSV de corpo + JSON de metadados
# 2. Identifica juntas de referência (ex.: SpineShoulder, ShoulderLeft, ShoulderRight)
# 3. Define origem e eixos do frame corporal:
#    - origin = (ShoulderLeft + ShoulderRight) / 2  # ponto médio dos ombros
#    - x_axis = normalize(ShoulderRight - ShoulderLeft)  # direita do sujeito
#    - z_axis = normalize(SpineShoulder - origin)        # frente do sujeito
#    - y_axis = cross(z_axis, x_axis)                    # cima (ortogonal)
# 4. T_base_cam = [R | -R @ origin]  # 4x4 homogênea
# 5. Para cada frame e cada junta: p_base = R @ (p_cam - origin)
```

### Vantagens do camera space bruto:
- **Reprocessável**: Se a calibração de origem mudar, não precisa regravar
- **Auditável**: Pode verificar qualidade do tracking frame a frame
- **Flexível**: Múltiplas definições de frame corporal possíveis (esterno, cabeça, ombro médio)
- **Sincronização**: Mesmo `t_mono_s` do JSON de eventos e CSV de movimento

---

## 5. Sincronização com Outros Streams

| Stream | Relógio | Como alinhar |
|--------|---------|--------------|
| EEG (500 Hz) | `t_mono_s` no JSON de eventos (`sample_index` + `monotonic_s`) | Interpolar `t_mono_s` do corpo para amostras de EEG |
| Movimento (KT/ARM/PALM, 30 Hz) | `t_mono_s` no CSV de movimento | Mesmo relógio — join direto por `t_mono_s` |
| Eventos (markers) | `monotonic_s` no JSON de eventos | Mesmo relógio — join direto |
| IMU (100 Hz) | `timestamp_us` no ESP32 + offset no JSON | Requer sincronização separada (hardware marker ou xcorr) |

### Exemplo de alinhamento EEG ↔ Corpo

```python
# JSON de eventos tem: eventos[i]["monotonic_s"] e eventos[i]["sample_index"]
# CSV de corpo tem: t_mono_s por linha

# 1. Dois eventos conhecidos (ex.: CODE_SESSION e CODE_SESSION_END)
#    dão a relação linear: sample_index = a * t_mono_s + b
# 2. Para cada linha do corpo: sample_eeg = a * t_mono_s + b
# 3. Interpolar EEG para o tempo do corpo (ou vice-versa)
```

---

## 6. Juntas Gravadas (PyKinectV2 JointType)

| Nome no CSV | PyKinectV2 JointType | Descrição Anatômica |
|-------------|---------------------|---------------------|
| `Head` | `JointType_Head` | Topo da cabeça |
| `Neck` | `JointType_Neck` | Pescoço (base do crânio) |
| `SpineShoulder` | `JointType_SpineShoulder` | Clavícula / topo do tronco |
| `SpineMid` | `JointType_SpineMid` | Meio do tronco (coluna torácica) |
| `SpineBase` | `JointType_SpineBase` | Base do tronco (pelvis/quadril) |
| `ShoulderLeft` | `JointType_ShoulderLeft` | Ombro esquerdo |
| `ShoulderRight` | `JointType_ShoulderRight` | Ombro direito |
| `ElbowLeft` | `JointType_ElbowLeft` | Cotovelo esquerdo |
| `ElbowRight` | `JointType_ElbowRight` | Cotovelo direito |
| `WristLeft` | `JointType_WristLeft` | Punho esquerdo |
| `WristRight` | `JointType_WristRight` | Punho direito |
| `HandLeft` | `JointType_HandLeft` | Mão esquerda (palma) |
| `HandRight` | `JointType_HandRight` | Mão direita (palma) |

---

## 7. Uso na Linha de Comando

```bash
# Ligado por padrão quando Kinect está ativo
python eeg_motor_paradigm.py --source gnautilus --sessao 1

# Desligar explicitamente
python eeg_motor_paradigm.py --source gnautilus --sessao 1 --sem-corpo

# Forçar ligado (útil para testes com generator)
python eeg_motor_paradigm.py --source generator --sem-kinect --corpo
```

### Comportamento padrão:
- **Com Kinect (`--sem-kinect` NÃO passado)**: `--corpo` é **ligado por padrão**
- **Sem Kinect (`--sem-kinect` passado)**: `--corpo` é **desligado** (não há esqueleto)
- `--sem-corpo` força desligado mesmo com Kinect
- `--corpo` força ligado (útil para testes)

---

## 8. Arquivos Gerados

Para uma sessão `gravacao_MEMI_P01_s1_20261115_143022.csv`:

```
gravacoes/
├── gravacao_MEMI_P01_s1_20261115_143022.csv          # EEG (500 Hz)
├── gravacao_MEMI_P01_s1_20261115_143022_eventos.json # Eventos + metadados
├── gravacao_MEMI_P01_s1_20261115_143022_movimento.csv # KT/ARM/PALM (30 Hz)
├── corpo_P01_s1_20261115_143022.csv                   # Juntas 3D corpo (30 Hz)  ← NOVO
├── corpo_P01_s1_20261115_143022.json                  # Metadados do corpo        ← NOVO
└── priming/                                           # Clipes (se --video-real)
    └── t001_garrafa_direita.avi
```

### Metadados JSON do Corpo (`corpo_*.json`)

```json
{
  "joints": ["Head", "Neck", "SpineShoulder", "SpineMid", "SpineBase",
             "ShoulderLeft", "ShoulderRight", "ElbowLeft", "ElbowRight",
             "WristLeft", "WristRight", "HandLeft", "HandRight"],
  "joint_count": 13,
  "coordinate_frame": "kinect_camera_space",
  "units": "meters",
  "tracking_state_codes": {
    "0": "NotTracked",
    "1": "Inferred",
    "2": "Tracked"
  },
  "version": "1.0",
  "description": "3D body joints from Kinect v2 in raw camera space coordinates",
  "csv_file": "corpo_P01_s1_20261115_143022.csv",
  "rows_written": 18000,
  "duration_s": 600.0,
  "created_at": "2026-11-15 14:30:22"
}
```

---

## 9. Teste com Mock (Sem Hardware)

O módulo `body_joints_recorder.py` inclui um helper de teste que aceita um objeto mock de `last_bodies` e devolve as linhas esperadas:

```python
from body_joints_recorder import (
    create_mock_last_bodies,
    extract_body_joints,
    build_body_joints_row,
    test_extract_body_joints_with_mock
)

# 1. Teste rápido (roda sem Kinect)
test_extract_body_joints_with_mock()

# 2. Uso customizado com posições próprias
custom_positions = {
    "Head": (0.0, 1.7, 2.1),
    "Neck": (0.0, 1.55, 2.1),
    # ... outras juntas
}
mock_bodies = create_mock_last_bodies(custom_positions)
joints_data = extract_body_joints(mock_bodies)
row = build_body_joints_row(123.456, 1700000000.123, 42, joints_data)
print(",".join(row))
```

### Saída esperada do teste:
```
[OK] Teste extract_body_joints com mock PASSOU
  Linha CSV: 123.456000,1700000000.123000,42,0.000000,1.600000,2.000000,2,...

[OK] Todos os testes do helper mock passaram!
```

---

## 10. Limitações Conhecidas

1. **Requer Kinect v2 + SDK 2.0**: Não funciona com `--sem-kinect` (exceto com mock para testes)
2. **Tracking de corpo inteiro**: Qualidade depende de o sujeito estar visível no FOV do Kinect (corpo todo)
3. **Juntas inferidas**: `ShoulderLeft/Right`, `ElbowLeft/Right`, `HandLeft/Right` são frequentemente `Inferred` (state=1) quando oclusão
4. **Sem sincronização hardware com IMU**: O relógio do ESP32 é independente; alinhar requer pós-processamento
5. **Latência**: O `last_bodies` é atualizado a ~30 Hz no loop de tracking; pode haver jitter de ±1 frame
6. **Não grava pernas**: `HipLeft/Right`, `KneeLeft/Right`, `AnkleLeft/Right`, `FootLeft/Right` não incluídas (fora do escopo de braço/tronco)

---

## 11. Versionamento

| Versão | Data | Mudanças |
|--------|------|----------|
| 1.0 | 2026-11-15 | Implementação inicial: 13 juntas, camera space raw, CSV + JSON separado |

---

## 12. Referências de Código

- **Recorder principal**: `body_joints_recorder.py` — classe `BodyJointsRecorder`
- **Integração no paradigma**: `eeg_motor_paradigm.py` — função `main()` (criação, start, stop)
- **Teste mock**: `body_joints_recorder.py` — `create_mock_last_bodies()`, `test_extract_body_joints_with_mock()`
- **Tracking thread**: `kinect_imu_groundtruth.py` — `KinectHandTracker.last_bodies`
- **Joint types**: `kinect_imu_groundtruth.py` — `BODY_CONNECTIONS`, `JOINT_TYPE_MAP` (local em `body_joints_recorder.py`)