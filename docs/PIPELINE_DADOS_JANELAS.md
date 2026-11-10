# Pipeline de dados e janelas — EEG → pose da mão

> Documento para iniciante. Ele conecta o schema do dataset
> (`docs/DATASET_POSE_SCHEMA.md`), o treino (`docs/SAND_POSE_ARCHITECTURE.md`) e
> o tempo real. A pergunta central é: **qual pedaço de EEG eu mostro ao modelo
> para prever qual pedaço de pose?**

## 1. Visão em 5 camadas

```text
[1] AQUISIÇÃO CRUA
    EEG 500 Hz | câmeras 30 fps | IMU 100 Hz | eventos
              ↓
[2] LINHA DO TEMPO SINCRONIZADA
    cada amostra/label sabe seu t_us no relógio do PC
              ↓
[3] LABELS DE POSE
    T_base_tcp(t), abertura(t), landmarks crus para o futuro
              ↓
[4] JANELAS DE TREINO
    X_eeg[janela] -> Y_pos[janela], Y_rot6d[janela], Y_abertura[janela]
              ↓
[5] INFERÊNCIA ONLINE
    buffer circular -> janela causal -> modelo -> suavização -> robô
```

A regra de ouro: **a camada 1 é crua e nunca é sobrescrita**. Todo
processamento (filtro, resample, normalização) é reproduzível a partir dela.

## 2. Camada 1 — aquisição crua

Gravar cada sensor na taxa nativa:

| Sensor | Taxa | O que guardar |
| --- | --- | --- |
| EEG | 500 Hz | `eeg[N,C]`, `t_us[N]`, nomes dos canais |
| câmeras | ~30 fps | frames e/ou landmarks 2D + confiança |
| IMU | ~100 Hz | acc, gyro, quaternion, lado, boot_id |
| eventos | quando ocorrer | trial start/end, LED, onset |

Nunca grave só a versão filtrada. O filtro muda entre treino e tempo real; se
você não tiver o cru, não consegue reproduzir.

## 3. Camada 2 — sincronização

O problema: EEG, câmera e IMU têm relógios/atrasos diferentes.

Você precisa de:

1. **timestamp do PC** para tudo que chega nele;
2. um **offset** por stream (`sync.csv`);
3. um evento comum (LED, trigger, clique) para ancorar.

Exemplo de timeline:

```text
t_us:        0        100000    200000    300000
EEG:         ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  (500 Hz)
Câmera:      ●                 ●                 (30 fps)
Label pose:  ●                 ●                 (30 Hz)
```

Para treinar, você não precisa reamostrar o EEG. Para cada label de pose em
`t_label`, você recorta uma janela de EEG ao redor dele.

Para o tempo real, você mantém buffers separados e usa o **último instante
comum** — nunca “espera o futuro” para montar a janela.

### Latência

Meça e registre:

- latência da câmera;
- latência do MediaPipe;
- latência da triangulação;
- latência do modelo;
- latência do robô.

O atraso total entra no orçamento. Para controle, mire em **< 100–150 ms** no
total; se a câmera atrasa 60 ms e o modelo 30 ms, o alvo deve ser preditivo
(futuro), não concorrente.

## 4. Camada 3 — labels

Para o piloto:

```text
T_base_tcp(t) = posição (3) + quaternion (4) + abertura (1)
```

Onde:

- `base`: tronco/ombros;
- `tcp`: ponto de ferramenta da garra;
- abertura: distância normalizada entre as garras (ou entre polegar e
  indicador), `0 = fechada`, `1 = aberta`.

Guarde também, para o futuro:

- 21 landmarks 3D triangulados;
- ângulos das juntas dos dedos (se o modelo cinemático existir);
- normal da palma e direção dos dedos.

Assim você não precisa recoletar quando quiser controlar dedos individualmente.

## 5. Camada 4 — janelas de treino

### 5.1. O desenho da janela

```text
                         t_anchor (fim da janela de EEG)
                                      │
EEG:   [──────── janela de entrada ────]│
Pose:                                  [─── alvo futuro ───]
       t_anchor - 2.0 s               t_anchor      t_anchor + 1.0 s
```

- **entrada:** EEG dos últimos 2 s;
- **alvo:** pose de `t_anchor` até `t_anchor + 1 s` (preditivo), ou pose
  concorrente à janela (offline).

Para o **piloto offline**, comece concorrente: a janela de EEG cobre o mesmo
intervalo do alvo. É mais fácil de depurar e compara com o SAND atual.

Para o **controle online**, mude para preditivo: janela só do passado; alvo no
futuro. Aí você mede o horizonte que o modelo consegue antecipar.

### 5.2. Parâmetros iniciais

| Parâmetro | Piloto |
| --- | --- |
| `fs_eeg` | 500 Hz |
| `window_sec` | 2.0 s |
| `window_samples` | 1000 |
| `stride` | 0.1 s (10 Hz de janelas) |
| `target_fs` | 30 Hz |
| `target_seq_len` | 30 (1 s) |
| `target_start_sec` | 0.0 (concorrente) ou +0.1–0.3 (preditivo) |
| `target_end_sec` | 1.0 |
| filtro | causal (`sosfilt`) |
| banda | 1–30 Hz |
| normalização | CAR + z-score |

Por que 2 s? É o que o SAND já usa; dá contexto motor e cabe no orçamento de
inferência. Depois teste 1 s e 3 s como ablação.

### 5.3. Como gerar as janelas

Para cada label `t_k`:

```text
inicio_eeg = t_k + window_start * fs
fim_eeg    = inicio_eeg + window_n
X = eeg[inicio_eeg:fim_eeg]
Y_pos = pose_pos[t_k + target_start : t_k + target_end]  # reamostrado em S
Y_rot = pose_rot[...]
Y_ap  = abertura[...]
```

- `window_start` negativo significa passado;
- descarte janelas que invadem o início/fim da gravação;
- se não houver pose válida, `mask=0` e a janela pode ser mantida para o
  modelo aprender “sem label”? No piloto, descarte ou mascare; não invente
  label.
- **nunca** deixe uma janela de treino e uma de validação da mesma sessão se
  sobreporem no tempo. O split é por sessão, não por janela.

### 5.4. Tensores de treino

```text
x_eeg      (N, C, T)      # 500 Hz, T=1000
y_pos      (N, S, 3)      # m, relativo ao repouso
y_rot6d    (N, S, 6)      # 6D de R
y_ap       (N, S, 1)      # [0,1]
y_mask     (N, S)         # pose válida
```

Opcional, para o futuro:

```text
y_joints   (N, S, J)      # ângulos dos dedos
x_imu      (N, 10, T)     # IMU reamostrado
```

### 5.5. Tratamento da abertura

- no piloto, use **uma** variável escalar;
- derive de `thumb_tip` e `index_tip` triangulados:
  `abertura = clip(dist / max_dist, 0, 1)`;
- se a garra não for proporcional, use classes (`fechada`, `meia`, `aberta`)
  ou regressão + limiar;
- no futuro, troque a cabeça escalar por um vetor de juntas.

### 5.6. Janela de análise (4 s) NÃO é a janela de entrada (2 s)

Essa é a confusão mais comum. Existem três coisas diferentes:

| Coisa | Exemplo | Para que serve |
| --- | --- | --- |
| **janela do trial** | cue -0,5 s até cue + 3,5 s = 4 s | contexto experimental e análise de fases |
| **janela de entrada do modelo** | últimos 2 s de EEG | o tensor que a rede recebe |
| **janela de saída** | pose atual, ou 0,5–1 s à frente | o que a rede prevê |

Se o modelo foi treinado com entrada de **4 s**, no online ele precisa receber
**4 s**. Não dá para alimentar 2 s sem retreinar ou sem um modelo que aceite
comprimento variável. O contrário também vale: se treinou com 2 s, o online usa
2 s.

#### O que eu recomendo

- **offline:** analise o trial de 4 s (fases, ERD/ERS, pose ao longo do trial);
- **modelo:** comece com entrada fixa de **2 s**, a mesma do SAND atual
  (`T = 1000` amostras a 500 Hz);
- **online:** janela rolante de 2 s, deslizando a cada 33–100 ms.

Assim o modelo é treinado e usado no mesmo formato. O trial de 4 s continua
existindo para a análise e para o desenho experimental.

#### “Mas a janela de 2 s não me atrasa 2 s?”

Não. A janela é **histórico**, não espera. O loop online funciona assim:

```text
a cada 33 ms:
  pega os últimos 2 s do ring buffer
  pré-processa
  roda o modelo
  prevê a pose de agora (ou de +100 ms)
  envia ao robô
```

Depois que o buffer encheu (2 s após o início), você recebe uma predição a
cada 33 ms. O atraso real é:

```text
latência = aquisição + comunicação + pré-processamento + inferência + suavização + robô
```

A janela de 2 s **não** soma 2 s de atraso. Ela é o contexto que o modelo olha
para trás.

#### Se você quiser mesmo treinar com 4 s

Aí o online precisa de um ring buffer de 4 s. É possível, mas:

- o modelo processa 2× mais amostras;
- a resposta fica mais “lenta” para mudanças recentes;
- para MI, que é lenta, pode até ajudar;
- para controle fino, 2 s costuma ser suficiente.

Não misture: **treinar em 4 s e usar 2 s no online é erro de distribuição.**

#### Como comparar 2 s vs 4 s cientificamente

Treine dois modelos separados, com o mesmo split:

```text
Modelo A: entrada 2 s, stride 0,1 s
Modelo B: entrada 4 s, stride 0,1 s
```

Compare posição, orientação e classificação. O trial de 4 s é o mesmo para os
dois; muda só o recorte de entrada.

## 6. Camada 5 — inferência online

### 6.1. Loop

```text
a cada 33 ms (30 Hz):
  lê EEG mais recente
  lê câmera/visão (não para o label; só para monitorar/qualidade)
  mantém ring buffer de EEG dos últimos 2 s
  aplica filtro causal + CAR + z-score (stats do treino)
  monta X (1, C, 1000)
  modelo -> posição, rotação, abertura
  suaviza / valida / converte para comando
  envia ao robô
```

### 6.2. Predição e atraso

O modelo pode prever:

- **pose atual** (mais simples): o atraso do sistema aparece;
- **pose futura** em `t + h` (h = 100–300 ms): compensa atraso;
- **sequência futura** (1 s): permite receding horizon control.

Para começar: prever a sequência de 1 s e usar só o primeiro passo (ou uma
média dos primeiros 100 ms) para o robô.

### 6.3. Suavização e segurança

- quaternion: use **SLERP**, nunca média aritmética;
- posição: filtro exponencial ou Kalman;
- abertura: saturação e taxa máxima;
- rotação: limite de velocidade angular;
- se `mask=0` ou incerteza alta: mantenha última pose/abertura por pouco tempo
  e desacelere o robô.

### 6.4. O que não fazer no online

- não usar `filtfilt` (zero-fase) no online: ele usa o futuro;
- não normalizar por estatística da janela se o treino usou estatística do
  dataset;
- não usar label de câmera para “corrigir” a saída se o objetivo é EEG-only;
- não trocar o frame do alvo entre treino e online.

## 7. Pipeline de arquivos

```text
dados_pose/ (cru, schema)
        ↓  scripts de sincronização + labels
dados_pose_processados/
  S01_2026-11-10_01/
    eeg.npz
    pose.npz
    sync.json
        ↓  janelamento
dados_treino/
  windows.npz
  windows_index.csv
  windows_meta.json
        ↓  treino
sand_pose_piloto.pt
        ↓  online
buffer circular + modelo + suavização + robô
```

Versione os **scripts e o schema**, não os dados brutos no Git.

## 8. Separação de dados (split)

Para o piloto:

```text
sessão 1 -> treino
sessão 2 -> treino
sessão 3 -> validação
```

Depois:

```text
LOSO: leave-one-subject-out
```

Nunca divida frames aleatórios da mesma sessão: janelas vizinhas são quase
idênticas e inflam o resultado.

## 9. Quantidade de dados

Uma sessão de 15 min:

- 15 min × 60 s × 30 labels/s = **27.000 labels**;
- com stride de 0,1 s = **9.000 janelas**;
- 3 sessões ≈ 27.000 janelas.

Parece muito, mas a variabilidade de sujeito/sessão é o gargalo. Para o piloto,
priorize **cobertura** (poses, alturas, rotações, repouso, dinâmica) em vez de
minutos repetitivos parados.

## 10. Roadmap de modelos

1. **Posição only** (SAND atual): controle de trajetória.
2. **Posição + orientação 6D** (SAND-Pose): garra rígida.
3. **+ abertura** (garra com 1 GDL).
4. **+ juntas dos dedos** (mão robótica completa).
5. **IMU como entrada auxiliar** (não como label; sem T/f).
6. **Online com predição + suavização** (receding horizon).

Cada etapa deve ser comparada com a anterior no mesmo split. Não pule para a
etapa 4 antes de a 2 funcionar.

## 11. Checklist do iniciante

- [ ] uma sessão crua bem gravada;
- [ ] sincronização conferida com um evento;
- [ ] labels de pose com `pct_with_quat >= 0,80`;
- [ ] janelas geradas com máscara;
- [ ] split por sessão;
- [ ] baseline posição-only rodando;
- [ ] posição + orientação rodando;
- [ ] abertura rodando;
- [ ] online consumindo o mesmo pré-processamento do treino;
- [ ] latência medida de ponta a ponta;
- [ ] só então pensar em dedos individualmente.

## 12. O que responder antes de coletar

1. O alvo do piloto é **concorrente** (offline) ou **preditivo** (online)?
   Recomendo começar concorrente.
2. A garra tem abertura proporcional ou só classes?
3. O frame `base` será o tronco ou a base do robô?
4. A câmera vai ficar fixa ou no robô?
5. O controle vai usar a predição como posição absoluta ou delta?

Responda essas cinco antes de gravar; elas mudam o significado dos labels.
