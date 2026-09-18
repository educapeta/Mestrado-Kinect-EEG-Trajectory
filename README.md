# Mestradopy — Ground truth de mão (Kinect v2 + webcams + IMU) com EEG g.Nautilus

Pipeline de aquisição **offline** (ME/MI com 6 condições objeto×mão) e sistema **online**
(trajectória prevista pela rede SAND sobreposta ao vídeo), para a sessão de 12 s/trial:

```
[Repouso 2 s] → [Cue + ME 4 s] → [Pausa + vídeo 0,5× 2 s] → [MI 4 s]
```

## Programas

| Programa | Arquivo | O que faz |
|---|---|---|
| **1 — Aquisição offline** | `eeg_motor_paradigm.py` | EEG cru (500 Hz) + movimento (~30 Hz) + marcadores + vídeo de priming + questionário do participante |
| Treino SAND trajetória | `sand_traj_treino.py` | Treina a rede (EEG 2 s → trajetória 30×3 em metros) com as gravações do Programa 1 |
| **2 — Tempo real** | `sand_traj_tempo_real.py` | Inferência assíncrona + overlay da trajetória prevista no vídeo |
| Calibração stereo | `stereo_calibration.py` | Kinect ✕ webcam auxiliar N (`--aux-index`); gera `stereo_calibration_auxN.npz` |
| Ground truth autônomo | `kinect_imu_groundtruth.py` | Ferramenta de calibração/diagnóstico standalone (uso pontual) |

Biblioteca compartilhada: `kinect_imu_groundtruth.py` (stereo/triangulação, MediaPipe,
IK de 2 elos `ArmLinkModel`, esqueleto do SDK) e **`imu.py`** (IMU do ESP32 por UDP +
`PositionFusion` com zeragem — **fonte única**, importada pelo módulo do Kinect).
Modelo: `sand_trajectory_model.py`.

## Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Requisito externo: Kinect for Windows SDK 2.0 + g.HIcopy/g.HIldriver (g tec USB)
```

**Importante:** o pacote `gtec_gds` só inicializa o GDS na Python do `.venv`
(instalação 1.6.0 com cdefs compilados). A Python do PATH falha — o programa
detecta e orienta trocar de interpretador.

## Layout de câmeras

```
        [Kinect]   ← centro (logo acima/abaixo da tela do laptop)
        /      \
  [webcam esq]  [webcam dir]   ← ±45°, elevadas, baseline 0,6–1,2 m do Kinect
```

Nunca posicione um par em 180° (cada câmera veria uma face diferente do tabuleiro).

```powershell
python -c "import cv2;[print(i, cv2.VideoCapture(i).isOpened()) for i in range(4)]"
python stereo_calibration.py --aux-index 1   # par Kinect ✕ webcam 1
python stereo_calibration.py --aux-index 2   # par Kinect ✕ webcam 2
python eeg_motor_paradigm.py --aux-cameras "1,2"
```

Durante a calibração, oclusão pontual só descarta a captura (colete 15+ pares bons).

## Execução (Programa 1)

```powershell
python eeg_motor_paradigm.py --participante P01 --sessao S1 --montagem "C3,Cz,C4,P3,P4"
# úteis: --source generator (teste sem hardware) · --sem-kinect · --sem-zeroing
#         --aux-cameras "1,2" · --tela-cheia · --sem-priming-arquivo
#         --sem-painel-impedancia · --sem-questionario · --mov-hz 30
#         --mov-no-eeg-csv (formato antigo: movimento embutido no CSV de EEG)
```

Fluxo da sessão: PREPARAÇÃO (painel de impedâncias ao vivo + calibração do
Kinect + teste de sinal → ESPAÇO)
→ BASELINE (2 min olhos abertos, 2 min fechados, 1 min repouso ativo)
→ calibração de origem (10 s, mão na mesa = 0,0,0; também mede os elos do braço)
→ 5 blocos × 50 trials (12 s) com pausa de 2 min entre blocos (ESPAÇO pula)
→ encerramento com questionário do participante.

Enquanto a sessão roda: a janela de tracking mostra a fase, a sobreposição da
mão auxiliar e **"LINK EEG PERDIDO"** se o amplificador parar de enviar amostras
(o vigia também grava 899/898 no CSV).

## Layout do projeto

```
*.py                  programas e bibliotecas (rodam da raiz)
imu.py                IMU do ESP32 (UDP) + PositionFusion: FONTE UNICA
test_*.py             suites de teste (sem hardware)
_smoke_*.py           testes ponta-a-ponta sintéticos
_demo_trial.py        demo da plataforma gráfica com a webcam
*.npz / *.json        calibrações: stereo_calibration_auxN, camera_alignment,
                      hand_landmark_calibration, arm_model, overlay_trim
*.task / *.pt         modelos (MediaPipe e checkpoints do SAND)
gravacoes/            sessoes gravadas (ignorado pelo git; --pasta-sessoes)
data/                 dados antigos/BCI IV e resultados de treino anteriores
results/              saídas de execuções de teste
docs/                 histórico, críticas e documentos de decisão
                      (ESTADO_DA_ARTE_JANELAS.md, ARQUITETURA_ONLINE_JANELA_
                      HORIZONTE_E_MCU.md, PRE_TREINO_WAY_EEG_GAL.md,
                      ANALISE_PAPER_JANELA_ALVO_E_VIDEO.md,
                      MIGRACAO_FORA_DO_ONEDRIVE.md, PUBLICAR_NO_GITHUB.md)
tools/                utilitários e a ferramenta autônoma do Kinect
                      (kinect_groundtruth_tool.py: calibração/diagnóstico,
                      gerar_sessao_sintetica.py: piloto sem hardware,
                      compara_alvos.py: baseline do alvo medio,
                      mede_custo_modelo.py: FLOPs/tempo por janela,
                      analisa_pdfs_janelas.py: extracao dos artigos,
                      inventario_way_eeg_gal.py, publicar.ps1, limpeza_admin.ps1)
gravacoes_sinteticas/ sessoes FICTICIAS do piloto (ignorado pelo git)
legacy/               zip do código legado (subir no Drive manualmente)
```


## Dados gerados (por sessão, em `gravacoes/` — mude com `--pasta-sessoes`)

| Arquivo | Conteúdo |
|---|---|
| `gravacao_MEMI_<stamp>.csv` | **EEG cru** (µV, sem filtros) a 500 Hz: `Time, EEG_Ch01..32, Marker` — 34 colunas, ~281 B/amostra (≈440 MB por sessão de 50 min) |
| `gravacao_MEMI_<stamp>_movimento.csv` | Movimento a 30 Hz (45 colunas): `t_mono_s, t_epoch_s` (**absolutos**) + as 43 de `MOTION_COLUMNS` (KT_* da mão ativa, IMU_* roll/pitch/yaw + posição fusionada, ZERO_lock, KTT_valid, ARM_* da IK, PALM_* da palma, KT_hand, KT_src, KT_onset) |
| `gravacao_MEMI_<stamp>_eventos.json` | Marcadores com a **amostra exata** do EEG + metadados (`meta`: participante, sessão, unidades, versões dos pacotes, **sha1 das calibrações**, ordem dos trials, tempos de fase) |
| `gravacao_MEMI_<stamp>_participante.json` | Questionário do fim da sessão (ID, idade, sexo, dominância, sono, cafeína, observações) |
| `gravacao_MEMI_<stamp>_priming/tNNN_<condicao>.avi` | Clipe de priming 0,5× com a trajetória 3D sobreposta (~200–400 kB/trial) |

**Sincronização EEG↔movimento:** cada evento do JSON traz `sample_index` (índice
exato da amostra de EEG) **e** `monotonic_s`; o CSV de movimento traz `t_mono_s`
no mesmo relógio (`time.monotonic`). Dois eventos definem a reta
amostra↔relógio — alinhe por ela, não por suposição de taxa.

**Colunas de diagnóstico do movimento:** `KT_hand` (0 desconhecida, 1 direita,
2 esquerda — lateralidade da mão efetivamente rastreada) e `KT_src`
(0 sem medida, 1 triangulado, 2 depth do Kinect, 3 esqueleto do SDK,
4 modelo, 5 amostrada, 6 última válida, 7 nominal, 8 IMU). Filtrar por
`KT_src <= 3` mantém apenas amostras de **medida real**.

## Marcadores (1 amostra de hold = 0,1 s @ 500 Hz)

| Código | Evento | Código | Evento |
|---|---|---|---|
| 32766 / 32765 | início / fim da sessão | 768 | início do trial (repouso/home) |
| 276 / 277 / 784 | baseline olhos abertos / fechados / repouso ativo | 769 / 770 | cue mão esquerda / direita |
| 790 / 791 | início / fim de bloco | 778/779, 780/781 | início/fim ME, início/fim MI |
| 792 / 793 | início / fim da pausa | 811–816 | condição (objeto × mão) |
| 794 | início do vídeo 0,5× | **795** | **início do movimento (onset detectado pelo Kinect)** |
| 500 / 501 | início / fim da calibração de origem | 898 / 899 | link EEG recuperado / **LINK EEG PERDIDO** (vigia) |

## Treino do SAND de trajetória (como recortar a janela e o alvo)

A aquisição marca o **início do movimento** (marcador 795 + coluna `KT_onset`), o
que permite ancorar a janela de EEG no **planejamento motor** (o planejamento
começa 500–1000 ms antes do movimento visível; a organização da sequência,
~500–350 ms antes). Por isso a janela recomendada começa **0,5 s antes** do onset:

```powershell
# Alvo CONCORRENTE, janela ancorada no planejamento (recomendado hoje):
python sand_traj_treino.py --data gravacoes --event-code 795 `
       --window-start-sec -0.5 --window-sec 2.0

# Alvo PREDITIVO (o que o controle de prótese exige: prever o futuro):
python sand_traj_treino.py --data gravacoes --event-code 795 `
       --window-start-sec -0.5 --window-sec 2.0 `
       --target-start-sec 1.5 --target-end-sec 2.5

# Com REGULARIZADOR ANATÔMICO (envelope de alcance + limites do cotovelo):
python sand_traj_treino.py --data gravacoes --event-code 795 `
       --window-start-sec -0.5 --window-sec 2.0 `
       --peso-anatomico 0.1 --peso-angulo-cotovelo 0.005
```

Sem `--target-start-sec` o alvo é **concorrente** (descreve a própria janela);
com ele, o alvo passa a ser a trajetória **futura** — e o checkpoint registra
`target_concurrente`/`target_start_sec` para a análise.

### Piloto sem hardware (ondas fictícias)

Para ensaiar/validar o pipeline inteiro (formato dos arquivos, marcadores, janelas,
alvos, treino, checkpoint, inferência) antes de ter o g.Nautilus em mãos:

```powershell
# sessões fictícias no formato real (CSV + _movimento.csv + _eventos.json)
.venv\Scripts\python.exe tools\gerar_sessao_sintetica.py --sessoes 6 --trials 24
# treino em cima delas (mesmos comandos do treino real, só muda --data)
.venv\Scripts\python.exe sand_traj_treino.py --data gravacoes_sinteticas `
       --event-code 795 --window-start-sec -0.5 --window-sec 2.0
# baseline trivial (trajetória média, sem EEG): obrigatório para ler qualquer PCC
.venv\Scripts\python.exe tools\compara_alvos.py --data gravacoes_sinteticas --max-sessoes 2
```

O EEG sintético traz ERD de mu/beta, MRCP 1,2 s antes do onset e um ganho por
canal que codifica a lateralidade do alvo; o movimento é mínimo-jerk com onset
aleatório 0,35–0,90 s após a cue. Para rodar o **paradigma** sem EEG/Kinect:
`python eeg_motor_paradigm.py --source generator --sem-kinect --sem-questionario`.
Resultados e limites declarados: `docs/ESTADO_DA_ARTE_JANELAS.md` (seção 5).

## Testes

```powershell
.venv\Scripts\python.exe tools\roda_testes.py   # roda todas e resume (OK/FALHA)
foreach ($t in Get-ChildItem test_*.py) { python $t.Name }
python _smoke_eeg.py        # sessão sintética ponta-a-ponta
python _smoke_sand_traj.py  # treino + tempo real em modo headless
python _demo_trial.py       # demo da plataforma gráfica com a webcam
```

| Suite | Cobre |
|---|---|
| `test_paradigm_protocol.py` | trials balanceados, trial de 12 s, códigos únicos com legenda, contrato de colunas (EEG 34 / movimento 44), vigia 899/898, painel de impedâncias, questionário, `KT_src` |
| `test_arm_csv.py` | colunas ARM_*/PALM_*/KT_hand/KT_src: ordem, unidades, relatividade à origem |
| `test_arm_ik.py` | cinemática inversa de 2 elos (8 casos) |
| `test_body_anchor.py` | esqueleto do SDK como âncora de profundidade (6 casos) |
| `test_cameras_multi.py` | calibração por índice de webcam e lista personalizada |
| `test_fusion_zeroing.py` | zeragem do IMU com as câmeras (bias/velocidade/posição) |
| `test_triangulation.py` | triangulação estéreo 3D da mão (mm) |
| `test_hand_calib.py` | calibração 2D→2D da mão (sintética) |
| `test_stereo_solve.py` | reparo da ordem dos cantos do tabuleiro no stereo |
| `test_sand_traj.py` | rede/alvo/normalizador do SAND de trajetória |
| `test_move_onset.py` | detector de início do movimento (`MovementOnsetDetector`) e coluna `KT_onset` |
| `test_treino_loader.py` | leitura dos arquivos no treino: X (N, canais, amostras), montagem, CAR/z-score, 1 época, checkpoint + `SandTrajectoryBCI`, alvo concorrente × preditivo |
| `test_anatomical_reg.py` | regularizador anatômico: lei dos cossenos, consistência com a IK do projeto, envelope + gradiente, ângulo medido, limites articulares, NaN seguro, colunas `ARM_*` no caminho de arquivo, treino com o termo |
