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
| Calibração stereo | `stereo_calibration.py` | Kinect ✕ webcam auxiliar N (`--aux-index 2 0`); gera `stereo_calibration_auxN.npz` |
| Ground truth autônomo | `tools/kinect_groundtruth_tool.py` | ferramenta de calibração/diagnóstico standalone (uso pontual) |

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

**Tabuleiro (calibração):** 6×4 quadrados de **57 mm** = **5×3 cantos internos**
(placa impressa em 23/09/2026). Os valores vivem em `stereo_calibration.py`
(`CHECKERBOARD_COLS/ROWS`, `SQUARE_SIZE_M`) e em `kinect_imu_groundtruth.py`
(`CHECKERBOARD_SIZE`, `SQUARE_SIZE_M`). Se trocar de placa, mude nos dois (ou use
`--checkerboard CxR --quadrado MM`) e **recalibre**: o npz grava placa e quadrado,
e o programa recusa calibração de placa diferente dizendo o motivo
(`stereo_status` → `tools\diagnostico_cameras.py` mostra por câmera).

**Janelas de vídeo:** cada câmera tem a **sua** janela (Kinect RGB, Kinect depth e
**uma por webcam auxiliar**), criadas e posicionadas em **grade sem sobreposição**
por `GradeDeJanelas` + `grade_de_janelas` (em `kinect_imu_groundtruth.py`). A
escala de exibição é automática: o maior valor **≤ 0,5** (`WINDOW_SCALE`, usado
como teto) que faz todas caberem na tela — em 1080p dá 0,5 (Kinect RGB 960×540,
auxiliar 640×360, depth 256×212) e em telas com DPI alto cai para ~0,4 (nesta
máquina o Windows entrega 1536×960 úteis num painel 1920×1080 a 125%). Se a grade
não couber nem no mínimo, as janelas vão em **cascata** (degraus: nenhuma fica
exatamente atrás de outra) e o programa diz isso. `--escala-janelas F` fixa a
escala; `--salvar-quadros S` grava os PNGs. É redução de **exibição** apenas —
detecção, profundidade e calibração seguem na resolução cheia (e o texto de status
é desenhado *depois* da redução, então continua legível).

Com as **duas** auxiliares calibradas contra o Kinect, a relação aux1 ✕ aux2 sai da
composição das duas calibrações (não existe npz de par): é ela que permite ao par
**substituir o Kinect** — quando o Kinect perde a mão, as duas webcams triangulam a
posição 3D nos três eixos (nos programas, `hand_source = trianguladoAux`). E o
esqueleto do Kinect (verde) + o de **cada** auxiliar aparecem no mesmo quadro
(`draw_auxiliary_cameras_on_color`), cada um na sua cor.

```powershell
# 1) QUEM E' QUEM: imprime o NOME (Windows) de cada indice do OpenCV. O indice
#    NAO e' estavel -- ele muda quando uma webcam entra/sai do USB, e foi assim
#    que "a camera auxiliar" virou a webcam do LAPTOP (rosto de quem esta' no
#    laptop na janela de tracking).
python tools\diagnostico_cameras.py --salvar
# 2) calibre CADA webcam EXTERNA no indice que o passo 1 mostrou. UMA SESSÃO DE
#    CAPTURA POR CAMERA, no mesmo comando (15+ poses por camera, tabuleiro PARADO):
#      - tabuleiro: 6x4 quadrados de 57 mm (5x3 cantos) -> ja' e' o padrao dos
#        programas; se usar outra placa, acrescente --checkerboard 5x3 --quadrado 57
#      - 3-8 s de pausa com a placa parada para aceitar cada pose
python stereo_calibration.py --aux-index 2 0
python stereo_calibration.py --aux-index 2   # ou uma camera so'
# 2b) conferiu? cada npz diz a placa que gravou; calibracao de placa diferente e'
#     recusada com o motivo (recalibre) em vez de um "RMS alto" enganoso
python tools\diagnostico_cameras.py
# 3) CONFIRA os TRES esqueletos no video do Kinect (a ativa = --aux-index; as
#    outras em --aux-cameras, vazio = auto = todas menos a do laptop). Cada
#    camera tem a SUA janela de video, em grade (sem uma tapar a outra);
#    --escala-janelas F fixa o tamanho se preferir maior e arrastar.
python tools\kinect_groundtruth_tool.py --aux-index 2 --aux-cameras "0"
# 4) no paradigma o padrao e' `--aux-cameras auto` = todas as webcams MENOS a do
#    laptop, pelo nome; para fixar os indices use --aux-cameras "2,0"
python eeg_motor_paradigm.py --participante P01 --sessao S1
```

Durante a calibração, oclusão pontual só descarta a captura (colete 15+ pares bons).

### Rotina da bancada: o que apertar em cada programa

**`stereo_calibration.py`** (Kinect RGB ✕ cada webcam auxiliar — 1 sessão por
câmera, 15+ poses): a captura é automática, 2 s com o tabuleiro parado em cada
pose aceita, e ele pede sozinho mais poses se o RMS ficar acima de 2 px.
`ESC` encerra a câmera atual (e passa para a próxima, se pediu várias).

**`tools/kinect_groundtruth_tool.py`** — é este que se abre no dia a dia (a
biblioteca `kinect_imu_groundtruth.py` **não tem mais programa próprio**; ela só
é importada). Ordem sugerida:

| Passo | Tecla | O que faz |
|---|---|---|
| 1 | — | abre e já mostra: mapa de câmeras por NOME, estado do stereo (com o motivo) e os esqueletos (Kinect verde + cada auxiliar na sua cor) — e **uma janela de vídeo por câmera**, em grade |
| 2 | `T` | diagnóstico: depth válido/medido **e** erro ponta-a-ponta da sobreposição no tabuleiro |
| 3 | `L` | liga o **log de desvio** (`overlay_deviation_log.csv`, 1 linha/frame com o desvio por nó) |
| 4 | `K` | mede o **trim fixo** do overlay: 3 s com a mão PARADA e visível nas duas câmeras |
| 5 | `H` | **calibração do esqueleto da mão**: segure a mão visível nas DUAS câmeras e VARIE posição, profundidade (0,8–1,5 m) e inclinação; `P` descarta, `H` pausa |
| 6 | `C` | **origem**: mão na mesa = (0,0,0) e mede os elos do braço (deixe ~30 amostras passarem) |
| 7 | `B` | bias do acelerômetro (2 s com a mão parada) |
| 8 | `R` | reset da fusão, se precisar recomeçar |
| 9 | `O` | **alinhamento da palma por luva** (IMU ✕ câmera): ~14 s girando a mão com a palma virada para o Kinect; imprime o resíduo em graus e grava `imu_palm_alignment_{right,left}.json` |

No vídeo aparece, por lado, **`Palma R/L: erro Kinect x IMU`** (e a coluna
`palm_err_deg` no CSV e no log da tecla `L`): é o número que diz se a luva daquele
lado está alinhada. Sem a tecla `O`, o vetor da palma é ancorado numa única amostra
e o erro **cresce conforme a mão gira** — e cresce diferente em cada luva (a
esquerda, montada espelhada, é a que mais aparece). Se o resíduo da `O` ficar alto
em **um** lado, o suspeito é o par de portas: use
`--imu-portas "4210,4211"` (ordem **direita,esquerda**) para casar com o firmware.

`ESC` sai. O CSV `kinect_imu_groundtruth.csv` grava `hand_source`
(`Kinect`/`triangulado`/`trianguladoAux`) e `aux_pair_x/y/z_m` — as colunas que
mostram quando o **par de auxiliares** substituiu o Kinect.

## Execução (Programa 1)

```powershell
python eeg_motor_paradigm.py --participante P01 --sessao S1 --montagem "C3,Cz,C4,P3,P4"
# úteis: --source generator (teste sem hardware) · --sem-kinect · --sem-zeroing
#         --aux-cameras auto (padrao) · --tela-cheia · --sem-priming-arquivo
#         --sem-painel-impedancia · --sem-questionario · --mov-hz 30
#         --mov-no-eeg-csv (formato antigo: movimento embutido no CSV de EEG)
#         --sem-janela-tracking (nao abre a janela do Kinect/OpenCV)
#         --sem-hotkeys · --sem-painel-controle (controle da sessao)
```

Fluxo da sessão: PREPARAÇÃO (painel de impedâncias ao vivo + calibração do
Kinect + teste de sinal → ESPAÇO)
→ BASELINE (2 min olhos abertos, 2 min fechados, 1 min repouso ativo)
→ calibração de origem (10 s, mão na mesa = 0,0,0; também mede os elos do braço)
→ 5 blocos × 50 trials (12 s) com pausa de 2 min entre blocos (ESPAÇO pula)
→ encerramento com questionário do participante.

### Cancelar / controlar a sessão (3 caminhos — `hotkeys.py`)

| Tecla | Ação | Efeito |
|-------|------|--------|
| `ESPAÇO` | `skip` | encerra a espera/pausa atual (preparação, baseline, pausa de bloco) |
| `C` | `cancel_trial` | descarta a trial em andamento (marca **796** no CSV) e vai à próxima |
| `ESC` | `abort` | encerra a sessão: fecha CSV/JSON, imprime o resumo e sai |
| `Ctrl+C` | `abort` | idem ESC (a saída é garantida, com tempo limite por parada) |

As teclas são **globais**: `hotkeys.GlobalHotkeyWatcher` lê o estado do teclado no
Windows inteiro (`GetAsyncKeyState`), então elas funcionam **mesmo com a janela de
vídeo do Kinect (OpenCV/Win32) na frente e com o foco** — era exatamente aí que
ESPAÇO/ESC “morriam” e o trial não andava (bancada de 25/09/2026). Com
`--sem-hotkeys` elas voltam a valer só na janela do Qt. Sem tela cheia aparece
também o **painel do experimentador** (janela pequena sempre no topo, com os
botões pular/cancelar/abortar e o status bloco/trial/fase); `--sem-painel-controle`
o desliga. A saída nunca trava: `CancelamentoGarantido` roda as paradas com tempo
limite (`ABORT_TEARDOWN_SEC`) e termina com `os._exit`. Um 2º ESC/Ctrl+C sai na
hora. `--sem-janela-tracking` evita a janela do OpenCV (não rouba mais o foco;
medição e gravação seguem idênticas).

Enquanto a sessão roda: a janela de tracking mostra a fase, a sobreposição da
mão auxiliar e **"LINK EEG PERDIDO"** se o amplificador parar de enviar amostras
(o vigia também grava 899/898 no CSV).

## Layout do projeto

```
*.py                  programas e bibliotecas (rodam da raiz)
imu.py                IMU do ESP32 (UDP) + PositionFusion: FONTE UNICA
hotkeys.py            teclas GLOBAIS de cancelamento (GetAsyncKeyState) + hard_exit
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
                      HORIZONTE_E_MCU.md, PROTOCOLO_IDLE_VELOCIDADE_E_CONTEXTO.md,
                      IMU_DUPLO_ESP32_DESENHO.md, PRE_TREINO_WAY_EEG_GAL.md,
                      ANALISE_PAPER_JANELA_ALVO_E_VIDEO.md,
                      MIGRACAO_FORA_DO_ONEDRIVE.md, PUBLICAR_NO_GITHUB.md)
tools/                utilitários e a ferramenta autônoma do Kinect
                      (kinect_groundtruth_tool.py: calibração/diagnóstico,
                      gerar_sessao_sintetica.py: piloto sem hardware,
                      compara_alvos.py: baseline do alvo medio,
                      mede_custo_modelo.py: FLOPs/tempo por janela,
                      diagnostico_contexto.py: ocupacao/vazamento de contexto,
                      roda_testes.py: roda todas as suites e resume,
                      teste_imu_dois_esp32.py: bancada das duas luvas (UDP),
                      emissores_imu_falsos.py: luvas falsas para testar sem HW,
                      analisa_pdfs_janelas.py: extracao dos artigos,
                      inventario_way_eeg_gal.py, publicar.ps1, limpeza_admin.ps1)
gravacoes_sinteticas/ sessoes FICTICIAS do piloto (ignorado pelo git)
legacy/               zip do código legado (subir no Drive manualmente)
```


## Dados gerados (por sessão, em `gravacoes/` — mude com `--pasta-sessoes`)

| Arquivo | Conteúdo |
|---|---|
| `gravacao_MEMI_<stamp>.csv` | **EEG cru** (µV, sem filtros) a 500 Hz: `Time, EEG_Ch01..32, Marker` — 34 colunas, ~281 B/amostra (≈440 MB por sessão de 50 min) |
| `gravacao_MEMI_<stamp>_movimento.csv` | Movimento a 30 Hz (61 colunas), tempos **absolutos**: `t_mono_s, t_epoch_s` + KT (3) + blocos `IMU_L_*`/`IMU_R_*` (11 cada: rpy, aceleração bruta em g, posição fusionada, valid, ZERO_lock) + `IMU_hand` + `KTT_valid` + `ARM_*` (18, IK) + `PALM_*` (11) + `KT_hand`, `KT_src`, `KT_onset` |
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

# Alvo de VELOCIDADE (m/s) — o que os trials de repouso/IDLE exigem:
python sand_traj_treino.py --data gravacoes --event-code 795 `
       --window-start-sec -0.5 --window-sec 2.0 --alvo velocidade
```

Com `--alvo velocidade` o modelo prevê **m/s** e o tempo real
(`sand_traj_tempo_real.py`) reconstrói a **posição** integrando a velocidade num
**filtro de Kalman** que usa a **aceleração do MPU6050** como medida direta
(`imu.KalmanTrajectory`); o log mostra `| KF pos=(...) bias=... m/s^2`.

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
| `test_paradigm_abort.py` | cancelamento da sessão: borda das teclas ESPAÇO/C/ESC, vigia global de hotkeys (backend injetado), `hard_exit`, `cancel_trial` (descarta a trial e marca 796; respeita a pausa de bloco), `abort` executado uma vez, painel/fila de hotkeys e `CancelamentoGarantido` que sai mesmo com uma parada travada |
| `test_palm_alignment.py` | vetor da palma por luva: Kabsch/SVD, alinhamento IMU✕câmera resolvido de amostras (montagem até 180°), o bug antigo medido (erro médio 19°, máx. 78° ao girar a mão) × a correção (erro máx. <2°), recusas, JSON (salvar/carregar), captura da tecla `O` na ferramenta e retrocompatibilidade (T=I é idêntico ao código antigo) |
| `test_janelas_grade.py` | janelas de vídeo: uma por câmera, grade sem sobreposição dentro da tela, escala de exibição automática (0,5 em 1080p → ~0,4 em 1536×960), cascata quando não cabe e `GradeDeJanelas` (cria/reposiciona/destrói, com cv2 falso) |
| `test_arm_csv.py` | colunas ARM_*/PALM_*/KT_hand/KT_src: ordem, unidades, relatividade à origem |
| `test_arm_ik.py` | cinemática inversa de 2 elos (8 casos) |
| `test_body_anchor.py` | esqueleto do SDK como âncora de profundidade (6 casos) |
| `test_cameras_multi.py` | calibração por índice de webcam e lista personalizada |
| `test_camera_names.py` | nomes das webcams pelo Windows: mapa índice→nome, detecção da webcam do laptop, degradação silenciosa sem enumeração |
| `test_aux_pair_stereo.py` | par de auxiliares SEM Kinect: relação vinda da composição das duas calibrações, pixels por câmera, triangulação sub-cm, recusas e escolha do melhor par |
| `test_overlay_auxiliares.py` | 3 esqueletos no RGB do Kinect: cor por auxiliar, desvio contra o verde, profundidade rígida, gating de borda |
| `test_paradigm_aux_pair.py` | desvio de fonte do `_resolve_wrist3d`: o par de auxiliares assume com o Kinect cego; os caminhos do Kinect mantêm prioridade |
| `test_fusion_zeroing.py` | zeragem do IMU com as câmeras (bias/velocidade/posição) |
| `test_triangulation.py` | triangulação estéreo 3D da mão (mm) |
| `test_hand_calib.py` | calibração 2D→2D da mão (sintética) |
| `test_stereo_solve.py` | reparo da ordem dos cantos do tabuleiro no stereo |
| `test_sand_traj.py` | rede/alvo/normalizador do SAND de trajetória |
| `test_move_onset.py` | detector de início do movimento (`MovementOnsetDetector`) e coluna `KT_onset` |
| `test_treino_loader.py` | leitura dos arquivos no treino: X (N, canais, amostras), montagem, CAR/z-score, 1 época, checkpoint + `SandTrajectoryBCI`, alvo concorrente × preditivo |
| `test_anatomical_reg.py` | regularizador anatômico: lei dos cossenos, consistência com a IK do projeto, envelope + gradiente, ângulo medido, limites articulares, NaN seguro, colunas `ARM_*` no caminho de arquivo, treino com o termo |
| `test_alvo_velocidade.py` | alvo de velocidade (`--alvo velocidade`): rampa/senoide, punho parado → 0 (caso do trial IDLE), gradiente no caminho de arquivo, treino + checkpoint registrando a formulação |
| `test_kalman_trajectory.py` | filtro de Kalman da trajetória: referência da aceleração (g → m/s², rotação, gravidade, bias), integração trapezoidal, IDLE sem deriva, erro **quadrático** do bias (10 cm/2 s e 2,50 m/10 s na integração dupla × 5,8 cm com o filtro), suavização, degrau, predição |
| `test_imu_bank.py` | dois IMUs (um por mão): parse das portas, convenção dos lados (1=dir/4210, 2=esq/4211), atribuição por porta com dois emissores UDP falsos, fusões independentes, resumo/parada |
| `test_imu_gravacao_dual.py` | gravação SIMULTÂNEA dos dois IMUs: dois emissores UDP falsos com assinaturas distintas (dir roll +5°/1,2 g; esq −5°/1,0 g), `blocos_motion()` por lado, linha do CSV com cada lado no seu bloco e `IMU_hand`, lado ausente em NaN/valid=0 |
