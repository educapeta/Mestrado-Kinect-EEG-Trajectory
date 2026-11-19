# Gravacao e processamento OFFLINE dos videos das 2 auxiliares

**Data:** 17/11/2026
**Pipeline:** aquisicao "burra" (grava video) -> processamento offline
(MediaPipe 2D -> triangulacao 3D), sem rodar MediaPipe/HandMvNet durante a
gravacao. O Kinect continua online em outro programa como referencia, mas nao e'
necessario para gravar nem para processar o par de auxiliares.

| Etapa | Ferramenta | O que faz |
|---|---|---|
| 1. Gravar | `src/mestradopy/tools/handmvnet_video_record.py` | abre aux0/aux1, grava MP4/AVI em disco configuravel, timestamp por frame, `meta.json`, checagem de espaco, sync por flash/clap |
| 2. Processar | `src/mestradopy/tools/handmvnet_video_process.py` | le os videos, roda MediaPipe offline, sincroniza as duas timelines, triangula o par aux (e opcionalmente Kinect+aux), grava `landmarks_3d.csv/npz` + relatorio de cobertura/erro |
| Biblioteca | `src/mestradopy/vision/handmvnet_video_io.py` | formato da sessao, CSV de timestamps, flash, sincronismo, espaco/codecs (sem duplicar codigo entre as duas ferramentas) |
| Teste sintetico | `tests/manual/test_handmvnet_video_offline.py` | 73 checks sem hardware (cameras fake + MediaPipe fake + calibracoes reais) |

**Nada de EEG foi tocado**; nenhum arquivo de `src/mestradopy/eeg/`,
`outputs/eeg/`, `docs/thesis/` ou `data/` foi alterado.

---

## 1. Por que offline?

O MediaPipe online disputa CPU/GPU com o Kinect e o tracking cai justamente
durante a aquisicao. Gravando as duas auxiliares em disco (so' `read()` +
`VideoWriter`, nenhuma inferencia), a aquisicao fica leve e o processamento
pesado (MediaPipe 2D, triangulacao) roda depois, em lote, quantas vezes for
preciso. Vantagens praticas:

- da' para reprocessar a MESMA sessao com parametros diferentes (confianca do
  MediaPipe, gate de reprojecao, `--every`, filtros de tempo);
- o experimento nao perde frames por causa de uma inferencia lenta;
- o video fica auditavel (dá para revisar visualmente e recalibrar/refinar).

Custo: armazenamento (secao 3) e um passo extra antes da analise.

---

## 2. Formato da sessao em disco

```
<dest>/<sessao>/
    meta.json                    # cameras, resolucao, fps, codec real, sync, espaco
    aux00.mp4 (ou .avi)          # video da auxiliar 0 (frames espelhados -> fisicos)
    aux01.mp4 (ou .avi)          # video da auxiliar 1
    timestamps_aux00.csv         # uma linha por frame gravado
    timestamps_aux01.csv
```

`timestamps_auxNN.csv` (gerado por frame; colunas):

| coluna | conteudo |
|---|---|
| `frame` | indice do frame no video (0-based) |
| `wall_s` / `wall_iso` | epoch (`time.time`) da leitura do frame |
| `monotonic_s` | `time.monotonic()` -- sincroniza as DUAS cameras do MESMO processo |
| `perf_s` | `time.perf_counter()` (diagnostico) |
| `pos_msec` | `CAP_PROP_POS_MSEC` do driver (pode ser 0/mentir) |
| `brightness` | brilho medio 0-255 (deteccao do flash) |
| `flash` | 1 no frame em que o flash foi detectado |
| `manual_sync` | 1 no frame marcado pela tecla **S** |

`meta.json` inclui (`schema_version=1`): `cameras[]` (indice, nome do
Windows/UVC, video, timestamps, codec real, resolucao, fps pedido/medido,
frames, calibracao encontrada e RMS), `sync_events[]` (flash/manual com frame e
`t_monotonic`), espaco livre inicio/fim, estimativas de GB/h e `status`.

Convencao de imagem: o driver da webcam entrega o stream **espelhado**; o
gravador aplica `transform_auxiliary_frame` (flip) e grava a imagem **fisica**,
que e' a convencao da calibracao (`stereo_calibration*.npz`). Por isso o
processador NAO espelha de novo (`meta.mirror=true`). Se o video veio de fora,
use `--mirror on/off` no processador.

---

## 3. Gravacao

### Comandos

```powershell
# ver configuracao sem abrir camera (indices, calibracao, espaco, estimativa)
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_video_record.py --dry-run

# gravar 2 minutos em F:\handmvnet_videos (default se F: existir)
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_video_record.py --duration 120

# escolher cameras/resolucao/codec e destino
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_video_record.py ^
    --dest F:\handmvnet_videos --aux-indices 0 1 --width 1280 --height 720 --fps 30 ^
    --codec auto --duration 120

# bancada remota (sem janela); tecla S nao existe -> use flash automatico
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_video_record.py --no-window --duration 60
```

Destino default: `--dest` > env `HANDMVNET_VIDEO_DIR` >
`F:\handmvnet_videos` (se o disco existir) > `outputs/handmvnet_videos`.

Teclas na janela: **S** = marca sincronismo manual na proxima frame das duas
cameras; **Q/ESC** = encerrar. `--countdown 3` faz contagem regressiva com
preview antes de comecar (aquecimento das cameras).

### Codecs e armazenamento (2 cameras)

| Codec | Extensao | Estimativa 720p30, 2 cameras |
|---|---|---|
| H.264 (`avc1`) | `.mp4` | **~3-7 GB/h** (a tarefa estimou 2-4 GB/h por camera em H.264) |
| MPEG-4 (`mp4v`) | `.mp4` | ~5-10 GB/h |
| MJPEG | `.avi` | ~35-70 GB/h |

O tool tenta H.264 -> MP4V -> MJPEG e grava no `meta.json` qual abriu de
VERDADE (no Windows o H.264 depende do build de OpenCV/FFmpeg; quando nao ha'
codec, `isOpened()` e' False e ele cai para o proximo). **Se cair em MJPEG o
disco enche ~10x mais rapido**: no inicio o tool imprime estimativa e o
`--dry-run` mostra antes. Regra pratica: libere no minimo
`1.5 x estimativa` do codec que abrir + a reserva (`--min-free-gb`, default
2 GB). A checagem e' refeita a cada ~2 s e a gravacao para com `status=
space_low` se o livre cair abaixo da reserva.

Outras protecoes: `--duration`/`--max-frames` para parar sozinho; erro de
leitura persistente (driver caiu) encerra com `status=camera_error`; o
`finally` sempre fecha os writers e grava o `meta.json` (mesmo em erro).

### Exposicao / UVC

Por padrao congela exposicao/foco (mesmo `camera_uvc.trava_exposicao` do
projeto) e pede MJPEG na abertura (720p30 nas duas cameras da bancada).
`--no-exposure-lock` desliga; `--exposicao -7` / `--ganho N` forcam valores do
driver; `--no-mjpeg` desfaz o pedido de MJPEG. O relatorio UVC vai no
`meta.json` (`uvc`).

### Sincronismo (flash / clap / tecla S)

Protocolo de bancada:

1. inicie a gravacao (com `--countdown` a janela mostra a contagem);
2. ~2 s depois, da' **um flash** (lanterna/celular) ou **clap** na frente das
   DUAS cameras -- o salto de brilho e' detectado por camera e vira um evento
   `flash` no CSV/meta;
3. repita o flash 2-3x no inicio (mais eventos = offset mais robusto);
4. se quiser marcar um instante especifico (ex.: palma), aperte **S**;
5. encerre (duracao ou Q/ESC) e processe.

Precisao: com 1 evento o alinhamento fica em ~1 frame; com 2-3 eventos o
processador usa a mediana dos offsets (mais estavel). Sem flash/clap nas duas
cameras, o processador cai no alinhamento por inicio simultaneo
(`method=lockstep_start`) e avisa no relatorio.

---

## 4. Processamento offline

### Comandos

```powershell
# 1) checagem rapida: sync, pares, calibracao, resolucao (SEM MediaPipe)
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_video_process.py ^
    --session F:\handmvnet_videos\aux2v_20261117_140000 --dry-run

# 2) processar o par aux0+aux1 (caminho padrao) -> <sessao>\processed\
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_video_process.py ^
    --session F:\handmvnet_videos\aux2v_20261117_140000

# 3) opcional: Kinect + aux0 + aux1 (3 vistas). Precisa do video do Kinect e
#    de um CSV com frame,monotonic_s (ou o CSV do proprio recorder).
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_video_process.py ^
    --session <sessao> --mode three ^
    --kinect-video C:\dados\kinect.mp4 --kinect-timestamps C:\dados\kinect_timestamps.csv

# 4) opcional: estagio HandMvNet (requer --mode three + checkpoint; ver
#    docs/guias/HANDMVNET_INTEGRACAO.md; sem pesos ele apenas registra o motivo)
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_video_process.py ^
    --session <sessao> --mode three --kinect-video ... --kinect-timestamps ... ^
    --handmvnet auto
```

Flags uteis: `--out <pasta>`, `--every N` (1 a cada N pares),
`--start-s/--max-seconds`, `--max-sync-dt-ms 34`, `--max-reproj-px 10`,
`--mediapipe-gpu`, `--mirror auto|on|off`, `--aux-indices 0 1`, `--json`.

### O que o processador faz (e qual codigo reutiliza)

1. le `meta.json` + os dois videos + CSVs de timestamps;
2. carrega a calibracao canonicamente (`kinect_imu_groundtruth.CameraAlignment`)
   e valida `stereo_usable`/RMS das duas auxiliares;
3. **sincroniza as timelines** pelos eventos `flash`/`manual_sync` (mesmo
   `time.monotonic` do processo de gravacao) ou, sem eventos, pelo inicio
   simultaneo; estima o `offset` (`t_b + offset ~= t_a`) e casa frames com
   `pair_by_time` (1:1, tolerancia `--max-sync-dt-ms`);
4. **se a resolucao do video diferir da resolucao calibrada**, reescala os
   pixels dos landmarks para a resolucao do `npz` antes da geometria (avisa no
   log; equivalente a gravar na resolucao calibrada);
5. roda o **MediaPipe HandLandmarker offline** por vista, com o MESMO
   `criar_hand_landmarker` do pipeline online e a lateralidade corrigida por
   `aux_handedness_flip` (mesma convencao; por isso o resultado e' comparavel ao
   online). Um detector por vista (VideoMode nao e' reentrante);
6. para cada par + cada mao presente nas duas vistas:
   - triangula com **`triangulate_pair_points`** (a funcao canonica do projeto:
     undistort + DLT + retorno ao frame do Kinect via `relative_transform`);
   - calcula erro de reprojecao por junção com **`erro_reprojecao_px`** e
     aplica o gate (`--max-reproj-px`) + profundidade positiva;
   - no `--mode three`, usa **`triangulate_three_views`** (Kinect + aux0 + aux1)
     com o mesmo gate; a funcao pode cair no melhor par por frame (por design);
7. grava saidas (abaixo) e o relatorio de cobertura.

**Nao ha' matematica duplicada**: triangulacao, undistort, reprojecao e
composicao de calibracoes sao as funcoes do `kinect_imu_groundtruth.py`. A
classe `HandMvNetView` (do adaptador HandMvNet, tarefa anterior) e' reutilizada
em `make_handmvnet_views(...)` para o estagio opcional de modelo -- os mesmos
frames/deteccoes viram o contrato do `HandMvNetAdapter`.

### Saidas (`<sessao>\processed\`)

| arquivo | conteudo |
|---|---|
| `landmarks_3d.csv` | formato LONGO: 1 linha por (par de frames, mao, junção) com `x_m,y_m,z_m`, reproj por vista, `valid`, `method` |
| `landmarks_3d.npz` | arrays: `joints_xyz (N,21,3)` (m, frame do Kinect), `valid (N,21)`, `reproj_aux0/aux1 (N,21)`, `reproj_kinect` (modo 3 vistas), `pixels_aux0/aux1`, `undist_aux0/aux1`, `frame0/frame1/frame_kinect`, `time0_s/time1_s/time_sync_s`, `hand`, `method`, `meta_json` |
| `coverage_report.json` | cobertura (pares/frames/maos/junções), erro de reprojecao (mediana/p90 por vista e por lado), sync (metodo/offset/residual), calibracao, tempos, limitacoes |
| `coverage_report.txt` | resumo legivel do mesmo relatorio |
| `meta_processado.json` | parametros do processamento + resumo (auditoria/reprodutibilidade) |

Interpretacao rapida: `counts.pairs_total` vs `pairs_written` = quantos pares
viraram medição; `quality.reproj_*` e `quality.joint_coverage` dizem a
qualidade; `sync.method/offset_ms/residual_ms` dizem se o alinhamento e' de
confianca. Se `method != flash/manual`, alinhe com flash na proxima sessao.

---

## 5. Teste sintetico (sem hardware)

```powershell
.\.venv\Scripts\python.exe tests\manual\test_handmvnet_video_offline.py
```

Resultado esperado (17/11/2026): **`HANDMVNET_VIDEO_OFFLINE_OK: 73/73 checks`
(~6 s)**. O teste:

- gera uma mao 3D sintetica movendo-se e projeta-a com as calibracoes REAIS
  (`stereo_calibration.npz` + `stereo_calibration_aux1.npz`, inclusive
  distorcao) para as duas auxiliares e para um Kinect fake;
- grava com cameras FAKE usando o **gravador real** (video MJPG, CSV, meta,
  flash no frame 8, aborto por falta de espaco);
- monta sessoes com timestamps deterministicos (pareamento exato) e injeta um
  detector FAKE (MediaPipe "perfeito", ruido 0.2 px), cobrindo: plano/dry-run,
  sincronismo por flash, pares com mao so' numa vista / sem mao, triangulacao
  par (**erro mediano 0.79 mm** vs ground truth), tres vistas com Kinect fake
  (**0.80 mm**), CSV/NPZ/relatorio, `every`/filtros, HandMvNet auto sem
  checkpoint (pula com motivo) e a cadeia completa gravacao->processamento
  (mao parada, 0.78 mm).

O que o teste NAO cobre (limites honestos): detector MediaPipe real (o teste
injeta landmarks perfeitos; o caminho real usa `criar_hand_landmarker` do
projeto), cameras fisicas/USB, escrita H.264 (no teste forca MJPEG que sempre
existe) e desempenho/FPS em sessao longa.

---

## 6. Limitacoes e riscos

1. **Sincronismo**: 1 flash da' alinhamento de ~1 frame; 2-3 flashes no inicio
   melhoram. `time.monotonic` vale para as duas cameras do MESMO processo de
   gravacao; para o Kinect (outro processo) o pareamento usa relogio de
   parede/offset manual (`--kinect-offset-s`) -- confira `--dry-run`.
2. **Sem Kinect no modo par**: a pose 3D e' do par auxiliar (triangulacao),
   sem correcao por profundidade do Kinect. No `--mode three` o Kinect entra
   como terceira vista (e raiz do frame), mas exige video+timestamps.
3. **Espelhamento**: o gravador grava a imagem FISICA (flip do stream) para
   casar com a calibracao. Video externo cru precisa `--mirror on`.
4. **Resolucao fora da calibrada**: o processador reescala os pixels (e avisa),
   mas o ideal continua sendo gravar na resolucao da calibracao (1280x720 das
   auxiliares; o tool avisa se a camera entregar outra).
5. **Codec**: se cair em MJPEG o disco enche ~10x mais rapido; confira o codec
   real no `meta.json`/terminal (nao confie no que foi PEDIDO).
6. **Espaco livre**: checado antes e durante; o `finally` grava meta mesmo em
   erro, mas um desligamento abrupto do PC pode truncar o ultimo video.
7. **MediaPipe offline** processa os frames em ordem (modo VIDEO), entao o
   tracking e' equivalente ao online em ordem; sem ROI dinamico do pipeline
   online (o processador manda o frame inteiro, como o fallback online).
8. **LGPD/CEP**: os videos mostram a bancada/participante. Grave em disco
   local, nao versione (`outputs/`, `data/` e `external/` sao gitignored) e
   siga o protocolo do projeto para armazenamento/descarte.
9. **Escopos**: o pipeline nao toca EEG/`outputs/eeg`/`docs/thesis`; a
   integracao com a fusao/EEG continua sendo um passo separado (o
   `landmarks_3d.npz/csv` e' a interface).

---

## 7. Arquivos entregues

| Arquivo | O que e' |
|---|---|
| `src/mestradopy/vision/handmvnet_video_io.py` | biblioteca compartilhada (sessao, timestamps, flash, sync, espaco, codecs) |
| `src/mestradopy/tools/handmvnet_video_record.py` | gravador das 2 auxiliares (CLI + nucleo testavel `record_session`) |
| `src/mestradopy/tools/handmvnet_video_process.py` | processador offline (MediaPipe + triangulacao par/3 vistas + relatorio) |
| `tests/manual/test_handmvnet_video_offline.py` | suite sintetica (73 checks, sem hardware) |
| `docs/guias/HANDMVNET_VIDEO_OFFLINE.md` | este guia |

Proximos passos sugeridos ao Lead: (1) revisar/commitar; (2) gravar uma sessao
curta real com flash e rodar `--dry-run` + processamento para medir cobertura e
RMS na bancada; (3) usar o `landmarks_3d.npz` como entrada do experimento
(comparar handmvnet vs triangulacao, alimentar a fusao/EEG); (4) se o Kinect for
usado, definir o formato de exportacao dos timestamps dele no processo de
aquisicao online (a tarefa aceita `frame,monotonic_s`).
