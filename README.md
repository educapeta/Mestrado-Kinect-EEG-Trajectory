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
| Congelamento do Kinect | `freeze_kinect_direto.py` | UMA vez por bancada: 12+ poses do tabuleiro a 0,30–0,45 m (arranjo Kinect-only) e grava `kinect_intrinsics.npz` — só anuncia `MODELO GRAVADO` quando o arquivo existe (`NADA GRAVADO: <motivo>` caso contrário) |
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

**Importante (três coisas, não uma):** o `gtec_gds` só inicializa o GDS na
Python do `.venv` (cdefs compiladas dentro do pacote; a Python do PATH falha);
**e** o g.tec Suite/GDS tem de estar instalado na máquina; **e**, com a Suite
instalada, a pasta `C:\Program Files\gtec\gNEEDaccess` (de onde saem
`GDSClientAPI.dll`/`GDSServer.dll`, que o `gtec_gds` procura **pelo nome**) tem de
estar no PATH **da sessão** — o instalador só mexe no PATH de máquina, então um
terminal/IDE aberto **antes** da instalação falha com a mesma mensagem do "Suite
ausente" (medido em 29/09/2026, g.tec Suite 2024). O programa separa as três
causas na mensagem e **conserta a terceira sozinho** (`GDS_LIBRARY` + PATH na
própria sessão, com aviso no console; o reparo é *preemptivo* porque o wrapper
compilado cacheia a falha na primeira chamada). Não defina `GNEEDACCESS_DIR`:
neste pacote ela faz o wrapper exigir os headers C — o programa a ignora, com
aviso, quando a pasta não tem os `.h`. Com o GDS funcionando e o amplificador
desligado a mensagem é acionável ("NENHUM amplificador..."); sem EEG real a
bancada inteira roda com `--source generator`.

## Layout de câmeras

```
        [Kinect]   ← centro (logo acima/abaixo da tela do laptop)
        /      \
  [webcam esq]  [webcam dir]   ← ±45°, elevadas, baseline 0,6–1,2 m do Kinect
```

Nunca posicione um par em 180° (cada câmera veria uma face diferente do tabuleiro).

**Tabuleiro (calibração):** 8×6 quadrados de **45 mm** = **7×5 cantos internos**
(placa impressa em 01/10/2026; aresta de cantos 270×180 mm, diagonal 324 mm —
1,27× a da placa de 57 mm, o que faz o tabuleiro chegar a 35% da diagonal do
Kinect a ~0,44 m em vez de 0,35 m). Os valores vivem em `stereo_calibration.py`
(`CHECKERBOARD_COLS/ROWS`, `SQUARE_SIZE_M`) e em `kinect_imu_groundtruth.py`
(`CHECKERBOARD_SIZE`, `SQUARE_SIZE_M`) e **os quatro têm de casar**: com o
calibrador numa placa e a biblioteca na outra, todo npz recém-gravado volta
`rejeitada` ("tabuleiro gravado … != …") e o `kinect_intrinsics.npz` deixa de ser
aceito (a sessão volta a re-estimar o Kinect). `test_intrinsecas_kinect.py` trava
essa igualdade. Se trocar de placa, mude nos dois (ou use
`--checkerboard CxR --quadrado MM`) e **recalibre**: o npz grava placa e quadrado,
e o programa recusa calibração de placa diferente dizendo o motivo
(`stereo_status` → `tools\diagnostico_cameras.py` mostra por câmera).

> **O que está em disco agora é a calibração de 01/10/2026** (placa 7×5 @ 45 mm;
> `stereo_calibration.npz` 13:04 com RMS do par 1,39 px e `_aux1.npz` 13:05 com
> 1,01 px, `camera_name` conferido nas duas). O arquivo *versionado no git* é da
> bancada antiga e projetava ~490 px fora do tabuleiro **mesmo passando** no RMS —
> o RMS mede só a coerência *interna* do arquivo. Confirme **sempre** com a tecla
> `T`: deu certo quando a mediana fica abaixo de 25 px e o esqueleto vermelho
> encosta na mão.

Cada npz stereo passa a gravar também o **nome (Windows) da câmera** que o gerou
(`camera_name`): o índice do OpenCV é a *posição* dela na enumeração do USB, então
sem esse campo um arquivo do **outro** par era carregado sem sintoma nenhum — RMS
lindo e a sobreposição centenas de px fora. Ao carregar, o nome gravado é
comparado com o da câmera daquele índice hoje: se forem diferentes (e ambos
conhecidos) o stereo é **recusado** com os dois nomes na mensagem
(`tools\diagnostico_cameras.py`, linha `camera`, mostra `gravado=... | agora=...`).
npz anteriores a 29/09/2026 não têm o campo e seguem aceitos por placa + quadrado
+ RMS, como antes.

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
#    (os indices sao os DESTA maquina: 0 = SIGMA-W780M, 1 = Trust USB -- e o
#    PADRAO ja' e' AUTOMATICO por NOME: sem argumento, calibra as externas)
python stereo_calibration.py                 # as externas desta maquina, uma a uma
python stereo_calibration.py --aux-index 1   # forcando uma so'
# 2b) conferiu? cada npz diz a placa que gravou E o NOME da camera (campo novo);
#     calibracao de placa diferente -- ou de OUTRA camera, quando o indice do
#     OpenCV troca (posicao no USB) -- e' recusada com o motivo por extenso
python tools\diagnostico_cameras.py
# 3) CONFIRA os TRES esqueletos no video do Kinect. SEM `--aux-index`, a camera
#    ATIVA (teclas K/L/H, trim e calibracao do esqueleto) e' escolhida PELO NOME
#    (as externas, as que tem stereo_calibration_aux{N}.npz primeiro); as OUTRAS
#    vem em --aux-cameras (vazio = auto = todas menos a do laptop). Cada camera
#    tem a SUA janela de video, em grade (sem uma tapar a outra);
#    --escala-janelas F fixa o tamanho se preferir maior e arrastar.
python tools\kinect_groundtruth_tool.py
python tools\kinect_groundtruth_tool.py --aux-index 1 --aux-cameras "0"  # forcando
# 4) no paradigma o padrao e' `--aux-cameras auto` = todas as webcams MENOS a do
#    laptop, pelo nome; para fixar os indices use --aux-cameras "1,0"
python eeg_motor_paradigm.py --participante P01 --sessao S1
```

> **A lateralidade das mãos não se adivinha: mede-se.** Com duas webcams de
> **modelos diferentes** (nesta bancada: SIGMA-W780M e Trust USB) o driver de uma
> delas pode já entregar o fluxo **espelhado**; como o rótulo do MediaPipe assume
> imagem espelhada, aquela câmera passa a trocar as mãos — e o casamento **por
> lado** (que liga cada mão ao IMU 4210/4211, ao seu overlay e à sua triangulação)
> fica trocado **só nela**. O programa mede isso pela **geometria** (punho 3D do
> Kinect ✕ **stereo 3D daquela câmera** — a mesma projeção que desenha o esqueleto
> auxiliar no RGB; a homografia 2D da mão fica só de **reserva**, e o motivo diz
> qual via foi usada), mostra no vídeo (`Lateralidade aux N`) e grava a correção em
> `aux_handedness_flip.json` quando há consenso (≥20 frames e ≥90% discordando do
> rótulo) **e o frame é conferível inteiro** (as **duas** mãos vistas pelas duas
> câmeras, todas atribuídas, e erro ≤ 60 px: 120 px ainda decide o quadro, mas não
> serve de prova). Sem consenso, o rótulo é mantido e a linha do vídeo diz **por
> quê**
> (`margem N px` = separe mais as mãos; `projecao a N px` = a projeção caiu longe —
> e o motivo **nomeia a via**: `homografia 2D da mao` é imprecisa por natureza e o
> que falta é o **stereo**; `sem calibracao da mao desta camera` = falta
> `stereo_calibration.py`, ou a tecla `H`, naquela câmera). O veredito sai **uma
> vez por mudança** e no máximo a cada 2 s — o valor em px muda a cada frame e,
> antes, reimprimia as mesmas linhas a sessão inteira.
>
> ```powershell
> .venv\Scripts\python.exe tools\probe_lateralidade.py            # veredito por camera (nao grava)
> .venv\Scripts\python.exe tools\probe_lateralidade.py --gravar   # fixa a correcao aprendida
> .venv\Scripts\python.exe tools\kinect_groundtruth_tool.py --aux-flip-lateralidade 2        # na mao
> .venv\Scripts\python.exe tools\kinect_groundtruth_tool.py --aux-flip-lateralidade limpar   # esquece
> ```
>
> Leitura do probe: `CERTO` = o rótulo concorda com a geometria; `INVERTIDO` = o
> rótulo está trocado (é o bug); `INDECISO` = mãos juntas demais no quadro, projeção
> longe ou duas detecções no mesmo lado — **nenhuma** troca é feita nesse caso. A
> inversão é **por índice** (como as próprias calibrações `stereo_calibration_aux{N}.npz`)
> e se **auto-cura**: se a webcam mudar de porta USB, a geometria percebe e desfaz.
> Aparece também no arranque do tool ("Lateralidade da camera auxiliar N: rotulo do
> MediaPipe INVERTIDO") — de propósito, porque era um bug mudo.

Durante a calibração, oclusão pontual só descarta a captura (colete 15+ pares bons).

### Rotina da bancada: o que apertar em cada programa

**`stereo_calibration.py`** (Kinect RGB ✕ cada webcam auxiliar — 1 sessão por
câmera, 15+ poses): a captura é automática, 2 s com o tabuleiro parado em cada
pose aceita, e ele pede sozinho mais poses se o RMS ficar acima de 2 px.
`ESC` encerra a câmera atual (e passa para a próxima, se pediu várias).

> **O Kinect RGB é UMA câmera: congele o modelo dela UMA vez por bancada.** Com
> o RMS do par em 0,44–0,80 px (dentro do limite de 2 px) o esqueleto da
> auxiliar ainda saía dezenas de px fora da mão no vídeo do Kinect: cada sessão
> re-estimava o modelo do Kinect (fx/fy/cx/cy + distorção) e **três sessões da
> mesma bancada projetavam o mesmo ponto 3D com 24–50 px de diferença** (até
> 176 px na borda). É erro **sistemático**, e o RMS não denuncia: ele mede a
> coerência *interna* de cada arquivo, não qual modelo é o certo.
>
> ```powershell
> # 1) UMA vez por bancada: congele com o arranjo KINECT-ONLY, que leva o
> #    tabuleiro a 0,35–0,40 m (o span tem de ficar >= 35% da diagonal do quadro
> #    do Kinect = 771 px; a 0,45 m ele fica EXATAMENTE no limite).
> #    Pelo caminho do PAR isso NAO sai: quem manda na distancia e' a auxiliar
> #    (ela fecha o quadro primeiro), o tabuleiro fica em 11-13% e o portao
> #    recusa -- foi assim que a bancada ficou sem modelo congelado em 30/09.
> #    12+ poses (junte 15-20: a limpeza de outliers descarta as piores),
> #    variando posicao, distancia e inclinacao; o passo minimo entre poses e'
> #    30% do span (~256 px a 0,40 m = ~10 cm de deslocamento lateral).
> #    O PORTAO OLHA O SPAN *MEDIANO*, nao a melhor pose: TODAS as poses contam,
> #    e uma a 0,5-0,8 m (span ~32-20%) derruba o conjunto -- 18 poses em 05/10
> #    com 6 delas longe deram mediano 34,9% e NADA foi gravado. O console diz,
> #    a cada pose, "span mediano X,Y% (meta 35,0%): n/N poses cumprem" e avisa
> #    ("ATENCAO: pose a X% ...") quando a pose aceita esta' puxando o mediano
> #    para baixo.
> #    O RMS (portao 0,6 px) e' medido por POSE: se ele reprovar, o console diz
> #    QUAIS poses refazer (#8 5,3 px, #13 2,9 px ...) ou que a captura inteira
> #    esta' borrada (luz/apoio), e grava SEM as piores (ate' 4, dizendo quais).
> #    O detector deste arranjo roda a 960 px (escala cheia como rede): o erro de
> #    canto a 427 px, primeiro alvo do padrao do projeto, e' 2,20 px MEDIDO --
> #    do tamanho do proprio portao de RMS. `--largura-deteccao 0` forca a escala
> #    cheia; `--max-descartes 0` desliga o descarte.
> #    A janela ajuda nos três números: distância estimada (`~0,44 m` no alvo,
> #    derivada do span e do fx do congelado), as poses já aceitas (X verde) e
> #    o anel laranja da separação mínima (dentro dele a pose seria "quase
> #    igual" e o portão reprova) — dá para retomar sem repetir pose. Ele só
> #    escreve `MODELO GRAVADO` quando o arquivo existe; se os portões
> #    reprovarem sai `NADA GRAVADO: <motivo>` + "o arquivo de destino NAO foi
> #    tocado" (antes ele dizia "PRONTO" com o portão reprovando e sem gravar
> #    nada — bug da linha 319, coberto por `test_freeze_kinect_direto.py`).
> python freeze_kinect_direto.py
> #    ANTES de montar o tabuleiro, 10 s que economizam a bancada inteira: se o
> #     sensor estiver mudo, o freeze para em `sc.abre_kinect()` e sai
> #     `NADA GRAVADO` DEPOIS de voce ter posicionado 12+ poses a 0,35 m.
> #     `.venv\Scripts\python.exe tools\checa_kinect.py` responde em segundos se
> #     o Kinect ENTREGA frame (saida 0 = pode ir; 1 = NAO monte o tabuleiro;
> #     2 = falta SDK/pykinect2). "Abriu" NAO e' "entrega": em 07/10/2026 o
> #     runtime abria em 0,0 s reportando 1920x1080 e nenhum frame saia em 20 s
> #     (depois do uso do Kinect Studio). `abre_kinect()` agora REABRE o runtime
> #     ate' `TENTATIVAS_KINECT` vezes antes de desistir.
> #    Se o `kinect_intrinsics.npz` em disco for um arquivo que a LEITURA recusa
> #     (o paliativo de 30/09/2026 nao registra `poses`/`span`), o freeze fica
> #     competindo com ele; aposente-o sem apagar nada:
> #     `.venv\Scripts\python.exe tools\conserta_intrinsecas_kinect.py` (dry-run)
> #     e `... --sim` para renomear para `<nome>.arquivado` (que a leitura nao
> #     carrega). `--restaurar` desfaz.
> #    Se o sensor ABRIR e nao entregar frame (o Kinect Studio mostra imagem
> #     e todo programa do SDK fica mudo), o passo que decide e' o
> #     `sensor.IsAvailable` -- `tools\probe_kinect_runtime.py` mostra os
> #     HRESULTs crus, reinicia o servico (`--reiniciar-servico`, elevado) e
> #     observa o IsAvailable enquanto o USB e' desligado/religado (`--observar`).
> #     Medido em 07/10/2026: GetDefaultKinectSensor S_OK, Open() S_OK,
> #     IsAvailable FALSE, eventos de color/depth/body assinados e ZERO frames.
> #     Causa conhecida, com correcao documentada: as MELHORIAS DE AUDIO do
> #     dispositivo de audio do Kinect -- Painel de Controle > Som > Gravacao >
> #     'Xbox NUI Sensor' > Propriedades > Aprimoramentos > desmarcar todas.
> #    (o mesmo portao, dentro da sessao de par, quando o tabuleiro couber nas
> #     duas cameras ao mesmo tempo:)
> python stereo_calibration.py --salvar-intrinsecas-kinect
> # 2) rotina (1 sessão por auxiliar): reusa o modelo congelado, resolve só R/T
> python stereo_calibration.py
> # 3) confira SEM hardware o que ficou (e o que 1 cm de Z custa em px)
> python tools\diagnostico_calibracao.py
> ```
>
> Com o `kinect_intrinsics.npz` presente **e aceito** o modelo do Kinect
> **nunca** é re-estimado (só R/T mudam de sessão para sessão): cada npz do par
> grava `kinect_intrinsics_source` (`congelado`) e a mediana da divergência em px
> do modelo que a sessão estimaria contra o congelado (`kinect_deviation_px`; a
> sessão avisa a partir de 10 px — sinal de tabuleiro sem cobertura do quadro),
> e `tools\diagnostico_calibracao.py` compara cada npz com o congelado e grita
> nos mesmos 25 px da tecla `T`.
>
> Quem decide se o arquivo vale são **poses + cobertura (span) + RMS**, os
> MESMOS portões ao gravar e ao ler (`portoes_do_modelo_kinect`); a **placa** em
> que ele foi gravado virou AVISO, não recusa — as intrínsecas são da CÂMERA, e
> trocar o tabuleiro impresso (5×3/57 mm → 7×5/45 mm, em 01/10/2026) não muda
> fx/fy/cx/cy de ninguém. Recusar por placa transformava uma troca de placa em
> "toda sessão volta a re-estimar o Kinect": medido, dois npz da MESMA bancada a
> 68,8 px um do outro. O diagnóstico agora diz se a calibração **aceita** o
> arquivo, e não apenas se ele existe. Regressão: `test_intrinsecas_kinect.py`,
> `test_freeze_kinect_direto.py` e `test_diagnostico_calibracao.py`.

**`tools/kinect_groundtruth_tool.py`** — é este que se abre no dia a dia (a
biblioteca `kinect_imu_groundtruth.py` **não tem mais programa próprio**; ela só
é importada). Ordem sugerida:

| Passo | Tecla | O que faz |
|---|---|---|
| 1 | — | abre e já mostra: mapa de câmeras por NOME, estado do stereo (com o motivo) e os esqueletos (Kinect verde + cada auxiliar na sua cor) — e **uma janela de vídeo por câmera**, em grade |
| 2 | `T` | diagnóstico: depth válido/medido **e** erro ponta-a-ponta da sobreposição no tabuleiro — testa as 4 simetrias da grade e fica com a de menor erro; ajusta uma **homografia nova** com os mesmos cantos para separar *captura ruim* de *calibração de outra bancada*; se a mediana passar de 25 px ele **grita** (`RECALIBRE nesta maquina: python stereo_calibration.py` + as 3 causas em ordem), e se o npz disser ter sido gravado em **outra câmera** ele avisa antes de tudo |
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

### A rede das luvas (hotspot do PC)

O firmware do ESP32 manda **unicast** para o PC na rede do **Ponto de acesso
móvel** (`192.168.137.1`, `ScopeAddress` do ICS), SSID **`EDUARDO-PC 9049`**
(senha `0011223344`) — o nome do Windows é `<computador> <4 dígitos>`, então
**renomear o PC muda o SSID**. Três coisas fazem o datagrama chegar, e as três já
falharam em silêncio: hotspot ligado **na faixa 2,4 GHz** (o ESP32 não vê 5 GHz),
IP do ICS existindo de fato e uma regra de firewall de **entrada UDP** para o
`.venv\Scripts\python.exe` (a regra "Python" que o Windows cria costuma ser do
*outro* python — regra de programa vale só para aquele executável).

```powershell
.venv\Scripts\python.exe tools\diagnostico_hotspot.py --escutar 20
.venv\Scripts\python.exe tools\diagnostico_hotspot.py --criar-regra-firewall  # admin
```

Detalhes e o snippet para o sketch do ESP32: `docs/IMU_DUPLO_ESP32_DESENHO.md`
§2.0. Regressão: `test_diagnostico_hotspot.py`.

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

Para a janela **fechar sozinha** quando a sessão terminar, use o launcher (5 s
no fim, 20 s em erro; repassa todos os argumentos acima):

```powershell
tools\rodar_sessao.bat --participante P01 --sessao S1 --tela-cheia
```

Em 30/09 a janela ficava parada depois da sessão porque ela era aberta com
`-NoExit` (ou era o terminal do VS Code, que é um terminal e não fecha sozinho);
o processo em si sempre sai — `ESC`/Ctrl+C termina via `hard_exit` (`os._exit`).

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

Para **testar o cancelamento sem ficar apertando o teclado** no meio da sessão
(bancada de 28/09/2026, com o paradigma rodando de verdade):

```powershell
.venv\Scripts\python.exe eeg_motor_paradigm.py --source generator `
       --blocos 1 --trials-por-bloco 4 --sem-questionario   # sessão curta
.venv\Scripts\python.exe tools\tecla_global.py c      # descarta a trial (796)
.venv\Scripts\python.exe tools\tecla_global.py esc    # aborta e sai na hora
```

O `C` deve aparecer como `trial N/M CANCELADO pelo experimentador` e marcar
**796** no CSV; o `ESC` imprime o resumo com os caminhos **reais** dos arquivos e
encerra o processo em poucos segundos (`CancelamentoGarantido`).

Enquanto a sessão roda: a janela de tracking mostra a fase, a sobreposição da
mão auxiliar e **"LINK EEG PERDIDO"** se o amplificador parar de enviar amostras
(o vigia também grava 899/898 no CSV).

## Kinect "mudo" (abre e nao entrega frame): o rotulo de integridade

Se o Kinect Studio mostra imagem e **todo** programa do projeto fica mudo
(`Open()` = S_OK, handle de evento valido e o EVENTO NUNCA DISPARA), confira antes
de qualquer outra coisa o **rotulo de integridade da pasta do projeto**:

```powershell
icacls C:\mestradopy
# se aparecer "Rotulo Obrigatorio\Nivel Obrigatorio Baixo:(OI)(CI)(NW)", e' ESTE o problema:
icacls C:\mestradopy /setintegritylevel (OI)(CI)M
```

O rotulo `Mandatory Low` herdado faz o processo filho nascer em integridade LOW, e
o `KinectService` (integridade MEDIA) **nao sinaliza os callbacks** para clientes
LOW. O sintoma e' cruel porque tudo o mais parece certo: o sensor esta' enumerado,
o driver carregado, o `PyKinectRuntime` abre em 0,0 s, o `SubscribeFrameArrived`
devolve handle valido — e nenhum frame chega. MEDIDO em 07-08/10/2026: com o
rotulo LOW o mesmo `.venv` na mesma pasta ficava mudo; depois de
`/setintegritylevel (OI)(CI)M` o mesmo comando entregou frame em 1,2 s.
O rotulo **ja' voltou sozinho** uma vez — se o Kinect "morrer" de novo, comece por
aqui.

Diagnostico: `tools\preview_kinect.py` (janela do sensor; nasce so' com o 1o
frame) e `tools\checa_kinect.py`. Lembrando que **`IsAvailable = FALSE` nao decide
nada** — foi medido FALSE tambem em processo que entregava frame; o termometro
real e' o EVENTO / `has_new_color_frame`.

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
firmware/esp32_luvas/ firmware das duas luvas (ESP32 + MPU6050, PlatformIO):
                      leitura do IMU, Kalman, Wi-Fi e UDP nas portas 4210/4211
                      (o `credenciais.h`, com a senha das redes, NÃO é versionado)
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
                      tecla_global.py: injeta ESPAÇO/C/ESC globalmente (testar
                      o cancelamento sem tocar no teclado),
                      teste_imu_dois_esp32.py: bancada das duas luvas (UDP),
                      diagnostico_hotspot.py: hotspot do PC para os ESP32
                      (SSID/ICS/firewall/escuta + snippet do firmware),
                      diagnostico_calibracao.py: auditoria OFFLINE do overlay
                      (modelos do Kinect de cada npz, divergencia entre arquivos
                      e o que 1 cm de erro em Z custa em px),
                      emissores_imu_falsos.py: luvas falsas para testar sem HW,
                      analisa_pdfs_janelas.py: extracao dos artigos,
                      checa_kinect.py: PRE-CHECAGEM do sensor em segundos e sem
                      tabuleiro -- o pykinect2 importa? o runtime abre? cada
                      stream (color/depth/body) responde? o caminho oficial
                      entrega frame? quais webcams existem por indice? o que o
                      disco diz do congelado? Saida 0 = pode ir para a bancada,
                      1 = NAO monte o tabuleiro, 2 = falta SDK/pykinect2,
                      conserta_intrinsecas_kinect.py: aposenta (renomeia para
                      `<nome>.arquivado`) o `kinect_intrinsics.npz` que a LEITURA
                      recusa, sem apagar nada e sem forjar `poses`/`span`
                      (dry-run por padrao; `--sim`, `--forcar`, `--restaurar`),
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

**Nomes no disco (o CSV de EEG recebe um timestamp extra):** o CSV de EEG é
escrito pelo `gp.FileWriter` do g.Pype, que **anexa o próprio timestamp** (e um
sufixo `_N` em caso de colisão) ao nome pedido — o arquivo em disco é
`gravacao_MEMI_<stamp>_<stamp-do-g.Pype>.csv`, e o JSON de eventos é renomeado
para o mesmo prefixo. A referência para os scripts de análise é a chave
`meta.arquivos` **dentro do JSON** (`eeg_csv`, `movimento_csv`, `eventos_json`,
`priming_dir`), que aponta para os arquivos que existem (o resumo do fim da
sessão imprime os mesmos caminhos).

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
python _render_cue.py       # 9 telas do cue em PNG (arranjo LOCAL de inspeção)
```

| Suite | Cobre |
|---|---|
| `test_paradigm_protocol.py` | trials balanceados, trial de 12 s, códigos únicos com legenda, contrato de colunas (EEG 34 / movimento 44), vigia 899/898, painel de impedâncias, questionário, `KT_src`, nome **real** do CSV/JSON nos metadados (o `FileWriter` do g.Pype anexa timestamp) e a pré-checagem do g.Nautilus separando "Python errada" de "falta o g.tec Suite (GDS)" |
| `test_gtec_runtime.py` | descoberta do runtime do GDS depois do g.tec Suite 2024: busca no disco (`GDS_LIBRARY` / `GNEEDACCESS_DIR` / `Program Files\gtec\gNEEDaccess`), **reparo preemptivo** do PATH da sessão (o wrapper cacheia a falha: reparo *depois* do erro não serve), `GNEEDACCESS_DIR` sem headers C ignorada com aviso, os três ramos da mensagem e o `build_source` sem aparelho virando mensagem acionável |
| `test_paradigm_abort.py` | cancelamento da sessão: borda das teclas ESPAÇO/C/ESC, vigia global de hotkeys (backend injetado), `hard_exit`, `cancel_trial` (descarta a trial e marca 796; respeita a pausa de bloco), `abort` executado uma vez, painel/fila de hotkeys e `CancelamentoGarantido` que sai mesmo com uma parada travada |
| `test_video_priming.py` | o vídeo de priming da pausa (0,5×): playback **cronometrado pelo relógio** (`passo = restante/n`) — o clipe nunca acaba antes da deadline (sem congelar; a pose final do movimento aparece) nem depois (a MI começa com o cue), `_start_mi` restaurando o `show_condition` (o canvas ficava em modo `video` e a MI inteira mostrava o último quadro **congelado**), clipe curto espacado até o fim da pausa, clipe atrasado com AVISO de >0,5×, `video_snapshot` sem consumir (a API legada continua de pé) e a **taxa de captura (fps)** medida no `_publish_slowmo` (5–7 fps nesta bancada: 3× MediaPipe por iteração) |
| `test_cue_estimulos.py` | o cue da trial: a seta **não invade** o ícone em 9 telas × 2 mãos (a folga vem de `geometria_cue` — 5% do menor lado, ponta na borda certa; a regressão de −9 px em 900×900 e −7 px em 700×900 ficou travada) e a bola de tênis tem as **duas costuras** do Twemoji: 4 pés sobre o círculo, mergulho de 0,39 R no meio (limitado entre 0,30 e 0,45 R), nada fora do círculo e simetria de espelho ponto a ponto (o cálculo antigo deixava uma reta atravessada e arcos 10% maiores para fora) |
| `test_parada_threads.py` | parada limpa das threads de bancada (`MotionRecorder`, `LinkWatchdog`, `ImpedanceMonitor`): o `self._stop` que elas criavam **sombreava o `Thread._stop`** e o `join()` morria com `TypeError` justamente no caminho de cancelar/abortar; a suite trava os nomes internos do `Thread` e o `stop()`/`join` dentro do limite |
| `test_palm_alignment.py` | vetor da palma por luva: Kabsch/SVD, alinhamento IMU✕câmera resolvido de amostras (montagem até 180°), o bug antigo medido (erro médio 19°, máx. 78° ao girar a mão) × a correção (erro máx. <2°), recusas, JSON (salvar/carregar), captura da tecla `O` na ferramenta e retrocompatibilidade (T=I é idêntico ao código antigo) |
| `test_janelas_grade.py` | janelas de vídeo: uma por câmera, grade sem sobreposição dentro da tela, escala de exibição automática (0,5 em 1080p → ~0,4 em 1536×960), cascata quando não cabe e `GradeDeJanelas` (cria/reposiciona/destrói, com cv2 falso) |
| `test_arm_csv.py` | colunas ARM_*/PALM_*/KT_hand/KT_src: ordem, unidades, relatividade à origem |
| `test_arm_ik.py` | cinemática inversa de 2 elos (8 casos) |
| `test_body_anchor.py` | esqueleto do SDK como âncora de profundidade (6 casos) |
| `test_cameras_multi.py` | calibração por índice de webcam e lista personalizada; a recusa de calibração de placa é medida pelos **dois** lados (quadrado antigo de 52 mm e **geometria de cantos** diferente), sempre sem falar de RMS |
| `test_camera_names.py` | nomes das webcams pelo Windows: mapa índice→nome, detecção da webcam do laptop, degradação silenciosa sem enumeração |
| `test_aux_pair_stereo.py` | par de auxiliares SEM Kinect: relação vinda da composição das duas calibrações, pixels por câmera, triangulação sub-cm, recusas e escolha do melhor par |
| `test_overlay_auxiliares.py` | 3 esqueletos no RGB do Kinect: cor por auxiliar, desvio contra o verde, profundidade rígida, gating de borda |
| `test_paradigm_aux_pair.py` | desvio de fonte do `_resolve_wrist3d`: o par de auxiliares assume com o Kinect cego; os caminhos do Kinect mantêm prioridade |
| `test_fusion_zeroing.py` | zeragem do IMU com as câmeras (bias/velocidade/posição) |
| `test_triangulation.py` | triangulação estéreo 3D da mão (mm) |
| `test_hand_calib.py` | calibração 2D→2D da mão (sintética) |
| `test_lateralidade_maos.py` | **as mãos trocadas numa câmera auxiliar** (`right`↔`left`): o bug demonstrado com número (o casamento por rótulo erra 672 px quando o rótulo do MediaPipe está invertido naquela câmera — o driver pode já entregar o fluxo espelhado, e o pipeline espelha TODAS as auxiliares com uma constante única), o conserto pela **geometria** (palma projetada no Kinect pela homografia da mão de cada câmera ✕ punho 3D do Kinect, com margem mínima entre as duas atribuições), o **aprendizado por câmera** (`aux_handedness_flip.json`: grava com consenso, confirma sem reescrever e **desfaz** inversão velha), o Kinect cego (cai no rótulo e **declara** a fonte), mãos ambíguas (não decide, sem votar), **lado repetido na mesma câmera** (descarta a atribuição inteira — defeito real pego no probe ao vivo) e as degradações (sem homografia/sem Kinect/vazio/sem rótulo) sem exceção e sem trocar lado por conta própria. Do **item 39** (o log real da bancada): a **via da projeção** — com a mesma homografia 2D ruim do log a geometria fica indecisa a **300 px** e o motivo **nomeia a via** (mandando calibrar o stereo, não o esqueleto), e com o **stereo 3D** daquela câmera decide com erro ~0 px pedindo o Z do punho **candidato** do Kinect; a câmera **sem homografia da mão** (Trust USB) sendo medida pelo stereo; e o **silêncio do log** (o estado ignora os números do motivo, então o mesmo motivo não reimprime); o **portão do aprendizado** (frame com contagens diferentes entre Kinect e auxiliar, ou erro acima de 60 px, corrige mas **não** vota — quem ensina tem de ser conferível inteiro) |
| `test_stereo_solve.py` | reparo da ordem dos cantos do tabuleiro no stereo |
| `test_intrinsecas_kinect.py` | modelo do Kinect CONGELADO (`--salvar-intrinsecas-kinect`): o npz é só o Kinect (sem R/T), a rotina reusa o congelado em vez de re-estimar, a sessão mede e **grava** o quanto o modelo daquela sessão divergiria (com o caso "tabuleiro pequeno", em que a divergência é o motivo de congelar) e o round-trip do arquivo. A cena é um **fixture** da medição de 30/09 (5×3 @ 57 mm): medido em 01/10/2026 com a placa de hoje (35 cantos) a divergência do modelo por sessão cai de 14,4 para 1,5 px e a armadilha desaparece, então a cena **não** segue a placa impressa. O fim tem a **guarda de migração**: a placa real declarada em `kinect_imu_groundtruth.py` tem de casar com a de `stereo_calibration.py` |
| `test_freeze_kinect_direto.py` | o **congelamento do Kinect** (arranjo Kinect-only): o loop da captura roda **inteiro sem hardware** (`_KinectFalso` + `coleta_poses` de verdade — fita roteirizada de cantos com cada pose 2×, `DETECTION_INTERVAL_S=0`/`STABLE_CAPTURE_SECONDS=0` restaurados no `finally`), a grade determinística (14 poses, span 33,9–39,7%, `tvec` iterado pela projeção **centrada** — a origem de `board_object_points` é um *canto* da placa, e o `tvec` ingênuo jogava o tabuleiro para fora do quadro), **grava de verdade** (12 poses, RMS 0,275 px, lido de volta pelo `carrega_intrinsecas_kinect` a 0,64 px do verdadeiro) e a cena na distância que o **par** impõe (19 poses, span 12%) **recusa sem gravar** (`gravou=False`, `rms=None`, arquivo nunca nasce). A linha 319 virou **regressão**: com `is not None` reinstalado a suíte falha no console que mostra o falso "(o modelo JÁ está gravado)" na pose 1. **O span mediano** (o número que decide): `spans_das_poses` = `sc.board_span`/diagonal por pose, `resumo_do_span` (mediana do **mesmo** `sc.span_do_tabuleiro` + quantas poses cumprem a meta), `texto_do_span` com 1 casa decimal (o motivo do `sc` arredonda 34,9% para "35%" — a bancada de 05/10 leu "ocupou 35% (minimo 35%)" e concluiu que o programa estava quebrado), o **aviso na hora** da pose fraca (`aviso_span_fraco`), o `motivo_com_span` no colchete da recusa e o **motivo final vindo do portão** (e não do último frame, que reportava "tabuleiro nao visto no Kinect"): 6c cobre tudo isso como função pura e 9 roda o fracasso real da bancada (grade 0,45 m entrelaçada com grade 0,80 m → 23 poses, mediano 34,1%, 6/23 cumprem, arquivo não criado, console avisando e motivo com o mediano). **O RMS por pose** (o `calibrateCamera` devolve UM número e não diz qual pose estragou o conjunto): `erros_por_pose` (solvePnP por pose com as intrínsecas resolvidas), `piores_poses`, `descartes_sugeridos` (só `erro > 3×mediana` — conjunto homogêneo NÃO descarta pose boa) e `veredito_do_rms` (pose isolada × captura inteira, com o mesmo corte: comparar a mediana com o limite foi **medido** como errado, porque a pose estragada puxa o K/D e a mediana das boas sobe para 0,59 px). O solver também **grava sem as piores** (`--max-descartes`, padrão 4), reavaliando os três portões a cada descarte e dizendo no motivo quais saíram e com que erro — 6d cobre as funções puras (inclusive o `threading.excepthook` que transforma o *wart* do pykinect2 no fechamento em uma linha, sem engolir outros erros) e 10 roda o caso da bancada com FakeKinect (14 poses, uma delas borrada: 4,449 px reprovado → 12 poses gravadas a 0,269 px, pelo `tenta_gravar` e pelo loop inteiro). Trava também o portão de captura (span/separação com o motivo e o número), a medida de movimento, a fonte única das constantes e o overlay puro (`35% -> 0,46 m`, `60% -> 0,27 m`, anel `0,30 × span`, X/anel/círculo atual conferidos **nos pixels** da imagem) |
| `test_diagnostico_calibracao.py` | auditoria OFFLINE (`tools\diagnostico_calibracao.py`): a grade espelhada do calibrador **ponto a ponto** (e `compara_modelos_kinect` = `espalhamento_px` = o número do cv2, com o bias de principal point de +32/+26 px que a bancada mediu), a projeção pela profundidade **idêntica** ao `to_kinect_stereo` (o código que desenha o overlay), o custo de 1 cm de Z contra o oráculo, os alertas com número (divergência entre arquivos, `fora do CONGELADO`, npz velho, RMS alto), o npz do modelo congelado **só com o Kinect** sendo comparado (era o bug: exigia o par inteiro e ficava mudo) e a ordem/nomes dos arquivos |
| `test_sand_traj.py` | rede/alvo/normalizador do SAND de trajetória |
| `test_move_onset.py` | detector de início do movimento (`MovementOnsetDetector`) e coluna `KT_onset` |
| `test_treino_loader.py` | leitura dos arquivos no treino: X (N, canais, amostras), montagem, CAR/z-score, 1 época, checkpoint + `SandTrajectoryBCI`, alvo concorrente × preditivo |
| `test_anatomical_reg.py` | regularizador anatômico: lei dos cossenos, consistência com a IK do projeto, envelope + gradiente, ângulo medido, limites articulares, NaN seguro, colunas `ARM_*` no caminho de arquivo, treino com o termo |
| `test_alvo_velocidade.py` | alvo de velocidade (`--alvo velocidade`): rampa/senoide, punho parado → 0 (caso do trial IDLE), gradiente no caminho de arquivo, treino + checkpoint registrando a formulação |
| `test_kalman_trajectory.py` | filtro de Kalman da trajetória: referência da aceleração (g → m/s², rotação, gravidade, bias), integração trapezoidal, IDLE sem deriva, erro **quadrático** do bias (10 cm/2 s e 2,50 m/10 s na integração dupla × 5,8 cm com o filtro), suavização, degrau, predição |
| `test_imu_bank.py` | dois IMUs (um por mão): parse das portas, convenção dos lados (1=dir/4210, 2=esq/4211), atribuição por porta com dois emissores UDP falsos, fusões independentes, resumo/parada |
| `test_imu_gravacao_dual.py` | gravação SIMULTÂNEA dos dois IMUs: dois emissores UDP falsos com assinaturas distintas (dir roll +5°/1,2 g; esq −5°/1,0 g), `blocos_motion()` por lado, linha do CSV com cada lado no seu bloco e `IMU_hand`, lado ausente em NaN/valid=0 |
| `test_diagnostico_hotspot.py` | rede das luvas: parser das regras do firewall do **registro** (porta em lista/intervalo/vazio, sem depender do idioma do Windows), o filtro que dava falso "tem regra" — regra de **programa** vale só para aquele executável, então a regra UDP do *python da Loja* **não** cobre o `.venv` —, cobertura × caso "quase", veredito (hotspot desligado, `BlockInbound` sem regra, porta ocupada, ICS diferente do documentado, SSID que não casa com o hostname), snippet do firmware, escuta ao vivo em porta de teste e **documento × ferramenta em sincronia** |
| `test_firmware_luvas.py` | contrato das luvas com o **firmware** (`firmware/esp32_luvas`): env do PlatformIO × fórmula da porta no `main.cpp` × `imu.IMU_PORTAS_PADRAO` (direita **4210** / esquerda **4211**), pacote de **10 campos sem ID de lado** (o lado vem da porta), `UDP_USAR_UNICAST 1` para o IP do ICS, `credenciais.h` real apontando para o hotspot atual e **fora do git** e nota `unicast`/README em sincronia (teste negativo: trocar o `GLOVE_SIDE` da esquerda derruba a suíte) |
| `test_imu_serial_bridge.py` | a **ponte serial→UDP** (`tools/imu_serial_bridge.py`, transporte serial do ESP32 dual-IMU): regex de 10 campos float (rejeita lixo sem quebrar), payload **idêntico ao `ImuReceiver`** real (mesmo bytes que o Wi-Fi manda), roteamento `IMU_PORTAS_PADRAO` 1→4210 / 2→4211, descarte de linha ruim sem derrubar o laço, janela de latência e `--escutar N` (modo só-imprime, sem abrir socket) — 27 asserções, sem porta física |
| `test_preview_kinect.py` | o **preview do sensor** (`tools/preview_kinect.py`): contrato do frame do pykinect2 (buffer `h*w*4` BGRA→BGR **nos pixels**, mesmo caminho do `coleta_poses`), overlay de status, loop com FakeKinect **entregando** (conta frames e respeita `--duracao-s`) e **mudo** (`n_frames=0` em vez de espera infinita + aviso do1º frame), teto do `--sem-janela` (sem GUI não há ESC), `main()` devolvendo **2** quando `sc.abre_kinect` recusa e **1** quando 0 frames, `close()` em todo caminho — 7 casos |
