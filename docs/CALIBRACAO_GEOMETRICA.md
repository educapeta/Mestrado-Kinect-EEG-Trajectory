# Calibração geométrica da palma (item 127, tecla `X`)

Calibração do vetor da palma do IMU (o par `T`/`f` do modelo `v = T·R·f`)
usando ALVOS GEOMÉTRICOS EXATOS do referencial da câmera, com um plano-guia
desenhado no vídeo. É o complemento — não o substituto — do `J`/`V`
("Calibração da palma por TAKES"): o `J` calibra com a verdade *mao→lente*
(por posição), o `X` calibra com a verdade *geométrica* (alvo do referencial da
câmera) e ancora a direção absoluta com uma pose de mira deslocada.

## Por que existe

- A verdade do `J` é a direção do centro da palma até a lente:
  `-normalize(posição da mão)`. Duas limitações medidas:
  - a **posição 3D também é uma estimativa** (depth + resgate pelo esqueleto),
    com ruído/vieses próprios;
  - as poses de mira ficam todas num **cone estreito** em volta do eixo óptico
    (`-Z` da câmera): pouco espalhamento deixa o ajuste mal condicionado e o
    vetor deixa de acompanhar a mão fora daquele cone (item 122: poses quase
    iguais → erro mediano 12–39 graus na simulação).
- Na calibração geométrica a verdade é o **alvo em si** (normal da palma
  paralela a um eixo/diagonal do referencial da câmera): por construção o
  erro de **MEDIDA** da verdade é ~0. O que sobra é apenas o erro de
  **EXECUÇÃO** humana (alinhar a mão ao guia) — e é exatamente isso que o QA
  mede a cada quadro: a normal medida pelo Kinect vira **controle de
  qualidade, não verdade**.
- A última pose (**mira deslocada**, verdade por posição) ancora a direção
  absoluta: só alvos de plano não distinguem pequenas rotações em torno do
  alvo.

## Convenção do referencial da câmera

O projeto usa o camera space do Kinect v2 assim:

| Eixo | Direção |
|---|---|
| `X` | DIREITA na imagem |
| `Y` | CIMA na imagem |
| `Z` | Para DENTRO da cena (afastando-se da lente; a lente, vista da mão, fica em −Z) |

Os alvos da tabela abaixo seguem exatamente essa convenção. Na prática o
usuário não precisa decorar eixos: o alvo é desenhado no vídeo — quadro
**MAGENTA** = plano que a folha deve "vestir", seta **MAGENTA** = normal-alvo
(a direção exata que entra na verdade), seta **CIANO** = referência dos dedos.
A projeção usa a mesma `camera_para_pixel_color` do esqueleto.

> É a convenção do camera space do Kinect v2 (o mesmo frame usado pelo
> `camera_para_pixel_color` do esqueleto). Na prática não é preciso decorar
> eixos: o guia no vídeo é a projeção do alvo, então basta alinhar a folha ao
> quadro e os dedos à seta.

## O guia só aparece com a mão detectada

O quadro MAGENTA e as setas são desenhados **ancorados no centro da palma
detectada** (landmark 9 + profundidade válida via `camera_para_pixel_color`).
Se a mão não estiver em quadro, se o MediaPipe não achar os landmarks ou se o
depth não devolver um ponto 3D para o centro da palma, **não existe origem
para desenhar o plano** — a tela mostra o banner vermelho "MÃO NÃO DETECTADA"
e o console repete o aviso a cada ~5 s; o take não avança até a mão aparecer.
Não é bug e não tem relação com a pose atual: é o guia que é ancorado na mão.

## Sinal da normal (corrigido no item 131)

A normal medida pelo Kinect tem sinal ambíguo (SVD) e é orientada por uma
referência anatômica (`transversal = index−pinky`, `longitudinal = middle−punho`,
`cross(transversal, longitudinal)`, espelhada para a esquerda). O estabilizador
de janela (`normal_robusta`) alinhava o sinal pela **normal anterior**; isso
impedia uma virada REAL de ~180° entre poses (palma→dorso, ±Z/±X): a mediana
ficava com o sinal antigo, o QA ia a ~170° e o take era recusado. No modo X a
janela agora usa a **normal anatômica instantânea** como referência de sinal
(continua valendo para J/V); a pose 1 também foi marcada para ser feita **SEM
folha** (com a palma virada para a lente, a folha bloqueia a mão/landmarks).

## As 9 poses (por mão)

O campo `dedos` é só a direção de referência para padronizar a **rolagem** do
punho (seta ciana) — não entra na verdade.

| # | Nome | Alvo (x, y, z) | Instrução resumida | Tipo |
|---|---|---|---|---|
| 1 | PALMA PARA A CAMERA | (0, 0, −1) | Palma de FRENTE para a câmera **SEM folha** (ela bloquearia a mão): mão aberta, palma paralela ao plano da imagem; dedos para CIMA | geométrica |
| 2 | DORSO PARA A CAMERA | (0, 0, +1) | Dorso de frente (palma para longe): plano do dorso paralelo à imagem; dedos para CIMA | geométrica |
| 3 | PALMA PARA A DIREITA | (+1, 0, 0) | Mão de perfil: palma apontando para a DIREITA da imagem; dedos para CIMA | geométrica |
| 4 | PALMA PARA A ESQUERDA | (−1, 0, 0) | Mão de perfil: palma apontando para a ESQUERDA; dedos para CIMA | geométrica |
| 5 | PALMA PARA CIMA (cam) | (0, +1, 0) | Palma para CIMA **no referencial da câmera** (ela está inclinada ~51°: use o plano-guia, NÃO a gravidade); dedos para a DIREITA | geométrica |
| 6 | PALMA PARA BAIXO (cam) | (0, −1, 0) | Palma para BAIXO no referencial da câmera (use o guia); dedos para a DIREITA | geométrica |
| 7 | DIAGONAL FRENTE-DIREITA | (0,707, 0, −0,707) | Entre as poses 1 e 3; dedos para CIMA | geométrica |
| 8 | DIAGONAL FRENTE-CIMA | (0, 0,707, −0,707) | Entre as poses 1 e 5; dedos para a DIREITA | geométrica |
| 9 | MIRA DESLOCADA (palma na lente) | — (verdade = `−normalize(posição da mão)`) | Mão a ~45–50 cm e DESLOCADA ~20 cm para a ESQUERDA da imagem; palma MIRANDO a lente | posição |

## Protocolo na bancada

1. **Uma mão por vez**: a tecla `X` começa pela DIREITA; terminada a sequência,
   o próprio modo abre a ESQUERDA ("AGORA A MAO ESQUERDA..."). A outra mão
   fica FORA do quadro ou parada.
2. Cole uma **FOLHA RÍGIDA** na palma (papelão/plástico): é ela que alinha o
   plano da mão ao quadro magenta. NÃO use papel mole (entorta e mente para o
   QA) e NÃO cubra os dedos/landmarks — o Kinect precisa vê-los para medir a
   normal e a posição.
3. Alinhe a folha ao quadro **MAGENTA** e os dedos à seta **CIANA**. O QA
   (ângulo entre a normal medida e o alvo) precisa ficar **≤ 8 graus** — o
   contador fica verde quando está dentro.
4. Fique **PARADO 2 s** (rotação estável < 4 graus, posição < 3 cm): o take é
   gravado automaticamente ("TAKE GRAVADO").
5. **Gate de movimento**: depois de um take é preciso MOVER a mão (≥ 8 graus de
   rotação ou ≥ 10 cm) antes do próximo armar — evita take duplicado da mesma
   pose.
6. **`N` pula a pose** sem gravar (útil nas de perfil, cujo QA de plano é mais
   ruidoso — a qualidade final é avaliada no gate). **`X` cancela** tudo a
   qualquer momento (nada é gravado).
7. Mínimo para o ajuste: **6 takes** (`CALIB_GEO_MIN_TAKES`) — o ideal é
   completar as 9. Para o JSON OFICIAL ser sobrescrito é preciso um pouco mais:
   **6 geométricas + a MIRA** (≥ 7 takes) e a validação independente da mira
   tem de existir (pule no máximo 2 poses geométricas); se pular mais, a sessão
   só grava o JSON piloto.

## Como o T/f é resolvido (ajuste híbrido ponderado)

- Modelo: `v = T·R·f` — `R` é a atitude do IMU (corpo → mundo), `f` a direção
  da palma no corpo do sensor e `T` a rotação mundo do IMU → câmera.
- **Pesos (híbrido)**: as 8 poses geométricas entram com peso **1.0**
  (`CALIB_GEO_PESO_GEOMETRICA` — verdade exata, sem erro de medida) e a MIRA
  DESLOCADA com peso **1.5** (`CALIB_GEO_PESO_POSICAO` — verdade por posição,
  com erro de medida maior, mas independente do plano do Kinect; o peso maior
  ancora a direção absoluta sem deixar que ela distorça o conjunto).
- **Solver**: `imu.solve_palm_alignment_pesos` — Kabsch **alternado
  ponderado** (`H = Σ w·v·(R·f)ᵀ`; com `T` fixo, `f` = média ponderada
  normalizada de `RᵀTᵀv`), inicializado pela amostra de MAIOR peso (começar
  por uma amostra de peso baixo podia travar o ajuste num mínimo local).
  Peso ≤ 0 ou NaN ignora a amostra. `solve_palm_alignment` (sem pesos) é a
  mesma conta com peso 1.
- **Diagnóstico** no `resultado()`:
  - residuo ponderado médio (graus) e erro por pose (`qa_deg` de execução ×
    `erro_deg` do ajuste);
  - **validação da MIRA FORA do ajuste**: resolve de novo só com as poses
    geométricas e mede o erro na mira (mediana/máx) — valida a direção
    absoluta sem contaminar o ajuste;
  - **LOO (deixa-um-de-fora)**: refaz o ajuste sem cada take e mede o erro
    nele (mediana/máx) — mostra a dependência de um take ruim.

## O que é gravado

- **Sempre que o ajuste resolve** (≥ 6 takes e convergiu): o JSON piloto
  `calibracao_geometrica_<lado>_<timestamp>.json` na raiz do projeto (ex.:
  `calibracao_geometrica_right_20261022_153000.json`), com os dados brutos de
  cada take (alvo, verdade, QA, rotação do IMU, posição da mão, normal
  medida), a solução (`rotation`, `body`, `residual_deg`), o diagnóstico (erro
  por pose, validação da mira, LOO) e as constantes usadas — auditoria e
  reanálise sem repetir a sessão. Com **< 6 takes** ou sem convergência,
  NADA é gravado.
- O alinhamento OFICIAL `imu_palm_alignment_<lado>.json` só é substituído se o
  **gate** passar:
  - ≥ 7 takes = 6 geométricas + a MIRA (`CALIB_GEO_MIN_TAKES + 1`);
  - residuo ponderado ≤ 12 graus (`CALIB_GEO_MAX_RESIDUO_DEG`);
  - pior pose geométrica ≤ 20 graus (`CALIB_GEO_MAX_POSE_DEG`);
  - a validação independente da MIRA existe e tem mediana ≤ 20 graus
    (`CALIB_GEO_MAX_VALIDACAO_MIRA_DEG`) — é a única prova de generalização
    para a direção absoluta;
  - se a validação cruzada LOO estiver disponível, mediana ≤ 15 graus
    (`CALIB_GEO_MAX_LOO_MEDIANA_DEG`).
  Passando, o `T`/`f` entra na fusão e é gravado pelo mesmo caminho do
  `J`/`O` — o arquivo anterior vira `imu_palm_alignment_<lado>.json.bak`.
  Falhando, o console lista os motivos e o oficial **não é tocado**.

## Análise offline

`tools/analisa_calibracao_geometrica.py` relê o JSON piloto (o mais recente por
padrão, ou `--arquivo`) sem hardware: recalcula o erro por pose, a validação
cruzada LOO e compara com o alinhamento oficial (`--oficial`), imprimindo um
veredito e podendo gravar `--json-out` com todos os números. Serve para
comparar sessões/versões e para auditar uma calibração antes de confiar nela.

## Relação com o J/V e com o firmware

- O `J` (calibrar por takes) e o `V` (verificar/certificado) **continuam
  existindo** e seguem sendo a referência de bancada; iniciar um cancela o
  outro e o `X` (um modo por vez). O `X` cobre direções que o cone do `J` não
  vê (perfil, cima/baixo no referencial da câmera, diagonais); o `V` segue
  sendo o teste de mira independente.
- Recomendação atual: rodar `J` + `V` como de costume e usar o `X` na mesma
  sessão, comparando os residuos/erros no **teste de mira** e o comportamento
  do vetor vermelho ao girar a mão.
- **Firmware (item 124)**: o código do ESP32 já tem a calibração dos OFFSETS
  do acelerômetro no boot (luvas PARADAS e DEITADAS), faixa do giroscópio
  ±1000 °/s e DLPF ~42 Hz / ODR ~100 Hz — mas **ainda não foi gravado no
  ESP32**. Enquanto o firmware novo não estiver gravado, o erro dependente da
  orientação (1,5–7 graus, item 123) continua e aparece como residuo tanto no
  `J` quanto no `X`.

## Limitações e hipóteses

- O termo dominante é o **erro de execução humana** (típico 3–8 graus sem
  guia). O guia magenta + QA ≤ 8 graus existem para contê-lo, mas um take com
  QA bom ainda carrega o erro residual do alinhamento manual.
- **Poses de perfil** (3, 4, 5, 6, 8) têm QA de plano mais ruidoso: com a mão
  contendo o eixo da câmera sobra pouco plano visível no depth. Se a pose não
  armar, use `N` e siga (o gate final cuida da qualidade).
- **Sem magnetômetro o yaw não é absoluto**: o `T` absorve o yaw do boot
  atual (igual ao `J`). Depois de religar/reflashar o ESP32, refazer a
  calibração (ou ancorar o yaw).
- **Ainda NÃO testado na bancada** — até agora só o teste estático/sintético
  (`test_calibracao_geometrica.py`). A primeira sessão real deve comparar com
  `J` + `V` no teste de mira.

## Fluxo resumido

```powershell
# 0) luvas PARADAS e DEITADAS no boot (bias de giroscópio + offsets do accel)
# 1) ferramenta de bancada:
.venv\Scripts\python.exe tools\kinect_groundtruth_tool.py
# 2) tecla X -> MAO DIREITA primeiro; siga o quadro MAGENTA, dedos na seta CIANA
# 3) 2 s PARADO com QA <= 8 graus = take automático; N pula pose; X cancela
# 4) no fim da direita a esquerda abre sozinha; repita as 9 poses
# 5) console: residuo, erro por pose, validação da mira/LOO, gate e arquivos
# 6) valide com o V (teste de mira) e a conferência visual do vetor vermelho
```

Código envolvido: `kinect_imu_groundtruth.py` (constantes `CALIB_GEO_*`,
`POSES_CALIBRACAO_GEOMETRICA`, classe `CalibracaoGeometrica`, helpers
`_rotacao_a_para_b`/`_matriz_rotacao_eixo`/`_angulo_rotacoes`), `imu.py`
(`solve_palm_alignment_pesos`), `tools/kinect_groundtruth_tool.py` (teclas
`X`/`N`, overlay `desenhar_guia_plano_palma`,
`finalizar_calibracao_geometrica`) e `test_calibracao_geometrica.py`.
