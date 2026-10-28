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

## Tamanho do guia (item 140)

O plano MAGENTA tem **0,30 m de lado** (antes 0,16 m), preenchimento translúcido
(14%), moldura de 3 px, cantos marcados e diagonais/cruz central; a seta da
normal tem 0,26 m e a dos dedos 0,20 m. Foi aumentado por pedido da bancada
("está pequeno demais na tela") sem cobrir a mão. Se ainda ficar pequeno no
vídeo, o tamanho é o parâmetro `tamanho_m` de `desenhar_guia_plano_palma`.

## O guia só aparece com a mão detectada

O quadro MAGENTA e as setas são desenhados **ancorados no centro da palma
detectada** (landmark 9 + profundidade válida via `camera_para_pixel_color`).
Se a mão não estiver em quadro, se o MediaPipe não achar os landmarks ou se o
depth não devolver um ponto 3D para o centro da palma, **não existe origem
para desenhar o plano** — a tela mostra o banner vermelho "MÃO NÃO DETECTADA"
e o console repete o aviso a cada ~5 s; o take não avança até a mão aparecer.
Não é bug e não tem relação com a pose atual: é o guia que é ancorado na mão.

## Por que as poses 1 e 2 usam POSIÇÃO (item 134)

A normal medida pelo Kinect tem dois problemas nas poses “de frente”: (a) o
estabilizador de janela podia congelar o sinal (corrigido no item 131); (b) a
estimativa do plano da mão (punho+MCPs) **não é exatamente a normal da palma**
quando a mão está fora do eixo da câmera ou com luva. Medido na bancada
(28/10/2026): com a mão apontando a palma para a lente mas deslocada do eixo, o
QA dava 30-45° e a normal medida era exatamente a direção **mão→lente**. Ou
seja, o gesto natural é mirar a lente, não alinhar o plano ao plano da imagem.
Por isso as poses 1 e 2 passaram a usar a **verdade por posição** (`−normalize`
para palma, `+normalize` para dorso), a mesma que o J já usava com sucesso. O
QA do Kinect nessas 3 poses (palma, dorso, mira) usa o ângulo **dobrado**
`min(θ, 180−θ)`: o Kinect pode inverter palma/dorso (~180°), então o gate
confere o alinhamento do plano à linha de mira **sem depender do sentido**
(limite 10°, item 136). Se o plano estiver perpendicular à linha o take é
bloqueado; se só o sentido estiver invertido, grava normalmente. A verdade
continua sendo a posição 3D. As 6 poses geométricas restantes (±X, ±Y e
diagonais) continuam com alvo EXATO e QA **não dobrado** (o sentido importa).

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

## PROTOCOLO ATUAL (item 139): 9 POSIÇÕES DE MIRA, SEM EDGE-ON

O protocolo mudou em 28/10/2026. As poses de perfil (±X/±Y e diagonais fixas)
foram **removidas**: com a mão perpendicular aos raios do Kinect o plano de
profundidade é mal condicionado e a normal do vetor verde se perde/inverte,
tornando o QA impossível. Agora são **9 posições de mira deslocadas**, todas
com a **PALMA MIRANDO A LENTE**: centro (~45 cm), esquerda (~25 cm), direita,
cima (~20 cm), baixo e as 4 diagonais. A verdade continua sendo a posição 3D
(`−normalize(posição)`), o plano da mão fica **oblíquo** (bem condicionado para
o Kinect), e o QA usa o ângulo dobrado (limite 10°). O quadro MAGENTA é o
plano perpendicular à linha mão→lente. O gate exige **espalhamento**: cada take
novo precisa estar a ≥ 15° da direção de qualquer take anterior (senão aparece
"MOVA A MÃO") e o conjunto precisa de espalhamento mínimo ≥ 15° e máximo ≥ 30°.
A validação independente é o **deixa-um-de-fora (LOO)** — não há mais pose de
mira separada, porque todas são de mira. A tabela e o protocolo abaixo são
**históricos** (itens 127–138).

## As 9 poses (por mão)

O campo `dedos` é só a direção de referência para padronizar a **rolagem** do
punho (seta ciana) — não entra na verdade.

| # | Nome | Alvo (x, y, z) | Instrução resumida | Tipo |
|---|---|---|---|---|
| 1 | PALMA MIRANDO A LENTE | — (verdade = `−normalize(posição da mão)`) | Gesto natural de apontar a PALMA para a lente, em qualquer posição do quadro (de preferência perto do centro). **SEM folha** (ela bloquearia a mão). Verdade por POSIÇÃO 3D | posição |
| 2 | DORSO MIRANDO A LENTE | — (verdade = `+normalize(posição da mão)`) | Dorso de frente para a lente (palma para longe, como empurrar a lente com as costas da mão). Verdade por POSIÇÃO 3D | posição |
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
7. Mínimo para o ajuste: **6 takes** (`CALIB_GEO_MIN_TAKES`). O ideal é
   completar as 9 (6 geométricas + 3 de posição). Para o JSON OFICIAL ser
   sobrescrito é preciso ≥ 7 takes E a validação independente (que exige
   **as 6 geométricas**); ou seja, **não pule nenhuma das 6 geométricas** —
   as 3 de posição podem ser puladas no máximo 2; se pular uma geométrica,
   a sessão só grava o JSON piloto.

## Gravação parcial e tecla F (itens 137)

- A cada take o tool grava `calibracao_geometrica_<lado>_parcial.json`
  (sobrescrito): se a sessão for interrompida, os takes já feitos não são
  perdidos. Ao cancelar com `X` ou fechar o tool, o parcial também é salvo.
- O console (e o arquivo `calibracao_geometrica_diag.log`) registram a cada
  ~1 s a pose/alvo/normal/QA/QAef — é o que permite analisar depois uma pose
  que não fechou.
- **Tecla `F`**: força o take da pose atual quando o QA do Kinect está torto
  (entre o limite normal e 35°) mas você **confirmou visualmente** que a mão
  está alinhada ao guia. A verdade continua sendo o alvo exato/posição; o take
  fica marcado `"forcado": true` no JSON, com QA bruto/efetivo registrado.
  QA > 35° não é aceito nem forçando — aí é ajustar a mão.
- As 3 poses de posição usam QA dobrado (item 136); as 6 geométricas usam QA
  bruto ≤ 8° (ou `F` até 35° com confirmação).

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
  - ≥ 7 takes, com **as 6 geométricas** completas + poses de posição
    (`CALIB_GEO_MIN_TAKES + 1`);
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

## Edge-on e estabilizador de sinal (item 138)

Para as poses de perfil (mão perpendicular ao Kinect), o plano de 5 pontos
(punho + 4 MCPs) fica mal condicionado e o vetor verde se perdia. O tool agora,
no modo X, quando a normal base está fraca/edge-on (`|nz| < 0.5` ou spread > 5°),
estima a normal por **PCA com os 21 landmarks 3D** (`palm_normal_edge_on.py`,
qualidade/RMS/confiável) e passa o resultado por um **estabilizador de sinal com
histerese** (`palm_orientacao.py`): o sinal só vira depois de 8 amostras opostas
consecutivas, evitando flip espúrio no meio da pose. O log `[GEO]` mostra
`nfonte=edge21(...)` quando a normal robusta foi usada; se a qualidade for ruim,
cai na normal antiga (nada quebra). As posições continuam com QA dobrado; as
geométricas com QA bruto/`F` até 35°.

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


## ADENDO (item 141): 4 POSES GEOMÉTRICAS INCLINADAS

Depois das 9 posições de mira entram **4 poses geométricas intermediárias**
(alvos EXATOS, para o operador não precisar estimar posição):

| # | Nome | Alvo (x, y, z) | Instrução |
|---|---|---|---|
| 10 | INCLINADA 50 DIREITA | (0,766; 0; −0,643) | mão no centro; incline a palma ~50° para a DIREITA da imagem; dedos para CIMA |
| 11 | INCLINADA 50 ESQUERDA | (−0,766; 0; −0,643) | mão no centro; incline a palma ~50° para a ESQUERDA; dedos para CIMA |
| 12 | INCLINADA 60 CIMA | (0; 0,866; −0,5) | mão no centro; incline a palma ~60° para CIMA na imagem; dedos para a DIREITA |
| 13 | INCLINADA 60 BAIXO | (0; −0,866; −0,5) | mão no centro; incline a palma ~60° para BAIXO; dedos para a DIREITA |

São **intermediárias**: nem paralelas ao eixo −Z nem edge-on (perpendiculares).
O componente `z` ainda é grande (−0,64 ou −0,50), então a profundidade condiciona
bem a normal; e os alvos exatos aumentam o espalhamento do ajuste. Nestas 4 o
QA é o **bruto** (≤8°; `F` permite até 35° com confirmação), ao contrário das 9
de mira que usam QA dobrado. A ordem final do X é: 9 posições (1–9) + 4
inclinadas (10–13), depois a mão esquerda. O gate continua o mesmo (≥7 takes,
espalhamento, resíduo, pior inclinada ≤20°, LOO ≤15°).
