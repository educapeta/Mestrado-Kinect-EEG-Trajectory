# Arquitetura online: janela de contexto × horizonte de predição × latência × MCU

> Responde, com números, às perguntas de 18/09: *"o SAND ganhou por usar janela
> grande — devo usar contexto longo no online? isso não fica lento? e num
> microcontrolador na prótese? 2 s serve para movimentos coordenados, mas e para
> **sustentar** um gesto?"*
> Complementa `docs/ESTADO_DA_ARTE_JANELAS.md`.

## 1. Duas coisas diferentes que estavam sendo tratadas como uma só

| Dimensão | O que é | Quem manda nela |
| --- | --- | --- |
| **Contexto de entrada** | quantos segundos de EEG o modelo olha | a informação neural (planejamento, ERD/ERS, MRCP) |
| **Horizonte de predição** | quanto **à frente** do instante atual está o alvo | o **controle** (prótese, atuador) |

Elas são **independentes**. É perfeitamente possível (e é o desenho recomendado)
ter contexto longo com horizonte curto:

```
   contexto (o que o modelo VE)            horizonte (o que ele PREVE)
   |<---------------------->|              |<----->|
   ----------+---------------+--------------+-------+---------> tempo
        onset-3 s        onset+1 s      instante   alvo a +0,5 s
```

Mapa dos 7 artigos nessas duas dimensões (números do texto integral):

| Estudo | Contexto | Horizonte | Observação |
| --- | --- | --- | --- |
| SAND | **9,6 s** (T=960 @100 Hz, trial inteiro) | o próprio trial (0 s à frente) | offline; 2,27 s de inferência |
| E2T | **3 s** (1 500 @500 Hz, passo 0,5 s) | quadro-a-quadro (velocidade) | offline; integra a velocidade |
| M3T | **5 s** (500 @100 Hz) | 5 s | offline |
| Tang | **2 s** (Δt escolhido) | 0 s (posição atual) | causal, treino offline |
| Korik | **0,25 s** (band power) | 0 s | **online** (MI, 2 braços) |
| **Nosso (hoje)** | **2 s** (1 000 @500 Hz) | 0 s (concorrente) ou **+1,5 a +2,5 s** (preditivo) | online, 6–10 ms/janela |

## 2. Então o SAND ganhou só por causa da janela grande?

Não — e isso importa muito para a dissertação. Três efeitos misturados:

1. **Contexto maior ajuda de verdade** quando o alvo é longo (o trial inteiro):
   para desenhar uma trajetória de 9,4 s, ver 9,4 s de EEG é o mínimo razoável.
   Mas isso é *descrição do mesmo trecho*, não controle.
2. **A métrica é generosa**: medimos nesta base sintética que a **trajetória média
   do treino** (baseline trivial, zero EEG) atinge PCC **0,89–0,97**. Ou seja,
   parte de qualquer PCC alto de trajetória vem da forma estereotipada do
   movimento — e quanto mais longo o trecho, mais forte fica o baseline (no alvo
   +1,5 a +2,5 s o baseline sobe para 0,90–0,98).
3. **Não há restrição de latência**: 9,6 s de contexto + 2,27 s de inferência =
   ~11,9 s de atraso. Para uma prótese, isso é inutilizável; para um artigo
   offline, é irrelevante.

Conclusão: **copiar a janela do SAND não é a lição**. A lição é separar contexto
de horizonte e sempre reportar o baseline.

## 3. Contexto longo **sem** pagar latência: decimar antes de alongar

O custo de entrada é ≈ canais × amostras. Como a banda útil da trajetória é
1–30 Hz (a nossa!), a 100 Hz ainda temos Nyquist = 50 Hz — margem suficiente:

| Janela de contexto | taxa | amostras | custo relativo | atraso de buffer |
| --- | --- | --- | --- | --- |
| **2,0 s** (hoje) | 500 Hz | 1 000 | **1,0×** | 2,0 s |
| 4,0 s | 250 Hz | 1 000 | 1,0× | 4,0 s |
| 4,0 s | 100 Hz | 400 | 0,40× | 4,0 s |
| 6,0 s | 100 Hz | 600 | 0,60× | 6,0 s |
| **9,6 s** (igual ao SAND) | **100 Hz** | **960** | **0,96×** | 9,6 s |

Ou seja: **o SAND usa exatamente 960 amostras — o MESMO tamanho de entrada que a
nossa janela de 2 s a 500 Hz** — só que distribuídas em 9,6 s. Se a informação
útil está abaixo de 50 Hz, trocar taxa por contexto é quase grátis.

Dois cuidados honestos:

- **Perde-se alta frequência** (>50 Hz: EMG, gama baixo 30–50 Hz) e **resolução
  temporal** (10 ms por amostra em vez de 2 ms). Para a **trajetória** (fenômeno
  lento, <10 Hz) não importa; para **detectar o onset** importa — e por isso o
  detector de onset deve continuar rodando na cinemática (30 Hz) e, se preciso,
  num ramo de EEG em taxa cheia.
- **Filtro causal**: em tempo real não existe `filtfilt` (zero-phase). O ramo
  online deve usar IIR de 4ª ordem causal, e vale treinar com o mesmo filtro
  causal para não haver descasamento treino/inferência (o nosso treino usa hoje
  `sosfiltfilt`, que é offline — ponto a corrigir antes de falar em produto).

### 3.1 Desenho recomendado: dois ramos (rápido + lento), horizonte curto

```
       rápido:  1,0 s @ 500 Hz  (500 amostras)  -> estado/onset/curto prazo
       lento :  6,0 s @ 100 Hz  (600 amostras)  -> planejamento/contexto
                              \  atenção  /
                               \    |    /
                        cabeça de regressão -> trajetória a +0,5 s
```

## 4. Orçamento de latência: contexto **não** é latência

Ponto que resolve a dúvida "não ficaria muito lento?": com janela **causal**
`[t−C, t]` transmitida em fluxo, o **comprimento do contexto C não entra na
latência de controle** — o atraso é decidido por *quando esperamos a última
amostra*, e não por quanto passado olhamos. O que entra é:

```
latencia_controle = t_inferencia + passo_de_atualizacao/2 + atraso_do_atuador
                    (+ atraso de grupo do filtro causal)
```

| Sistema | Contexto | t_inferência | Passo | Latência de controle |
| --- | --- | --- | --- | --- |
| SAND | 9,6 s **não causal** (precisa do trial inteiro) | **2,27 s** | por trial | **inviável** (~11,9 s) |
| Nosso hoje (2 s) | 2,0 s causal | 6–10 ms (modelo) / dezenas de ms (pipeline) | 0,25–0,5 s | **~0,3–0,5 s** |
| Alvo recomendado (rápido 1 s + lento 6 s) | 1,0 s + 6,0 s (ambos causais) | ~10–30 ms | **0,10 s** | **~0,15–0,35 s** |

Três consequências práticas:

1. **Podemos alongar o contexto sem piorar o controle** — só aumenta o custo de
   computação (e por isso vale decimar, seção 3). O que estraga o controle do
   SAND é (a) precisar do trial inteiro para prever e (b) gastar 2,27 s nisso.
2. **O passo de atualização é o que a prótese sente**: 0,25 s de passo dá um
   movimento "aos saltos"; 0,10 s já é fluido para a mão. Reduzir o passo obriga
   a reduzir o custo por inferência — de novo: decimar/quantizar, não encurtar o
   contexto.
3. **Há um custo escondido**: em tempo real não existe filtro zero-phase
   (`sosfiltfilt`); um IIR causal de 4ª ordem atrasa ~3–10 ms e distorce a fase.
   Treinar com o **mesmo filtro causal** do tempo real é um ajuste obrigatório
   antes de qualquer medida de precisão "de produto" (hoje o treino e o tempo
   real não estão 100% casados nesse ponto).

## 5. "Sustentar um gesto" é outro problema — e o culpado não é a janela de 2 s

Em regime de sustentação a trajetória é **quase constante**, e aí:

- o **alvo tem variância ~0** → o PCC perde sentido e o baseline trivial ganha
  por construção. Não é teoria: medimos que, no alvo preditivo (+1,5 a +2,5 s da
  mesma base), o baseline da trajetória média chega a **0,90–0,98** de PCC;
- um horizonte longo em hold é pior ainda: extrapolar 2 s um sinal que está
  parado só amplifica ruído (o modelo "inventa" movimento que não existe);
- a neurofisiologia joga contra: durante uma postura mantida o ERD **retorna ao
  nível de base** (ERS de rebote), então o sinal decodificável de *posição
  estática* é bem mais fraco do que o de *velocidade durante o movimento*.

Desenho correto para hold (o que a literatura e as próteses comerciais fazem):

| Alternativa | Como funciona | Vantagem no hold |
| --- | --- | --- |
| **Prever velocidade** (como o E2T) e integrar | o alvo é `dx/dt`; a posição vem de integração (com Kalman / α-β) | velocidade ~0 ⇒ o estado **para** naturalmente, sem deriva |
| **Máquina de estados + detecção de intenção** | classificador de intenção (pegar / segurar / soltar / ir para X) + controlador assistido | é o que existe em prótese comercial; robusto e interpretável |
| **Alvo curto** (0,2–0,5 s) em vez de 2 s | o horizonte preditivo acompanha o movimento em vez de extrapolar | em hold o erro fica pequeno e o tremor é filtrado |

E as métricas de hold **não** são PCC: são RMSE em regime, *jitter* (desvio
padrão da previsão no platô), tempo de estabilização e taxa de falso movimento.

## 6. Custo medido e o caminho até o microcontrolador da prótese

Medido com `tools/mede_custo_modelo.py` (d_model=32, 4 camadas, 30 pontos de
saída, 60 427 parâmetros — o modelo atual):

| Configuração | amostras de entrada | MFLOPs/janela | vs. hoje | INT8 do modelo |
| --- | --- | --- | --- | --- |
| **2,0 s @ 500 Hz (atual)** | 1 000 | **113,8** | 1,00× | 59 kB |
| 1,0 s @ 500 Hz | 500 | 55,0 | 0,48× | 59 kB |
| 4,0 s @ 250 Hz | 1 000 | 113,8 | 1,00× | 59 kB |
| **4,0 s @ 100 Hz** | 400 | **43,2** | **0,38×** | 59 kB |
| 6,0 s @ 100 Hz | 600 | 66,7 | 0,59× | 59 kB |
| **9,6 s @ 100 Hz (janela do SAND)** | 960 | **109,1** | **0,96×** | 59 kB |
| 18,0 s @ 100 Hz | 1 800 | 208,0 | 1,83× | 59 kB |

**A conclusão é direta e responde à sua pergunta**: dá para ter **9,6 s de
contexto pelo preço da janela de 2 s que já usamos**, ou **4 s de contexto por
38 % do custo**, simplesmente decimando de 500 Hz para 100 Hz. O tempo medido em
CPU (49,7 ms na linha de base) inclui a disputa de CPU nesta execução de teste
(5 processos de treino rodando em paralelo) — os **FLOPs** são a grandeza
comparável; o tempo absoluto deve ser re-medido com a máquina ociosa.

### 6.1 Viabilidade num MCU (conta explícita, para projeto futuro)

| Recurso | Valor | Cabe em |
| --- | --- | --- |
| pesos FP32 (60 427 param.) | 242 kB | MCU com ≥ 512 kB de flash |
| pesos **INT8** | **59 kB** | quase qualquer Cortex-M4F |
| pesos INT8 + poda 50 % | ~30 kB | idem, com folga |
| entrada INT8 (32 ch × 960) | 30,7 kB | — |
| entrada FP32 (32 ch × 960) | 123 kB | precisa ≥ 256 kB de SRAM |
| ativações (3 s de contexto) | < 64 kB | STM32F4+/H7, ESP32-S3 |

Ordem de grandeza do tempo num Cortex-M7 a 480 MHz com CMSIS-NN (INT8,
~200–500 MMAC/s na prática): 109 MFLOPs ≈ 55 MMACs ≈ **0,1–0,3 s por janela**;
na configuração de 4 s @ 100 Hz (43 MFLOPs ≈ 22 MMACs) ≈ **0,05–0,1 s**. Ou seja:

- **com quantização INT8 e janela de 4–6 s @ 100 Hz, é viável** num MCU de linha
  (com passo de atualização de 0,1–0,25 s);
- **com o modelo FP32 de hoje, não é** (242 kB de pesos + 123 kB de entrada +
  ativações em FP32 estouram a SRAM de um M4 e ficam lentos).

Os três gargalos reais num protótipo embarcado **não** são o modelo:

1. **extração de features causal**: CAR + z-score + passa-banda 1–30 Hz em 32
   canais (IIR de 4ª ordem por canal) — trivial para o MCU, mas precisa ser a
   mesma cadeia do treino (hoje o treino usa `sosfiltfilt`, que é offline);
2. **aquisição/transmissão do EEG**: 32 ch × 500 Hz × 3 B = 48 kB/s (USB/WiFi
   resolvem; BLE obriga a decimar para 100–250 Hz — outra razão para decimar);
3. **energia**: quem pesa é a rádio, não a inferência — mais um argumento para
   **decimar e reduzir o passo** em vez de encurtar o contexto.

Ferramentas sugeridas: exportar PyTorch → ONNX → TFLite Micro (ou CMSIS-NN
direto), quantização pós-treino INT8 calibrada nas **nossas** janelas, e usar o
truque de atenção por FFT (`FFTAttention`) como vantagem — atencao em O(T log T)
é justamente o que o SAND explorou para sair de 43,68 s para 2,27 s.

## 7. O que eu faria, em ordem (barato → caro)

| # | Experimento | Custo | O que decide |
| --- | --- | --- | --- |
| 1 | **Decimar para 100 Hz** no carregamento (`--fs-alvo`) e treinar a mesma janela de 2 s | ~30 linhas | se não perder PCC, ganhamos 2,6× de folga (4 s de contexto pelo mesmo custo) |
| 2 | **Contexto 4–6 s @ 100 Hz** (janela causal única) | pequeno | o contexto longo realmente ajuda a trajetória? |
| 3 | **Dois ramos** (1 s @ 500 Hz + 6 s @ 100 Hz) | médio | mantém a resolução temporal fina para onset e o contexto longo para trajetória |
| 4 | **Alvo = velocidade** + integração com Kalman/α-β | médio | resolve o hold (seção 5) e casa com o E2T |
| 5 | **Filtro causal no treino** (igual ao tempo real) | pequeno | tira o viés otimista do `sosfiltfilt` |
| 6 | Quantização INT8 + medição em MCU dev board | médio | viabilidade da prótese (seção 6) |

E duas regras de método que valem mais que qualquer arquitetura:

- **sempre reportar o baseline do alvo médio** (`tools/compara_alvos.py`) — no
  hold e nos alvos longos ele chega a 0,90–0,98;
- **declarar o filtro, a taxa e a causalidade** em qualquer número comparado com
  o SAND/E2T (offline zero-phase versus causal muda o resultado).

## 8. Integração da velocidade com a aceleração do MPU6050 (filtro de Kalman)

Implementado em 18/09 no `imu.py` (fonte única do IMU) e ligado no tempo real.

**Por que velocidade e acelerômetro juntos.** O modelo passou a prever
**velocidade** (`--alvo velocidade`, seção 1.2 do doc de protocolo) e a posição
tem de ser reconstruída. As duas rotas ruins:

- integrar a **posição que o modelo prevê** diretamente: perde a dinâmica e
  herda a estereotipia do alvo;
- integrar a **aceleração do IMU duas vezes**: o erro de posição cresce com
  **t²** e o bias do acelerômetro manda nesse erro.

**O argumento do erro quadrático, medido** (`test_kalman_trajectory.py`, caso 5).
A lei é simples e é **só integração**: um erro **constante** na **aceleração**
(bias ε) produz erro **linear** na velocidade (ε·t) e **quadrático** na
**posição** (½·ε·t²). O erro está na posição — a aceleração não melhora nem
piora; quem amplifica é a dupla integração. Com ε = **5 cm/s²** (≈5 mg, típico de
MPU6050 sem zeragem) e integrando a aceleração duas vezes (o teste verifica a
integração numérica contra a fórmula fechada):

| | erro de posição |
| --- | --- |
| Integração dupla da aceleração | **10 cm em 2 s** e **2,50 m em 10 s** (×25 ao quintuplicar o tempo = t²) |
| Filtro de Kalman + velocidade da EEG | **5,8 cm em 10 s** → **43× menor**, e o bias foi **estimado** (0,052 contra 0,05 m/s²) |

Como o filtro é construído (`KalmanTrajectory`, 3 estados por eixo:
`[posição, velocidade, bias de aceleração]`):

```
predicao : p <- p + v*dt + 0.5*(a_medido - bias)*dt^2 ;  v <- v + (a_medido - bias)*dt
medidas  : velocidade decodificada da EEG (a cada inferencia, sigma_vel_eeg)
           posicao do punho vista pelo Kinect (ancora lenta, sigma_pos_camera)
```

**O que o filtro faz — e o que ele NÃO faz** (importante para não vender demais):
ele **não** melhora a medição da aceleração (o MPU6050 continua com o bias dele).
Ele faz duas coisas: (1) o **bias é um estado**, então a parte *sistemática* do
erro é absorvida em vez de integrar; (2) a integração é **ancorada por uma medida
independente** (velocidade da EEG, e a posição do Kinect como âncora lenta), de
modo que o erro **não acumula indefinidamente** — ele é reiniciado a cada
atualização.

Daí sai uma alavanca de projeto que vale registrar: entre duas atualizações o erro
volta a crescer como **½·ε·T²**, com `T` = intervalo de atualização. Com
ε = 5 cm/s²: `T = 0,5 s` → **6 mm**; `T = 2 s` → **10 cm**. Ou seja, a **cadência
de inferência** importa quadraticamente — mais um argumento para passo de 0,1–0,5 s
(seção 4) e para decimar o EEG em vez de alongar o passo.


No tempo real (`sand_traj_tempo_real.py`): quando o checkpoint diz
`alvo = velocidade`, o nó de inferência cria o filtro, lê a aceleração do
`ImuReceiver`, usa a **última velocidade prevista** como medida e a **posição do
punho** (Kinect, no referencial da origem) como âncora lenta, e entrega ao
overlay/log a trajetória já em **metros** (integrada e filtrada). O log ganhou
`| KF pos=(...) bias=.... m/s^2`.

> ⚠️ **Bug corrigido no caminho**: o `ImuReceiver` era criado *dentro* do
> `TrackingThread` e **nunca recebia `start()`** — na prática o IMU estava morto
> no tempo real (a zeragem e a âncora por acelerômetro não aconteciam). Agora há
> **um** receiver por processo, iniciado em `main()` e compartilhado entre o
> rastreador e o nó de inferência.

Efeito colateral desejado: como o IMU passa a funcionar, a **zeragem** por
movimento parado (bias + âncora espacial, que já existia no `PositionFusion`)
volta a alimentar o filtro — é ela que mantém o bias pequeno de verdade em uso
real.


## 9. Regularizador anatômico: implementação, A/B e como usar

Implementado em 18/09 (`st.AnatomicalRegularizer`, `test_anatomical_reg.py` 8/8),
seguindo o MTRT: **geometria como perda, não como arquitetura**. Como o alvo é o
punho, o que a geometria de 2 elos determina univocamente é o **raio**
`r = ||punho − ombro||` e o **ângulo de cotovelo implicado** (lei dos cossenos).
Daí três termos: **envelope** de alcance, **ângulo** implicado × medido e
**limites** articulares.

### 8.1 A/B medido (6 sessões sintéticas, 144 trials, mesma seed, 20 épocas)

| Configuração | melhor val_loss | PCC x/y/z | `anat env` na época 20 |
| --- | --- | --- | --- |
| `λ = 0` (inativo) | 0,6645 | **0,767 / 0,475 / 0,742** | não medido |
| `λ = 1e-6` (só mede) | ~0,66 | 0,696 / 0,518 / 0,746 | **1,22** |
| **`λ = 1`** | **0,5744** | 0,742 / **0,547** / **0,757** | **0,081** (15× menor) |
| `λ = 10` | 0,7417 | 0,643 / 0,365 / 0,658 | **0,004** (305× menor) |
| `λ = 1` + `ângulo = 1` | 1,0042 | 0,153 / 0,072 / 0,154 | 0,000 |

Três leituras que valem mais que a tabela:

1. **Sem o termo, a violação CRESCE durante o treino**: com `λ = 1e-6` ela sai de
   0,10 (época 2) para **1,22** (época 20) — a rede aprende a imitar também os
   trechos fisicamente impossíveis do alvo (outliers do Kinect). Com `λ = 1` ela
   fica **estável em 0,08** e com `λ = 10` em 0,004. O regularizador não só
   melhora uma métrica: ele **impede uma inconsistência que piora com o treino**.
2. **`λ = 1` é ganho líquido**: melhor val_loss (0,574 contra 0,664) e melhor
   *y*, com PCC praticamente igual em *x*/*z* — é regularização de verdade.
3. **Escala importa**: o termo de ângulo tem valor bruto ~16 contra um MSE ~1; com
   peso 1 ele domina e colapsa o PCC. Comece em `--peso-anatomico 0.1` e
   `--peso-angulo-cotovelo 0.005`, e suba olhando as colunas `anat env/ang/lim`
   do log.

### 8.2 Como usar

```powershell
# 2 s causal ancorado no onset + regularizador anatomico
.venv\Scripts\python.exe sand_traj_treino.py --data gravacoes `
    --event-code 795 --window-start-sec -0.5 --window-sec 2.0 `
    --peso-anatomico 0.1 --peso-angulo-cotovelo 0.005
```

- `--peso-anatomico` liga **envelope de alcance** + **limites articulares**
  (15–178°) e `--peso-angulo-cotovelo` liga o termo do ângulo medido;
- os comprimentos de elo vêm das próprias gravações (`ARM_upper_len_m` /
  `ARM_fore_len_m`), com `--elo1/--elo2` para forçar;
- sem colunas `ARM_*` o termo fica inerte e avisa no log — nada quebra;
- o checkpoint registra toda a configuração usada (rastreabilidade da análise).

Documentação vinculada: `docs/ESTADO_DA_ARTE_JANELAS.md` (seções 2.4 e 5),
`test_anatomical_reg.py` (8 casos) e `tools/mede_custo_modelo.py`.


