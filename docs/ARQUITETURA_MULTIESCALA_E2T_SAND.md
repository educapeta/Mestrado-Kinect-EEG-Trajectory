# Arquitetura multiescala: contexto longo do SAND + janelas locais do E2T

## 1. O que E2T e SAND realmente usam

| Trabalho | Trial | Janela de entrada | Alvo | Observação |
| --- | --- | --- | --- | --- |
| E2T (Kwon et al., 2026) | 8 s de ME + 8 s de MI | 3 s deslizante, stride 0,5 s (1500 amostras a 500 Hz) | velocidade 3D | janelas sobrepostas dentro do trial; ME e MI separadas |
| SAND (Fu et al., 2026) | grasp-and-lift (WAY-EEG-GAL) | 9,6 s a 100 Hz (T=960) | trajetória xyz | contexto longo, quase offline |
| Nosso piloto atual | ME 4 s + MI 4 s | 2 s causal | pose final + velocidade auxiliar | MI de 4 s |

O artigo novo que apareceu na pasta é **“Opportunistic Calibration of Sensor
Orientation using the Kinect and IMU Sensor Fusion”** (Wireless Health 2015),
não um artigo de calibração multi-câmera. Ele pode ajudar na parte de
Kinect+IMU, mas não no problema das câmeras auxiliares.

## 2. A ideia em uma frase

Sim, faz sentido pensar em:

```text
CONTEXTO LONGO (8–9,6 s)  ->  pose final / trajetória completa  (professor)
JANELAS LOCAIS (2 s)      ->  velocidade xyz                    (aluno/online)
```

Mas o papel correto é:

- o contexto longo aprende a **pose final** e a estrutura global;
- as janelas locais aprendem **dinâmica/velocidade**;
- a integração da velocidade **não deve substituir** a pose final: ela serve
  de consistência e de representação para o online.

## 3. O problema crítico: 8 s de MI não cabem em 4 s

Este é o ponto que precisa ficar claro antes de desenhar o modelo.

Se você pegar uma janela de 8 s durante a MI de 4 s, a janela **inclui
necessariamente a ME anterior, a animação ou o repouso**. Isso cria
vazamento: o modelo pode usar EEG de execução para “prever” a MI. O resultado
offline fica ótimo e a transferência para MI real fica inválida.

Opções:

1. **Manter MI de 4 s** e usar somente janelas de 2–4 s na MI. O contexto
   longo de 8 s fica restrito à execução e vira professor/auxiliar.
2. **Estender a MI para 8 s** (como E2T). O trial passa a ter
   `2 + 4 + 2 + 8 = 16 s`; 250 trials = ~67 min de protocolo ativo. Aumenta
   fadiga e muda o CEP.
3. Criar dois modelos: um de contexto longo só para execução (offline) e um
   de 2 s para MI/online. É o caminho mais defensável no mestrado.

Minha recomendação: **opção 3** para o piloto; opção 2 só se oito segundos de
MI forem realmente necessários.

## 4. Arquitetura proposta

```text
                         ┌─────────────────────────────────────┐
EEG 8 s (execução) ──────┤ Encoder de contexto longo (SAND)    │
                         │  espectro + Transformer             │
                         └──────────────┬──────────────────────┘
                                        ├─ cabeça de pose final (p, R6D, abertura)
                                        └─ cabeça de trajetória (opcional)

EEG 2 s (execução/MI) ───┐
                         ├─ Encoder local (Conv1D + Transformer)
                         │  (pode compartilhar pesos com o longo)
                         └─ cabeça de velocidade 3D

                         ┌─ consistência: p_final ≈ p0 + ∫v dt
                         └─ destilação: representação local aprende com a longa
```

### 4.1. Cabeça de pose final (principal)

- entrada: contexto longo de execução (8–9,6 s);
- saída: `p` (3), `R` 6D, abertura;
- label: pose de hold medida;
- uso: professor offline e alvo principal da tese.

### 4.2. Cabeça de velocidade (auxiliar/online)

- entrada: janela local de 2 s;
- saída: velocidade linear 3D no fim da janela;
- label: derivada da trajetória medida na execução;
- uso: regularização, consistência e predição causal no online.

### 4.3. Integração da velocidade

Não use a integral da velocidade como pose final principal. Ela acumula
deriva. Use:

```text
L_consist = || p_final_pred - (p0 + ∫ v_pred dt) ||²
```

como perda auxiliar, com peso menor.

Se quiser corrigir a deriva, use um filtro (Kalman) ou uma segunda cabeça
`p_final` que recebe a integral como entrada, mas isso é etapa 2.

## 5. Como treinar isso sem vazamento

### Estágio 1 — professor de execução

- entradas de 8 s de ME;
- alvo: pose final/hold + trajetória;
- treina o encoder longo.

### Estágio 2 — aluno local

- janelas de 2 s de ME;
- alvos: velocidade e pose final do mesmo trial;
- inicializa com o encoder longo (ou compartilha pesos).

### Estágio 3 — transferência para MI

- janelas de 2 s de MI;
- zero-shot: avalia sem ajuste;
- fine-tune: poucos trials de MI com rótulos de classe e, em análise
  exploratória, com rótulo fraco de pose herdado da ME.

### Estágio 4 — consistência multiescala

- no mesmo trial, a pose final prevista pelo ramo longo e a integral da
  velocidade do ramo curto devem concordar;
- peso pequeno, para não dominar o erro de pose.

## 6. Janelas e taxas

Para não explodir computação:

| Ramo | Taxa | Janela | Amostras |
| --- | --- | --- | --- |
| longo | 100 Hz | 8–9,6 s | 800–960 |
| curto | 100 Hz | 2 s | 200 |
| curto (alta resolução) | 500 Hz | 2 s | 1000 |

O SAND usa 100 Hz e T=960; o E2T usa 500 Hz e 1500 amostras (3 s). Não misture
taxas sem reamostrar e sem registrar no checkpoint.

Stride sugerido:

- treino offline: 0,25–0,5 s;
- online: 0,25 s (4 Hz) ou 0,1 s (10 Hz), dependendo da latência.

## 7. O que cada cabeça acrescenta de verdade

| Cabeça | Acrescenta | Não acrescenta |
| --- | --- | --- |
| pose final longa | alvo global, menos drift, contexto | causalidade online |
| velocidade curta | dinâmica, causalidade, regularização | pose final absoluta |
| classificação | sanity check e comparação | pose contínua |
| integral da velocidade | consistência | posição final confiável sozinha |

## 8. Decisão prática para o mestrado

Para o piloto, eu faria:

1. **Ramo principal de 2 s** para MI e online;
2. **Ramo longo de 8 s em execução e MI** (com ME = MI = 8 s), como professor/offline;
3. pose final direta + velocidade auxiliar;
4. consistência multiescala com peso pequeno;
5. nenhuma janela de 8 s atravessando ME→MI.

Assim você usa a ideia do E2T (velocidade/janelas) e a do SAND (contexto
longo/pose) sem violar a causalidade nem inflar o resultado da MI.

## 9. Correção sobre o volume do E2T

O E2T **informa o número de trials** no texto de métodos:

- 12 participantes destros;
- 50 trials por tarefa;
- 4 tarefas (FVM, FLM, FRM, LVM);
- 200 trials por participante;
- **2.400 trials no total**.

Além disso, o “trial de 8 s” não é o trial inteiro. A sequência é:

```text
repouso 5 s + instrução 8 s + ready 2 s
+ ME 8 s + ready 2 s + MI 8 s
= 33 s por trial
```

O que tem 8 s é cada fase (ME e MI), não o trial completo. O artigo descreve
um sistema de **64 canais** (não 32). A 500 Hz e 64 canais, uma fase de 8 s tem:

```text
8 s × 500 Hz × 64 canais = 256.000 valores
```

Se contarmos ME + MI por trial: 512.000 valores por trial. Em 2.400 trials:
1.228,8 milhões de valores, aproximadamente **4,92 GB em float32**, só das
fases analisadas.

Isso também corrige a conta que usava 32 canais: o E2T gravou com sistema de
64 canais.

## 10. Se o nosso ME e MI passarem para 8 s cada

Faz sentido igualar as duas fases. O trial passa a ser:

```text
HOME 2 s + ME 8 s + animação 2 s + MI 8 s = 20 s
```

Com 125 trials por sessão:

```text
125 × 20 s = 2.500 s = 41,7 min de protocolo ativo por sessão
```

Com 2 sessões (250 trials/participante):

```text
250 × 20 s = 5.000 s = 83,3 min de protocolo ativo por participante
```

O tempo de MI por participante será:

```text
250 × 8 s = 2.000 s = 33,3 min de EEG de MI
```

O tempo de ME será o mesmo: 33,3 min. O restante (HOME + animação) soma 16,7 min.

Comparação de volume bruto de EEG (32 canais, 500 Hz, float32):

| Desenho | Trials | Tempo/trial | Valores de EEG | Float32 |
| --- | --- | --- | --- | --- |
| 5 participantes × 250 | 1.250 | 20 s | 400 milhões | ~1,60 GB |
| 6 participantes × 250 | 1.500 | 20 s | 480 milhões | ~1,92 GB |
| E2T (64 canais, ME+MI) | 2.400 | 16 s (2×8 s) | 1.229 milhões | ~4,92 GB |

Ou seja: **125 trials por sessão × 2 sessões = 250 trials/participante** já
produz uma quantidade grande de **pontos de EEG**. O gargalo não é o número de
amostras; é o número de **trials independentes** (1.250–1.500), a qualidade dos
labels, a fadiga e a divisão treino/teste.

### O que muda na arquitetura

Com ME e MI de 8 s, o ramo longo pode ser treinado nas duas condições sem
atravessar fases:

```text
janela longa: 8 s de ME ou 8 s de MI, começando no respectivo onset
janela curta: 2 s deslizante dentro desses 8 s
```

Isso resolve o problema de vazamento que existia com MI de 4 s. Em
contrapartida:

- o trial passa para 20 s;
- 125 trials/sessão = 41,7 min de protocolo ativo; com pausas e setup, a visita
  fica em ~1h40–2h;
- a fadiga aumenta;
- o ramo longo de 8 s continua **não causal para online** (só está disponível
  no fim da MI); o ramo curto de 2 s é o que controla em tempo real.

Se a fadiga for um problema, uma alternativa é **100 trials/sessão × 2
sessões = 200/participante**, como no E2T:

```text
200 × 20 s = 4.000 s = 66,7 min ativos
200 × 8 s = 1.600 s = 26,7 min de MI
```

Ainda é bastante dado e reduz a carga.

## 11. O E2T diz que a quantidade de dados foi suficiente?

Não explicitamente. O artigo:

- usa **12 participantes**;
- **200 trials por participante** (50 por tarefa × 4 tarefas);
- **2.400 trials** no total;
- treina modelos **por sujeito**;
- não apresenta cálculo de poder ou análise formal de suficiência amostral.

Nas limitações, eles reconhecem que:

- o treinamento por sujeito limita a generalização entre sujeitos;
- não existe dataset público com EEG + trajetória 3D contínua em MI, então
  treinaram e avaliaram apenas no próprio dataset;
- assumem que a trajetória imaginada espelha a executada, e reconhecem o viés
  dessa suposição;
- a MI tem SNR menor que a ME.

Ou seja: **2.400 trials foram suficientes para o resultado dentro do desenho
deles, com 12 sujeitos e treino por sujeito, mas eles não provam suficiência
geral**. O nosso 1.250–1.500 é menor em número de sujeitos; por outro lado,
temos mais trials por participante (250 vs 200). Para um piloto é razoável;
para generalização entre sujeitos, é limitado.

## 12. O que ainda precisa ser decidido

- ME = MI = 8 s (trial de 20 s) ou manter ME 4 s + MI 8 s?
- 125 trials/sessão × 2 ou 100 × 2?
- contexto longo em 100 Hz ou 500 Hz?
- pesos das perdas de pose, velocidade e consistência;
- usar destilação do ramo longo para o curto ou treinar separado;
- avaliar o ramo longo apenas como professor ou também como preditor offline.
