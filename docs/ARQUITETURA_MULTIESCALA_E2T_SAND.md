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
2. **Ramo longo de 8 s apenas em execução**, como professor;
3. pose final direta + velocidade auxiliar;
4. consistência multiescala com peso pequeno;
5. nenhuma janela de 8 s atravessando ME→MI.

Assim você usa a ideia do E2T (velocidade/janelas) e a do SAND (contexto
longo/pose) sem violar a causalidade nem inflar o resultado da MI.

## 9. O que ainda precisa ser decidido

- manter MI de 4 s ou estender para 8 s?
- contexto longo em 100 Hz ou 500 Hz?
- pesos das perdas de pose, velocidade e consistência;
- usar destilação do ramo longo para o curto ou treinar separado;
- avaliar o ramo longo apenas como professor ou também como preditor offline.
