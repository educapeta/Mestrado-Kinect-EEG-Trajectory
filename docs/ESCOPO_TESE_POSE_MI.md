# Escopo da tese — pose contínua com seeding por execução

> Documento de trabalho para alinhar tese, CEP e experimento. Não é a versão
> final do capítulo; é a decisão de escopo antes de reescrever a metodologia.

## 1. Objetivo principal (decidido)

Desenvolver e avaliar um decodificador de **pose contínua da mão** a partir de
EEG, usando **execução motora como seeding** para permitir que a imagética
motora (MI) posterior seja pareada com um alvo de pose.

Pose = posição `xyz` + orientação (`R`/quaternion) + abertura da garra.

A classificação MI (mão direita/esquerda/repouso) continua como **tarefa
secundária/baseline**, não como objetivo principal.

### 1.1. Formulação científica que evita promessa excessiva

A contribuição não é “a execução ensina a MI” em abstrato. É:

> **Usar a execução motora como supervisão auxiliar para aprender uma
> representação de pose que possa ser decodificada a partir de MI, reduzindo a
> dependência de labels de execução no momento do uso.**

O que isso permite afirmar, no máximo:

- execução fornece labels de pose que a MI não tem;
- um modelo treinado nessa representação pode funcionar em MI sem labels de
  pose;
- isso é uma **hipótese de transferência**, a ser medida.

O que **não** permite afirmar automaticamente:

- que o modelo funcionará em amputados, AVC ou ALS;
- que a MI de um paciente é equivalente à MI de um saudável;
- que a execução de pessoas saudáveis resolve a falta de execução do usuário
  final;
- que o seeding por execução pode ser feito pelo próprio paciente.

Para a limitação de pacientes, a lógica correta é:

```text
treino: execução de uma coorte saudável (ou mista, quando possível)
alvo:   MI do usuário que não consegue executar
```

O usuário final não precisa executar no treino; ele precisa que a
representação aprendida com execução transfira para a MI dele. Isso é
exatamente o que H2 testa, e é uma limitação em aberto — não um resultado
garantido.

Se o objetivo for “pessoas que não têm braço”, a tese deve dizer:

- a coorte de treino é saudável;
- a MI é o sinal de uso;
- a validação em pacientes é trabalho futuro, salvo se houver coorte clínica.

Também não chame isso de “impossibilidade de aplicar em pessoas não
saudáveis”: BCIs de MI já são usadas em pacientes; a limitação real é
**desempenho, variabilidade e generalização**, não impossibilidade.

## 2. Hipóteses

- **H1 (execução):** EEG de execução permite decodificar pose contínua acima do
  acaso.
- **H2 (transferência):** um modelo treinado em execução transfere para MI de
  forma útil (zero-shot ou few-shot).
- **H3 (classificação):** as janelas de MI permitem classificar mão
  direita/esquerda/repouso com acurácia compatível com a literatura.
- **H4 (online, se houver tempo):** a pose prevista consegue comandar a mão
  robótica em tempo real com latência aceitável.

H2 é a hipótese de maior risco. A tese precisa sobreviver se ela falhar.

## 3. Delineamento proposto

### 3.1. Participantes

- 5 participantes (máximo 6), saudáveis, destros/ambidestros.
- 6 runs por participante.
- ~42 trials por run = ~250 trials por participante.
- Total: 1.250 trials (5 participantes) ou 1.500 (6).

Atenção: **1.250 trials não são 1.250 exemplos independentes de frame.** Cada
trial gera muitos frames, mas eles são altamente correlacionados. O número
efetivo para estatística é de trials/sessões/sujeitos.

### 3.2. Estrutura de um trial

Proposta inicial (a validar com orientador):

```text
repouso 1–2 s
  ↓
cue (1 s)
  ↓
[EXECUÇÃO] reach → grasp → lift → hold 2–3 s → return
  ↓
pausa
  ↓
[MI] mesma instrução, sem movimento, 4 s
```

Duas opções de seeding:

**A. Bloco execução → bloco MI (recomendado para o piloto)**
- primeiro runs de execução, com pose medida;
- depois runs de MI com os mesmos alvos;
- mais simples de separar treino e avaliação.

**B. ME → MI dentro do mesmo trial (como E2T)**
- execução e MI na mesma trial;
- priming mais forte, mas trial longo e risco de contaminação motora;
- precisa de washout e randomização.

### 3.2.1. Timing do trial de 12 s e fadiga

O `eeg_motor_paradigm.py` atual usa:

```text
HOME/repouso      2 s
CUE + ME          4 s
pausa + vídeo      2 s
MI                4 s
----------------------
total             12 s
```

O “pausa 2 s que engata no próximo” é o **HOME do próximo trial**; não são
14 s.

Com 250 trials em uma sessão:

```text
250 × 12 s = 3.000 s = 50 min de trials puros
5 blocos × 50 trials = 5 × 10 min
4 pausas de 2 min = 8 min
--------------------------------
~58 min de protocolo ativo
+ baseline (5 min) + prática (6 × 12 s ≈ 1,2 min)
+ setup de EEG/touca/gel (20–40 min)
= ~1 h 25 min a 1 h 45 min de visita
```

Isso é **cansativo**, principalmente porque são 250 repetições de
alcance/preensão + MI. O tempo de MI real por participante é:

```text
250 × 4 s = 1.000 s = 16,7 min de EEG de MI
```

Ou seja: 50 min de sessão produzem só ~17 min de MI. Os outros ~17 min são ME
e ~17 min são pausa/vídeo/home.

**Recomendação:** não use 50 trials por bloco. Use 25–30 trials por bloco
(5–6 min), com pausas de 3–5 min. Se 250 é o total do participante em 6
sessões, cada sessão tem ~42 trials e fica muito mais viável.

Cuidado com a ambiguidade:

- **250 trials por sessão × 6 sessões** = 1.500 trials/participante — muito
  provavelmente inviável;
- **250 trials totais por participante em 6 sessões** = ~42/sessão — viável
  para piloto.

### 3.3. Tarefa e objetos

- 3 objetos/posições diferentes;
- mão direita e esquerda;
- alcançar, agarrar, levantar, segurar e retornar;
- aleatorizar objeto, lado e ordem;
- não usar peso/fricção como no WAY-EEG-GAL: aqui o interesse é **pose**,
  não força.

## 3.4. Zero-shot, fine-tune e “MI pura” — mesmo modelo

Não são três arquiteturas. É o **mesmo modelo**, com protocolos de treino
diferentes:

| Protocolo | Treino | Teste | O que prova |
| --- | --- | --- | --- |
| **Zero-shot** | execução | MI | transferência sem nenhum label de MI |
| **Fine-tune** | execução + poucas MI | MI restante | adaptação com calibração curta |
| **MI pura (classificação)** | MI com labels de classe | MI | classificação direita/esquerda/repouso |
| **MI pura (pose)** | MI com label fraco da execução | MI | só é possível assumindo o pareamento ME→MI |

“MI pura” para pose **não existe sem algum label**. Se você usar a pose da
execução como label da MI, isso é aprendizado fraco/supervisionado por
transferência, não ground truth de MI.

O caminho mais leve:

```text
1 modelo com encoder compartilhado
  ├─ cabeça de pose (posição + 6D + abertura)
  └─ cabeça de classificação (direita/esquerda/repouso)
```

Treine primeiro em execução; depois avalie zero-shot em MI; depois faça
fine-tune com poucas trials de MI; a classificação pode ser treinada desde o
início como tarefa auxiliar.

## 3.5. Alvo principal: pose final/hold; velocidade/trajetória como auxiliar

Essa é uma decisão metodológica forte e defensável:

```text
ALVO PRINCIPAL
  pose final/hold da mão:
  posição + rotação (6D/quaternion) + abertura

ALVO AUXILIAR
  velocidade 3D (e talvez velocidade angular) da ME,
  usada para dar consistência temporal/suavidade
```

Por que faz sentido:

- o robô precisa chegar a uma pose final para segurar o objeto;
- a pose final é menos sensível a diferenças de ritmo da MI do que a
  trajetória inteira;
- a velocidade auxiliar aproveita a lógica temporal do E2T sem exigir que a
  MI reproduza exatamente os 4 s da ME;
- a MI pode ser mais lenta/rápida; a pose final continua sendo o alvo
  semântico correto.

O que **não** fazer:

- usar a trajetória inteira como alvo principal e MSE no tempo exato;
- chamar a trajetória da ME de “ground truth da MI”;
- exigir que a MI tenha a mesma duração/velocidade da ME.

Se quiser usar a trajetória como auxiliar:

- treine uma cabeça de velocidade/trajetória com peso menor;
- alinhe por fase/DTW para o cálculo da perda auxiliar;
- reporte as duas métricas: erro temporal estrito e erro com DTW.

O E2T previu **velocidade 3D** em MI com referência da ME; isso é evidência de
que a ideia é viável, não prova de replay temporal exato.

### 3.5.1. Pose final: label da trial, não label de toda janela

Sua preocupação é correta: se você pegar todas as janelas dos 4 s de MI e der a
mesma pose final, algumas janelas podem representar a fase de alcance, não o
hold. Isso é **ruído de label**.

Três soluções, da mais simples para a mais sofisticada:

1. **Só janelas tardias (recomendado no piloto):**
   - usar apenas os últimos 1,5–2 s da MI;
   - label = pose final/hold da ME anterior;
   - janelas iniciais ficam fora da perda de pose (podem entrar na classificação
     ou no auxiliar).
2. **Pooling por atenção/MIL (multiple-instance learning):**
   - o modelo vê várias janelas da mesma trial;
   - produz **uma** predição de pose por trial;
   - o label é único por trial;
   - o modelo aprende a dar peso às janelas mais informativas.
3. **Alvo resolvido no tempo (trajetória/velocidade):**
   - cada janela tem alvo correspondente à fase da ME, alinhado por DTW/fase;
   - é o caminho do E2T, mas exige alinhamento temporal.

No piloto, comece com (1). Se sobrar tempo, compare com (2). Não use (3) como
alvo principal.

### 3.5.2. Como fica o modelo final do piloto

```text
encoder SAND compartilhado
  ├─ cabeça de pose final: xyz + 6D + abertura
  ├─ cabeça auxiliar de velocidade linear 3D (ME)
  └─ cabeça de classificação: direita/esquerda/repouso (opcional)
```

- pose final: label da trial, janelas tardias;
- velocidade: label da ME, janelas alinhadas ao ME onset/DTW, peso menor;
- classificação: todas as janelas de MI, label de classe.

## 3.6. Janelas de treino: ME onset × MI onset

```text
EXECUÇÃO (ME):
  janela de 2 s em torno do onset do movimento
  ex.: [ME_onset - 0,5 s, ME_onset + 1,5 s]
  ou janelas deslizantes durante a ME

MI:
  janela de 2 s ancorada no INÍCIO DA MI (beep/cue), com offset
  ex.: [MI_onset + 0,5 s, MI_onset + 2,5 s]
```

Não use “ME_onset + 2 s” para a MI: na MI não há movimento medido; o marcador
correto é o início da MI. O offset de 0,5 s é para evitar o transiente
visual/ERP do cue; não é regra universal, mas é comum em ERD/ERS.

Se o alvo principal é a pose final, a janela de MI pode ser escolhida dentro da
fase de MI (por exemplo, 0,5–2,5 s ou 2–4 s) com o mesmo label final; isso
reduz a dependência de alinhamento temporal.

## 3.7. Estímulo visual antes da MI

O E2T usa uma **esfera branca** como guia e esferas laranja/arrows para
trajetória; não usa vídeo real da mão. Para o seu paradigma:

- troque o vídeo real por uma **animação esquemática** da trajetória;
- fundo neutro/escuro, sem mão, sem textura, sem detalhes;
- mostre o alvo final (esfera/caixa) e um ponto/cursor percorrendo o caminho;
- durante a MI, tela neutra;
- registre `video/animation_start` e `animation_end` no CSV;
- mantenha o mesmo estímulo em todas as trials.

O vídeo real pode ser um confound visual/atencional; a animação geométrica é
mais controlada e ainda serve de priming.

## 4. Janela de 4 s: o que ela significa

Sua ideia: `-0,5 s antes do cue` + `3,5 s depois do cue` = 4 s.

Isso é bom para capturar:

- baseline pré-cue (ERD/ERS de referência);
- preparação;
- reação;
- movimento;
- hold.

Mas há três janelas diferentes que precisam ser separadas:

| Janela | Uso |
| --- | --- |
| **janela do trial** (~6–8 s) | contexto experimental, análise de fases |
| **janela de entrada do modelo** (2–4 s) | o que a rede vê |
| **janela de saída** (0,5–1 s ou pose atual) | o que a rede prevê |

Recomendação:

- **offline:** use a janela de 4 s para análise/treino trial-level;
- **online:** use janela deslizante causal de 1–2 s, com predição futura de
  100–300 ms;
- não use a mesma janela de 4 s para tudo sem distinguir esses papéis.

O pré-cue de 0,5 s é ótimo para **baseline**, mas não deve virar “o modelo
aprende a esperar o cue”. Ele precisa funcionar com ritmo variável.

## 5. O ponto que ainda falta definir: qual é o label da MI?

Esta é a decisão mais importante da tese.

### Opção 1 — treinar em execução, testar em MI (zero-shot)

- treino: EEG de execução + pose medida;
- teste: EEG de MI, sem label de pose medido;
- avaliação: comparação com a pose-alvo/instrução e com a classificação.
- **vantagem:** não inventa label;
- **risco:** MI pode não reproduzir a trajetória de execução.

### Opção 2 — treinar em execução + fine-tune com poucas trials de MI

- poucas trials de MI com alvo instruído (pose do objeto/posição);
- **vantagem:** mais realista para BCI;
- **risco:** precisa de labels de MI; a pose “imaginada” não é medida.

### Opção 3 — treinar diretamente em MI usando o label da execução anterior

- cada trial de MI herda a pose medida da execução correspondente;
- **vantagem:** permite treinar o modelo em MI;
- **risco:** assume que a MI reproduz a execução; é uma hipótese forte, não um
  ground truth.

Minha recomendação: **Opção 1 como avaliação principal, Opção 2 como
fine-tune, Opção 3 só como análise exploratória.** Assim a tese não depende de
uma suposição não verificada.

## 6. O que o modelo aprende em cada fase

```text
FASE 1 — EXECUÇÃO
EEG(t) + pose medida(t)  ->  modelo aprende EEG -> pose

FASE 2 — MI ZERO-SHOT
EEG_MI(t)                ->  modelo prevê pose
comparar com alvo instruído / pose da execução anterior

FASE 3 — MI FEW-SHOT
poucas trials de MI com alvo -> ajuste fino

FASE 4 — ONLINE
EEG_MI -> pose -> mão robótica
```

## 7. Classificação MI como tarefa secundária

Usar as mesmas janelas de MI para classificar:

- mão direita;
- mão esquerda;
- repouso.

Isso dá um resultado mais robusto e comparável à literatura (Graz, EEGNeX), e
serve de sanity check do dataset. Mas não é o objetivo principal.

## 8. CEP

Mudanças em relação ao projeto atual:

- execução motora real (não só imagética);
- câmeras/multi-view para ground truth;
- sessões mais longas ou diferentes;
- novos desfechos (pose, orientação, abertura);
- possível EMG para monitorar execução;
- TCLE e riscos atualizados.

Isso exige **emenda ao CEP** antes de coletar. Não usar a aprovação antiga para
um protocolo novo.

## 9. Riscos e planos B

| Risco | Plano B |
| --- | --- |
| MI não transfere para pose | tese vira “pose offline em execução + classificação MI” |
| 4 s é janela ruim | testar 2 s e 3 s; usar janelas deslizantes |
| poucos participantes | piloto 3–5; LOSO; pré-treino WAY-EEG-GAL |
| labels de câmera ruins | multi-view + QC; descartar frames ruins |
| CEP atrasa | escrever metodologia e pipeline antes; coletar só depois |
| fadiga/ordem | randomizar, pausas, washout |
| modelo grande para 1.250 trials | SAND compacto + pré-treino + augmentation leve |

## 10. Decisões ainda abertas

1. O label da MI será alvo instruído, pose da execução anterior ou os dois?
2. O seeding será bloco execução→MI ou ME→MI no mesmo trial?
3. A janela de entrada será 4 s ou 2 s para o online?
4. 5 ou 6 participantes? 250 trials é por participante ou por run?
5. A classificação MI será capítulo separado ou apêndice?
6. O CEP será emenda com novo TCLE ou projeto novo?

## 11. O que a tese pode afirmar se tudo der certo

- é possível decodificar pose contínua de execução a partir de EEG;
- existe transferência parcial execução→MI;
- a classificação MI permanece viável nas mesmas janelas;
- o pipeline é compatível com controle online (se testado).

## 12. O que a tese não deve prometer

- controle de todos os dedos individualmente;
- controle online robusto em pacientes;
- generalização entre sujeitos sem calibração;
- que MI é equivalente à execução;
- que 1.250 trials são “suficientes” para qualquer arquitetura.
