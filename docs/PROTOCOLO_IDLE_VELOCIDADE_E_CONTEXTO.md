# Protocolo: condição IDLE, alvo de velocidade e ocupação de contexto

> Decisões de 18/09 (a partir da discussão sobre sinal de posição estática) e as
> consequências técnicas que elas trazem. Complementa
> `docs/ARQUITETURA_ONLINE_JANELA_HORIZONTE_E_MCU.md`.

## 1. Decisões tomadas

| Decisão | Status |
| --- | --- |
| Adicionar a condição **IDLE** (nenhum movimento, nenhuma mão) | **a implementar** (código 817 proposto) |
| Adotar o **alvo de velocidade** (m/s) como formulação principal | **implementado** (`--alvo velocidade`, teste 5/5) |
| Ancorar a janela no **onset** e usar **âncora mista** para incluir o IDLE | a implementar |
| Alongar o **repouso antes do cue** e a **pausa antes da MI** (seção 3) | decisão de protocolo a fechar |

### 1.1 Por que velocidade (e não posição)

1. **A mesma trajetória pode ser executada em velocidades/intensidades
   completamente diferentes** — e a informação neural que distingue "rápido" de
   "devagar" não existe no alvo de posição. A amplitude do MRCP/ERD escala com a
   força/velocidade executada (a revisão Wang 2023 registra: "the negative
   amplitude of MRCP is related to participants' level of participation in
   performing exercise tasks").
2. **No IDLE o alvo de velocidade é exatamente zero** — informação real. Com
   posição, o trial de repouso tem alvo constante e cai na armadilha do baseline
   trivial (medimos 0,90–0,98 de PCC para a trajetória média).
3. **É o que o controle de prótese consome**: integrar a velocidade estimada com
   Kalman/α-β devolve a posição **e** filtra tremor no regime de sustentação
   (hold), sem a deriva que "extrapolar posição parada" produziria.

### 1.2 Como a velocidade está implementada

- `--alvo velocidade` no treino: o alvo passa a ser o gradiente central do alvo
  de posição reamostrado, em **m/s** (`st.velocity_from_trajectory`);
- o checkpoint grava `alvo: "velocidade"` (a inferência precisa saber que deve
  integrar, não plotar direto);
- `TrajectoryNormalizer` já é agnóstico (z-score por eixo) e funciona igual;
- validação: rampa linear → velocidade exata; senoide 0,5 Hz/0,12 m → pico
  0,375 m/s (teórico 0,377); punho parado → **0 exato**; no caminho de arquivo o
  alvo de velocidade é idêntico ao gradiente do alvo de posição.

## 2. O cue de repouso: o que tem de ser igualado (e o que não)

O trial de IDLE **precisa de cue** (é o evento "agora não é nada → fico parado"
que queremos capturar). O cuidado é outro: o cue é um **estímulo visual** e gera
EEG visual (P1/N1/P300, sacadas, mudança de luminância) **time-locked** nele. Se o
cue do IDLE for visualmente diferente (cor, tamanho, posição, brilho, movimento),
um classificador separa "IDLE" de "movimento" por **canais occipitais** — a
acurácia vem da tela, não da intenção motora, e não sobrevive numa prótese sem
tela.

Duas medidas:

1. **Igualar a saliência sensorial**: mesma posição/duração/contraste/luminância e
   o mesmo instante de cue; muda **só o conteúdo semântico** (objeto vs. mão
   parada/relaxe). Ideal: mesmo molde visual, símbolo diferente.
2. **Análise de controle de pré-movimento**: treinar/avaliar o mesmo pipeline numa
   janela **sem movimento** (p.ex. `[cue − 0,5 s, cue + 0,5 s]`). Se as condições
   já forem discrimináveis ali, antes de qualquer movimento, então o sinal lido é
   **visual/antecipatório** — não motor. Complemento: olhar a **topografia**
   (motor → central/contralateral C3/C4/FC1; visual → occipital O1/O2/Pz).

## 3. Ocupação de contexto (medido com `tools/diagnostico_contexto.py`)

Estrutura atual do trial: **home 2 s → ME 4 s → pausa 2 s → MI 4 s**. Resultado
nas sessões (o mesmo vale para o dado real):

| Contexto causal | ME ancorada no onset (offset −0,5 s) | MI (janela de 4 s) |
| --- | --- | --- |
| 1,0 s | OK | OK |
| **2,0 s** | **invade 0,50 s do trial anterior** | OK |
| 3,0 s | invade 1,50 s | **1,00 s de ME do mesmo trial (33 %)** |
| 4,0 s | invade 2,50 s | **2,00 s (50 %)** |
| 6,0 s | invade 4,50 s | **4,00 s (67 %)** |
| 8,0 s | invade 6,50 s | **6,00 s (75 %)** |

Leitura e recomendações:

- **ME**: o "histórico limpo" antes do cue é o home (2 s) **menos** o offset da
  janela (0,5 s) → 1,5 s. Para contexto `C`, é preciso `home ≥ C + 0,5 s`
  (hoje o home de 2 s já invade 0,5 s com `C = 2 s`, o que é aceitável porque cai
  no fim do trial anterior, mas com `C ≥ 3 s` a contaminação fica relevante).
  Se adotarmos `C = 4 s`, o home deve ir a **4,5–5 s**.
- **MI (vazamento sério)**: a pausa de 2 s limita a MI a contexto ≤ 2 s. Para
  `C = 4 s`, metade da janela da MI seria a **execução do mesmo trial** — o modelo
  aprenderia a MI "espiando" o movimento executado.
  Três saídas (recomendo 1+3): (1) **pausa ≥ C** antes da MI; (2) **máscara** das
  amostras da ME na perda; (3) **contexto curto na MI** (2 s) e longo na ME,
  declarando a assimetria.
- Custo de sessão: +1 s de home e +2 s de pausa por trial ≈ +25 % de duração
  (250 trials). Vale mais que perder a validade da MI.

## 4. O que falta implementar (ordem)

| # | Item | Onde | Observação |
| --- | --- | --- | --- |
| 1 | condição **IDLE**: código 817, `--com-repouso`, cue neutra, sem vídeo de priming, sem onset | `eeg_motor_paradigm.py` + `build_trial_list` + `test_paradigm_protocol.py` | balanceamento sugerido: 1 IDLE a cada 6 trials de movimento |
| 2 | **âncora mista**: ancorar em 778 (cue) e "reancorar" no 795 quando ele existir na janela | `sand_traj_treino.py` | sem isso o IDLE fica fora do dataset (não tem onset) |
| 3 | **filtro causal** no treino (igual ao tempo real) | `sand_traj_treino.py` | tira o viés otimista do `sosfiltfilt` |
| 4 | **integração Kalman/α-β** da velocidade no tempo real | `sand_traj_tempo_real.py` | + `from_normalized` deixa de devolver posição e passa a devolver m/s |
| 5 | **taxa de falso movimento** nos períodos de baseline (olhos abertos/fechados) | `tools/` | métrica que os 7 artigos não reportam |
| 6 | alongar home/pausa conforme a seção 3 | protocolo | decisão a fechar com o orientador |
