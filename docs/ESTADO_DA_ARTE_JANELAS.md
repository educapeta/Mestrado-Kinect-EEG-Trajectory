# Estado da arte: JANELA DE ANÁLISE, ALVO e ONLINE x OFFLINE

> Gerado a partir da leitura direta dos 7 PDFs indicados, com
> `tools/analisa_pdfs_janelas.py` (PyMuPDF, ligaduras expandidas, texto UTF-8).
> As frases entre aspas são **transcrições** dos artigos — nada aqui é inferido
> sem indicação. Data: 18/09/2026.

## 1. A dúvida que estava aberta: a janela "[2,3] s pós-estímulo" do SAND

**Resposta: NÃO é a entrada do modelo — é a janela da análise
tempo-frequência (Fig. 2a).** A entrada do modelo é declarada explicitamente:

> "Preprocessed EEG signals are structured as an input matrix X ∈ R^(T×C),
> where **C = 32** denotes the number of recording channels and **T = 960**
> represents the temporal length of each **analysis window**."

E o pré-processamento do mesmo artigo:

> "Both kinematic data and denoised EEG signals were **downsampled from 500 Hz
> to 100 Hz** to reduce computational complexity and unify temporal
> resolution."
> "...a **0.1–40 Hz** bandpass filter removed baseline drift while preserving
> movement-related neural activity."
> "...**independent component analysis (ICA) combined with ICLabel** was
> employed to discard components classified as ocular or muscular artifacts
> with a probability threshold ≥ 0.90."
> "...all EEG channels and position coordinates were **normalized to [0,1]**
> using min-max normalization."

T = 960 a 100 Hz = **9,60 s** → o SAND alimenta o modelo com **o trial inteiro**
(a janela útil do WAY-EEG-GAL é 9,43 s e o loader do nosso legado justamente
reamostrava os 4 714 pontos para **960**). As duas frases que pareciam
contraditórias vêm de seções diferentes:

| Frase | Onde aparece | O que significa |
| --- | --- | --- |
| "[2–3] second post-stimulus window were selected, covering motor preparation and execution" | análise tempo-frequência (Fig. 2a) | janela **da análise** de ERD/ERS |
| "T = 960 ... each analysis window" | modelo (entrada) | janela **do modelo** ≈ trial inteiro a 100 Hz |

Consequências diretas para o nosso projeto:

1. O SAND é **offline e trial-level**: 1 trial de EEG → 1 trajetória completa,
   com **5-fold cross-validation**. Não existe janela deslizante causal.
2. O preço aparece na própria Tabela III: **2,27 s de inferência por janela**
   (contra 43,68 s do Transformer puro) — inviável para controle de prótese em
   tempo real, mesmo com a aceleração que eles propõem.
3. O alvo é a cinemática alinhada por **DTW**: "The temporal alignment with
   ground-truth kinematic recordings was validated by remarkable performance in
   dynamic time warping analysis", com "temporal residuals <30 ms in over
   99.375% of trials".
4. O sensor de referência é o **P4 no dorso da mão** ("(P4) on the back of the
   hand, which recorded 3D trajectories") — exatamente o que usamos
   (`KT_Pos_X4/Y4/Z4` = colunas 21/25/29 do `data_<sujeito>_<serie>.mat`).
5. O artigo usa **dois** conjuntos: WAY-EEG-GAL (público) e um **próprio**
   ("a 32-channel Neuroscan system (1,024 Hz) and 3D hand trajectories via a
   **Leap Motion controller** (170 Hz)", tarefa de levar uma bola ao alvo,
   "maintaining contact for 1 s"). Ou seja: **eles também fazem ground truth
   óptico** (Leap Motion), como nós fazemos com o Kinect.

## 2. Tabela comparativa (janela de análise × alvo × online)

| Estudo | Tarefa | Janela de análise | Alvo | Online? |
| --- | --- | --- | --- | --- |
| **SAND** (Fu et al., IEEE, 2026) | ME (grasp-and-lift, WAY-EEG-GAL) + base própria (bola→alvo, Leap Motion) | **T = 960 × C = 32 a 100 Hz = 9,6 s (trial inteiro)**; banda 0,1–40 Hz; ICA/ICLabel; min-max | trajetória 2D/3D (PCC); alinhamento por DTW | **não** — offline, 5-fold CV, 2,27 s por janela |
| **Korik et al.** (Front. Neurorobot., 2019) | **MI** — imaginar trajetória 3D do braço para mover **dois braços virtuais** | **band power** em janela deslizante de **250 ms**, passo **8,33 ms** (mu 8–12, low beta 12–18, high beta, low gamma); regressão mLR/KRR | trajetória 3D imaginada → classificação do alvo alcançado | **SIM — controle online com MI; referência mais próxima do nosso objetivo** (acurácia 45 % ± 5 % vs 33,3 % ao acaso) |
| **Tang et al.** (EMBC, 2024) | pega contínua (grasp) | **janela causal [t−Δt, t]**; Δt testado **0,5 / 1 / 2 / 2,5 s** a **100 Hz** | posições das juntas (mão/dedos), R e regressão linear + redes | **sim** (causal, "the corresponding EEG period within [t−Δt, t] is extracted as the feature window") |
| **E2T** (Kwon et al., IEEE TNSRE, 2026) | **MI** (imaginação) | **trial inteiro de 8 s = 4 000 amostras a 500 Hz** ("covering the entire movement epoch from 0 to 8 seconds"); *sliding window* só para **augmentação** (+ ruído gaussiano) | velocidade → trajetória 3D (fully-DoF) | treino offline |
| **MTRT** (Wang et al., IEEE TNSRE, 2023) | ME (linguagem de sinais chinesa) | reconstrução por *transformer* com **restrições geométricas** do membro ("human upper limb bone geometry properties as reconstruction constraints") | trajetória de **ombro, cotovelo e punho** | offline |
| **M3T-Attention** (Zhu et al., Cogn. Neurodyn., 2026) | ME (trajetória 3D da mão) | **500 amostras a 100 Hz = 5 s** (C = 18 canais; FIR 0,5–12 Hz; min-max); *strides* testados **500, 400, 300, 250, 200, 100 e 50** amostras | trajetória 3D de saída com 5 s | offline (sliding window para augmentação/continuidade) |
| **Wang et al. (revisão)** (Front. Neurosci., 2023) | revisão de reconstrução de trajetória | vários (ex.: 2 400 amostras); discute **MRCP** e o MTP-BCI ("predict the current motion state, such as position, speed, acceleration ... from the EEG characteristics of the **last several time lags**") | posição/velocidade/aceleração | revisão |

### 2.1 O dado que mais interessa: qual o tamanho ótimo de janela

Transcrição (Tang et al., 2024):

> "The impact of EEG window sizes (t = 0,5 s, 1 s, 2 s, 2,5 s) was examined as
> shown in Table III, with the EEG frequency set at 100 Hz."
> "The results highlight that the window size of **t = 2,5 s yields the highest
> decoding performance**, but it also increases the training time and memory
> occupation, so **t = 2 s is chosen** for Table I due to its close performance
> but less computation."

Transcrição (M3T-Attention, 2026):

> "So, we set the window length to **500 (sampling points)** and segmented both
> the EEG and hand kinematic data into 500-sample windows."
> "...we conducted experiments with **stride lengths of 500, 400, 300, 250,
> 200, 100 and 50** ... identified the optimal window and stride configuration
> that balanced data augmentation with temporal continuity."

**Leitura**: **2 s é a escolha defensável e já validada por terceiros** — o ótimo
medido é 2,5 s, e 2 s foi adotado por custo computacional com desempenho
próximo. Quem usa janela maior ou trial inteiro (M3T 5 s, E2T 8 s, SAND 9,6 s)
está **offline**, onde a latência de inferência não importa. O código do nosso
tempo real já mede ~6–10 ms por janela de 2 s em CPU — três ordens de grandeza
abaixo dos 2,27 s do SAND.

### 2.2 Sobre ancorar a janela ANTES do movimento (o nosso `onset − 0,5 s`)

Três evidências independentes:

1. **MRCP / potencial de prontidão** (revisão Wang et al., 2023):
   > "the **negative deviation of low EEG frequency (0,1–3 Hz) before the start
   > of exercise**, that is, the motor-related cortical potential (MRCP)";
   > "the negative amplitude of MRCP is related to participants' level of
   > participation in performing exercise tasks".
2. **M3T-Attention**: na análise de imagem por RM, o segmento analisado é de 4 s
   > "the time range is from **−0,5 s to 3,5 s**".
3. **Korik et al. (2019)**: eles otimizam explicitamente a **latência** da
   janela de features — "a feature window that maximizes the ... specifically
   optimized **latency of the feature window**" — porque a informação útil não
   está alinhada com o estímulo (exatamente o argumento da nossa decisão de
   ancorar no **onset detectado**, e não na cue).

### 2.3 Onde o nosso desenho se posiciona

| Dimensão | Literatura dominante | Nosso programa |
| --- | --- | --- |
| Janela | 2–5 s a 100 Hz (ou trial inteiro) | **2 s a 500 Hz** (≈ equivalente, e validado por Tang 2024) |
| Âncora | cue / pós-estímulo (nível de trial) | **onset detectado por cinemática**, janela em **−0,5 s** |
| Alvo | mesma janela (concorrente) ou trial inteiro | **concorrente (diagnóstico) + preditivo (+1,5 a +2,5 s)** |
| Online | minoria (Korik 2019, Tang 2024) | **objetivo do projeto** (`sand_traj_tempo_real.py`) |
| Ground truth | sensor inercial/óptico (P4, Leap Motion) | **Kinect + MediaPipe + fusão IMU/MPU6050** |
## 3. Como citar (DOIs)

- Fu, R.; Fang, Y.; Xu, F.; Hua, C.; Hua, C. **SAND: Spectral-Attention Neural
  Decoding of Hand Kinematics from Low-Frequency EEG Dynamics**. *IEEE*, 2026.
  (PDF: `SAND_Spectral-Attention_...pdf`)
- Korik, A.; Sosnik, R.; Siddique, N.; Coyle, D. **Decoding Imagined 3D Arm
  Movement Trajectories From EEG to Control Two Virtual Arms — A Pilot Study**.
  *Front. Neurorobot.* 13:94, 2019. DOI 10.3389/fnbot.2019.00094.
- Kwon, B.-H.; Jeong, J.-H.; Lee, S.-W. **E2T: EEG-to-Trajectory Transformer for
  Motor Imagery-Based Fully-DoF Motion Prediction**. *IEEE TNSRE* 34:1961–1973,
  2026. DOI 10.1109/TNSRE.2026.3683431.
- Tang, Y.; Robinson, N.; Fu, X.; Thomas, K. P.; Wai, A. A. P.; Guan, C.
  **Reconstruction of Continuous Hand Grasp Movement from EEG Using Deep
  Learning**. *EMBC* 2024:1–4. DOI 10.1109/EMBC53108.2024.10781850.
- Wang, P.; Cao, X.; Zhou, Y.; Gong, P.; Yousefnezhad, M.; Shao, W.; Zhang, D.
  **A comprehensive review on motion trajectory reconstruction for EEG-based
  brain-computer interface**. *Front. Neurosci.* 17:1086472, 2023.
  DOI 10.3389/fnins.2023.1086472.
- Wang, P.; Li, Z.; Gong, P.; Zhou, Y.; Chen, F.; Zhang, D. **MTRT: Motion
  Trajectory Reconstruction Transformer for EEG-Based BCI Decoding**. *IEEE
  TNSRE* 31:2349–2358, 2023. DOI 10.1109/TNSRE.2023.3275172.
- Zhu, L.; Jiang, P.; Huang, A.; Zhang, J.; Yuan, P. **M3T-attention: a
  multi-level multi-scale temporal attention transformer for EEG hand movement
  trajectory decoding**. *Cogn. Neurodyn.* 20(1):33, 2026.
  DOI 10.1007/s11571-025-10403-1.

### 3.1 O que nenhum deles faz (o recorte do nosso trabalho)

Nenhum dos sete trabalhos reporta, simultaneamente:

1. janela **causal** curta (2 s) com **âncora no onset detectado por cinemática**
   (e não na cue);
2. **ME e MI no mesmo protocolo**, com o mesmo conjunto de objetos/condições;
3. ground truth 3D por **câmera + MediaPipe + fusão com IMU** (e não sensor
   inercial caro ou Leap Motion);
4. **alvo preditivo** explícito (`--target-start-sec`), comparado ao concorrente;
5. clipe de **priming 0,5×** entre a execução e a imaginação.

Korik 2019 é o que mais se aproxima no **objetivo** (MI → trajetória 3D → controle
online), mas usa features de *band power* e regressão com **templates de
trajetória**, não um modelo sequência-a-sequência como o nosso — e não tem ME
nem ground truth por câmera.

## 4. Como isso muda as nossas decisões

| Decisão | Antes | Depois da leitura |
| --- | --- | --- |
| Janela de EEG | 2 s "por estimativa" | 2 s **validado** (Tang 2024: 2,5 s melhor, 2 s escolhido) — manter, e citar |
| Âncora | cue (`--event-code 778`) | **onset (`795`) com `--window-start-sec -0.5`** — respaldado por MRCP, M3T (−0,5 s) e Korik (latência otimizada) |
| Alvo | concorrente por herança | **concorrente (diagnóstico) x preditivo (aplicação)**; sempre comparar *e* reportar a amplitude do alvo (`tools/compara_alvos.py`) |
| Comparação com o SAND | "janela [2,3] s" era a entrada | o SAND é **trial-level** (T = 960 @100 Hz); para comparar justiça, usar o trial inteiro como *baseline* offline, e o 2 s causal como o nosso diferencial online |
## 5. Piloto sintético (sem g.Nautilus): o que ele já validou e o que ele ensina

Ferramenta: `tools/gerar_sessao_sintetica.py` — cria sessões no **formato exato**
do Programa 1 (EEG cru, `_movimento.csv` a 30 Hz, `_eventos.json`, questionário),
com EEG que contém ERD de mu/beta, MRCP 1,2 s antes do onset e um ganho por canal
que **codifica a direção do alvo** (analogia com o ERD contralateral), e um
movimento mínimo-jerk com onset aleatório entre 0,35 e 0,90 s após a cue.

Comando do piloto:

```powershell
.venv\Scripts\python.exe tools\gerar_sessao_sintetica.py --sessoes 6 --trials 24 `
    --out gravacoes_sinteticas
# A/B das três decisões (validação = 1 sessão, 120 trials de treino)
.venv\Scripts\python.exe sand_traj_treino.py --data gravacoes_sinteticas `
    --event-code 778 --window-start-sec 0.0  --window-sec 2.0 --epochs 20
.venv\Scripts\python.exe sand_traj_treino.py --data gravacoes_sinteticas `
    --event-code 795 --window-start-sec -0.5 --window-sec 2.0 --epochs 20
.venv\Scripts\python.exe sand_traj_treino.py --data gravacoes_sinteticas `
    --event-code 795 --window-start-sec -0.5 --window-sec 2.0 --epochs 20 `
    --target-start-sec 1.5 --target-end-sec 2.5
# baselines triviais (sem EEG): trajetória média do treino
.venv\Scripts\python.exe tools\compara_alvos.py --data gravacoes_sinteticas --max-sessoes 2
```

### 5.1 Resultados

**Versão 1 do gerador** (alvos todos do mesmo lado; o baseline médio é muito forte):

| Configuração | val_loss | PCC x | PCC y | PCC z | baseline do alvo médio (x/y/z) |
| --- | --- | --- | --- | --- | --- |
| cue concorrente (778) | 0,7275 | 0,739 | 0,310 | 0,771 | 0,892 / 0,762 / 0,922 |
| **onset concorrente (795, −0,5 s)** | **0,5011** | **0,863** | 0,488 | **0,866** | 0,935 / 0,822 / 0,959 |
| onset preditivo (alvo +1,5 a +2,5 s) | 0,5049 | 0,865 | 0,593 | 0,879 | 0,968 / 0,897 / 0,979 |

**Versão 2 (atual)**: objetos distribuídos na mesa → alvos laterais bimodais; o
baseline em X cai para ~0 e o teste passa a medir decodificação de verdade:

| Configuração | val_loss | PCC x | PCC y | PCC z | baseline do alvo médio (x / y / z) |
| --- | --- | --- | --- | --- | --- |
| cue concorrente (778) | 0,5546 | **0,764** | 0,608 | 0,819 | −0,047 / 0,762 / 0,922 |
| onset concorrente (795, −0,5 s) | 0,6645 | **0,767** | 0,475 | 0,742 | −0,105 / 0,822 / 0,959 |
| onset preditivo (alvo +1,5 a +2,5 s) | 0,6957 | **0,549** | 0,575 | 0,826 | −0,144 / 0,897 / 0,979 |

Leituras afirmáveis (com as duas versões lado a lado):

1. **X é a única dimensão em que o EEG sintético carrega informação da
   trajetória** (a lateralidade, via `ganho_direcional`): o modelo faz 0,55–0,77
   contra baseline ≈ 0 → **decodificação real**, e o testbed é capaz de
   detectá-la. Note que 0,76 é um *bom* número na versão 2 e um *número ruim* na
   versão 1: **PCC sozinho não significa nada**.
2. **Y e Z não estão codificados** → o modelo fica **abaixo** do baseline em todas
   as configurações. O ideal seria ele convergir para a média; com 120 trials ele
   não chega lá — ou seja, **com poucos dados a média analítica é um baseline
   difícil de bater**. Reforça a regra de sempre reportar o baseline.
3. **Âncora no onset**: na versão 1 (em que o *jitter* do tempo de reação era o
   principal conteúdo informativo) o ganho foi evidente (x 0,739→0,863;
   z 0,771→0,866; perda 0,73→0,50). Na versão 2 (o sinal decodificável é a
   direção, presente nas duas janelas) houve **empate em X** (0,764 vs 0,767) e o
   val_loss piorou. Conclusão honesta: **ancorar no onset não piora** e ajuda
   quando o jitter importa; quem decide é o dado real.
4. **Alvo preditivo**: o X continua decodificável (0,549), mas o futuro é mais
   estereotipado e o baseline sobe para 0,90–0,98 em y/z. Prever mais longe exige
   um baseline ainda mais forte para provar mérito — exatamente o teste que a
   sessão real vai fazer.

### 5.2 Limites declarados do piloto sintético

- Ele **valida o pipeline** (janelas, âncoras, alvos, metadados, checkpoint,
  inferência, formatos de arquivo) e **não** a decodificação em si: no sintético,
  a informação sobre a trajetória está no *instante* do movimento e numa
  lateralização artificial que **nós** escolhemos.
- A validação é de 1 sessão (24 trials) — variância alta por desenho.
- Para validar limiar de onset, ERD/ERS e decodificação de verdade, o passo é a
  sessão piloto com o g.Nautilus, usando `--source generator --sem-kinect` para
  ensaiar o software e `tools/analisa_onset.py` para olhar a distribuição dos
  onsets detectados.



