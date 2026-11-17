# Relatório de Datasets EEG para Decodificação de Pose/Cinemática da Mão

**Projeto:** Mestrado em Decodificação de Pose/Cinemática da Mão a partir de EEG  
**Data:** 12/11/2026  
**Contexto:** Decodificar pose final/hold da mão (posição 3D, rotação 6D, abertura) a partir de EEG, com execução motora (ME) como referência e imagética motora (MI) subsequente; velocidade linear auxiliar; classificação MI secundária.  
**Setup alvo:** 4 classes (2 objetos × 2 mãos), 2 sessões, 200 trials/participante, 6 participantes, g.Nautilus 32 canais a 500 Hz, câmeras multi-view para pose.  
**Dataset base conhecido:** WAY-EEG-GAL (NÃO incluído neste relatório).
> **Revisão do Lead (12/11/2026):** este relatório foi produzido por um
> subagente. As fichas de MOVING2024, IACKD, Thapa2025, GuttmannFlury2025,
> BCIComp2020 e BNCI são **candidatos a verificar**, não recomendações finais.
> Vários detalhes de canais, taxa, licença e labels foram inferidos de
> resumos/repositórios e precisam ser confirmados no artigo/portal antes de
> baixar ou usar. O artigo do DOI `s41597-026-07242-y` foi identificado e
> corrigido na seção 1.


---

## 1. Dataset do artigo Scientific Data 2026 (DOI s41597-026-07242-y)

> **Correção do Lead (12/11/2026):** o DOI **não é 404**. O artigo existe:
> **“High-Density EEG and Multi-Muscle EMG Dataset during Object Prehension
> with a sensorized Grasping Box in Humans”**, Scientific Data (2026),
> DOI `10.1038/s41597-026-07242-y`. A página da Nature/Springer redireciona
> para autenticação (`idp.nature.com`/`idp.springer.com`), então o texto completo
> não foi lido nesta revisão. Os dados abaixo são **apenas o que o título/escopo
> permitem afirmar**; participantes, canais, taxa, trials e licença precisam ser
> confirmados no artigo/portal de dados antes de usar.

| Campo | Informação |
|---|---|
| **Título** | High-Density EEG and Multi-Muscle EMG Dataset during Object Prehension with a sensorized Grasping Box in Humans |
| **DOI** | 10.1038/s41597-026-07242-y |
| **Modalidade** | EEG de alta densidade + EMG multi-músculo; preensão de objetos com caixa instrumentada |
| **Tarefa** | Preensão/alcance de objetos (execução), com sensores na caixa |
| **MI?** | Não confirmado — o título/escopo aponta para execução/preensão, não imagética |
| **Cinemática/pose 3D?** | Não confirmado; a caixa instrumentada fornece força/eventos de preensão, não necessariamente pose 6D da mão |
| **Acesso/licença** | Confirmar no artigo/portal; página exige autenticação |
| **Relevância para nós** | Média–alta para **pré-treino de preensão/EMG** e eventos de grasp; baixa para pose contínua enquanto não houver label 3D |
| **Limitações** | Não é MI; provavelmente não tem pose 6D/abertura; acesso a confirmar |

**Ação necessária:** baixar o artigo via acesso institucional/contato com autores e preencher os campos faltantes. Não usar este dataset como se tivesse pose/MI até confirmar.

## 2. Datasets Candidatos Prioritários

### 2.1 IACKD — Intention-Action Conflict EEG-Hand Kinematics Dataset (ds006840 / on006840)

| Campo | Informação |
|-------|------------|
| **Nome** | IACKD (Intention Action Conflict EEG-Hand Kinematics Dataset) |
| **Artigo/URL** | Scientific Data 2026, DOI: [10.1038/s41597-026-07146-x](https://doi.org/10.1038/s41597-026-07146-x) • OpenNeuro: [ds006840](https://openneuro.org/datasets/ds006840) • NEMAR: [on006840](https://nemar.org/dataset/on006840) |
| **Modalidade** | EEG (26 canais, 10-10) + cinemática de mão (trajetória contínua) |
| **Participantes** | 15 saudáveis |
| **Tarefas** | Controle unimanual sob condições congruentes/incongruentes (conflito intenção-ação); alcance e preensão |
| **Canais/Taxa** | 26 EEG (10-10), taxa não explicitada no resumo NEMAR (verificar BrainVision) |
| **Labels** | **Trajetória contínua da mão** (cinemática 3D), condições congruente/incongruente, cues visuais |
| **Acesso/Licença** | **Aberto (CC0)** — download via NEMAR CLI, OpenNeuro, DataLad, zip 6.4 GB |
| **Adequação ao alvo** | **ALTA** — EEG + cinemática contínua de mão, tarefa de alcance/preensão, licença aberta, BIDS |
| **Riscos** | - Apenas 15 participantes<br>- 26 canais (vs. 32 alvo) — montagem 10-10 diferente<br>- Taxa de amostragem a confirmar<br>- Paradigma de conflito intenção-ação ≠ ME/MI puro<br>- Cinemática pode ser 3D posição, não pose completa (rotação 6D + abertura) |
| **Prioridade** | **ALTA** |

---

### 2.2 MOVING2024 — Multi-Modal EEG + Virtual Glove Hand Tracking

| Campo | Informação |
|-------|------------|
| **Nome** | MOVING2024 |
| **Artigo/URL** | Sensors 2024, DOI: [10.3390/s24165207](https://doi.org/10.3390/s24165207) • Zenodo: [10.5281/zenodo.12804784](https://doi.org/10.5281/zenodo.12804784) • MOABB: `MOVING2024` |
| **Modalidade** | EEG seco 32 canais (Enobio, 10-20) + **Virtual Glove** (2× Leap Motion) rastreamento cinemático da mão |
| **Participantes** | 11 saudáveis |
| **Tarefas** | 3 movimentos mão direita: abrir/fechar, rotação punho, tamborilar dedos — **cada um como MI e ME** (rest → MI → ME) |
| **Canais/Taxa** | 32 EEG secos + 3 acelerômetro (misc), **500 Hz**, referência CMS/DRL, notch 50 Hz |
| **Labels** | **Classes discretas** (rest, open_close, wrist_rotation, finger_tapping) × 2 modalidades (MI/ME); **cinemática contínua via Leap Motion** (posição/orientação dedos/palma) |
| **Acesso/Licença** | **Aberto (CC BY 4.0)** — Zenodo, download via MOABB |
| **Adequação ao alvo** | **ALTA** — 32 canais a 500 Hz (match exato g.Nautilus), MI + ME pareados, rastreamento de mão (Leap Motion), protocolo rest→MI→ME alinhado ao nosso |
| **Riscos** | - Apenas 11 participantes, 1 sessão<br>- Apenas mão direita<br>- Movimentos simples (abrir/fechar, rotação, tamborilar) — não grasp de objetos com pose 3D/6D<br>- Leap Motion ≠ câmeras multi-view + marcadores (ruído, oclusão)<br>- Labels de classe, não pose contínua hold final (embora cinemática bruta exista) |
| **Prioridade** | **ALTA** |

---

### 2.3 Thapa2025 — Large EEG Database of Freewill Reaching and Grasping

| Campo | Informação |
|-------|------------|
| **Nome** | Thapa2025 (Freewill Reach-and-Grasp) |
| **Artigo/URL** | Scientific Data 2025, DOI: [10.1038/s41597-025-06039-9](https://doi.org/10.1038/s41597-025-06039-9) • Figshare: [10.6084/m9.figshare.28632599](https://doi.org/10.6084/m9.figshare.28632599) • MOABB: `Thapa2025` |
| **Modalidade** | EEG 31 canais (actiCHamp Plus, 10-20 estendido) + **alvo de preensão** (4 copos: 2 com água, 2 vazios) |
| **Participantes** | 23 saudáveis (18-24 anos, 8F/15M, destros) |
| **Tarefas** | **Alcance e preensão em livre-arbítrio** — escolha livre de 1 de 4 copos, tempo auto-iniciado (~1-2s após cue auditivo) |
| **Canais/Taxa** | 31 EEG + 4 EOG + 1 áudio + 3 acelerômetro (ulna), **250 Hz** (1000 Hz para 2 sujeitos) |
| **Labels** | **Classes discretas** (Tgt1–Tgt4 = copo alvo), **livre-arbítrio** (não cued), marcador de início de movimento via acelerômetro |
| **Acesso/Licença** | **Aberto (CC BY 4.0)** — Figshare, BIDS (BrainVision), MOABB |
| **Adequação ao alvo** | **MÉDIA-ALTA** — Maior coorte (23), tarefa alcance/preensão naturalística, multi-sessão (49 sessões, 6808 trials), acel. marca movement onset |
| **Riscos** | - **Apenas ME** (sem MI)<br>- **Sem cinemática contínua da mão** — apenas classe do alvo (copo)<br>- 31 canais a 250 Hz (vs. 32/500 Hz)<br>- Livre-arbítrio ≠ paradigma cued ME→MI<br>- Objetos simples (copos), não pose 6D + abertura |
| **Prioridade** | **MÉDIA** (bom para pré-treino ME, mas sem MI nem pose contínua) |

---

### 2.4 BNCI2025-002 — Continuous 2D Trajectory Decoding (Attempted Movement)

| Campo | Informação |
|-------|------------|
| **Nome** | BNCI2025-002 |
| **Artigo/URL** | J. Neural Eng. 2022, DOI: [10.1088/1741-2552/ac689f](https://doi.org/10.1088/1741-2552/ac689f) • MOABB: `BNCI2025_002` |
| **Modalidade** | EEG 60 canais (actiCAP, 10-10 modificado) + **trajetória 2D contínua** (alvo na tela) |
| **Participantes** | 10 saudáveis + 1 lesionado medular (SCI) |
| **Tarefas** | **Movimento tentado** (braço imobilizado) — rastrear alvo em movimento (snake run) ou traçar formas (free run) com feedback visual variável (0/50/100%) |
| **Canais/Taxa** | 60 EEG + 4 EOG, **200 Hz** (original), pré-processado para 20 Hz |
| **Labels** | **Trajetória 2D contínua** (x, y) decodificada de EEG, 3 classes de run (snake, free, eye) |
| **Acesso/Licença** | **Aberto (CC BY 4.0)** — BNCI server (apenas 2/20 sujeitos disponíveis), MOABB |
| **Adequação ao alvo** | **MÉDIA** — Único com **labels de trajetória contínua** para decodificação regressiva, paradigma "attempted movement" relevante para MI |
| **Riscos** | - **Apenas 2 sujeitos disponíveis** publicamente<br>- Trajetória 2D tela ≠ pose 3D/6D mão real<br>- 60 canais a 200 Hz (downsampled para 20 Hz no paper)<br>- Movimento tentado (braço imobilizado) ≠ ME natural + MI<br>- Feedback visual variável confunde decodificação |
| **Prioridade** | **MÉDIA** (valioso para decodificação contínua, mas dados limitados) |

---

### 2.5 GuttmannFlury2025 (MI + ME) — EEG + Eye-tracking + High-speed Video

| Campo | Informação |
|-------|------------|
| **Nome** | GuttmannFlury2025_MI / GuttmannFlury2025_ME |
| **Artigo/URL** | Scientific Data 2025, DOI: [10.1038/s41597-025-04861-9](https://doi.org/10.1038/s41597-025-04861-9) • MOABB: `GuttmannFlury2025_MI` e `GuttmannFlury2025_ME` |
| **Modalidade** | EEG 62 canais (Neuroscan Quik-Cap, SynAmps2) + eye-tracking + vídeo alta velocidade |
| **Participantes** | 31 saudáveis (20-57 anos, média 28.3) |
| **Tarefas** | **MI e ME de preensão** esquerda/direita (2 classes) — cue visual retângulo, 2s fixação + 4s imaginação/execução |
| **Canais/Taxa** | 62 EEG + 1 EOG, **1000 Hz**, referência mastoide direito |
| **Labels** | **Classes discretas** (left_hand, right_hand) para MI e ME separados, 40 trials/classe/sessão, 3 sessões |
| **Acesso/Licença** | **Aberto** — Zenodo (re-hospedado do Synapse), BDF, MOABB |
| **Adequação ao alvo** | **MÉDIA** — Maior coorte com MI+ME pareados, alta taxa (1000 Hz), multi-sessão |
| **Riscos** | - **Apenas 2 classes** (esquerda/direita), não 4 (objeto×mão)<br>- **Sem cinemática/pose de mão** — apenas classe de lateralidade<br>- 62 canais a 1000 Hz (downsample necessário)<br>- Paradigma simples preensão genérica, não objetos específicos com pose |
| **Prioridade** | **MÉDIA** (bom para transfer MI↔ME, mas labels fracos para pose) |

---

### 2.6 BCIComp2020UpperLimb — BCI Competition 2020 Track 4 (Grasping MI)

| Campo | Informação |
|-------|------------|
| **Nome** | BCIComp2020UpperLimb |
| **Artigo/URL** | Front. Hum. Neurosci. 2022, DOI: [10.3389/fnhum.2022.898300](https://doi.org/10.3389/fnhum.2022.898300) • OSF: [pq7vb](https://osf.io/pq7vb/) • MOABB: `BCIComp2020UpperLimb` |
| **Modalidade** | EEG 60 canais (BrainAmp, 10-20) |
| **Participantes** | 15 saudáveis (20-34 anos, destros) |
| **Tarefas** | **MI de 3 tipos de preensão** mão direita: cilíndrica (copo), esférica (bola), lumbrical (cartão) |
| **Canais/Taxa** | 60 EEG, **250 Hz**, FCz ref, Fpz ground, notch 60 Hz |
| **Labels** | **Classes discretas** (cylindrical, spherical, lumbrical), 50 trials/classe/sessão, 3 sessões (dias 1, 8, 15) |
| **Acesso/Licença** | **Aberto (CC BY 4.0)** — OSF, .mat, MOABB |
| **Adequação ao alvo** | **MÉDIA** — 3 tipos de grasp semanticamente relevantes, multi-sessão (transfer sessão-a-sessão) |
| **Riscos** | - **Apenas MI** (sem ME, sem cinemática/pose)<br>- Apenas mão direita, 3 classes (não 4 objeto×mão)<br>- 60 canais a 250 Hz<br>- Dia 3 (teste) tem cue visual zerado — labels só via answer sheet |
| **Prioridade** | **MÉDIA** (bom para classificação grasp MI, mas sem pose contínua) |

---

### 2.7 MILimbEEG — Upper/Lower Limb MI/ME (OpenBCI 16ch)

| Campo | Informação |
|-------|------------|
| **Nome** | MILimbEEG |
| **Artigo/URL** | Data in Brief 2023, DOI: [10.1016/j.dib.2023.109540](https://doi.org/10.1016/j.dib.2023.109540) • Mendeley: [10.17632/x8psbz3f6x.2](https://doi.org/10.17632/x8psbz3f6x.2) • MOABB: `MILimbEEG` |
| **Modalidade** | EEG 16 canais (OpenBCI Cyton+Daisy, seco) |
| **Participantes** | 60 (população mista: 2 amputados, 1 hidrocefalia, 18 pós-COVID, 39 saudáveis) |
| **Tarefas** | **ME seguido de MI**: fechar mão, flexão dorsal/plantar pé — 8 classes (beo, clh, crh, dlf, plf, drf, prf, rest) |
| **Canais/Taxa** | 16 EEG secos, **125 Hz**, monopolar orelha, band-pass 5-50 Hz, notch 60 Hz |
| **Labels** | **Classes discretas** (MI apenas no loader MOABB), ME disponível nos CSVs originais |
| **Acesso/Licença** | **Aberto (CC BY 4.0)** — Mendeley Data (URL assinada, download manual) |
| **Adequação ao alvo** | **BAIXA** — Grande N, mas 16 canais/125 Hz muito abaixo do alvo, população heterogênea clínica |
| **Riscos** | - 16 canais, 125 Hz (incompatível)<br>- Eletrodos secos OpenBCI (ruído alto)<br>- População não homogênea saudável<br>- Sem cinemática/pose<br>- Download problemático (Mendeley signed URL) |
| **Prioridade** | **BAIXA** |

---

### 2.8 Pan2025 — Cross-Session MI (Left/Right Hand)

| Campo | Informação |
|-------|------------|
| **Nome** | Pan2025 |
| **Artigo/URL** | Harvard Dataverse, DOI: [10.7910/DVN/GH74ZG](https://doi.org/10.7910/DVN/GH74ZG) • Ref. paper: J. Neural Eng. 2023, DOI: [10.1088/1741-2552/ad0a01](https://doi.org/10.1088/1741-2552/ad0a01) • MOABB: `Pan2025` |
| **Modalidade** | EEG 28 canais (Neuroscan SynAmps2, 10-10 sensorimotor) |
| **Participantes** | 10 saudáveis (22-25 anos, 3F, 2 canhotos) |
| **Tarefas** | MI cueada mão esquerda vs. direita, 2 sessões (dias separados), ~180 trials/sessão |
| **Canais/Taxa** | 28 EEG, **250 Hz** (downsample de 1000 Hz), band-pass 0.01-200 Hz, notch 50 Hz |
| **Labels** | **Classes discretas** (left_hand, right_hand), sessão 2 tem trials treino + teste c/ feedback online |
| **Acesso/Licença** | **Aberto (CC0 1.0)** — Harvard Dataverse, .mat, MOABB |
| **Adequação ao alvo** | **BAIXA** — Apenas 2 classes MI, sem ME, sem cinemática, 28ch/250Hz |
| **Riscos** | - Muito simples para nosso alvo (pose 4 classes + ME+MI) |
| **Prioridade** | **BAIXA** |

---

### 2.9 Reach&Grasp (Di Domenico et al. 2025) — EMG + Cinemática + TÁTIL (SEM EEG)

| Campo | Informação |
|-------|------------|
| **Nome** | Reach&Grasp |
| **Artigo/URL** | Scientific Data 2025, DOI: [10.1038/s41597-025-04552-5](https://doi.org/10.1038/s41597-025-04552-5) • IIT Dataverse |
| **Modalidade** | **EMG** (HD 64ch + bipolar 10ch) + **Cinemática** (Vicon 23 marcadores + luva sensorizada CyberGlove) + **Tátil** (58 taxels) |
| **Participantes** | 10 saudáveis (24-36 anos, 4F/6M, destros) |
| **Tarefas** | 16 tarefas: movimentos simples (abrir/fechar, pronação/supinação, grasps) + complexos (alcance+preensão, manipulação) |
| **Canais/Taxa** | EMG 2kHz, Cinemática Vicon 100Hz + CyberGlove 100Hz, Tátil 100Hz |
| **Labels** | **Cinemática completa mão/braço** (ângulos articulares Vicon + luva), forças táteis, 10 reps/tarefa |
| **Acesso/Licença** | **Aberto** — IIT Dataverse, BIDS |
| **Adequação ao alvo** | **N/A — SEM EEG** |
| **Riscos** | **Não tem EEG** — apenas EMG + cinemática + tátil. Útil apenas se fusão multimodal futura. |
| **Prioridade** | **N/A** (registrado para completude) |

---

## 3. Tabela Comparativa Resumida

| Dataset | EEG Canais/Hz | Participantes | MI | ME | Cinemática/Pose Mão | License | Acesso | Prioridade |
|---------|---------------|---------------|----|----|---------------------|---------|--------|------------|
| **IACKD (ds006840)** | 26 ch, taxa? | 15 | Implícito (conflito) | Sim (execução) | **Trajetória 3D contínua** | CC0 | Aberto (NEMAR/OpenNeuro) | **ALTA** |
| **MOVING2024** | **32 ch, 500 Hz** | 11 | **Sim** | **Sim** | **Leap Motion (contínua)** | CC BY 4.0 | Aberto (Zenodo/MOABB) | **ALTA** |
| **Thapa2025** | 31 ch, 250 Hz | 23 | Não | **Sim** | Apenas classe alvo (4 copos) | CC BY 4.0 | Aberto (Figshare/MOABB) | MÉDIA |
| **BNCI2025-002** | 60 ch, 200 Hz | 2/10+1 SCI | Tentado | Não (braço imobilizado) | **Trajetória 2D contínua (tela)** | CC BY 4.0 | Parcial (BNCI server) | MÉDIA |
| **GuttmannFlury2025** | 62 ch, 1000 Hz | 31 | **Sim** | **Sim** | Apenas lateralidade (E/D) | Aberto | Zenodo/MOABB | MÉDIA |
| **BCIComp2020UpperLimb** | 60 ch, 250 Hz | 15 | **Sim** (3 grasps) | Não | Apenas classe grasp (3) | CC BY 4.0 | Aberto (OSF/MOABB) | MÉDIA |
| **MILimbEEG** | 16 ch, 125 Hz | 60 | Sim | Sim (CSVs) | Não | CC BY 4.0 | Mendeley (manual) | BAIXA |
| **Pan2025** | 28 ch, 250 Hz | 10 | Sim (E/D) | Não | Não | CC0 | Dataverse/MOABB | BAIXA |
| **Reach&Grasp** | **NÃO TEM EEG** | 10 | N/A | Sim | **Cinemática completa (Vicon+luva)** | Aberto | IIT Dataverse | N/A |

---

## 4. Ranking e Recomendação dos Top 3 para Pré-Treino/Transfer Learning

### 🥇 1º Lugar: **MOVING2024** (Prioridade ALTA)
**Por que:** Match quase perfeito de hardware (32 ch, 500 Hz, 10-20), paradigma **rest → MI → ME** idêntico ao nosso, rastreamento de mão contínuo (Leap Motion), licença aberta, disponível no MOABB.

**O que falta para uso:**
- [ ] Verificar taxa de amostragem exata dos arquivos EDF (MOABB reporta 500 Hz)
- [ ] Alinhar montagem 10-20 Enobio → 10-10 g.Nautilus (mapeamento canais)
- [ ] Extrair cinemática bruta Leap Motion (posição 3D palma + orientação + abertura dedos) dos arquivos brutos Zenodo
- [ ] Definir janela "hold final" nos trials ME (últimos 1-2s do período 6s de ação)
- [ ] Criar labels de pose contínua (posição 3D + rotação 6D + abertura) a partir do Leap Motion
- [ ] Apenas 11 sujeitos, 1 sessão → augmentação / data synthesis necessária
- [ ] Apenas mão direita → para generalizar mão esquerda, precisa data augmentation (espelhamento) ou outro dataset

---

### 🥈 2º Lugar: **IACKD (ds006840 / on006840)** (Prioridade ALTA)
**Por que:** Único com **trajetória 3D contínua da mão** + EEG em tarefa alcance/preensão, licença CC0 (domínio público), 15 participantes, BIDS completo, disponível via NEMAR/OpenNeuro.

**O que falta para uso:**
- [ ] Confirmar taxa de amostragem EEG (não explicitada no NEMAR; ler headers BrainVision)
- [ ] Mapear 26 canais 10-10 → 32 canais g.Nautilus (interpolação espacial ou subseleção)
- [ ] Entender formato da cinemática: é posição 3D da ponta dos dedos? Palma? Marcadores?
- [ ] Paradigma de "conflito intenção-ação" ≠ ME puro + MI puro → precisa isolar trials congruentes (intenção=ação)
- [ ] Verificar se há dados de MI pura (condição incongruente pode servir como proxy)
- [ ] Download 6.4 GB — planejar armazenamento/processamento

---

### 🥉 3º Lugar: **Thapa2025** (Prioridade MÉDIA-ALTA para pré-treino ME)
**Por que:** Maior coorte (23), multi-sessão (49 sessões), tarefa alcance/preensão naturalística livre-arbítrio, acelerômetro marca movement onset, 6808 trials, BIDS, MOABB-ready.

**O que falta para uso:**
- [ ] **Não tem MI** — só serve para pré-treino de features ME (encoder compartilhado)
- [ ] **Sem cinemática contínua** — apenas classe do alvo (4 copos). Para pose, precisaria fusão com outro dataset ou pseudo-labels
- [ ] 31 canais a 250 Hz → downsample/upsample + interpolação espacial para 32ch/500Hz
- [ ] Livre-arbítrio → alinhamento temporal variável; usar acelerômetro para epoching relativo ao movement onset
- [ ] Objetos simples (copos) → pose genérica, não 6D + abertura específica

---

## 5. Estratégia de Uso Combinado (Recomendada)

| Etapa | Dataset(s) | Objetivo |
|-------|------------|----------|
| **1. Pré-treino ME (encoder espacial-temporal)** | Thapa2025 (23 subj, 6808 trials) + GuttmannFlury2025_ME (31 subj, 2520 trials) | Aprender representações EEG de execução motora de preensão/alcance com alta variabilidade inter-sujeto |
| **2. Pré-treino ME+MI pareado (alinhamento latente)** | MOVING2024 (11 subj, MI+ME pareados, 32ch/500Hz) | Alinhar espaços latentes ME↔MI no mesmo hardware/protocolo do alvo |
| **3. Regressão de pose contínua (fine-tuning)** | IACKD (15 subj, trajetória 3D contínua) + MOVING2024 (cinemática Leap Motion bruta) | Treinar cabeça de regressão para pose 3D/6D/abertura; pseudo-labels hold final |
| **4. Classificação grasp MI (auxiliar)** | BCIComp2020UpperLimb (15 subj, 3 grasps MI) + GuttmannFlury2025_MI (31 subj) | Head de classificação secundária para tipo de grasp/objeto |

---

## 6. Próximos Passos Imediatos

1. **Baixar MOVING2024 via MOABB** — teste rápido de loading, inspecionar estrutura Leap Motion, extrair cinemática hold.
2. **Clonar IACKD via NEMAR CLI** — `nemar dataset download on006840 --subjects sub-01,sub-02` para amostra; ler `participants.tsv` e `dataset_description.json`.
3. **Baixar Thapa2025 via MOABB** — verificar sessões, runs, acelerômetro para epoching.
4. **Criar pipeline de harmonização**: re-referenciamento (common average), band-pass 0.5-100 Hz, notch 50 Hz, resample 500 Hz, interpolação espacial 10-20 → layout g.Nautilus.
5. **Definir janela "hold"**: nos trials ME, últimos 500-1000ms do período de ação; nos MI, período equivalente pós-cue.

---

## 7. Referências Principais

1. **MOVING2024**: Mattei et al., *MOVING: A Multi-Modal Dataset of EEG Signals and Virtual Glove Hand Tracking*, Sensors 2024, DOI: 10.3390/s24165207
2. **IACKD**: Cai et al., *Intention-Action Conflict EEG-Hand Kinematics Dataset*, Scientific Data 2026, DOI: 10.1038/s41597-026-07146-x (OpenNeuro ds006840)
3. **Thapa2025**: Thapa et al., *A large electroencephalogram database of freewill reaching and grasping tasks for brain machine interfaces*, Scientific Data 2025, DOI: 10.1038/s41597-025-06039-9
4. **BNCI2025-002**: Kobler et al., *Continuous 2D trajectory decoding from attempted movement*, J. Neural Eng. 2022, DOI: 10.1088/1741-2552/ac689f
5. **GuttmannFlury2025**: Guttmann-Flury et al., *Dataset combining EEG, eye-tracking, and high-speed video for ocular activity analysis across BCI paradigms*, Scientific Data 2025, DOI: 10.1038/s41597-025-04861-9
6. **BCIComp2020UpperLimb**: Jeong et al., *2020 International brain-computer interface competition: A review*, Front. Hum. Neurosci. 2022, DOI: 10.3389/fnhum.2022.898300
7. **Reach&Grasp**: Di Domenico et al., *Reach&Grasp: a multimodal dataset of the whole upper-limb during simple and complex movements*, Scientific Data 2025, DOI: 10.1038/s41597-025-04552-5

---

*Relatório gerado automaticamente com base em buscas em bases públicas (MOABB, NEMAR, OpenNeuro, Zenodo, Figshare, OSF, Harvard Dataverse, IIT Dataverse) e documentação dos respectivos repositórios. Verificar detalhes nos artigos originais antes do uso.*