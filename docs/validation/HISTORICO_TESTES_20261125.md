# Histórico consolidado de testes e dados — EEG/BCI

**Atualizado em:** 25/11/2026
**Objetivo:** registrar, de forma honesta e rastreável, o que foi adquirido,
testado e concluído no piloto EEG/BCI.

---

## 1. Dados disponíveis

### Piloto principal (120 trials)

| sessão | trials | canais | fs | janelas MI | arquivo |
|---|---:|---|---:|---:|---|
| s01 | 60 (30 L/30 R) | C3/C4/Cz | 500 Hz | 300 | `outputs/eeg/piloto_me_mi_s01/` |
| s02 | 60 (30 L/30 R) | C3/C4/Cz | 500 Hz | 300 | `outputs/eeg/piloto_me_mi_s02/` |
| total | 120 | — | — | 600 | — |

- Protocolo: cue 1 s → ME 4 s → transição 1 s → MI 4 s → ITI.
- Janelas de análise: 2 s, passo 0,5 s, 4–40 Hz, reamostrado para 128 Hz.
- Folds canônicos: `docs/validation/folds_piloto_groupkf.json` (5-fold,
  agrupado por trial, 96 treino / 24 teste).
- Dataset processado portátil: `docs/validation/piloto_me_mi_2s.npz`.

### Sessões com feedback online

| sessão | trials | modelo | BA online | BA offline |
|---|---:|---|---:|---:|
| s01 | 40 | CSP+LDA calibrado no bloco 1 | 0,600 / 0,500 | 0,650 / 0,550 |
| s02 | 40 | modelo adaptado | — | 0,700 / 0,650 |
| s03 | não realizada | congelado | — | — |

### Dados auxiliares/de ruído

- Desktop: `outputs/eeg/ruido_bruto/` (5 condições A–E).
- Laptop: branch `resultados/ruido-eeg-20261123` e
  `outputs/eeg/ruido_laptop/` (fora do git).
- Resultados de ruído: `docs/validation/RUIDO_EEG_CRU_20261123.md` e
  `docs/validation/REVISAO_BRANCH_RUIDO_20261123.md`.

---

## 2. Linha do tempo dos testes

| data | teste | resultado principal |
|---|---|---|
| 17/11 | aquisição piloto s01/s02 | 120 trials, 600 janelas |
| 19/11 | ERD/ERS + permutação | sinal fraco; janela late não significativa |
| 19/11 | transferência BCIC 2b → piloto | CSP+LDA não transferiu; target ~0,475 |
| 23/11 | feedback online s01 | 40/40; BA .60/.50 |
| 23/11 | feedback online s02 | 40/40; BA .70/.65 |
| 23/11 | ruído desktop | 60 Hz ~84–99 µV; E ~1,5 mV |
| 23/11 | ruído laptop | bateria reduz 60 Hz ~18×; ainda ~97 µV |
| 24/11 | FBCSP 2b | BA 0,6785 (<0,70) |
| 24/11 | EEGCCT 2b | BA 0,6412 (<0,70) |
| 25/11 | FBCSP piloto | 0,892–0,908 por trial |
| 25/11 | controles FBCSP | baseline 0,462; ME 0,783; MI 0,892 |
| 25/11 | EEGCCT piloto real | target 0,692; source 0,541; fine-tune 0,566 |
| 25/11 | CIACNet | 2b 0,6985; piloto com bug de avaliação |

---

## 3. Resultados detalhados

### 3.1 FBCSP

**BCIC 2b — within-subject cross-session:**

- BA média: **0,6785 ± 0,1079**;
- IC95%: [0,6144; 0,7464];
- alvo ≥0,70: **não atingido**;
- por sujeito: B01 .694, B02 .604, B03 .500, B04 .892, B05 .748,
  B06 .637, B07 .650, B08 .657, B09 .726.

**Piloto — folds canônicos, BA por trial:**

| configuração | BA |
|---|---:|
| FBCSP full (9 bandas, 4 CSP) | **0,908** |
| versão reduzida (4 bandas, 2 CSP) | 0,867 |
| versão reduzida (6 bandas, 2 CSP) | 0,883 |
| primeira janela/trial | 0,908 |
| última janela/trial | 0,842 |
| canais embaralhados | 0,567 |
| features simples | 0,683 |
| CSP+LDA | 0,583 |

Permutação de labels por trial: p ≈ 0,01 (100 permutações).
Por sessão: s01 0,933; s02 0,850.

**Controle por fase (FBCSP):**

| fase | BA trial |
|---|---:|
| baseline pré-cue 2 s | **0,462** |
| ME | **0,783** |
| MI | **0,892** |

**Bandas mais informativas (MI):**

- 12–16 Hz: 0,767;
- 16–20 Hz: 0,800;
- 20–24 Hz: 0,808;
- 8–12 Hz: 0,508 (chance).

**Leitura:** o classificador não usa informação pré-cue; o efeito aparece na
ME e cresce na MI. É robusto a reduzir bandas/componentes e a usar uma janela
por trial. EMG ainda precisa ser registrado para descartar contaminação
muscular.

### 3.2 EEGCCT

**BCIC 2b (100 epochs):**

- BA: **0,6412 ± 0,1210**;
- IC95%: [0,5955; 0,6868];
- alvo ≥0,70: **não atingido**.

**Piloto real (120 trials, folds canônicos):**

| cenário | parâmetros | BA |
|---|---:|---:|
| target-only | 188.419 | **0,6921 ± 0,1122** |
| source-only | 0 | 0,5408 ± 0,0540 |
| fine-tune do classificador | 130 | 0,5658 ± 0,0703 |

- target-only: fold 1 colapsou (0,501), outros 0,68–0,78;
- source-only ≈ acaso;
- fine-tune melhora só ~2,5 pp.

**Leitura:** resultado real e plausível; a arquitetura não supera o FBCSP e a
transferência 2b→piloto falha.

### 3.3 CIACNet

**BCIC 2b LOSO:**

- BA: **0,6985 ± 0,0762**;
- paper: 0,9005;
- não reproduzido (-20,2 pp).

**Piloto:**

- resultado anterior de 100% era inválido (dados sintéticos/40 trials);
- a versão com 120 trials tinha **bug de avaliação**: usava o fold de treino
  como teste;
- correção indicada em `finetune.py`: trocar
  `test_trials, _ = get_piloto_folds_indices(...)` por
  `_, test_trials = get_piloto_folds_indices(...)`;
- precisa ser reexecutado; nenhum resultado válido de CIACNet no piloto ainda.

### 3.4 Transferência

| método | 2b | piloto target | piloto source |
|---|---:|---:|---:|
| CSP+LDA (antigo) | 0,658 | 0,475 | 0,508 |
| FBCSP+LDA | 0,6785 | 0,892 (auditado) | 0,500 |
| EEGCCT | 0,6412 | 0,692 | 0,541 |
| CIACNet | 0,6985 | inválido (bug) | — |

A transferência 2b→piloto **não funcionou** em nenhum método avaliado de forma
válida. O melhor resultado no piloto veio do FBCSP treinado **no próprio
piloto**, não da transferência.

---

## 4. Ruído

- Desktop: 60 Hz dominante em todas as condições; 50 Hz baixo; modo comum.
- Laptop/bateria: 60 Hz cai ~18×, mas ainda ~97 µV na melhor condição.
- Recomendação: corrigir terra/referência, manter cap/cabos longe do desktop,
  operar na bateria e registrar EMG.

---

## 5. Limitações atuais

1. n=120 trials (60/classe), 2 sessões do mesmo dia.
2. Sem EMG registrado neste piloto.
3. Ruído de 60 Hz alto no ambiente original.
4. FBCSP do piloto ainda pode estar contaminado por EMG/postura.
5. 2b não atingiu 0,70 em nenhum dos métodos.
6. CIACNet no piloto ainda inválido.
7. `s03` congelada não foi realizada.
8. Alguns resultados foram obtidos em sessões/ambientes diferentes, o que limita
   comparações cruzadas.

---

## 6. Próximos passos

1. **Adicionar EMG** aos próximos trials (FDS/ED, 20–200 Hz, marcadores
   sincronizados).
2. Repetir FBCSP em trials EMG-limpos vs todos.
3. Corrigir e rodar CIACNet no piloto real.
4. Definir se o protocolo de 3 sessões será reiniciado em ambiente limpo.
5. Registrar na tese:
   - controle baseline/ME/MI;
   - bandas 12–24 Hz;
   - limitação de EMG;
   - falha de transferência 2b→piloto.

---

## 7. Arquivos de referência

- `docs/validation/CONTROLE_FBCSP_20261125.md`
- `docs/validation/BENCHMARK_TRANSFER_20261123.md`
- `docs/validation/folds_piloto_groupkf.json`
- `outputs/eeg/auditoria_piloto/fbcsp_controles.json`
- `outputs/eeg/auditoria_piloto/fbcsp_comparacao.json`
- `outputs/eeg/eegcct_pilot_real_v3/resultado.json`
- `outputs/eeg/ciacnet_out/resultado.json`
