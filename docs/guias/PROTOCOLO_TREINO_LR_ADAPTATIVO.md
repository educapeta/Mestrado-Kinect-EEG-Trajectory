# Protocolo — Treino L/R adaptativo com feedback e sessão final congelada

**Versão:** 18/11/2026 (task-6) — plano **fixado antes da coleta**
**Objetivo:** tratar BCI illiteracy com **treino**, não só trocando classificador.
Desenho gradual **ME → MI no mesmo trial** (inspirado no E2T), com feedback visual
contínuo do decoder, adaptação por bloco e uma sessão final **congelada** para medir
o efeito do treino sem vazamento.
**Ferramenta:** `src/mestradopy/tools/feedback_adaptativo.py`
**Aquisição:** `src/mestradopy/tools/validacao_eeg_3classes.py` (CSV+JSON do piloto)
**EEG:** g.Nautilus, 3 canais úteis **C3, Cz, C4**, 500 Hz.

---

## 1. Por que este desenho

- BCI illiteracy / baixa responsividade é um problema de **aprendizado do usuário**;
  trocar o classificador não resolve se a estratégia motora não gera ERD/ERS
  distinguível.
- Treino gradual ME→MI: no mesmo trial o participante **executa** o movimento e
  depois **imagina** o mesmo movimento, vendo o feedback do decoder imediatamente.
  Isso dá contexto sensório-motor e permite ajustar a estratégia sessão a sessão.
- O feedback é **adaptativo** (decoder se atualiza entre blocos com os labels do cue
  e a dificuldade acompanha o desempenho), mas a **sessão final é congelada**: o
  decoder não muda, então a métrica final é honesta e comparável entre participantes.
- Regras e limiares abaixo foram fixados **antes** de olhar dados. Mudanças exigem
  nova versão do protocolo (e devem ser relatadas na tese).

---

## 2. Plano de sessões e blocos (obrigatório)

| item | valor |
|---|---|
| sessões por participante | **3** |
| trials/classe por sessão | **20** (total de 40 trials/sessão) |
| blocos por sessão | **2** (pausa de 2 min entre eles) |
| trials/classe por bloco | **10** |
| total | **60 trials/classe** (120 trials) em 3 sessões |
| sessões de treino adaptativo | sessões 1 e 2 |
| sessão final congelada | sessão **3** |

Regras:
- **Nunca** fazer 60 trials/classe em sequência: 20/classe por sessão, 10/classe por
  bloco, com pausa; o participante pode sair da tarefa nos intervalos.
- Sessões em **dias diferentes** (idealmente 2–4 dias entre elas).
- As 3 sessões somam 60/classe: a sessão 3 é o **teste congelado**, sem adaptação.
- Se o participante faltar uma sessão, o protocolo determina **reiniciar** a contagem
  (não completar 60/classe em 2 sessões).

Aquisição de cada sessão (não sobrescreve nenhum resultado existente; cria pasta nova):

```powershell
Set-Location C:\mestradopy
$env:PYTHONPATH = "$PWD\src\mestradopy;$PWD\src\mestradopy\core;$PWD\src\mestradopy\tools"
.\.venv\Scripts\python.exe src\mestradopy\tools\validacao_eeg_3classes.py `
  --source gnautilus --classes esquerda,direita `
  --blocos 2 --trials-por-classe 10 `
  --cue-sec 1 --me-sec 4 --transicao-sec 1 --mi-sec 4 `
  --repouso-sec 2 --iti-min 1 --iti-max 2 --baseline-olhos-sec 60 `
  --sujeito S01 --sessao 01 --pasta-saida outputs\eeg\treino_lr_s01
```

(`--trials-por-classe 10` × 2 blocos = 20/classe/sessão.)

---

## 3. Trial: ME → MI sem vazamento de fase

```text
repouso 2 s | cue 1 s (L/R) | ME 4 s | transição 1 s | MI 4 s | ITI 1–2 s
                                                      ↑
                        o decoder só age AQUI (fase MI, marcadores 780/781)
```

- Marcadores (ver `eeg_motor_paradigm.py`/`validacao_eeg_3classes.py`):
  768 trial start, 769/770 cue L/R, 778/779 ME, 780/781 MI, 790/791 bloco.
- **Regra de ouro implementada no tool:** a janela de decisão (2 s, passo 0,25 s) é
  recortada **dentro do segmento MI** (teste unitário
  `tests/manual/test_feedback_adaptativo.py::TestJanelasNaoCruzamFases` garante que
  nenhuma janela atravessa a transição ME→MI). Nada de ME no treino do decoder de MI.
- O baseline de ERD é 1 s imediatamente antes do cue.

---

## 4. Decoder, score contínuo e feedback

- **Classificador:** CSP + LDA (mesmo pipeline de `csp_lda_duas_sessoes.py`:
  `mne.decoding.CSP(n_components=2, reg="ledoit_wolf", log=True)` +
  `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`), treinado nas
  janelas de 2 s da MI (FS 128 Hz, banda 8–30 Hz, janela deslizando 0,25 s).
- **Score contínuo:** `score = 2·P(direita) − 1 ∈ [−1, +1]`, atualizado a cada
  janela durante os 4 s de MI. Score por trial = média das janelas (soft vote).
- **Feedback visual** (OpenCV; opcional em bancada, obrigatório para o participante):
  - barra horizontal L↔R com **zona morta de tolerância**;
  - **cursor** e **mão virtual** que andam para o lado que o decoder "puxa";
  - `ACERTO`/`ERRO`/`AGUARDE` + score e tolerância na tela;
  - o participante deve **empurrar** a mão para o lado do cue (L=esquerda, R=direita).
  Frames de exemplo podem ser salvos com `--render <pasta>` (PNG) para auditoria.
- **Calibração inicial:** bloco 1 da sessão 1 é usado para treinar o CSP+LDA
  (papel `calibracao`); a partir daí cada bloco tem papel `adaptacao`.

---

## 5. Adaptação (por bloco, com labels do cue) e dificuldade

Após cada bloco de treino (labels do cue já conhecidos — é treino supervisionado
offline, não online não-supervisionado):

1. **Refit** do CSP+LDA com **todos** os trials rotulados até aquele bloco
   (memória cumulativa: sessão 1 + sessão 2).
2. **Recalibração do limiar** de decisão para equilibrar L/R:
   `limiar = média(mediana(score|L), mediana(score|R))` — corrige o viés do LDA.
3. **Dificuldade adaptativa** (muda a cada bloco em função da BA do bloco):

| BA do bloco | tolerância (|score| p/ decidir) | ganho do cursor | ação |
|---|---|---:|---|
| ≥ 0,80 | ×0,85 (mín. 0,12) | ×1,10 (máx. 2,5) | mais difícil |
| ≤ 0,55 | ×1,15 (máx. 0,80) | ×0,90 (mín. 0,40) | mais fácil |
| 0,55–0,80 | mantém | mantém | mantido |

O participante só recebe "ACERTO/ERRO" quando `|score| ≥ tolerância`; abaixo disso o
feedback é "AGUARDE" (isso reduz frustração e evita forçar decisão fraca).

---

## 6. Sessão final congelada (sessão 3)

- O modelo salvo ao fim da sessão 2 é carregado com `--modo teste-congelado` e
  **não é mais atualizado** (sem refit, sem mudança de limiar, sem dificuldade
  adaptativa). O feedback continua sendo mostrado ao participante, mas a decisão é
  do decoder congelado.
- Métricas da sessão 3 são as métricas de **resultado** do treino.
- Sessões 1–2 nunca entram como teste; o teste é sempre em sessão/dia separado.

---

## 7. Métricas e critérios (pré-fixados)

### 7.1 Sucesso / não-respondedor

| critério | limiar | significado |
|---|---|---|
| **respondedor** | BA ≥ **0,55** com IC95% inferior ≥ 0,55 na sessão congelada | respondeu ao treino |
| **sucesso** | BA ≥ **0,65** na sessão congelada | controle confiável L/R |
| **não-respondedor** | IC95% inferior da BA < 0,55 | BCI illiteracy: pausar treino, revisar estratégia/ERD, considerar mudar o desenho |

BA = acurácia balanceada (média das recalls L e R); IC = **intervalo de confiança
bootstrap 95%** (2.000 reamostragens; `estat_bci.bootstrap`/`bootstrap_ci_ba`).
Sempre reportar n, confusão L/R e IC — nunca só a acurácia.

### 7.2 Métricas obrigatórias por sessão

- **BA + IC95%** por bloco (treino) e na sessão congelada;
- **ERD/ERS** mu (8–13 Hz) e beta (13–30 Hz) da MI relativo ao baseline, por canal
  (C3/Cz/C4) e por classe — negativo = dessincronização (esperado contralateral);
- **score médio |score|** e nº de decisões (calibração do feedback);
- **tolerância/ganho** por bloco (curva de dificuldade);
- **questionário de estratégia** (abaixo);
- (opcional) ITR de Wolpaw se quiser comparar com a literatura — incluir ITI no T.

### 7.3 Questionário de estratégia (preencher ao fim de cada sessão)

`--questionario arquivo.json` cria o template. Itens (escala 0–4 quando indicado):
1. O que imaginou durante a MI? (movimento/sensação/outro)
2. Sentiu movimento/contração muscular? (0–4)
3. Esforço percebido na MI (0–4)
4. Fadiga (0–4)
5. Confiança no lado imaginado (0–4)
6. Estratégia visual em 1ª ou 3ª pessoa
7. Fatores externos (ruído, luz, desconforto)
8. Observações do experimentador

Mudanças de estratégia entre sessões devem ser registradas (é parte do resultado).

---

## 8. Comandos (tool)

```powershell
Set-Location C:\mestradopy
$env:PYTHONPATH = "$PWD\src\mestradopy;$PWD\src\mestradopy\core;$PWD\src\mestradopy\tools"
$env:PYTHONIOENCODING = "utf-8"

# 0) plano do protocolo (valida 60/classe em 3 sessoes e mostra o cronograma)
.\.venv\Scripts\python.exe src\mestradopy\tools\feedback_adaptativo.py --plano

# 1) smoke sintetico (sem hardware; gera relatorio, questionario e frames PNG)
.\.venv\Scripts\python.exe src\mestradopy\tools\feedback_adaptativo.py --smoke ^
  --saida outputs\visao\feedback_adaptativo_smoke ^
  --render outputs\visao\feedback_adaptativo_smoke\render ^
  --questionario outputs\visao\feedback_adaptativo_smoke\questionario_estrategia.json

# 2) SESSAO 1 — treino: calibra no bloco 1 e adapta no bloco 2
.\.venv\Scripts\python.exe src\mestradopy\tools\feedback_adaptativo.py ^
  --sessao-dir outputs\eeg\treino_lr_s01 --modo treino ^
  --salvar-modelo outputs\visao\treino_lr\modelo_s01.joblib ^
  --questionario outputs\visao\treino_lr\questionario_s01.json ^
  --saida outputs\visao\treino_lr\s01

# 3) SESSAO 2 — treino continua do modelo da sessao 1 (adaptacao cumulativa)
.\.venv\Scripts\python.exe src\mestradopy\tools\feedback_adaptativo.py ^
  --sessao-dir outputs\eeg\treino_lr_s02 --modo treino ^
  --modelo outputs\visao\treino_lr\modelo_s01.joblib ^
  --salvar-modelo outputs\visao\treino_lr\modelo_s02.joblib ^
  --saida outputs\visao\treino_lr\s02

# 4) SESSAO 3 — TESTE CONGELADO (sem adaptacao)
.\.venv\Scripts\python.exe src\mestradopy\tools\feedback_adaptativo.py ^
  --sessao-dir outputs\eeg\treino_lr_s03 --modo teste-congelado ^
  --modelo outputs\visao\treino_lr\modelo_s02.joblib ^
  --saida outputs\visao\treino_lr\s03_congelado ^
  --questionario outputs\visao\treino_lr\questionario_s03.json

# 5) teste unitario (sem hardware)
.\.venv\Scripts\python.exe tests\manual\test_feedback_adaptativo.py
```

Observação: se a sessão for mais curta (piloto), o tool funciona igual; o protocolo
de 3×20/classe é o desenho oficial do treino.

---

## 9. Arquivos gerados e o que NÃO tocar

- `feedback_adaptativo.py` (novo) — aditivo; **não altera** `eeg_motor_paradigm.py`.
- Saídas do tool: `relatorio.md`, `resultados.json`, `modelo_*.joblib`,
  `questionario_*.json`, frames PNG — em `--saida` (default: pasta temporária).
- **Não editar/sobrescrever:** `outputs/eeg/` existentes (dados de aquisição são
  crus e não se reescrevem), `docs/thesis/`, `external/HandMvNet/`,
  `configs/calibracao/*.npz`, `Comite Etica/`.
- Nada de commit/push sem revisão do Lead.

---

## 10. Riscos e limitações (declarar na tese)

1. **Adaptação supervisionada por bloco** usa os labels do cue (legítimo em treino),
   mas não é "online não-supervisionado": os números de treino não são teste.
2. **Poucos trials** (20/classe/sessão): IC95% da BA é largo; não superinterpretar
   BA de 0,6 com n=40 sem o IC.
3. **Seleção por desempenho**: se o participante desistir por frustração, isso é
   resultado (não-respondedor) e deve ser registrado; não repetir até "passar".
4. **Vazamento se sessões forem do mesmo dia**: manter sessões em dias diferentes.
5. **CSP+LDA é baseline**: se o objetivo for comparar com EEGNet, treinar ambos no
   mesmo split/frozen session (mesmo protocolo).
6. **Feedback pode mudar a estratégia** (aprendizado de artefatos): controlar com
   questionário de estratégia e, idealmente, uma sessão sem feedback para checar
   generalização.
7. **Sincronização**: o trial ME→MI usa marcadores do paradigma; janelas nunca
   atravessam a fase (garantido em teste unitário) — validar com o JSON de eventos.

---

*Referências internas: `docs/guias/HISTORICO_ITERACOES.txt` (itens do piloto novo
ME/MI), `docs/PIPELINE_DADOS_JANELAS.md`, `docs/ESTADO_DA_ARTE_ESTATISTICA.md`
(BA/IC/ITR), `src/mestradopy/tools/csp_lda_duas_sessoes.py` (pipeline CSP+LDA),
`src/mestradopy/tools/validacao_eeg_3classes.py` (aquisição).*
