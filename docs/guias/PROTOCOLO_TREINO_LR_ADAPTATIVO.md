# Protocolo â€” Treino L/R adaptativo com feedback e sessÃ£o final congelada

**VersÃ£o:** 18/11/2026 (task-6) â€” plano **fixado antes da coleta**
**Objetivo:** tratar BCI illiteracy com **treino**, nÃ£o sÃ³ trocando classificador.
Desenho gradual **ME â†’ MI no mesmo trial** (inspirado no E2T), com feedback visual
contÃ­nuo do decoder, adaptaÃ§Ã£o por bloco e uma sessÃ£o final **congelada** para medir
o efeito do treino sem vazamento.
**Ferramenta:** `src/mestradopy/tools/feedback_adaptativo.py`
**AquisiÃ§Ã£o:** `src/mestradopy/tools/validacao_eeg_3classes.py` (CSV+JSON do piloto)
**EEG:** g.Nautilus, 3 canais Ãºteis **C3, Cz, C4**, 500 Hz.

---

## 1. Por que este desenho

- BCI illiteracy / baixa responsividade Ã© um problema de **aprendizado do usuÃ¡rio**;
  trocar o classificador nÃ£o resolve se a estratÃ©gia motora nÃ£o gera ERD/ERS
  distinguÃ­vel.
- Treino gradual MEâ†’MI: no mesmo trial o participante **executa** o movimento e
  depois **imagina** o mesmo movimento, vendo o feedback do decoder imediatamente.
  Isso dÃ¡ contexto sensÃ³rio-motor e permite ajustar a estratÃ©gia sessÃ£o a sessÃ£o.
- O feedback Ã© **adaptativo** (decoder se atualiza entre blocos com os labels do cue
  e a dificuldade acompanha o desempenho), mas a **sessÃ£o final Ã© congelada**: o
  decoder nÃ£o muda, entÃ£o a mÃ©trica final Ã© honesta e comparÃ¡vel entre participantes.
- Regras e limiares abaixo foram fixados **antes** de olhar dados. MudanÃ§as exigem
  nova versÃ£o do protocolo (e devem ser relatadas na tese).

---

## 2. Plano de sessÃµes e blocos (obrigatÃ³rio)

| item | valor |
|---|---|
| sessÃµes por participante | **3** |
| trials/classe por sessÃ£o | **20** (total de 40 trials/sessÃ£o) |
| blocos por sessÃ£o | **2** (pausa entre blocos indefinida até pressionar ESPAÇO) |
| trials/classe por bloco | **10** |
| total | **60 trials/classe** (120 trials) em 3 sessÃµes |
| sessÃµes de treino adaptativo | sessÃµes 1 e 2 |
| sessÃ£o final congelada | sessÃ£o **3** |

Regras:
- **Nunca** fazer 60 trials/classe em sequÃªncia: 20/classe por sessÃ£o, 10/classe por
  bloco, com pausa; o participante pode sair da tarefa nos intervalos.
- SessÃµes em **dias diferentes** (idealmente 2â€“4 dias entre elas).
- As 3 sessÃµes somam 60/classe: a sessÃ£o 3 Ã© o **teste congelado**, sem adaptaÃ§Ã£o.
- Se o participante faltar uma sessÃ£o, o protocolo determina **reiniciar** a contagem
  (nÃ£o completar 60/classe em 2 sessÃµes).

AquisiÃ§Ã£o de cada sessÃ£o (nÃ£o sobrescreve nenhum resultado existente; cria pasta nova):

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

(`--trials-por-classe 10` Ã— 2 blocos = 20/classe/sessÃ£o.)

---

## 3. Trial: ME â†’ MI sem vazamento de fase

```text
repouso 2 s | cue 1 s (L/R) | ME 4 s | transiÃ§Ã£o 1 s | MI 4 s | ITI 1â€“2 s
                                                      â†‘
                        o decoder sÃ³ age AQUI (fase MI, marcadores 780/781)
```

- Marcadores (ver `eeg_motor_paradigm.py`/`validacao_eeg_3classes.py`):
  768 trial start, 769/770 cue L/R, 778/779 ME, 780/781 MI, 790/791 bloco.
- **Regra de ouro implementada no tool:** a janela de decisÃ£o (2 s, passo 0,25 s) Ã©
  recortada **dentro do segmento MI** (teste unitÃ¡rio
  `tests/manual/test_feedback_adaptativo.py::TestJanelasNaoCruzamFases` garante que
  nenhuma janela atravessa a transiÃ§Ã£o MEâ†’MI). Nada de ME no treino do decoder de MI.
- O baseline de ERD Ã© 1 s imediatamente antes do cue.

---

## 4. Decoder, score contÃ­nuo e feedback

- **Classificador:** CSP + LDA (mesmo pipeline de `csp_lda_duas_sessoes.py`:
  `mne.decoding.CSP(n_components=2, reg="ledoit_wolf", log=True)` +
  `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`), treinado nas
  janelas de 2 s da MI (FS 128 Hz, banda 8â€“30 Hz, janela deslizando 0,25 s).
- **Score contÃ­nuo:** `score = 2Â·P(direita) âˆ’ 1 âˆˆ [âˆ’1, +1]`, atualizado a cada
  janela durante os 4 s de MI. Score por trial = mÃ©dia das janelas (soft vote).
- **Feedback visual** (OpenCV; opcional em bancada, obrigatÃ³rio para o participante):
  - barra horizontal Lâ†”R com **zona morta de tolerÃ¢ncia**;
  - **cursor** e **mÃ£o virtual** que andam para o lado que o decoder "puxa";
  - `ACERTO`/`ERRO`/`AGUARDE` + score e tolerÃ¢ncia na tela;
  - o participante deve **empurrar** a mÃ£o para o lado do cue (L=esquerda, R=direita).
  Frames de exemplo podem ser salvos com `--render <pasta>` (PNG) para auditoria.
- **CalibraÃ§Ã£o inicial:** bloco 1 da sessÃ£o 1 Ã© usado para treinar o CSP+LDA
  (papel `calibracao`); a partir daÃ­ cada bloco tem papel `adaptacao`.

---

## 5. AdaptaÃ§Ã£o (por bloco, com labels do cue) e dificuldade

ApÃ³s cada bloco de treino (labels do cue jÃ¡ conhecidos â€” Ã© treino supervisionado
offline, nÃ£o online nÃ£o-supervisionado):

1. **Refit** do CSP+LDA com **todos** os trials rotulados atÃ© aquele bloco
   (memÃ³ria cumulativa: sessÃ£o 1 + sessÃ£o 2).
2. **RecalibraÃ§Ã£o do limiar** de decisÃ£o para equilibrar L/R:
   `limiar = mÃ©dia(mediana(score|L), mediana(score|R))` â€” corrige o viÃ©s do LDA.
3. **Dificuldade adaptativa** (muda a cada bloco em funÃ§Ã£o da BA do bloco):

| BA do bloco | tolerÃ¢ncia (|score| p/ decidir) | ganho do cursor | aÃ§Ã£o |
|---|---|---:|---|
| â‰¥ 0,80 | Ã—0,85 (mÃ­n. 0,12) | Ã—1,10 (mÃ¡x. 2,5) | mais difÃ­cil |
| â‰¤ 0,55 | Ã—1,15 (mÃ¡x. 0,80) | Ã—0,90 (mÃ­n. 0,40) | mais fÃ¡cil |
| 0,55â€“0,80 | mantÃ©m | mantÃ©m | mantido |

O participante sÃ³ recebe "ACERTO/ERRO" quando `|score| â‰¥ tolerÃ¢ncia`; abaixo disso o
feedback Ã© "AGUARDE" (isso reduz frustraÃ§Ã£o e evita forÃ§ar decisÃ£o fraca).

---

## 6. SessÃ£o final congelada (sessÃ£o 3)

- O modelo salvo ao fim da sessÃ£o 2 Ã© carregado com `--modo teste-congelado` e
  **nÃ£o Ã© mais atualizado** (sem refit, sem mudanÃ§a de limiar, sem dificuldade
  adaptativa). O feedback continua sendo mostrado ao participante, mas a decisÃ£o Ã©
  do decoder congelado.
- MÃ©tricas da sessÃ£o 3 sÃ£o as mÃ©tricas de **resultado** do treino.
- SessÃµes 1â€“2 nunca entram como teste; o teste Ã© sempre em sessÃ£o/dia separado.

---

## 7. MÃ©tricas e critÃ©rios (prÃ©-fixados)

### 7.1 Sucesso / nÃ£o-respondedor

| critÃ©rio | limiar | significado |
|---|---|---|
| **respondedor** | BA â‰¥ **0,55** com IC95% inferior â‰¥ 0,55 na sessÃ£o congelada | respondeu ao treino |
| **sucesso** | BA â‰¥ **0,65** na sessÃ£o congelada | controle confiÃ¡vel L/R |
| **nÃ£o-respondedor** | IC95% inferior da BA < 0,55 | BCI illiteracy: pausar treino, revisar estratÃ©gia/ERD, considerar mudar o desenho |

BA = acurÃ¡cia balanceada (mÃ©dia das recalls L e R); IC = **intervalo de confianÃ§a
bootstrap 95%** (2.000 reamostragens; `estat_bci.bootstrap`/`bootstrap_ci_ba`).
Sempre reportar n, confusÃ£o L/R e IC â€” nunca sÃ³ a acurÃ¡cia.

### 7.2 MÃ©tricas obrigatÃ³rias por sessÃ£o

- **BA + IC95%** por bloco (treino) e na sessÃ£o congelada;
- **ERD/ERS** mu (8â€“13 Hz) e beta (13â€“30 Hz) da MI relativo ao baseline, por canal
  (C3/Cz/C4) e por classe â€” negativo = dessincronizaÃ§Ã£o (esperado contralateral);
- **score mÃ©dio |score|** e nÂº de decisÃµes (calibraÃ§Ã£o do feedback);
- **tolerÃ¢ncia/ganho** por bloco (curva de dificuldade);
- **questionÃ¡rio de estratÃ©gia** (abaixo);
- (opcional) ITR de Wolpaw se quiser comparar com a literatura â€” incluir ITI no T.

### 7.3 QuestionÃ¡rio de estratÃ©gia (preencher ao fim de cada sessÃ£o)

`--questionario arquivo.json` cria o template. Itens (escala 0â€“4 quando indicado):
1. O que imaginou durante a MI? (movimento/sensaÃ§Ã£o/outro)
2. Sentiu movimento/contraÃ§Ã£o muscular? (0â€“4)
3. EsforÃ§o percebido na MI (0â€“4)
4. Fadiga (0â€“4)
5. ConfianÃ§a no lado imaginado (0â€“4)
6. EstratÃ©gia visual em 1Âª ou 3Âª pessoa
7. Fatores externos (ruÃ­do, luz, desconforto)
8. ObservaÃ§Ãµes do experimentador

MudanÃ§as de estratÃ©gia entre sessÃµes devem ser registradas (Ã© parte do resultado).

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

# 2) SESSAO 1 â€” treino: calibra no bloco 1 e adapta no bloco 2
.\.venv\Scripts\python.exe src\mestradopy\tools\feedback_adaptativo.py ^
  --sessao-dir outputs\eeg\treino_lr_s01 --modo treino ^
  --salvar-modelo outputs\visao\treino_lr\modelo_s01.joblib ^
  --questionario outputs\visao\treino_lr\questionario_s01.json ^
  --saida outputs\visao\treino_lr\s01

# 3) SESSAO 2 â€” treino continua do modelo da sessao 1 (adaptacao cumulativa)
.\.venv\Scripts\python.exe src\mestradopy\tools\feedback_adaptativo.py ^
  --sessao-dir outputs\eeg\treino_lr_s02 --modo treino ^
  --modelo outputs\visao\treino_lr\modelo_s01.joblib ^
  --salvar-modelo outputs\visao\treino_lr\modelo_s02.joblib ^
  --saida outputs\visao\treino_lr\s02

# 4) SESSAO 3 â€” TESTE CONGELADO (sem adaptacao)
.\.venv\Scripts\python.exe src\mestradopy\tools\feedback_adaptativo.py ^
  --sessao-dir outputs\eeg\treino_lr_s03 --modo teste-congelado ^
  --modelo outputs\visao\treino_lr\modelo_s02.joblib ^
  --saida outputs\visao\treino_lr\s03_congelado ^
  --questionario outputs\visao\treino_lr\questionario_s03.json

# 5) teste unitario (sem hardware)
.\.venv\Scripts\python.exe tests\manual\test_feedback_adaptativo.py
```

ObservaÃ§Ã£o: se a sessÃ£o for mais curta (piloto), o tool funciona igual; o protocolo
de 3Ã—20/classe Ã© o desenho oficial do treino.

---

## 9. Arquivos gerados e o que NÃƒO tocar

- `feedback_adaptativo.py` (novo) â€” aditivo; **nÃ£o altera** `eeg_motor_paradigm.py`.
- SaÃ­das do tool: `relatorio.md`, `resultados.json`, `modelo_*.joblib`,
  `questionario_*.json`, frames PNG â€” em `--saida` (default: pasta temporÃ¡ria).
- **NÃ£o editar/sobrescrever:** `outputs/eeg/` existentes (dados de aquisiÃ§Ã£o sÃ£o
  crus e nÃ£o se reescrevem), `docs/thesis/`, `external/HandMvNet/`,
  `configs/calibracao/*.npz`, `Comite Etica/`.
- Nada de commit/push sem revisÃ£o do Lead.

---

## 10. Riscos e limitaÃ§Ãµes (declarar na tese)

1. **AdaptaÃ§Ã£o supervisionada por bloco** usa os labels do cue (legÃ­timo em treino),
   mas nÃ£o Ã© "online nÃ£o-supervisionado": os nÃºmeros de treino nÃ£o sÃ£o teste.
2. **Poucos trials** (20/classe/sessÃ£o): IC95% da BA Ã© largo; nÃ£o superinterpretar
   BA de 0,6 com n=40 sem o IC.
3. **SeleÃ§Ã£o por desempenho**: se o participante desistir por frustraÃ§Ã£o, isso Ã©
   resultado (nÃ£o-respondedor) e deve ser registrado; nÃ£o repetir atÃ© "passar".
4. **Vazamento se sessÃµes forem do mesmo dia**: manter sessÃµes em dias diferentes.
5. **CSP+LDA Ã© baseline**: se o objetivo for comparar com EEGNet, treinar ambos no
   mesmo split/frozen session (mesmo protocolo).
6. **Feedback pode mudar a estratÃ©gia** (aprendizado de artefatos): controlar com
   questionÃ¡rio de estratÃ©gia e, idealmente, uma sessÃ£o sem feedback para checar
   generalizaÃ§Ã£o.
7. **SincronizaÃ§Ã£o**: o trial MEâ†’MI usa marcadores do paradigma; janelas nunca
   atravessam a fase (garantido em teste unitÃ¡rio) â€” validar com o JSON de eventos.

---

*ReferÃªncias internas: `docs/guias/HISTORICO_ITERACOES.txt` (itens do piloto novo
ME/MI), `docs/PIPELINE_DADOS_JANELAS.md`, `docs/ESTADO_DA_ARTE_ESTATISTICA.md`
(BA/IC/ITR), `src/mestradopy/tools/csp_lda_duas_sessoes.py` (pipeline CSP+LDA),
`src/mestradopy/tools/validacao_eeg_3classes.py` (aquisiÃ§Ã£o).*
---

## Runbook operacional

Antes de cada sessão, rodar:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verificar_preflight_lr.ps1
```

O passo a passo completo (sessões 01/02/03, feedback, critérios e checklist)
está em `docs/guias/RUNBOOK_AQUISICAO_LR.md`.

