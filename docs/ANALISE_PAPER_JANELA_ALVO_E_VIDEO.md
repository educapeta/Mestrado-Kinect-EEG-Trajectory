# Análise: janela do artigo SAND, alvo concorrente × preditivo, priming e pré-treino no WAY-EEG-GAL

Referência do artigo (apontada pelo autor do projeto):
Fu, R.; Fang, Y.; Xu, F.; Hua, C.; Hua, C. (2026). *SAND: Spectral-Attention Neural
Decoding of Hand Kinematics from Low-Frequency EEG Dynamics.* IEEE TBME.
DOI: 10.1109/TBME.2026.3694935.

---

## 1. O que existe no WAY-EEG-GAL local (`C:\Mestradopy\WAY-EEG-GAL`)

Confirmado por inspeção (`tools/inventario_way_eeg_gal.py`):

- **12 sujeitos** (`P1`…`P12`) × **9 séries** `WS_P{n}_S{1..9}.mat` (~40–50 MB cada)
  + `P{n}_AllLifts.mat`. Total: 120 arquivos, **5,38 GB**.
- `P{n}_AllLifts.mat` **não** contém EEG: é a **tabela de eventos por lift**
  (`P.AllLifts` 294×43 + `P.ColNames`) com tempos (`tLiftOff`, `tHandStart`,
  `tHandStop`, `tPeakVelHandReach`…), forças (`GF_Max`, `LF_Max`) e durações.
- Cada `WS_*_S*.mat` traz a struct `ws` com `ws.win` = **34 janelas (1 por lift)**.

| Campo da janela | Forma | Conteúdo |
|---|---|---|
| `eeg` | (4714, **32**) float32 | EEG em **µV** (min −595, max 3101); a 500 Hz → janela de **9,43 s** |
| `kin` | (4714, **45**) | cinemática: ângulos, forças, **posições Px1-4/Py1-4/Pz1-4**, torques, LF/GF |
| `emg` | (37712, 5) | deltoide anterior, braquiorradial, flexor/extensor dos dedos, 1º interósseo dorsal |
| `eeg_t` / `emg_t` | (1,4714) / (1,37712) | eixos de tempo |
| `LEDon` / `LEDoff` | escalares | **estímulo ("go") a +2,0 s**; LED apagado a +6,43 s |
| `weight` / `weight_id` | 1..3 / texto | 165 g / 330 g / 660 g |
| `surf` / `surf_id` | 1..3 / texto | superfície (cork / sandpaper / …) |

- **Montagem dos 32 canais** (`ws.names.eeg`): `Fp1, Fp2, F7, F3, Fz, F4, F8, FC5,
  FC1, FC2, FC6, T7, C3, Cz, C4, T8, TP9, CP5, CP1, CP2, CP6, TP10, P7, P3, Pz,
  P4, P8, PO9, O1, Oz, O2, PO10` → **contém exatamente os canais que o artigo
  destaca** (C3, C4, Cz, F3, F4, FC1, CP1, CP2). Índices 0-based: F3=3, F4=5,
  FC1=8, C3=12, Cz=13, C4=14, CP1=18, CP2=19.
- **Cinemática do alvo**: `ws.names.kin` tem 45 nomes; as posições são
  `Px1..Px4`, `Py1..Py4`, `Pz1..Pz4`. Logo **[21, 25, 29] = Px4, Py4, Pz4 =
  "sensor de posição nº 4"** — exatamente o que o código legado fazia em
  `WAYEEGDataset._trajectory_from_kin` (*"The paper uses the P4 position sensor"*).
  **P4 = sensor de posição 4** (não o eletrodo P4 nem o sujeito 4). Faixas:
  x ≈ −72,9…−60,1 · y ≈ 7,7…31,6 · z ≈ 27,5…36,3 (dezenas → provavelmente **cm**).
- **Cache já gerado pelo código legado** (`SAND_cache/`): `sample_count = 3528`
  janelas `(3528, 32, 960)` + alvo `(3528, 30, 3)`; há chaves `p4_*` (sensor P4) e
  `loso_raw_*` (com rótulo de sujeito, para LOSO). O corpus de pré-treino já foi
  montado uma vez — 3.528 amostras, ordem de grandeza igual à sua coleta
  (5 participantes × 5 sessões × 250 trials = 6.250 trials).
- ⚠️ `WAY-EEG-GAL/`, `*.mat`, `*.dat` e `SAND_cache/` estão no `.gitignore`
  (5,4 GB nunca devem ir para o GitHub).

## 2. A janela no artigo (do trecho enviado)

O trecho diz, literalmente:

- **Pré-processamento**: banda **0,1–40 Hz**; ICA + ICLabel descartando componentes
  ocular/muscular com **p ≥ 0,90**; *downsampling* **500 → 100 Hz**; **normalização
  min-max para [0,1]** tanto do EEG quanto das **coordenadas de posição** (min/max
  por canal).
- **Métrica**: PCC entre trajetória prevista e ground truth.
- **Janela**: *"EEG signals from the **2–3 second post-stimulus window** were
  selected, covering motor preparation and execution."*
- **Tempo-frequência**: alinhada ao **movimento (0 s)**, janela **±300 ms**,
  0–50 Hz, destacando μ (8–12 Hz) e β (13–30 Hz); o texto conclui que a informação
  útil está em **0–12 Hz** (δ 0,5–3, θ 3–7, α baixo 7–12 Hz).

**Interpretação e implicações:**

1. Se **2–3 s pós-estímulo** é a **janela de entrada do modelo**, ela tem **1 s**
   (100 amostras a 100 Hz, 32 canais) — diferente da nossa de **2 s a 500 Hz**
   (1000 amostras). Reproduzir o do artigo é possível **porque gravamos o EEG
   cru**: basta filtrar 0,1–40 Hz, reamostrar para 100 Hz e normalizar min-max.
2. No WAY-EEG-GAL o "pós-estímulo" tem marco explícito: `LEDon = +2,0 s` e
   `LEDoff = +6,43 s` → **[2,3] s pós-estímulo = [LEDon+2, LEDon+3] = [4,0 s,
   5,0 s]** no eixo da janela (fase de carga/lift).
3. **O trecho não define o ALVO** — o que decide a arquitetura: (a) a trajetória
   **do mesmo intervalo** (decodificação **concorrente**) ou (b) a do movimento
   que ainda vai ocorrer (predição). A leitura sugere (a); confirmar no parágrafo
   de *input/target* do artigo.

---

## 3. Alvo "concorrente" × "preditivo" (o que eu quis dizer)

Duas coisas distintas: **(i)** o que existe em tempo de inferência e **(ii)** o
que o modelo aprende a relacionar.

- **(i) Você está certo:** em uso só há EEG; o Kinect existe **apenas** para criar
  os pares (EEG, trajetória) no treino.
- **(ii) A questão é o deslocamento temporal do alvo em relação à janela:**

```
CONCORRENTE (como esta' hoje):
  EEG   [====================]          (2 s terminando agora, t)
  alvo      [====================]      (a trajetoria DESSE mesmo intervalo)
  -> descreve o movimento que JA' aconteceu; a saida so' fica pronta no fim
     da janela (atraso >= janela).

PREDITIVO (o que controle exige):
  EEG   [====================]
  alvo                        [====]    (a trajetoria dos proximos 0,5 s)
  -> a saida se refere ao FUTURO: e' isso que permite COMANDAR a protese
     enquanto o movimento ainda nao aconteceu.
```

Para uma prótese/braço robótico, o alvo concorrente implica comando atrasado
(o robô reproduz ~1 janela depois). O alvo preditivo é o que torna a aplicação
viável — e é mensurável na **ME**, onde há movimento real adiante.

**Na MI não há movimento executado** → o alvo medido pelo Kinect durante a MI é
~constante (degenerado). O desenho coerente: treinar na **ME** (movimento real)
com o horizonte escolhido (0 = concorrente, > 0 = preditivo) e, na **MI**, avaliar
a saída contra o **template do ME da mesma condição** — papel do **vídeo 0,5×**
(priming): cria a referência compartilhada da trajetória imaginada, permitindo
calcular PCC previsto × template (mesma métrica do artigo, adaptada).

**Estado no nosso código:** o `--window-start-sec` desloca a **janela**, não o
alvo; o alvo é sempre `[início, início + window_sec]`. Faltam (poucas linhas em
`Recording.build_epochs`): `--target-start-sec` / `--target-end-sec` (ou
`--target-lead-sec`) e um modo MI que use o template do ME.

## 4. Vídeo de priming: o que o código faz hoje (e o que ajustar)

**Fatos do código (`eeg_motor_paradigm.py`):**
`HOME 2 s → CUE+ME 4 s → PAUSA+VÍDEO 2 s → MI 4 s`;
`SLOWMO_SPEED = 0,5` · `SLOWMO_SPAN_SEC = 1,0` · `SLOWMO_END_LAG_SEC = 1,0` ·
`SLOWMO_BUFFER_SEC = 2,5`; `_clip_window()` devolve
`[fim_da_ME − 2,0 s, fim_da_ME − 1,0 s]`.

Portanto:

1. O vídeo toca **durante a pausa de 2 s**; a **MI fica limpa** (sem vídeo) ✔.
2. Mostra **1 s real de movimento** (o trecho **2–3 s após a cue**, o *meio* da ME)
   e, em 0,5×, ocupa exatamente os 2 s da pausa.
3. **Não** começa no início do movimento: é um recorte fixo por tempo, **não
   ancorado no movimento**. Sua observação está correta.
4. **Não é o movimento completo** — é o núcleo (transporte + pegada), porque a
   pausa de 2 s comporta só 1 s de material em 0,5×.

| Opção | Como | Prós | Contras |
|---|---|---|---|
| **A (recomendada)** | ancorar o clipe no **início do movimento** detectado pela velocidade do KT (limiar + 100 ms sustentados), mostrando 1 s a 0,5× | usa o início real do movimento; robusto a atrasos do participante | depende da qualidade do KT (validável por `KT_src`) |
| B | pausa de **3 s** + clipe de 1,5 s a 0,5× | mostra mais do movimento | muda o protocolo (trial de 13 s) |
| C | clipe de 1 s a **1,0×** na pausa de 2 s | movimento completo em tempo real | perde o efeito "câmera lenta" |
| D | clipe no **início da MI** (2 s de MI) | vê a ação enquanto imagina | contamina a MI com estímulo visual |

> Recomendação: **A**, mantendo 0,5× e pausa de 2 s, com o clipe salvo em disco
> (já implementado) para auditoria.

---

## 5. Os dois modelos offline (discreto e trajetória)

- **Trajetória:** `sand_traj_treino.py` já treina (alvo 30×3 em metros).
- **Discreto (3 classes):** existe no legado (`sand_bci_model.py`,
  `SAND_BCI_Treino.py`, em `legacy/LEGADO_classificacao_BCI.zip`), mas ele lê o
  **BCI IV 2a** (`.gdf`/cache), **não** as nossas sessões. Para o modelo discreto
  no seu protocolo falta um treinador adaptado que use os rótulos dos
  `*_eventos.json` (811–816 = condição; 769/770 = mão; "nada" = repouso).
- A aquisição offline já gera tudo o que os dois precisam (EEG cru a 500 Hz +
  eventos com amostra exata + trajetória a 30 Hz + metadados).

---

## 6. Pré-treino com WAY-EEG-GAL: viável? **Sim** — com ressalvas

**Por que faz sentido:** 3.528 janelas de 32 canais de uma tarefa de **pegar e
levantar** (3 pesos) com cinemática de sensor de posição — dimensão comparável à
sua coleta; e a montagem de 32 canais **inclui** os canais que você vai analisar.
Serve para o backbone SAND aprender o acoplamento **EEG de baixa frequência ↔
cinemática** antes de ver os seus dados.

**Ressalvas honestas (domain shift):**
1. tarefa diferente (pegar-e-levantar × alcançar-e-pegar 3 objetos, duas mãos);
2. amplificador/montagem/participantes diferentes (32 ch padrão × seu g.Nautilus);
3. **unidades/escala do alvo** (posições do P4 em cm; o artigo normaliza min-max
   [0,1]) × nosso alvo em **metros** com z-score → a transferência exige
   **fine-tuning** (o normalizador resolve a escala);
4. janela/faixa: nossa 2 s @ 500 Hz e 1–30 Hz × do artigo 1 s @ 100 Hz,
   0,1–40 Hz, min-max → escolher o que replicar.

**Plano em 4 passos (cada um verificável):**
1. `tools/inventario_way_eeg_gal.py` — inventário (feito).
2. Carregador `--dataset waygal` no `sand_traj_treino.py`: lê `ws.win`, aplica o
   pré-processamento escolhido (`--norm minmax|zscore`, `--downsample-to 100`),
   recorta o tempo (`LEDon+2 s` etc.) e monta `(X, Y, sessão)` com alvo
   `kin[:, [21, 25, 29]]` (Px4/Py4/Pz4).
3. Pré-treinar → `sand_traj_pretrain_waygal.pt` (+ validação LOSO dentro do
   WAY-EEG-GAL, que com 12 sujeitos já é uma validação cruzada séria).
4. `--init-from sand_traj_pretrain_waygal.pt` para **fine-tuning** nas suas
   sessões (sua montagem, alvo em metros).

**Riscos:** com poucos dados seus o fine-tuning pode não superar treinar do zero;
o teste honesto é comparar (a) do zero × (b) pré-treinado e afinado, sempre com
**hold-out por sessão** (já implementado) e, idealmente, **LOSO**.

---

## 7. Decisões pendentes (para o autor do projeto)

1. Replicar a janela do artigo (1 s @ 100 Hz, pós-estímulo, min-max [0,1]) ou
   manter a nossa (2 s @ 500 Hz, z-score)? (afeta pré-treino e comparação)
2. Alvo **concorrente** ou **preditivo** (horizonte de 0,5–1 s)? Para controle de
   prótese, preditivo; para comparar com o artigo, concorrente.
3. Na MI: alvo = **template do ME da mesma condição**? (métrica = PCC vs template)
4. Vídeo de priming: ancorar no **início do movimento** (opção A)?
5. O modelo **discreto** (3 classes) sai do legado adaptado às nossas sessões?
6. Confirmar no artigo: a janela [2,3] s pós-estímulo é a **entrada do modelo** ou
   só a janela de análise? E qual é o **alvo** (mesmo intervalo ou movimento
   completo)?

---

## 8. Nota de manutenção (incidente evitado)

Durante esta análise, um `git add -A` chegou a preparar os **5,4 GB** do
WAY-EEG-GAL (o dataset não estava no `.gitignore`). O processo foi interrompido, o
`index.lock` removido, o staging desfeito e o `.gitignore` passou a excluir
`WAY-EEG-GAL/`, `*.mat`, `*.dat` e `SAND_cache/`. Os objetos órfãos (908 MB) foram
limpos com `git gc --prune=now` → `.git` voltou a **10,6 MB**, remoto intacto.
O script `tools/publicar.ps1` já bloqueia arquivos > 50 MB como segunda barreira.