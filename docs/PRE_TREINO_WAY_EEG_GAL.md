# Pré-treinamento com o WAY-EEG-GAL — procedimento exato

> Este documento é o **passo a passo** do pré-treinamento, para ser executado
> **depois** que o tamanho/âncora da janela estiver fechado (ver
> `docs/ESTADO_DA_ARTE_JANELAS.md`). Ele separa o que **já existe** no código do
> que **precisa ser implementado**, para não prometer nada pronto.

## 0. O que precisa estar decidido antes (30 min de conversa, não de código)

| Decisão | Valor de hoje | Por que importa para o pré-treinamento |
| --- | --- | --- |
| Janela `T` e taxa | 2 s @ **500 Hz = 1000 amostras** | o cache do WAY-EEG-GAL tem de usar **a mesma T**; como ele é 500 Hz nativo, a janela casa 1:1 (SAND usa 100 Hz/T=960, M3T 100 Hz/T=500 — se migrarmos para 100 Hz, o cache muda) |
| Banda | `--f-lo 1 --f-hi 30` | precisa ser **idêntica** no pré-treino e no ajuste fino (SAND usa 0,1–40 Hz) |
| Âncora | `--event-code 795 --window-start-sec -0.5` | o WAY-EEG-GAL não tem Kinect: o onset tem de ser **detectado da cinemática do P4** — usamos o nosso `MovementOnsetDetector`, com limiar reescalado |
| Alvo | preditivo (`--target-start-sec 1.5`) | o alvo do pré-treino tem de ter o **mesmo deslocamento** do ajuste fino |
| Montagem | 32 canais do g.Nautilus | o cache do WAY-EEG-GAL deve ser montado com a **interseção por nome** das duas montagens, **na nossa ordem** (senão o modelo pré-treinado não encaixa) |
| `output_seq_len` | 30 | igual nos dois lados |

Pré-requisito de dados (já atendido): o inventário
`tools/inventario_way_eeg_gal.py` confirmou **12 sujeitos × 9 séries, 5,38 GB**,
com `eeg` (L × 32 @ 500 Hz), `kin` (L × 62), `t` e os eventos de LED.
O sensor de referência é o **P4 no dorso da mão** → `kin[:, [21, 25, 29]]`
(Px4/Py4/Pz4), como já validamos.

## 1. Passo 1 — construir o cache no MESMO formato dos nossos dados

**Objetivo**: transformar 5,38 GB de `.mat` em um `.npz` por sujeito com
`X (N, 32, T)` e `Y (N, 30, 3)`, **exatamente** como `load_all_epochs` produz —
assim o pré-treino reaproveita `preprocess_epochs`, `TrajectoryNormalizer` e
`run_training` sem nenhuma linha nova.

Ainda **não implementado** (script a criar: `tools/prepara_way_eeg_gal.py`):

```text
para cada sujeito (01..12):
    para cada série (1..9):
        scipy.io.loadmat(<sujeito>/data_<S>_<N>.mat)
        eeg  = data["eeg"]            # (L, 32) @ 500 Hz
        kin  = data["kin"]            # (L, 62)
        t    = data["t"]
        kt   = kin[:, [21, 25, 29]]   # Px4/Py4/Pz4 (m)
        # âncora: onset detectado por cinemática (mesmo detector do programa 1)
        onset_detector = MovementOnsetDetector(limiar_velocidade=<recalibrado>)
        para cada LEDon (início do trial):
            onset = detector.detecta(kt, a partir de LEDon)
            X[i] = eeg[onset - 0.5 s : onset + 1.5 s]      # (32, 1000)
            Y[i] = kt[onset + 1.5 s : onset + 2.5 s]       # reamostrado em 30 pontos
        # igual ao nosso pipeline: reordena para 32 canais na ORDEM da nossa
        # montagem (interseção por nome), z-score por janela + CAR
    salvar way_eeg_gal_<S>.npz  com X, Y, sujeito, série, trial
    salvar way_eeg_gal_meta.json  (fs, banda, T, âncora, alvo, montagem, SHA1)
```

Ponto de atenção 1 — **onset sem Kinect**: no WAY-EEG-GAL o movimento é um
*reach-and-grasp* de ~0,5 m; o limiar de 0,03 m/s que calibramos para a mão
apoiada continua plausível, mas **tem de ser validado** (sugestão: rodar o
detector em 5 trials de 3 sujeitos e olhar a distribuição dos instantes
detectados em relação ao LED; se ficar bimodal, o limiar está errado).

Ponto de atenção 2 — **alinhamento temporal**: `t` do `.mat` é o tempo da
coleta; o LED e a cinemática estão na mesma base, então a janela pode ser
recortada por amostra, sem usar o nosso `MotionTrack`.

## 2. Passo 2 — pré-treino LOSO interno (leave-one-subject-out)

Com o cache pronto, o pré-treino é o **mesmo treino** de hoje, repetido 12 vezes
com o sujeito de validação trocando — e nenhuma sessão do sujeito de validação
entrando no treino:

```text
para k em 1..12:                      # LOSO
    treino  = sujeitos != k           # ~11 x 9 séries x ~5-6 trials
    valid.  = sujeito k
    resultado_k = run_training(args_pretreino, X_treino, Y_treino,
                               X_val_k, Y_val_k, channel_map,
                               meta={"dataset": "way-eeg-gal",
                                     "fold": k, "sujeito_val": k})
    guardar (val_loss_k, PCC_k)  e o config

checkpoint final (pre_treino_way.pt):
    - escolher o fold com a melhor val_loss media,
      OU refazer o treino em TODOS os sujeitos com o numero de epocas
        descoberto no LOCO (pratica mais comum);
    - config: {'pretreino': 'way-eeg-gal', 'loso_folds': 12,
               'pcc_medio_loso': {...}, 'T': 1000, 'f_lo': 1, 'f_hi': 30}
```

Interface sugerida (a implementar em `sand_traj_treino.py`, ~60 linhas):

```powershell
.venv\Scripts\python.exe sand_traj_treino.py --cache-way way_eeg_gal_cache `
    --pretreino-loso --epochs 30 --window-sec 2.0 --window-start-sec -0.5 `
    --target-start-sec 1.5 --target-end-sec 2.5 --f-lo 1 --f-hi 30 `
    --model-out pre_treino_way.pt
```

O que **já existe e é reutilizado sem mudança**: `preprocess_epochs`,
`TrajectoryNormalizer`, `SANDTrajectory`, `run_training`, `save_checkpoint`,
`validate_epoch` (PCC), *early stopping*, `ReduceLROnPlateau`.

## 3. Passo 3 — transferência para as nossas gravações

```powershell
# (a) ajuste fino de TUDO, LR menor
.venv\Scripts\python.exe sand_traj_treino.py --data gravacoes `
    --init-from pre_treino_way.pt --lr 1e-4 --epochs 30 `
    --event-code 795 --window-start-sec -0.5 --window-sec 2.0 `
    --target-start-sec 1.5 --target-end-sec 2.5 `
    --model-out sand_ajustado_way.pt

# (b) só a CABEÇA (encoder congelado) -- útil com poucas sessões
.venv\Scripts\python.exe sand_traj_treino.py --data gravacoes `
    --init-from pre_treino_way.pt --congela-encoder --lr 1e-3 ...
```

`--init-from` (a implementar, ~20 linhas em `run_training`): carregar
`torch.load(...)["model_state"]` com `strict=True` **antes** de criar o
`optimizer`, e registrar no `config` do checkpoint novo:
`'init_from': path, 'init_sha1': <sha1 do arquivo>`. Se `strict=True` falhar,
o motivo será montagem/T diferentes → **não** usar `strict=False` (isso esconde
erro de engenharia); corrigir a montagem.

Três cuidados que decidem o resultado:

1. **Ordem dos canais**: o cache do WAY-EEG-GAL tem de sair já na ordem da nossa
   montagem (`channel_map` gravado no JSON). Sem isso o modelo aprende a
   mapear canais errados e o ajuste fino **piora** em relação a treinar do zero.
2. **Banda e normalização idênticas** nos dois lados (`--f-lo/--f-hi`,
   `--zscore`, CAR ligado/desligado).
3. **Reportar os dois números**: com e sem pré-treinamento, mesmas sessões e
   mesmo split. É a única forma de sustentar a afirmação de que o
   pré-treinamento ajuda.

## 4. Alternativa (sem o WAY-EEG-GAL) — pré-treino auto-supervisionado

Se a montagem/conjunto atrapalhar, há um caminho que usa **só as nossas
gravações** e é mais fácil de defender: pré-treinar o *encoder* com
reconstrução mascarada (mascarar 20 % das amostras e reconstruir o sinal),
descartar a cabeça e depois treinar o alvo de trajetória. Custo: ~80 linhas
(uma cabeça de reconstrução + o loop) e **nenhum** dataset externo. Vantagens:
sem problema de montagem, sem dúvida de vazamento e funciona mesmo com 3-4
sessões. Vale comparar os dois caminhos.

## 5. Checklist do que falta no código (ordem de execução)

| # | Item | Onde | Tamanho |
| --- | --- | --- | --- |
| 1 | validação do `MovementOnsetDetector` no WAY-EEG-GAL (limiar + mínimo de deslocamento) | `tools/prepara_way_eeg_gal.py` | médio |
| 2 | `tools/prepara_way_eeg_gal.py` (cache `.npz` por sujeito + `meta.json` com SHA1) | novo | médio |
| 3 | mapa de canais por NOME (interseção das montagens, na nossa ordem) | `docs/` + JSON | pequeno |
| 4 | `--cache-way` + `--pretreino-loso` | `sand_traj_treino.py` | médio |
| 5 | `--init-from` e `--congela-encoder` | `sand_traj_treino.py` | pequeno |
| 6 | teste de regressão do cache (formas, LOSO sem vazamento, SHA1) | `test_pre_treino.py` | pequeno |

