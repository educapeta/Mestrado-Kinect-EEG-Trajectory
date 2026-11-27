# WAY-EEG-GAL — validação final do pipeline (26/11/2026)

## 1. Dataset local

- 12 participantes (`P1`–`P12`);
- 108 séries (`WS_P*_S*.mat`);
- 3.528 trials carregados (294 por participante);
- 32 canais de EEG a 500 Hz;
- 45 colunas de cinemática a 500 Hz;
- 5 canais de EMG a 4 kHz;
- mão direita, ME (não há MI).

**Auditoria:** o paper descreve 3.936 trials. A cópia local contém 3.528
trials válidos em `P*_AllLifts.mat` (294 por participante) e 108 séries. Essa
diferença precisa ser explicada na tese como característica da release local,
não como falha do loader.

## 2. Definição dos alvos

Janela ME:

```text
[tHandStart - 0,2 s, tReplace + 0,3 s]
```

Hold real:

```text
[tLiftOff, tReplace]
```

Alvos principais (relativos):

| alvo | definição | unidade |
|---|---|---|
| `wrist_pos_disp` | P4 no hold − P4 no repouso | cm |
| `wrist_vel_mean` | velocidade média do reach | cm/s |
| `wrist_vel_peak` | pico de velocidade | cm/s |
| `wrist_vel_at_lift_off` | velocidade no lift-off | cm/s |
| `wrist_orient_disp` | orientação no hold − repouso | deg |
| `grip_aperture_disp` | abertura P2–P3 no hold − repouso | cm |
| `grip_force_disp` | GF no hold − repouso | N |

## 3. Within-participant (validação principal)

GroupKFold por trial, 5 folds, todos os 12 participantes. Média dos
participantes; baseline = média do treino no mesmo fold.

### Melhor feature por alvo

| alvo | feature | PCC (média ± dp) | MAE | R² | MAE baseline |
|---|---|---:|---:|---:|---:|
| `wrist_pos_disp` | time_bins | **0,234 ± 0,170** | 1,326 cm | 0,075 | 1,715 cm |
| `wrist_vel_mean` | bandpower | **0,383 ± 0,139** | 1,330 cm/s | 0,051 | 2,792 cm/s |
| `wrist_vel_peak` | time_bins | 0,264 ± 0,146 | 4,809 cm/s | 0,083 | 7,748 cm/s |
| `wrist_vel_at_lift_off` | simple | 0,003 ± 0,098 | 0,840 cm/s | −0,005 | 0,926 cm/s |
| `wrist_orient_disp` | time_bins | 0,275 ± 0,183 | 3,450° | 0,096 | 6,480° |
| `grip_aperture_disp` | time_bins | 0,213 ± 0,131 | 0,791 cm | 0,056 | 1,641 cm |
| `grip_force_disp` | bandpower | **0,305 ± 0,100** | 2,017 N | 0,025 | 2,603 N |

Leitura:

- existe sinal acima da média para posição, velocidade, orientação,
  abertura e força;
- os PCCs são modestos (0,20–0,38), como esperado para 32 canais e ME;
- `wrist_vel_at_lift_off` não traz informação útil;
- a arquitetura não está validada para pose completa; está validada apenas
  como **regressão relativa dentro do participante**.

## 4. LOSO cross-participant

O LOSO com alvos relativos continua negativo para praticamente tudo
(PCC −0,05 a −0,44). Isso não invalida a arquitetura: a posição/velocidade
absoluta e a biomecânica variam entre participantes. O LOSO é uma referência
de generalização entre sujeitos, não o critério de validação da arquitetura.

## 5. EEGNet neural (teste cross-participant)

Modelo: EEGNet configurável, 32 canais, 2 s iniciais do ME, 3 saídas de
deslocamento, GroupKFold por participante, 25 épocas, scaler fit no treino.

| métrica | valor |
|---|---:|
| PCC eixo X | 0,203 |
| PCC eixo Y | 0,212 |
| PCC eixo Z | −0,131 |
| **PCC médio** | **0,095** |
| MAE médio | 1,963 cm |
| MAE baseline | 1,723 cm |
| R² médio | −0,184 |

Conclusão: o EEGNet **não generalizou entre participantes** para deslocamento
3D. Ele não superou o baseline de MAE. Isso é um resultado negativo válido e
deve ser reportado como tal.

## 6. EEGNet within-participant (protocolo SAND-like)

Protocolo:

- um modelo por participante;
- split por **série/sessão**, não por trial:
  - 6 séries de treino;
  - 1 série de validação interna (early stopping);
  - 2 séries de teste, nunca usadas no treino/validação;
- z-score de EEG com estatísticas do treino;
- 32 canais, 2 s do ME, 256 amostras;
- 30 épocas, Adam, early stopping;
- alvo `wrist_pos_disp` (3D).

Resultado por participante (PCC médio dos 3 eixos):

| participante | PCC |
|---|---:|
| P1 | 0,365 |
| P2 | 0,147 |
| P3 | −0,229 |
| P4 | 0,157 |
| P5 | −0,143 |
| P6 | 0,398 |
| P7 | 0,426 |
| P8 | 0,042 |
| P9 | 0,257 |
| P10 | 0,093 |
| P11 | 0,493 |
| P12 | −0,006 |

Agregado:

| métrica | valor |
|---|---:|
| **PCC médio por participante** | **0,167** |
| desvio entre participantes | 0,220 |
| PCC eixo X | 0,363 |
| PCC eixo Y | 0,507 |
| PCC eixo Z | 0,789 |
| MAE médio | 1,308 cm |

Importante: o PCC “agregado” de 0,553 (calculado concatenando participantes)
é inflado pela diferença de escala entre participantes; a métrica honesta para
comparar com o Ridge within é a **média por participante = 0,167**.

Leitura:

- o EEGNet within-participant **não superou** o baseline Ridge within
  (`time_bins`, PCC ≈ 0,234 para posição);
- 4 participantes ficaram negativos/zero;
- a arquitetura neural ainda não está validada para pose 3D;
- o resultado é útil como linha de base negativa e mostra que a abordagem
  SAND-like precisa de mais dados, regularização ou alvo de trajetória.

### Como o SAND deste repositório foi treinado

- modelo por gravação/sujeito;
- split por **sessão** (`--test-frac 0.2`), não LOSO;
- z-score com estatísticas do treino;
- saída de trajetória (30 pontos × 3);
- MSE + termo de suavidade + regularizador anatômico opcional;
- early stopping na validação;
- não é um protocolo cross-subject.

O EEGNet within acima copia o **espírito do protocolo** (modelo por sujeito,
separação por sessão, validação interna e teste separado), mas não é o SAND
completo: falta a saída de trajetória temporal, o termo de suavidade e o
regularizador anatômico.

## 7. Arquivos

- loader: `src/mestradopy/eeg/way_gal/loader.py`
- benchmark: `src/mestradopy/tools/way_gal_bench.py`
- CNN cross-participant: `src/mestradopy/tools/way_gal_cnn.py`
- CNN within-participant: `src/mestradopy/tools/way_gal_cnn_within.py`
- QC do loader: `outputs/eeg/way_gal/loader_qc.json`
- auditoria: `outputs/eeg/way_gal/auditoria_trials.json`
- alvos: `outputs/eeg/way_gal/targets.npz`
- benchmark: `outputs/eeg/way_gal/regression_results.json`
- relatório do benchmark: `outputs/eeg/way_gal/WAY_GAL_BENCH_20261126.md`
- CNN cross-participant: `outputs/eeg/way_gal/cnn/cnn_results.json`
- CNN within-participant: `outputs/eeg/way_gal/cnn_within/cnn_within_results.json`

## 8. Comandos reproduzíveis

```powershell
cd C:\mestradopy
$env:PYTHONPATH="C:\mestradopy;C:\mestradopy\src"

.\.venv\Scripts\python.exe src\mestradopy\tools\way_gal_bench.py --save-targets

.\.venv\Scripts\python.exe src\mestradopy\tools\way_gal_cnn.py --epochs 25

.\.venv\Scripts\python.exe src\mestradopy\tools\way_gal_cnn_within.py --epochs 30
```

## 9. Limitações e o que falta

- WAY tem ME, não MI; não valida decodificação de imagética;
- apenas mão direita;
- alvo final de pose é quase constante (mesmo objeto/local);
- não há validação de orientação/grip como pose completa;
- LOSO cross-participant permanece um problema em aberto;
- o EEGNet within-participant (PCC médio 0,167) não superou o baseline
  Ridge within (0,234) e teve 4 participantes negativos/zero;
- o resultado neural ainda é uma linha de base negativa, não uma arquitetura
  validada;
- a unidade de posição é assumida como cm e deve ser confirmada na
  documentação do dataset;
- a contagem 3.528 vs 3.936 precisa ser explicada na tese.
