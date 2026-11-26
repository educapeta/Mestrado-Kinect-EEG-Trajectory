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
deve ser reportado como tal. A validação neural precisa ser refeita
within-participant ou com normalização por participante antes de afirmar que
a arquitetura neural funciona.

## 6. Arquivos

- loader: `src/mestradopy/eeg/way_gal/loader.py`
- benchmark: `src/mestradopy/tools/way_gal_bench.py`
- CNN: `src/mestradopy/tools/way_gal_cnn.py`
- QC do loader: `outputs/eeg/way_gal/loader_qc.json`
- auditoria: `outputs/eeg/way_gal/auditoria_trials.json`
- alvos: `outputs/eeg/way_gal/targets.npz`
- benchmark: `outputs/eeg/way_gal/regression_results.json`
- relatório do benchmark: `outputs/eeg/way_gal/WAY_GAL_BENCH_20261126.md`
- CNN: `outputs/eeg/way_gal/cnn/cnn_results.json`

## 7. Comandos reproduzíveis

```powershell
cd C:\mestradopy
$env:PYTHONPATH="C:\mestradopy;C:\mestradopy\src"

.\.venv\Scripts\python.exe src\mestradopy\tools\way_gal_bench.py --save-targets

.\.venv\Scripts\python.exe src\mestradopy\tools\way_gal_cnn.py --epochs 25
```

## 8. Limitações e o que falta

- WAY tem ME, não MI; não valida decodificação de imagética;
- apenas mão direita;
- alvo final de pose é quase constante (mesmo objeto/local);
- não há validação de orientação/grip como pose completa;
- LOSO cross-participant permanece um problema em aberto;
- o EEGNet não superou o baseline;
- a unidade de posição é assumida como cm e deve ser confirmada na
  documentação do dataset;
- a contagem 3.528 vs 3.936 precisa ser explicada na tese.
