# mi_eegnet — EEGNet configurável e validação honesta

Esta pasta implementa a metodologia consolidada para MI binária
(esquerda × direita) com:

- BCIC-IV 2b local em GDF (`Datasets/BCICIV_2b_gdf`);
- pré-treino **LOSO** (9 sujeitos);
- fine-tuning em 20 trials próprios (100 janelas);
- teste final reservado em 20 trials;
- CSP+LDA e EEGNet do zero como baselines;
- validação agrupada por trial, sem vazamento;
- métricas por janela e por trial com soft voting;
- bootstrap por trial.

## Modelo

EEGNet padrão (blocos Lawhern et al., 2018) + camada densa oculta
configurável. Com `classifier_hidden=36`:

```text
5.870 parâmetros treináveis
entrada: (B, 3, 256)
```

O texto da metodologia citava ~5.900; a variante foi calibrada para
**5.870** parâmetros. Não é o EEGNet original de 1.410 parâmetros.

## Estrutura

```text
mi_eegnet/
├── configs/default.yaml
├── data.py          # loaders 2b local + dataset próprio
├── model.py         # EEGNet configuravel
├── train_utils.py   # scaler, treino, soft voting, bootstrap
├── pretrain.py      # LOSO no 2b
├── finetune.py      # fine-tuning + teste final
├── baselines.py     # CSP+LDA, EEGNet do zero, teste final
└── tests/
    ├── test_eegnet.py
    ├── test_no_leakage.py
    ├── test_scaler.py
    └── test_prediction.py
```

## Instalação

No ambiente do projeto:

```powershell
cd C:\mestradopy
.\.venv\Scripts\Activate.ps1
python -m pip install pyyaml
```

## Testes

```powershell
python -m unittest discover -s mi_eegnet/tests -v
```

## 1. Pré-treino LOSO no 2b

```powershell
python -m mi_eegnet.pretrain --device xpu
```

No laptop com 4060: `--device cuda`.

Saídas:

```text
_mi_eegnet_out/pretrain/
  eegnet_pretrained.pt
  eegnet_fold_s01.pt ... eegnet_fold_s09.pt
  history.csv
  summary.json
```

Critério: BA média LOSO. O melhor fold é salvo como modelo pré-treinado.

## 2. Fine-tuning + teste final

```powershell
python -m mi_eegnet.finetune `
  --pretrained _mi_eegnet_out/pretrain/eegnet_pretrained.pt `
  --device xpu
```

Saídas:

```text
_mi_eegnet_out/finetune/
  pretrained_A_folds.csv
  pretrained_B_folds.csv
  eegnet_finetuned.pt
  results.json
```

O teste final usa **apenas os 20 trials reservados** e é feito uma única vez,
depois da escolha da estratégia.

## 3. Baselines no mesmo split

```powershell
python -m mi_eegnet.baselines --device xpu
```

Saídas:

```text
_mi_eegnet_out/baselines/baselines.json
```

Inclui:

- CSP+LDA (5-fold + teste final);
- EEGNet do zero (5-fold + teste final);
- resultado do EEGNet pré-treinado + fine-tuning.

## Preprocessamento

- 2b: filtro 8–32 Hz, 128 Hz, janela de 2 s;
- próprio: mesmo filtro/resample, janela de 2 s com passo de 0,5 s;
- scaler por canal ajustado **somente** no treino de cada fold;
- GroupKFold por trial: nenhuma janela do mesmo trial cruza treino/teste.

## Observações

- O 2b é lido das sessões **T** rotuladas; as E não têm labels de classe.
- O dataset próprio tem 40 trials L/R; 20 treino, 20 teste final.
- Todas as janelas de um trial ficam no mesmo fold.
- O resultado de 20 trials tem variância alta; reporte o IC bootstrap, não só
  a acurácia pontual.
