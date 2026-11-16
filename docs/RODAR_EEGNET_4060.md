# Rodar EEGNet do Faria na RTX 4060

## 1. Ambiente no laptop

Se ainda não tiver PyTorch com CUDA:

```powershell
cd C:\caminho\mestradopy
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install mne pandas scikit-learn
```

Confirme:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Deve aparecer `True` e `NVIDIA GeForce RTX 4060 ...`.

## 2. Dados

Coloque os arquivos em:

```text
C:\caminho\mestradopy\Datasets\BCICIV_2a_gdf\A01T.gdf ...
C:\caminho\mestradopy\Datasets\BCICIV_2b_gdf\*.gdf
```

## 3. Rodar os casos da Tabela 4.1

Use `--device cuda --epochs 500`:

```powershell
.\.venv\Scripts\python.exe tools\eegnet_faria_benchmark.py `
  --dataset 2a --n-canais 22 --n-classes 4 `
  --folds 4 --epochs 500 --device cuda `
  --saida _eegnet_faria_500

.\.venv\Scripts\python.exe tools\eegnet_faria_benchmark.py `
  --dataset 2a --n-canais 3 --n-classes 4 `
  --folds 4 --epochs 500 --device cuda `
  --saida _eegnet_faria_500

.\.venv\Scripts\python.exe tools\eegnet_faria_benchmark.py `
  --dataset 2a --n-canais 22 --n-classes 2 `
  --folds 4 --epochs 500 --device cuda `
  --saida _eegnet_faria_500
```

Quando o 2b estiver na pasta:

```powershell
.\.venv\Scripts\python.exe tools\eegnet_faria_benchmark.py `
  --dataset 2b --n-canais 3 --n-classes 2 `
  --folds 4 --epochs 500 --device cuda `
  --saida _eegnet_faria_500
```

Ou simplesmente execute:

```powershell
.\run_eegnet_4060.ps1
```

## 4. O que o script valida

- EEGNet idêntico ao `project/models/eegnet.py` do repositório do Faria:
  - `kernel_length = 32` (`n_times//2//4`);
  - `F1=8`, `D=2`, `F2=16`;
  - convoluções sem bias, FC com bias;
  - saída com `log_softmax`? na nossa versão usamos logits + CrossEntropy,
    mantendo a mesma contagem de parâmetros.
- Contagens exatas da Table 4.1:
  - 22ch/4cls = 1.716;
  - 3ch/4cls = 1.412;
  - 22ch/2cls = 1.458;
  - 3ch/2cls = 1.154.
- Pré-processamento: 4–40 Hz, 128 Hz, padronização exponencial,
  janela de 2 s (0,5–2,5 s após o cue), 4-fold com validação interna.

## 5. Limitação importante

O GDF local do 2a tem labels só na sessão **T** (a E tem cue 783 sem
rótulo). Portanto o 4-fold é feito dentro da T, não é o cross-session
completo do Faria. O número de referência do Faria (0,566 para 3 canais/4
classes) usa as duas sessões; a comparação é aproximada.

O 2b será carregado do mesmo jeito (3 canais, 2 classes) quando você subir
os `.gdf`.
