# Rodar e acompanhar o EEGNet no PowerShell

Há dois ambientes possíveis:

- **Intel Arc A580 desta máquina**: `--device xpu`;
- **RTX 4060 do laptop**: `--device cuda` (instale PyTorch CUDA).

## 1. Abrir o PowerShell na raiz do projeto

```powershell
cd C:\mestradopy
```

## 2. Ativar o `.venv`

```powershell
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Conferir o dispositivo

Arc:

```powershell
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```

4060:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Rodar um caso com log ao vivo

Arc — 3 canais / 4 classes:

```powershell
python tools\eegnet_faria_benchmark.py `
  --dataset 2a --n-canais 3 --n-classes 4 `
  --folds 4 --epochs 500 --device xpu --verbose `
  --saida _eegnet_faria_500 2>&1 |
  Tee-Object -FilePath _eegnet_faria_500\log_2a_3ch_4cls.txt
```

4060 — troque apenas `xpu` por `cuda`.

O `--verbose` mostra:

- sujeito/fold que está treinando;
- a cada 50 epochs: `val_loss`;
- ao final de cada fold: `test_acc`;
- ao final do caso: acurácia média ± desvio.

## 5. Acompanhar em outra janela

```powershell
Get-Content .\_eegnet_faria_500\log_2a_3ch_4cls.txt -Wait -Tail 30
```

Para parar: `Ctrl+C` na janela do treino.

## 6. Rodar todos os casos

Arc:

```powershell
.\run_eegnet_arc.ps1
```

Só um caso:

```powershell
.\run_eegnet_arc.ps1 -Caso 3ch4cls
```

Teste rápido de 20 epochs:

```powershell
.\run_eegnet_arc.ps1 -Caso 3ch4cls -Epochs 20
```

4060:

```powershell
.\run_eegnet_4060.ps1
```

## 7. O que significam os números

- 500 epochs é o protocolo do `run-example.sh` do repositório do Faria;
- nosso 4-fold é sobre a sessão T (a E não tem labels de classe);
- não compare ainda com 0,566/0,696/0,798 até rodar os 500 epochs completos.
