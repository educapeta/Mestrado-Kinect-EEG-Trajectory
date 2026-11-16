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
python -W ignore tools\eegnet_faria_benchmark.py `
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


---

# Caso focado: BCIC-IV 2a 3 canais / 4 classes (1.412 parâmetros)

## 1. Treinar o benchmark 4-fold (Faria)

```powershell
python -W ignore tools\eegnet_faria_benchmark.py `
  --dataset 2a --n-canais 3 --n-classes 4 `
  --folds 4 --epochs 500 --device xpu --verbose `
  --saida _eegnet_faria_500 2>&1 |
  Tee-Object -FilePath _eegnet_faria_500\log_3ch4cls.txt
```

No laptop com 4060, use `--device cuda`.

## 2. Treinar o modelo final e aplicar no MI do piloto

Isso treina no BCI inteiro, salva `eegnet_2a_3ch_4cls.pt` e aplica nos seus
4 s de MI com janelas de 2 s (passo 0,5 s):

```powershell
python -W ignore tools\eegnet_faria_aplicar_mi.py `
  --device xpu --epochs 500 `
  --saida _eegnet_faria_3ch4cls 2>&1 |
  Tee-Object -FilePath _eegnet_faria_3ch4cls\log_aplicacao.txt
```

O script salva:

- `eegnet_2a_3ch_4cls.pt`;
- `janelas_mi_2s.npz` com `X=(60,5,3,256)`, `y`, `blocos`;
- `predicoes_piloto.npz`;
- `resultado_aplicacao_mi.md`.

## 3. Aplicar depois sem retreinar

```powershell
python -W ignore tools\eegnet_faria_aplicar_mi.py `
  --device xpu `
  --carregar-modelo _eegnet_faria_3ch4cls\eegnet_2a_3ch_4cls.pt `
  --saida _eegnet_faria_3ch4cls_aplicacao 2>&1 |
  Tee-Object -FilePath _eegnet_faria_3ch4cls_aplicacao\log.txt
```

## 4. Apenas gerar as janelas deslizantes

```powershell
python tools\gera_janelas_mi_piloto.py `
  --piloto _piloto3c_memi `
  --saida _janelas_mi_piloto
```

Saída: `janelas_mi_2s.npz` e `janelas_mi_2s.json`.

## 5. O que o relatório de aplicação mostra

- predição do modelo BCI para cada trial do piloto;
- acurácia 4-class apenas nos trials esquerda/direita;
- acurácia binária restrita left/right;
- distribuição dos trials de repouso (o BCI não tem classe repouso; tem
  pés/língua, então não existe acurácia 3-class equivalente).
