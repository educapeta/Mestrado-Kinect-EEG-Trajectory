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


---

# BCIC-IV 2b — 3 canais / 2 classes / 1.154 parâmetros

## 1. Benchmark 4-fold

```powershell
python -W ignore tools\eegnet_faria_benchmark.py `
  --dataset 2b --n-canais 3 --n-classes 2 `
  --folds 4 --epochs 500 --device xpu --verbose `
  --saida _eegnet_faria_2b_benchmark 2>&1 |
  Tee-Object -FilePath _eegnet_faria_2b_benchmark\log_2b_3ch_2cls.txt
```

No laptop com 4060, troque `--device xpu` por `--device cuda`.

## 2. Treinar o modelo final e testar left/right do piloto

```powershell
python -W ignore tools\eegnet_faria_2b_aplicar_mi.py `
  --device xpu --epochs 500 --batch-size 32 --progress-every 10 `
  --saida _eegnet_faria_2b_3ch2cls 2>&1 |
  Tee-Object -FilePath _eegnet_faria_2b_3ch2cls\log_aplicacao_2b.txt
```

O `--batch-size 32` segue o Faria; use `128` para ir mais rápido.

Saídas:

```text
_eegnet_faria_2b_3ch2cls/eegnet_2b_3ch_2cls.pt
_eegnet_faria_2b_3ch2cls/cache_bci_2b.npz
_eegnet_faria_2b_3ch2cls/janelas_mi_left_right.npz
_eegnet_faria_2b_3ch2cls/resultado_2b_teste_mi.md
```

## 3. Aplicar sem retreinar

```powershell
python -W ignore tools\eegnet_faria_2b_aplicar_mi.py `
  --device xpu `
  --carregar-modelo _eegnet_faria_2b_3ch2cls\eegnet_2b_3ch_2cls.pt `
  --saida _eegnet_faria_2b_3ch2cls_aplicacao
```

## 4. Interpretação

O relatório mostra:

```text
BINÁRIO NO PILOTO: acc=... | BA=... | n=40
Confusão (linha=verdade, coluna=pred):
[[... ...]
 [... ...]]
```

- chance binária = 50%;
- são usados **somente** os 20 trials de esquerda + 20 de direita;
- repouso é descartado;
- o modelo vem do **2b**, com 3.680 épocas T de 9 sujeitos.


---

# Validação 1 e 2 na CPU — ME positivo + labels embaralhados

## Rodar

```powershell
python -W ignore tools\validacao_me_shuffle.py --permutacoes 1000
```

Saídas:

```text
_validacao_me_shuffle/relatorio.md
_validacao_me_shuffle/resultados.json
```

## O que o script faz

- carrega ME e MI de `_piloto3c_memi`;
- usa somente esquerda/direita (20 + 20 trials);
- gera 5 janelas de 2 s por trial (passo 0,5 s);
- classifica bandpower alpha/beta com LDA shrinkage;
- faz 5-fold agrupado por trial (sem vazamento);
- calcula acurácia, balanced accuracy e confusão;
- embaralha labels 1.000 vezes e calcula p-valor.

## Leitura dos resultados

- ME: BA 0,675, p=0,043 (controle positivo);
- MI: BA 0,575, p=0,251 (não significativo);
- p95 do shuffle = 0,650; com 40 trials, resultados ~0,65 podem ser acaso.


---

# Validação alternativa: CNN-Transformer leve na CPU

```powershell
python -W ignore tools\validacao_me_transformer.py --epochs 60
```

Saídas em `_validacao_transformer/relatorio.md`.

Resultado com 40 trials left/right:

- ME BA = 0,425;
- MI BA = 0,550;
- LDA bandpower: ME 0,675 / MI 0,575.

O transformer tem 13.410 parâmetros e overfitou no ME; com 40 trials
independentes o LDA foi mais robusto.


---

# Baselines 3.1 e 3.2 na CPU

```powershell
python -W ignore tools\baselines_csp_lda_eegnet.py
```

- CSP+LDA: `CSP(n_components=2, reg="ledoit_wolf", log=True)`; com 3 canais
  `n_components=4` não é válido.
- EEGNet compacto: F1=8, D=2, F2=16, K=64, dropout 0.5, early stopping por
  validação agrupada por trial.
- Mesmo protocolo: 2 s, 5 janelas por trial, 5-fold agrupado.

Saídas: `_baselines_csp_eegnet/relatorio.md`.


---

# Permutação dos baselines

```powershell
python -W ignore tools\permutacao_baselines.py `
  --perm-csp 500 --perm-eegnet 20
```

- CSP 500 permutações: ME p=0,108; MI p=0,064;
- EEGNet 20 permutações: p=0,048 nas duas fases, mas nulo instável
  (média ~0,405) e p mínimo 1/21; não interpretar como evidência forte.
- Saídas: `_permutacao_baselines/relatorio.md` e `resultados.json`.
