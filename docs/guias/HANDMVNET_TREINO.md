# HandMvNet — captura, dataset e treino (rig de 3 vistas)

## Status

- Ainda **não há checkpoint oficial** publicado.
- O pipeline abaixo treina a arquitetura oficial **do zero** com o nosso dataset.
- Quando os pesos forem liberados, o mesmo trainer aceita `--checkpoint` para
  fine-tune (e `--freeze-backbone`, se desejado).
- Licença: confirmar que cobre **código e pesos** antes de redistribuir.

## 1. Calibrar as câmeras

Depois de mover qualquer câmera, recalibrar:

```powershell
$env:PYTHONPATH = "$PWD\src\mestradopy\core"
.\.venv\Scripts\python.exe src\mestradopy\core\stereo_calibration.py --aux-index 1
```

- 15–20 poses boas do tabuleiro por par;
- ~5–10 min;
- os npz vão para `configs/calibracao/`;
- não misturar sessões gravadas com calibrações diferentes sem guardar a
  calibração de cada sessão.

## 2. Captura em disco livre (ex.: F:)

```powershell
.\scripts\coletar_handmvnet.ps1 -Saida "F:\handmvnet" -Cada 5
```

Ou sem preview:

```powershell
.\scripts\coletar_handmvnet.ps1 -Saida "F:\handmvnet" -Cada 10
```

Cada sessão cria:

```text
F:\handmvnet\sessao_YYYYMMDD_HHMMSS\
  frames\000000_0.jpg   # Kinect
  frames\000000_1.jpg   # aux0
  frames\000000_2.jpg   # aux1
  labels.npz
  meta.json
```

O script:

- grava JPEG direto no destino, sem passar pela raiz do projeto;
- verifica espaço livre e aborta abaixo de `--min-free-gb` (2 GB);
- salva 1 amostra a cada `--cada` frames (30 fps → `--cada 10` = 3 amostras/s);
- usa a triangulação 3-view do projeto para gerar o pseudo-3D;
- só salva quando há pelo menos 8 joints triangulados.

### Quanto ocupa

Ordem de grandeza com JPEG qualidade 85:

| Resolução | Por amostra (3 vistas) | 10 min a 3 amostras/s |
|---|---:|---:|
| Kinect 1080p + aux 720p | ~0,3–0,45 MB | ~0,5–0,8 GB |
| Kinect 720p + aux 720p | ~0,15–0,25 MB | ~0,3–0,45 GB |

`--cada 5` (6 amostras/s) dobra esses valores. Para treino, **não use 30 fps**:
frames consecutivos são altamente correlacionados. 3–6 amostras/s já é
suficiente.

## 3. Smoke test sem hardware

O trainer gera um dataset sintético e roda 1 época:

```powershell
.\scripts\treinar_handmvnet.ps1 -Dados "" -Epocas 1 -Device xpu
```

Melhor use diretamente:

```powershell
$env:PYTHONPATH = "$PWD\src\mestradopy;$PWD\src\mestradopy\core;$PWD\src\mestradopy\tools;$PWD\src\mestradopy\vision"
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_train.py --smoke --device xpu
```

## 4. Treino do zero

```powershell
$env:PYTHONPATH = "$PWD\src\mestradopy;$PWD\src\mestradopy\core;$PWD\src\mestradopy\tools;$PWD\src\mestradopy\vision"
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_train.py `
  --data "F:\handmvnet\sessao_01" `
  --data-val "F:\handmvnet\sessao_02" `
  --saida "outputs\visao\handmvnet_treino" `
  --epochs 40 --batch-size 2 --device xpu
```

Saída:

```text
outputs/visao/handmvnet_treino/
  handmvnet_checkpoint.pt
  train_history.json
```

## 5. Fine-tune com checkpoint

Quando os autores liberarem os pesos:

```powershell
.\.venv\Scripts\python.exe src\mestradopy\tools\handmvnet_train.py `
  --data "F:\handmvnet\sessao_01" `
  --data-val "F:\handmvnet\sessao_02" `
  --checkpoint "external\HandMvNet\weights\handmvnet.ckpt" `
  --freeze-backbone --epochs 20 --device xpu
```

O checkpoint pode ter sido treinado com 4/5/8 vistas; o trainer usa
`adapt_num_views=True` para reconstruir com 3 vistas, mas isso é experimental.

## 6. Split correto

- separar **sessões inteiras** para treino/validação/teste;
- nunca separar frames consecutivos aleatoriamente;
- se a geometria das câmeras mudou, tratar como **novo domínio**:
  recalibrar, gravar novas sessões e fine-tunar/validar de novo.

## 7. Quando mover as câmeras

- Se mudou posição/orientação: **recalibrar** e não reutilizar
  `labels.npz` antigos com a nova calibração;
- se mudou foco/zoom: recalibrar intrínsecas também;
- se só mudou a cena/iluminação: a calibração pode continuar, mas o domínio
  visual muda; fine-tune com dados novos;
- mantenha a geometria fixa durante uma coleta/training; trate cada mudança
  como sessão/domínio separado.

## 8. Limitações atuais

- Os labels 3D são pseudo-labels de triangulação; erros de MediaPipe e
  calibração entram no treino.
- O modelo oficial usa MANO/vertices em alguns caminhos; nosso trainer usa
  somente joints e desativa `get_vertices`.
- A qualidade do checkpoint depende do volume e da diversidade dos frames;
  o smoke test valida o encanamento, não a acurácia.
