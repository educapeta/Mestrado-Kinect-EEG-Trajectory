# Migração para fora do OneDrive (e organização da "garagem")

Este documento responde ao item **#10 da revisão crítica** ("a raiz do projeto é
uma garagem") e ao problema de manter `.venv` + dezenas de GB de dados dentro de
uma pasta sincronizada.

## 1. Por que tirar do OneDrive

| Problema | Consequência |
|---|---|
| `.venv` tem **~40 000 arquivos** | o cliente do OneDrive passa a monitorar/sincronizar dezenas de milhares de arquivos; a máquina fica lenta e o upload nunca termina |
| Dados de sessão: **~440 MB de EEG + ~18 MB/h de movimento + clipes** por sessão (25 sessões ≈ 12–15 GB) | sync de binários grandes gera conflitos (`arquivo-cópia-conflitante.csv`) e corrompe o meio de uma sessão |
| Repositório **git** dentro do OneDrive | o OneDrive mexe nos arquivos de `.git/` enquanto o git trabalha → índices corrompidos, `git status` lento |
| Um arquivo bloqueado pelo sync | a gravação falha no meio de uma trial |

**Regra de ouro:** OneDrive só para documentos (`.md`, `.txt`, PDFs, figuras).
Código → pasta local + GitHub. Dados → disco local ou HD externo.

## 2. Estrutura-alvo (fora do OneDrive)

```
C:\Mestrado\projeto\        <- CÓDIGO (repo git; ~15 MB com modelos .task/.pt)
    *.py  test_*.py  imu.py  README.md  requirements.txt  HISTORICO_ITERACOES.txt
    stereo_calibration_auxN.npz / camera_alignment.npz / arm_model.json ...
    gravacoes -> (junction para o disco de dados)

D:\Mestrado_Dados\          <- DADOS (nunca no OneDrive)
    gravacoes\              <- sessoes (CSV + JSON + clipes) — o treino le daqui
    data\                   <- dados antigos (BCI IV, resultados de treino)
    results\                <- saidas de teste
    legacy\                 <- zips do codigo legado (subir no Drive manualmente)
```

Se não houver `D:`, use `C:\Mestrado_Dados\`. O código **não assume** nada: a
pasta das sessões é escolhida na linha de comando (`--pasta-sessoes`).

## 3. Passo a passo (PowerShell)

```powershell
# 0) Feche o programa de aquisição e o VS Code. Pause o OneDrive (opcional).

# 1) CODIGO: copia SEM o ambiente virtual e SEM os dados
robocopy "C:\Users\elgut\OneDrive\Desktop\Mestrado_Projeto\Mestradopy" `
         "C:\Mestrado\projeto" /E /COPY:DAT /R:1 /W:1 `
         /XD .venv __pycache__ .git gpype BCICIV_2a_gdf `
         /XF *.csv *.avi
# /E inclui subpastas; depois copie o historico do git, se quiser:
robocopy "C:\Users\elgut\OneDrive\Desktop\Mestrado_Projeto\Mestradopy\.git" `
         "C:\Mestrado\projeto\.git" /E /R:1 /W:1

# 2) DADOS: mova para o disco de dados
robocopy "C:\Users\elgut\OneDrive\Desktop\Mestrado_Projeto\Mestradopy\data" `
         "D:\Mestrado_Dados\data" /E /MOVE
robocopy "C:\Users\elgut\OneDrive\Desktop\Mestrado_Projeto\Mestradopy\results" `
         "D:\Mestrado_Dados\results" /E /MOVE
robocopy "C:\Users\elgut\OneDrive\Desktop\Mestrado_Projeto\Mestradopy\legacy" `
         "D:\Mestrado_Dados\legacy" /E /MOVE

# 3) AMBIENTE: recriar o .venv FORA do OneDrive (o venv guarda caminhos
#    absolutos nos executaveis; recriar e' mais seguro que mover)
cd C:\Mestrado\projeto
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
#    gtec-gds e pykinect2 costumam ser instalados a parte (ver README)

# 4) Liga a pasta de dados ao projeto (opcional, mantem o caminho padrao)
New-Item -ItemType Directory -Force -Path "D:\Mestrado_Dados\gravacoes" | Out-Null
New-Item -ItemType Junction -Path "C:\Mestrado\projeto\gravacoes" `
                             -Target "D:\Mestrado_Dados\gravacoes"

# 5) Verificacao (tudo tem que passar)
.\.venv\Scripts\python.exe test_paradigm_protocol.py
.\.venv\Scripts\python.exe test_fusion_zeroing.py
.\.venv\Scripts\python.exe _smoke_eeg.py
git status

# 6) Só depois de tudo verde: apague a pasta antiga do OneDrive
```

## 4. Uso no dia a dia

```powershell
# Sessao indo para o disco de dados (sem junction):
python eeg_motor_paradigm.py --participante P03 --sessao S2 `
       --pasta-sessoes "D:\Mestrado_Dados\gravacoes"

# Treino lendo de la:
python sand_traj_treino.py --data "D:\Mestrado_Dados\gravacoes"
```

## 5. O que NÃO vai para o git (`.gitignore`)

`.venv/`, `gravacoes/`, `data/`, `results/`, `legacy/`, `gpype/` (clone de
referência), `*.csv`, `*.avi`, `*.pt`, `__pycache__/`.

**Vão para o GitHub:** código, testes, README, histórico, `requirements.txt`,
calibrações (`.npz`, `.json` pequenos) e os modelos `.task` (13 MB, para o clone
funcionar offline).

**Vão para o Drive/HD (manual):** `legacy/*.zip` e, se quiser, os dados das
sessões (grandes) — o GitHub não é lugar para eles.

## 7. E se eu nao tiver um segundo disco? (pendrive de 2 TB)

Um pendrive **serve como arquivo**, mas nao como destino da gravacao ao vivo:

1. **Nao grave a sessao direto nele.** O CSV de EEG e' escrito a 500 Hz com
   flush constante; se o pendrive engasgar (ou o USB cair), a trial e' perdida.
   Grave no SSD local (padrao `gravacoes/`) e **copie no fim** (ou entre
   participantes):

   ```powershell
   robocopy "C:\Mestrado\projeto\gravacoes" "M:\Mestrado_Dados\gravacoes" `
            /E /XO /R:1 /W:1     # /XO = nao recopia o que ja' esta' la'
   ```

2. **Letra fixa.** O Windows troca a letra do pendrive ao reconectar. Fixe uma
   letra (Gerenciamento de Disco -> Alterar letra de unidade -> `M:`) e use
   sempre ela; sem isso um `--pasta-sessoes` ou junction aponta para o vazio.
3. **Formato:** use **exFAT** (ou NTFS). FAT32 limita arquivo a 4 GB.
4. **Junction para o pendrive:** funciona, mas se o pendrive estiver fora a
   pasta `gravacoes` fica quebrada e a gravacao falha. Prefira `--pasta-sessoes`
   explicito e copie no fim.
5. **Velocidade:** um pendrive USB 3.x grava ~30–100 MB/s e le ~100–200 MB/s —
   suficiente para copiar ~600 MB por sessao em poucos segundos. Para gravar
   *direto* nele, um SSD externo USB-C e' o minimo aceitavel.

**Tamanho real para planejar (medido/tabelado):**

| Item | Tamanho |
|---|---|
| EEG por sessao (50 min, 34 colunas) | ~440 MB |
| Movimento por sessao (30 Hz, 44 colunas) | ~18 MB/h |
| Metadados (JSON + questionario) | < 100 kB |
| Clipes de priming (250 trials) | ~60–100 MB |
| **Total por sessao** | **~0,5 GB** |
| **25 sessoes** | **~13 GB** |
| Dataset BCI IV 2a (se usar) | 707 MB |
| Dados antigos (`data/`) | 78 MB |
| Codigo + modelos + calibracoes | ~27 MB |

Ou seja: 2 TB e' folgadissimo (sobra ~99%). Se quiser guardar os **videos brutos
do Kinect** (ainda nao gravamos), ai sim o espaco importa: 1080p a 30 fps da'
~2–4 GB por hora de sessao.

## 8. Checklist rápido de "higiene" (repetir a cada sessão nova)

- [ ] `git status` limpo antes de começar a gravar
- [ ] `python test_paradigm_protocol.py` verde depois de qualquer mudança de protocolo
- [ ] nada de `*.csv`/`*.avi`/`.pt` novo versionado por engano
- [ ] ao fim do dia: `git add -A && git commit -m "..."` e `git push`
- [ ] dados copiados para o HD (o GitHub não é backup de dados)
