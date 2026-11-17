# Próximo piloto — ME 4 s + MI 4 s

## Desenho

- **2 sessões** em dias diferentes.
- Cada sessão: **30 trials/classe** (15 esquerda + 15 direita por bloco × 2 blocos).
- Total: **60 trials/classe = 120 trials L/R**.
- Apenas mão esquerda e direita no trial.
- ME 4 s como controle positivo; MI 4 s como alvo.
- Início: **60 s olhos abertos + 60 s olhos fechados** para QC/alpha.
- O baseline de olhos **não entra no treino de MI**.

## Sequência por trial

```text
cue 1 s
ME 4 s
transição 1 s
MI 4 s
ITI 1–2 s
```

Duração aproximada: 12–13 s por trial.

## Rodar no PowerShell

Feche o gRecorder/GtecSuite antes.

Sessão 1:

```powershell
cd C:\mestradopy
.\.venv\Scripts\Activate.ps1
.\scripts\run_piloto_me_mi.ps1 -Sessao 01 -Sujeito S01
```

Sessão 2 (outro dia):

```powershell
.\scripts\run_piloto_me_mi.ps1 -Sessao 02 -Sujeito S01
```

Ou diretamente:

```powershell
python -W ignore src\mestradopy\tools\validacao_eeg_3classes.py `
  --source gnautilus `
  --blocos 2 --trials-por-classe 15 `
  --classes esquerda,direita `
  --me-sec 4 --mi-sec 4 --transicao-sec 1 --cue-sec 1 `
  --repouso-sec 2 --iti-min 1 --iti-max 2 `
  --baseline-olhos-sec 60 `
  --sujeito S01 --sessao 01 `
  --pasta-saida outputs/eeg/piloto_me_mi_s01
```

Para a sessão 2, troque `--sessao 02` e `outputs/eeg/piloto_me_mi_s02`.

## Marcadores

- 769/770: cue esquerda/direita;
- 778/779: início/fim ME;
- 780/781: início/fim MI;
- 901/902: olhos abertos;
- 903/904: olhos fechados;
- 790/791: blocos.

## Depois da aquisição

1. Verificar alpha dos 2 min iniciais.
2. Avaliar ME left/right como controle positivo.
3. Reservar a **sessão 2 inteira** como teste final do MI; não usá-la para ajuste.
4. Não tratar as 5 janelas de 2 s como amostras independentes; o N é o número de trials.

## Critério

- ME deve ficar acima do acaso para indicar que o pipeline/sinal está bom.
- MI só será considerado validado se ficar acima do acaso no teste final, com IC/bootstrap e, idealmente, em mais de uma sessão.
