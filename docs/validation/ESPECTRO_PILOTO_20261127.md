# Análise espectral do piloto ME/MI — validação do FBCSP

**Data:** 27/11/2026  
**Dados:** piloto `s01` + `s02`, 120 trials (60 esquerda, 60 direita)  
**Canais:** C3, C4, Cz  
**Janelas:** 2 s, fs=128 Hz, filtro 4–40 Hz, passo 0,5 s — **as mesmas
janelas do FBCSP**.

## Método

- PSD por Welch em cada janela de 2 s (`nperseg=256`, sobreposição 128);
- média por trial e depois por classe/fase;
- potência por banda: 4–8, 8–12, 12–16, 16–20, 20–24, 24–28, 28–32,
  32–36, 36–40 Hz;
- efeito de classe: direita − esquerda, Cohen's d;
- significância: permutação de rótulos no nível do trial (2000 permutações);
- lateralização: contralateral − ipsilateral (direita→C3, esquerda→C4).

## MI — resultado

Efeitos de classe relevantes (p < 0,05):

| banda | canal | d | p | log2(direita/esquerda) |
|---|---|---:|---:|---:|
| 12–16 | C3 | 0,504 | 0,0055 | −0,295 |
| 16–20 | C3 | −0,492 | 0,0060 | 0,253 |
| 16–20 | C4 | 0,524 | 0,0045 | −0,374 |
| 20–24 | C3 | −0,423 | 0,0220 | 0,299 |
| **20–24** | **C4** | **0,835** | **0,0005** | **−0,562** |
| 24–28 | Cz | −0,399 | 0,0310 | 0,216 |
| 32–36 | C4 | −0,421 | 0,0140 | 0,365 |
| 32–36 | Cz | −0,390 | 0,0275 | 0,317 |

**8–12 Hz:** d ≈ 0,0 e p > 0,6 em todos os canais.  
**4–8 Hz:** d entre −0,21 e +0,26, p > 0,15.

O efeito está concentrado em **12–24 Hz**, com destaque em C4 20–24 Hz.

### Comparação com o FBCSP

Bandas do FBCSP no MI:

| banda | FBCSP BA | efeito espectral médio \|d\| |
|---|---:|---:|
| 4–8 | 0,658 | 0,168 |
| 8–12 | 0,508 | 0,048 |
| 12–16 | 0,767 | 0,332 |
| 16–20 | 0,800 | 0,369 |
| 20–24 | 0,808 | 0,425 |
| 24–28 | 0,525 | 0,189 |
| 28–32 | 0,667 | 0,241 |
| 32–36 | 0,592 | 0,376 |
| 36–40 | 0,708 | 0,234 |

Correlação entre \|d\| espectral e BA do FBCSP por banda: **r = 0,735**.

Isso **valida o FBCSP**: as bandas com maior efeito espectral são justamente
12–16, 16–20 e 20–24 Hz, que são as bandas com maior BA no FBCSP. A banda
8–12 não mostra efeito de classe e o FBCSP também fica no acaso nela.

## ME — resultado

O ME mostra um padrão **broadband**:

- direita tem potência maior que esquerda em **todos os canais**;
- efeito cresce com a frequência:
  - 12–16: d ≈ −0,7 a −0,96;
  - 20–24: d ≈ −0,9 a −1,07;
  - 24–40: d ≈ −1,1 a −1,35;
- p < 0,001 em quase todas as bandas/canais;
- o padrão é global (C3, C4 e Cz), não lateralizado.

Isso é compatível com **diferença de amplitude/movimento/EMG**, não com um
padrão específico de MI. O ME do piloto não pode ser usado como evidência de
MI pura.

## Implicações para o FBCSP

1. No **MI**, o FBCSP está capturando o efeito certo (12–24 Hz), com
   lateralização parcial e correlação alta entre bandas espectrais e BA.
2. No **ME**, o FBCSP pode estar capturando diferença global de amplitude
   (movimento/EMG), o que explica BA alta em ME e reforça a necessidade de
   EMG no protocolo.
3. O resultado do MI não é tão limpo quanto o BA sugere: há efeitos em C3 e
   C4, mas o padrão não é uma lateralização clássica perfeita.
4. A validação espectral **não prova causalidade**; é consistência
   espaço-frequencial.

## Arquivos

- script: `src/mestradopy/tools/analise_espectral_piloto.py`
- resultados: `outputs/eeg/espectro_piloto/resultados.json`
- relatório detalhado: `outputs/eeg/espectro_piloto/relatorio.md`
- figura ME: `outputs/eeg/espectro_piloto/espectro_ME.png`
- figura MI: `outputs/eeg/espectro_piloto/espectro_MI.png`
- figura de efeitos por banda: `outputs/eeg/espectro_piloto/bandas_efeito.png`

## Comando

```powershell
cd C:\mestradopy
$env:PYTHONPATH="C:\mestradopy;C:\mestradopy\src"

.\.venv\Scripts\python.exe src\mestradopy\tools\analise_espectral_piloto.py
```
