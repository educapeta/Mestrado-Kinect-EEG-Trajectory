# ERD/ERS cue-locked — janela de −1 s a +3 s

**Data:** 27/11/2026  
**Dados:** piloto s01 + s02, 120 trials (60 esquerda, 60 direita)  
**Canais:** C3, C4, Cz

## Protocolo

- **Janela:** do cue (769/770) −1 s até +3 s; 4 s no total.
- **Baseline:** bloco de olhos abertos (901–903), 60 s no início da sessão.
- **Pré-processamento:** 4–40 Hz, fs=128 Hz.
- **ERD%:** `100*(P_janela/P_olhos_abertos - 1)`.
- **Bandas:** mu 8–13, beta baixo 13–20, beta alto 20–30 Hz.

**Atenção:** no protocolo atual, o cue é seguido pela **ME**; a MI começa
depois da ME (marcador 780). Portanto, esta janela cobre **pré-cue + ME**,
não a MI.

## Tempo-frequência

Figura: `outputs/eeg/erd_ers_cue/tempo_frequencia_cue_por_canal.png`.

Padrão visual:

- **mão esquerda:** ERD (azul) amplo após o cue, principalmente em
  10–40 Hz, em C3, C4 e Cz;
- **mão direita:** ERS (vermelho) forte em beta (20–40 Hz), especialmente
  depois de ~0,5 s;
- a diferença entre as classes é **muito maior** que na janela de MI;
- o padrão é broadband e não lateralizado de forma limpa — compatível com
  diferenças de movimento/EMG, não com MI.

## Médias por janela

`*` = p < 0,05 contra olhos abertos.

### mu

| janela | classe | C3 | C4 | Cz |
|---|---|---|---|---|
| pré-cue | esquerda | 25,4%* | 23,2%* | 32,9%* |
|  | direita | 50,9%* | 50,6%* | 59,9%* |
| 0–1 s | esquerda | 13,2% | 14,7%* | 19,4%* |
|  | direita | 19,7%* | 28,0%* | 25,8%* |
| 1–2 s | esquerda | −0,8% | −9,6% | 5,0% |
|  | direita | 11,2% | 0,8% | 16,7%* |
| 2–3 s | esquerda | −13,2% | −17,9%* | −2,0% |
|  | direita | 10,3% | −3,1% | 16,3%* |

### beta baixo

| janela | classe | C3 | C4 | Cz |
|---|---|---|---|---|
| 1–2 s | esquerda | **−25,4%*** | **−32,5%*** | **−19,9%*** |
|  | direita | **+28,9%*** | **+17,3%*** | **+27,6%*** |
| 2–3 s | esquerda | **−24,0%*** | **−28,0%*** | **−16,2%*** |
|  | direita | **+30,0%*** | **+22,5%*** | **+31,3%*** |

### beta alto

| janela | classe | C3 | C4 | Cz |
|---|---|---|---|---|
| 1–2 s | esquerda | −44,5%* | −40,2%* | −33,9%* |
|  | direita | +38,2%* | +43,8%* | +44,0%* |
| 2–3 s | esquerda | −34,6%* | −35,4%* | −29,6%* |
|  | direita | +61,7%* | +66,2%* | +66,2%* |

## Leitura

1. Existe uma diferença enorme entre esquerda e direita na janela pós-cue,
   muito maior que a diferença encontrada na MI.
2. O padrão é **broadband, bilateral e com sinais opostos**: esquerda ERD,
   direita ERS.
3. Isso é mais compatível com **diferença de execução motora/EMG** do que com
   imagética motora.
4. A análise da MI (janela 780–781) é a que deve ser usada para ERD/ERS de
   imagética; esta janela serve para mostrar que a ME domina o período
   imediatamente pós-cue.
5. Reforça a necessidade de EMG no protocolo: a ME não pode ser interpretada
   como MI.

## Arquivos

- script: `src/mestradopy/tools/analise_erd_ers_cue_piloto.py`
- resultados: `outputs/eeg/erd_ers_cue/resultados.json`
- relatório: `outputs/eeg/erd_ers_cue/relatorio.md`
- mapa TF: `outputs/eeg/erd_ers_cue/tempo_frequencia_cue_por_canal.png`
- curvas: `outputs/eeg/erd_ers_cue/erd_ers_cue_mu.png`,
  `erd_ers_cue_beta_low.png`, `erd_ers_cue_beta_high.png`

## Comando

```powershell
cd C:\mestradopy
$env:PYTHONPATH="C:\mestradopy;C:\mestradopy\src"
.\.venv\Scripts\python.exe src\mestradopy\tools\analise_erd_ers_cue_piloto.py
```
