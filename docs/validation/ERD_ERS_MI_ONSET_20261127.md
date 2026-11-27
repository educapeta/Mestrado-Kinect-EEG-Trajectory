# ERD/ERS em torno do início da MI — janela −1 s a +3 s

**Data:** 27/11/2026  
**Dados:** piloto s01 + s02, 120 trials (60 esquerda, 60 direita)  
**Canais:** C3, C4, Cz

## Protocolo e transição

- Cue: 769/770.
- ME: 778 → 779 (4 s).
- Transição ME→MI: **1,0 s** (`transicao_sec=1.0`; medido nos eventos:
  `mi_start − me_end ≈ 1,015 s`).
- MI: 780 → 781 (4 s).

A janela usada agora é:

```text
[início da MI − 1 s, início da MI + 3 s]
```

Ou seja: **os 1 s finais da transição ME→MI + os primeiros 3 s da MI**.

Baseline: bloco de **olhos abertos** de 60 s (901→903).  
Pré-processamento: 4–40 Hz, fs=128.

## Tempo-frequência

Figura: `outputs/eeg/erd_ers_mi_onset/tempo_frequencia_mi_onset_por_canal.png`.

- linhas: C3, C4, Cz;
- colunas: esquerda, direita;
- eixo X: −1 s a +3 s em relação ao início da MI;
- eixo Y: 4–40 Hz;
- azul = ERD; vermelho = ERS.

Padrão:

- **pré-MI (−1 a 0 s, transição):** ERS em mu e beta baixo;
- **MI 0–1 s:** ERS ainda presente, principalmente em mu;
- **MI 1–2 s:** ERD em beta baixo em todos os canais e classes;
- **MI 2–3 s:** ERD persiste em beta baixo; mu volta a ERS.

## Médias por janela

`*` = p < 0,05 contra olhos abertos.

### mu

| janela | classe | C3 | C4 | Cz |
|---|---|---|---|---|
| pré-MI | esquerda | 24,9%* | 34,5%* | 31,3%* |
|  | direita | 39,3%* | 42,1%* | 35,5%* |
| MI 0–1 s | esquerda | 9,7% | 28,9%* | 27,4%* |
|  | direita | 32,7%* | 23,0%* | 32,2%* |
| MI 1–2 s | esquerda | 9,7% | −7,0% | 8,5% |
|  | direita | 2,3% | −1,7% | 14,5% |
| MI 2–3 s | esquerda | 37,9%* | 13,8% | 38,1%* |
|  | direita | 31,0%* | 12,1% | 35,9%* |

### beta baixo (13–20 Hz)

| janela | classe | C3 | C4 | Cz |
|---|---|---|---|---|
| pré-MI | esquerda | 21,6%* | 28,2%* | 14,3%* |
|  | direita | 53,4%* | 27,8%* | 19,3%* |
| MI 0–1 s | esquerda | 4,4% | −6,1% | −5,7% |
|  | direita | 3,5% | 5,4% | −1,6% |
| **MI 1–2 s** | esquerda | **−21,8%*** | **−21,1%*** | **−22,8%*** |
|  | direita | **−19,1%*** | **−24,8%*** | **−19,1%*** |
| MI 2–3 s | esquerda | **−18,4%*** | −7,1% | **−17,2%*** |
|  | direita | **−11,8%*** | **−23,1%*** | **−14,0%*** |

### beta alto (20–30 Hz)

| janela | classe | C3 | C4 | Cz |
|---|---|---|---|---|
| MI 1–2 s | esquerda | **−25,9%*** | −0,3% | **−26,4%*** |
|  | direita | **−18,8%*** | **−17,6%*** | **−20,8%*** |
| MI 2–3 s | esquerda | **−23,0%*** | 11,9% | **−20,6%*** |
|  | direita | −9,4% | **−20,1%*** | **−21,8%*** |

## Leitura

1. A **transição ME→MI** ainda mostra ERS em relação a olhos abertos.
2. O **ERD em beta baixo** aparece claramente **1 s após o início da MI** e
   persiste; é o achado mais robusto.
3. Em mu, o padrão é bifásico: ERS inicial, queda transitória e ERS tardio.
4. A separação esquerda/direita ainda não é uma lateralização clássica limpa;
   a diferença principal está no nível/amplitude do ERS inicial e em algumas
   bandas.
5. O intervalo de análise deve ser descrito como **−1 a +3 s em relação ao
   início da MI**, não ao cue inicial.

## Arquivos

- script: `src/mestradopy/tools/analise_erd_ers_mi_onset_piloto.py`
- resultados: `outputs/eeg/erd_ers_mi_onset/resultados.json`
- relatório: `outputs/eeg/erd_ers_mi_onset/relatorio.md`
- TF: `outputs/eeg/erd_ers_mi_onset/tempo_frequencia_mi_onset_por_canal.png`
- curvas: `erd_ers_mi_onset_mu.png`, `erd_ers_mi_onset_beta_low.png`,
  `erd_ers_mi_onset_beta_high.png`

## Comando

```powershell
cd C:\mestradopy
$env:PYTHONPATH="C:\mestradopy;C:\mestradopy\src"
.\.venv\Scripts\python.exe src\mestradopy\tools\analise_erd_ers_mi_onset_piloto.py
```
