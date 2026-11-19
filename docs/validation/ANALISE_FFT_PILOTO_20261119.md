# Análise FFT/PSD e tempo-frequência (Morlet) — épocas de MI de 4 s

**Gerado por:** `src/mestradopy/tools/fft_epocas_piloto.py` em 19/11/2026 17:07
**Dados:** piloto ME/MI do `validacao_eeg_3classes.py` (17/11/2026),
`outputs/eeg/piloto_me_mi_s01` e `piloto_me_mi_s02` (somente leitura; nada foi
alterado nas pastas de origem).
**Saída:** `outputs/eeg/analises_fft_piloto/` (figuras, tabelas, JSON, este
relatório).

## 1. Resumo

- Épocas de **imagens motora (MI) de 4 s** extraídas do marcador `mi_start=780`
  (2000 amostras a 500 Hz) até 780+4 s; **não** usam ME como MI e **não**
  cruzam a transição ME→MI (o 780 vem depois do fim do ME, 779).
- Rótulo por trial = último cue antes do 780 (`769`=esquerda, `770`=direita).
- **n = 60 trials L + 60 trials R** (piloto_me_mi_s01, piloto_me_mi_s02).
- Canais: C3/C4/Cz (canais físicos 15/16/17 = colunas Ch15/Ch16/Ch17 do CSV;
  índices 0-based 14/15/16 do `CH_OWN_IDX` do pacote `eeg.mi_eegnet`).
- Figuras: PSD média ± SEM, diferença/razão R−L (mu/beta destacados),
  espectrogramas médios Morlet L e R (mesma escala), diferença no
  tempo-frequência, heatmap de PSD por trial e barras de potência por banda.
- Leitura exploratória: com 3 canais, 1 participante e 2 sessões do mesmo dia,
  qualquer diferença entre classes é **descritiva**, sem correção para múltiplas
  comparações.

## 2. Sessões usadas

| sessao | CSV | fs (Hz) | trials validos | L/R | eventos | tamanho CSV |
|---|---|---|---|---|---|---|
| piloto_me_mi_s01 | eeg_mi3c_S01_s01_20261117_155319_20261117_155331_696.csv | 500 | 60 | 30/30 | eventos_S01_s01_20261117_160802.json | 272.0 MB |
| piloto_me_mi_s02 | eeg_mi3c_S01_s02_20261117_161204_20261117_161213_789.csv | 500 | 60 | 30/30 | eventos_S01_s02_20261117_162645.json | 272.6 MB |

Sanidade do epoching:

| sessao | marcadores 780 | marcadores 781 | dur MI (amostras) | gap 779→780 (amostras) | baseline valida | avisos |
|---|---|---|---|---|---|---|
| piloto_me_mi_s01 | 60 | 60 | 1993–2025 | 506 | 60/60 | — |
| piloto_me_mi_s02 | 60 | 60 | 1985–2025 | 506 | 60/60 | — |

Notas: `dur MI` é `mi_end - mi_start` no CSV (mediana ~1999 amostras); `gap 779→780` mede a transição (~1 s
esperado). O epoch usa exatamente 4 s a partir do 780 (o `mi_end` do marcador
varia alguns samples por jitter do muxer, por isso a duração é validada com
tolerância e o epoch é fixo, sem atravessar a transição).

## 3. Métodos

1. **Loader**: `eeg.mi_eegnet.data._load_own_csv` (mesmo leitor usado pelas
   análises CSP+LDA/EEGNet do projeto) — CSV `eeg_mi3c*.csv` + `eventos*.json`.
2. **Canais**: colunas Ch15/Ch16/Ch17 (índices 0-based 14/15/16) = C3/C4/Cz.
   Conferência: desvios-padrão da sessão s01 = 15.51, 15.23, 14.53 µV,
   batendo com o registrado no HISTÓRICO item 189 (Ch15=C3 15,51; Ch16=C4 15,23;
   Ch17=Cz 14,53).
3. **Pré-processamento**: filtro Butterworth bandpass 1–45 Hz
   (ordem 4, zero-phase `sosfiltfilt`) no sinal **contínuo** antes de cortar as
   épocas (evita transiente de filtro por época).
4. **PSD**: Welch por trial/canal, janela Hann de 2 s (`nperseg=1000`,
   `noverlap=500`, detrend constante) → df = 0.50 Hz;
   intervalo útil 1–45 Hz.
5. **Bandas**: mu 8–13 Hz, beta 13–30 Hz (pedido do Lead) + theta 4–8 e total
   1–45 Hz como referência; potência = integral da PSD na banda.
6. **Tempo-frequência**: Morlet via `mne.time_frequency.tfr_array_morlet`,
   `n_cycles=max(freq/2, 3)`, 2–45 Hz (passo 1 Hz), potência média por
   classe em dB (10·log10); escala de cor **compartilhada L×R**.
7. **Estatística**: Mann-Whitney U bilateral (p **não corrigido**, exploratório)
   e Cohen d para R−L da potência por banda/canal, por escopo (pooled e por
   sessão). Nenhuma conclusão inferencial: n pequeno e múltiplas comparações.
8. **Baseline**: 1 s de repouso **antes do cue** (não cruza ME→MI), usado só
   como coluna de referência em `potencia_bandas.csv`; as figuras são de
   potência absoluta.

## 4. Resultados (potência média por banda, pooled)

| banda | canal | L (µV²) mean ± SEM | R (µV²) mean ± SEM | razão R/L (dB) | p (não corrigido) | Cohen d |
|---|---|---|---|---|---|---|
| theta | C3 | 11.004 ± 0.409 | 12.541 ± 0.652 | 0.57 | 0.178 | 0.37 |
| theta | C4 | 10.674 ± 0.499 | 9.844 ± 0.533 | -0.35 | 0.173 | -0.21 |
| theta | Cz | 13.758 ± 0.549 | 13.890 ± 0.613 | 0.04 | 0.918 | 0.03 |
| mu | C3 | 25.288 ± 1.596 | 26.070 ± 1.872 | 0.13 | 0.783 | 0.06 |
| mu | C4 | 15.975 ± 0.991 | 15.852 ± 0.907 | -0.03 | 0.990 | -0.02 |
| mu | Cz | 21.461 ± 1.257 | 22.411 ± 1.415 | 0.19 | 0.898 | 0.09 |
| beta | C3 | 8.998 ± 0.288 | 9.459 ± 0.371 | 0.22 | 0.562 | 0.18 |
| beta | C4 | 8.001 ± 0.293 | 7.150 ± 0.289 | -0.49 | 0.022 | -0.38 |
| beta | Cz | 8.574 ± 0.262 | 8.884 ± 0.300 | 0.15 | 0.624 | 0.14 |
| total_1_45 | C3 | 75.933 ± 3.017 | 90.889 ± 4.643 | 0.78 | 0.018 | 0.49 |
| total_1_45 | C4 | 76.839 ± 3.867 | 66.231 ± 3.434 | -0.65 | 0.050 | -0.37 |
| total_1_45 | Cz | 83.029 ± 3.419 | 87.440 ± 3.981 | 0.22 | 0.430 | 0.15 |

Picos espectrais (frequência do máximo da PSD média):

| canal | pico mu L (Hz) | pico mu R (Hz) | pico beta L (Hz) | pico beta R (Hz) |
|---|---|---|---|---|
| C3 | 10.5 | 10.5 | 13.0 | 13.0 |
| C4 | 10.5 | 10.5 | 13.0 | 13.0 |
| Cz | 10.0 | 10.5 | 13.0 | 13.0 |

O pico de **mu** (~10,5 Hz) é consistente entre canais/classes. O pico de
**beta** cair no limite inferior (13 Hz) significa que o espectro cai
monotonicamente naquela faixa (1/f), ou seja, **não há** um pico beta destacado
neste piloto — o valor é apenas o máximo dentro da banda e não deve ser lido
como oscilação beta.

## 4b. Leitura descritiva (exploratória, sem inferência)

- **Mu (8–13 Hz)**: potências médias praticamente iguais entre classes (razão R/L: 0.13, -0.03, 0.19 dB por canal; menor p não corrigido = 0.783); o pico de mu fica em ~10.5 Hz em todos os canais/classes.
- **Beta (13–30 Hz)**: C4 tem potência menor na direita (razão -0.49 dB; p=0.022 não corrigido; d=-0.38). Não há pico beta destacado (espectro cai em 1/f — ver nota acima).
- **Banda total (1–45 Hz)**: C3 com R>L e C4 com R<L (razões 0.78 / -0.65 dB; p não corrigidos 0.018 / 0.050) — sinal lateralizado fraco, mas as SESSÕES individuais têm sinais opostos no mu (s1 R/L: -0.84, -0.63, -0.50 dB; s2: 0.91, 0.54, 0.81 dB), então o padrão NÃO é estável entre sessões e não deve ser promovido a achado.
- Nenhuma conclusão inferencial: p de Mann-Whitney sem correção para múltiplas comparações e sem cluster/permutação; n=120 épocas de um único participante.

Diferença entre sessões (potência mu, L/R):
- C3: potência mu L/R s1 = 25.07/20.68 µV²; s2 = 25.50/31.46 µV².
- C4: potência mu L/R s1 = 16.67/14.40 µV²; s2 = 15.28/17.30 µV².
- Cz: potência mu L/R s1 = 21.93/19.53 µV²; s2 = 20.99/25.29 µV².

Razão R/L em dB por canal e banda (calculada da potência integrada):
| canal | mu R/L (dB) | beta R/L (dB) |
|---|---|---|
| C3 | 0.13 | 0.22 |
| C4 | -0.03 | -0.49 |
| Cz | 0.19 | 0.15 |

## 5. Figuras geradas

| arquivo | tamanho |
|---|---|
| fig0_media_temporal_MI_4s.png | 477 kB |
| fig1_psd_media_sem_esquerda_direita.png | 202 kB |
| fig2_diferenca_razao_R_menos_L.png | 432 kB |
| fig3_tfr_morlet_media_L_R_mesma_escala.png | 883 kB |
| fig4_tfr_morlet_diferenca_R_menos_L.png | 1080 kB |
| fig5_psd_por_trial_heatmap.png | 224 kB |
| fig6_potencia_bandas_mu_beta.png | 58 kB |

Todas as figuras foram salvas em 200 dpi e verificadas (abrem via Pillow,
> 10000 bytes).

## 6. Tabelas e dados gerados

- `tabelas/psd_media.csv` — PSD média/SEM por classe/canal/frequência
  (pooled e por sessão).
- `tabelas/potencia_bandas.csv` — potência por banda/canal/classe, razão R/L,
  lateralização, p de Mann-Whitney **não corrigido**, Cohen d e baseline.
- `tabelas/picos_espectrais.csv` — frequência/PSD de pico em mu e beta.
- `tabelas/trials.csv` — amostras de cue/ME/MI/baseline e rótulo por trial.
- `tabelas/epocas_e_psd.npz` — épocas MI (4 s), baseline, PSD por
  trial e TFR médio L/R (para reprodutibilidade).
- `resumo_analise_fft.json` — parâmetros, números e lista de arquivos.

## 7. Limitações

- 3 canais apenas (C3/C4/Cz), sem topografia/espacial; nao ha' como testar padroes fora do strip motor.
- 1 participante (S01), 2 sessoes do MESMO dia (17/11/2026); a generalizacao e' nula e a variabilidade entre sessoes nao e' independente.
- 60 trials/classe no total (30/sessao), poucos para um achado confirmatorio; p-valores de Mann-Whitney NAO corrigidos para multiplas comparacoes (12 testes por banda/escopo) -> exploratorios.
- Welch com janela de 2 s (df=0.5 Hz) em epoca de 4 s: resolucao frequencial moderada; o pico de mu/beta e' arredondado a 0.5 Hz.
- Baseline = 1 s de repouso ANTES do cue (nao cruza ME->MI); nao e' a linha de base de olhos abertos/fechados, e parte dos trials pode nao ter baseline valida (registrada em trials.csv).
- Sem rejeicao de artefatos/ICA/regressao de EOG; epoca de 4 s inteira entra na media, inclusive transientes de movimento no inicio/fim.
- Morlet absoluto (dB rel. 1 uV^2), sem normalizacao por baseline: diferencas de potencia entre classes aparecem como estao; a escala de cor e' compartilhada L x R (comparacao direta).
- O mapeamento dos canais (Ch15=C3, Ch16=C4, Ch17=Cz) vem da configuracao de aquisicao/HISTORICO do projeto e foi conferido pelos desvios-padrao do HISTORICO item 189; o campo channel_names do JSON e' metadado do gerador sintetico e NAO foi usado.
- Nao ha' correcao de referencia/common average nem checagem de impedancia por trial; efeitos de referencia comum podem inflar correlacoes entre canais.

## 8. Comandos

```powershell
# análise completa (padrão: s01 + s02, saída em outputs/eeg/analises_fft_piloto)
.\.venv\Scripts\python.exe src\mestradopy\tools\fft_epocas_piloto.py

# uma sessão só / saída alternativa
.\.venv\Scripts\python.exe src\mestradopy\tools\fft_epocas_piloto.py --sessoes piloto_me_mi_s01
.\.venv\Scripts\python.exe src\mestradopy\tools\fft_epocas_piloto.py --saida outputs/eeg/analises_fft_piloto_v2
```

Tempo de execução desta rodada: 25.6 s. Nada foi commitado
(`git add -A` não foi usado) e nenhum arquivo de `outputs/eeg/piloto_me_mi_s01`,
`s02` foi alterado.
