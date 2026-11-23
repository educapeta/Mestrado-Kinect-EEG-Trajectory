# Teste de ruído EEG cru — perfil laptop (60 Hz / bateria × tomada / proximidade)

Sessão de bancada em **23/11/2026**, no laptop de aquisição, com o **g.Nautilus
RESEARCH** (32 canais, 500 Hz, dongle USB `NR-2018.10.53`, licença HASP ativa).

- **Commit do código:** `7343180` — *"Adiciona perfil laptop ao teste de ruido."*
- **Ferramentas:** `src/mestradopy/tools/aquisicao_ruido_eeg.py` (aquisição) e
  `src/mestradopy/tools/analisa_ruido_eeg.py` (análise) — **nenhuma linha alterada**.
- **Ambiente:** Python 3.13.12 · gpype 3.0.9 · gtec_gds 1.6.0 · numpy 2.5.2
- **Sinal:** EEG cru, **sem notch e sem bandpass** (a única filtragem no caminho é
  a de 59–61 Hz feita pelo analisador para a métrica `RMS60`).

> O g.Pype só executa no terminal de uma IDE suportada ("Execution outside
> supported IDEs detected"); a aquisição rodou no terminal do VS Code, como o
> `HISTORICO_ITERACOES.txt` já documenta.

---

## 1. Protocolo pretendido (perfil `laptop`)

| Cond. | Alimentação | Distância do desktop | Tela |
|---|---|---|---|
| A | bateria | longe (≥ 1 m) | desligada |
| B | bateria | perto (~0,3 m) | desligada |
| C | tomada | longe | desligada |
| D | tomada | perto | desligada |
| E | bateria | perto | **ligada** |

Cada condição: 60 s, com 3 s de estabilização após o ENTER e o sujeito imóvel.
O script grava **um CSV contínuo** com uma coluna `Marker` (1..5) marcando o
início de cada condição; o analisador segmenta pelos marcadores.

## 2. O que realmente aconteceu (importante)

Foram necessárias **duas sessões**, e há um incidente de integridade.

### Sessão 1 — 17:02 (`ruido_bruto_20261123_170201…csv`, 99 MB)

A, B, C, D, E rodaram em sequência, **mas o stream congelou em t = 247,660 s**,
0,8 s antes do marcador da condição C (248,434 s):

```
ultima mudanca em ch1: indice 123830 (t=247.660 s)
apos esse ponto: valores unicos por canal = 2
A: std_mediano =  791.0 uV   OK
B: std_mediano = 1815.7 uV   OK
C: std_mediano =    0.0 uV   CONGELADO
D: std_mediano =    0.0 uV   CONGELADO
E: std_mediano =    0.0 uV   CONGELADO
```

O CSV continuou crescendo por mais 260 s repetindo dois valores fixos por canal.
**C, D e E desta sessão são inválidos e não entram em nenhuma conclusão.**
Hipótese (não confirmada): a pausa de ~94 s entre B e C levou o amplificador a
standby. O script não tem vigia de enlace, então gravou sem avisar.

### Sessão 2 — 17:15 (`ruido_bruto_20261123_171500…csv`, 55 MB)

Refez **C, D e E**. Integridade confirmada (sem congelamento, sinal real nas três).

### Desvios em relação ao protocolo (declarados pelo operador)

- Na sessão 1, **D e E** foram feitas com a **tela ligada** (protocolo pedia
  desligada) e **E na tomada** (protocolo pedia bateria) — sem efeito prático,
  pois essas três condições foram descartadas por congelamento.
- Na sessão 2 não foi registrado o estado da tela; **os rótulos de condição no
  JSON vêm do perfil do script, não de medição** — C/D "tomada" e E "bateria"
  são o que o operador executou, não algo inferido do sinal.

---

## 3. Métricas (saída do `analisa_ruido_eeg.py`)

| sessão | cond | dur | std (µV) | **RMS 60 Hz** | RMS 50 Hz | PSD 40–55 | PSD 65–80 | corr bruta | corr 59–61 Hz |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 17:02 | A | 60,0 s | 789,4 | **246,9** | 1,89 | 3,73 | 2,23 | +0,721 | +0,996 |
| 17:02 | B | 60,0 s | 1761,8 | **1129,5** | 13,51 | 200,59 | 158,66 | +0,773 | +0,991 |
| 17:02 | C | 60,0 s | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | nan | — |
| 17:02 | D | 60,0 s | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | nan | — |
| 17:02 | E | 59,6 s | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | nan | — |
| 17:15 | C | 60,0 s | 9166,3 | **957,6** | 12,14 | 7,48 | 4,90 | −0,868 | +0,997 |
| 17:15 | D | 60,0 s | 3655,2 | **1727,1** | 5,35 | 3,48 | 2,03 | −0,266 | +0,997 |
| 17:15 | E | 60,0 s | 2602,8 | **96,7** | 1,84 | 4,06 | 2,29 | −0,885 | +0,985 |

`std`, `PSD` e `corr` são **exatamente** os valores de `sessao*/resultado.json`.
As colunas `corr 59–61 Hz` **não** estão no JSON do analisador: foram calculadas à
parte (mesma conta do analisador — mediana das correlações par-a-par entre os 8
primeiros canais —, porém sobre o sinal filtrado 59–61 Hz).

### Como o analisador calcula (para não interpretar errado)

- `std_mediano_uV` = mediana, entre canais, do desvio-padrão do segmento cru de 60 s.
- `rms_60_uV` = mediana do desvio-padrão por canal após Butterworth 4ª ordem 59–61 Hz.
- `psd_40_55` / `psd_65_80` = potência mediana (PSD de Welch, todos os canais) nas bandas.
- `corr_media_canais` = **mediana** das correlações par-a-par entre os **8 primeiros
  canais**, no **sinal cru**.

### Por que a `corr bruta` é negativa na sessão 2

No sinal cru, a variância é dominada por **drift de baixa frequência** (std de 2,6 a
9,2 mV, contra 0,8–1,8 mV na sessão 1), e esse drift anda em sentidos opostos entre
canais. Já na faixa de 60 Hz a correlação é **+0,98 a +0,997 em todas as condições
das duas sessões** — a assinatura clássica de interferência de rede em modo comum.
**Conclusão prática: para falar de 60 Hz, use `rms_60_uV`/`corr 59–61 Hz`, não a
correlação crua.**

---

## 4. Comparações

### 4.1 Bateria × tomada — o contraste limpo (sessão 2: D → E, mesma posição)

| | D (tomada) | E (bateria) | fator |
|---|---:|---:|---:|
| RMS 60 Hz | 1727,1 | **96,7** | **17,9× menor na bateria** |
| PSD 40–55 | 3,48 | 4,06 | ~igual |
| PSD 65–80 | 2,03 | 2,29 | ~igual |

**O 60 Hz cai 94,4% na bateria**, sem mexer na banda larga.
Ressalva: vale só se a tela estava no mesmo estado em D e E (não registrado).

### 4.2 Perto × longe do desktop

| comparação | longe | perto | fator |
|---|---:|---:|---:|
| sessão 2 (tomada): C → D | 957,6 | 1727,1 | **1,8× pior perto** |
| sessão 1 (bateria): A → B | 246,9 | 1129,5 | **4,6× pior perto** |

### 4.3 Banda larga 40–80 Hz

Só na sessão 1, ao aproximar: B = 200,6 / 158,7 contra A = 3,73 / 2,23
(**54× e 71×**). **Não se repetiu** na sessão 2 (C/D/E entre 2,0 e 7,5).
Portanto: efeito **não reprodutível** com os dados atuais.

### 4.4 Tela (A/B × E)

Não isolável: E foi medida em outra sessão; qualquer diferença mistura tela,
posição e sessão.

### 4.5 Aviso sobre comparações entre sessões

A×C e B×D cruzam sessões, com drift 3–5× diferente e correlação crua de sinal
oposto. A **direção** coincide com o efeito da bateria (tomada pior), mas a
**magnitude não é confiável**. Use apenas os contrastes internos: A→B, C→D, D→E.

---

## 5. Conteúdo deste diretório

```
resultados/ruido_eeg_20261123/
├── README.md                      (este arquivo)
├── checksums_sha256.txt           SHA-256 dos dois CSV completos (não versionados)
├── sessao1_AB/
│   ├── meta_sessao1.json          metadados da aquisição (marcadores 1..5)
│   ├── relatorio.md               saída do analisador (inclui C/D/E inválidos)
│   ├── resultado.json             métricas por condição
│   └── psd_condicoes.png          PSD por condição
├── sessao2_CDE/
│   ├── meta_sessao2.json          metadados (marcadores 3..5)
│   ├── relatorio.md
│   ├── resultado.json
│   └── psd_condicoes.png
└── amostras/                      recortes de 10 s por condição, sinal cru
    ├── sessao1_A_10s.csv  … sessao1_E_10s.csv
    └── sessao2_C_10s.csv  … sessao2_E_10s.csv
```

### Sobre as amostras

Cada `*_10s.csv` tem **10 s** a partir do marcador da condição, mesmas 34 colunas
do original (`Time`, `Ch1…Ch32`, `Marker`), **sem filtro**. Servem para
recalcular 60 Hz, PSD e correlação de forma independente — **não substituem** os
60 s (o valor de `rms_60` recalculado numa amostra de 10 s não vai bater
exatamente com a tabela).

As de `sessao1_C/D/E` são justamente as **congeladas**: úteis para reproduzir a
detecção do incidente (todos os canais com std = 0).

### Brutos completos (fora do git)

| arquivo | bytes | SHA-256 |
|---|---:|---|
| `ruido_bruto_20261123_170201_20261123_170211.csv` | 103 838 468 | `c71d91e0e2a87f7f33c9f48050ec8b201b8b01fe66f1838f5e11c693c0ce66a8` |
| `ruido_bruto_20261123_171500_20261123_171508.csv` | 57 867 567 | `2055f1a0e6852941ce7cd67a7f3d8f3328ae1ae1e7364efcd003ea7bc0cdb3db` |

Ficam em `outputs/eeg/ruido_laptop/` na máquina de aquisição (`outputs/` e `*.csv`
são ignorados pelo `.gitignore` do repositório).

---

## 6. Como reproduzir a análise

```powershell
python src/mestradopy/tools/analisa_ruido_eeg.py `
  outputs/eeg/ruido_laptop/ruido_bruto_20261123_171500_20261123_171508.csv `
  --meta outputs/eeg/ruido_laptop/ruido_bruto_20261123_171500.json `
  --saida outputs/eeg/ruido_laptop/analise_cde
```

---

## 7. Perguntas abertas / o que falta para fechar

1. **Rodar A–E numa única sessão** (idealmente 2–3 ciclos, pausas curtas) para
   eliminar o confundimento de drift entre sessões e permitir as quatro
   comparações previstas (A×B, A×C, B×D, A/B×E).
2. **Registrar o estado da tela** em cada condição (não foi anotado na sessão 2).
3. **Repetir B** (bateria + perto): foi o único caso com banda larga 40–80 Hz
   alta (200/159), e não se repetiu.
4. **Vigia de enlace na aquisição**: o congelamento passou silencioso; um teste de
   `std == 0` após cada condição evitaria gravar 4 min de lixo.
