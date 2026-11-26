# Protocolo de deslocamento de 0,5 m — par aux0/aux1 × Kinect

**Data:** 26/11/2026

## Objetivo

Validar a **escala absoluta** e a trajetória 3D do novo par de câmeras
auxiliares 0/1 contra uma referência externa conhecida: um deslocamento de
**0,5 m** medido com régua/fita na mesa.

O Kinect **não é verdade absoluta**; ele é comparado, mas a referência é a
régua. O protocolo serve para responder:

- o par reconstrói 0,5 m corretamente?
- o Kinect reconhece 0,5 m corretamente?
- há drift de retorno?
- a trajetória é fisicamente plausível?
- o erro par×Kinect é aceitável?

## Setup

- Kinect RGB+depth ligado;
- aux0 e aux1 abertas, calibradas;
- mão direita do participante;
- marcar na mesa:
  - **A**: posição inicial;
  - **B**: 0,50 m de A, medido com régua/fita.
- usar fita/marcador para conferir a distância;
- manter a mão visível nas duas auxiliares e no Kinect;
- evitar oclusão e mudar o menos possível a orientação da palma.

Calibração atual verificada:

```text
aux0 RMS = 1.699 px
aux1 RMS = 1.315 px
```

## Fases (35 s)

| tempo | fase | ação |
|---|---|---|
| 0–5 s | `A_parado` | mão parada em A |
| 5–15 s | `ida_0p5m` | mover A→B em ~10 s |
| 15–20 s | `B_parado` | mão parada em B |
| 20–30 s | `volta_0p5m` | mover B→A em ~10 s |
| 30–35 s | `A_retorno` | mão parada em A |

## Comando

```powershell
cd C:\mestradopy
$env:PYTHONPATH="C:\mestradopy;C:\mestradopy\src\mestradopy;C:\mestradopy\src\mestradopy\core;C:\mestradopy\src\mestradopy\tools"

.\.venv\Scripts\python.exe src\mestradopy\tools\compara_pair_kinect_online.py `
  --protocolo-0p5m `
  --num-hands 1 `
  --exposicao "0:-7,1:-7" `
  --csv outputs\visao\protocolo_0p5m_metricas.csv `
  --trajetoria-csv outputs\visao\protocolo_0p5m_traj.csv
```

## Análise

```powershell
.\.venv\Scripts\python.exe src\mestradopy\tools\analisa_protocolo_0p5m.py `
  outputs\visao\protocolo_0p5m_traj.csv `
  --saida outputs\visao\protocolo_0p5m_analise
```

## Métricas

- deslocamento A→B do par e do Kinect em metros;
- erro absoluto contra 0,5 m;
- drift de retorno a A;
- comprimento de trajetória ida/volta;
- erro 3D par×Kinect (mediana/p90);
- cobertura/nº de frames válidos.

## Critério de leitura

- comparar **par vs régua** e **Kinect vs régua**;
- se o par acertar 0,5 m dentro de poucos cm e o Kinect errar mais, o par tem
  melhor escala;
- se os dois tiverem erro parecido, o Kinect depth ainda pode ser usado como
  referência relativa, mas não absoluta;
- se o drift de retorno for grande, investigar tracking/identidade/tempo.

## Limitações

- é uma medida de deslocamento, não de pose completa;
- não valida orientação/grip;
- depende de a mão permanecer visível nas duas auxiliares;
- Kinect depth tem falhas e correção depth→cor;
- uma única trajetória não caracteriza repetibilidade; repetir 3–5 vezes.
