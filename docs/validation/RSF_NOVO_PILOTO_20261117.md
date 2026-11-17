# RSF no novo piloto ME/MI (17/11/2026)

**Método:** Riemannian Geometry-Based Spatial Filtering (Pan et al.,
*Neural Networks* 188:107511, 2025). O filtro `W` maximiza a distância
riemanniana entre as covariâncias médias das classes:

- RSF ajustado **apenas na sessão de treino**, em janelas de 4 s;
- CSP+LDA e EEGNet usam janelas de 2 s transformadas por `W`;
- teste na outra sessão (cross-session), 60 trials;
- RSF CSP: `dim=2`; RSF EEGNet: `dim=3`;
- sem uso de dados de teste no ajuste.

## Resultados — ME (controle positivo)

| Direção | Modelo | Baseline | RSF | Δ |
|---|---|---:|---:|---:|
| s1→s2 | CSP+LDA | 0,633 | 0,633 | 0,000 |
| s1→s2 | EEGNet A | 0,550 | 0,650 | +0,100 |
| s1→s2 | EEGNet B | 0,633 | **0,750** | **+0,117** |
| s2→s1 | CSP+LDA | 0,750 | 0,767 | +0,017 |
| s2→s1 | EEGNet A | 0,533 | 0,583 | +0,050 |
| s2→s1 | EEGNet B | 0,733 | 0,767 | +0,034 |

## Resultados — MI (alvo)

| Direção | Modelo | Baseline | RSF | Δ |
|---|---|---:|---:|---:|
| s1→s2 | CSP+LDA | 0,550 | 0,550 | 0,000 |
| s1→s2 | EEGNet A | 0,483 | 0,500 | +0,017 |
| s1→s2 | EEGNet B | 0,517 | 0,500 | −0,017 |
| s2→s1 | CSP+LDA | 0,533 | 0,550 | +0,017 |
| s2→s1 | EEGNet A | 0,467 | 0,467 | 0,000 |
| s2→s1 | EEGNet B | 0,717 | 0,500 | **−0,217** |

## Distância riemanniana entre classes

- ME: 0,36–0,43;
- MI: 0,19–0,21.

A distância entre as covariâncias de classe é aproximadamente **o dobro no
ME**. O RSF encontra um subespaço mais discriminativo para ME, mas não para
MI.

## Conclusão

- **CSP+LDA:** RSF praticamente não muda (0 a +1,7 ponto percentual).
- **EEGNet:** RSF melhora consistentemente o **ME** (+3 a +12 pontos), mas
  **não melhora o MI**; em uma direção chega a colapsar o EEGNet para uma
  classe (BA 0,500).
- O resultado do RSF é **dependente da fase**: ele amplifica a separabilidade
  do movimento executado, não cria separabilidade de imagética.
- Não promover a melhora do ME como se fosse melhora de MI. O gargalo do MI
  continua sendo o sinal, não apenas a filtragem espacial.
- Próximo teste justo: repetir com mais sessões/sujeitos; usar validação
  interna para escolher o RSF, nunca o teste final.
