# Janela curta de MI vs janela completa

**Data:** 19/11/2026
**Script:** `src/mestradopy/tools/analisa_janela_mi.py`
**Dados:** `piloto_me_mi_s01` + `s02` (60 trials/sessão, 120 trials, 600 janelas).
**Canais:** C3/C4/Cz; 128 Hz; 8–30 Hz.

## Ponto metodológico

No nosso trial o **cue visual (769/770) vem antes da ME**; a MI começa no
marcador **780**. “2 s após o cue” incluiria ME e não mediria MI. Este teste usa
“primeiros 2 s do bloco de MI”, que é a comparação correta.

## Resultados — 5-fold agrupado por trial

| variante | janelas/trial | BA | confusão |
|---|---:|---:|---|
| **early 0–2 s** | 1 | **0,442** | [[32,28],[39,21]] |
| mid 1–3 s | 1 | 0,542 | [[33,27],[28,32]] |
| late 2–4 s | 1 | 0,567 | [[34,26],[26,34]] |
| full 0–4 s | 5 | 0,475 | [[31,29],[34,26]] |

Por sessão:

| sessão | early | mid | late | full |
|---|---:|---:|---:|---:|
| s01 | 0,533 | **0,683** | 0,583 | 0,567 |
| s02 | 0,500 | 0,517 | 0,533 | 0,533 |

## Cross-session

| variante | s01→s02 | s02→s01 |
|---|---:|---:|
| early 0–2 s | 0,467 | 0,433 |
| mid 1–3 s | **0,617** | 0,550 |
| late 2–4 s | 0,550 | 0,583 |
| full 0–4 s | 0,550 | 0,533 |

## Interpretação

1. A hipótese “a janela longa está atrapalhando” **não se confirmou**:
   - early 0–2 s foi a **pior** variante (0,442 no pooled; 0,467/0,433 no
     cross-session);
   - full ficou ~0,55 no cross-session.
2. O mid 1–3 s teve 0,683 no s01 e 0,617 em s01→s02, mas 0,550 no sentido
   inverso e 0,542 no pooled. É um achado **exploratório**, não pré-registrado;
   não deve ser promovido sem replicação.
3. O sinal de MI não aparece nos primeiros 2 s; se algo aparece, é em uma
   janela mais central/tardia — mas com n=60/sessão e 3 canais, a variabilidade
   é grande.
4. O resultado não salva a hipótese de que “janela longa” era o problema
   principal.

## Limitações

- 2 sessões do mesmo dia; 120 trials;
- CSP+LDA sem rejeição de artefatos/ICA;
- pós-hoc: várias janelas testadas; risco de seleção;
- 3 canais;
- janelas sobrepostas na variante full, agrupadas por trial, mas dependência
  residual.

## Próximo passo

Se quiser investigar a tendência do mid 1–3 s, o teste correto é
**pré-registrar** essa janela e repetir em novas sessões, além de comparar com
CSP-Riemanniano/EEGNet sob validação agrupada.
