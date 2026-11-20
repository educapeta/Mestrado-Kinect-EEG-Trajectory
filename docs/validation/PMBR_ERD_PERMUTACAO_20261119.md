# ERD/ERS por banda e permutação — teste do PMBR

**Data:** 19/11/2026
**Script:** `src/mestradopy/tools/erd_pmbr_permutacao.py`
**Dados:** `piloto_me_mi_s01` + `s02` (120 trials, 60 L/60 R).
**Baseline:** 1 s imediatamente antes do cue; MI começa em 780.

## 1. ERD/ERS por janela de 1 s

Valores em %, medianas por classe; colunas C3/C4/Cz.
Negativo = ERD; positivo = ERS.

### Mu 8–12 Hz

| janela | esquerda (C3/C4/Cz) | direita (C3/C4/Cz) |
|---|---|---|
| 0–1 s | −8,4 / +3,9 / −4,0 | −24,8 / −10,6 / −22,9 |
| 1–2 s | −12,3 / −16,9 / −20,6 | −57,7 / −35,7 / −49,3 |
| 2–3 s | +23,8 / −0,5 / −19,8 | −22,8 / −16,1 / −36,6 |
| 3–4 s | +2,3 / −18,6 / −4,8 | −22,2 / −6,5 / −20,8 |

### Beta 13–30 Hz

| janela | esquerda (C3/C4/Cz) | direita (C3/C4/Cz) |
|---|---|---|
| 0–1 s | +1,5 / −15,8 / −18,3 | **+9,8 / +21,5 / +11,6** |
| 1–2 s | −25,2 / −18,0 / −37,4 | −10,6 / −21,2 / −1,7 |
| 2–3 s | −26,4 / +0,1 / −32,5 | −1,5 / −19,1 / −14,9 |
| 3–4 s | −21,6 / −14,9 / −26,2 | +5,2 / −21,7 / −10,9 |

## 2. Leitura do PMBR

O padrão esperado era:

- early beta **aumentado** (rebound do ME);
- late beta **diminuído** (ERD da MI).

O que os dados mostram:

- **direita:** early beta aumentado em C4/Cz (+21,5/+11,6%), compatível com
  rebound;
- **esquerda:** early beta **não** aumentado (C4/Cz negativos);
- **1–2 s:** beta já negativo nas duas classes;
- **late:** beta majoritariamente negativo (ERD), mas com exceções (+5,2 em C3
  na direita).

Portanto o PMBR **não se confirma de forma limpa**: aparece em uma classe/canal,
não como padrão consistente.

## 3. Permutação — late 2–4 s

Janela de 2 s, CSP+LDA, 5-fold por trial, 1.000 permutações de labels no nível
do trial:

```text
BA observado:    0,567
p unilateral:    0,1149
nulo mediana:    0,483
nulo p95:        0,583
nulo máximo:     0,675
```

O BA observado está **abaixo do p95 do nulo**; não é significativo.

## 4. Conclusão honesta

1. **PMBR não confirmado** como explicação principal: o padrão early/late não é
   consistente entre classes/canais.
2. **Late 2–4 s não tem sinal estatisticamente convincente** (p=0,115).
3. O 0,567 observado é compatível com flutuação/ruído estruturado, não com um
   ERD lateralizado estável.
4. Não há justificativa, com estes dados, para mudar o paradigma ou promover
   “janela late” como solução.

## 5. Limitações

- 120 trials, 3 canais, 2 sessões do mesmo dia;
- ERD por janela de 1 s com Welch de 1 s (resolução 1 Hz);
- múltiplas janelas/bandas/canais testados; não corrigido;
- permutação de labels no nível de trial, mas janelas de 2 s dentro de cada
  trial são dependentes;
- sem rejeição de artefatos/ICA;
- o baseline pré-cue não é um baseline de ME.

## 6. Próximo passo

Se quiser insistir no PMBR, o teste correto é **pré-registrar** uma janela e um
canal (ex.: beta C4 early) e replicar em sessões novas, com marcador de ME
separado e mais trials. Sem isso, é exploração pós-hoc e não deve virar
conclusão da tese.
