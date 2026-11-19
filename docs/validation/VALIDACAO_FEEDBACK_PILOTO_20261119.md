# Validação do feedback L/R adaptativo em dados reais — piloto 17/11

**Data:** 19/11/2026
**Fonte:** `outputs/eeg/piloto_me_mi_s01` (treino) e
`outputs/eeg/piloto_me_mi_s02` (teste congelado), piloto ME/MI de 17/11.
**Tool:** `src/mestradopy/tools/feedback_adaptativo.py`.
**Modelo:** `outputs/visao/lr_adaptativo/modelo_s01_v3.joblib`.

---

## 1. Correções importantes

Durante a validação foram encontrados dois bugs no loader do feedback:

1. **Mapeamento de canais**: usava `channel_names` do JSON do g.Nautilus
   (32 nomes), mas o CSV tem os canais físicos `Ch01..Ch33`; os canais úteis
   são `Ch15/Ch16/Ch17` (0-based 14/15/16) = **C3/C4/Cz**. Antes, o decoder
   usava colunas erradas.
2. **Orientação no resample**: `_resample` recebia `(amostras, canais)` em vez
   de `(canais, amostras)` e reamostrava o eixo dos canais, deixando **1 canal**
   em vez de 3.

Esses bugs invalidavam o BA/ERD anteriores. Resultados corrigidos abaixo.

## 2. Treino adaptativo — s01

| sessão | bloco | papel | trials | BA | IC95% | tolerância | ganho | ação |
|---:|---:|---|---:|---:|---|---:|---:|---|
| 1 | 1 | calibração | 30 | 0,700 | [0,525; 0,864] | 0,35 | 1,00 | mantido |
| 1 | 2 | adaptação | 30 | 0,533 | [0,400; 0,667] | 0,40 | 0,90 | mais fácil |

## 3. Teste congelado — s02

```text
BA = 0,567
IC95% = [0,437; 0,688]
n = 60
confusão (linhas=verdade L/R): [[20, 10], [16, 14]]
|score| médio = 0,197
veredito: NÃO-RESPONDEDOR (IC95% inferior < 0,55)
```

## 4. ERD/ERS mu/beta da MI (s02, relativo ao baseline)

| banda | classe | C3 | C4 | Cz |
|---|---|---:|---:|---:|
| mu 8–13 Hz | esquerda | −14,3% | −18,8% | −10,8% |
| mu 8–13 Hz | direita | +16,6% | −8,1% | −15,6% |
| beta 13–30 Hz | esquerda | −21,8% | −22,1% | −19,3% |
| beta 13–30 Hz | direita | +2,6% | −11,4% | −2,0% |

Há dessincronização mu/beta na MI esquerda, mas a direita não mostra o padrão
contralateral esperado de forma consistente; o decoder congelado não generaliza
para s02.

## 5. Interpretação

- Consistente com o histórico: **ME é detectável, MI é fraca**;
- o treino no s01 teve BA 0,700 no primeiro bloco, mas caiu para 0,533 no
  segundo;
- o teste congelado no s02 ficou em **0,567**, com IC incluindo 0,55;
- o participante deve ser tratado como **não-respondedor no estado atual**;
- não promover o BA de treino como resultado: o teste congelado é a métrica.

## 6. Limitações

- 2 sessões do mesmo dia; não é o protocolo completo de 3 sessões;
- o piloto tinha 30 trials/classe por sessão (15×2), não 20/classe;
- o modelo foi treinado no s01 e testado no s02 sem nova aquisição;
- ERD usa baseline de 1 s antes do cue, não olhos abertos/fechados;
- 3 canais e 1 participante; sem correção de artefatos/ICA.

## 7. Próximos passos

1. Rodar o protocolo completo: 3 sessões × 20 trials/classe (2 blocos de 10),
   com a sessão 3 congelada.
2. Manter o modelo v3 como baseline congelado.
3. Se o BA do teste congelado não passar de 0,55 com IC inferior ≥0,55,
   registrar não-respondedor e revisar estratégia/treino.
