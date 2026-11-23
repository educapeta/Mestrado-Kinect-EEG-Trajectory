# S02 — feedback, qualidade de sinal e hipótese de interferência

**Data:** 23/11/2026
**Sessão:** `outputs/eeg/lr_adaptativo_s02/`
**Modelo:** `outputs/visao/lr_adaptativo/modelo_s02.joblib`
**Feedback online:** 320 scores (40 trials × 8 janelas).

## 1. Resultado comportamental/classificador

Retreino offline da sessão (bloco 1 calibra, bloco 2 testa):

| bloco | trials | BA | IC95% |
|---|---:|---:|---|
| 1 — calibração | 20 | 0,700 | [0,494; 0,893] |
| 2 — feedback | 20 | 0,650 | [0,437; 0,850] |

Melhor que a s01, mas ainda com n=20/bloco e IC largos. **Não é o teste
congelado**; a confirmação é a s03.

## 2. Scores online

| métrica | bloco 1 | bloco 2 |
|---|---:|---:|
| mediana de \|score\| | 0,007 | 0,054 |
| score máximo absoluto | ~0,01 | ~0,18 |
| mediana de std da MI | 12,2 µV | 10,5 µV |
| mediana de pico-a-pico da MI | 88 µV | 68 µV |

O bloco 2 ficou mais confiante **e** com menos artefato na MI. A correlação
entre |score| e std da MI foi **−0,45** (mais artefato → menos score).

## 3. Interferência do monitor?

Métricas espectrais dentro da MI (mediana por trial):

| métrica | valor |
|---|---:|
| razão 60 Hz / 1–40 Hz | ~0,0002 |
| razão 45–100 Hz / 1–40 Hz | ~0,001 |
| correlação C3–C4 | 0,451 (bloco 1) → 0,706 (bloco 2) |
| correlação \|score\| × linha 60 Hz | −0,155 |
| correlação \|score\| × alta frequência | −0,226 |

O sinal entregue ao decoder já passou por 0,5–40 Hz e o decoder reaplica
8–30 Hz. **Line noise de 60 Hz não explica o feedback.** A maior correlação
C3–C4 no bloco 2 pode indicar componente comum (postura/referência), mas ela
não acompanha o aumento de score (correlação negativa).

## 4. O que ainda pode explicar a percepção de “sinais mais evidentes”

1. **Aprendizado/estratégia** na s02 (modelo adaptado, participante mais
   treinado).
2. **Efeito de sessão/dia** e mudança de postura/atenção.
3. **Artefato de movimento da ME** contaminando a MI (os picos de ~1000 µV
   ocorrem sobretudo na ME; na MI o std cai).
4. **Percepção subjetiva da barra**: ela é mais sensível que o BA real; score
   de 0,05–0,18 move poucos caracteres.
5. **Interferência elétrica de banda larga** não medida pelo CSV filtrado —
   para provar, seria preciso gravar EEG **cru** antes do filtro.

## 5. Decisão para a s03

Não mudar o setup no meio do teste congelado. Para a s03:

- manter a mesma posição de monitor/cadeira/cabos da s02;
- registrar no log qualquer mudança de setup;
- fazer os 60 s de baseline de olhos abertos/fechados como já previsto;
- rodar **congelado**, sem adaptação;
- não olhar fixamente para a barra; usá-la como feedback periférico.

Se quiser investigar interferência de verdade, o teste correto é uma sessão
curta separada, com EEG cru e monitor ligado/desligado/afastado — sem misturar
com o teste congelado.

## 6. Limitações

- 20 trials por bloco;
- o CSV já está filtrado (0,5–40 Hz), então 60 Hz não é visível nele;
- sem marcador de posição do monitor;
- artefato medido por std/pico, não por ICA/EOG;
- duas sessões não permitem separar aprendizado de efeito de dia.
