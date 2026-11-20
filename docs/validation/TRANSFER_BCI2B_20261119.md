# Transferência BCIC IV-2b → piloto próprio com CSP+LDA e EA

**Data:** 19/11/2026
**Script:** `src/mestradopy/tools/transfer_bci2b_csp_ea.py`
**Saída:** `outputs/eeg/transfer_bci2b_ea_full/`
**Canais:** C3/C4/Cz; janelas de 2 s, passo 0,5 s, 128 Hz, banda 8–30 Hz.

---

## 1. Dados usados

### BCIC IV-2b
- 9 sujeitos × 3 sessões T = 27 arquivos;
- **3.680 janelas** (2 classes balanceadas);
- event IDs 769/770 → esquerda/direita;
- validação within-subject: treina 2 sessões T, testa a terceira (27 folds).

### Alvo (nossos dados)
- `piloto_me_mi_s01` + `s02`;
- **120 trials** (60 por sessão), **600 janelas** (5 janelas/trial);
- 5-fold agrupado por trial (todas as janelas do mesmo trial no mesmo fold).

**Observação:** o pedido menciona 40 trials/200 janelas; os dados disponíveis
neste projeto têm 60 trials por sessão (300 janelas). O experimento usa os
dados reais disponíveis; a diferença é registrada no JSON.

---

## 2. Validação do pipeline no BCIC IV-2b

| métrica | valor |
|---|---:|
| BA média within-subject | **0,658** |
| desvio-padrão entre folds | 0,122 |
| folds | 27 |

Compatível com o baseline da literatura (~0,68) para CSP+LDA no 2b; o pipeline
básico está correto.

---

## 3. Comparação no alvo

| método | BA | n trials | confusão |
|---|---:|---:|---|
| CSP+LDA só no alvo | 0,475 | 120 | [[31,29],[34,26]] |
| CSP+LDA só no alvo + EA | 0,475 | 120 | [[31,29],[34,26]] |
| BCIC 2b + EA (fonte congelada) | 0,508 | 120 | [[1,59],[0,60]] |
| CSP do 2b + LDA re-treinado no alvo | 0,500 | 120 | [[30,30],[30,30]] |
| CSP+LDA 2b + alvo combinados | 0,500 | 120 | [[39,21],[39,21]] |

Nenhum método transferido superou o acaso de forma útil. O modelo de fonte
congelada colapsou para uma classe, e o fine-tuning não recuperou a
informação.

---

## 4. Interpretação sincera

1. **O pipeline está correto**: no 2b, CSP+LDA reproduz ~0,66, próximo da
   literatura. O problema não é o código do CSP.
2. **Volume de dados não é a única explicação**: mesmo com 27 sujeitos do 2b,
   a transferência para o piloto não funcionou. O domínio é muito diferente:
   sujeito, equipamento, referência, tarefa e paradigma (2b tem 2 classes
   discretas; o piloto tem ME→MI).
3. **EA não salvou**: alinhar covariâncias por sujeito não resolveu a mudança
   de domínio; no alvo, o EA nem mudou a BA.
4. **Fine-tuning com poucos trials não é suficiente**: 120 trials e 3 canais
   são poucos para adaptar filtros espaciais; o LDA re-treinado ficou no acaso.
5. **O resultado do alvo sozinho é ~0,475**, abaixo do “~0,55” de análises
   anteriores. A diferença vem do protocolo: aqui as janelas do mesmo trial
   ficam no mesmo fold e as duas sessões são combinadas. Isso é mais honesto
   contra vazamento, mas o n ainda é pequeno e a variabilidade entre sessões é
   alta.

**Conclusão:** não há evidência de que pré-treinar no BCIC IV-2b com CSP+LDA +
EA melhore a classificação L/R do nosso piloto. O gargalo parece ser a
qualidade/paradigma do sinal de MI, não apenas o volume de dados.

---

## 5. Limitações

- alvo com 2 sessões do mesmo dia, 120 trials, 3 canais;
- validação 2b usa sessões T; as sessões E não foram usadas (sem labels no GDF);
- CSP global no 2b é uma escolha simples; não foi feito CSP por sujeito nem
  seleção de hiperparâmetros;
- sem rejeição de artefatos/ICA/EOG;
- sem correção para múltiplas comparações;
- janelas sobrepostas (passo 0,5 s) — a avaliação agrupa por trial, mas a
  dependência residual existe;
- os resultados são descritivos, com n pequeno.

---

## 6. Próximos passos defensáveis

1. Repetir com o protocolo completo de 3 sessões;
2. avaliar **target cross-session** (s01→s02 e s02→s01) separadamente;
3. tentar CSP+Riemanniano/EEGNet com validação agrupada;
4. registrar EA como negativo para este alvo, não como solução;
5. se o alvo continuar no acaso, discutir BCI illiteracy/paradigma em vez de
   aumentar apenas o volume de dados.
