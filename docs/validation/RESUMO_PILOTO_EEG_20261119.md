# Resumo do piloto EEG — ME/MI, espectro e feedback L/R

**Data:** 19/11/2026
**Dados:** `outputs/eeg/piloto_me_mi_s01` e `s02` (17/11/2026)
**Canais:** C3/C4/Cz (Ch15/Ch16/Ch17; 0-based 14/15/16), 500 Hz.

---

## 1. Decodificação CSP+LDA

| tarefa | dentro s01 | dentro s02 | cross s01→s02 | cross s02→s01 |
|---|---:|---:|---:|---:|
| ME | 0,767 | 0,683 | **0,633** | **0,750** |
| MI | 0,567 | 0,533 | **0,550** | **0,533** |

Leitura: ME tem sinal cross-sessão utilizável; MI fica no nível do acaso
(0,50–0,55) e não sustenta controle confiável.

## 2. Espectro (FFT/PSD e Morlet)

- 60 trials L + 60 R (30/30 por sessão), MI de 4 s a partir do marcador 780;
- mu (8–13 Hz): sem diferença L/R; pico ~10,5 Hz;
- beta (13–30 Hz): C4 levemente menor à direita (−0,49 dB; p=0,022 **não
  corrigido**; d=−0,38), mas sem pico beta destacado;
- banda total 1–45 Hz: padrão lateralizado fraco, com sessões individuais em
  direções opostas → não é achado estável.

## 3. Feedback L/R adaptativo (modelo congelado)

- treino s01: BA 0,700 no bloco de calibração; 0,533 no bloco de adaptação;
- teste congelado s02: **BA=0,567, IC95% [0,437; 0,688]**, n=60;
- confusão: [[20,10],[16,14]];
- veredito: **não-respondedor** pelo critério pré-fixado (IC inferior < 0,55).

ERD mu/beta em s02:
- mu esquerda: C3 −14,3%, C4 −18,8%, Cz −10,8%;
- mu direita: C3 +16,6%, C4 −8,1%, Cz −15,6%;
- beta esquerda: −19 a −22%;
- beta direita: fraca/inconsistente.

## 4. Interpretação

O conjunto dos resultados é coerente com o histórico:

1. **ME é decodificável** e generaliza parcialmente entre sessões;
2. **MI não apresenta ERD/diferença L/R robusta** com 3 canais;
3. o feedback adaptativo não reverteu a illiteracy nas 2 sessões do piloto;
4. o participante deve ser tratado como **não-respondedor no estado atual**,
   não como “sucesso” por um BA alto de calibração.

## 5. Limitações

- 1 participante, 2 sessões do mesmo dia;
- 3 canais, sem topografia;
- piloto com 30 trials/classe por sessão, não os 20/classe do protocolo final;
- p-valores exploratórios sem correção múltipla;
- sem ICA/rejeição de artefatos/EOG;
- o teste congelado usa um modelo treinado em outra sessão do mesmo dia.

## 6. Próximo passo

Executar o protocolo completo:

```text
3 sessões × 20 trials/classe (2 blocos de 10)
sessões 1–2: treino adaptativo
sessão 3: teste congelado
```

Se o teste congelado continuar <0,55 com IC inferior <0,55, registrar
não-respondedor e revisar estratégia/treino antes de novas sessões.
