# Controle FBCSP — baseline, ME, MI e bandas

**Data:** 25/11/2026
**Dados:** piloto s01 + s02, 120 trials, 600 janelas de 2 s, folds canônicos.
**Método:** FBCSP (9 bandas 4–40 Hz, 4 componentes CSP por banda, LDA).
**Métrica:** BA por trial, 5-fold agrupado por trial.

## 1. Controle por fase

| fase | janelas | BA trial | dp | folds |
|---|---:|---:|---:|---|
| baseline pré-cue 2 s | 120 | **0,462** | 0,143 | [0,299; 0,643; 0,375; 0,417; 0,575] |
| ME | 557 | **0,783** | 0,085 | [0,708; 0,875; 0,708; 0,875; 0,750] |
| MI | 559 | **0,892** | 0,076 | [0,792; 0,833; 0,958; 0,958; 0,917] |

Interpretação:

- baseline 2 s fica em **chance** → não há informação de classe antes do cue;
- ME é discriminável (movimento executado);
- MI é ainda mais discriminável que ME;
- portanto, o sinal aprendido pelo FBCSP está **na fase de tarefa**, não no
  baseline/ordem/estratégia pré-cue.

A ressalva é que ME/MI podem conter EMG, postura ou pequenos movimentos. O
baseline limpo reduz a suspeita de artefato pré-cue, mas **não elimina
contaminação muscular durante a MI**.

## 2. Reprodutibilidade do resultado

| configuração | BA trial | dp | folds |
|---|---:|---:|---|
| FBCSP full (9 bandas, 4 CSP) | 0,908 | 0,075 | [0,833; 0,833; 0,958; 1,000; 0,917] |
| FBCSP reduzido (4 bandas, 2 CSP) | 0,867 | 0,080 | [0,833; 0,750; 0,917; 0,958; 0,875] |
| FBCSP reduzido (6 bandas, 2 CSP) | 0,883 | 0,095 | [0,833; 0,750; 0,917; 1,000; 0,917] |
| primeira janela/trial | 0,908 | — | [0,875; 0,875; 0,958; 0,875; 0,958] |
| última janela/trial | 0,842 | — | [0,833; 0,792; 0,833; 0,875; 0,875] |
| canais embaralhados | 0,567 | — | [0,542; 0,542; 0,542; 0,750; 0,458] |
| features simples (var + mu/beta) | 0,683 | 0,127 | [0,583; 0,583; 0,625; 0,875; 0,750] |
| CSP+LDA | 0,583 | 0,141 | [0,417; 0,750; 0,458; 0,667; 0,625] |

Por sessão: s01 = 0,933; s02 = 0,850.
Permutação de labels no nível do trial (100 permutações): p ≈ 0,01.
Canais embaralhados caem para ~0,57 → o padrão espacial importa.

## 3. Bandas que o FBCSP está usando

| banda | BA trial (MI) |
|---|---:|
| 4–8 Hz | 0,658 |
| 8–12 Hz | **0,508** |
| 12–16 Hz | **0,767** |
| 16–20 Hz | **0,800** |
| 20–24 Hz | **0,808** |
| 24–28 Hz | 0,525 |
| 28–32 Hz | 0,667 |
| 32–36 Hz | 0,592 |
| 36–40 Hz | 0,708 |

O sinal está concentrado em **12–24 Hz** (mu baixo/beta), com contribuições de
4–8 e 36–40 Hz. A banda 8–12 Hz sozinha fica no acaso. Isso é consistente com
ERD/ERS sensorimotor, mas não é prova de ausência de EMG.

## 4. O que isso significa

- O FBCSP não está aprendendo um artefato de baseline/ordem.
- O efeito é robusto a reduzir bandas/componentes, usar uma janela por trial e
  aparece nas duas sessões.
- Ainda falta descartar EMG/artefato motor: **o próximo protocolo deve gravar
  EMG nos trials**.

## 5. Plano de EMG para os próximos trials

### Hardware
- 2 canais bipolares de EMG de superfície:
  - flexor superficial dos dedos (FDS);
  - extensor dos dedos (ED).
- Mesma referência/terra do EEG, cabos afastados dos de EEG.
- Amostragem ≥ 500 Hz (ideal 1000 Hz), banda 20–200 Hz, notch 60 Hz.

### Aquisição
- Marcadores de EMG sincronizados com EEG (mesmo evento de cue/ME/MI).
- Registrar EMG cru e envelope (retificação + passa-baixa ~10 Hz).
- Guardar arquivo separado com o mesmo relógio de eventos.

### Análise
- RMS/envelope de EMG por janela de 2 s em baseline, ME e MI.
- Definir “trial limpo” usando apenas baseline pré-cue (ex.: MI ≤ baseline +
  2 DP), sem olhar o label.
- Comparar FBCSP:
  - todos os trials;
  - só trials EMG-limpos;
  - com EMG como covariável/regressão.
- Reportar correlação entre score do FBCSP e EMG.
- Se o efeito desaparecer nos trials limpos, o sinal era EMG; se persistir, o
  argumento de MI fica muito mais forte.

## 6. Texto curto para a tese

> “Para verificar se o classificador FBCSP capturava informação relacionada à
> tarefa e não a artefatos pré-estímulo, aplicou-se o mesmo pipeline a janelas
> de baseline pré-cue, ME e MI. O baseline ficou no nível do acaso (BA = 0,462),
> enquanto ME e MI apresentaram BA = 0,783 e 0,892, respectivamente. O efeito
> foi reproduzido em ambas as sessões e caiu para 0,567 quando a topografia foi
> destruída pelo embaralhamento dos canais. As bandas 12–24 Hz foram as mais
> informativas. Como EMG não foi registrado neste piloto, a próxima etapa
> incluirá canais de EMG para descartar contaminação muscular.”
