# Ruído EEG cru — aquisição de 23/11/2026

**Arquivo:** `outputs/eeg/ruido_bruto/ruido_bruto_20261123_161002_...csv`
**Duração:** 60 s por condição (após 1 s de estabilização).
**EEG:** 32 canais, 500 Hz, **sem notch e sem bandpass**.
**Análise:** `outputs/eeg/ruido_bruto/analise/`.

## Condições reais

| cond | monitor | posição |
|---|---|---|
| A | ligado | normal, ~1 m |
| B | desligado | normal, ~1 m |
| C | ligado | afastado |
| D | desligado | afastado |
| E | **ligado** | **muito perto do desktop (~0,3 m)** |

O E não foi “monitor desligado”: a tela ficou ligada e o cap se aproximou do
desktop. Isso confunde monitor + desktop + distância, mas ainda é informativo.

## Resultados

| cond | std mediano | RMS 60 Hz | RMS 50 Hz | PSD 40–55 | PSD 65–80 | p99,9 \|x\| |
|---|---:|---:|---:|---:|---:|---:|
| A | 260,5 µV | 98,8 µV | 1,75 µV | 3,63 | 1,89 | 1092 µV |
| B | 177,4 µV | 94,9 µV | 1,64 µV | 3,33 | 1,93 | 484 µV |
| C | 349,8 µV | 88,2 µV | 1,74 µV | 3,31 | 1,91 | 910 µV |
| D | 366,1 µV | 83,7 µV | 1,57 µV | 3,43 | 1,87 | 761 µV |
| E | **1553,8 µV** | **1469,6 µV** | **9,62 µV** | **47,7** | **41,0** | **3035 µV** |

![PSD por condição](outputs/eeg/ruido_bruto/analise/psd_condicoes.png)

## Perfil do ruído

1. **60 Hz é dominante em todas as condições**:
   - pico estreito e enorme em 60 Hz;
   - RMS de 84–99 µV nas condições A–D;
   - **1,47 mV no E** (~15× maior).
2. **50 Hz é baixo** em A–D (~1,6 µV) e sobe um pouco no E (~9,6 µV).
3. **Harmônicos 100/120 Hz são baixos** em todas; não é um perfil típico de
   distorção não linear forte.
4. **E tem ruído de banda larga**:
   - platô elevado entre ~40–80 Hz;
   - PSD 40–55 Hz ~14× maior que A–D;
   - PSD 65–80 Hz ~21× maior.
5. **Modo comum alto**:
   - correlação média entre canais 0,75–0,90 em A/B/D/E;
   - 60 Hz aparece em todos os canais, mas com amplitude desigual
     (RMS por canal de ~3 a 134 µV em A–D; até ~1,9 mV no E).
6. **Amplitude extrema no E**: p99,9 de |x| ≈ 3 mV, compatível com
   saturação/artefato de campo próximo.

## O que isso significa

- **O monitor ligado/desligado não explica o 60 Hz a ~1 m**:
  A (ligado) ≈ B (desligado); C (ligado) ≈ D (desligado).
- O problema principal é **60 Hz de modo comum** — provavelmente
  rede/terra/referência/cabos, não o monitor.
- Perto do desktop (E) aparece **ruído de banda larga + 50 Hz + 60 Hz gigante**:
  isso aponta para **fonte chaveada/GPU/desktop** e acoplamento capacitivo,
  não apenas para a tela.
- O 60 Hz de 84–99 µV é enorme para EEG (sinal de mu/beta é da ordem de
  poucos µV). Mesmo com notch em 60 Hz, essa interferência pode saturar o
  amplificador e gerar resíduos não lineares.

## Próximos testes para separar as fontes

1. **Repetir E com o monitor realmente desligado**, mesma distância do
   desktop.
2. **Adicionar F**: monitor ligado, muito perto do desktop.
   - E (monitor off + perto) vs F (monitor on + perto) separa tela de desktop.
3. Repetir A–D com:
   - receiver/cabos mais longe do desktop;
   - ferrites no cabo USB;
   - laptop na bateria (se possível);
   - outra tomada/região da sala.
4. Verificar **referência/terra** e impedância de cada eletrodo; um eletrodo
   ruim pode injetar 60 Hz em modo comum.
5. Teste com entradas curto-circuitadas ou cap fora da cabeça para separar
   ruído do amplificador do ruído captado pelos eletrodos.

## Conclusão curta

O perfil é **60 Hz de modo comum em todas as condições**, com agravamento
extremo perto do desktop, acompanhado de banda larga 40–80 Hz. O monitor, por
si só, não é o vilão principal a 1 m; o conjunto **desktop/fonte/terra/cabos**
e a proximidade física são. Antes de continuar as sessões, vale corrigir
referência/terra e manter o cap longe do desktop.
