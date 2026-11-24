# Protocolo comum — benchmark 2b e transferência para o piloto

**Data:** 23/11/2026
**Objetivo:** comparar FBCSP+LDA, EEGCCT e CIACNet no BCIC IV-2b e no nosso
piloto, sem vazamento e com métricas comuns.

## Dados

### BCIC IV-2b (fonte)
- `data/raw/Datasets/BCICIV_2b_gdf/*T.gdf` (9 sujeitos × 3 sessões T);
- canais C3, Cz, C4; 250 Hz; eventos 769=esquerda, 770=direita;
- uma janela de 2 s por trial (início 1 s após o cue), como no loader atual;
- **não usar as sessões E** para treino/teste rotulado.

### Piloto (alvo)
- `outputs/eeg/piloto_me_mi_s01` e `s02`;
- 60 trials/sessão, 5 janelas de 2 s por trial, 3 canais C3/C4/Cz;
- labels: esquerda=0, direita=1;
- agrupar janelas por trial em todos os splits.

## Protocolo mínimo obrigatório

1. **BCIC 2b — validação within-subject cross-session**:
   - para cada sujeito, treinar em 2 sessões T e testar na terceira;
   - repetir os 3 sentidos (leave-one-session-out);
   - reportar BA média, desvio e BA por sujeito;
   - alvo: BA média ≥ 0,70 (se não atingir, reportar honestamente).
2. **Sem vazamento**:
   - scaler/CSP/qualquer ajuste não supervisionado deve ser fitado só no treino;
   - no alvo, `StratifiedGroupKFold`/`GroupKFold` por trial;
   - não misturar 2b e piloto no mesmo treino, exceto no cenário explicitamente
     chamado “source+target fine-tuning”.
3. **Comparação no piloto** (mesmos folds por trial):
   - target-only;
   - source-only (modelo congelado);
   - fine-tuning do classificador com encoder congelado;
   - se o modelo permitir, fine-tuning parcial — declarar quantos parâmetros.
4. **Métricas**: BA, IC95% bootstrap (2.000 reamostragens no nível do trial),
   matriz de confusão, n de trials/janelas, custo computacional.
5. **Artefatos**: JSON com resultados, Markdown com tabela e limitações,
   script reproduzível, seed fixa.

## Regras de escopo (evitar conflito)

| agente | arquivos permitidos | não tocar |
|---|---|---|
| FBCSP | `src/mestradopy/eeg/fbcsp/**`, `src/mestradopy/tools/bench_fbcsp*.py`, `outputs/eeg/fbcsp*/**`, `docs/validation/FBCSP*.md` | `data.py`, `transfer_bci2b_csp_ea.py`, modelos EEGCCT/CIACNet |
| EEGCCT | `src/mestradopy/eeg/eegcct/**`, `src/mestradopy/tools/bench_eegcct*.py`, `outputs/eeg/eegcct*/**`, `docs/validation/EEGCCT*.md` | `data.py`, arquivos FBCSP/CIACNet |
| CIACNet | `src/mestradopy/eeg/ciacnet/**`, `src/mestradopy/tools/bench_ciacnet*.py`, `outputs/eeg/ciacnet*/**`, `docs/validation/CIACNET*.md` | `data.py`, arquivos FBCSP/EEGCCT |

- Não commitar; o Lead revisa e commita.
- Pode ler `src/mestradopy/eeg/mi_eegnet/data.py` e `carrega_own`.
- Se precisar de dependência nova, instalar no `.venv` e registrar no relatório.

## Entregável mínimo de cada agente

1. código executável;
2. resultado no 2b com a validação acima;
3. resultado no piloto (target-only/source-only/fine-tuning quando aplicável);
4. JSON + Markdown;
5. lista de limitações e o que não foi possível reproduzir.
