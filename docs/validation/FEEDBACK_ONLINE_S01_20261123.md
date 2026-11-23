# Sessão 01 com feedback online — 23/11/2026

**Sessão:** `outputs/eeg/lr_adaptativo_s01/`
**Protocolo:** 2 blocos ×10 trials/classe (40 trials, 20/classe).
**Modelo online:** `outputs/visao/lr_adaptativo/modelo_s01.joblib`.

## 1. Execução técnica

- 40/40 trials completados;
- bloco 1: calibração com **160 janelas** (8 por trial);
- modelo CSP+LDA ajustado ao fim do bloco 1;
- bloco 2: feedback online ativo;
- CSV, JSON e log gravados;
- modelo salvo com `limiar` calibrado.

### Problema corrigido depois da sessão

O formato inicial do modelo não salvava `limiar`, e o `FunctionTransformer`
não era portátil entre processos. Corrigido para:

- `limiar` salvo/carregado;
- função de limpeza serializável;
- modelo reconstruído a partir do bloco de calibração;
- carregamento validado em processo limpo.

## 2. Desempenho do modelo online

Avaliação do modelo salvo (não retreinado) nos dados da sessão:

| bloco | trials | BA |
|---|---:|---:|
| bloco 1 (calibração) | 20 | **0,600** |
| bloco 2 (não usado na calibração) | 20 | **0,500** |

Não houve ganho no bloco com feedback.

### Retreino offline da mesma sessão

| bloco | trials | BA | IC95% |
|---|---:|---:|---|
| calibração | 20 | 0,650 | [0,417; 0,850] |
| adaptação | 20 | 0,550 | [0,338; 0,750] |

A diferença entre online (0,600/0,500) e offline (0,650/0,550) é compatível
com o número diferente de janelas/filtro de treino; em ambos os casos o bloco 2
fica no acaso.

## 3. Interpretação

1. O **pipeline online funciona tecnicamente** e completou a sessão.
2. O feedback online **não melhorou** o desempenho no bloco 2.
3. 20 trials por bloco é pouco; os IC95% são muito largos.
4. Não há evidência, nesta sessão, de que o feedback em tempo real tenha
   ajudado o participante.
5. Ainda é a sessão 1 de treino, não o teste congelado; a decisão final depende
   das sessões 2 e 3.

## 4. Próximo passo

Rodar a **sessão 02** carregando este modelo e adaptando ao fim dos blocos:

```powershell
.\scripts\aquisicao_lr_adaptativo.ps1 -Sessao 02 -Sujeito S01
```

Depois, a sessão 03 com o modelo congelado. O resultado final honesto é o
**teste congelado da s03**, não o BA de calibração.

## 5. Limitações

- 20 trials/classe por sessão, não 60;
- 2 blocos, sem repetição em dias diferentes;
- feedback subjetivo não avaliado formalmente (questionário ainda pendente);
- uma sessão não permite concluir sobre BCI illiteracy;
- a adaptação online só ocorre entre blocos.
