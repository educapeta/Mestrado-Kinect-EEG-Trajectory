# Runbook de aquisição — protocolo L/R adaptativo

**Versão:** 19/11/2026
**Protocolo:** `docs/guias/PROTOCOLO_TREINO_LR_ADAPTATIVO.md`
**Objetivo:** 3 sessões × 20 trials/classe (2 blocos de 10), sendo s01–s02 de treino adaptativo e s03 de teste congelado.

---

## 0. Antes da primeira sessão

1. Fechar **gRecorder/GtecSuite** (o g.Nautilus só entrega dados com o receptor livre).
2. Conferir:
   - Kinect conectado;
   - aux0 SIGMA e aux1 Trust conectadas;
   - luz de bancada suficiente;
   - espaço em disco (vídeos vão para `F:`).
3. Rodar o preflight:

```powershell
cd C:\mestradopy
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verificar_preflight_lr.ps1
```

Critérios:
- `RAIZ_OK`;
- calibração aux0/aux1 utilizável (RMS ≤ 2 px);
- câmeras abrem na resolução calibrada;
- plano do feedback = 3 sessões × 20/classe;
- testes `73/73` e `10/10`.

Não iniciar a sessão se qualquer critério falhar.

---

## 1. Sessão 01 — treino adaptativo

```powershell
.\scripts\aquisicao_lr_adaptativo.ps1 -Sessao 01 -Sujeito S01
```

Durante:
- trial: cue 1 s → ME 4 s → transição 1 s → MI 4 s → ITI;
- 2 blocos × 10/classe = 20/classe;
- pausa de 2 min entre blocos;
- não conversar/mexer durante MI;
- anotar estratégia e fadiga no questionário.

Depois:

```powershell
.\scripts\rodar_feedback_lr.ps1 `
  -SessaoDir outputs\eeg\lr_adaptativo_s01 `
  -Modo treino `
  -SalvarModelo outputs\visao\lr_adaptativo\modelo_s01.joblib
```

Guardar:
- `outputs/eeg/lr_adaptativo_s01/` (CSV + JSON + log);
- `outputs/visao/lr_adaptativo/modelo_s01.joblib`;
- relatório do feedback.

---

## 2. Sessão 02 — treino adaptativo (continuação)

```powershell
.\scripts\aquisicao_lr_adaptativo.ps1 -Sessao 02 -Sujeito S01
```

Depois:

```powershell
.\scripts\rodar_feedback_lr.ps1 `
  -SessaoDir outputs\eeg\lr_adaptativo_s02 `
  -Modo treino `
  -Modelo outputs\visao\lr_adaptativo\modelo_s01.joblib `
  -SalvarModelo outputs\visao\lr_adaptativo\modelo_s02.joblib
```

---

## 3. Sessão 03 — teste congelado

Mesma aquisição, mas **sem adaptação** depois:

```powershell
.\scripts\aquisicao_lr_adaptativo.ps1 -Sessao 03 -Sujeito S01
```

Depois:

```powershell
.\scripts\rodar_feedback_lr.ps1 `
  -SessaoDir outputs\eeg\lr_adaptativo_s03 `
  -Modo teste-congelado `
  -Modelo outputs\visao\lr_adaptativo\modelo_s02.joblib
```

Critérios pré-fixados:
- respondedor: BA ≥ 0,55 com IC95% inferior ≥ 0,55;
- sucesso: BA ≥ 0,65;
- não-respondedor: IC95% inferior < 0,55 → revisar estratégia/ERD antes de continuar.

---

## 4. Checklist pós-sessão

- [ ] n trials/classe = 20 (2 blocos × 10);
- [ ] marcadores 768/769/770/778/779/780/781/790/791 presentes;
- [ ] nenhum bloco abortado;
- [ ] questionário de estratégia preenchido;
- [ ] modelo salvo apenas em s01/s02;
- [ ] s03 avaliada com modelo congelado;
- [ ] CSV/JSON/log copiados para o backup do projeto;
- [ ] registrar data/hora, fadiga e intercorrências no histórico.

---

## 5. Critério para a próxima sessão

- se BA do bloco anterior ≥ 0,55 e IC inferior ≥ 0,45: seguir o plano;
- se BA < 0,50 por dois blocos: pausar, revisar estratégia/ERD e considerar não-respondedor;
- não aumentar trials/classe nem rodar 60 em sequência;
- não usar s03 para treinar/adaptar.
