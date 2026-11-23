# Runbook de aquisiÃ§Ã£o â€” protocolo L/R adaptativo

**VersÃ£o:** 19/11/2026
**Protocolo:** `docs/guias/PROTOCOLO_TREINO_LR_ADAPTATIVO.md`
**Objetivo:** 3 sessÃµes Ã— 20 trials/classe (2 blocos de 10), sendo s01â€“s02 de treino adaptativo e s03 de teste congelado.

---

## 0. Antes da primeira sessÃ£o

1. Fechar **gRecorder/GtecSuite** (o g.Nautilus sÃ³ entrega dados com o receptor livre).
2. Conferir:
   - Kinect conectado;
   - aux0 SIGMA e aux1 Trust conectadas;
   - luz de bancada suficiente;
   - espaÃ§o em disco (vÃ­deos vÃ£o para `F:`).
3. Rodar o preflight:

```powershell
cd C:\mestradopy
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verificar_preflight_lr.ps1
```

CritÃ©rios:
- `RAIZ_OK`;
- calibraÃ§Ã£o aux0/aux1 utilizÃ¡vel (RMS â‰¤ 2 px);
- cÃ¢meras abrem na resoluÃ§Ã£o calibrada;
- plano do feedback = 3 sessÃµes Ã— 20/classe;
- testes `73/73` e `10/10`.

NÃ£o iniciar a sessÃ£o se qualquer critÃ©rio falhar.

---

## Feedback online (novo)

A aquisiÃ§Ã£o agora abre o feedback online junto com o protocolo:

- **s01:** bloco 1 Ã© calibraÃ§Ã£o (CSP+LDA); o bloco 2 jÃ¡ mostra o feedback e o
  modelo Ã© salvo em `outputs/visao/lr_adaptativo/modelo_s01.joblib`;
- **s02:** carrega o modelo s01, mostra feedback nos dois blocos e adapta ao
  fim de cada bloco; salva `modelo_s02.joblib`;
- **s03:** carrega o modelo s02 e roda **congelado**; sem adaptaÃ§Ã£o.

Na tela aparecem: cue, texto da MI e a barra de feedback
(`ESQUERDA ... DIREITA`, score e p(dir)). O decoder sÃ³ vÃª janelas de 2 s
dentro da MI; nenhuma janela atravessa MEâ†’MI. O `rodar_feedback_lr.ps1`
continua disponÃ­vel para consolidar o relatÃ³rio offline depois da sessÃ£o.

## 1. SessÃ£o 01 â€” treino adaptativo

```powershell
.\scripts\aquisicao_lr_adaptativo.ps1 -Sessao 01 -Sujeito S01
```

Durante:
- trial: cue 1 s â†’ ME 4 s â†’ transiÃ§Ã£o 1 s â†’ MI 4 s â†’ ITI;
- 2 blocos Ã— 10/classe = 20/classe;
- pausa entre blocos indefinida até pressionar **ESPAÇO**;
- nÃ£o conversar/mexer durante MI;
- anotar estratÃ©gia e fadiga no questionÃ¡rio.

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
- relatÃ³rio do feedback.

---

## 2. SessÃ£o 02 â€” treino adaptativo (continuaÃ§Ã£o)

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

## 3. SessÃ£o 03 â€” teste congelado

Mesma aquisiÃ§Ã£o, mas **sem adaptaÃ§Ã£o** depois:

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

CritÃ©rios prÃ©-fixados:
- respondedor: BA â‰¥ 0,55 com IC95% inferior â‰¥ 0,55;
- sucesso: BA â‰¥ 0,65;
- nÃ£o-respondedor: IC95% inferior < 0,55 â†’ revisar estratÃ©gia/ERD antes de continuar.

---

## 4. Checklist pÃ³s-sessÃ£o

- [ ] n trials/classe = 20 (2 blocos Ã— 10);
- [ ] marcadores 768/769/770/778/779/780/781/790/791 presentes;
- [ ] nenhum bloco abortado;
- [ ] questionÃ¡rio de estratÃ©gia preenchido;
- [ ] modelo salvo apenas em s01/s02;
- [ ] s03 avaliada com modelo congelado;
- [ ] CSV/JSON/log copiados para o backup do projeto;
- [ ] registrar data/hora, fadiga e intercorrÃªncias no histÃ³rico.

---

## 5. CritÃ©rio para a prÃ³xima sessÃ£o

- se BA do bloco anterior â‰¥ 0,55 e IC inferior â‰¥ 0,45: seguir o plano;
- se BA < 0,50 por dois blocos: pausar, revisar estratÃ©gia/ERD e considerar nÃ£o-respondedor;
- nÃ£o aumentar trials/classe nem rodar 60 em sequÃªncia;
- nÃ£o usar s03 para treinar/adaptar.

