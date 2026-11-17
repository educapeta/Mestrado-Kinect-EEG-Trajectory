# Integração das Colunas FUSED_* no EEG Motor Paradigm

**Versão:** 1.0 (schema v1)  
**Data:** 05/11/2026  
**Status:** Scaffold — integração real só após validação da Fase 4

---

## 1. Resumo

Este documento descreve **onde e como** plugar as colunas `FUSED_*` (schema v1, 19 colunas) no
`eeg_motor_paradigm.py`, mantendo as colunas legadas `PALM_*` / `IMU_*` **intactas**,
protegidas por feature flag `--fusion-enabled` (default `False`).

> **Regra de ouro:** O EEG **não quebra** se a fusão falhar. `PALM_*`/`IMU_*` continuam
> sendo gravadas exatamente como hoje. `FUSED_*` são **adicionais**, versionadas, e só
> aparecem no CSV quando `--fusion-enabled` é passado explicitamente.

---

## 2. Arquivos Envolvidos

| Arquivo | Papel | Pode alterar agora? |
|---------|-------|---------------------|
| `fusion/eeg_export.py` | Contrato de dados: `FUSED_COLUMNS`, `fused_row()`, `validate_fused_row()`, `append_fused_columns()` | **SIM** (este scaffold) |
| `test_fusion_eeg_export.py` | Testes unitários (sem hardware) | **SIM** (este scaffold) |
| `docs/INTEGRACAO_EEG_FUSED.md` | Este documento | **SIM** (este scaffold) |
| `eeg_motor_paradigm.py` | Paradigma principal (CSV, MotionRecorder, SharedState) | **NÃO** — só após Fase 4 |
| `fusion/contracts.py` | `FusedState` dataclass (já existe) | Só se necessário para contrato |
| `fusion/sensor_manager.py` | `SensorManager` → publica `FusedState` | Futuro (Fase 3/4) |

---

## 3. Onde Plugar no `eeg_motor_paradigm.py`

### 3.1 Pontos de Injeção (read-only nesta fase)

| Local no código | O que fazer (futuro) |
|-----------------|----------------------|
| **Imports** | `from fusion.eeg_export import FUSED_COLUMNS, fused_row, append_fused_columns, validate_fused_row, FUSION_ENABLED` |
| **Argparse** | Adicionar `--fusion-enabled` (action="store_true", default=False) |
| **MotionRecorder.__init__** | Se `fusion_enabled`: `self.header = append_fused_columns(MOTION_COLUMNS)`; gravar `FUSED_schema_ver=1` no JSON de sessão |
| **MotionRecorder._write_header** | Escrever header extendido |
| **MotionRecorder.write_motion** | Se `fusion_enabled` e `fused_state` disponível: `row + fused_row(fused_state)` |
| **TrackingThread / SharedState** | Publicar `FusedState` em `state.motion["fused"]` (ou campo dedicado) a 30 Hz |
| **Main loop (motion tick)** | Consumir `FusedState` do `SensorManager` / `FusionEngine` e passar para `MotionRecorder` |

### 3.2 Exemplo de Integração Mínima (pseudocódigo futuro)

```python
# No argparse
parser.add_argument("--fusion-enabled", action="store_true", default=False,
                    help="Habilita colunas FUSED_* no CSV de motion (schema v1)")

# No MotionRecorder.__init__
self.fusion_enabled = args.fusion_enabled
if self.fusion_enabled:
    from fusion.eeg_export import FUSED_COLUMNS, append_fused_columns, FUSED_SCHEMA_VERSION
    self.motion_columns = append_fused_columns(MOTION_COLUMNS)
    self.session_metadata["fused_schema_ver"] = FUSED_SCHEMA_VERSION
else:
    self.motion_columns = MOTION_COLUMNS

# No write_motion (chamado a 30 Hz)
def write_motion(self, motion_dict, fused_state=None):
    row = build_motion_row(motion_dict)  # existente
    if self.fusion_enabled and fused_state is not None:
        from fusion.eeg_export import fused_row, validate_fused_row
        fused_vals = fused_row(fused_state)
        vr = validate_fused_row(fused_vals)
        if not vr.ok:
            logger.warning(f"FUSED row validation failed: {vr.errors}")
        row.extend(fused_vals)
    self.writer.writerow(row)
```

---

## 4. Mantendo `PALM_*` / `IMU_*` Intactas (Regressão Zero)

| Coluna Legada | Status | Garantia |
|---------------|--------|----------|
| `PALM_nx, PALM_ny, PALM_nz` | **Inalterada** | Normal do plano médio do dorso (Kinect) — referência absoluta |
| `PALM_dirx, PALM_diry, PALM_dirz` | **Inalterada** | Direção dos dedos (Kinect) |
| `PALM_elev_deg, PALM_azim_deg, PALM_pitch_deg, PALM_yaw_deg` | **Inalterada** | Ângulos derivados do Kinect |
| `IMU_L_* / IMU_R_*` (11+11 colunas) | **Inalterada** | RPY + accel bruta + posição fusionada + valid + ZERO_lock |
| `IMU_hand` | **Inalterada** | Qual mão o trial executava |

**Nenhuma coluna existente é removida, renomeada ou reordenada.**
`FUSED_*` são **append-only** ao final do header de motion.

---

## 5. Feature Flag `--fusion-enabled`

| Valor | Comportamento |
|-------|---------------|
| `False` (default) | CSV de motion = `MOTION_COLUMNS` (74 colunas atuais). Fusão roda opcionalmente para visualização, mas **não** grava no CSV. |
| `True` | CSV de motion = `MOTION_COLUMNS + FUSED_COLUMNS` (93 colunas). `FUSED_schema_ver=1` gravado no JSON de sessão. Requer Fase 4 validada. |

**Onde definir:** `FUSION_ENABLED = False` em `fusion/eeg_export.py` (constante).
O argparse `--fusion-enabled` sobrescreve no `MotionRecorder`.

---

## 6. Ordem dos Testes de Regressão (Obrigatória Antes de Merge)

```bash
# 1. Suite raiz existente (74/74 deve continuar passando)
python -m pytest test_*.py -v

# 2. Novos testes do scaffold FUSED_*
python test_fusion_eeg_export.py -v
# ou: python -m pytest test_fusion_eeg_export.py -v

# 3. Teste de integração OFFLINE (sem hardware, com dados gravados)
#    Reprocessa CSVs antigos + fusão offline → compara PALM_* vs FUSED_*
python -m fusion.replay --input sessao_xyz --compare-palm-fused  # futuro

# 4. Teste de fumaça com --fusion-enabled (generator source, sem Kinect/IMU)
python src\mestradopy\core\eeg_motor_paradigm.py --source generator --fusion-enabled --blocos 1 --trials-por-bloco 2 --sem-kinect
# Verifica: CSV tem 93 colunas, header inclui FUSED_*, JSON tem fused_schema_ver=1
```

**Critério de passagem:** Zero regressões nos 74 testes originais + 28 testes novos passam.

---

## 7. Riscos e Mitigações

| Risco | Prob. | Impacto | Mitigação no Scaffold |
|-------|-------|---------|----------------------|
| **Arquivo grande** (`eeg_motor_paradigm.py` ~3000 linhas) | Alta | Conflitos de merge, bugs sutis | Integração **mínima**: só 3 pontos de injeção (argparse, MotionRecorder init, write_motion). Nenhuma lógica de fusão no arquivo principal. |
| **Sincronização EEG ↔ Motion** | Média | Timestamps dessincronizados | Motion roda a 30 Hz (mesmo loop atual). `FusedState.timestamp` é monotônico. `FUSED_latency_ms` expõe latência da fusão para diagnóstico offline. |
| **Schema versioning** | Baixa | CSVs antigos ilegíveis | `FUSED_schema_ver` no header + no JSON. `validate_fused_row()` checa versão. Futuro v2 = nova coluna no final, v1 ainda lido. |
| **Fusão degrada (fallback)** | Alta | `FUSED_source=0`, confiança baixa | `validate_fused_row()` aceita `source=0` e `confidence≈0`. Não falha o CSV — apenas marca qualidade. |
| **NaN em vetores (visão perdida)** | Média | Linhas com NaN no CSV | `fused_row()` escreve `NaN` para componentes inválidos. CSV reader (pandas) lida nativamente. |
| **Dependência circular** | Baixa | Import errors | `fusion/eeg_export.py` **só importa** `fusion.contracts` e `fusion.quaternion`. **NÃO** importa `eeg_motor_paradigm.py`. |
| **Performance (30 Hz + fusão)** | Baixa | Latência extra no loop | `fused_row()` é puro Python/NumPy (~50 µs). `validate_fused_row()` só em debug. Produção: desabilitar validação. |

---

## 8. Validação da Fase 4 (Pré-requisito para Ativar)

Antes de flippar `--fusion-enabled` para `True` por default, a Fase 4 deve demonstrar:

1. **Erro angular mediano** (fundido vs. Kinect plano médio) < 10° em movimento livre (5 min).
2. **Taxa de fallback** (`source ≠ fused`) < 15% dos frames.
3. **Latência de fusão** (mediana) < 30 ms.
4. **Yaw boot anchor** converge < 10 s, erro < 5°.
5. **Diagnóstico `D`** passa nas 3 poses programadas (mediana ≤ 15°).
6. **Fallback gracioso**: visão some → `imu_only` (bias corrigido mantém posição por segundos); visão volta → `fused` sem salto.
7. **Reprocessamento offline** de dados gravados produz `FUSED_*` consistentes com fusão online.

---

## 9. Plano Mínimo de 3 Passos para Ativar Após Validação

### Passo 1: Injetar Feature Flag e Header Estendido
```bash
# Editar eeg_motor_paradigm.py (3 locais apenas)
# 1. argparse: add --fusion-enabled
# 2. MotionRecorder.__init__: import append_fused_columns, condicional header
# 3. MotionRecorder.write_motion: import fused_row, validate_fused_row, condicional extend
```
**Teste:** `python src\mestradopy\core\eeg_motor_paradigm.py --source generator --fusion-enabled --blocos 1 --trials-por-bloco 2 --sem-kinect` → CSV com 93 colunas, JSON com `fused_schema_ver=1`.

### Passo 2: Conectar `FusedState` do SensorManager ao MotionRecorder
```bash
# Em TrackingThread / main loop:
# - Instanciar FusionEngine / SensorManager (já existe em fusion/)
# - A cada tick de motion (30 Hz): fused_state = sensor_manager.get_fused_state()
# - Passar para motion_recorder.write_motion(motion_dict, fused_state=fused_state)
```
**Teste:** Rodar com Kinect+IMU reais, `--fusion-enabled`, gravar 1 min → validar `FUSED_source ∈ {1,2,3}`, `FUSED_confidence > 0`, sem NaN em quaternion.

### Passo 3: Rodar Suite de Regressão Completa e Documentar
```bash
# 1. Suite raiz (74 testes)
python -m pytest test_*.py -v

# 2. Testes FUSED_* (28 testes)
python test_fusion_eeg_export.py -v

# 3. Teste de integração offline (replay)
python -m fusion.replay --input dados_fase4 --compare-palm-fused --report regressao_fused.md

# 4. Atualizar docs/INTEGRACAO_EEG_FUSED.md com resultados reais (latência, erro, taxa fallback)
# 5. Flippar FUSION_ENABLED = True em fusion/eeg_export.py (opcional, default ainda False no argparse)
```

---

## 10. Referências Cruzadas

- `docs/ARQUITETURA_FUSAO.md` §9.2 — Tabela completa das colunas `FUSED_*` (schema v1)
- `fusion/contracts.py` — `FusedState` dataclass (contrato de entrada para `fused_row()`)
- `fusion/eeg_export.py` — Implementação do scaffold (este documento)
- `test_fusion_eeg_export.py` — Testes de contrato (28 testes, sem hardware)

---

## 11. Changelog

| Versão | Data | Autor | Mudança |
|--------|------|-------|---------|
| 1.0 | 05/11/2026 | Scaffold A3-F5 | Criação inicial: scaffold + testes + documentação |

---

*Fim do documento. Próximo: implementar `fusion/sensor_manager.py` → `SensorManager.get_fused_state()` e validar Fase 4.*