# PLANO DE VALIDAÇÃO DE HARDWARE — GATE B ATUALIZADO

**Versão:** 2.1 (com Ground Truth de Deslocamento 0,500 m)  
**Data:** 2026-11-13  
**Responsável:** Engenharia de Visão Computacional/BCI

---

## 1. VISÃO GERAL DO GATE B

O **GATE B** valida a **acurácia posicional 3D** do sistema de tracking de mãos (Kinect v2 + MediaPipe + câmeras auxiliares) contra um **ground truth físico rastreável**: duas fitas pretas de **0,500 m** na mesa, uma para mão direita e outra para esquerda.

> **Objetivo:** Demonstrar que o deslocamento medido pelos métodos de tracking (Kinect depth, multi-view aux0/aux1, 3-view) está dentro dos critérios de aceite definidos, para ambas as mãos, em três velocidades.

---

## 2. GROUND TRUTH DE DESLOCAMENTO (NOVO — GATE B)

### 2.1 Layout Experimental

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MESA (vista superior)                          │
│                                                                             │
│    KINECT v2 (centrado, ~1,0 m acima, inclinado ~15° para baixo)           │
│         │                                                                   │
│         ▼                                                                   │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │                                                                 │     │
│    │   AUX0 (esq, ~45°)                          AUX1 (dir, ~45°)   │     │
│    │      ↖                                              ↗           │     │
│    │                                                                 │     │
│    │   ┌─────────────────────────┐         ┌─────────────────────┐  │     │
│    │   │ FITA ESQUERDA (0,500 m) │         │ FITA DIREITA (0,500 m)│  │     │
│    │   │                         │         │                     │  │     │
│    │   │  ●──────────────────●   │         │   ●─────────────●   │  │     │
│    │   │  INÍCIO (0,000 m)   │   │         │   INÍCIO (0,000 m)  │  │     │
│    │   │       0,500 m       │   │         │       0,500 m       │  │     │
│    │   │  ●──────────────────●   │         │   ●─────────────●   │  │     │
│    │   │  FIM (0,500 m)      │   │         │   FIM (0,500 m)     │  │     │
│    │   └─────────────────────────┘         └─────────────────────┘  │     │
│    │                                                                 │     │
│    └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Especificações das fitas:**
- **Comprimento:** 0,500 m ± 1 mm (verificado com paquímetro/régua de aço classe II)
- **Material:** Fita adesiva preta fosca 19-25 mm, baixa reflexão IR
- **Marcações:** Extremidades (INÍCIO/FIM) com fita branca contrastante 20×20 mm
- **Orientação:** Paralelas ao eixo X do Kinect (horizontal na imagem RGB)
- **Separação:** ≥ 30 cm entre eixos centrais das fitas (evita oclusão cruzada)
- **Altura de tracking:** Palma ~2-3 cm acima da mesa (evita ruído de depth na superfície)

### 2.2 Métodos a Comparar

| Método | Fonte de Dados | Pipeline | Saída 3D |
|--------|----------------|----------|----------|
| **Kinect Depth** | Kinect depth + MediaPipe RGB | `KinectHandTracker._landmark_camera_point()` | Palm center (lmk 9) em camera space |
| **Multi-view Aux0** | Kinect RGB + Aux0 | `AuxCameraView.triangulate_with_kinect()` | Triangulação estéreo (RANSAC) |
| **Multi-view Aux1** | Kinect RGB + Aux1 | `AuxCameraView.triangulate_with_kinect()` | Triangulação estéreo (RANSAC) |
| **3-View** | Kinect RGB + Aux0 + Aux1 | `triangulate_three_views()` (DLT + RANSAC/Huber) | Fusão robusta 3 vistas |

> **Reutilização de código:** Todos os métodos usam **exclusivamente** as classes/funções de `kinect_imu_groundtruth.py`:
> - `CameraAlignment` + `AuxCameraView` para calibração/projeção
> - `KinectHandTracker` para depth mapping + MediaPipe
> - `triangulate_three_views` / `build_projection_matrices` para 3-view
> - **NÃO** reimplementar tracker, triangulação ou calibração.

### 2.3 Protocolo de Movimento

| Velocidade | Tempo alvo (0,5 m) | Velocidade média | Repetições por mão |
|------------|-------------------|------------------|-------------------|
| **Lenta** | ~2,0 s | 0,25 m/s | 10 |
| **Média** | ~1,0 s | 0,50 m/s | 10 |
| **Rápida** | ~0,5 s | 1,00 m/s | 10 |

**Total:** 60 trilhas (2 mãos × 3 velocidades × 10 repetições)

**Sequência por repetição:**
1. **Hold inicial** — Palma no INÍCIO (0,000 m), parada ≥ 0,3 s → `ESPAÇO`
2. **Movimento** — Desliza palma ao longo da fita até FIM (0,500 m)
3. **Hold final** — Palma no FIM, parada ≥ 0,3 s → `ESPAÇO` → repetição completa

> **Medição no hold:** Mediana geométrica dos últimos 5+ frames (≈0,17 s @ 30 FPS) — **não** frame isolado.

### 2.4 Métricas e Critérios de Aceite

#### Métricas Primárias (por repetição)
| Métrica | Definição |
|---------|-----------|
| **Deslocamento medido** | ‖pos_hold_fim - pos_hold_inicio‖₂ (mediana no hold) |
| **Erro absoluto** | \|desloc_medido - 0,500\| [m] → [cm] |
| **Erro relativo** | erro_abs / 0,500 × 100 [%] |
| **Velocidade medida** | desloc_medido / Δt_hold [m/s] |
| **Erro velocidade** | \|vel_medida - vel_esperada\| / vel_esperada × 100 [%] |
| **Cobertura landmarks** | % frames com ≥ 21 landmarks válidos (média temporal) |
| **Jitter hold** | σ_posição no hold (norma do vetor σ_x,σ_y,σ_z) [mm] |
| **Lateralidade** | Rótulo MediaPipe = mão física (100% frames) |
| **Troca de mão** | Mudança de lateralidade durante movimento (0 ocorrências) |

#### Estatísticas Agregadas (n=10 repetições)
- **Mediana** — Tendência central robusta
- **P95** — Pior caso operacional (95% das repetições melhores)

#### Critérios de Aceite (Gates de Aprovação)

| Sistema | Métrica | Limite | Gate |
|---------|---------|--------|------|
| **Kinect Depth** | Erro absoluto mediano | ≤ 1 cm | ✅ PASS / ❌ FAIL |
| **Kinect Depth** | Erro absoluto P95 | ≤ 2 cm | ✅ PASS / ❌ FAIL |
| **Multi-view Aux0** | Erro absoluto mediano | ≤ 1,5 cm | ✅ PASS / ❌ FAIL |
| **Kinect Depth** | Cobertura landmarks | ≥ 95% | ✅ PASS / ❌ FAIL |
| **Multi-view** | Cobertura landmarks | ≥ 90% | ✅ PASS / ❌ FAIL |
| **Kinect Depth** | Jitter no hold | ≤ 5 mm | ✅ PASS / ❌ FAIL |
| **Todos** | Erro velocidade mediano | ≤ 15% | ✅ PASS / ❌ FAIL |
| **Todos** | Lateralidade | 100% | ✅ PASS / ❌ FAIL |
| **Todos** | Troca de mão | 0 | ✅ PASS / ❌ FAIL |

> **Base técnica:** Kinect v2 depth noise ~1-4 mm @ 0,5 m (sigma); triangulação estéreo baseline 40-60 cm → ~2-5 mm; MediaPipe hands coverage >95% em condições ideais. Critérios incluem margem para condições reais (mesa, iluminação, oclusão parcial).

---

## 3. PRÉ-REQUISITOS DE CALIBRAÇÃO (Obrigatórios antes do GATE B)

| Calibração | Arquivo(s) | Critério de Validação | Ferramenta |
|------------|------------|----------------------|------------|
| Intrínseca Kinect | `kinect_intrinsics.npz` | RMS reprojeção < 1 px | `tools/conserta_intrinsecas_kinect.py` + `test_intrinsecas_kinect.py` |
| Estéreo Kinect↔Aux0 | `stereo_calibration.npz` | RMS ≤ 2 px (aceita) / ≤ 5 px (usável) | `stereo_calibration.py` |
| Estéreo Kinect↔Aux1 | `stereo_calibration_aux1.npz` | RMS ≤ 2 px / ≤ 5 px | `stereo_calibration.py --aux-index 1` |
| Landmarks Aux0 | `hand_landmark_calibration.npz` | ≥ 18 amostras, span depth ≥ 0,4 m | `tools/kinect_groundtruth_tool.py` (tecla H) |
| Landmarks Aux1 | `hand_landmark_calibration_aux1.npz` | ≥ 18 amostras, span depth ≥ 0,4 m | `tools/kinect_groundtruth_tool.py --aux-index 1` |
| **Lateralidade** | `aux_handedness_flip.json` | 100% concordância Kinect↔Aux0↔Aux1 | `tools/probe_lateralidade.py` |

> **Regra:** `CameraAlignment.stereo_status()` deve retornar `"aceita"` para ambas auxiliares. Se `"aproximada"`, GATE B roda mas overlay tem precisão reduzida.

---

## 4. FERRAMENTA DE EXECUÇÃO

### 4.1 Script Principal
```
tools/validacao_groundtruth_deslocamento.py
```

### 4.2 Modos de Operação

| Modo | Comando | Uso |
|------|---------|-----|
| **Hardware (mão única)** | `python src/mestradopy/tools/validacao_groundtruth_deslocamento.py --mao direita --velocidade media --repeticoes 10` | Coleta real com Kinect + auxiliares |
| **Hardware (ambas)** | `python src/mestradopy/tools/validacao_groundtruth_deslocamento.py --mao ambas --velocidade media --repeticoes 10` | Coleta sequencial direita + esquerda |
| **Sem janela (headless)** | `... --sem-janela` | CI/servidor sem display |
| **Sintético (teste)** | `python src/mestradopy/tools/validacao_groundtruth_deslocamento.py --sintetico --mao direita --velocidade media --repeticoes 10` | Validação de pipeline sem hardware |

### 4.3 Interface Interativa (Hardware)
- `ESPAÇO` — Marca hold_inicio → movimento → hold_fim
- `R` — Repete repetição atual
- `Q` / `ESC` — Aborta

### 4.4 Saídas Geradas
```
validacao_groundtruth/
├── validacao_direita_media.csv      # Dados brutos por frame
├── validacao_esquerda_media.csv
├── relatorio_direita_media.json     # Métricas agregadas + gates
├── relatorio_esquerda_media.json
└── validacao_sintetica_*.csv/.json  # Modo --sintetico
```

---

## 5. PLANO DE ANÁLISE ESTATÍSTICA

### 5.1 Comparações Principais
1. **Absoluta:** Cada método vs. 0,500 m (ground truth)
2. **Lateralidade:** Mão direita vs. esquerda (mesmo método, mesma velocidade)
3. **Dinâmica:** Lenta vs. Média vs. Rápida (dependência de velocidade)
4. **Fusão:** 3-View vs. melhor par (ganho de robustez)

### 5.2 Testes Estatísticos

| Hipótese | Teste | α | Potência (n=10) |
|----------|-------|---|-----------------|
| Erro mediano Kinect ≤ 1 cm | Wilcoxon signed-rank (unilateral) | 0,05 | ~80% p/ d=0,9 |
| Erro P95 Kinect ≤ 2 cm | Bootstrap percentile CI 95% | 0,05 | - |
| Direita = Esquerda (erro) | Mann-Whitney U | 0,05 | ~80% p/ d=0,9 |
| Velocidade afeta erro | Friedman + Nemenyi post-hoc | 0,05 | - |
| Cobertura Kinect ≥ 95% | Binomial exact | 0,05 | - |
| Jitter hold ≤ 5 mm | Wilcoxon signed-rank | 0,05 | - |

### 5.3 Relatório Final por Condição
```json
{
  "lado": "direita",
  "velocidade": "media",
  "n_repeticoes": 10,
  "kinect_depth": {
    "deslocamento_mediano_m": 0.502,
    "deslocamento_p95_m": 0.518,
    "erro_abs_mediano_cm": 0.8,
    "erro_abs_p95_cm": 1.8,
    "velocidade_erro_mediano_pct": 4.2,
    "cobertura_media_pct": 97.3,
    "jitter_hold_mediano_mm": 2.1
  },
  "multi_view_aux0": { ... },
  "multi_view_3view": { ... },
  "criterios_aceite": {
    "kinect_erro_mediano_cm_ok": true,
    "kinect_erro_p95_cm_ok": true,
    "mv_aux0_erro_mediano_cm_ok": true,
    "kinect_cobertura_ok": true,
    "mv_cobertura_ok": true,
    "kinect_jitter_ok": true,
    "velocidade_erro_ok": true,
    "lateralidade_ok": true
  }
}
```

---

## 6. CRONOGRAMA ESTIMADO

| Etapa | Duração | Dependências |
|-------|---------|--------------|
| Verificação calibrações (Seção 3) | 30 min | Arquivos .npz existentes |
| Checagem lateralidade (`probe_lateralidade.py`) | 15 min | Calibrações OK |
| Posicionamento fitas + medição paquímetro | 15 min | Mesa limpa, fita 0,500 m |
| Coleta mão direita (3 vel × 10 rep) | ~45 min | Participante/operador |
| Coleta mão esquerda (3 vel × 10 rep) | ~45 min | Participante/operador |
| Processamento + relatórios | 10 min | Script roda inline |
| Análise estatística + decisão Gate | 30 min | Python/pandas/scipy |

**Total estimado:** ~3 horas (uma sessão)

---

## 7. RISCOS E MITIGAÇÕES

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Calibração estéreo RMS > 5 px | Média | Multi-view inutilizável | Recalibrar tabuleiro (7×5/45mm) na mesma pose |
| Lateralidade trocada em uma aux | Alta (já ocorreu) | Mão errada triangulada | `probe_lateralidade.py` obrigatório antes |
| Depth mapping falha nas bordas | Média | Kinect depth cobertura < 95% | ROI dinâmico MediaPipe (`DSH_MEDIAPIPE_ROI=1`) |
| Motion blur na velocidade rápida | Alta | Erro velocidade > 15% | Aceitar critério mais frouxo p/ rápida (documentar) |
| Oclusão fita pela mão | Baixa | Ground truth visual perdido | Fita fina (19mm), palma 2-3cm acima |

---

## 8. DOCUMENTAÇÃO ASSOCIADA

| Documento | Localização | Finalidade |
|-----------|-------------|------------|
| Protocolo completo | `Tese_Latex_Agente/Validacao/PROTOCOLO_GROUNDTRUTH_DESLOCAMENTO.md` | Detalhamento científico, ética, análise |
| Biblioteca base | `kinect_imu_groundtruth.py` | CameraAlignment, KinectHandTracker, triangulação |
| Ferramenta calibração | `tools/kinect_groundtruth_tool.py` | Calibração estéreo, landmarks, lateralidade |
| Probe lateralidade | `tools/probe_lateralidade.py` | Verificação ao vivo por câmera |
| Testes unitários | `test_triangulation.py`, `test_deteccao_cascata.py` | Validação de triangulação 2/3-view |

---

## 9. CRITÉRIOS DE DECISÃO DO GATE B

| Resultado | Ação |
|-----------|------|
| **Todos os gates ✅ PASS** | GATE B **APROVADO** → Prosseguir para GATE C (validação dinâmica/velocidade) |
| **Alguns gates ❌ FAIL** (ex.: jitter, cobertura) | Investigar causa (calibração, iluminação, oclusão); corrigir; repetir condição |
| **Falha sistemática** (ex.: Kinect erro > 2 cm P95) | Revisar pipeline depth mapping / hardware; não prosseguir |

> **Regra:** GATE B é **binário por condição** (mão × velocidade). Para aprovação geral, **todas as 6 condições** (2 mãos × 3 velocidades) devem passar nos gates críticos (erro mediano, P95, lateralidade).

---

## 10. HISTÓRICO

| Versão | Data | Alteração |
|--------|------|-----------|
| 1.0 | 2026-10-XX | Plano original (calibração + overlay) |
| 2.0 | 2026-11-13 | **Adicionada Seção 2 — Ground Truth de Deslocamento (GATE B)** (1,000 m) |
| 2.1 | 2026-11-13 | **Correção: 0,500 m (50 cm) em toda a Seção 2** |

---

**Fim do Plano Atualizado**