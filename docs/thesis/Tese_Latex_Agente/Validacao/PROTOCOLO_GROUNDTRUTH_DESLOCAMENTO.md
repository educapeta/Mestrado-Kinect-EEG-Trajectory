# PROTOCOLO GROUND TRUTH DE DESLOCAMENTO — GATE B

**Versão:** 1.1  
**Data:** 2026-11-13  
**Responsável:** Engenharia de Visão Computacional/BCI  
**Workspace:** `C:\mestradopy`

---

## 1. OBJETIVO CIENTÍFICO

Validar a acurácia do **deslocamento 3D da palma da mão** medido por três sistemas de tracking independentes, comparando contra um **ground truth físico conhecido de 0,500 m**:

| Sistema | Descrição |
|---------|-----------|
| **Kinect Depth 3D** | Profundidade do Kinect v2 + MediaPipe Hands no RGB (color→depth mapping) |
| **Multi-view Aux0** | Triangulação estéreo Kinect RGB + Câmera Auxiliar 0 (homografia + stereo) |
| **Multi-view Aux1** | Triangulação estéreo Kinect RGB + Câmera Auxiliar 1 |
| **3-View (Kinect+Aux0+Aux1)** | Triangulação robusta de 3 vistas com RANSAC/Huber (DLT multi-view) |

A validação compara **mão DIREITA vs. ESQUERDA** (lateralidade, troca de lado, cobertura, jitter) e mede **velocidade** a partir de timestamps.

---

## 2. LAYOUT EXPERIMENTAL

### 2.1 Duas Fitas Pretas de 0,500 m na Mesa

```
┌─────────────────────────────────────────────────────────────────┐
│                        MESA (topo)                              │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  KINECT v2 (acima, centrado, ~1.0 m da mesa)            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   ┌─────────────────────────┐    ┌─────────────────────────┐   │
│   │   FITA ESQUERDA         │    │   FITA DIREITA          │   │
│   │   (mão esquerda)        │    │   (mão direita)         │   │
│   │                         │    │                         │   │
│   │   ●─────────────────●   │    │   ●─────────────────●   │   │
│   │   INÍCIO (0.000 m)    │    │   INÍCIO (0.000 m)      │   │
│   │       0,500 m         │    │       0,500 m           │   │
│   │   ●─────────────────●   │    │   ●─────────────────●   │   │
│   │   FIM (0,500 m)       │    │   FIM (0,500 m)         │   │
│   │                         │    │                         │   │
│   │   Paralela ao eixo X    │    │   Paralela ao eixo X    │   │
│   │   do Kinect (esq→dir)   │    │   do Kinect (esq→dir)   │   │
│   └─────────────────────────┘    └─────────────────────────┘   │
│                                                                 │
│   Aux0 (esquerda, ~45°)          Aux1 (direita, ~45°)          │
└─────────────────────────────────────────────────────────────────┘
```

**Especificações das fitas:**
- **Material:** Fita adesiva preta fosca, largura 19-25 mm
- **Comprimento exato:** 0,500 m ± 1 mm (medido com paquímetro/régua de aço)
- **Posicionamento:** Paralelas ao eixo X do Kinect (horizontal na imagem)
- **Altura:** Sobre a mesa (Z ≈ 0 no referencial da mesa)
- **Distância entre fitas:** ≥ 30 cm (evita oclusão mútua)
- **Marcações:** Pontos de **INÍCIO** e **FIM** claramente visíveis (marcador branco ou fita contrastante nas extremidades)

### 2.2 Postura e Orientação da Palma

| Parâmetro | Especificação |
|-----------|---------------|
| **Altura da palma** | ~2-3 cm acima da mesa (não toca a fita; evita oclusão de depth) |
| **Orientação** | Palma **para baixo** (normal ≈ -Y do Kinect), dedos apontando para **frente** (+Z) |
| **Punho** | Neutro, sem rotação excessiva (yaw ≈ 0°) |
| **Cotovelo** | Apoiado ou flutuante confortável; ombro relaxado |
| **Campo de visão** | Palma inteira visível no Kinect RGB + ambas auxiliares durante todo o percurso |

### 2.3 Protocolo de Movimento

| Velocidade | Tempo alvo (0,5 m) | Velocidade média | Tolerância |
|------------|-------------------|------------------|------------|
| **Lenta** | ~2,0 s | 0,25 m/s | ±15% |
| **Média** | ~1,0 s | 0,50 m/s | ±15% |
| **Rápida** | ~0,5 s | 1,00 m/s | ±15% |

**Repetições:** 10 por mão por velocidade (ida = 1 repetição; volta = repetição separada se aplicável)  
**Total por condição:** 10 repetições × 2 mãos × 3 velocidades = **60 trilhas**

---

## 3. PROCEDIMENTO DE CALIBRAÇÃO (Pré-requisito)

### 3.1 Calibração Intrínseca do Kinect
- Arquivo: `kinect_intrinsics.npz` (gerado por `tools/conserta_intrinsecas_kinect.py`)
- Validação: `test_intrinsecas_kinect.py` (RMS reprojeção < 1 px)

### 3.2 Calibração Estéreo Kinect ↔ Auxiliares
- Arquivos: `stereo_calibration.npz` (aux0), `stereo_calibration_aux1.npz` (aux1)
- Tabuleiro: 7×5 cantos internos (8×6 quadrados), 45 mm/quadrado
- Critério: RMS estéreo ≤ 2,0 px (ideal) / ≤ 5,0 px (usável para overlay)
- Verificação: `CameraAlignment.stereo_status()` deve retornar `"aceita"`

### 3.3 Calibração de Landmarks (Homografia Nó-a-Nó)
- Arquivos: `hand_landmark_calibration.npz`, `hand_landmark_calibration_aux1.npz`
- Amostras: ≥ 18 poses variadas (escala + profundidade)
- Modelo 3D: `scale_depth_params` + `node_z_offset` para estimativa de Z por landmark

### 3.4 Checagem de Lateralidade (CRÍTICO)
- **Problema conhecido:** MediaPipe assume imagem espelhada (selfie); Kinect RGB **não** é espelhado; auxiliares **são** espelhadas (`AUXILIARY_MIRROR_HORIZONTAL = True`).
- **Solução:** `MEDIAPIPE_KINECT_HANDEDNESS_FLIP = True` inverte rótulo do Kinect.
- **Por câmera auxiliar:** `AUX_HANDEDNESS_FLIP_FILE` (`aux_handedness_flip.json`) armazena inversão aprendida por índice.
- **Validação:** `tools/probe_lateralidade.py` — coloca as duas mãos no quadro; verifica se cada auxiliar concorda com o Kinect. Se uma auxiliar troca as mãos, pressiona tecla para inverter (`--aux-flip-lateralidade N`).
- **Critério de aceite:** Lateralidade 100% correta em ambas as mãos, sem troca durante o movimento.

---

## 4. PROCEDIMENTO DE COLETA (Por Condição)

### 4.1 Preparação
1. Ligue Kinect + auxiliares; aguarde estabilização térmica (~5 min).
2. Carregue calibrações (`CameraAlignment` auto-carrega `.npz`).
3. Execute `tools/probe_lateralidade.py` e confirme lateralidade 100%.
4. Posicione o participante sentado, cotovelo apoiado, palma ~2-3 cm acima da fita correspondente.

### 4.2 Sequência por Repetição (Interface: `tools/validacao_groundtruth_deslocamento.py`)

```
ESTADO: AGUARDANDO HOLD INICIAL
  → Usuário posiciona palma no INÍCIO da fita (marcador 0,000 m)
  → Mantém PARADO por ~0,4 s (mediana de ≥5 frames)
  → Pressiona ESPAÇO → sistema marca "hold_inicio"

ESTADO: MOVIMENTO
  → Usuário move palma ao longo da fita até o FIM (0,500 m)
  → Velocidade alvo conforme tabela (lenta/média/rápida)
  → Sistema grava frames contínuos (timestamp + posição 3D)

ESTADO: AGUARDANDO HOLD FINAL
  → Usuário para no FIM da fita (marcador 0,500 m)
  → Mantém PARADO por ~0,4 s (mediana de ≥5 frames)
  → Pressiona ESPAÇO → sistema marca "hold_fim" e finaliza repetição
```

**Teclas durante coleta:**
- `ESPAÇO` — Marca transição hold_inicio → movimento → hold_fim
- `R` — Repete a repetição atual (descarta dados)
- `Q` / `ESC` — Aborta e sai

### 4.3 Medição no Hold (Não Frame Isolado)
- **Por que:** Ruído frame-a-frame (~mm a cm) contaminaria deslocamento/velocidade.
- **Método:** Mediana geométrica dos últimos `MIN_HOLD_FRAMES = 5` frames do hold (≈0,17 s @ 30 FPS; alvo 0,3-0,5 s).
- **Aplicável a:** Posição 3D (início/fim), jitter (desvio padrão no hold), velocidade (Δposição_mediana / Δtempo_hold).

---

## 5. MÉTRICAS E CRITÉRIOS DE ACEITE

### 5.1 Métricas Primárias (por repetição)

| Métrica | Definição | Método de Cálculo |
|---------|-----------|-------------------|
| **Deslocamento medido** | ‖pos_hold_fim - pos_hold_inicio‖ | Mediana geométrica 3D no hold |
| **Erro absoluto** | \|desloc_medido - 0,500\| | Em metros → cm |
| **Erro relativo** | erro_abs / 0,500 × 100% | Percentual |
| **Velocidade medida** | desloc_medido / (t_hold_fim - t_hold_inicio) | m/s |
| **Erro velocidade** | \|vel_medida - vel_esperada\| / vel_esperada × 100% | Percentual |
| **Cobertura landmarks** | % de 21 landmarks válidos/frame | Média temporal |
| **Jitter no hold** | Desvio padrão da posição no hold | mm (norma do vetor σ_x, σ_y, σ_z) |
| **Lateralidade** | Rótulo MediaPipe = mão física | Binário (ok/falha) por frame |
| **Troca de mão** | Mudança de lateralidade durante movimento | Binário (ocorreu/não) |

### 5.2 Estatísticas Agregadas (10 repetições)

| Estatística | Uso |
|-------------|-----|
| **Mediana** | Tendência central robusta (insensível a outliers) |
| **P95** | Pior caso operacional (95% das repetições melhores) |
| **Máximo** | Limite absoluto |

### 5.3 Critérios de Aceite (Documentados — NÃO inventar resultados)

| Sistema | Métrica | Limite | Justificativa |
|---------|---------|--------|---------------|
| **Kinect Depth** | Erro absoluto mediano | ≤ 1 cm | 2% de 0,5 m (precisão nativa do depth @ 0,5 m) |
| **Kinect Depth** | Erro absoluto P95 | ≤ 2 cm | 4% de 0,5 m (outliers de borda/oclusão) |
| **Multi-view Aux0** | Erro absoluto mediano | ≤ 1,5 cm | 3% de 0,5 m (baseline ~45°, triangulação) |
| **Kinect Depth** | Cobertura 21 landmarks | ≥ 95% | MediaPipe + depth mapping |
| **Multi-view** | Cobertura landmarks | ≥ 90% | Homografia + triangulação |
| **Kinect Depth** | Jitter no hold | ≤ 5 mm | Estabilidade depth + mapping |
| **Todos** | Erro de velocidade | ≤ 10-15% | Timestamp sync + Δposição |
| **Todos** | Lateralidade | 100% | Zero troca de mão |
| **Todos** | Troca de mão | 0 ocorrências | Consistência temporal |

> **Nota:** Critérios baseados em literatura (Kinect v2 depth noise ~1-4 mm @ 0,5 m; triangulação estéreo ~2-5 mm com baseline 40-60 cm) e medições internas de bancada. Valores servem como *gates* para aprovação do hardware no GATE B.

---

## 6. PLANO DE ANÁLISE ESTATÍSTICA

### 6.1 Comparações Principais

1. **Kinect vs. Ground Truth** — Validação absoluta do depth 3D
2. **Multi-view Aux0 vs. Ground Truth** — Validação estéreo
3. **Multi-view Aux1 vs. Ground Truth** — Validação estéreo (lado oposto)
4. **3-View vs. Ground Truth** — Fusão multi-view (melhor caso)
5. **Mão Direita vs. Esquerda** — Lateralidade, assimetria sistemática
6. **Velocidades (Lenta/Média/Rápida)** — Dependência dinâmica (motion blur, latency)

### 6.2 Testes Estatísticos

| Comparação | Teste | Variável |
|------------|-------|----------|
| Erro mediano Kinect vs. 1 cm | Wilcoxon signed-rank (unilateral) | Erro absoluto por repetição |
| Erro P95 Kinect vs. 2 cm | Percentil bootstrap (IC 95%) | Erro absoluto por repetição |
| Direita vs. Esquerda (erro) | Mann-Whitney U | Erro mediano por repetição |
| Velocidades (erro) | Friedman + post-hoc Nemenyi | Erro velocidade por repetição |
| Cobertura Kinect vs. 95% | Binomial exact test | Frames válidos / total |
| Jitter hold vs. 5 mm | Wilcoxon signed-rank | Jitter mediano por repetição |

### 6.3 Tamanho de Efeito e Potência

- **n = 10 repetições** por condição (mão × velocidade)
- **Potência alvo:** 80% para detectar diferença de 0,5 cm no erro mediano (α = 0,05)
- **Tamanho de efeito mínimo detectável:** Cohen's d ≈ 0,9 (grande) com n=10 pareado
- **Recomendação:** Se efeito for pequeno (d < 0,5), aumentar para n=20 em coletas futuras.

### 6.4 Relatório Final (por condição)

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
    "erro_rel_mediano_pct": 1.6,
    "erro_rel_p95_pct": 3.6,
    "velocidade_erro_mediano_pct": 4.2,
    "velocidade_erro_p95_pct": 11.5,
    "cobertura_media_pct": 97.3,
    "jitter_hold_mediano_mm": 2.1,
    "jitter_hold_p95_mm": 4.8
  },
  "multi_view_aux0": { ... },
  "multi_view_3view": { ... },
  "lateralidade": { "lateralidade_100_pct": true, "troca_mao_ocorreu": false },
  "criterios_aceite": { ... }
}
```

---

## 7. FRONTEIRA ÉTICA — TESTE DE BANCADA vs. COLETA COM SERES HUMANOS

### 7.1 Classificação da Atividade

| Atividade | Classificação | Enquadramento |
|-----------|---------------|---------------|
| **Teste sintético** (`--sintetico`) | Simulação computacional | **NÃO é pesquisa com seres humanos** — sem coleta de dados pessoais/biométricos |
| **Validação com operador** (movimento da própria mão do engenheiro) | Teste de bancada / verificação de equipamento | **NÃO é pesquisa com seres humanos** — o operador é parte da equipe técnica, não "participante de pesquisa" |
| **Coleta com voluntários externos** (pessoas não da equipe) | Pesquisa com seres humanos | **EXIGE aprovação do Comitê de Ética (CAAE)**, TCLE, anonimização, etc. |

### 7.2 Regra Prática

> **Se a pessoa movendo a mão é autor do código / engenheiro responsável / membro da equipe técnica que conhece o protocolo e o hardware → TESTE DE BANCADA.**  
> **Se a pessoa é voluntário recrutado, não conhece o sistema, assina TCLE → PESQUISA COM SERES HUMANOS.**

### 7.3 Documentação para Auditoria

- Este protocolo (`PROTOCOLO_GROUNDTRUTH_DESLOCAMENTO.md`) registra a **finalidade de validação de equipamento**.
- O script `tools/validacao_groundtruth_deslocamento.py` possui **modo sintético (`--sintetico`)** que executa toda a pipeline sem câmeras nem pessoa.
- Qualquer coleta com voluntários **deve** ter protocolo separado submetido ao CEP/ConEp, com este documento como anexo técnico apenas.

---

## 8. ENTREGÁVEIS DO GATE B

| Arquivo | Descrição |
|---------|-----------|
| `tools/validacao_groundtruth_deslocamento.py` | Ferramenta principal (hardware + sintético) |
| `Tese_Latex_Agente/Validacao/PROTOCOLO_GROUNDTRUTH_DESLOCAMENTO.md` | Este documento |
| `Tese_Latex_Agente/Validacao/PLANO_VALIDACAO_HARDWARE.md` | Atualizado (Seção 2) |
| `validacao_groundtruth/*.csv` | Dados brutos por frame (rastreabilidade) |
| `validacao_groundtruth/relatorio_*.json` | Métricas agregadas + critérios de aceite |

---

## 9. REFERÊNCIAS TÉCNICAS INTERNAS

- `kinect_imu_groundtruth.py` — Biblioteca base (CameraAlignment, KinectHandTracker, AuxCameraView, triangulate_three_views)
- `tools/kinect_groundtruth_tool.py` — Ferramenta de calibração/diagnóstico
- `tools/probe_lateralidade.py` — Verificação de lateralidade por câmera
- `tools/diagnostico_calibracao.py` — Checagem de RMS estéreo e homografia
- `test_triangulation.py` — Testes unitários de triangulação 2-view e 3-view

---

## 10. HISTÓRICO DE VERSÕES

| Versão | Data | Autor | Alterações |
|--------|------|-------|------------|
| 1.0 | 2026-11-13 | Eng. Visão/BCI | Criação inicial — GATE B completo (1,000 m) |
| 1.1 | 2026-11-13 | Eng. Visão/BCI | **Correção: 0,500 m (50 cm) em toda a documentação, thresholds e testes** |

---

**Fim do Protocolo**