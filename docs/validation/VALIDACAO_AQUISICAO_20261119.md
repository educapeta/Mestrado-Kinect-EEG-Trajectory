# Validação da aquisição — calibração, vídeo offline e sincronismo

**Data:** 19/11/2026
**Escopo:** validar a geometria Kinect + auxiliares 0/1 e o pipeline de vídeo
offline antes de retomar o protocolo EEG.
**Sessão de vídeo usada:** `F:\handmvnet_videos\teste_sync3`
**Processamento:** `processed/` (2 mãos) e `processed_1mao/` (1 mão).

---

## 1. Calibração vigente

As câmeras auxiliares foram remontadas antes da calibração de 17/11. Portanto, a
calibração de 17/11 é a geometria **atual**; os dados de 12/11 pertencem à
geometria antiga e **não** devem ser usados para labels/treino da montagem atual.

| item | valor | estado |
|---|---:|---|
| Kinect intrinsics | RMS 0,631 px | congelado, não reestimar sem mover o Kinect |
| aux0 SIGMA-W780M | RMS **1,6988 px**, 15 poses | aceito (meta < 2 px) |
| aux1 Trust USB | RMS **1,1580 px**, 15 poses | aceito (meta < 2 px) |

Arquivos:

- `configs/calibracao/kinect_intrinsics.npz`
- `configs/calibracao/stereo_calibration.npz` (aux0)
- `configs/calibracao/stereo_calibration_aux1.npz` (aux1)
- `configs/calibracao/camera_alignment.npz`

**Ressalva:** ambos os conjuntos têm poucas poses inclinadas (2/15). O RMS está
dentro do critério, mas para a próxima calibração incluir mais poses de 20–45°
e o tabuleiro na distância de trabalho da mão (1,0–1,4 m).

### Correção depth→color

O 3D do depth do Kinect vem do frame da câmera de **profundidade**, enquanto a
extrínseca das auxiliares está no frame da câmera de **cor**. A correção é feita
por `configs/calibracao/kinect_depth_para_cor.npz` (PnP, resíduo ~1,9 px).

Foi corrigido um bug de caminho após a reorganização: o tool procurava
`src/configs/...` em vez de `configs/...`. Sem a correção, o diagnóstico
`depth 3D na AUXILIAR` ficava inflado em ~25 px (39/51 px → 24/28 px depois da
correção). Commit: `5e14e42`.

---

## 2. Sincronismo dos vídeos

O detector automático de flash não funcionou com a iluminação/exposição da
bancada. A solução operacional é o **sincronismo manual**:

1. iniciar a gravação;
2. dar um flash/clap visível nas duas câmeras;
3. pressionar **S** no preview no mesmo instante;
4. repetir no fim.

Resultado da sessão `teste_sync3`:

```text
sync: manual
offset: +16,0 ms
residual: 0,0 ms
eventos manuais: 1 em cada câmera
```

Comparação com a tentativa anterior:

| tentativa | método | residual |
|---|---|---:|
| `teste_20261119` | lockstep_start | 843 ms |
| `teste_sync3` | manual (S) | **0 ms** |

O `S` é obrigatório quando o flash automático falhar. Não processar vídeos sem
um evento de sync nas duas câmeras.

---

## 3. Vídeo e processamento offline

Configuração da sessão:

| item | aux0 | aux1 |
|---|---:|---:|
| resolução | 1280×720 | 1280×720 |
| codec | H.264 | H.264 |
| frames | 196 | 160 |
| FPS medido | ~10,6 | ~8,0 |
| duração | 20 s | 20 s |
| calibração | 1,699 px | 1,315 px |

O FPS ficou abaixo dos 30 pedidos (limite de USB/exposição). Para movimentos
lentos e processamento offline é aceitável, mas para movimentos rápidos o ideal
é reduzir resolução ou separar as câmeras em controladores USB diferentes.

### 3.1 Duas mãos (padrão antigo)

```text
pares: 124 | saídas: 187
hand: 115 right / 70 left / 2 unknown
trocas de rótulo: 143
cobertura por junção: min 73,8% / mediana 97,3%
reproj aux0/aux1: 2,03 / 1,73 px
```

Esse modo **não é adequado** para uma sessão de uma mão: ele introduz detecções
duplicadas e troca de lateralidade, criando dois níveis de profundidade e
trajetórias inconsistentes.

### 3.2 Uma mão (`--num-hands 1`)

```text
pares: 124 | saídas: 112
hand: 85 right / 27 unknown
trocas de rótulo: 2
cobertura por junção: min 49,1% / mediana 75,9%
reproj aux0/aux1: 1,37 / 1,40 px
triangulação: 112/112 ok
```

Conclusão: para vídeos de uma mão, usar **sempre `--num-hands 1`**. A cobertura
cai em relação ao modo de 2 mãos porque este preenchia frames com detecções
falsas; a consistência melhora.

### 3.3 Visualização 3D

Gerada a partir de `processed_1mao`:

- `outputs/visao/landmarks3d_teste_sync3_1mao/landmarks3d.gif`
- `outputs/visao/landmarks3d_teste_sync3_1mao/landmarks3d_trajetorias.png`
- tool: `src/mestradopy/tools/visualiza_landmarks3d.py --npz ...`

A profundidade Z cai de ~1,35 m para ~1,0–1,2 m por volta de 7,5 s. Se o
participante aproximou a mão nesse instante, é consistente; se não, é troca de
identidade/lateralidade e precisa de rastreamento temporal.

---

## 4. Decisões

1. **Triangulação é o baseline atual** para os 21 joints 3D; com duas vistas e
   MediaPipe funcionando, ela cobre os frames válidos com reprojeção ~1,4 px.
2. **HandMvNet não será treinado do zero agora.** Sem checkpoint, treinar com
   pseudo-labels da própria triangulação seria circular. Só fine-tune/avaliação
   se os pesos forem liberados ou se a triangulação falhar em oclusão/cobertura.
3. Kinect depth continua como fonte primária para o pipeline online; o overlay
   magenta do depth na auxiliar é apenas diagnóstico.
4. Próximas gravações: uma mão por sessão, `S` no início/fim, mais poses de mão,
   duração maior e meta de cobertura > 80% por junção.

---

## 5. Verificações automatizadas

| verificação | resultado |
|---|---|
| `tests/manual/test_handmvnet_video_offline.py` | 73/73 |
| `tests/manual/test_feedback_adaptativo.py` | 10/10 |
| `feedback_adaptativo.py --plano` | 3 sessões × 20/classe, teste congelado |
| `tests/manual/test_estrutura_raiz.py` | raiz limpa |

Higiene da raiz: os PNGs `_hb_*.png` eram gravados pelo `stereo_calibration.py`
na raiz; o código foi corrigido para gravar em
`outputs/visao/debug_calibracao/` e a checagem `scripts/verificar_raiz.ps1` foi
adicionada.
