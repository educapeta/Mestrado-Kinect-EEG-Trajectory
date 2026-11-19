# Validação rápida — rigidez de ossos e comparação online

**Data:** 19/11/2026
**Dados:** `outputs/visao/comparacao_pair_kinect_v3.csv` e
`comparacao_pair_kinect_v3_rigidez.csv` (sessão online com Kinect + aux0 + aux1).
**Duração:** ~20 s; FPS mediano 11,7.

## 1. Comparação 3D par × Kinect

| fase | lado | erro3D mediana | p90 |
|---|---|---:|---:|
| 0–5 s (parado) | esquerda | 22,2 mm | 33,9 mm |
| 0–5 s (parado) | direita | 21,8 mm | 29,1 mm |
| 5–10 s | esquerda | 30,5 mm | 42,0 mm |
| 5–10 s | direita | 35,2 mm | 60,7 mm |
| 10–20 s (movimento) | esquerda | 35,8 mm | 61,2 mm |
| 10–20 s (movimento) | direita | 34,7 mm | 42,8 mm |

Reprojeção do 3D do par nas auxiliares (após correção de extrínseca):

| lado | aux0 | aux1 |
|---|---:|---:|
| esquerda | 2,53 px | 1,90 px |
| direita | 2,54 px | 2,51 px |

Reprojeção no Kinect: ~16,2 px (mediana), mesmo parado.

## 2. Rigidez de ossos

CV = desvio-padrão / média do comprimento do osso em todos os frames.

| lado | fonte | CV mediano | p90 | comprimento mediano |
|---|---|---:|---:|---:|
| direita | par aux0↔aux1 | 11,44% | 18,30% | 26,65 mm |
| direita | Kinect depth | 151,32% | 457,19% | 34,59 mm |
| esquerda | par aux0↔aux1 | 10,21% | 13,22% | 28,73 mm |
| esquerda | Kinect depth | 73,08% | 158,93% | 36,78 mm |

Offline, no `teste_sync3` (par, 42 ossos): CV mediano **2,66%**, p90 6,19%.

Razão por osso Kinect/par: mediana **1,05** (direita) e **1,02** (esquerda),
sem viés global de escala; o Kinect tem cauda de outliers.

## 3. Leitura

1. O par aux0↔aux1 tem reprojeção baixa e rigidez de osso razoável no online
   (10–11%), mas pior que o offline (2,7%) — provável mistura de movimento,
   timestamps e frames de identidade instável.
2. O Kinect depth **não é uma referência limpa** para CV: 73–151% de CV,
   dominado por outliers de profundidade; a mediana por osso até concorda em
   escala (~1,0), mas a dispersão é enorme.
3. O erro 3D par×Kinect (~22 mm parado) não deve ser atribuído automaticamente
   ao par: o Kinect também tem erro de depth/map e correção depth→cor.
4. Para decidir o melhor pipeline, usar:
   - CV de osso **robusto** (mediana/MAD, com filtro de outliers);
   - comprimento mediano de ossos vs medida antropométrica (régua/caliper);
   - Kinect+aux0 e Kinect+aux1 como triangulações independentes, não só depth.

## 4. Próximos passos

- Adicionar CV robusto (descartar ossos fora de mediana ± 3 MAD) no comparador.
- Medir 2–3 ossos da mão do participante com régua para checar escala absoluta.
- Repetir com mão parada marcada (tecla ou fase) para separar estático de
  movimento e comparar CV antes/depois da suavização.
