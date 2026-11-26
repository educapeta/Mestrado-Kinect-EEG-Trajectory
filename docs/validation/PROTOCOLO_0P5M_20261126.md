# Protocolo de deslocamento de 0,5 m — resultado

**Data:** 26/11/2026  
**Lado:** mão direita (`--lado-mao right`)  
**Frames válidos:** 580  
**Referência externa:** 0,500 m medidos com régua/fita

## Resultado

| métrica | par aux0/aux1 | Kinect |
|---|---:|---:|
| deslocamento A→B | **0,4821 m** | **0,4901 m** |
| erro absoluto vs régua | **0,0179 m** (1,79 cm) | **0,0099 m** (0,99 cm) |
| drift de retorno a A | 0,0155 m (1,55 cm) | 0,0074 m (0,74 cm) |
| comprimento trajeto ida | 0,4853 m | 0,5603 m |
| comprimento trajeto volta | 0,5358 m | 0,5978 m |

Erro 3D par×Kinect durante o trajeto:

- mediana: **42,0 mm**
- p90: **46,5 mm**
- frames: 578

Rigidez dos ossos:

| fonte | CV bruto | CV robusto |
|---|---:|---:|
| par aux0/aux1 | 2,52% | **2,20%** |
| Kinect | 247,60% | 12,13% |

## Leitura

1. O **par aux0/aux1 reconstruiu 0,482 m** para um deslocamento real de
   0,500 m: erro de **1,8 cm (3,6%)**. É uma validação positiva de escala
   absoluta para o par.
2. O **Kinect** acertou mais o deslocamento absoluto neste teste
   (0,490 m; erro 1,0 cm), mas teve trajetória mais ruidosa: comprimento de
   ida/volta 0,56–0,60 m para um deslocamento de 0,49 m.
3. O **drift de retorno** foi pequeno nos dois: 1,55 cm no par e 0,74 cm no
   Kinect.
4. A **rigidez robusta** do par foi 2,20%; a do Kinect, 12,13%. O Kinect tem
   muito mais outliers de profundidade.
5. O erro par×Kinect de 42 mm é **sistemático** e maior que o esperado para
   ruído; não usar o Kinect como verdade absoluta. Ele serve como QA grosseiro,
   não como ground truth fino.

## Conclusão

- O par aux0/aux1 é viável para medir **deslocamento absoluto** da mão dentro
  de ~2 cm neste protocolo.
- O Kinect depth não é adequado como ground truth fino de trajetória.
- A validação atual é de **posição/deslocamento**, não de orientação, grip ou
  pose completa.
- Ainda é uma única repetição; para repetibilidade, repetir 3–5 vezes.

## Próximos passos

1. Repetir o protocolo 3–5 vezes com as mesmas marcas A/B.
2. Manter a mão direita e o trajeto dentro da sobreposição das duas auxiliares.
3. Reportar média/desvio do erro de deslocamento e do drift.
4. Investigar/calibrar o offset de 42 mm par×Kinect antes de usar o Kinect como
   referência de pose.
5. Só depois avançar para orientação/grip.
