# Trajetória suavizada por landmark — kinect

Deslocamento físico: 50.0 cm; tolerância: 2.0 cm (mediana do erro).

Landmarks dentro da tolerância: **17/21**

| Landmark | Desloc. mediano | Erro mediano | Erro p95 | Caminho integrado | OK |
|---:|---:|---:|---:|---:|:---:|
| 0 | 0.4957 m | 0.73 cm | 10.45 cm | 0.4872 m | SIM |
| 1 | 0.5111 m | 1.27 cm | 10.16 cm | 0.4944 m | SIM |
| 2 | 0.5049 m | 1.17 cm | 11.00 cm | 0.4977 m | SIM |
| 3 | 0.5109 m | 1.67 cm | 11.94 cm | 0.4996 m | SIM |
| 4 | 0.5259 m | 4.02 cm | 12.51 cm | 0.5103 m | NÃO |
| 5 | 0.5175 m | 2.09 cm | 9.86 cm | 0.5049 m | NÃO |
| 6 | 0.5103 m | 1.75 cm | 10.55 cm | 0.5057 m | SIM |
| 7 | 0.5128 m | 1.95 cm | 10.77 cm | 0.5058 m | SIM |
| 8 | 0.5140 m | 1.68 cm | 10.37 cm | 0.5040 m | SIM |
| 9 | 0.5046 m | 1.19 cm | 10.15 cm | 0.4978 m | SIM |
| 10 | 0.5086 m | 2.04 cm | 10.54 cm | 0.5017 m | NÃO |
| 11 | 0.5145 m | 2.42 cm | 9.02 cm | 0.5054 m | NÃO |
| 12 | 0.5086 m | 1.41 cm | 6.23 cm | 0.5018 m | SIM |
| 13 | 0.5020 m | 0.84 cm | 10.14 cm | 0.4923 m | SIM |
| 14 | 0.5064 m | 1.50 cm | 9.82 cm | 0.4993 m | SIM |
| 15 | 0.5105 m | 1.97 cm | 6.05 cm | 0.5030 m | SIM |
| 16 | 0.5085 m | 1.82 cm | 5.46 cm | 0.4959 m | SIM |
| 17 | 0.4969 m | 0.76 cm | 9.97 cm | 0.4870 m | SIM |
| 18 | 0.4974 m | 1.16 cm | 9.90 cm | 0.4888 m | SIM |
| 19 | 0.4982 m | 1.29 cm | 9.57 cm | 0.4910 m | SIM |
| 20 | 0.4976 m | 1.22 cm | 8.60 cm | 0.4896 m | SIM |

## Interpretação

- deslocamento = distância entre as medianas do hold inicial e final;
- caminho integrado = soma das distâncias entre amostras suavizadas;
- se o caminho for muito maior que 50 cm, há ruído/curvatura; se for muito menor, o movimento foi menor que o alvo.
